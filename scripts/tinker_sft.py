"""Minimal Tinker SFT runner for this repo's JSON datasets."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger("tinker_sft")
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.sharded_multiturn import ShardedTask, initial_messages, shard_message
from scripts.tinker_grpo import (
    acquire_run_lock,
    is_sharded_multiturn_row,
    load_json_records,
    release_run_lock,
    resolve_split_file,
    row_to_messages,
    shuffle_records,
    wait_result,
)

DEFAULT_CLARIFYING_QUESTION = "Could you provide the next missing detail?"
MATH_DATASETS = {"math", "math500", "dapo_math", "gsm8k"}
MCQ_DATASETS = {"sciknoweval", "gpqa"}


def write_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=True) + "\n")


def _extract_final_numeric_answer(answer: str) -> str:
    text = answer.strip()
    if "####" in text:
        text = text.rsplit("####", 1)[1].strip()
    return text


def format_tooluse_target(answer: Any) -> str:
    if isinstance(answer, str):
        try:
            payload = json.loads(answer)
        except json.JSONDecodeError:
            return str(answer).strip()
    else:
        payload = answer
    if not isinstance(payload, list):
        return str(answer).strip()

    formatted = []
    for item in payload:
        if not isinstance(item, dict):
            return str(answer).strip()
        action = item.get("Action") or item.get("action")
        action_input = item.get("Action_Input") or item.get("action_input") or item.get("Action Input")
        if action is None or action_input is None:
            return str(answer).strip()
        if isinstance(action_input, str):
            try:
                action_input = json.dumps(json.loads(action_input), ensure_ascii=True, sort_keys=True)
            except json.JSONDecodeError:
                action_input = action_input.strip()
        else:
            action_input = json.dumps(action_input, ensure_ascii=True, sort_keys=True)
        formatted.append(f"Action: {action}\nAction Input: {action_input}")
    return "\n\n".join(formatted)


def format_math_target(answer: str) -> str:
    stripped = answer.strip()
    if "\\boxed{" in stripped:
        return stripped
    return f"\\boxed{{{_extract_final_numeric_answer(stripped)}}}"


def format_mcq_target(answer: str) -> str:
    candidate = answer.strip()
    if "<answer>" in candidate.lower():
        return candidate
    return f"<answer>\n{candidate}\n</answer>"


def format_sharded_final_target(task: ShardedTask) -> str:
    answer = task.answer.strip()
    if "<final" in answer.lower() or "<answer>" in answer.lower():
        return answer
    return f"<final>{answer}</final>"


def format_sft_target_for_row(row: dict[str, Any]) -> str:
    data_source = str(row.get("dataset", "")).lower()
    answer = str(row.get("answer", "")).strip()
    kind = str(row.get("kind", "")).lower()
    if data_source == "tooluse":
        return format_tooluse_target(answer)
    if data_source in MCQ_DATASETS or kind in {"mcq", "multiple_choice"}:
        return format_mcq_target(answer)
    if data_source in MATH_DATASETS:
        return format_math_target(answer)
    return answer


def build_supervised_examples(
    row: dict[str, Any],
    prompt_style: str | None = None,
    allow_untagged_final: bool = False,
) -> list[dict[str, Any]]:
    if is_sharded_multiturn_row(row):
        task = ShardedTask.from_row(row, allow_untagged_final=allow_untagged_final)
        messages = initial_messages(task, prompt_style=prompt_style)
        examples: list[dict[str, Any]] = []
        for shard_index, shard in enumerate(task.shards):
            examples.append(
                {
                    "messages": [dict(message) for message in messages],
                    "target": DEFAULT_CLARIFYING_QUESTION,
                    "target_type": "clarify",
                }
            )
            messages.append({"role": "assistant", "content": DEFAULT_CLARIFYING_QUESTION})
            messages.append(
                {
                    "role": "user",
                    "content": shard_message(shard, shard_index, len(task.shards), prompt_style=prompt_style),
                }
            )
        examples.append(
            {
                "messages": [dict(message) for message in messages],
                "target": format_sharded_final_target(task),
                "target_type": "final",
            }
        )
        return examples
    return [{"messages": row_to_messages(row, prompt_style=prompt_style), "target": format_sft_target_for_row(row), "target_type": "single"}]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a small Tinker SFT job.")
    parser.add_argument("--data-path", default=os.environ.get("DATA_PATH", "datasets/tooluse"))
    parser.add_argument("--split", default=os.environ.get("SPLIT", "train"))
    parser.add_argument("--model-name", default=os.environ.get("MODEL_NAME", "Qwen/Qwen3-8B"))
    parser.add_argument(
        "--renderer-name",
        default=os.environ.get("RENDERER_NAME"),
        help="Override the Tinker cookbook renderer; defaults to the recommended renderer for the model.",
    )
    parser.add_argument("--base-url", default=os.environ.get("TINKER_BASE_URL"))
    parser.add_argument("--log-dir", default=os.environ.get("TINKER_SFT_LOG_DIR", "_logs/tinker_sft"))
    parser.add_argument("--run-name", default=os.environ.get("RUN_NAME", "tinker-sft-smoke"))
    parser.add_argument("--batch-size", type=int, default=int(os.environ.get("BATCH_SIZE", "1")))
    parser.add_argument("--max-steps", type=int, default=int(os.environ.get("MAX_STEPS", "1")))
    parser.add_argument(
        "--shuffle-seed",
        type=int,
        default=int(os.environ.get("SHUFFLE_SEED", "-1")),
        help="Shuffle training rows before selecting max_steps * batch_size rows; negative disables shuffling.",
    )
    parser.add_argument(
        "--sharded-prompt-style",
        default=os.environ.get("SHARDED_PROMPT_STYLE", "default"),
        help="Prompt style for sharded_multiturn rows: default, minimal, linc_math, or tool_schema.",
    )
    parser.add_argument(
        "--sharded-allow-untagged-final",
        action="store_true",
        default=os.environ.get("SHARDED_ALLOW_UNTAGGED_FINAL", "0").strip().lower()
        in {"1", "true", "yes", "on"},
        help="Allow untagged responses to be scored as final answers for sharded_multiturn rows.",
    )
    parser.add_argument("--learning-rate", type=float, default=float(os.environ.get("LR", "1e-5")))
    parser.add_argument("--lora-rank", type=int, default=int(os.environ.get("LORA_RANK", "32")))
    parser.add_argument("--save-every", type=int, default=int(os.environ.get("SAVE_EVERY", "0")))
    parser.add_argument("--dry-run", action="store_true", help="Validate local inputs without contacting Tinker.")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()

    split_file = resolve_split_file(args.data_path, args.split)
    rows = load_json_records(split_file)
    if not rows:
        raise SystemExit(f"No records found in {split_file}")
    rows = shuffle_records(rows, args.shuffle_seed)
    max_examples = min(len(rows), args.batch_size * args.max_steps)
    rows = rows[:max_examples]
    LOGGER.info("Loaded %d %s records from %s", len(rows), args.split, split_file)

    if args.dry_run:
        examples = build_supervised_examples(
            rows[0],
            prompt_style=args.sharded_prompt_style,
            allow_untagged_final=args.sharded_allow_untagged_final,
        )
        LOGGER.info("Dry run prompt roles: %s", [msg["role"] for msg in examples[0]["messages"]])
        LOGGER.info(
            "Dry run dataset=%s target_type=%s target=%s",
            rows[0].get("dataset"),
            examples[0]["target_type"],
            examples[0]["target"][:200],
        )
        return

    if not os.environ.get("TINKER_API_KEY"):
        raise SystemExit("Set TINKER_API_KEY in the environment before running Tinker jobs.")

    try:
        import tinker
        import torch
        from tinker import types
        from tinker_cookbook import model_info, renderers
        from tinker_cookbook.supervised.common import datum_from_model_input_weights
        from tinker_cookbook.tokenizer_utils import get_tokenizer
    except ImportError as exc:
        raise SystemExit("Install Tinker dependencies with: uv pip install -r requirements-tinker.txt") from exc

    log_dir = Path(args.log_dir)
    lock_path = acquire_run_lock(log_dir, args.run_name)
    metrics_path = log_dir / f"{args.run_name}-metrics.jsonl"
    samples_path = log_dir / f"{args.run_name}-samples.jsonl"

    try:
        for path in (metrics_path, samples_path):
            if path.exists():
                path.unlink()

        tokenizer = get_tokenizer(args.model_name)
        renderer_name = args.renderer_name or model_info.get_recommended_renderer_name(args.model_name)
        renderer = renderers.get_renderer(renderer_name, tokenizer, model_name=args.model_name)
        adam_params = types.AdamParams(
            learning_rate=args.learning_rate,
            beta1=0.9,
            beta2=0.95,
            eps=1e-8,
        )
        service_client = tinker.ServiceClient(base_url=args.base_url)
        training_client = service_client.create_lora_training_client(
            base_model=args.model_name,
            rank=args.lora_rank,
        )
        total_steps = min(args.max_steps, (len(rows) + args.batch_size - 1) // args.batch_size)

        def build_sft_datum(messages: list[dict[str, str]], target_text: str) -> tuple[Any, int] | None:
            target_tokens = tokenizer.encode(target_text, add_special_tokens=False)
            if not target_tokens:
                return None
            prompt = renderer.build_generation_prompt(messages)
            model_input = prompt.append(types.EncodedTextChunk(tokens=target_tokens))
            weights = torch.zeros(model_input.length, dtype=torch.float32)
            weights[prompt.length :] = 1.0
            return datum_from_model_input_weights(model_input, weights), len(target_tokens)

        for step in range(total_steps):
            start_time = time.time()
            batch_rows = rows[step * args.batch_size : (step + 1) * args.batch_size]
            datums = []
            sample_count = 0
            token_count = 0

            for row in batch_rows:
                examples = build_supervised_examples(
                    row,
                    prompt_style=args.sharded_prompt_style,
                    allow_untagged_final=args.sharded_allow_untagged_final,
                )
                for example_index, example in enumerate(examples):
                    built = build_sft_datum(example["messages"], example["target"])
                    if built is None:
                        continue
                    datum, target_tokens = built
                    datums.append(datum)
                    sample_count += 1
                    token_count += target_tokens
                    write_jsonl(
                        samples_path,
                        {
                            "step": step,
                            "idx": row.get("idx"),
                            "dataset": row.get("dataset"),
                            "example_index": example_index,
                            "target_type": example["target_type"],
                            "prompt_messages": example["messages"],
                            "target": example["target"],
                            "target_tokens": target_tokens,
                        },
                    )

            train_loss = None
            fwd_bwd_metrics = {}
            optim_metrics = {}
            if datums:
                fwd_bwd_result = wait_result(training_client.forward_backward(datums, loss_fn="cross_entropy"))
                train_loss = getattr(fwd_bwd_result, "loss", None)
                fwd_bwd_metrics = getattr(fwd_bwd_result, "metrics", None) or {}
                optim_result = wait_result(training_client.optim_step(adam_params))
                optim_metrics = getattr(optim_result, "metrics", None) or {}
            else:
                LOGGER.warning("Step %d produced no SFT datums; skipped optimizer step.", step)

            if args.save_every > 0 and (step + 1) % args.save_every == 0:
                wait_result(training_client.save_state(name=f"{args.run_name}-step-{step + 1:06d}"))

            metrics = {
                "step": step,
                "examples": len(batch_rows),
                "samples_used": sample_count,
                "supervised_tokens": token_count,
                "loss": train_loss,
                "time_sec": time.time() - start_time,
                **{f"fwd_bwd/{key}": value for key, value in fwd_bwd_metrics.items()},
                **optim_metrics,
            }
            write_jsonl(metrics_path, metrics)
            LOGGER.info(
                "step=%d samples_used=%d supervised_tokens=%d loss=%s",
                step,
                sample_count,
                token_count,
                "None" if train_loss is None else f"{train_loss:.6f}",
            )

        final_state = wait_result(training_client.save_state(name=f"{args.run_name}-final"))
        final_sampler = wait_result(training_client.save_weights_for_sampler(name=f"{args.run_name}-final-sampler"))
        LOGGER.info("Saved final Tinker state: %s", final_state)
        LOGGER.info("Saved final Tinker sampler weights: %s", final_sampler)
    finally:
        release_run_lock(lock_path)


if __name__ == "__main__":
    main()

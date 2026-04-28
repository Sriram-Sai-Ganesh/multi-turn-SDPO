"""Evaluate a base model or saved Tinker checkpoint on this repo's JSON datasets."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.sharded_multiturn import SampledResponse, ShardedTask, run_sharded_interaction
from scripts.tinker_grpo import (
    is_sharded_multiturn_row,
    load_json_records,
    resolve_split_file,
    row_to_messages,
    score_response,
    wait_result,
)

LOGGER = logging.getLogger("tinker_eval")


def mean(values: list[float]) -> float:
    return sum(values) / max(len(values), 1)


def write_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=True) + "\n")


def is_training_state_path(path: str | None) -> bool:
    return bool(path and "/weights/" in path and "/sampler_weights/" not in path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Tinker sampling on a held-out split.")
    parser.add_argument("--data-path", default=os.environ.get("DATA_PATH", "datasets/tooluse"))
    parser.add_argument("--split", default=os.environ.get("SPLIT", "test"))
    parser.add_argument("--model-name", default=os.environ.get("MODEL_NAME", "Qwen/Qwen3-8B"))
    parser.add_argument("--model-path", default=os.environ.get("MODEL_PATH"))
    parser.add_argument(
        "--renderer-name",
        default=os.environ.get("RENDERER_NAME"),
        help="Override the Tinker cookbook renderer; defaults to the recommended renderer for the model.",
    )
    parser.add_argument("--base-url", default=os.environ.get("TINKER_BASE_URL"))
    parser.add_argument("--log-dir", default=os.environ.get("TINKER_EVAL_LOG_DIR", "_logs/tinker_eval"))
    parser.add_argument("--run-name", default=os.environ.get("RUN_NAME", "tinker-eval"))
    parser.add_argument("--batch-size", type=int, default=int(os.environ.get("BATCH_SIZE", "8")))
    parser.add_argument("--num-samples", type=int, default=int(os.environ.get("NUM_SAMPLES", "1")))
    parser.add_argument("--max-examples", type=int, default=int(os.environ.get("MAX_EXAMPLES", "0")))
    parser.add_argument(
        "--max-turns",
        type=int,
        default=int(os.environ.get("MAX_TURNS", "0")),
        help="Max assistant turns for sharded_multiturn rows; 0 uses len(shards)+1.",
    )
    parser.add_argument(
        "--sharded-prompt-style",
        default=os.environ.get("SHARDED_PROMPT_STYLE", "default"),
        help="Prompt style for sharded_multiturn rows: default, minimal, or linc_math.",
    )
    parser.add_argument(
        "--sharded-allow-untagged-final",
        action="store_true",
        default=os.environ.get("SHARDED_ALLOW_UNTAGGED_FINAL", "0").strip().lower()
        in {"1", "true", "yes", "on"},
        help="Allow untagged responses to be scored as final answers for sharded_multiturn rows.",
    )
    parser.add_argument("--max-tokens", type=int, default=int(os.environ.get("MAX_TOKENS", "256")))
    parser.add_argument("--temperature", type=float, default=float(os.environ.get("TEMPERATURE", "0.0")))
    parser.add_argument("--dry-run", action="store_true", help="Validate local inputs without contacting Tinker.")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()

    split_file = resolve_split_file(args.data_path, args.split)
    rows = load_json_records(split_file)
    if args.max_examples > 0:
        rows = rows[: args.max_examples]
    if not rows:
        raise SystemExit(f"No records found in {split_file}")
    LOGGER.info("Loaded %d %s records from %s", len(rows), args.split, split_file)

    if args.dry_run:
        first_messages = row_to_messages(rows[0], prompt_style=args.sharded_prompt_style)
        LOGGER.info("Dry run prompt roles: %s", [msg["role"] for msg in first_messages])
        LOGGER.info(
            "Dry run dataset=%s model=%s model_path=%s sharded_prompt_style=%s allow_untagged_final=%s",
            rows[0].get("dataset"),
            args.model_name,
            args.model_path,
            args.sharded_prompt_style,
            args.sharded_allow_untagged_final,
        )
        return

    if not os.environ.get("TINKER_API_KEY"):
        raise SystemExit("Set TINKER_API_KEY in the environment before running Tinker jobs.")

    try:
        import tinker
        from tinker import types
        from tinker_cookbook import model_info, renderers
        from tinker_cookbook.tokenizer_utils import get_tokenizer
    except ImportError as exc:
        raise SystemExit("Install Tinker dependencies with: uv pip install -r requirements-tinker.txt") from exc

    tokenizer = get_tokenizer(args.model_name)
    renderer_name = args.renderer_name or model_info.get_recommended_renderer_name(args.model_name)
    renderer = renderers.get_renderer(renderer_name, tokenizer, model_name=args.model_name)
    sampling_params = types.SamplingParams(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        stop=renderer.get_stop_sequences(),
    )

    log_dir = Path(args.log_dir)
    metrics_path = log_dir / f"{args.run_name}-metrics.json"
    samples_path = log_dir / f"{args.run_name}-samples.jsonl"
    if samples_path.exists():
        samples_path.unlink()

    service_client = tinker.ServiceClient(base_url=args.base_url)
    if args.model_path:
        if is_training_state_path(args.model_path):
            LOGGER.info(
                "MODEL_PATH points to a training checkpoint; exporting sampler weights before eval."
            )
            training_client = service_client.create_training_client_from_state(args.model_path)
            sampler_response = wait_result(
                training_client.save_weights_for_sampler(name=f"{args.run_name}-sampler")
            )
            LOGGER.info("Saved sampler checkpoint for eval: %s", sampler_response.path)
            sampling_client = service_client.create_sampling_client(model_path=sampler_response.path)
            model_label = sampler_response.path
        else:
            sampling_client = service_client.create_sampling_client(model_path=args.model_path)
            model_label = args.model_path
    else:
        sampling_client = service_client.create_sampling_client(base_model=args.model_name)
        model_label = args.model_name

    LOGGER.info(
        "Evaluating model=%s renderer=%s examples=%d num_samples=%d sharded_prompt_style=%s allow_untagged_final=%s",
        model_label,
        renderer_name,
        len(rows),
        args.num_samples,
        args.sharded_prompt_style,
        args.sharded_allow_untagged_final,
    )
    start_time = time.time()
    rewards: list[float] = []
    format_errors = 0
    examples_with_success = 0

    for start in range(0, len(rows), args.batch_size):
        batch_rows = rows[start : start + args.batch_size]
        futures = []
        sharded_flags = [is_sharded_multiturn_row(row) for row in batch_rows]
        if any(sharded_flags) and not all(sharded_flags):
            raise ValueError("Mixed sharded_multiturn and single-turn rows in one eval batch are not supported.")

        if all(sharded_flags):
            for row in batch_rows:
                task = ShardedTask.from_row(row, allow_untagged_final=args.sharded_allow_untagged_final)
                row_rewards: list[float] = []
                for sample_idx in range(args.num_samples):
                    def sample_fn(messages: list[dict[str, str]]) -> SampledResponse:
                        prompt = renderer.build_generation_prompt(messages)
                        sample_result = wait_result(
                            sampling_client.sample(
                                prompt=prompt,
                                num_samples=1,
                                sampling_params=sampling_params,
                            )
                        )
                        sequence = sample_result.sequences[0]
                        sampled_tokens = list(sequence.tokens)
                        parsed_message, _ = renderer.parse_response(sampled_tokens)
                        response = renderers.get_text_content(parsed_message)
                        return SampledResponse(text=response, tokens=sampled_tokens, prompt=prompt)

                    rollout = run_sharded_interaction(
                        task,
                        sample_fn,
                        max_turns=args.max_turns if args.max_turns > 0 else None,
                        prompt_style=args.sharded_prompt_style,
                    )
                    reward = rollout.reward
                    rewards.append(reward)
                    row_rewards.append(reward)
                    format_errors += int(rollout.score.get("incorrect_format", 0))
                    record = rollout.to_log_record()
                    record.update(
                        {
                            "idx": row.get("idx"),
                            "dataset": row.get("dataset"),
                            "sample_idx": sample_idx,
                        }
                    )
                    write_jsonl(samples_path, record)
                examples_with_success += int(any(reward > 0 for reward in row_rewards))
        else:
            for row in batch_rows:
                prompt = renderer.build_generation_prompt(row_to_messages(row))
                futures.append(
                    sampling_client.sample(
                        prompt=prompt,
                        num_samples=args.num_samples,
                        sampling_params=sampling_params,
                    )
                )

            for row, future in zip(batch_rows, futures):
                sample_result = wait_result(future)
                row_rewards: list[float] = []
                for sample_idx, sequence in enumerate(sample_result.sequences):
                    sampled_tokens = list(sequence.tokens)
                    parsed_message, _ = renderer.parse_response(sampled_tokens)
                    response = renderers.get_text_content(parsed_message)
                    score = score_response(row, response, args.split)
                    reward = float(score.get("score", 0.0))
                    rewards.append(reward)
                    row_rewards.append(reward)
                    format_errors += int(score.get("incorrect_format", 0))
                    write_jsonl(
                        samples_path,
                        {
                            "idx": row.get("idx"),
                            "dataset": row.get("dataset"),
                            "sample_idx": sample_idx,
                            "reward": reward,
                            "response": response,
                            "score": score,
                        },
                    )
                examples_with_success += int(any(reward > 0 for reward in row_rewards))

        LOGGER.info(
            "evaluated=%d/%d reward_mean=%.4f",
            min(start + len(batch_rows), len(rows)),
            len(rows),
            mean(rewards),
        )

    metrics = {
        "run_name": args.run_name,
        "model": model_label,
        "model_name": args.model_name,
        "model_path": args.model_path,
        "resolved_model_path": model_label if str(model_label).startswith("tinker://") else None,
        "renderer": renderer_name,
        "data_path": str(split_file),
        "examples": len(rows),
        "num_samples": args.num_samples,
        "samples": len(rewards),
        "reward_mean": mean(rewards),
        "examples_with_success": examples_with_success,
        "example_success_rate": examples_with_success / len(rows),
        "format_errors": format_errors,
        "format_error_rate": format_errors / max(len(rewards), 1),
        "time_sec": time.time() - start_time,
    }
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    LOGGER.info("Saved eval metrics to %s", metrics_path)
    LOGGER.info("reward_mean=%.4f format_error_rate=%.4f", metrics["reward_mean"], metrics["format_error_rate"])


if __name__ == "__main__":
    main()

"""Minimal Tinker GRPO runner for this repo's JSON datasets.

This is intentionally separate from the verl/Ray training path. Tinker owns
remote sampling, forward/backward, optimizer steps, and checkpoint storage;
this script owns data loading, reward computation, and advantage construction.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
import types as py_types
from functools import lru_cache
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger("tinker_grpo")
REPO_ROOT = Path(__file__).resolve().parents[1]


def load_json_records(path: Path) -> list[dict[str, Any]]:
    """Load either a JSON array or newline-delimited JSON records."""
    text = path.read_text(encoding="utf-8")
    stripped = text.lstrip()
    if not stripped:
        return []
    if stripped[0] == "[":
        records = json.loads(stripped)
    else:
        records = [json.loads(line) for line in text.splitlines() if line.strip()]
    if not isinstance(records, list):
        raise ValueError(f"Expected a list of records in {path}")
    return records


def resolve_split_file(data_path: str, split: str) -> Path:
    path = Path(data_path)
    if path.is_dir():
        path = path / f"{split}.json"
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    return path


def row_to_messages(row: dict[str, Any]) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    system = row.get("system")
    if system:
        messages.append({"role": "system", "content": str(system)})
    messages.append({"role": "user", "content": str(row["prompt"])})
    return messages


def ground_truth_for_row(row: dict[str, Any]) -> Any:
    if row.get("kind") == "code" and row.get("tests") not in (None, "", "-"):
        return row["tests"]
    return row["answer"]


def extra_info_for_row(row: dict[str, Any], split: str) -> dict[str, Any]:
    return {
        "split": split,
        "index": str(row.get("idx", "")),
        "description": row.get("description", row.get("prompt", "")),
        "problem": row.get("prompt", ""),
        "elo": row.get("elo", 1500),
        "achievement_prior": row.get("achievement_prior", 0),
        "truncated": False,
    }


@lru_cache(maxsize=None)
def load_feedback_module(name: str) -> Any:
    module_path = REPO_ROOT / "verl" / "utils" / "reward_score" / "feedback" / f"{name}.py"
    source = module_path.read_text(encoding="utf-8")
    if "from __future__ import annotations" not in source.splitlines()[:5]:
        source = "from __future__ import annotations\n" + source
    module = py_types.ModuleType(f"_tinker_feedback_{name}")
    module.__file__ = str(module_path)
    exec(compile(source, str(module_path), "exec"), module.__dict__)
    return module


def score_response(row: dict[str, Any], response: str, split: str) -> dict[str, Any]:
    data_source = str(row["dataset"])
    ground_truth = ground_truth_for_row(row)
    extra_info = extra_info_for_row(row, split)

    if data_source in {"code", "livecodebench", "humanevalplus"}:
        code = load_feedback_module("code")

        return code.compute_score(response, ground_truth, extra_info, sparse_rewards=True, max_test_cases=None)
    if data_source in {"math", "math500", "dapo_math", "gsm8k"}:
        math = load_feedback_module("math")

        return math.compute_score(response, ground_truth, extra_info)
    if data_source == "gpqa":
        gpqa = load_feedback_module("gpqa")

        return gpqa.compute_score(response, ground_truth)
    if data_source == "sciknoweval":
        mcq = load_feedback_module("mcq")

        return mcq.compute_score(response, ground_truth)
    if data_source == "tooluse":
        tooluse = load_feedback_module("tooluse")

        return tooluse.compute_score(response, ground_truth)
    raise ValueError(f"Reward style {data_source} not found.")


def mean(values: list[float]) -> float:
    return sum(values) / max(len(values), 1)


def write_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=True) + "\n")


def wait_result(value: Any) -> Any:
    if hasattr(value, "result"):
        return value.result()
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a small GRPO-style Tinker training job.")
    parser.add_argument("--data-path", default=os.environ.get("DATA_PATH", "datasets/tooluse"))
    parser.add_argument("--split", default=os.environ.get("SPLIT", "train"))
    parser.add_argument("--model-name", default=os.environ.get("MODEL_NAME", "Qwen/Qwen3-8B"))
    parser.add_argument(
        "--renderer-name",
        default=os.environ.get("RENDERER_NAME"),
        help="Override the Tinker cookbook renderer; defaults to the recommended renderer for the model.",
    )
    parser.add_argument("--base-url", default=os.environ.get("TINKER_BASE_URL"))
    parser.add_argument("--log-dir", default=os.environ.get("TINKER_LOG_DIR", "_logs/tinker_grpo"))
    parser.add_argument("--run-name", default=os.environ.get("RUN_NAME", "tinker-grpo-smoke"))
    parser.add_argument("--batch-size", type=int, default=int(os.environ.get("BATCH_SIZE", "1")))
    parser.add_argument("--rollout-n", type=int, default=int(os.environ.get("ROLLOUT_N", "2")))
    parser.add_argument("--max-steps", type=int, default=int(os.environ.get("MAX_STEPS", "1")))
    parser.add_argument("--max-tokens", type=int, default=int(os.environ.get("MAX_TOKENS", "512")))
    parser.add_argument("--temperature", type=float, default=float(os.environ.get("TEMPERATURE", "1.0")))
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
    max_examples = min(len(rows), args.batch_size * args.max_steps)
    rows = rows[:max_examples]
    LOGGER.info("Loaded %d %s records from %s", len(rows), args.split, split_file)

    if args.dry_run:
        first_messages = row_to_messages(rows[0])
        LOGGER.info("Dry run prompt roles: %s", [msg["role"] for msg in first_messages])
        LOGGER.info("Dry run dataset=%s model=%s", rows[0].get("dataset"), args.model_name)
        return

    if not os.environ.get("TINKER_API_KEY"):
        raise SystemExit("Set TINKER_API_KEY in the environment before running Tinker jobs.")

    try:
        import tinker
        import torch
        from tinker import types
        from tinker.types.tensor_data import TensorData
        from tinker_cookbook import model_info, renderers
        from tinker_cookbook.tokenizer_utils import get_tokenizer
    except ImportError as exc:
        raise SystemExit("Install Tinker dependencies with: uv pip install -r requirements-tinker.txt") from exc

    log_dir = Path(args.log_dir)
    metrics_path = log_dir / f"{args.run_name}-metrics.jsonl"
    samples_path = log_dir / f"{args.run_name}-samples.jsonl"
    for path in (metrics_path, samples_path):
        if path.exists():
            path.unlink()

    tokenizer = get_tokenizer(args.model_name)
    renderer_name = args.renderer_name or model_info.get_recommended_renderer_name(args.model_name)
    renderer = renderers.get_renderer(renderer_name, tokenizer, model_name=args.model_name)
    sampling_params = types.SamplingParams(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        stop=renderer.get_stop_sequences(),
    )
    adam_params = types.AdamParams(
        learning_rate=args.learning_rate,
        beta1=0.9,
        beta2=0.95,
        eps=1e-8,
    )

    LOGGER.info("Using Tinker model=%s renderer=%s", args.model_name, renderer_name)
    service_client = tinker.ServiceClient(base_url=args.base_url)
    training_client = service_client.create_lora_training_client(
        base_model=args.model_name,
        rank=args.lora_rank,
    )
    total_steps = min(args.max_steps, (len(rows) + args.batch_size - 1) // args.batch_size)

    for step in range(total_steps):
        start_time = time.time()
        batch_rows = rows[step * args.batch_size : (step + 1) * args.batch_size]
        sampling_client = training_client.save_weights_and_get_sampling_client()

        futures = []
        prompts = []
        for row in batch_rows:
            prompt = renderer.build_generation_prompt(row_to_messages(row))
            futures.append(
                sampling_client.sample(
                    prompt=prompt,
                    num_samples=args.rollout_n,
                    sampling_params=sampling_params,
                )
            )
            prompts.append(prompt)

        datums = []
        group_rewards: list[float] = []
        sample_count = 0
        for row, prompt, future in zip(batch_rows, prompts, futures):
            sample_result = wait_result(future)
            rewards: list[float] = []
            sampled_payloads = []
            for sequence in sample_result.sequences:
                sampled_tokens = list(sequence.tokens)
                sampled_logprobs = sequence.logprobs
                if sampled_logprobs is None or len(sampled_tokens) == 0:
                    continue
                parsed_message, _ = renderer.parse_response(sampled_tokens)
                response = renderers.get_text_content(parsed_message)
                score = score_response(row, response, args.split)
                reward = float(score.get("score", 0.0))
                rewards.append(reward)
                sampled_payloads.append((sampled_tokens, list(sampled_logprobs), response, score))
                write_jsonl(
                    samples_path,
                    {
                        "step": step,
                        "idx": row.get("idx"),
                        "dataset": row.get("dataset"),
                        "reward": reward,
                        "response": response,
                        "score": score,
                    },
                )

            if not rewards:
                continue
            reward_mean = mean(rewards)
            advantages = [reward - reward_mean for reward in rewards]
            group_rewards.append(reward_mean)

            for (sampled_tokens, sampled_logprobs, _response, _score), advantage in zip(sampled_payloads, advantages):
                ob_len = prompt.length - 1
                model_input = prompt.append(types.EncodedTextChunk(tokens=sampled_tokens[:-1]))
                target_tokens = [0] * ob_len + sampled_tokens
                padded_logprobs = [0.0] * ob_len + sampled_logprobs
                padded_advantages = [0.0] * ob_len + [advantage] * (model_input.length - ob_len)
                if not (
                    model_input.length == len(target_tokens) == len(padded_logprobs) == len(padded_advantages)
                ):
                    raise RuntimeError("Tinker datum length mismatch while assembling rollout data.")
                datums.append(
                    types.Datum(
                        model_input=model_input,
                        loss_fn_inputs={
                            "target_tokens": TensorData.from_torch(torch.tensor(target_tokens)),
                            "logprobs": TensorData.from_torch(torch.tensor(padded_logprobs)),
                            "advantages": TensorData.from_torch(torch.tensor(padded_advantages)),
                        },
                    )
                )
                sample_count += 1

        train_loss = None
        fwd_bwd_metrics = {}
        optim_metrics = {}
        if datums:
            fwd_bwd_future = training_client.forward_backward(datums, loss_fn="importance_sampling")
            optim_future = training_client.optim_step(adam_params)
            fwd_bwd_result = wait_result(fwd_bwd_future)
            optim_result = wait_result(optim_future)
            train_loss = getattr(fwd_bwd_result, "loss", None)
            fwd_bwd_metrics = {
                f"fwd_bwd/{key}": value for key, value in (getattr(fwd_bwd_result, "metrics", None) or {}).items()
            }
            optim_metrics = getattr(optim_result, "metrics", None) or {}
        else:
            LOGGER.warning("Step %d produced no non-zero advantages; skipped optimizer step.", step)

        if args.save_every > 0 and (step + 1) % args.save_every == 0:
            wait_result(training_client.save_state(name=f"{args.run_name}-step-{step + 1:06d}"))

        metrics = {
            "step": step,
            "examples": len(batch_rows),
            "samples_used": sample_count,
            "reward_mean": mean(group_rewards),
            "loss": train_loss,
            "time_sec": time.time() - start_time,
            **fwd_bwd_metrics,
            **optim_metrics,
        }
        write_jsonl(metrics_path, metrics)
        LOGGER.info("step=%d reward_mean=%.4f samples_used=%d", step, metrics["reward_mean"], sample_count)

    final_state = wait_result(training_client.save_state(name=f"{args.run_name}-final"))
    final_sampler = wait_result(
        training_client.save_weights_for_sampler(name=f"{args.run_name}-final-sampler")
    )
    LOGGER.info("Saved final Tinker state: %s", final_state)
    LOGGER.info("Saved final Tinker sampler weights: %s", final_sampler)


if __name__ == "__main__":
    main()

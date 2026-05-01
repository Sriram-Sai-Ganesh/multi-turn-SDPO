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
import random
import sys
import time
import types as py_types
from functools import lru_cache
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger("tinker_grpo")
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.sharded_multiturn import (
    SDPO_REWARD_MODE,
    SPARSE_REWARD_MODE,
    SampledResponse,
    ShardedTask,
    build_sdpo_teacher_messages,
    normalize_reward_mode,
    rollout_training_reward,
    rollout_training_rewards,
    run_sharded_interaction,
    select_sdpo_distillation_turn,
)


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


def shuffle_records(records: list[dict[str, Any]], seed: int | None) -> list[dict[str, Any]]:
    if seed is None or seed < 0:
        return records
    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)
    return shuffled


def row_to_messages(row: dict[str, Any], prompt_style: str | None = None) -> list[dict[str, str]]:
    if is_sharded_multiturn_row(row):
        from scripts.sharded_multiturn import initial_messages

        return initial_messages(ShardedTask.from_row(row), prompt_style=prompt_style)
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


def is_sharded_multiturn_row(row: dict[str, Any]) -> bool:
    return str(row.get("dataset")) == "sharded_multiturn"


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
    if data_source == "sharded_multiturn":
        from scripts.sharded_multiturn import score_sharded_response

        return score_sharded_response(response, ShardedTask.from_row(row))
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


def centered_turn_advantages(turn_reward_groups: list[list[float]]) -> list[list[float]]:
    """Center dense rewards by turn index within one prompt's rollout group."""
    max_turns = max((len(rewards) for rewards in turn_reward_groups), default=0)
    turn_means: list[float] = []
    for turn_idx in range(max_turns):
        rewards_at_turn = [rewards[turn_idx] for rewards in turn_reward_groups if turn_idx < len(rewards)]
        turn_means.append(mean(rewards_at_turn))
    return [
        [reward - turn_means[turn_idx] for turn_idx, reward in enumerate(rewards)]
        for rewards in turn_reward_groups
    ]


def first_successful_response(rollouts: list[Any], exclude_index: int | None = None) -> str | None:
    for index, rollout in enumerate(rollouts):
        if exclude_index is not None and index == exclude_index:
            continue
        if float(getattr(rollout, "reward", 0.0)) >= 1.0:
            response = str(getattr(rollout, "final_response", "")).strip()
            if response:
                return response
    return None


def write_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=True) + "\n")


def wait_result(value: Any) -> Any:
    if hasattr(value, "result"):
        return value.result()
    return value


def _pid_is_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def acquire_run_lock(log_dir: Path, run_name: str) -> Path:
    log_dir.mkdir(parents=True, exist_ok=True)
    lock_path = log_dir / f"{run_name}.lock"
    if lock_path.exists():
        try:
            pid = int(lock_path.read_text(encoding="utf-8").strip())
        except ValueError:
            pid = -1
        if pid > 0 and _pid_is_running(pid):
            raise SystemExit(
                f"Run {run_name!r} already appears to be active with pid {pid}. "
                "Use a different run name or wait for it to finish."
            )
        lock_path.unlink()
    fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(str(os.getpid()))
    return lock_path


def release_run_lock(lock_path: Path | None) -> None:
    if lock_path is None:
        return
    try:
        if lock_path.read_text(encoding="utf-8").strip() == str(os.getpid()):
            lock_path.unlink()
    except FileNotFoundError:
        return


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
    parser.add_argument(
        "--max-turns",
        type=int,
        default=int(os.environ.get("MAX_TURNS", "0")),
        help="Max assistant turns for sharded_multiturn rows; 0 uses len(shards)+1.",
    )
    parser.add_argument("--max-tokens", type=int, default=int(os.environ.get("MAX_TOKENS", "512")))
    parser.add_argument("--temperature", type=float, default=float(os.environ.get("TEMPERATURE", "1.0")))
    parser.add_argument(
        "--shuffle-seed",
        type=int,
        default=int(os.environ.get("SHUFFLE_SEED", "-1")),
        help="Shuffle training rows before selecting max_steps * batch_size rows; negative disables shuffling.",
    )
    parser.add_argument(
        "--sharded-reward-mode",
        default=os.environ.get("SHARDED_REWARD_MODE", SPARSE_REWARD_MODE),
        help="Reward mode for sharded_multiturn training: sparse, dense/rlrf, or sdpo.",
    )
    parser.add_argument(
        "--sharded-prompt-style",
        default=os.environ.get("SHARDED_PROMPT_STYLE", "default"),
        help="Prompt style for sharded_multiturn rows: default, minimal, linc_math, or tool_schema.",
    )
    parser.add_argument(
        "--sharded-reveal-policy",
        default=os.environ.get("SHARDED_REVEAL_POLICY", "always"),
        help="Hidden-shard reveal policy for sharded_multiturn rows: always or clarify_only.",
    )
    parser.add_argument(
        "--sharded-allow-untagged-final",
        action="store_true",
        default=os.environ.get("SHARDED_ALLOW_UNTAGGED_FINAL", "0").strip().lower()
        in {"1", "true", "yes", "on"},
        help="Allow untagged responses to be scored as final answers for sharded_multiturn rows.",
    )
    parser.add_argument(
        "--sdpo-distill-weight",
        type=float,
        default=float(os.environ.get("SDPO_DISTILL_WEIGHT", "0.1")),
        help="Cross-entropy token weight for Tinker SDPO-style self-teacher targets.",
    )
    parser.add_argument(
        "--sdpo-topk",
        type=int,
        default=int(os.environ.get("SDPO_TOPK", "20")),
        help="Top-k teacher distribution size for SDPO distillation; 0 uses generated-target CE fallback.",
    )
    parser.add_argument(
        "--sdpo-skip-first-n-tokens",
        type=int,
        default=int(os.environ.get("SDPO_SKIP_FIRST_N_TOKENS", "3")),
        help="Skip this many sampled response tokens when applying top-k SDPO CE.",
    )
    parser.add_argument(
        "--sdpo-max-teacher-tokens",
        type=int,
        default=int(os.environ.get("SDPO_MAX_TEACHER_TOKENS", "256")),
        help="Maximum tokens for feedback-conditioned self-teacher completions.",
    )
    parser.add_argument(
        "--sdpo-teacher-temperature",
        type=float,
        default=float(os.environ.get("SDPO_TEACHER_TEMPERATURE", "0.0")),
        help="Sampling temperature for SDPO-style self-teacher completions.",
    )
    parser.add_argument(
        "--sdpo-distill-on",
        default=os.environ.get("SDPO_DISTILL_ON", "failed"),
        help="Which sharded rollouts get SDPO distillation targets: failed, all, or none.",
    )
    parser.add_argument(
        "--sdpo-teacher-prompt-style",
        default=os.environ.get("SDPO_TEACHER_PROMPT_STYLE", "minimal_teacher"),
        help="Feedback teacher prompt style for SDPO: minimal_teacher, enhanced, or brief.",
    )
    parser.add_argument("--learning-rate", type=float, default=float(os.environ.get("LR", "1e-5")))
    parser.add_argument("--lora-rank", type=int, default=int(os.environ.get("LORA_RANK", "32")))
    parser.add_argument("--save-every", type=int, default=int(os.environ.get("SAVE_EVERY", "0")))
    parser.add_argument("--dry-run", action="store_true", help="Validate local inputs without contacting Tinker.")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    args.sharded_reward_mode = normalize_reward_mode(args.sharded_reward_mode)

    split_file = resolve_split_file(args.data_path, args.split)
    rows = load_json_records(split_file)
    if not rows:
        raise SystemExit(f"No records found in {split_file}")
    rows = shuffle_records(rows, args.shuffle_seed)
    max_examples = min(len(rows), args.batch_size * args.max_steps)
    rows = rows[:max_examples]
    LOGGER.info("Loaded %d %s records from %s", len(rows), args.split, split_file)

    if args.dry_run:
        first_messages = row_to_messages(rows[0], prompt_style=args.sharded_prompt_style)
        LOGGER.info("Dry run prompt roles: %s", [msg["role"] for msg in first_messages])
        LOGGER.info(
            "Dry run dataset=%s model=%s sharded_reward_mode=%s sharded_reveal_policy=%s",
            rows[0].get("dataset"),
            args.model_name,
            args.sharded_reward_mode,
            args.sharded_reveal_policy,
        )
        return

    if not os.environ.get("TINKER_API_KEY"):
        raise SystemExit("Set TINKER_API_KEY in the environment before running Tinker jobs.")

    try:
        import tinker
        import torch
        from tinker import types
        from tinker.types.tensor_data import TensorData
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
        sampling_params = types.SamplingParams(
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            stop=renderer.get_stop_sequences(),
        )
        sdpo_sampling_params = types.SamplingParams(
            max_tokens=args.sdpo_max_teacher_tokens,
            temperature=args.sdpo_teacher_temperature,
            stop=renderer.get_stop_sequences(),
        )
        sdpo_forced_sampling_params = types.SamplingParams(
            max_tokens=1,
            temperature=0.0,
            stop=renderer.get_stop_sequences(),
        )
        adam_params = types.AdamParams(
            learning_rate=args.learning_rate,
            beta1=0.9,
            beta2=0.95,
            eps=1e-8,
        )

        LOGGER.info(
            "Using Tinker model=%s renderer=%s sharded_reward_mode=%s sharded_prompt_style=%s sharded_reveal_policy=%s allow_untagged_final=%s",
            args.model_name,
            renderer_name,
            args.sharded_reward_mode,
            args.sharded_prompt_style,
            args.sharded_reveal_policy,
            args.sharded_allow_untagged_final,
        )
        service_client = tinker.ServiceClient(base_url=args.base_url)
        training_client = service_client.create_lora_training_client(
            base_model=args.model_name,
            rank=args.lora_rank,
        )
        total_steps = min(args.max_steps, (len(rows) + args.batch_size - 1) // args.batch_size)

        def add_training_datum(
            datums: list[Any],
            prompt: Any,
            sampled_tokens: list[int],
            sampled_logprobs: list[float],
            advantage: float,
        ) -> bool:
            if abs(advantage) < 1e-12:
                return False
            if not sampled_tokens or not sampled_logprobs:
                return False
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
            return True

        def add_sdpo_distillation_datum(
            datums: list[Any],
            prompt: Any,
            teacher_tokens: list[int],
            weight: float,
        ) -> bool:
            if weight <= 0.0 or not teacher_tokens:
                return False
            model_input = prompt.append(types.EncodedTextChunk(tokens=teacher_tokens))
            weights = torch.zeros(model_input.length, dtype=torch.float32)
            weights[prompt.length :] = float(weight)
            datums.append(datum_from_model_input_weights(model_input, weights))
            return True

        def add_sdpo_topk_distillation_datum(
            datums: list[Any],
            sampling_client: Any,
            student_prompt: Any,
            teacher_prompt: Any,
            sampled_tokens: list[int],
            topk: int,
            weight: float,
            skip_first_n_tokens: int,
        ) -> int:
            if weight <= 0.0 or topk <= 0 or not sampled_tokens:
                return 0

            teacher_forced = teacher_prompt.append(types.EncodedTextChunk(tokens=sampled_tokens))
            teacher_result = wait_result(
                sampling_client.sample(
                    prompt=teacher_forced,
                    num_samples=1,
                    sampling_params=sdpo_forced_sampling_params,
                    include_prompt_logprobs=True,
                    topk_prompt_logprobs=topk,
                )
            )
            topk_all = teacher_result.topk_prompt_logprobs
            if not topk_all:
                return 0

            model_input = student_prompt.append(types.EncodedTextChunk(tokens=sampled_tokens[:-1]))
            target_tokens = torch.zeros(model_input.length, topk, dtype=torch.long)
            weights = torch.zeros(model_input.length, topk, dtype=torch.float32)
            student_completion_start = student_prompt.length - 1
            teacher_completion_start = teacher_prompt.length
            positions_used = 0

            for token_offset in range(len(sampled_tokens)):
                if token_offset < skip_first_n_tokens:
                    continue
                student_pos = student_completion_start + token_offset
                teacher_pos = teacher_completion_start + token_offset
                if student_pos >= model_input.length or teacher_pos >= len(topk_all):
                    continue
                topk_entries = topk_all[teacher_pos]
                if not topk_entries:
                    continue
                token_ids = torch.tensor([token_id for token_id, _logprob in topk_entries[:topk]], dtype=torch.long)
                logprobs = torch.tensor([logprob for _token_id, logprob in topk_entries[:topk]], dtype=torch.float32)
                if token_ids.numel() == 0:
                    continue
                logprobs = logprobs - torch.logsumexp(logprobs, dim=0)
                probs = logprobs.exp() * float(weight)
                k_actual = token_ids.numel()
                target_tokens[student_pos, :k_actual] = token_ids
                weights[student_pos, :k_actual] = probs
                positions_used += 1

            if positions_used == 0:
                return 0
            datums.append(
                types.Datum(
                    model_input=model_input,
                    loss_fn_inputs={
                        "target_tokens": TensorData.from_torch(target_tokens),
                        "weights": TensorData.from_torch(weights),
                    },
                )
            )
            return positions_used

        for step in range(total_steps):
            start_time = time.time()
            batch_rows = rows[step * args.batch_size : (step + 1) * args.batch_size]
            sampling_client = training_client.save_weights_and_get_sampling_client()

            futures = []
            prompts = []
            datums = []
            sdpo_datums = []
            group_rewards: list[float] = []
            sample_count = 0
            sdpo_sample_count = 0
            sharded_flags = [is_sharded_multiturn_row(row) for row in batch_rows]
            if any(sharded_flags) and not all(sharded_flags):
                raise ValueError("Mixed sharded_multiturn and single-turn rows in one batch are not supported.")

            if all(sharded_flags):
                for row in batch_rows:
                    task = ShardedTask.from_row(row, allow_untagged_final=args.sharded_allow_untagged_final)
                    rollouts = []
                    rollout_rewards: list[float] = []
                    turn_reward_groups: list[list[float]] = []
                    for rollout_idx in range(args.rollout_n):
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
                            sampled_logprobs = list(sequence.logprobs or [])
                            parsed_message, _ = renderer.parse_response(sampled_tokens)
                            response = renderers.get_text_content(parsed_message)
                            return SampledResponse(
                                text=response,
                                tokens=sampled_tokens,
                                logprobs=sampled_logprobs,
                                prompt=prompt,
                            )

                        rollout = run_sharded_interaction(
                            task,
                            sample_fn,
                            max_turns=args.max_turns if args.max_turns > 0 else None,
                            prompt_style=args.sharded_prompt_style,
                            reveal_policy=args.sharded_reveal_policy,
                        )
                        rollouts.append(rollout)
                        turn_rewards = rollout_training_rewards(rollout, args.sharded_reward_mode)
                        rollout_rewards.append(rollout_training_reward(rollout, args.sharded_reward_mode))
                        turn_reward_groups.append(turn_rewards)
                        record = rollout.to_log_record()
                        record.update(
                            {
                                "step": step,
                                "idx": row.get("idx"),
                                "dataset": row.get("dataset"),
                                "rollout_idx": rollout_idx,
                                "training_reward": rollout_rewards[-1],
                                "training_turn_rewards": turn_rewards,
                                "sharded_reward_mode": args.sharded_reward_mode,
                                "sharded_reveal_policy": args.sharded_reveal_policy,
                            }
                        )
                        write_jsonl(samples_path, record)

                    reward_mean = mean(rollout_rewards)
                    group_rewards.append(reward_mean)
                    if args.sharded_reward_mode == SPARSE_REWARD_MODE:
                        advantages = [
                            [reward - reward_mean for _turn in rollout.turns]
                            for rollout, reward in zip(rollouts, rollout_rewards)
                        ]
                    else:
                        advantages = centered_turn_advantages(turn_reward_groups)
                    for rollout, rollout_advantages in zip(rollouts, advantages):
                        for turn, advantage in zip(rollout.turns, rollout_advantages):
                            if add_training_datum(
                                datums,
                                turn.prompt,
                                turn.sampled_tokens,
                                turn.sampled_logprobs,
                                advantage,
                            ):
                                sample_count += 1

                    if args.sharded_reward_mode == SDPO_REWARD_MODE:
                        for rollout_idx, rollout in enumerate(rollouts):
                            turn_index = select_sdpo_distillation_turn(rollout, args.sdpo_distill_on)
                            if turn_index is None:
                                continue
                            successful_attempt = first_successful_response(rollouts, exclude_index=rollout_idx)
                            teacher_messages = build_sdpo_teacher_messages(
                                rollout,
                                turn_index,
                                successful_previous_attempt=successful_attempt,
                                teacher_prompt_style=args.sdpo_teacher_prompt_style,
                            )
                            teacher_prompt = renderer.build_generation_prompt(teacher_messages)
                            selected_turn = rollout.turns[turn_index]
                            teacher_response = ""
                            target_tokens = len(selected_turn.sampled_tokens)
                            distill_positions = 0
                            target_mode = "topk"
                            if args.sdpo_topk > 0:
                                distill_positions = add_sdpo_topk_distillation_datum(
                                    sdpo_datums,
                                    sampling_client,
                                    selected_turn.prompt,
                                    teacher_prompt,
                                    selected_turn.sampled_tokens,
                                    args.sdpo_topk,
                                    args.sdpo_distill_weight,
                                    args.sdpo_skip_first_n_tokens,
                                )
                            else:
                                target_mode = "generated"
                                teacher_result = wait_result(
                                    sampling_client.sample(
                                        prompt=teacher_prompt,
                                        num_samples=1,
                                        sampling_params=sdpo_sampling_params,
                                    )
                                )
                                sequence = teacher_result.sequences[0]
                                teacher_tokens = list(sequence.tokens)
                                parsed_message, _ = renderer.parse_response(teacher_tokens)
                                teacher_response = renderers.get_text_content(parsed_message)
                                target_tokens = len(teacher_tokens)
                                if add_sdpo_distillation_datum(
                                    sdpo_datums,
                                    selected_turn.prompt,
                                    teacher_tokens,
                                    args.sdpo_distill_weight,
                                ):
                                    distill_positions = target_tokens

                            if distill_positions > 0:
                                sdpo_sample_count += 1
                                write_jsonl(
                                    samples_path,
                                    {
                                        "step": step,
                                        "idx": row.get("idx"),
                                        "dataset": row.get("dataset"),
                                        "rollout_idx": rollout_idx,
                                        "record_type": "sdpo_teacher",
                                        "target_mode": target_mode,
                                        "distill_turn": selected_turn.turn,
                                        "student_response": selected_turn.response,
                                        "teacher_response": teacher_response,
                                        "teacher_target_tokens": target_tokens,
                                        "distill_positions": distill_positions,
                                        "sdpo_topk": args.sdpo_topk,
                                        "has_successful_peer": successful_attempt is not None,
                                        "sharded_reward_mode": args.sharded_reward_mode,
                                        "sharded_reveal_policy": args.sharded_reveal_policy,
                                    },
                                )
            else:
                for row in batch_rows:
                    prompt = renderer.build_generation_prompt(row_to_messages(row, prompt_style=args.sharded_prompt_style))
                    futures.append(
                        sampling_client.sample(
                            prompt=prompt,
                            num_samples=args.rollout_n,
                            sampling_params=sampling_params,
                        )
                    )
                    prompts.append(prompt)

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

                    for (sampled_tokens, sampled_logprobs, _response, _score), advantage in zip(
                        sampled_payloads, advantages
                    ):
                        if add_training_datum(datums, prompt, sampled_tokens, sampled_logprobs, advantage):
                            sample_count += 1

            train_loss = None
            loss_metrics = {}
            fwd_bwd_metrics = {}
            optim_metrics = {}
            training_batches = []
            if datums:
                training_batches.append(("rl", datums, "importance_sampling"))
            if sdpo_datums:
                training_batches.append(("sdpo", sdpo_datums, "cross_entropy"))

            if training_batches:
                losses: dict[str, Any] = {}
                for label, batch_datums, loss_fn in training_batches:
                    fwd_bwd_result = wait_result(training_client.forward_backward(batch_datums, loss_fn=loss_fn))
                    losses[label] = getattr(fwd_bwd_result, "loss", None)
                    prefix = "fwd_bwd" if label == "rl" and len(training_batches) == 1 else f"fwd_bwd/{label}"
                    fwd_bwd_metrics.update(
                        {
                            f"{prefix}/{key}": value
                            for key, value in (getattr(fwd_bwd_result, "metrics", None) or {}).items()
                        }
                    )
                optim_result = wait_result(training_client.optim_step(adam_params))
                train_loss = losses.get("rl", losses.get("sdpo"))
                loss_metrics = {f"loss/{label}": value for label, value in losses.items() if value is not None}
                optim_metrics = getattr(optim_result, "metrics", None) or {}
            else:
                LOGGER.warning("Step %d produced no trainable RL or SDPO datums; skipped optimizer step.", step)

            if args.save_every > 0 and (step + 1) % args.save_every == 0:
                wait_result(training_client.save_state(name=f"{args.run_name}-step-{step + 1:06d}"))

            metrics = {
                "step": step,
                "examples": len(batch_rows),
                "samples_used": sample_count,
                "sdpo_samples_used": sdpo_sample_count,
                "reward_mean": mean(group_rewards),
                "sharded_reward_mode": args.sharded_reward_mode,
                "sharded_reveal_policy": args.sharded_reveal_policy,
                "loss": train_loss,
                "time_sec": time.time() - start_time,
                **loss_metrics,
                **fwd_bwd_metrics,
                **optim_metrics,
            }
            write_jsonl(metrics_path, metrics)
            LOGGER.info(
                "step=%d reward_mean=%.4f samples_used=%d sdpo_samples_used=%d",
                step,
                metrics["reward_mean"],
                sample_count,
                sdpo_sample_count,
            )

        final_state = wait_result(training_client.save_state(name=f"{args.run_name}-final"))
        final_sampler = wait_result(training_client.save_weights_for_sampler(name=f"{args.run_name}-final-sampler"))
        LOGGER.info("Saved final Tinker state: %s", final_state)
        LOGGER.info("Saved final Tinker sampler weights: %s", final_sampler)
    finally:
        release_run_lock(lock_path)


if __name__ == "__main__":
    main()

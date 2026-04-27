from __future__ import annotations

from typing import Any

from scripts.sharded_multiturn import (
    DENSE_REWARD_MODE,
    ShardedTask,
    contains_environment_impersonation,
    extract_final_answer,
    normalize_reward_mode,
    score_final_answer,
    score_sharded_turn,
)


def compute_score(solution: str, ground_truth: str, extra_info: dict[str, Any] | None = None) -> dict[str, Any]:
    extra_info = extra_info or {}
    kind = str(extra_info.get("reward_kind") or extra_info.get("kind") or "exact")
    reward_mode = normalize_reward_mode(str(extra_info.get("sharded_reward_mode") or "sparse"))
    if reward_mode == DENSE_REWARD_MODE and extra_info.get("revealed_shards_before_turn") is not None:
        task = ShardedTask(
            task_id=str(extra_info.get("index", "")),
            prompt=str(extra_info.get("problem", "")),
            shards=[str(shard) for shard in extra_info.get("shards", [])],
            answer=str(ground_truth),
            kind=kind,
            full_prompt=str(extra_info.get("full_prompt") or "") or None,
        )
        return score_sharded_turn(solution, task, int(extra_info["revealed_shards_before_turn"]))
    if contains_environment_impersonation(solution):
        return {
            "score": 0.0,
            "acc": 0.0,
            "pred": None,
            "incorrect_format": 1,
            "feedback": "Do not invent or write hidden user detail messages; only the user may provide them.",
        }
    return score_final_answer(extract_final_answer(solution), ground_truth, kind)

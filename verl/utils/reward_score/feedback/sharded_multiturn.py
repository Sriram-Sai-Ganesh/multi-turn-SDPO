from __future__ import annotations

from typing import Any

from scripts.sharded_multiturn import contains_environment_impersonation, extract_final_answer, score_final_answer


def compute_score(solution: str, ground_truth: str, extra_info: dict[str, Any] | None = None) -> dict[str, Any]:
    extra_info = extra_info or {}
    kind = str(extra_info.get("reward_kind") or extra_info.get("kind") or "exact")
    if contains_environment_impersonation(solution):
        return {
            "score": 0.0,
            "acc": 0.0,
            "pred": None,
            "incorrect_format": 1,
            "feedback": "Do not invent or write hidden user detail messages; only the user may provide them.",
        }
    return score_final_answer(extract_final_answer(solution), ground_truth, kind)

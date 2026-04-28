"""Helpers for the final-project sharded multi-turn task.

The task format is intentionally JSON-friendly so it can be used by both the
Tinker runner and the local/JHU preprocessing path:

{
  "idx": "example-id",
  "dataset": "sharded_multiturn",
  "kind": "exact",
  "prompt": "underspecified user request",
  "shards": ["hidden fact 1", "hidden fact 2"],
  "answer": "reference final answer"
}
"""

from __future__ import annotations

import json
import math
import re
import string
from dataclasses import dataclass, field
from typing import Any, Callable

DEFAULT_SYSTEM_PROMPT = (
    "You are solving an underspecified multi-turn task. Ask concise clarifying "
    "questions when required information is missing. When you have enough "
    "information, reply with the final answer inside <final_answer>...</final_answer> "
    "or <final>...</final> XML tags. Only user messages may provide hidden "
    "details; never invent hidden details or write messages that pretend to "
    "come from the user."
)
LINC_MATH_SYSTEM_PROMPT = "As an expert problem solver solve step by step the following mathematical questions."
BRIEF_UNDERSPECIFIED_PREFIX = (
    "This may be under-specified. If needed, ask a concise clarifying question "
    "before answering."
)
DEFAULT_PROMPT_STYLE = "default"
MINIMAL_PROMPT_STYLE = "minimal"
LINC_MATH_PROMPT_STYLE = "linc_math"
DEFAULT_TEACHER_PROMPT_STYLE = "default"
BRIEF_TEACHER_PROMPT_STYLE = "brief"

FINAL_TAG_RE = re.compile(
    r"<(?P<tag>final|final[-_]answer|answer)>\s*(?P<answer>.*?)\s*</(?P=tag)>",
    re.IGNORECASE | re.DOTALL,
)
FINAL_PREFIX_RE = re.compile(r"^\s*(?:final answer|answer)\s*:\s*(.+)", re.IGNORECASE | re.DOTALL)
ENVIRONMENT_MESSAGE_RE = re.compile(
    r"\b(?:Additional information|User-provided detail)\s+\d+\s*(?:/|of)\s*\d+\s*:",
    re.IGNORECASE,
)
MCQ_RE = re.compile(r"\b([A-E])\b", re.IGNORECASE)
NUMBER_RE = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")
CLARIFYING_PREFIXES = (
    "clarify",
    "please provide",
    "provide",
    "could you provide",
    "can you provide",
    "please tell",
    "could you tell",
    "can you tell",
    "tell me",
    "share",
    "what are",
    "what is",
    "which",
    "please specify",
    "i need",
    "i need more",
    "i do not have enough",
    "i don't have enough",
    "more information",
)
FINAL_MARKUP_RE = re.compile(
    r"<\s*/?\s*(?:final|final[-_]answer|answer)\b|^\s*(?:final answer|answer)\s*:",
    re.IGNORECASE | re.MULTILINE,
)
SPARSE_REWARD_MODE = "sparse"
DENSE_REWARD_MODE = "dense"
SDPO_REWARD_MODE = "sdpo"
DENSE_CLARIFICATION_REWARD = 0.25


@dataclass(frozen=True)
class ShardedTask:
    task_id: str
    prompt: str
    shards: list[str]
    answer: str
    kind: str = "exact"
    system: str = DEFAULT_SYSTEM_PROMPT
    full_prompt: str | None = None
    allow_untagged_final: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_row(cls, row: dict[str, Any], allow_untagged_final: bool = False) -> "ShardedTask":
        prompt = _first_present(
            row,
            "prompt",
            "underspecified_prompt",
            "initial_prompt",
            "sharded_prompt",
            "question",
        )
        answer = _first_present(row, "answer", "final_answer", "target", "reference", "ground_truth")
        shards = _extract_shards(row)
        kind = str(row.get("reward_kind") or row.get("answer_kind") or row.get("kind") or "exact")
        if kind == "sharded_multiturn":
            kind = "exact"
        return cls(
            task_id=str(row.get("idx", row.get("task_id", row.get("id", row.get("question_id", ""))))),
            prompt=str(prompt),
            shards=shards,
            answer=str(answer),
            kind=kind,
            system=str(row.get("system") or DEFAULT_SYSTEM_PROMPT),
            full_prompt=_optional_str(row.get("full_prompt") or row.get("fully_specified_prompt")),
            allow_untagged_final=allow_untagged_final,
            metadata={k: v for k, v in row.items() if k not in {"prompt", "answer", "shards", "system"}},
        )


@dataclass
class SampledResponse:
    text: str
    tokens: list[int] = field(default_factory=list)
    logprobs: list[float] = field(default_factory=list)
    prompt: Any = None


@dataclass
class AssistantTurn:
    turn: int
    prompt_messages: list[dict[str, str]]
    response: str
    sampled_tokens: list[int] = field(default_factory=list)
    sampled_logprobs: list[float] = field(default_factory=list)
    prompt: Any = None


@dataclass
class MultiturnRollout:
    task: ShardedTask
    turns: list[AssistantTurn]
    score: dict[str, Any]
    revealed_shards: int
    transcript: list[dict[str, str]]
    turn_scores: list[dict[str, Any]] = field(default_factory=list)

    @property
    def reward(self) -> float:
        return float(self.score.get("score", 0.0))

    @property
    def dense_reward(self) -> float:
        if not self.turn_scores:
            return self.reward
        return sum(float(score.get("score", 0.0)) for score in self.turn_scores) / len(self.turn_scores)

    @property
    def final_response(self) -> str:
        if not self.turns:
            return ""
        return self.turns[-1].response

    def to_log_record(self) -> dict[str, Any]:
        return {
            "task_id": self.task.task_id,
            "reward": self.reward,
            "score": self.score,
            "dense_reward": self.dense_reward,
            "revealed_shards": self.revealed_shards,
            "turns": len(self.turns),
            "turn_scores": self.turn_scores,
            "final_response": self.final_response,
            "transcript": self.transcript,
        }


def _first_present(row: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return value
    raise ValueError(f"Missing required field; tried: {', '.join(keys)}")


def _optional_str(value: Any) -> str | None:
    if value in (None, ""):
        return None
    return str(value)


def _extract_shards(row: dict[str, Any]) -> list[str]:
    raw = None
    for key in ("shards", "instruction_shards", "user_shards", "hidden_shards", "turns"):
        if row.get(key) not in (None, ""):
            raw = row[key]
            break
    if raw is None:
        return []
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            raw = parsed
        except json.JSONDecodeError:
            return [raw]
    if not isinstance(raw, list):
        raw = [raw]
    return [_stringify_shard(shard) for shard in raw]


def _stringify_shard(shard: Any) -> str:
    if isinstance(shard, dict):
        for key in ("shard_text", "content", "text", "shard", "message", "value"):
            if shard.get(key) not in (None, ""):
                return str(shard[key])
        return json.dumps(shard, sort_keys=True)
    return str(shard)


def normalize_prompt_style(style: str | None) -> str:
    normalized = (style or DEFAULT_PROMPT_STYLE).strip().lower().replace("-", "_")
    if normalized in {DEFAULT_PROMPT_STYLE, "verbose"}:
        return DEFAULT_PROMPT_STYLE
    if normalized in {MINIMAL_PROMPT_STYLE, "bare", "question_only", "question"}:
        return MINIMAL_PROMPT_STYLE
    if normalized in {LINC_MATH_PROMPT_STYLE, "paper_math", "lost_in_conversation_math"}:
        return LINC_MATH_PROMPT_STYLE
    raise ValueError(f"Unsupported sharded prompt style: {style!r}")


def normalize_teacher_prompt_style(style: str | None) -> str:
    normalized = (style or DEFAULT_TEACHER_PROMPT_STYLE).strip().lower().replace("-", "_")
    if normalized in {DEFAULT_TEACHER_PROMPT_STYLE, "verbose"}:
        return DEFAULT_TEACHER_PROMPT_STYLE
    if normalized in {BRIEF_TEACHER_PROMPT_STYLE, "minimal", "short"}:
        return BRIEF_TEACHER_PROMPT_STYLE
    raise ValueError(f"Unsupported teacher prompt style: {style!r}")


def _is_math_task(task: ShardedTask) -> bool:
    return str(task.metadata.get("source_task") or "").lower() == "math" or task.kind.lower() in {"number", "numeric", "gsm8k"}


def initial_messages(task: ShardedTask, prompt_style: str | None = None) -> list[dict[str, str]]:
    style = normalize_prompt_style(prompt_style)
    if style == MINIMAL_PROMPT_STYLE:
        return [{"role": "user", "content": task.prompt}]
    if style == LINC_MATH_PROMPT_STYLE and _is_math_task(task):
        return [
            {"role": "system", "content": LINC_MATH_SYSTEM_PROMPT},
            {"role": "user", "content": f"Q: {task.prompt}\nA:"},
        ]
    if style == LINC_MATH_PROMPT_STYLE:
        return [{"role": "user", "content": task.prompt}]
    return [
        {"role": "system", "content": task.system},
        {"role": "user", "content": task.prompt},
    ]


def shard_message(shard: str, shard_index: int, shard_count: int, prompt_style: str | None = None) -> str:
    if normalize_prompt_style(prompt_style) != DEFAULT_PROMPT_STYLE:
        return shard
    return (
        f"{shard}\n\n"
        "Ask another concise clarifying question if more information is needed. "
        "If you can answer now, use the required final-answer XML format."
    )


def no_more_shards_message(prompt_style: str | None = None) -> str:
    if normalize_prompt_style(prompt_style) != DEFAULT_PROMPT_STYLE:
        return "No more information is available."
    return "No more hidden information is available. Reply with your best final answer using final XML tags."


def extract_final_answer(response: str, allow_untagged: bool = False) -> str | None:
    tag_match = FINAL_TAG_RE.search(response)
    if tag_match:
        candidate = tag_match.group("answer").strip()
        if is_clarifying_text(candidate) or is_placeholder_final_text(candidate):
            return None
        return candidate
    lines = [line.strip() for line in response.splitlines() if line.strip()]
    prefix_match = FINAL_PREFIX_RE.search(lines[-1] if lines else response)
    if prefix_match:
        candidate = prefix_match.group(1).strip()
        if is_clarifying_text(candidate) or is_placeholder_final_text(candidate):
            return None
        return candidate
    if allow_untagged and not contains_final_answer_markup(response):
        candidate = response.strip()
        if is_clarifying_text(candidate) or is_placeholder_final_text(candidate):
            return None
        return candidate
    return None


def contains_environment_impersonation(response: str) -> bool:
    return ENVIRONMENT_MESSAGE_RE.search(response) is not None


def _environment_impersonation_score() -> dict[str, Any]:
    return {
            "score": 0.0,
            "acc": 0.0,
            "pred": None,
            "incorrect_format": 1,
            "feedback": "Do not invent or write hidden user detail messages; only the user may provide them.",
        }


def is_clarifying_text(value: str) -> bool:
    normalized = value.strip().lower()
    if not normalized:
        return False
    if "?" in normalized:
        return True
    return normalized.startswith(CLARIFYING_PREFIXES)


def contains_final_answer_markup(response: str) -> bool:
    return FINAL_MARKUP_RE.search(response) is not None


def is_placeholder_final_text(value: str) -> bool:
    normalized = value.strip().lower()
    if not normalized:
        return True
    if normalized in {"...", "answer", "your answer", "final answer"}:
        return True
    return not any(ch.isalnum() for ch in normalized)


def normalize_answer(value: str) -> str:
    value = value.strip().lower()
    value = value.translate(str.maketrans("", "", string.punctuation))
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def _last_number(value: str) -> float | None:
    matches = NUMBER_RE.findall(value.replace(",", ""))
    if not matches:
        return None
    try:
        return float(matches[-1])
    except ValueError:
        return None


def _mcq_letter(value: str) -> str | None:
    match = MCQ_RE.search(value)
    if not match:
        return None
    return match.group(1).upper()


def score_final_answer(prediction: str | None, reference: str, kind: str = "exact") -> dict[str, Any]:
    if prediction is None:
        return {
            "score": 0.0,
            "acc": 0.0,
            "pred": None,
            "incorrect_format": 1,
            "feedback": (
                "Your response must include the final answer inside "
                "<final_answer>...</final_answer>, <final>...</final>, or <answer>...</answer>."
            ),
        }

    kind = kind.lower()
    reference = str(reference)
    correct = False
    if kind in {"number", "numeric", "gsm8k"}:
        pred_num = _last_number(prediction)
        ref_num = _last_number(reference)
        correct = pred_num is not None and ref_num is not None and math.isclose(
            pred_num, ref_num, rel_tol=1e-6, abs_tol=1e-6
        )
    elif kind in {"mcq", "multiple_choice"}:
        correct = _mcq_letter(prediction) == _mcq_letter(reference)
    elif kind == "contains":
        correct = normalize_answer(reference) in normalize_answer(prediction)
    else:
        correct = normalize_answer(prediction) == normalize_answer(reference)

    return {
        "score": 1.0 if correct else 0.0,
        "acc": 1.0 if correct else 0.0,
        "pred": prediction,
        "incorrect_format": 0,
        "feedback": "" if correct else "The final answer is incorrect.",
    }


def score_sharded_response(response: str, task: ShardedTask) -> dict[str, Any]:
    if contains_environment_impersonation(response):
        return _environment_impersonation_score()
    return score_final_answer(extract_final_answer(response, task.allow_untagged_final), task.answer, task.kind)


def normalize_reward_mode(mode: str | None) -> str:
    normalized = (mode or SPARSE_REWARD_MODE).strip().lower().replace("-", "_")
    if normalized in {SPARSE_REWARD_MODE, "terminal", "final", "baseline"}:
        return SPARSE_REWARD_MODE
    if normalized in {DENSE_REWARD_MODE, "rlrf", "rich_feedback", "turn", "turn_level"}:
        return DENSE_REWARD_MODE
    if normalized in {SDPO_REWARD_MODE, "self_distill", "self_distillation", "feedback_distill"}:
        return SDPO_REWARD_MODE
    raise ValueError(f"Unsupported sharded reward mode: {mode!r}")


def score_sharded_turn(response: str, task: ShardedTask, revealed_shards_before_turn: int) -> dict[str, Any]:
    """Return the dense teacher/rubric score for one assistant turn.

    The rubric uses privileged knowledge of the hidden shard schedule. It does
    not reveal hidden content to the policy; it only distinguishes information
    seeking from premature final-answer behavior.
    """
    total_shards = len(task.shards)
    if contains_environment_impersonation(response):
        score = _environment_impersonation_score()
        score.update({"dense_action": "environment_impersonation", "revealed_shards": revealed_shards_before_turn})
        return score

    prediction = extract_final_answer(response, task.allow_untagged_final)
    if prediction is not None:
        if revealed_shards_before_turn < total_shards:
            missing = total_shards - revealed_shards_before_turn
            feedback_prefix = "The privileged full instruction still has hidden details." if task.full_prompt else "Hidden details remain."
            return {
                "score": 0.0,
                "acc": 0.0,
                "pred": prediction,
                "incorrect_format": 0,
                "premature_final": 1,
                "dense_action": "premature_final",
                "revealed_shards": revealed_shards_before_turn,
                "missing_shards": missing,
                "feedback": f"{feedback_prefix} Ask for more information before giving a final answer.",
            }
        final_score = score_final_answer(prediction, task.answer, task.kind)
        final_score.update(
            {
                "premature_final": 0,
                "dense_action": "final_answer",
                "revealed_shards": revealed_shards_before_turn,
                "missing_shards": 0,
            }
        )
        return final_score

    if contains_final_answer_markup(response):
        return {
            "score": 0.0,
            "acc": 0.0,
            "pred": None,
            "incorrect_format": 1,
            "premature_final": 0,
            "dense_action": "malformed_final_markup",
            "revealed_shards": revealed_shards_before_turn,
            "missing_shards": max(total_shards - revealed_shards_before_turn, 0),
            "feedback": "Do not put clarifying questions or placeholders inside final-answer tags.",
        }

    if revealed_shards_before_turn < total_shards:
        if is_clarifying_text(response):
            return {
                "score": DENSE_CLARIFICATION_REWARD,
                "acc": 0.0,
                "pred": None,
                "incorrect_format": 0,
                "premature_final": 0,
                "dense_action": "clarify",
                "revealed_shards": revealed_shards_before_turn,
                "missing_shards": total_shards - revealed_shards_before_turn,
                "feedback": "Good: ask for missing information before answering.",
            }
        return {
            "score": 0.0,
            "acc": 0.0,
            "pred": None,
            "incorrect_format": 0,
            "premature_final": 0,
            "dense_action": "non_clarifying",
            "revealed_shards": revealed_shards_before_turn,
            "missing_shards": total_shards - revealed_shards_before_turn,
            "feedback": "Hidden details remain; ask a concise clarifying question.",
        }

    return {
        "score": 0.0,
        "acc": 0.0,
        "pred": None,
        "incorrect_format": 1,
        "premature_final": 0,
        "dense_action": "missing_final_after_all_shards",
        "revealed_shards": revealed_shards_before_turn,
        "missing_shards": 0,
        "feedback": "All hidden details have been revealed; provide the final answer in XML tags.",
    }


def rollout_training_rewards(rollout: MultiturnRollout, reward_mode: str | None) -> list[float]:
    mode = normalize_reward_mode(reward_mode)
    if mode == SPARSE_REWARD_MODE:
        return [rollout.reward for _turn in rollout.turns]
    if len(rollout.turn_scores) == len(rollout.turns):
        return [float(score.get("score", 0.0)) for score in rollout.turn_scores]
    return [
        float(score_sharded_turn(turn.response, rollout.task, _revealed_before_turn(rollout, index)).get("score", 0.0))
        for index, turn in enumerate(rollout.turns)
    ]


def rollout_training_reward(rollout: MultiturnRollout, reward_mode: str | None) -> float:
    rewards = rollout_training_rewards(rollout, reward_mode)
    if not rewards:
        return 0.0
    return sum(rewards) / len(rewards)


def _is_good_dense_action(score: dict[str, Any]) -> bool:
    action = str(score.get("dense_action", ""))
    if action == "clarify":
        return True
    if action == "final_answer" and float(score.get("score", 0.0)) >= 1.0:
        return True
    return False


def select_sdpo_distillation_turn(rollout: MultiturnRollout, distill_on: str = "failed") -> int | None:
    """Pick one assistant turn whose behavior should receive self-teacher feedback.

    The Tinker SDPO approximation keeps the expensive feedback-conditioned
    distillation pass bounded by defaulting to one datum per failed rollout.
    It chooses the first turn where the dense teacher identified a behavioral
    mistake, such as a premature final answer or a missing final answer after
    all shards were revealed.
    """
    normalized = (distill_on or "failed").strip().lower().replace("-", "_")
    if normalized in {"none", "off", "false", "0"}:
        return None
    if normalized not in {"failed", "all"}:
        raise ValueError(f"Unsupported SDPO distillation selector: {distill_on!r}")
    if normalized == "failed" and rollout.reward >= 1.0:
        return None

    for index, score in enumerate(rollout.turn_scores):
        if not _is_good_dense_action(score):
            return index
    if normalized == "all" and rollout.turns:
        return len(rollout.turns) - 1
    if rollout.reward < 1.0 and rollout.turns:
        return len(rollout.turns) - 1
    return None


def rollout_feedback_summary(rollout: MultiturnRollout, turn_index: int) -> str:
    if not rollout.turns:
        return "No assistant response was sampled."
    turn_index = min(max(turn_index, 0), len(rollout.turns) - 1)
    turn = rollout.turns[turn_index]
    score = rollout.turn_scores[turn_index] if turn_index < len(rollout.turn_scores) else {}
    feedback = str(score.get("feedback") or rollout.score.get("feedback") or "No feedback was provided.")
    action = str(score.get("dense_action") or "unknown")
    revealed = int(score.get("revealed_shards", _revealed_before_turn(rollout, turn_index)))
    missing = max(len(rollout.task.shards) - revealed, 0)
    return (
        f"Turn {turn.turn} action: {action}.\n"
        f"Hidden shards revealed before this turn: {revealed}/{len(rollout.task.shards)}.\n"
        f"Hidden shards still unavailable to the student before this turn: {missing}.\n"
        f"Feedback: {feedback}\n"
        f"Student response on this turn:\n{turn.response}"
    )


def build_sdpo_teacher_messages(
    rollout: MultiturnRollout,
    turn_index: int,
    successful_previous_attempt: str | None = None,
    teacher_prompt_style: str | None = None,
) -> list[dict[str, str]]:
    """Build a feedback-conditioned self-teacher prompt for one sharded turn.

    The teacher is allowed to see privileged information, including the fully
    specified prompt and the rubric feedback. The generated target is later
    distilled onto the original student prompt for the selected turn, so this
    helper must not modify the student transcript itself.
    """
    turn_index = min(max(turn_index, 0), len(rollout.turns) - 1)
    turn = rollout.turns[turn_index]
    revealed = _revealed_before_turn(rollout, turn_index)
    missing = max(len(rollout.task.shards) - revealed, 0)
    style = normalize_teacher_prompt_style(teacher_prompt_style)

    if missing > 0:
        ideal_behavior = (
            "The student has not yet received all hidden details. The ideal next "
            "assistant message should ask one concise clarifying question and must "
            "not reveal, infer, or use hidden shard content that is unavailable in "
            "the original conversation."
        )
    else:
        ideal_behavior = (
            "All hidden details have been revealed in the original conversation. "
            "The ideal next assistant message should provide the final answer in "
            "the required XML final-answer format."
        )

    full_instruction = rollout.task.full_prompt or "(not provided)"
    shard_lines = "\n".join(f"{idx + 1}. {shard}" for idx, shard in enumerate(rollout.task.shards))
    demonstration = successful_previous_attempt or "(no successful peer attempt available)"
    conversation = "\n".join(
        f"{message.get('role', 'unknown')}: {message.get('content', '')}"
        for message in turn.prompt_messages
    )
    if style == BRIEF_TEACHER_PROMPT_STYLE:
        teacher_user = (
            f"{BRIEF_UNDERSPECIFIED_PREFIX}\n\n"
            f"Original conversation:\n{conversation}\n\n"
            f"Full instruction, visible only to you:\n{full_instruction}\n\n"
            f"Hidden details, visible only to you:\n{shard_lines or '(none)'}\n\n"
            f"Reference answer, visible only to you:\n{rollout.task.answer}\n\n"
            f"Feedback for the selected turn:\n{rollout_feedback_summary(rollout, turn_index)}\n\n"
            f"Ideal behavior:\n{ideal_behavior}\n\n"
            "Return only the next assistant message."
        )
        return [
            {"role": "system", "content": "Write only the target assistant message."},
            {"role": "user", "content": teacher_user},
        ]

    teacher_user = (
        "You are the feedback-conditioned self-teacher for a sharded multi-turn task.\n"
        "Use the privileged context and feedback to write the next assistant message "
        "that the student should have produced from the original conversation state.\n\n"
        f"Original conversation before the selected assistant turn:\n{conversation}\n\n"
        f"Fully specified instruction visible only to the teacher:\n{full_instruction}\n\n"
        f"All hidden shards visible only to the teacher:\n{shard_lines or '(none)'}\n\n"
        f"Reference final answer visible only to the teacher:\n{rollout.task.answer}\n\n"
        f"Successful peer attempt, if any:\n{demonstration}\n\n"
        f"Rubric feedback for the selected turn:\n{rollout_feedback_summary(rollout, turn_index)}\n\n"
        f"Ideal behavior:\n{ideal_behavior}\n\n"
        "Return only the replacement assistant message. Do not include analysis "
        "headers or mention that you saw privileged context."
    )
    return [
        {
            "role": "system",
            "content": (
                "You are a precise self-teacher for policy distillation. Produce "
                "only the target assistant message."
            ),
        },
        {"role": "user", "content": teacher_user},
    ]


def _revealed_before_turn(rollout: MultiturnRollout, turn_index: int) -> int:
    revealed = 0
    for message in rollout.turns[turn_index].prompt_messages:
        if message.get("role") == "user" and message.get("content") != rollout.task.prompt:
            revealed += 1
    return min(revealed, len(rollout.task.shards))


def _coerce_sample(sample: str | SampledResponse) -> SampledResponse:
    if isinstance(sample, SampledResponse):
        return sample
    return SampledResponse(text=str(sample))


def run_sharded_interaction(
    task: ShardedTask,
    sample_fn: Callable[[list[dict[str, str]]], str | SampledResponse],
    max_turns: int | None = None,
    prompt_style: str | None = None,
) -> MultiturnRollout:
    prompt_style = normalize_prompt_style(prompt_style)
    messages = initial_messages(task, prompt_style=prompt_style)
    turns: list[AssistantTurn] = []
    turn_scores: list[dict[str, Any]] = []
    revealed_shards = 0
    max_turns = max_turns or max(len(task.shards) + 1, 1)

    for turn_idx in range(max_turns):
        prompt_messages = [dict(message) for message in messages]
        revealed_before_turn = revealed_shards
        sample = _coerce_sample(sample_fn(prompt_messages))
        turns.append(
            AssistantTurn(
                turn=turn_idx,
                prompt_messages=prompt_messages,
                response=sample.text,
                sampled_tokens=sample.tokens,
                sampled_logprobs=sample.logprobs,
                prompt=sample.prompt,
            )
        )
        turn_scores.append(score_sharded_turn(sample.text, task, revealed_before_turn))
        messages.append({"role": "assistant", "content": sample.text})
        if contains_environment_impersonation(sample.text):
            break
        if extract_final_answer(sample.text) is not None:
            break
        if revealed_shards < len(task.shards):
            messages.append(
                {
                    "role": "user",
                    "content": shard_message(
                        task.shards[revealed_shards],
                        revealed_shards,
                        len(task.shards),
                        prompt_style=prompt_style,
                    ),
                }
            )
            revealed_shards += 1
        elif turn_idx < max_turns - 1:
            messages.append({"role": "user", "content": no_more_shards_message(prompt_style=prompt_style)})

    final_response = turns[-1].response if turns else ""
    if any(contains_environment_impersonation(turn.response) for turn in turns):
        score = _environment_impersonation_score()
    else:
        score = score_sharded_response(final_response, task)
    return MultiturnRollout(
        task=task,
        turns=turns,
        score=score,
        revealed_shards=revealed_shards,
        transcript=messages,
        turn_scores=turn_scores,
    )

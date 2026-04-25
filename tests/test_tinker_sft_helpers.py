from scripts.tinker_grpo import score_response
from scripts.tinker_sft import (
    DEFAULT_CLARIFYING_QUESTION,
    build_supervised_examples,
    format_math_target,
    format_sft_target_for_row,
    format_tooluse_target,
)


def test_format_tooluse_target_matches_reward_parser():
    row = {
        "idx": 1,
        "dataset": "tooluse",
        "kind": "tooluse",
        "prompt": "p",
        "answer": '[{"Action": "search", "Action_Input": "{\\"query\\": \\"abc\\"}"}]',
    }

    target = format_sft_target_for_row(row)

    assert target == 'Action: search\nAction Input: {"query": "abc"}'
    assert score_response(row, target, "train")["score"] == 1.0


def test_format_math_target_boxes_final_numeric_answer():
    assert format_math_target("21") == r"\boxed{21}"
    assert format_math_target("reasoning\n#### 21") == r"\boxed{21}"


def test_format_mcq_target_matches_reward_parser():
    row = {
        "idx": 1,
        "dataset": "sciknoweval",
        "kind": "mcq",
        "prompt": "p",
        "answer": "B",
    }

    target = format_sft_target_for_row(row)

    assert target == "<answer>\nB\n</answer>"
    assert score_response(row, target, "train")["score"] == 1.0


def test_build_supervised_examples_for_sharded_rows_expand_clarify_then_final():
    row = {
        "idx": "x",
        "dataset": "sharded_multiturn",
        "kind": "number",
        "prompt": "Add hidden numbers.",
        "shards": ["First is 1.", "Second is 2."],
        "answer": "3",
    }

    examples = build_supervised_examples(row, prompt_style="minimal")

    assert len(examples) == 3
    assert examples[0]["target"] == DEFAULT_CLARIFYING_QUESTION
    assert examples[1]["target"] == DEFAULT_CLARIFYING_QUESTION
    assert examples[2]["target"] == "<final>3</final>"
    assert examples[1]["messages"][-1] == {"role": "user", "content": "First is 1."}
    assert examples[2]["messages"][-1] == {"role": "user", "content": "Second is 2."}


def test_build_supervised_examples_for_sharded_actions_wraps_final_answer():
    row = {
        "idx": "x",
        "dataset": "sharded_multiturn",
        "kind": "exact",
        "prompt": "Calculate the CAGR for these investments",
        "shards": ["Investment 1 details.", "Investment 2 details."],
        "answer": '{"calculate_cagr": {"final_value": [7000], "initial_value": [5000], "period_in_years": [5]}}',
    }

    examples = build_supervised_examples(row)

    assert examples[-1]["target"].startswith("<final>")
    assert examples[-1]["target"].endswith("</final>")


def test_format_tooluse_target_preserves_fallback_for_unparseable_text():
    raw = "Action: search\nAction Input: {not json}"

    assert format_tooluse_target(raw) == raw

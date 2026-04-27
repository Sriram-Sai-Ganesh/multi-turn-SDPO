from scripts.sharded_multiturn import (
    SampledResponse,
    ShardedTask,
    contains_environment_impersonation,
    extract_final_answer,
    run_sharded_interaction,
    score_final_answer,
    score_sharded_response,
)
from scripts.tinker_grpo import load_feedback_module, row_to_messages, score_response


def test_extract_final_answer_from_tags_and_prefix():
    assert extract_final_answer("work\n<final>42</final>") == "42"
    assert extract_final_answer("<final-answer>24</final-answer>") == "24"
    assert extract_final_answer("<answer>44</answer>") == "44"
    assert extract_final_answer("Final answer: ocean blue") == "ocean blue"
    assert extract_final_answer("I need more info.") is None
    assert extract_final_answer("<final>Clarify: What are the two hidden numbers?</final>") is None
    assert extract_final_answer("reply with <final>...</final>.") is None
    assert extract_final_answer("Reasoning...\nFinal answer:\n<final") is None


def test_score_final_answer_supports_exact_number_and_mcq():
    assert score_final_answer("Ocean blue.", "ocean blue", "exact")["score"] == 1.0
    assert score_final_answer("The answer is 5.", "5", "number")["score"] == 1.0
    assert score_final_answer("I choose B", "B", "mcq")["score"] == 1.0
    assert score_final_answer(None, "B", "mcq")["incorrect_format"] == 1


def test_sharded_task_row_to_messages_uses_default_system():
    row = {
        "idx": "x",
        "dataset": "sharded_multiturn",
        "kind": "number",
        "prompt": "Add hidden numbers.",
        "shards": ["First is 1.", "Second is 2."],
        "answer": "3",
    }

    messages = row_to_messages(row)

    assert messages[0]["role"] == "system"
    assert messages[1] == {"role": "user", "content": "Add hidden numbers."}


def test_run_sharded_interaction_reveals_shards_until_final():
    task = ShardedTask(
        task_id="x",
        prompt="Add hidden numbers.",
        shards=["First is 1.", "Second is 2."],
        answer="3",
        kind="number",
    )
    responses = iter(
        [
            "What is the first number?",
            "What is the second number?",
            "<final>3</final>",
        ]
    )

    rollout = run_sharded_interaction(task, lambda _messages: SampledResponse(text=next(responses)))

    assert rollout.reward == 1.0
    assert rollout.revealed_shards == 2
    assert len(rollout.turns) == 3
    assert rollout.transcript[-1]["content"] == "<final>3</final>"


def test_score_response_handles_sharded_multiturn_rows():
    row = {
        "idx": "x",
        "dataset": "sharded_multiturn",
        "kind": "number",
        "prompt": "Add hidden numbers.",
        "shards": ["First is 1.", "Second is 2."],
        "answer": "3",
    }

    score = score_response(row, "<final>3</final>", "train")

    assert score["score"] == 1.0
    assert score["incorrect_format"] == 0


def test_score_rejects_assistant_environment_impersonation():
    task = ShardedTask(
        task_id="x",
        prompt="Solve the hidden task.",
        shards=["The answer is 8."],
        answer="8",
        kind="number",
    )
    response = "Additional information 1/1: The answer is 8.\n<final_answer>8</final_answer>"

    assert contains_environment_impersonation(response)
    assert contains_environment_impersonation("User-provided detail 1 of 1: The answer is 8.")
    score = score_sharded_response(response, task)

    assert score["score"] == 0.0
    assert score["incorrect_format"] == 1
    assert "user detail" in score["feedback"]


def test_run_sharded_interaction_penalizes_earlier_environment_impersonation():
    task = ShardedTask(
        task_id="x",
        prompt="Solve the hidden task.",
        shards=["The answer is 8."],
        answer="8",
        kind="number",
    )
    responses = iter(
        [
            "Additional information 1/1: The answer is 8.",
            "<final_answer>8</final_answer>",
        ]
    )

    rollout = run_sharded_interaction(task, lambda _messages: SampledResponse(text=next(responses)))

    assert rollout.reward == 0.0
    assert rollout.score["incorrect_format"] == 1
    assert len(rollout.turns) == 1


def test_local_reward_module_handles_sharded_multiturn_rows():
    sharded_multiturn = load_feedback_module("sharded_multiturn")

    score = sharded_multiturn.compute_score(
        "<final>24</final>",
        "24",
        {"reward_kind": "number"},
    )

    assert score["score"] == 1.0

    impersonation_score = sharded_multiturn.compute_score(
        "Additional information 1/1: hidden answer.\n<final>24</final>",
        "24",
        {"reward_kind": "number"},
    )

    assert impersonation_score["score"] == 0.0
    assert impersonation_score["incorrect_format"] == 1

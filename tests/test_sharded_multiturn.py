from scripts.sharded_multiturn import (
    DENSE_CLARIFICATION_REWARD,
    BRIEF_UNDERSPECIFIED_PREFIX,
    ENHANCED_TEACHER_PROMPT_STYLE,
    MINIMAL_TEACHER_PROMPT_STYLE,
    SDPO_REWARD_MODE,
    SampledResponse,
    ShardedTask,
    build_sdpo_teacher_messages,
    contains_environment_impersonation,
    extract_final_answer,
    initial_messages,
    normalize_reward_mode,
    run_sharded_interaction,
    shard_message,
    score_final_answer,
    score_sharded_response,
    score_sharded_turn,
    rollout_training_reward,
    rollout_training_rewards,
    select_sdpo_distillation_turn,
    system_prompt_for_task,
)
from scripts.tinker_grpo import centered_turn_advantages
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
    assert extract_final_answer("The answer is 42.", allow_untagged=True) == "The answer is 42."
    assert extract_final_answer("What number should I use?", allow_untagged=True) is None


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

    task = ShardedTask.from_row(row)
    assert messages[0] == {"role": "system", "content": system_prompt_for_task(task)}
    assert messages[1] == {"role": "user", "content": "Q: Add hidden numbers.\nA:"}


def test_minimal_prompt_style_uses_minimal_system_and_plain_shards():
    task = ShardedTask(
        task_id="x",
        prompt="Add hidden numbers.",
        shards=["First is 1."],
        answer="1",
        kind="number",
    )

    assert initial_messages(task, prompt_style="minimal") == [
        {"role": "system", "content": "As an expert problem solver solve step by step the following mathematical question."},
        {"role": "user", "content": "Q: Add hidden numbers.\nA:"},
    ]
    assert shard_message("First is 1.", 0, 1, prompt_style="minimal") == "First is 1."


def test_linc_math_prompt_style_uses_minimal_math_prompt_shape():
    task = ShardedTask(
        task_id="x",
        prompt="How many apples are left?",
        shards=["There were 5 apples."],
        answer="5",
        kind="number",
        metadata={"source_task": "math"},
    )

    messages = initial_messages(task, prompt_style="linc_math")

    assert messages[0]["content"] == "As an expert problem solver solve step by step the following mathematical question."
    assert messages[1] == {"role": "user", "content": "Q: How many apples are left?\nA:"}


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
    assert rollout.dense_reward == (DENSE_CLARIFICATION_REWARD + DENSE_CLARIFICATION_REWARD + 1.0) / 3
    assert rollout.revealed_shards == 2
    assert len(rollout.turns) == 3
    assert [score["dense_action"] for score in rollout.turn_scores] == ["clarify", "clarify", "final_answer"]
    assert rollout.transcript[-1]["content"] == "<final>3</final>"


def test_run_sharded_interaction_can_score_untagged_final_in_prompt_ablation():
    task = ShardedTask(
        task_id="x",
        prompt="Add hidden numbers.",
        shards=["First is 1.", "Second is 2."],
        answer="3",
        kind="number",
        allow_untagged_final=True,
    )
    responses = iter(["What is the first number?", "What is the second number?", "The answer is 3."])

    rollout = run_sharded_interaction(
        task,
        lambda _messages: SampledResponse(text=next(responses)),
        prompt_style="minimal",
    )

    assert rollout.reward == 1.0
    assert rollout.turn_scores[-1]["dense_action"] == "final_answer"


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


def test_dense_turn_scoring_rewards_clarification_and_penalizes_premature_final():
    task = ShardedTask(
        task_id="x",
        prompt="Add hidden numbers.",
        shards=["First is 1.", "Second is 2."],
        answer="3",
        kind="number",
        full_prompt="Add 1 and 2.",
    )

    clarify_score = score_sharded_turn("What is the first number?", task, revealed_shards_before_turn=0)
    premature_score = score_sharded_turn("<final>3</final>", task, revealed_shards_before_turn=1)
    final_score = score_sharded_turn("<final>3</final>", task, revealed_shards_before_turn=2)

    assert clarify_score["score"] == DENSE_CLARIFICATION_REWARD
    assert clarify_score["dense_action"] == "clarify"
    assert premature_score["score"] == 0.0
    assert premature_score["premature_final"] == 1
    assert "privileged full instruction" in premature_score["feedback"].lower()
    assert final_score["score"] == 1.0
    assert final_score["dense_action"] == "final_answer"


def test_dense_turn_scoring_rejects_final_tagged_clarifications():
    task = ShardedTask(
        task_id="x",
        prompt="Add hidden numbers.",
        shards=["First is 1."],
        answer="1",
        kind="number",
    )

    score = score_sharded_turn("<final>What is the first number?</final>", task, 0)

    assert score["score"] == 0.0
    assert score["incorrect_format"] == 1
    assert score["dense_action"] == "malformed_final_markup"


def test_rollout_training_rewards_support_sparse_and_dense_modes():
    task = ShardedTask(
        task_id="x",
        prompt="Add hidden numbers.",
        shards=["First is 1.", "Second is 2."],
        answer="3",
        kind="number",
    )
    responses = iter(["What is the first number?", "What is the second number?", "<final>3</final>"])
    rollout = run_sharded_interaction(task, lambda _messages: SampledResponse(text=next(responses)))

    assert rollout_training_rewards(rollout, "sparse") == [1.0, 1.0, 1.0]
    assert rollout_training_rewards(rollout, "dense") == [
        DENSE_CLARIFICATION_REWARD,
        DENSE_CLARIFICATION_REWARD,
        1.0,
    ]
    assert rollout_training_reward(rollout, "rlrf") == rollout.dense_reward
    assert rollout_training_rewards(rollout, "sdpo") == rollout_training_rewards(rollout, "dense")
    assert normalize_reward_mode("self-distillation") == SDPO_REWARD_MODE


def test_select_sdpo_distillation_turn_picks_first_bad_failed_turn():
    task = ShardedTask(
        task_id="x",
        prompt="Add hidden numbers.",
        shards=["First is 1.", "Second is 2."],
        answer="3",
        kind="number",
    )
    responses = iter(["<final>1</final>"])
    rollout = run_sharded_interaction(task, lambda _messages: SampledResponse(text=next(responses)))

    assert select_sdpo_distillation_turn(rollout, "failed") == 0

    successful_responses = iter(["What is the first number?", "What is the second number?", "<final>3</final>"])
    successful = run_sharded_interaction(task, lambda _messages: SampledResponse(text=next(successful_responses)))
    assert select_sdpo_distillation_turn(successful, "failed") is None


def test_build_sdpo_teacher_messages_defaults_to_minimal_teacher_without_privileged_context():
    task = ShardedTask(
        task_id="x",
        prompt="Add hidden numbers.",
        shards=["First is 1.", "Second is 2."],
        answer="3",
        kind="number",
        full_prompt="Add 1 and 2.",
    )
    responses = iter(["<final>1</final>"])
    rollout = run_sharded_interaction(task, lambda _messages: SampledResponse(text=next(responses)))
    original_transcript = list(rollout.transcript)

    messages = build_sdpo_teacher_messages(
        rollout,
        0,
        successful_previous_attempt="<final>3</final>",
    )

    assert rollout.transcript == original_transcript
    assert messages is not rollout.turns[0].prompt_messages
    assert messages[1:] == rollout.turns[0].prompt_messages[1:]
    assert messages[0]["content"].endswith(
        "There is a chance that the problem is underspecified. Either beforehand or during the problem solving process, ask clarifying questions for any potentially missing information."
    )


def test_build_sdpo_teacher_messages_enhanced_uses_privileged_context_without_mutating_rollout():
    task = ShardedTask(
        task_id="x",
        prompt="Add hidden numbers.",
        shards=["First is 1.", "Second is 2."],
        answer="3",
        kind="number",
        full_prompt="Add 1 and 2.",
    )
    responses = iter(["<final>1</final>"])
    rollout = run_sharded_interaction(task, lambda _messages: SampledResponse(text=next(responses)))
    original_transcript = list(rollout.transcript)

    messages = build_sdpo_teacher_messages(
        rollout,
        0,
        successful_previous_attempt="<final>3</final>",
        teacher_prompt_style=ENHANCED_TEACHER_PROMPT_STYLE,
    )

    assert rollout.transcript == original_transcript
    joined = "\n".join(message["content"] for message in messages)
    assert "Add 1 and 2." in joined
    assert "First is 1." in joined
    assert "Student response on this turn" in joined
    assert "<final>3</final>" in joined
    assert "ask one concise clarifying question" in joined


def test_build_sdpo_teacher_messages_supports_brief_prompt_style():
    task = ShardedTask(
        task_id="x",
        prompt="Add hidden numbers.",
        shards=["First is 1.", "Second is 2."],
        answer="3",
        kind="number",
        full_prompt="Add 1 and 2.",
    )
    responses = iter(["<final>1</final>"])
    rollout = run_sharded_interaction(task, lambda _messages: SampledResponse(text=next(responses)))

    messages = build_sdpo_teacher_messages(rollout, 0, teacher_prompt_style="brief")
    joined = "\n".join(message["content"] for message in messages)

    assert messages[0]["content"].endswith(
        "There is a chance that the problem is underspecified. Either beforehand or during the problem solving process, ask clarifying questions for any potentially missing information."
    )
    assert BRIEF_UNDERSPECIFIED_PREFIX in joined
    assert "Return only the next assistant message." in joined
    assert "feedback-conditioned self-teacher" not in joined


def test_teacher_prompt_style_constants_match_new_defaults():
    assert MINIMAL_TEACHER_PROMPT_STYLE == "minimal_teacher"
    assert ENHANCED_TEACHER_PROMPT_STYLE == "enhanced"


def test_centered_turn_advantages_aligns_rollouts_by_turn_index():
    advantages = centered_turn_advantages([[0.25, 1.0], [0.0], [0.25, 0.0]])

    assert advantages == [[0.08333333333333334, 0.5], [-0.16666666666666666], [0.08333333333333334, -0.5]]


def test_local_reward_module_supports_dense_turn_context():
    sharded_multiturn = load_feedback_module("sharded_multiturn")

    score = sharded_multiturn.compute_score(
        "What hidden number should I use?",
        "3",
        {
            "sharded_reward_mode": "dense",
            "reward_kind": "number",
            "problem": "Add hidden numbers.",
            "shards": ["First is 1.", "Second is 2."],
            "revealed_shards_before_turn": 0,
        },
    )

    assert score["score"] == DENSE_CLARIFICATION_REWARD
    assert score["dense_action"] == "clarify"

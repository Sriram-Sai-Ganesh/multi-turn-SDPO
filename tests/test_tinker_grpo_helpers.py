import json

from scripts.tinker_eval import is_training_state_path
from scripts.tinker_grpo import (
    acquire_run_lock,
    extra_info_for_row,
    first_successful_response,
    ground_truth_for_row,
    load_json_records,
    release_run_lock,
    row_to_messages,
    score_response,
    shuffle_records,
)


class _Rollout:
    def __init__(self, reward, final_response):
        self.reward = reward
        self.final_response = final_response


def test_load_json_records_supports_jsonl(tmp_path):
    path = tmp_path / "rows.jsonl"
    path.write_text('{"idx": 1}\n{"idx": 2}\n', encoding="utf-8")

    assert load_json_records(path) == [{"idx": 1}, {"idx": 2}]


def test_load_json_records_supports_arrays(tmp_path):
    path = tmp_path / "rows.json"
    path.write_text(json.dumps([{"idx": 1}, {"idx": 2}]), encoding="utf-8")

    assert load_json_records(path) == [{"idx": 1}, {"idx": 2}]


def test_shuffle_records_is_deterministic_and_non_mutating():
    rows = [{"idx": i} for i in range(10)]

    shuffled_a = shuffle_records(rows, seed=3)
    shuffled_b = shuffle_records(rows, seed=3)

    assert shuffled_a == shuffled_b
    assert shuffled_a != rows
    assert rows == [{"idx": i} for i in range(10)]
    assert shuffle_records(rows, seed=-1) is rows


def test_first_successful_response_can_exclude_self():
    rollouts = [
        _Rollout(1.0, "<final>self</final>"),
        _Rollout(0.0, "<final>wrong</final>"),
        _Rollout(1.0, "<final>peer</final>"),
    ]

    assert first_successful_response(rollouts) == "<final>self</final>"
    assert first_successful_response(rollouts, exclude_index=0) == "<final>peer</final>"
    assert first_successful_response([_Rollout(0.0, "")]) is None


def test_run_lock_blocks_active_duplicate(tmp_path):
    lock = acquire_run_lock(tmp_path, "run-a")
    try:
        try:
            acquire_run_lock(tmp_path, "run-a")
        except SystemExit as exc:
            assert "already appears to be active" in str(exc)
        else:
            raise AssertionError("duplicate active run lock was not rejected")
    finally:
        release_run_lock(lock)

    assert not lock.exists()


def test_row_to_messages_preserves_optional_system_prompt():
    row = {"system": "sys", "prompt": "question"}

    assert row_to_messages(row) == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "question"},
    ]


def test_code_ground_truth_prefers_tests():
    row = {"kind": "code", "answer": "reference", "tests": "unit tests"}

    assert ground_truth_for_row(row) == "unit tests"


def test_extra_info_has_reward_fields():
    row = {"idx": 7, "prompt": "p", "description": "d", "elo": 1234}

    assert extra_info_for_row(row, "train") == {
        "split": "train",
        "index": "7",
        "description": "d",
        "problem": "p",
        "elo": 1234,
        "achievement_prior": 0,
        "truncated": False,
    }


def test_score_response_loads_tooluse_without_full_verl_imports():
    row = {
        "idx": 1,
        "dataset": "tooluse",
        "kind": "tooluse",
        "prompt": "p",
        "answer": '[{"Action": "search", "Action_Input": "{\\"query\\": \\"abc\\"}"}]',
    }

    score = score_response(row, 'Action: search\nAction Input: {"query": "abc"}', "train")

    assert score["score"] == 1.0


def test_score_response_handles_nested_tooluse_action_input():
    row = {
        "idx": 1,
        "dataset": "tooluse",
        "kind": "tooluse",
        "prompt": "p",
        "answer": (
            '[{"Action": "sendHttpRequest", "Action_Input": '
            '"{\\"method\\": \\"POST\\", \\"url\\": \\"https://httpbin.org/post\\", '
            '\\"headers\\": {\\"Content-Type\\": \\"application/json\\"}, '
            '\\"data\\": {\\"name\\": \\"John Doe\\", \\"email\\": \\"john.doe@example.com\\"}}"}]'
        ),
    }
    response = (
        "Action: sendHttpRequest\n"
        'Action Input: {"method": "POST", "url": "https://httpbin.org/post", '
        '"headers": {"Content-Type": "application/json"}, '
        '"data": {"name": "John Doe", "email": "john.doe@example.com"}}'
    )

    score = score_response(row, response, "train")

    assert score["score"] == 1.0


def test_score_response_reports_sciknoweval_format_errors():
    row = {
        "idx": 1,
        "dataset": "sciknoweval",
        "kind": "mcq",
        "prompt": "p",
        "answer": "B",
    }

    valid = score_response(row, "<answer>\nB\n</answer>", "train")
    invalid = score_response(row, "B", "train")

    assert valid["score"] == 1.0
    assert valid["incorrect_format"] == 0
    assert invalid["incorrect_format"] == 1


def test_tinker_eval_detects_training_state_paths():
    assert is_training_state_path("tinker://abc/weights/run-final")
    assert not is_training_state_path("tinker://abc/sampler_weights/run-final-sampler")
    assert not is_training_state_path(None)

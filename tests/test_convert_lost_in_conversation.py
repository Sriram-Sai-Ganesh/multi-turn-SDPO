import json

from scripts.convert_lost_in_conversation import convert_record, split_records


def test_convert_lost_in_conversation_math_record():
    row = {
        "task_id": "sharded-gsm8k-1",
        "task": "math",
        "shards": [
            {"shard_id": 2, "shard": "He buys 3 more. How many apples does he have?"},
            {"shard_id": 1, "shard": "John has 2 apples."},
        ],
        "answer": "5",
        "fully_specified_question": ["John has 2 apples.", "He buys 3 more. How many apples does he have?"],
    }

    record = convert_record(row)

    assert record == {
        "idx": "sharded-gsm8k-1",
        "dataset": "sharded_multiturn",
        "kind": "number",
        "source_task": "math",
        "prompt": "John has 2 apples.",
        "shards": ["He buys 3 more. How many apples does he have?"],
        "answer": "5",
        "full_prompt": "John has 2 apples.\nHe buys 3 more. How many apples does he have?",
    }


def test_convert_lost_in_conversation_skips_records_without_answer():
    row = {
        "task_id": "x",
        "task": "math",
        "shards": [
            {"shard_id": 1, "shard_text": "First."},
            {"shard_id": 2, "shard_text": "Second."},
        ],
    }

    assert convert_record(row) is None


def test_split_records_is_deterministic():
    records = [{"idx": i} for i in range(10)]

    left_a, right_a = split_records(records, train_ratio=0.8, seed=7)
    left_b, right_b = split_records(records, train_ratio=0.8, seed=7)

    assert json.dumps(left_a, sort_keys=True) == json.dumps(left_b, sort_keys=True)
    assert json.dumps(right_a, sort_keys=True) == json.dumps(right_b, sort_keys=True)
    assert len(left_a) == 8
    assert len(right_a) == 2

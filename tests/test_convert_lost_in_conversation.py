import json

from scripts.build_clean_holdout import build_holdout
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


def test_convert_record_preserves_action_function_metadata():
    row = {
        "task_id": "sharded-BFCL/parallel_144",
        "task": "actions",
        "function": [
            {
                "name": "math.factorial",
                "description": "Calculate the factorial of a given number.",
                "parameters": {
                    "type": "dict",
                    "properties": {"number": {"type": "integer"}},
                    "required": ["number"],
                },
            }
        ],
        "language": "Python",
        "test_category": "parallel",
        "shards": [
            {"shard_id": 1, "shard": "What is the factorial of 5?"},
            {"shard_id": 2, "shard": "Now calculate the factorial of 3"},
        ],
        "reference_answer": [{"math.factorial": {"number": [5]}}],
        "fully_specified_question": [[{"role": "user", "content": "Calculate factorials."}]],
    }

    record = convert_record(row)

    assert record is not None
    assert record["idx"] == "sharded-BFCL/parallel_144"
    assert record["source_task"] == "actions"
    assert record["kind"] == "tool_call"
    assert record["prompt"] == "What is the factorial of 5?"
    assert record["shards"] == ["Now calculate the factorial of 3"]
    assert record["functions"][0]["name"] == "math.factorial"
    assert record["language"] == "Python"
    assert record["test_category"] == "parallel"


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


def test_build_clean_holdout_excludes_ids_and_shuffles():
    rows = [
        {
            "task_id": "keep-math",
            "task": "math",
            "shards": [
                {"shard_id": 1, "shard": "John has 2 apples."},
                {"shard_id": 2, "shard": "He buys 3 more. How many apples does he have?"},
            ],
            "answer": "5",
        },
        {
            "task_id": "exclude-math",
            "task": "math",
            "shards": [
                {"shard_id": 1, "shard": "A"},
                {"shard_id": 2, "shard": "B"},
            ],
            "answer": "C",
        },
        {
            "task_id": "skip-translation",
            "task": "translation",
            "shards": [
                {"shard_id": 1, "shard": "Translate hi."},
                {"shard_id": 2, "shard": "Into French."},
            ],
            "answer": "salut",
        },
    ]

    records, summary = build_holdout(
        rows=rows,
        task_filter={"math", "actions"},
        excluded_ids={"exclude-math"},
        seed=3,
        max_records=0,
    )

    assert [record["idx"] for record in records] == ["keep-math"]
    assert summary["converted_holdout"] == 1
    assert summary["excluded_ids_matched"] == 1
    assert summary["task_counts"] == {"math": 1}

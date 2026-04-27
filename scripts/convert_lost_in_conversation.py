"""Convert microsoft/lost_in_conversation rows into sharded_multiturn JSON.

The upstream dataset contains task-specific records with an ordered `shards`
field. This converter makes the first shard the initial user prompt and keeps
the remaining shards hidden for the environment to reveal over turns.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any

DEFAULT_SOURCE = "microsoft/lost_in_conversation"
TASK_TO_KIND = {
    "math": "number",
    "code": "exact",
    "database": "exact",
    "actions": "exact",
    "data2text": "exact",
    "summary": "contains",
    "translation": "exact",
}
ANSWER_KEYS = (
    "answer",
    "final_answer",
    "reference_answer",
    "reference",
    "target",
    "gold",
    "gold_answer",
    "expected_answer",
    "output",
    "solution",
)
FULL_PROMPT_KEYS = (
    "fully_specified_question",
    "fully_specified_instruction",
    "full_prompt",
    "question",
    "prompt",
    "instruction",
)


def load_json_records(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    stripped = text.lstrip()
    if not stripped:
        return []
    if stripped[0] == "[":
        payload = json.loads(stripped)
        if not isinstance(payload, list):
            raise ValueError(f"Expected a JSON list in {path}")
        return payload
    if stripped[0] == "{":
        payload = json.loads(stripped)
        if isinstance(payload, dict):
            for key in ("train", "data", "records", "examples"):
                if isinstance(payload.get(key), list):
                    return payload[key]
        return [payload]
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def load_records(source: str, split: str) -> list[dict[str, Any]]:
    path = Path(source)
    if path.exists():
        if path.is_dir():
            for name in (f"{split}.json", f"{split}.jsonl", "lost_in_conversation.json"):
                candidate = path / name
                if candidate.exists():
                    return load_json_records(candidate)
            raise FileNotFoundError(f"No {split}.json/jsonl or lost_in_conversation.json found in {path}")
        if path.suffix == ".parquet":
            import datasets

            return list(datasets.load_dataset("parquet", data_files=str(path), split="train"))
        return load_json_records(path)

    try:
        import datasets

        return list(datasets.load_dataset(source, split=split))
    except Exception:
        if "/" not in source:
            raise
        from huggingface_hub import hf_hub_download

        downloaded = hf_hub_download(repo_id=source, repo_type="dataset", filename="lost_in_conversation.json")
        return load_json_records(Path(downloaded))


def stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(stringify(item) for item in value if stringify(item))
    if isinstance(value, dict):
        for key in ("text", "content", "question", "instruction", "answer", "value"):
            if value.get(key) not in (None, ""):
                return stringify(value[key])
        return json.dumps(value, sort_keys=True)
    return str(value)


def shard_text(shard: Any) -> str:
    if isinstance(shard, dict):
        for key in ("shard_text", "shard", "text", "content", "message", "value"):
            if shard.get(key) not in (None, ""):
                return stringify(shard[key]).strip()
    return stringify(shard).strip()


def ordered_shards(row: dict[str, Any]) -> list[str]:
    shards = row.get("shards") or row.get("instruction_shards") or row.get("hidden_shards")
    if isinstance(shards, str):
        try:
            shards = json.loads(shards)
        except json.JSONDecodeError:
            shards = [shards]
    if not isinstance(shards, list):
        return []
    if all(isinstance(shard, dict) and "shard_id" in shard for shard in shards):
        shards = sorted(shards, key=lambda shard: shard.get("shard_id", 0))
    return [text for text in (shard_text(shard) for shard in shards) if text]


def first_value(row: dict[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        if row.get(key) not in (None, ""):
            return stringify(row[key]).strip()
    return ""


def infer_answer(row: dict[str, Any]) -> str:
    direct = first_value(row, ANSWER_KEYS)
    if direct:
        return direct
    evaluation = row.get("evaluation") or row.get("eval") or row.get("metadata") or {}
    if isinstance(evaluation, dict):
        return first_value(evaluation, ANSWER_KEYS)
    return ""


def infer_full_prompt(row: dict[str, Any], shards: list[str]) -> str:
    prompt = first_value(row, FULL_PROMPT_KEYS)
    if prompt:
        return prompt
    return "\n".join(shards)


def convert_record(row: dict[str, Any]) -> dict[str, Any] | None:
    shards = ordered_shards(row)
    if len(shards) < 2:
        return None
    answer = infer_answer(row)
    if not answer:
        return None
    task = str(row.get("task") or row.get("task_type") or "unknown")
    fallback_id = hashlib.sha1(json.dumps(row, sort_keys=True).encode("utf-8")).hexdigest()[:12]
    return {
        "idx": str(row.get("task_id") or row.get("idx") or row.get("id") or f"lost-{fallback_id}"),
        "dataset": "sharded_multiturn",
        "kind": TASK_TO_KIND.get(task, "exact"),
        "source_task": task,
        "prompt": shards[0],
        "shards": shards[1:],
        "answer": answer,
        "full_prompt": infer_full_prompt(row, shards),
    }


def split_records(records: list[dict[str, Any]], train_ratio: float, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records = list(records)
    rng = random.Random(seed)
    rng.shuffle(records)
    train_size = int(round(len(records) * train_ratio))
    train_size = min(max(train_size, 1 if len(records) > 1 else len(records)), len(records))
    return records[:train_size], records[train_size:]


def write_json(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(records, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert Lost in Conversation rows to sharded_multiturn JSON.")
    parser.add_argument("--source", default=DEFAULT_SOURCE, help="HF dataset id, JSON/JSONL file, parquet file, or directory.")
    parser.add_argument("--split", default="train", help="Input split when loading from Hugging Face.")
    parser.add_argument("--output-dir", default="datasets/sharded_multiturn/lost_in_conversation")
    parser.add_argument("--task", action="append", default=[], help="Keep only this upstream task. Repeatable.")
    parser.add_argument("--max-records", type=int, default=0, help="Limit converted records after filtering.")
    parser.add_argument("--train-ratio", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--strict", action="store_true", help="Fail if any record cannot be converted.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0 < args.train_ratio <= 1:
        raise SystemExit("--train-ratio must be in (0, 1].")

    rows = load_records(args.source, args.split)
    task_filter = {task.lower() for task in args.task}
    converted = []
    skipped = 0
    for row in rows:
        task = str(row.get("task") or row.get("task_type") or "unknown").lower()
        if task_filter and task not in task_filter:
            continue
        record = convert_record(row)
        if record is None:
            skipped += 1
            if args.strict:
                raise SystemExit(f"Could not convert record: {row}")
            continue
        converted.append(record)
        if args.max_records > 0 and len(converted) >= args.max_records:
            break

    if not converted:
        raise SystemExit("No records converted. Check source path, task filter, and answer fields.")

    train, test = split_records(converted, train_ratio=args.train_ratio, seed=args.seed)
    output_dir = Path(args.output_dir)
    write_json(output_dir / "train.json", train)
    write_json(output_dir / "test.json", test)
    summary = {
        "source": args.source,
        "input_rows": len(rows),
        "converted": len(converted),
        "skipped": skipped,
        "train": len(train),
        "test": len(test),
        "tasks": sorted({record["source_task"] for record in converted}),
    }
    write_json(output_dir / "conversion_summary.json", [summary])
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

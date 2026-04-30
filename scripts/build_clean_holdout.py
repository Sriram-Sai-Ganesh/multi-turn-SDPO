"""Build a clean sharded_multiturn holdout split from Lost in Conversation.

This helper is intended for final-project evaluation. It converts upstream
Lost-in-Conversation rows with the repo converter, excludes any task IDs that
were already used for training or repeatedly inspected pilot tests, and writes
a local JSON split that can be consumed by ``run_tinker_eval.sh``.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.convert_lost_in_conversation import convert_record, load_records


def load_json_records(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    stripped = text.lstrip()
    if not stripped:
        return []
    if stripped[0] == "[":
        payload = json.loads(stripped)
        if not isinstance(payload, list):
            raise ValueError(f"Expected JSON list in {path}")
        return payload
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def collect_excluded_ids(paths: list[str]) -> set[str]:
    excluded: set[str] = set()
    for raw_path in paths:
        path = Path(raw_path)
        if not path.exists():
            raise FileNotFoundError(f"Exclude file not found: {path}")
        for row in load_json_records(path):
            if row.get("idx") not in (None, ""):
                excluded.add(str(row["idx"]))
    return excluded


def build_holdout(
    rows: list[dict[str, Any]],
    task_filter: set[str],
    excluded_ids: set[str],
    seed: int,
    max_records: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    skipped = 0
    duplicate_ids = 0
    excluded_seen = 0
    seen_ids: set[str] = set()

    for row in rows:
        task = str(row.get("task") or row.get("task_type") or "unknown").lower()
        if task_filter and task not in task_filter:
            continue
        record = convert_record(row)
        if record is None:
            skipped += 1
            continue
        task_id = str(record.get("idx", ""))
        if task_id in seen_ids:
            duplicate_ids += 1
            continue
        seen_ids.add(task_id)
        if task_id in excluded_ids:
            excluded_seen += 1
            continue
        converted.append(record)

    rng = random.Random(seed)
    rng.shuffle(converted)
    if max_records > 0:
        converted = converted[:max_records]

    summary = {
        "converted_holdout": len(converted),
        "duplicate_ids_skipped": duplicate_ids,
        "excluded_ids_matched": excluded_seen,
        "excluded_ids_total": len(excluded_ids),
        "max_records": max_records,
        "seed": seed,
        "skipped_unconvertible": skipped,
        "tasks": sorted({record.get("source_task", "unknown") for record in converted}),
        "task_counts": {
            task: sum(1 for record in converted if record.get("source_task") == task)
            for task in sorted({record.get("source_task", "unknown") for record in converted})
        },
    }
    return converted, summary


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a clean sharded_multiturn holdout split.")
    parser.add_argument("--source", default="microsoft/lost_in_conversation")
    parser.add_argument("--split", default="train")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--task", action="append", default=[])
    parser.add_argument("--exclude-json", action="append", default=[])
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--max-records", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    task_filter = {task.lower() for task in (args.task or ["math", "actions"])}
    rows = load_records(args.source, args.split)
    excluded_ids = collect_excluded_ids(args.exclude_json)
    records, summary = build_holdout(
        rows=rows,
        task_filter=task_filter,
        excluded_ids=excluded_ids,
        seed=args.seed,
        max_records=args.max_records,
    )
    if not records:
        raise SystemExit("No holdout records produced after filtering.")

    output_dir = Path(args.output_dir)
    write_json(output_dir / "train.json", [])
    write_json(output_dir / "test.json", records)
    summary.update(
        {
            "source": args.source,
            "split": args.split,
            "exclude_json": args.exclude_json,
            "output_dir": str(output_dir),
        }
    )
    write_json(output_dir / "holdout_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

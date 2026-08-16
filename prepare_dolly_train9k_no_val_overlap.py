#!/usr/bin/env python
"""Remove validation-record overlap from the shuffled Dolly training JSON."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Hashable


def _record_signature(record: dict[str, Any]) -> Hashable:
    answers = tuple(
        sorted(
            (
                str(record.get(f"model{position}", "")),
                str(record.get(f"output{position}", "")),
                float(record[f"score{position}"]),
            )
            for position in "ABC"
        )
    )
    return (
        str(record.get("instruction", record.get("Instruction", ""))),
        str(record.get("input", "")),
        answers,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--val", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    train_path = Path(args.train)
    val_path = Path(args.val)
    out_path = Path(args.out)
    train = json.loads(train_path.read_text(encoding="utf-8"))
    val = json.loads(val_path.read_text(encoding="utf-8"))
    if not isinstance(train, list) or not isinstance(val, list):
        raise ValueError("train and validation JSON files must contain lists")

    remaining_val = Counter(_record_signature(record) for record in val)
    kept: list[dict[str, Any]] = []
    removed = 0
    for record in train:
        signature = _record_signature(record)
        if remaining_val[signature] > 0:
            remaining_val[signature] -= 1
            removed += 1
        else:
            kept.append(record)

    unmatched_val = sum(remaining_val.values())
    if unmatched_val:
        raise RuntimeError(f"{unmatched_val} validation records were not found in training data")
    if removed != len(val):
        raise RuntimeError(f"expected to remove {len(val)} records, removed {removed}")
    if len(kept) != len(train) - len(val):
        raise RuntimeError("unexpected output record count")

    val_signatures = {_record_signature(record) for record in val}
    overlap_after = sum(_record_signature(record) in val_signatures for record in kept)
    if overlap_after:
        raise RuntimeError(f"output still contains {overlap_after} validation overlaps")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    temp_path.write_text(
        json.dumps(kept, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temp_path.replace(out_path)
    print(
        json.dumps(
            {
                "input_train_records": len(train),
                "validation_records": len(val),
                "removed_records": removed,
                "output_train_records": len(kept),
                "overlap_after": overlap_after,
                "output": str(out_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

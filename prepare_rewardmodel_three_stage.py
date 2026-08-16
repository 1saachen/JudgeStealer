#!/usr/bin/env python3
"""Prepare aligned continuous-reward data for the three-stage SFT pipeline."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence


KINDS = ("pointwise", "pairwise", "listwise")


def _read(path: Path) -> List[Dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
        raise ValueError(f"{path} must contain a JSON list of objects")
    return value


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _score(row: Mapping[str, Any], key: str) -> float:
    try:
        value = float(row[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"row id={row.get('id')} has invalid {key}") from exc
    if not math.isfinite(value):
        raise ValueError(f"row id={row.get('id')} has non-finite {key}")
    return value


def _ranking_from_scores(scores: Sequence[float]) -> str:
    letters = ("A", "B", "C")
    groups = []
    for value in sorted(set(float(x) for x in scores), reverse=True):
        groups.append("=".join(letter for letter, score in zip(letters, scores) if float(score) == value))
    return ">".join(groups)


def _pair_choice_code(pair_name: str, choice: Any) -> str:
    text = str(choice or "").strip()
    first, second = pair_name
    if text == first:
        return "1"
    if text == second:
        return "2"
    if text in {"3", "T", "t", "tie", "Tie", "equal"}:
        return "3"
    raise ValueError(f"unsupported {pair_name} pairwise choice: {choice!r}")


def _normalize_pointwise(row: Mapping[str, Any]) -> Dict[str, Any]:
    out = dict(row)
    for letter in "ABC":
        out[f"score{letter}"] = _score(row, f"score{letter}")
    return out


def _normalize_pairwise(row: Mapping[str, Any]) -> Dict[str, Any]:
    out = dict(row)
    pairwise = row.get("pairwise")
    if not isinstance(pairwise, Mapping):
        raise ValueError(f"row id={row.get('id')} has no pairwise object")
    normalized: Dict[str, Any] = {}
    for pair_name in ("AB", "AC", "BC"):
        pair = pairwise.get(pair_name)
        if not isinstance(pair, Mapping):
            raise ValueError(f"row id={row.get('id')} has no pairwise.{pair_name}")
        normalized[pair_name] = dict(pair)
        normalized[pair_name]["choice_code"] = _pair_choice_code(pair_name, pair.get("choice"))
    out["pairwise"] = normalized
    return out


def _normalize_listwise(row: Mapping[str, Any]) -> Dict[str, Any]:
    out = dict(row)
    scores = [_score(row, f"listwise_score{letter}") for letter in "ABC"]
    out["ranking"] = _ranking_from_scores(scores)
    return out


def _index(records: Iterable[Mapping[str, Any]], kind: str) -> Dict[str, Dict[str, Any]]:
    normalizer = {
        "pointwise": _normalize_pointwise,
        "pairwise": _normalize_pairwise,
        "listwise": _normalize_listwise,
    }[kind]
    indexed: Dict[str, Dict[str, Any]] = {}
    for row in records:
        row_id = str(row.get("id"))
        if row_id in indexed:
            raise ValueError(f"duplicate {kind} id: {row_id}")
        indexed[row_id] = normalizer(row)
    return indexed


def _assert_text_alignment(rows: Mapping[str, Mapping[str, Any]], row_id: str) -> None:
    reference = rows["pointwise"]
    keys = ("instruction", "input", "outputA", "outputB", "outputC")
    for kind in ("pairwise", "listwise"):
        for key in keys:
            if str(rows[kind].get(key, "")) != str(reference.get(key, "")):
                raise ValueError(f"unaligned {kind} field {key!r} for id={row_id}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-size", type=int, default=1500)
    parser.add_argument("--mix-size", type=int, default=200)
    parser.add_argument("--eval-size", type=int, default=300)
    args = parser.parse_args()

    indexed = {kind: _index(_read(args.source / f"{kind}.json"), kind) for kind in KINDS}
    ids = list(indexed["pointwise"])
    for kind in ("pairwise", "listwise"):
        if set(indexed[kind]) != set(ids):
            raise ValueError(f"{kind} ids do not match pointwise ids")
    for row_id in ids:
        _assert_text_alignment({kind: indexed[kind][row_id] for kind in KINDS}, row_id)

    random.Random(int(args.seed)).shuffle(ids)
    heldout_size = int(args.mix_size) + int(args.eval_size)
    needed = int(args.train_size) + heldout_size
    if len(ids) < needed:
        raise ValueError(f"need {needed} aligned records, found {len(ids)}")
    selected_ids = ids[:needed]
    train_ids = selected_ids[: int(args.train_size)]
    heldout_ids = selected_ids[int(args.train_size) :]
    mix_ids = heldout_ids[: int(args.mix_size)]
    eval_ids = heldout_ids[int(args.mix_size) :]

    split_dir = args.source / f"split{args.train_size}_{heldout_size}"
    mix_dir = args.source / f"mix{args.mix_size}_eval{args.eval_size}"
    for kind in KINDS:
        train = [indexed[kind][row_id] for row_id in train_ids]
        heldout = [indexed[kind][row_id] for row_id in heldout_ids]
        mix = [indexed[kind][row_id] for row_id in mix_ids]
        evaluation = [indexed[kind][row_id] for row_id in eval_ids]
        _write(split_dir / f"{kind}_train{args.train_size}.json", train)
        _write(split_dir / f"{kind}_eval{heldout_size}.json", heldout)
        _write(mix_dir / f"{kind}_train{args.mix_size}.json", mix)
        _write(mix_dir / f"{kind}_eval{args.eval_size}.json", evaluation)

    _write(
        args.source / "three_stage_split.json",
        {
            "source": str(args.source),
            "seed": int(args.seed),
            "split_unit": "aligned_question_id",
            "continuous_pointwise_scores_preserved": True,
            "pairwise_choice_code_added": True,
            "listwise_ranking_from_listwise_scores": True,
            "train_size": len(train_ids),
            "mix_size": len(mix_ids),
            "eval_size": len(eval_ids),
            "train_ids": train_ids,
            "mix_ids": mix_ids,
            "eval_ids": eval_ids,
        },
    )
    print(f"prepared {len(train_ids)} train + {len(mix_ids)} mix + {len(eval_ids)} eval questions")
    print(f"split: {split_dir}")
    print(f"mix/eval: {mix_dir}")


if __name__ == "__main__":
    main()

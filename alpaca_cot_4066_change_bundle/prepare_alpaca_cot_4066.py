#!/usr/bin/env python
"""Recover the complete CoT rows and build a leakage-free Mix/eval split.

The currently available training JSON is truncated in the middle of a row.  This
script never edits that source file.  It decodes complete objects from its valid
prefix, validates them, and writes a new JSON file.  It also reserves the same
200 validation questions for one pointwise, one pairwise, and one listwise Mix
training example per question.  Every reserved question is removed from every
evaluation task.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = (
    ROOT
    / "train_with_selector"
    / "train_with_selector"
    / "data"
    / "Alpaca-cot-gpt"
    / "Alpaca-cot-gpt"
)
DEFAULT_TRAIN = DEFAULT_DATA_DIR / "train_pointwise_8k.json"
DEFAULT_VALIDATION = DEFAULT_DATA_DIR / "test_2k.json"
DEFAULT_OUTPUT = DEFAULT_DATA_DIR.parent / "prepared_4066"
POSITIONS = ("A", "B", "C")
PAIR_NAMES = ("AB", "AC", "BC")
CANONICAL_RANKINGS = {
    "A>B>C",
    "A>C>B",
    "B>A>C",
    "B>C>A",
    "C>A>B",
    "C>B>A",
    "A=B>C",
    "A=C>B",
    "B=C>A",
    "A>B=C",
    "B>A=C",
    "C>A=B",
    "A=B=C",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _recover_json_array_prefix(path: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    decoder = json.JSONDecoder()
    pos = 0
    while pos < len(text) and text[pos].isspace():
        pos += 1
    if pos >= len(text) or text[pos] != "[":
        raise ValueError(f"expected a top-level JSON array: {path}")
    pos += 1
    rows: List[Dict[str, Any]] = []
    decode_error = ""
    while True:
        while pos < len(text) and (text[pos].isspace() or text[pos] == ","):
            pos += 1
        if pos >= len(text) or text[pos] == "]":
            break
        try:
            value, end = decoder.raw_decode(text, pos)
        except json.JSONDecodeError as exc:
            decode_error = str(exc)
            break
        if not isinstance(value, dict):
            raise ValueError(f"training row {len(rows)} is not an object")
        rows.append(value)
        pos = int(end)
    return rows, {
        "complete_rows": len(rows),
        "source_bytes": len(text.encode("utf-8")),
        "stopped_at_character": int(pos),
        "decode_error": decode_error,
        "source_was_truncated": bool(decode_error),
    }


def _score(value: Any, *, context: str) -> int:
    try:
        result = int(value)
    except Exception as exc:
        raise ValueError(f"invalid score at {context}: {value!r}") from exc
    if not 1 <= result <= 10:
        raise ValueError(f"score out of range at {context}: {result}")
    return result


def _validate_pointwise_record(record: Mapping[str, Any], *, context: str) -> None:
    if not str(record.get("instruction", "")).strip():
        raise ValueError(f"missing instruction at {context}")
    pointwise = record.get("pointwise")
    if not isinstance(pointwise, Mapping):
        raise ValueError(f"missing pointwise object at {context}")
    for position in POSITIONS:
        if not str(record.get(f"answer{position}", "")).strip():
            raise ValueError(f"missing answer{position} at {context}")
        value = pointwise.get(position)
        if not isinstance(value, Mapping):
            raise ValueError(f"missing pointwise.{position} at {context}")
        _score(value.get("score"), context=f"{context}.pointwise.{position}")
        if not str(value.get("reason", "")).strip():
            raise ValueError(f"missing pointwise reason at {context}.{position}")


def _normalize_question(record: Mapping[str, Any]) -> Dict[str, Any]:
    pointwise = record["pointwise"]
    out: Dict[str, Any] = {
        "id": int(record["id"]),
        "source_id": int(record.get("source_id", record["id"])),
        "dataset": str(record.get("dataset", "alpaca_cot_gpt")),
        "instruction": str(record.get("instruction", "")),
        "input": str(record.get("input", "")),
    }
    for position in POSITIONS:
        out[f"model{position}"] = str(record.get(f"model{position}", position))
        out[f"output{position}"] = str(record[f"answer{position}"])
        out[f"score{position}"] = _score(
            pointwise[position]["score"], context=f"id={record['id']}.{position}"
        )
        out[f"reason{position}"] = str(pointwise[position]["reason"]).strip()
    out["pointwise"] = {
        position: {
            "score": int(out[f"score{position}"]),
            "reason": str(out[f"reason{position}"]),
        }
        for position in POSITIONS
    }
    return out


def _swap_pair_reason(reason: str) -> str:
    value = str(reason)
    value = value.replace("Assistant 1", "Assistant __TMP__")
    value = value.replace("Assistant 2", "Assistant 1")
    return value.replace("Assistant __TMP__", "Assistant 2")


def _replace_final_pair_label(reason: str, choice: int) -> str:
    body = re.sub(r"\s*\[\[[123]\]\]\s*$", "", str(reason).rstrip())
    return f"{body}\n[[{int(choice)}]]" if body else f"[[{int(choice)}]]"


def _replace_labels(text: str, mapping: Mapping[str, str]) -> str:
    value = str(text)
    for old in POSITIONS:
        value = value.replace(f"Assistant {old}", f"Assistant __{old}__")
    for old, new in mapping.items():
        value = value.replace(f"Assistant __{old}__", f"Assistant {new}")
    return value


def _replace_final_ranking(reason: str, ranking: str) -> str:
    body = re.sub(r"\s*Ranking:\s*\[[ABC=>]+\]\s*$", "", str(reason).rstrip())
    return f"{body}\nRanking:[{ranking}]" if body else f"Ranking:[{ranking}]"


def _map_ranking(ranking: str, old_to_new: Mapping[str, str]) -> str:
    tokens = re.findall(r"[ABC]|[=>]", str(ranking).replace(" ", ""))
    mapped = "".join(old_to_new.get(token, token) for token in tokens)
    # A permutation can turn a canonical tie such as A>B=C into C>B=A.
    # Sort labels inside each equality group so equivalent rankings use the
    # 13 canonical spellings expected by the existing listwise evaluator.
    groups = mapped.split(">")
    canonical = ">".join("=".join(sorted(group.split("="))) for group in groups)
    labels = re.findall(r"[ABC]", canonical)
    if sorted(labels) != list(POSITIONS) or canonical not in CANONICAL_RANKINGS:
        raise ValueError(f"invalid listwise ranking after permutation: {ranking!r} -> {canonical!r}")
    return canonical


def _pair_value(
    record: Mapping[str, Any], old_left: str, old_right: str
) -> Tuple[int, str]:
    canonical = "".join(sorted((old_left, old_right)))
    pairwise = record.get("pairwise")
    if not isinstance(pairwise, Mapping) or not isinstance(pairwise.get(canonical), Mapping):
        raise ValueError(f"missing pairwise.{canonical} for id={record.get('id')}")
    value = pairwise[canonical]
    choice = int(value.get("choice"))
    reason = str(value.get("reason", "")).strip()
    canonical_left, canonical_right = canonical
    if (old_left, old_right) != (canonical_left, canonical_right):
        choice = 2 if choice == 1 else 1 if choice == 2 else 3
        reason = _swap_pair_reason(reason)
    return choice, _replace_final_pair_label(reason, choice)


def _permuted_validation_record(
    record: Mapping[str, Any], *, permutation: Sequence[str]
) -> Dict[str, Any]:
    """Return a standard ABC record where new position -> original position."""
    if sorted(permutation) != list(POSITIONS):
        raise ValueError(f"invalid permutation: {permutation}")
    pointwise = record["pointwise"]
    old_to_new = {old: new for new, old in zip(POSITIONS, permutation)}
    out: Dict[str, Any] = {
        "id": int(record["id"]),
        "source_id": int(record.get("source_id", record["id"])),
        "dataset": str(record.get("dataset", "alpaca_cot_gpt")),
        "instruction": str(record.get("instruction", "")),
        "input": str(record.get("input", "")),
        "original_position_order": list(permutation),
    }
    for new_position, old_position in zip(POSITIONS, permutation):
        out[f"model{new_position}"] = str(record.get(f"model{old_position}", old_position))
        out[f"output{new_position}"] = str(record[f"answer{old_position}"])
        out[f"score{new_position}"] = _score(
            pointwise[old_position]["score"],
            context=f"id={record['id']}.pointwise.{old_position}",
        )
        out[f"reason{new_position}"] = str(pointwise[old_position]["reason"]).strip()
    out["pointwise"] = {
        position: {
            "score": int(out[f"score{position}"]),
            "reason": str(out[f"reason{position}"]),
        }
        for position in POSITIONS
    }

    out_pairwise: Dict[str, Any] = {}
    for pair_name in PAIR_NAMES:
        new_left, new_right = pair_name
        old_left = permutation[POSITIONS.index(new_left)]
        old_right = permutation[POSITIONS.index(new_right)]
        choice, reason = _pair_value(record, old_left, old_right)
        out_pairwise[pair_name] = {
            "choice": int(choice),
            "choice_code": int(choice),
            "reason": reason,
        }
        out[f"choice_{pair_name}"] = int(choice)
    out["pairwise"] = out_pairwise

    listwise = record.get("listwise")
    if not isinstance(listwise, Mapping):
        raise ValueError(f"missing listwise object for id={record.get('id')}")
    ranking = _map_ranking(str(listwise.get("choice", "")), old_to_new)
    reason = _replace_labels(str(listwise.get("reason", "")), old_to_new)
    reason = _replace_final_ranking(reason, ranking)
    out["ranking"] = ranking
    out["listwise"] = {"choice": ranking, "reason": reason}
    out["listwise_reason"] = reason
    return out


def _select_mix_examples(
    records: Sequence[Mapping[str, Any]], *, rng: random.Random
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    pointwise_rows: List[Dict[str, Any]] = []
    pairwise_rows: List[Dict[str, Any]] = []
    listwise_rows: List[Dict[str, Any]] = []
    for record in records:
        position = rng.choice(POSITIONS)
        pointwise_rows.append(
            {
                "id": int(record["id"]),
                "source_id": int(record["source_id"]),
                "dataset": str(record["dataset"]),
                "instruction": str(record["instruction"]),
                "input": str(record["input"]),
                "model": str(record[f"model{position}"]),
                "output": str(record[f"output{position}"]),
                "judge_reason": str(record[f"reason{position}"]),
                "judge_score": int(record[f"score{position}"]),
                "position": position,
            }
        )

        pair_name = rng.choice(PAIR_NAMES)
        left, right = pair_name
        pair_value = record["pairwise"][pair_name]
        pairwise_rows.append(
            {
                "id": int(record["id"]),
                "source_id": int(record["source_id"]),
                "dataset": str(record["dataset"]),
                "instruction": str(record["instruction"]),
                "input": str(record["input"]),
                "modelA": str(record[f"model{left}"]),
                "outputA": str(record[f"output{left}"]),
                "modelB": str(record[f"model{right}"]),
                "outputB": str(record[f"output{right}"]),
                "choice_AB": int(pair_value["choice"]),
                "pairwise": {
                    "AB": {
                        "choice": int(pair_value["choice"]),
                        "choice_code": int(pair_value["choice"]),
                        "reason": str(pair_value["reason"]),
                    }
                },
                "source_pair": pair_name,
            }
        )

        listwise_rows.append(dict(record))
    return pointwise_rows, pairwise_rows, listwise_rows


def _ids(rows: Iterable[Mapping[str, Any]]) -> List[int]:
    return [int(row["id"]) for row in rows]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--validation", type=Path, default=DEFAULT_VALIDATION)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mix-questions", type=int, default=200)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    recovered, recovery = _recover_json_array_prefix(args.train)
    for index, record in enumerate(recovered):
        _validate_pointwise_record(record, context=f"train[{index}]")
    recovered_ids = _ids(recovered)
    if len(recovered_ids) != len(set(recovered_ids)):
        raise ValueError("recovered training IDs are not unique")

    validation = json.loads(args.validation.read_text(encoding="utf-8"))
    if not isinstance(validation, list):
        raise ValueError("validation must be a JSON array")
    for index, record in enumerate(validation):
        _validate_pointwise_record(record, context=f"validation[{index}]")
        if not isinstance(record.get("pairwise"), Mapping):
            raise ValueError(f"missing pairwise at validation[{index}]")
        if not isinstance(record.get("listwise"), Mapping):
            raise ValueError(f"missing listwise at validation[{index}]")
    validation_ids = _ids(validation)
    if len(validation_ids) != len(set(validation_ids)):
        raise ValueError("validation IDs are not unique")
    overlap = sorted(set(recovered_ids) & set(validation_ids))
    if overlap:
        raise ValueError(f"train/validation ID overlap: {overlap[:10]}")
    if not 0 < int(args.mix_questions) < len(validation):
        raise ValueError("mix-questions must be between 1 and validation size - 1")

    rng = random.Random(int(args.seed))
    mix_indices = set(rng.sample(range(len(validation)), int(args.mix_questions)))
    mix_raw = [validation[i] for i in sorted(mix_indices)]
    eval_raw = [validation[i] for i in range(len(validation)) if i not in mix_indices]

    def permute_rows(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
        output: List[Dict[str, Any]] = []
        for record in rows:
            permutation = list(POSITIONS)
            rng.shuffle(permutation)
            output.append(_permuted_validation_record(record, permutation=permutation))
        return output

    mix_questions = permute_rows(mix_raw)
    eval_questions = permute_rows(eval_raw)
    mix_pointwise, mix_pairwise, mix_listwise = _select_mix_examples(mix_questions, rng=rng)
    normalized_train = [_normalize_question(record) for record in recovered]

    out = args.output_dir
    _write_json(out / "train_recovered_4066.json", recovered)
    _write_json(out / "train_questions_4066.json", normalized_train)
    _write_json(out / "mix_questions_200.json", mix_questions)
    _write_json(out / "mix_pointwise_train_200.json", mix_pointwise)
    _write_json(out / "mix_pairwise_train_200.json", mix_pairwise)
    _write_json(out / "mix_listwise_train_200.json", mix_listwise)
    _write_json(out / "eval_questions_1800.json", eval_questions)

    pair_hist = Counter(int(row["choice_AB"]) for row in mix_pairwise)
    point_hist = Counter(int(row["judge_score"]) for row in mix_pointwise)
    list_hist = Counter(str(row["ranking"]) for row in mix_listwise)
    manifest = {
        "seed": int(args.seed),
        "source": {
            "train": str(args.train),
            "train_sha256": _sha256(args.train),
            "validation": str(args.validation),
            "validation_sha256": _sha256(args.validation),
        },
        "recovery": recovery,
        "counts": {
            "train_questions": len(normalized_train),
            "train_pointwise_answers": len(normalized_train) * 3,
            "mix_source_questions": len(mix_questions),
            "mix_pointwise_examples": len(mix_pointwise),
            "mix_pairwise_examples": len(mix_pairwise),
            "mix_listwise_examples": len(mix_listwise),
            "eval_questions": len(eval_questions),
            "eval_pointwise_answers": len(eval_questions) * 3,
            "eval_pairwise_examples": len(eval_questions) * 3,
            "eval_listwise_examples": len(eval_questions),
        },
        "leakage": {
            "train_validation_id_overlap": 0,
            "mix_eval_id_overlap": len(set(_ids(mix_questions)) & set(_ids(eval_questions))),
            "mix_ids": _ids(mix_questions),
        },
        "mix_label_histograms": {
            "pointwise": dict(sorted(point_hist.items())),
            "pairwise": dict(sorted(pair_hist.items())),
            "listwise": dict(sorted(list_hist.items())),
        },
        "position_control": {
            "mix_and_eval_answer_positions_permuted": True,
            "permutation_seed": int(args.seed),
            "labels_transformed_with_answers": True,
        },
    }
    _write_json(out / "manifest.json", manifest)
    print(json.dumps(manifest["counts"], ensure_ascii=False, indent=2))
    print(f"Prepared data: {out}")


if __name__ == "__main__":
    main()

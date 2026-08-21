#!/usr/bin/env python3
"""Evaluate an already-trained checkpoint without running any training."""

from __future__ import annotations

import argparse
import gc
import json
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

import run_newnew_one_answer_trueval_three_stage_sft as trueval
import run_pointwise5answers_three_stage_pairwise_listwise_sft_v1 as three
import run_pointwise5answers_three_to_listwise_v1 as listwise
import run_pointwise5answers_two_to_pairwise_v1 as base


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Saved model directory, not the run root.")
    parser.add_argument("--pointwise-dataset", required=True, help="Original scored pointwise JSON dataset.")
    parser.add_argument(
        "--pairwise-dataset",
        default="",
        help="Optional pairwise ABC JSON. If omitted, expand pairwise examples from the listwise dataset.",
    )
    parser.add_argument("--listwise-dataset", required=True, help="Listwise JSON used for evaluation.")
    parser.add_argument("--out", required=True, help="Directory for evaluation metrics.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--val-split-seed", type=int, default=55)
    parser.add_argument("--pointwise-val-answer-seed", type=int, default=65)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--max-new-tokens-pointwise", type=int, default=16)
    parser.add_argument("--max-new-tokens-pairwise", type=int, default=8)
    parser.add_argument("--max-new-tokens-listwise", type=int, default=16)
    parser.add_argument("--score-min", type=int, default=1)
    parser.add_argument("--score-max", type=int, default=10)
    return parser.parse_args()


def _load_pairwise_eval_dataset(pairwise_path: str, listwise_path: str):
    """Match the training runner: use explicit ABC pairs, otherwise expand listwise triples."""
    if str(pairwise_path).strip():
        return base._load_pairwise_abc_eval_dataset(
            str(pairwise_path), pairwise_system_prompt=base.DEFAULT_PAIRWISE_SYSTEM_PROMPT
        )
    return three._load_pairwise_eval_from_listwise_dataset(str(listwise_path))


def _load_eval_data(args: argparse.Namespace) -> Dict[str, Any]:
    questions = base._load_scored_questions(
        str(args.pointwise_dataset), score_min=int(args.score_min), score_max=int(args.score_max)
    )[0]
    _, pointwise_eval_questions, split_info = base._split_questions(
        questions, seed=int(args.val_split_seed), val_ratio=float(args.val_ratio)
    )
    pointwise_eval, pointwise_rows, pointwise_stats = base._build_single_answer_pointwise_eval_examples(
        pointwise_eval_questions,
        seed=int(args.pointwise_val_answer_seed),
        score_min=int(args.score_min),
        score_max=int(args.score_max),
        judge_system_prompt=base.JUDGE_SYSTEM_PROMPT_SCORE_ONLY,
        fix_score_prefix_in_prompt=True,
    )
    pairwise_eval, pairwise_rows, pairwise_stats = _load_pairwise_eval_dataset(
        str(args.pairwise_dataset), str(args.listwise_dataset)
    )
    listwise_eval, listwise_rows, listwise_stats = listwise._load_listwise_eval_dataset(
        str(args.listwise_dataset)
    )
    return {
        "pointwise_eval": pointwise_eval,
        "pointwise_rows": pointwise_rows,
        "pointwise_stats": pointwise_stats,
        "pairwise_eval": pairwise_eval,
        "pairwise_rows": pairwise_rows,
        "pairwise_stats": pairwise_stats,
        "listwise_eval": listwise_eval,
        "listwise_rows": listwise_rows,
        "listwise_stats": listwise_stats,
        "split_info": split_info,
    }


def main() -> None:
    args = parse_args()
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    out = Path(args.out).expanduser().resolve()
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"checkpoint directory not found: {checkpoint}")
    if not (checkpoint / "config.json").is_file():
        raise FileNotFoundError(f"checkpoint has no config.json: {checkpoint}")

    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))

    data = _load_eval_data(args)
    out.mkdir(parents=True, exist_ok=True)
    _write_json(out / "eval_config.json", {
        "mode": "checkpoint_only",
        "checkpoint": str(checkpoint),
        "pointwise_dataset": str(Path(args.pointwise_dataset).expanduser().resolve()),
        "pairwise_dataset": str(Path(args.pairwise_dataset).expanduser().resolve()),
        "listwise_dataset": str(Path(args.listwise_dataset).expanduser().resolve()),
        "seed": int(args.seed),
        "val_ratio": float(args.val_ratio),
        "val_split_seed": int(args.val_split_seed),
        "pointwise_val_answer_seed": int(args.pointwise_val_answer_seed),
        "max_length": int(args.max_length),
        "eval_batch_size": int(args.eval_batch_size),
    })
    _write_json(out / "split_questions.json", data["split_info"])
    _write_json(out / "pointwise_eval_stats.json", data["pointwise_stats"])
    _write_json(out / "pairwise_eval_stats.json", data["pairwise_stats"])
    _write_json(out / "listwise_eval_stats.json", data["listwise_stats"])
    trueval._write_jsonl(out / "pointwise_eval.jsonl", data["pointwise_rows"])
    trueval._write_jsonl(out / "pairwise_eval.jsonl", data["pairwise_rows"])
    trueval._write_jsonl(out / "listwise_eval.jsonl", data["listwise_rows"])

    print(f"Loading checkpoint: {checkpoint}", flush=True)
    model, tokenizer, _ = base._load_sft_model_and_tokenizer(
        model_name_or_path=str(checkpoint),
        max_length=int(args.max_length),
        load_in_4bit=False,
    )
    tokenizer.model_max_length = int(args.max_length)
    tokenizer.padding_side = "left"

    cfg = type("EvalConfig", (), {
        "max_length": int(args.max_length),
        "eval_batch_size": int(args.eval_batch_size),
        "max_new_tokens_pointwise": int(args.max_new_tokens_pointwise),
        "max_new_tokens_pairwise": int(args.max_new_tokens_pairwise),
        "max_new_tokens_listwise": int(args.max_new_tokens_listwise),
        "score_min": int(args.score_min),
        "score_max": int(args.score_max),
    })()
    metrics = {
        "pointwise": base._evaluate_pointwise_sft(
            model=model, tokenizer=tokenizer, examples=data["pointwise_eval"],
            max_length=cfg.max_length, batch_size=cfg.eval_batch_size,
            max_new_tokens=cfg.max_new_tokens_pointwise,
            score_min=cfg.score_min, score_max=cfg.score_max,
        ),
        "pairwise": base._evaluate_pairwise_sft(
            model=model, tokenizer=tokenizer, examples=data["pairwise_eval"],
            max_length=cfg.max_length, batch_size=cfg.eval_batch_size,
            max_new_tokens=cfg.max_new_tokens_pairwise,
        ),
        "listwise": three._evaluate_listwise_sft(
            model=model, tokenizer=tokenizer, examples=data["listwise_eval"],
            max_length=cfg.max_length, batch_size=cfg.eval_batch_size,
            max_new_tokens=cfg.max_new_tokens_listwise,
        ),
    }
    summary = {
        "mode": "checkpoint_only",
        "checkpoint": str(checkpoint),
        "pointwise_metrics": {"after_stage4": metrics["pointwise"]},
        "pairwise_metrics": {"after_stage4": metrics["pairwise"]},
        "listwise_metrics": {"after_stage4": metrics["listwise"]},
        "evaluation_counts": {key: int(value["n"]) for key, value in metrics.items()},
    }
    _write_json(out / "summary.json", summary)
    compact = three._compact_metrics(summary)
    _write_json(out / "metrics_compact.json", compact)
    print(json.dumps(compact, ensure_ascii=False, indent=2), flush=True)
    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

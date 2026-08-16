#!/usr/bin/env python
"""Newnew SFT controls with one-answer pointwise sampling and true val tasks.

Modes
-----
pointwise_only_one_answer:
  Train one stage on N pointwise examples. Each selected question contributes
  exactly one randomly sampled answer.

trueval_three_stage:
  Stage 1 trains on one-answer pointwise examples from train-20k.
  Stage 2 trains on real pairwise examples sampled from the pairwise val file.
  Stage 3 trains on real listwise examples sampled from the listwise val file.
  Pairwise/listwise eval use the remaining validation examples after excluding
  the sampled training units.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import re
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch

import run_pointwise5answers_three_stage_pairwise_listwise_sft_v1 as three
import run_pointwise5answers_three_to_listwise_v1 as lw
import run_pointwise5answers_two_to_pairwise_v1 as base


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _select_one_answer_pointwise(
    questions: Sequence[Dict[str, Any]],
    *,
    samples: int,
    seed: int,
    score_min: int,
    score_max: int,
    fix_score_prefix_in_prompt: bool,
) -> Tuple[List[base.PointwiseScoredExample], List[Dict[str, Any]], Dict[str, Any]]:
    rng = np.random.default_rng(int(seed))
    question_order = [int(i) for i in rng.permutation(len(questions)).tolist()]
    examples: List[base.PointwiseScoredExample] = []
    rows: List[Dict[str, Any]] = []
    stats: Dict[str, Any] = {
        "input_questions": int(len(questions)),
        "requested_samples": int(samples),
        "sampling": "one_random_answer_per_random_question",
        "selected_questions": 0,
        "selected_answers": 0,
        "skipped_no_answers": 0,
    }

    for qi in question_order:
        if len(examples) >= int(samples):
            break
        q = questions[int(qi)]
        answers = list(q.get("answers", []))
        if not answers:
            stats["skipped_no_answers"] += 1
            continue
        answer_index = int(rng.integers(low=0, high=len(answers)))
        ans = answers[answer_index]
        label = base.score_to_class(int(ans.score), score_min=int(score_min), score_max=int(score_max))
        prompt = base.build_judge_prompt(
            system_prompt=base.JUDGE_SYSTEM_PROMPT_SCORE_ONLY,
            instruction=str(q["instruction"]),
            input_text=str(q["input_text"]),
            candidate_output=str(ans.output),
            include_gold_score=False,
            fix_score_prefix=bool(fix_score_prefix_in_prompt),
        )
        row_id = int(len(examples) + 1)
        examples.append(
            base.PointwiseScoredExample(
                row_id=row_id,
                question_id=int(q["question_id"]),
                source_id=int(q.get("source_id", q["question_id"])),
                dataset=str(q.get("dataset", "")),
                instruction=str(q.get("instruction", "")),
                input_text=str(q.get("input_text", "")),
                model=str(ans.model),
                output=str(ans.output),
                score=int(ans.score),
                label=int(label),
                prompt=prompt,
            )
        )
        rows.append(
            {
                "row_id": row_id,
                "question_index": int(qi),
                "question_id": int(q["question_id"]),
                "source_id": int(q.get("source_id", q["question_id"])),
                "dataset": str(q.get("dataset", "")),
                "answer_index": int(answer_index),
                "model": str(ans.model),
                "score": int(ans.score),
                "label": int(label),
            }
        )

    stats["selected_questions"] = int(len(examples))
    stats["selected_answers"] = int(len(examples))
    if len(examples) < int(samples):
        raise RuntimeError(f"only selected {len(examples)} one-answer pointwise samples, requested {samples}")
    return examples, rows, stats


def _split_pairwise_trueval(
    path: str,
    *,
    train_pairs: int,
    seed: int,
) -> Tuple[List[base.PairwiseExample], List[Dict[str, Any]], Dict[str, Any], List[base.PairwiseExample], List[Dict[str, Any]], Dict[str, Any], Dict[str, Any]]:
    records = base._load_pairwise_abc_raw_records(str(path))
    if len(records) <= 1:
        raise RuntimeError(f"pairwise true-val split needs at least 2 records: {path}")

    rng = np.random.default_rng(int(seed))
    order = [int(i) for i in rng.permutation(len(records)).tolist()]
    train_indices: List[int] = []
    generated = 0
    for idx in order:
        one_examples, _, _ = base._build_pairwise_abc_examples_from_records(
            [records[idx]],
            dataset_path=str(path),
            split_name="train_count_probe",
            pairwise_system_prompt=base.DEFAULT_PAIRWISE_SYSTEM_PROMPT,
        )
        if not one_examples:
            continue
        train_indices.append(int(idx))
        generated += int(len(one_examples))
        if generated >= int(train_pairs):
            break
    if generated < int(train_pairs):
        raise RuntimeError(f"only generated {generated} pairwise train examples, requested {train_pairs}")

    train_index_set = {int(i) for i in train_indices}
    train_records = [records[i] for i in train_indices]
    eval_records = [records[i] for i in range(len(records)) if i not in train_index_set]
    train_examples, train_rows, train_stats = base._build_pairwise_abc_examples_from_records(
        train_records,
        dataset_path=str(path),
        split_name="train_trueval_sampled",
        pairwise_system_prompt=base.DEFAULT_PAIRWISE_SYSTEM_PROMPT,
    )
    eval_examples, eval_rows, eval_stats = base._build_pairwise_abc_examples_from_records(
        eval_records,
        dataset_path=str(path),
        split_name="eval_remaining_trueval",
        pairwise_system_prompt=base.DEFAULT_PAIRWISE_SYSTEM_PROMPT,
    )
    train_examples = train_examples[: int(train_pairs)]
    train_rows = train_rows[: int(train_pairs)]
    if not eval_examples:
        raise RuntimeError("pairwise remaining eval split is empty")

    split_info = {
        "dataset_path": str(path),
        "seed": int(seed),
        "split_unit": "abc_record_for_leakage_guard",
        "requested_train_pairwise_examples": int(train_pairs),
        "train_pairwise_examples": int(len(train_examples)),
        "train_records_excluded_from_eval": int(len(train_indices)),
        "eval_records_remaining": int(len(eval_records)),
        "eval_pairwise_examples": int(len(eval_examples)),
        "train_record_indices": [int(i) for i in train_indices],
        "leakage_guard": "all pairwise examples from selected ABC records are excluded from pairwise eval",
    }
    return train_examples, train_rows, train_stats, eval_examples, eval_rows, eval_stats, split_info


def _split_listwise_trueval(
    path: str,
    *,
    train_examples_count: int,
    seed: int,
) -> Tuple[List[lw.ListwiseExample], List[Dict[str, Any]], Dict[str, Any], List[lw.ListwiseExample], List[Dict[str, Any]], Dict[str, Any], Dict[str, Any]]:
    examples, rows, stats = lw._load_listwise_eval_dataset(str(path))
    if len(examples) <= int(train_examples_count):
        raise RuntimeError(f"listwise true-val split too small: {len(examples)} <= {train_examples_count}")
    rng = np.random.default_rng(int(seed))
    train_indices = sorted(int(i) for i in rng.choice(len(examples), size=int(train_examples_count), replace=False).tolist())
    train_index_set = {int(i) for i in train_indices}
    eval_indices = [int(i) for i in range(len(examples)) if int(i) not in train_index_set]

    train_examples = [examples[i] for i in train_indices]
    train_rows = [dict(rows[i], split_index=int(i)) for i in train_indices]
    eval_examples = [examples[i] for i in eval_indices]
    eval_rows = [dict(rows[i], split_index=int(i)) for i in eval_indices]

    train_stats = dict(stats)
    train_stats["split"] = "train_trueval_sampled"
    train_stats["examples"] = int(len(train_examples))
    eval_stats = dict(stats)
    eval_stats["split"] = "eval_remaining_trueval"
    eval_stats["examples"] = int(len(eval_examples))
    split_info = {
        "dataset_path": str(path),
        "seed": int(seed),
        "split_unit": "listwise_example",
        "requested_train_listwise_examples": int(train_examples_count),
        "train_listwise_examples": int(len(train_examples)),
        "eval_listwise_examples": int(len(eval_examples)),
        "train_indices": [int(i) for i in train_indices],
        "leakage_guard": "sampled listwise examples are excluded from listwise eval",
    }
    return train_examples, train_rows, train_stats, eval_examples, eval_rows, eval_stats, split_info


def _make_cfg(args: argparse.Namespace) -> three.RunConfig:
    return three.RunConfig(
        seed=int(args.seed),
        val_ratio=float(args.val_ratio),
        val_split_seed=int(args.val_split_seed),
        pointwise_val_answer_seed=int(args.pointwise_val_answer_seed),
        train_selection_mode="selected_triple",
        fixed_selected_triples_path="",
        resume_stage1_model_dir=str(args.resume_stage1_model_dir),
        triple_selection_strategy="random",
        question_selection_strategy="random",
        randomize_listwise_order=True,
        candidate_selector_kind="bert",
        candidate_selector_init_triples=0,
        candidate_selector_batch_size=20,
        candidate_selector_epochs=4,
        candidate_selector_max_score_candidates=4096,
        candidate_selector_llama_rerank_candidates=1000,
        candidate_selector_buffer_maxlen=1000,
        candidate_selector_one_per_question=True,
        candidate_selector_target_task="pointwise",
        candidate_selector_score_range_weight=0.0,
        candidate_selector_gap_sum_weight=0.0,
        candidate_selector_uncertainty_weight=1.0,
        candidate_selector_pairwise_uncertainty_weight=0.0,
        candidate_selector_listwise_uncertainty_weight=0.0,
        candidate_selector_kl_weight=0.0,
        candidate_selector_score_bin_weight=0.0,
        candidate_selector_diversity_weight=0.25,
        candidate_selector_density_weight=0.15,
        candidate_selector_bias_weight=0.25,
        candidate_selector_coverage_weight=0.10,
        candidate_selector_pointwise_length_bias_weight=0.5,
        candidate_selector_pairwise_position_bias_weight=0.5,
        candidate_selector_pairwise_position_pairs=1,
        candidate_selector_pairwise_position_bias_scale=1.0,
        candidate_selector_signal_normalization="none",
        candidate_selector_uncertainty_view="pointwise",
        candidate_selector_length_aug_suffix="Additional context: This repeats the same answer without adding new useful information.",
        candidate_selector_density_k=10,
        candidate_selector_embedding_model=lw.DEFAULT_SELECTOR_EMBEDDING_MODEL,
        candidate_selector_embedding_max_length=lw.DEFAULT_SELECTOR_EMBEDDING_MAX_LENGTH,
        candidate_selector_embedding_batch_size=64,
        candidate_selector_embedding_device="auto",
        candidate_selector_embedding_pooling="cls",
        candidate_selector_diversity_view="pointwise",
        candidate_selector_exploration_ratio=0.0,
        candidate_selector_entropy_weight=1.0,
        candidate_selector_score_std_weight=0.0,
        candidate_selector_predicted_coverage_weight=0.0,
        candidate_selector_proxy_warmup_epochs=3,
        candidate_selector_proxy_update_epochs=1,
        candidate_selector_proxy_mode="classifier_heads",
        reuse_selection_proxy_for_stage1=False,
        candidate_bert_selector_model="bert-base-uncased",
        candidate_bert_selector_max_length=512,
        candidate_bert_selector_freeze=True,
        candidate_bert_selector_unfreeze_last_n_layers=0,
        proxy_lr=1e-4,
        proxy_max_length=768,
        llama_multitask_mode="shared_head",
        pointwise_loss_type="ce",
        pointwise_distance_weight=0.0,
        pointwise_class_weight_mode="none",
        pointwise_class_weight_strength=1.0,
        budget_units=int(args.budget),
        pointwise_epochs=int(args.pointwise_epochs),
        pairwise_epochs=int(args.pairwise_epochs),
        listwise_epochs=int(args.listwise_epochs),
        per_device_batch_size=int(args.per_device_batch_size),
        gradient_accumulation_steps=int(args.gradient_accumulation_steps),
        learning_rate=float(args.learning_rate),
        max_length=int(args.max_length),
        max_new_tokens_pointwise=int(args.max_new_tokens_pointwise),
        max_new_tokens_pairwise=int(args.max_new_tokens_pairwise),
        max_new_tokens_listwise=int(args.max_new_tokens_listwise),
        eval_batch_size=int(args.eval_batch_size),
        eval_stages=str(args.eval_stages),
        stage2_pointwise_replay_ratio=int(args.stage2_pointwise_replay_ratio),
        stage3_pointwise_replay_ratio=int(args.stage3_pointwise_replay_ratio),
        stage3_pairwise_replay_ratio=int(args.stage3_pairwise_replay_ratio),
        merge_stage2_stage3=False,
        stage23_pointwise_replay_ratio=1,
        stage23_pairwise_weight=1.0,
        stage23_listwise_weight=1.0,
        stage23_pointwise_weight=1.0,
        stage23_epochs=1,
        pointwise_teacher_distill_weight=0.0,
        pointwise_teacher_distill_temperature=2.0,
        stage4_task_teacher_distill_weight=0.0,
        stage4_task_teacher_distill_temperature=2.0,
        stage4_replay_strategy="none",
        stage4_replay_fraction=0.25,
        stage4_epochs=1,
        stage4_listwise_multiplier=1,
        pairwise_order_augmentation=False,
        listwise_order_augmentation=False,
        score_min=int(args.score_min),
        score_max=int(args.score_max),
        fix_score_prefix_in_prompt=not bool(args.no_fix_score_prefix),
        use_lora=bool(args.use_lora),
        load_in_4bit=bool(args.load_in_4bit),
        pointwise_global_smooth_alpha=0.0,
        pointwise_global_smooth_mode="global_prior",
        pointwise_global_smooth_gaussian_sigma=1.0,
        pointwise_global_smooth_stages="all",
        pointwise_global_smooth_start_step=0,
        pointwise_global_smooth_warmup_steps=0,
        pointwise_global_smooth_start_pointwise_seen=0,
        pointwise_global_smooth_warmup_pointwise_seen=0,
        pointwise_global_smooth_prior=1.0,
        pointwise_global_smooth_init_prior_from_stage1=False,
        pointwise_global_smooth_freeze_prior=False,
        pointwise_global_smooth_uniform_mix=0.0,
        pointwise_global_smooth_adaptive_entropy=False,
        pointwise_global_smooth_trainable_alpha=False,
        pointwise_global_smooth_alpha_max=0.2,
        pointwise_global_smooth_alpha_reg=0.0,
        pointwise_global_smooth_alpha_lr=0.0,
        max_pointwise_eval_samples=int(args.max_pointwise_eval_samples),
        max_pairwise_eval_samples=int(args.max_pairwise_eval_samples),
        max_listwise_eval_samples=int(args.max_listwise_eval_samples),
        fsdp=str(getattr(args, "fsdp", "")),
        fsdp_transformer_layer_cls_to_wrap=str(
            getattr(args, "fsdp_transformer_layer_cls_to_wrap", "")
        ),
        fsdp_state_dict_type=str(getattr(args, "fsdp_state_dict_type", "FULL_STATE_DICT")),
        fsdp_activation_checkpointing=bool(getattr(args, "fsdp_activation_checkpointing", False)),
        fsdp_use_orig_params=bool(getattr(args, "fsdp_use_orig_params", True)),
        fsdp_save_all_stages=bool(getattr(args, "fsdp_save_all_stages", False)),
    )


def _eval_all(
    *,
    model: Any,
    tokenizer: Any,
    cfg: three.RunConfig,
    pointwise_eval: Sequence[base.PointwiseScoredExample],
    pairwise_eval: Sequence[base.PairwiseExample],
    listwise_eval: Sequence[lw.ListwiseExample],
) -> Dict[str, Dict[str, Any]]:
    return {
        "pointwise": base._evaluate_pointwise_sft(
            model=model,
            tokenizer=tokenizer,
            examples=pointwise_eval,
            max_length=int(cfg.max_length),
            batch_size=int(cfg.eval_batch_size),
            max_new_tokens=int(cfg.max_new_tokens_pointwise),
            score_min=int(cfg.score_min),
            score_max=int(cfg.score_max),
        ),
        "pairwise": base._evaluate_pairwise_sft(
            model=model,
            tokenizer=tokenizer,
            examples=pairwise_eval,
            max_length=int(cfg.max_length),
            batch_size=int(cfg.eval_batch_size),
            max_new_tokens=int(cfg.max_new_tokens_pairwise),
        ),
        "listwise": three._evaluate_listwise_sft(
            model=model,
            tokenizer=tokenizer,
            examples=listwise_eval,
            max_length=int(cfg.max_length),
            batch_size=int(cfg.eval_batch_size),
            max_new_tokens=int(cfg.max_new_tokens_listwise),
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["pointwise_only_one_answer", "trueval_three_stage"], required=True)
    parser.add_argument("--pointwise-5answers-dataset", default="train_with_selector/train_with_selector/data/newnew/train-20k.json")
    parser.add_argument("--pairwise-val-dataset", default="train_with_selector/train_with_selector/data/newnew/val-2k-eval.json")
    parser.add_argument("--listwise-val-dataset", default="train_with_selector/train_with_selector/data/newnew/val-2k-eval-listwise.json")
    parser.add_argument("--resume-stage1-model-dir", default="")
    parser.add_argument("--pairwise-train-dataset", default="", help="Optional separate pairwise training file for true-val controls.")
    parser.add_argument("--listwise-train-dataset", default="", help="Optional separate listwise training file for true-val controls.")
    parser.add_argument("--llama", default="llama/Meta-Llama-3-8B-Instruct/")
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--val-split-seed", type=int, default=55)
    parser.add_argument("--pointwise-val-answer-seed", type=int, default=65)
    parser.add_argument("--budget", type=int, default=750)
    parser.add_argument("--pointwise-train-samples", type=int, default=750)
    parser.add_argument("--pairwise-train-pairs", type=int, default=250)
    parser.add_argument("--listwise-train-examples", type=int, default=250)
    parser.add_argument("--pointwise-epochs", type=int, default=1)
    parser.add_argument("--pairwise-epochs", type=int, default=1)
    parser.add_argument("--listwise-epochs", type=int, default=1)
    parser.add_argument("--per-device-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--max-new-tokens-pointwise", type=int, default=16)
    parser.add_argument("--max-new-tokens-pairwise", type=int, default=8)
    parser.add_argument("--max-new-tokens-listwise", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--eval-stages", choices=["all", "final"], default="all")
    parser.add_argument("--stage2-pointwise-replay-ratio", type=int, default=0)
    parser.add_argument("--stage3-pointwise-replay-ratio", type=int, default=0)
    parser.add_argument("--stage3-pairwise-replay-ratio", type=int, default=0)
    parser.add_argument("--score-min", type=int, default=1)
    parser.add_argument("--score-max", type=int, default=10)
    parser.add_argument("--no-fix-score-prefix", action="store_true")
    parser.add_argument("--use-lora", action="store_true")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--max-pointwise-eval-samples", type=int, default=0)
    parser.add_argument("--max-pairwise-eval-samples", type=int, default=0)
    parser.add_argument("--max-listwise-eval-samples", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = _make_cfg(args)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    random.seed(int(cfg.seed))
    np.random.seed(int(cfg.seed))
    torch.manual_seed(int(cfg.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(cfg.seed))

    ds_path = base._resolve_existing_path(str(args.pointwise_5answers_dataset))
    pairwise_path = base._resolve_existing_path(str(args.pairwise_val_dataset))
    listwise_path = base._resolve_existing_path(str(args.listwise_val_dataset))
    pairwise_train_path = base._resolve_existing_path(str(args.pairwise_train_dataset)) if str(args.pairwise_train_dataset).strip() else ""
    listwise_train_path = base._resolve_existing_path(str(args.listwise_train_dataset)) if str(args.listwise_train_dataset).strip() else ""
    if not ds_path:
        raise FileNotFoundError(args.pointwise_5answers_dataset)
    if not pairwise_path:
        raise FileNotFoundError(args.pairwise_val_dataset)
    if not listwise_path:
        raise FileNotFoundError(args.listwise_val_dataset)

    _write_json(
        out / "config.json",
        {
            **asdict(cfg),
            "mode": str(args.mode),
            "pointwise_5answers_dataset": str(ds_path),
            "pairwise_val_dataset": str(pairwise_path),
            "listwise_val_dataset": str(listwise_path),
            "pairwise_train_dataset": str(pairwise_train_path),
            "listwise_train_dataset": str(listwise_train_path),
            "llama": str(args.llama),
            "pointwise_train_samples": int(args.pointwise_train_samples),
            "pairwise_train_pairs": int(args.pairwise_train_pairs),
            "listwise_train_examples": int(args.listwise_train_examples),
        },
    )

    questions, load_stats = base._load_scored_questions(str(ds_path), score_min=int(cfg.score_min), score_max=int(cfg.score_max))
    train_questions, val_questions, split_info = base._split_questions(
        questions,
        seed=int(cfg.val_split_seed),
        val_ratio=float(cfg.val_ratio),
    )
    pointwise_train, pointwise_train_rows, pointwise_train_stats = _select_one_answer_pointwise(
        train_questions,
        samples=int(args.pointwise_train_samples),
        seed=int(cfg.seed) + 101,
        score_min=int(cfg.score_min),
        score_max=int(cfg.score_max),
        fix_score_prefix_in_prompt=bool(cfg.fix_score_prefix_in_prompt),
    )
    pointwise_eval, pointwise_eval_rows, pointwise_eval_stats = base._build_single_answer_pointwise_eval_examples(
        val_questions,
        seed=int(cfg.pointwise_val_answer_seed),
        score_min=int(cfg.score_min),
        score_max=int(cfg.score_max),
        judge_system_prompt=base.JUDGE_SYSTEM_PROMPT_SCORE_ONLY,
        fix_score_prefix_in_prompt=bool(cfg.fix_score_prefix_in_prompt),
    )

    if str(args.mode) == "trueval_three_stage":
        if pairwise_train_path:
            pairwise_train_records = base._load_pairwise_abc_raw_records(str(pairwise_train_path))
            pairwise_train, pairwise_train_rows, pairwise_train_stats = base._build_pairwise_abc_examples_from_records(
                pairwise_train_records,
                dataset_path=str(pairwise_train_path),
                split_name="train_explicit",
                pairwise_system_prompt=base.DEFAULT_PAIRWISE_SYSTEM_PROMPT,
            )
            pairwise_train = pairwise_train[: int(args.pairwise_train_pairs)]
            pairwise_train_rows = pairwise_train_rows[: int(args.pairwise_train_pairs)]
            pairwise_split = {"split": "explicit_train_eval", "train_dataset": str(pairwise_train_path), "eval_dataset": str(pairwise_path)}
            pairwise_eval, pairwise_eval_rows, pairwise_eval_stats = base._load_pairwise_abc_eval_dataset(
                str(pairwise_path), pairwise_system_prompt=base.DEFAULT_PAIRWISE_SYSTEM_PROMPT
            )
        else:
            pairwise_train, pairwise_train_rows, pairwise_train_stats, pairwise_eval, pairwise_eval_rows, pairwise_eval_stats, pairwise_split = _split_pairwise_trueval(
                str(pairwise_path), train_pairs=int(args.pairwise_train_pairs), seed=int(cfg.seed) + 202
            )
        if listwise_train_path:
            listwise_train, listwise_train_rows, listwise_train_stats = lw._load_listwise_eval_dataset(str(listwise_train_path))
            listwise_train = listwise_train[: int(args.listwise_train_examples)]
            listwise_train_rows = listwise_train_rows[: int(args.listwise_train_examples)]
            listwise_split = {"split": "explicit_train_eval", "train_dataset": str(listwise_train_path), "eval_dataset": str(listwise_path)}
            listwise_eval, listwise_eval_rows, listwise_eval_stats = lw._load_listwise_eval_dataset(str(listwise_path))
        else:
            listwise_train, listwise_train_rows, listwise_train_stats, listwise_eval, listwise_eval_rows, listwise_eval_stats, listwise_split = _split_listwise_trueval(
                str(listwise_path), train_examples_count=int(args.listwise_train_examples), seed=int(cfg.seed) + 303
            )
    else:
        pairwise_train, pairwise_train_rows, pairwise_train_stats = [], [], {"examples": 0, "split": "not_trained"}
        listwise_train, listwise_train_rows, listwise_train_stats = [], [], {"examples": 0, "split": "not_trained"}
        pairwise_eval, pairwise_eval_rows, pairwise_eval_stats = base._load_pairwise_abc_eval_dataset(
            str(pairwise_path),
            pairwise_system_prompt=base.DEFAULT_PAIRWISE_SYSTEM_PROMPT,
        )
        listwise_eval, listwise_eval_rows, listwise_eval_stats = lw._load_listwise_eval_dataset(str(listwise_path))
        pairwise_split = {"split": "full_pairwise_val_eval_no_pairwise_train"}
        listwise_split = {"split": "full_listwise_val_eval_no_listwise_train"}

    if int(cfg.max_pointwise_eval_samples) > 0:
        pointwise_eval = pointwise_eval[: int(cfg.max_pointwise_eval_samples)]
        pointwise_eval_rows = pointwise_eval_rows[: int(cfg.max_pointwise_eval_samples)]
        pointwise_eval_stats["eval_cap"] = int(cfg.max_pointwise_eval_samples)
    if int(cfg.max_pairwise_eval_samples) > 0:
        pairwise_eval = pairwise_eval[: int(cfg.max_pairwise_eval_samples)]
        pairwise_eval_rows = pairwise_eval_rows[: int(cfg.max_pairwise_eval_samples)]
        pairwise_eval_stats["eval_cap"] = int(cfg.max_pairwise_eval_samples)
    if int(cfg.max_listwise_eval_samples) > 0:
        listwise_eval = listwise_eval[: int(cfg.max_listwise_eval_samples)]
        listwise_eval_rows = listwise_eval_rows[: int(cfg.max_listwise_eval_samples)]
        listwise_eval_stats["eval_cap"] = int(cfg.max_listwise_eval_samples)

    _write_json(out / "dataset_load_stats.json", load_stats)
    _write_json(out / "split_questions.json", split_info)
    _write_json(out / "pointwise_train_stats.json", pointwise_train_stats)
    _write_json(out / "pointwise_eval_stats.json", pointwise_eval_stats)
    _write_json(out / "pairwise_train_stats.json", pairwise_train_stats)
    _write_json(out / "pairwise_eval_stats.json", pairwise_eval_stats)
    _write_json(out / "listwise_train_stats.json", listwise_train_stats)
    _write_json(out / "listwise_eval_stats.json", listwise_eval_stats)
    _write_json(out / "pairwise_trueval_split.json", pairwise_split)
    _write_json(out / "listwise_trueval_split.json", listwise_split)
    _write_jsonl(out / "pointwise_train.jsonl", pointwise_train_rows)
    _write_jsonl(out / "pointwise_eval.jsonl", pointwise_eval_rows)
    _write_jsonl(out / "pairwise_train.jsonl", pairwise_train_rows)
    _write_jsonl(out / "pairwise_eval.jsonl", pairwise_eval_rows)
    _write_jsonl(out / "listwise_train.jsonl", listwise_train_rows)
    _write_jsonl(out / "listwise_eval.jsonl", listwise_eval_rows)

    print("=" * 80)
    print(f"Start run: {args.mode}")
    print("=" * 80)
    print(f"output_dir={out}")
    print(f"pointwise train/eval = {len(pointwise_train)} / {len(pointwise_eval)}")
    print(f"pairwise train/eval  = {len(pairwise_train)} / {len(pairwise_eval)}")
    print(f"listwise train/eval  = {len(listwise_train)} / {len(listwise_eval)}")

    point_items = three._pointwise_items(pointwise_train, cfg)
    pair_items = three._pairwise_items(pairwise_train)
    list_items = three._listwise_items(listwise_train)

    train_stats: Dict[str, Any] = {}
    pointwise_metrics: Dict[str, Any] = {}
    pairwise_metrics: Dict[str, Any] = {}
    listwise_metrics: Dict[str, Any] = {}

    if str(args.resume_stage1_model_dir):
        model, tokenizer = three._load_stage1_resume_model(
            base_model_path=str(args.llama),
            adapter_dir=Path(args.resume_stage1_model_dir),
            cfg=cfg,
        )
        train_stats["stage1_pointwise"] = {
            "resumed": True,
            "adapter_dir": str(args.resume_stage1_model_dir),
        }
    else:
        train_stats["stage1_pointwise"], model, tokenizer = three._train_sft_on_items(
            model_name_or_path=str(args.llama),
            model=None,
            tokenizer=None,
            items=point_items,
            output_dir=out / "stage1_pointwise_sft_model",
            cfg=cfg,
            stage_name="stage1_pointwise",
        )
    _write_json(out / "train_stats_stage1_pointwise_sft.json", train_stats["stage1_pointwise"])
    if str(args.eval_stages) == "all" or str(args.mode) == "pointwise_only_one_answer":
        metrics = _eval_all(
            model=model,
            tokenizer=tokenizer,
            cfg=cfg,
            pointwise_eval=pointwise_eval,
            pairwise_eval=pairwise_eval,
            listwise_eval=listwise_eval,
        )
        pointwise_metrics["after_stage1"] = metrics["pointwise"]
        pairwise_metrics["after_stage1"] = metrics["pairwise"]
        listwise_metrics["after_stage1"] = metrics["listwise"]
        _write_json(out / "metrics_pointwise_after_stage1.json", metrics["pointwise"])
        _write_json(out / "metrics_pairwise_after_stage1.json", metrics["pairwise"])
        _write_json(out / "metrics_listwise_after_stage1.json", metrics["listwise"])

    if str(args.mode) == "trueval_three_stage":
        stage2_items = three._with_pointwise_replay(
            pair_items,
            point_items,
            replay_ratio=int(cfg.stage2_pointwise_replay_ratio),
            seed=int(cfg.seed) + 409,
        )
        train_stats["stage2_pairwise"], model, tokenizer = three._train_sft_on_items(
            model_name_or_path=None,
            model=model,
            tokenizer=tokenizer,
            items=stage2_items,
            output_dir=out / "stage2_pairwise_sft_model",
            cfg=cfg,
            stage_name="stage2_pairwise",
        )
        _write_json(out / "train_stats_stage2_pairwise_sft.json", train_stats["stage2_pairwise"])
        if str(args.eval_stages) == "all":
            metrics = _eval_all(
                model=model,
                tokenizer=tokenizer,
                cfg=cfg,
                pointwise_eval=pointwise_eval,
                pairwise_eval=pairwise_eval,
                listwise_eval=listwise_eval,
            )
            pointwise_metrics["after_stage2"] = metrics["pointwise"]
            pairwise_metrics["after_stage2"] = metrics["pairwise"]
            listwise_metrics["after_stage2"] = metrics["listwise"]
            _write_json(out / "metrics_pointwise_after_stage2.json", metrics["pointwise"])
            _write_json(out / "metrics_pairwise_after_stage2.json", metrics["pairwise"])
            _write_json(out / "metrics_listwise_after_stage2.json", metrics["listwise"])

        stage3_items = three._with_pointwise_replay(
            list_items,
            point_items,
            replay_ratio=int(cfg.stage3_pointwise_replay_ratio),
            seed=int(cfg.seed) + 509,
        )
        stage3_items = three._with_pointwise_replay(
            stage3_items,
            pair_items,
            replay_ratio=int(cfg.stage3_pairwise_replay_ratio),
            seed=int(cfg.seed) + 609,
            reference_count=len(list_items),
        )
        train_stats["stage3_listwise"], model, tokenizer = three._train_sft_on_items(
            model_name_or_path=None,
            model=model,
            tokenizer=tokenizer,
            items=stage3_items,
            output_dir=out / "stage3_listwise_sft_model",
            cfg=cfg,
            stage_name="stage3_listwise",
        )
        _write_json(out / "train_stats_stage3_listwise_sft.json", train_stats["stage3_listwise"])
        metrics = _eval_all(
            model=model,
            tokenizer=tokenizer,
            cfg=cfg,
            pointwise_eval=pointwise_eval,
            pairwise_eval=pairwise_eval,
            listwise_eval=listwise_eval,
        )
        pointwise_metrics["after_stage3"] = metrics["pointwise"]
        pairwise_metrics["after_stage3"] = metrics["pairwise"]
        listwise_metrics["after_stage3"] = metrics["listwise"]
        _write_json(out / "metrics_pointwise_after_stage3.json", metrics["pointwise"])
        _write_json(out / "metrics_pairwise_after_stage3.json", metrics["pairwise"])
        _write_json(out / "metrics_listwise_after_stage3.json", metrics["listwise"])

    summary = {
        "mode": str(args.mode),
        "train_budget": {
            "budget": int(args.budget),
            "pointwise_train": int(len(pointwise_train)),
            "pairwise_train": int(len(pairwise_train)),
            "listwise_train": int(len(listwise_train)),
            "pointwise_one_answer_per_question": True,
            "pairwise_trueval_training": str(args.mode) == "trueval_three_stage",
            "listwise_trueval_training": str(args.mode) == "trueval_three_stage",
            "stage2_pointwise_replay_ratio": int(cfg.stage2_pointwise_replay_ratio),
            "stage3_pointwise_replay_ratio": int(cfg.stage3_pointwise_replay_ratio),
            "stage3_pairwise_replay_ratio": int(cfg.stage3_pairwise_replay_ratio),
        },
        "split_by_question": split_info,
        "pairwise_trueval_split": pairwise_split,
        "listwise_trueval_split": listwise_split,
        "pointwise": {"train": pointwise_train_stats, "eval": pointwise_eval_stats},
        "pairwise": {"train": pairwise_train_stats, "eval": pairwise_eval_stats},
        "listwise": {"train": listwise_train_stats, "eval": listwise_eval_stats},
        "pointwise_metrics": pointwise_metrics,
        "pairwise_metrics": pairwise_metrics,
        "listwise_metrics": listwise_metrics,
        "train_stats": train_stats,
    }
    _write_json(out / "summary.json", summary)
    compact = three._compact_metrics(summary)
    _write_json(out / "metrics_compact.json", compact)
    print("Run finished")
    print(json.dumps(compact, ensure_ascii=False, indent=2))
    print(f"Output directory: {out}")

    del model
    del tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
        version = str(getattr(torch, "__version__", ""))
        match = re.match(r"^(\d+)\.(\d+)", version)
        if match is not None:
            major = int(match.group(1))
            minor = int(match.group(2))
            if (major > 2) or (major == 2 and minor >= 1):
                os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    main()

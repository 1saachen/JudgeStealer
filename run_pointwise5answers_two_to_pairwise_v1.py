"""Train with pointwise_5answers_score only: select 2 answers per question, then convert to pairwise.

Pipeline
--------
1) Load one scored 5-answers pointwise dataset.
2) For each question, pick exactly 2 answers (default: maximum score-gap pair).
3) Stage-1: train pointwise head on the 2 selected scored answers.
4) Natural conversion: each selected answer-pair becomes one gold pairwise sample.
5) Stage-2: alternating training on converted pairwise data with pointwise replay.

Rationale
---------
This script uses sequential two-stage training (pointwise -> pairwise), which is stable
and easy to analyze. It also aligns with the requirement that each selected two-answer
pointwise tuple should naturally become one pairwise sample.

Example

CUDA_VISIBLE_DEVICES=4 python run_pointwise5answers_two_to_pairwise_v1.py \
  --pointwise-5answers-dataset train_with_selector/train_with_selector/data/new/5answers_new.json \
  --llama llama/Meta-Llama-3-8B-Instruct/ \
  --seed 42 \
  --pointwise-training-mode proxy \
  --pairwise-abc-eval-dataset train_with_selector/train_with_selector/data/new/3k_pairwise_AB_BC.json \
  --pair-selection-strategy random \
  --pairwise-order-augmentation \
  --stage2-pointwise-replay-ratio 3 \
  --pointwise-loss-type ce \
  --pointwise-distance-weight 0.2 \
  --pairwise-abc-train-records 0 \
  --pointwise-epochs 1 \
  --pairwise-epochs 1 \
  --budget-units 500 \
  --pointwise-batch-size 32 \
  --pairwise-batch-size 32 \
  --out outputs/pointwise5answers_two_to_pairwise_compare_proxy_replay3_500_aug_abc0_ce
-------
CUDA_VISIBLE_DEVICES=7 python run_pointwise5answers_two_to_pairwise_v1.py \
  --pointwise-5answers-dataset train_with_selector/train_with_selector/data/new/5answers_new.json \
  --llama llama/Meta-Llama-3-8B-Instruct/ \
  --seed 42 \
  --pointwise-training-mode proxy \
  --train-selection-mode candidate_pair_selector \
  --candidate-selector-kind bert \
  --candidate-selector-target-task pointwise \
  --candidate-selector-init-pairs 100 \
  --candidate-selector-batch-size 32 \
  --candidate-selector-epochs 4 \
  --candidate-selector-gap-weight 1.0 \
  --candidate-selector-uncertainty-weight 1.0 \
  --pointwise-loss-type ce \
  --candidate-selector-kl-weight 0.0 \
  --pairwise-abc-eval-dataset train_with_selector/train_with_selector/data/new/3k_pairwise_AB_BC.json \
  --pair-selection-strategy random \
  --pairwise-order-augmentation \
  --stage2-pointwise-replay-ratio 3 \
  --pairwise-abc-train-records 0 \
  --pointwise-epochs 1 \
  --pairwise-epochs 1 \
  --budget-units 500 \
  --pointwise-batch-size 32 \
  --pairwise-batch-size 32 \
  --out outputs/2stage/pointwise5answers_two_to_pairwise_compare_bert_replay3_aug_abc0_500


CUDA_VISIBLE_DEVICES=0 python run_pointwise5answers_two_to_pairwise_v1.py \
  --pointwise-5answers-dataset train_with_selector/train_with_selector/data/new/5answers_new.json \
  --llama llama/Meta-Llama-3-8B-Instruct/ \
  --train-selection-mode candidate_pair_selector \
  --internal-val-mode question_single_answer \
  --candidate-selector-uncertainty-weight 0.8 \
  --candidate-selector-kl-weight 0.2 \
  --candidate-selector-init-pairs 500 \
  --candidate-selector-batch-size 32 \
  --candidate-selector-target-task pointwise \
  --candidate-selector-score-bin-weight 0.0 \
  --pointwise-batch-size 32 \
  --pairwise-batch-size 32 \
  --candidate-selector-epochs 4 \
  --budget-units 3000 \
  --pointwise-epochs 1 \
  --pairwise-epochs 1 \
  --pointwise-class-weight-mode inv_sqrt \
  --no-pointwise-class-weight \
  --pointwise-class-weight-strength 0 \
  --candidate-selector-kind distribution \
  --candidate-distribution-score-weight 1.0 \
  --candidate-distribution-dataset-weight 0.0 \
  --candidate-distribution-gap-weight 0.0 \
  --out outputs/5answers_new/candidate_pair_3000_init500_distribution

"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
import re
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

try:
    import psutil  # type: ignore
except Exception:  # pragma: no cover
    psutil = None

import copy

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments as HFTrainingArguments

_DEFAULT_POINTWISE_FIXED_VAL_IDS = (
    "train_with_selector/train_with_selector/data/new/"
    "fixed_val_compare_multitask_point_pair_mixed_bert_finetune/pointwise_val_ids.json"
)
_DEFAULT_PAIRWISE_FIXED_VAL_IDS = (
    "train_with_selector/train_with_selector/data/new/"
    "fixed_val_compare_multitask_point_pair_mixed_bert_finetune/pairwise_val_ids.json"
)
_DEFAULT_EXTERNAL_POINTWISE_EVAL_DATASET = "train_with_selector/train_with_selector/data/new/30K_pointwise.json"
_DEFAULT_EXTERNAL_PAIRWISE_EVAL_DATASET = "train_with_selector/train_with_selector/data/new/30k_pairwise.json"

from train_with_selector.train_with_selector.data.judge_dataset import (
    DEFAULT_JUDGE_SYSTEM_PROMPT,
    JUDGE_SYSTEM_PROMPT_SCORE_ONLY,
    JUDGE_SYSTEM_PROMPT_WITH_EXPLANATION,
    build_judge_prompt,
    get_judge_system_prompt,
    load_judge_json,
    score_to_class,
)
from train_with_selector.train_with_selector.data.pairwise_dataset import (
    DEFAULT_PAIRWISE_SYSTEM_PROMPT,
    LABEL_A,
    LABEL_B,
    LABEL_TIE,
    PairwiseExample,
    build_pairwise_prompt,
    label_to_token,
    load_pairwise_json,
)
from train_with_selector.train_with_selector.models.llama_shared_multitask_proxy import LlamaSharedMultiTaskProxyModel
from train_with_selector.train_with_selector.models.llama_shared_proxy import LlamaSharedProxyModel
from train_with_selector.train_with_selector.selector.binary_selector import BertBinarySelector
from train_with_selector.train_with_selector.selector.shared_llama_selector_v2 import SharedLlamaSelectorV2


@dataclass(frozen=True)
class AnswerWithScore:
    model: str
    output: str
    score: int
    reason: str = ""


@dataclass(frozen=True)
class SelectedQuestionPair:
    question_id: int
    source_id: int
    dataset: str
    instruction: str
    input_text: str
    answer_a: AnswerWithScore
    answer_b: AnswerWithScore


@dataclass(frozen=True)
class PointwiseScoredExample:
    row_id: int
    question_id: int
    source_id: int
    dataset: str
    instruction: str
    input_text: str
    model: str
    output: str
    score: int
    label: int
    prompt: str
    reason: str = ""

    def __str__(self) -> str:  # noqa: D105
        return self.prompt


@dataclass(frozen=True)
class CandidatePairExample:
    id: int
    group_id: int
    question_id: int
    source_id: int
    dataset: str
    model_a: str
    model_b: str
    score_a: int
    score_b: int
    score_gap: int
    prompt: str
    label: int
    selected_pair: SelectedQuestionPair

    def __str__(self) -> str:  # noqa: D105
        return self.prompt


@dataclass
class RunConfig:
    seed: int
    val_ratio: float
    val_split_seed: int
    pointwise_val_answer_seed: int
    internal_val_mode: str

    pair_selection_strategy: str
    train_selection_mode: str
    randomize_pair_order: bool
    budget_units: int
    budget_sampling_mode: str
    candidate_selector_kind: str
    candidate_selector_init_pairs: int
    candidate_selector_batch_size: int
    candidate_selector_epochs: int
    candidate_selector_buffer_maxlen: int
    candidate_selector_max_score_candidates: int
    candidate_distribution_score_weight: float
    candidate_distribution_dataset_weight: float
    candidate_distribution_gap_weight: float
    candidate_selector_distribution_rank_weight: float
    candidate_selector_distribution_rank_top_k: int
    candidate_selector_gap_weight: float
    candidate_selector_score_bin_weight: float
    candidate_selector_uncertainty_weight: float
    candidate_selector_kl_weight: float
    candidate_selector_target_task: str
    candidate_selector_one_per_question: bool
    candidate_bert_selector_model: str
    candidate_bert_selector_max_length: int
    candidate_bert_selector_freeze: bool
    candidate_bert_selector_unfreeze_last_n_layers: int
    pointwise_fixed_val_ids_file: str
    pairwise_fixed_val_ids_file: str
    strict_fixed_val_ids: bool
    use_external_fixed_eval: bool
    external_pointwise_eval_dataset: str
    external_pairwise_eval_dataset: str
    pairwise_abc_eval_dataset: str
    pairwise_abc_train_records: int
    pairwise_abc_train_ratio: float
    pairwise_abc_split_seed: int

    pointwise_epochs: int
    pairwise_epochs: int
    pointwise_batch_size: int
    pairwise_batch_size: int
    stage2_pointwise_replay_ratio: int

    score_min: int
    score_max: int
    drop_tie_pairwise: bool
    pairwise_order_augmentation: bool
    fix_score_prefix_in_prompt: bool

    # Proxy mode parameters
    proxy_lr: float
    proxy_max_length: int
    load_in_4bit: bool
    llama_multitask_mode: str

    pointwise_only: bool
    pointwise_training_mode: str  # "proxy" or "sft"
    pointwise_loss_type: str  # "ce", "ce_mse", "ce_cost", "ordinal", or "coral"
    pointwise_distance_weight: float
    pointwise_class_weight_mode: str  # "none" or "inv_sqrt"
    pointwise_class_weight_strength: float

    # SFT mode parameters
    training_mode: str  # "proxy" or "sft"
    sft_lr: float
    sft_per_device_batch_size: int
    sft_gradient_accumulation_steps: int
    sft_max_length: int
    sft_use_lora: bool
    sft_load_in_4bit: bool
    sft_stage2_mix_mode: str
    sft_stage2_pairs_per_batch: int
    sft_single_stage_pairbatch: bool
    sft_pointwise_global_smooth_alpha: float
    sft_pointwise_global_smooth_start_step: int
    sft_pointwise_global_smooth_warmup_steps: int
    sft_pointwise_global_smooth_prior: float
    sft_pointwise_global_smooth_trainable_alpha: bool
    sft_pointwise_global_smooth_alpha_max: float
    sft_pointwise_global_smooth_alpha_reg: float
    sft_pointwise_global_smooth_alpha_lr: float


def _safe_int(x: Any, default: int = -1) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _compute_pointwise_class_weights(
    examples: Sequence[PointwiseScoredExample],
    *,
    num_labels: int,
    mode: str,
    strength: float,
) -> Optional[np.ndarray]:
    mode_s = str(mode).strip().lower()
    if mode_s in {"", "none"}:
        return None
    if mode_s not in {"inv_sqrt"}:
        raise ValueError(f"unsupported pointwise_class_weight_mode={mode!r}")
    strength_f = float(strength)
    if strength_f < 0.0:
        raise ValueError(f"pointwise_class_weight_strength must be >= 0, got {strength!r}")

    counts = np.zeros((int(num_labels),), dtype=np.int64)
    for x in examples:
        label = int(x.label)
        if 0 <= label < int(num_labels):
            counts[label] += 1

    raw_weights = np.ones((int(num_labels),), dtype=np.float32)
    nonzero = counts > 0
    if bool(nonzero.any()):
        raw_weights[nonzero] = 1.0 / np.sqrt(counts[nonzero].astype(np.float32))
        raw_weights = raw_weights / float(raw_weights.mean())
    weights = 1.0 + strength_f * (raw_weights - 1.0)
    return weights.astype(np.float32)


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _first_present(metrics: Optional[Dict[str, Any]], *keys: str) -> Any:
    if not isinstance(metrics, dict):
        return None
    for key in keys:
        if key in metrics and metrics[key] is not None:
            return metrics[key]
    return None


def _compact_pointwise_metrics(metrics: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if not isinstance(metrics, dict):
        return out

    n = _first_present(metrics, "n")
    if n is not None:
        out["n"] = int(n)

    mappings = (
        ("acc", ("proxy_acc", "sft_acc")),
        ("within1_acc", ("proxy_within1", "sft_within1")),
        ("mae", ("proxy_mae", "sft_mae")),
        ("rmse", ("proxy_rmse", "sft_rmse")),
    )
    for out_key, src_keys in mappings:
        value = _first_present(metrics, *src_keys)
        if value is not None:
            out[out_key] = float(value)

    invalid_pred = _first_present(metrics, "sft_invalid_pred")
    if invalid_pred is not None:
        out["invalid_pred"] = int(invalid_pred)
    return out


def _compact_pairwise_metrics(metrics: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if not isinstance(metrics, dict):
        return out

    n = _first_present(metrics, "n")
    if n is not None:
        out["n"] = int(n)

    mappings = (
        ("acc", ("proxy_acc", "sft_acc")),
        ("tie_rate", ("proxy_tie_rate", "sft_tie_rate")),
    )
    for out_key, src_keys in mappings:
        value = _first_present(metrics, *src_keys)
        if value is not None:
            out[out_key] = float(value)

    invalid_pred = _first_present(metrics, "sft_invalid_pred")
    if invalid_pred is not None:
        out["invalid_pred"] = int(invalid_pred)
    return out


def _select_final_stage(stage_metrics: Dict[str, Dict[str, Any]]) -> Optional[str]:
    for stage_name in ("after_stage2", "after_stage1", "before_stage2", "before_stage1"):
        if stage_name in stage_metrics:
            return stage_name
    return None


def _build_compact_metrics_summary(summary: Dict[str, Any]) -> Dict[str, Any]:
    pointwise_raw = summary.get("pointwise_metrics") or {}
    pairwise_raw = summary.get("pairwise_metrics") or {}
    pointwise: Dict[str, Any] = {}
    pairwise: Dict[str, Any] = {}

    for stage_name in ("before_stage1", "after_stage1", "before_stage2", "after_stage2"):
        pw_metrics = _compact_pointwise_metrics(pointwise_raw.get(stage_name))
        if pw_metrics:
            pointwise[stage_name] = pw_metrics
        pr_metrics = _compact_pairwise_metrics(pairwise_raw.get(stage_name))
        if pr_metrics:
            pairwise[stage_name] = pr_metrics

    pointwise_final_stage = _select_final_stage(pointwise)
    pairwise_final_stage = _select_final_stage(pairwise)

    train_budget = summary.get("train_budget") or {}
    compact = {
        "mode": str(summary.get("mode", "proxy_stage2")),
        "pointwise_training_mode": summary.get("pointwise_training_mode"),
        "eval_split": dict(summary.get("eval_split") or {}),
        "budget": {
            "requested_units": train_budget.get("budget_units"),
            "effective_units": train_budget.get("effective_budget_units"),
            "train_pairs": train_budget.get("train_pairs_after_budget"),
            "train_answers": train_budget.get("train_answers_after_budget"),
        },
        "pointwise": pointwise,
        "pairwise": pairwise,
    }
    if pointwise_final_stage is not None:
        compact["pointwise_final_stage"] = pointwise_final_stage
        compact["pointwise_final"] = dict(pointwise[pointwise_final_stage])
    if pairwise_final_stage is not None:
        compact["pairwise_final_stage"] = pairwise_final_stage
        compact["pairwise_final"] = dict(pairwise[pairwise_final_stage])
    return compact


def _fmt_metric(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        value_f = float(value)
    except Exception:
        return str(value)
    if not np.isfinite(value_f):
        return "nan"
    return f"{value_f:.4f}"


def _write_compact_metrics(base_out: Path, summary: Dict[str, Any]) -> Dict[str, Any]:
    compact = _build_compact_metrics_summary(summary)
    _write_json(base_out / "metrics_compact.json", compact)
    return compact


def _print_compact_run_summary(base_out: Path, compact: Dict[str, Any]) -> None:
    print("\n" + "=" * 60)
    print("Run finished")
    print("=" * 60)

    pointwise = compact.get("pointwise") or {}
    pairwise = compact.get("pairwise") or {}
    pointwise_final = compact.get("pointwise_final") or {}
    pairwise_final = compact.get("pairwise_final") or {}

    pw_progress = []
    for stage_name in ("before_stage1", "after_stage1", "after_stage2"):
        stage_metrics = pointwise.get(stage_name)
        if isinstance(stage_metrics, dict) and "acc" in stage_metrics:
            pw_progress.append(f"{stage_name}={_fmt_metric(stage_metrics.get('acc'))}")
    if pw_progress:
        print(f"Pointwise acc: {' -> '.join(pw_progress)}")

    if pointwise_final:
        print(
            "Pointwise final: "
            f"acc={_fmt_metric(pointwise_final.get('acc'))} "
            f"within1={_fmt_metric(pointwise_final.get('within1_acc'))} "
            f"mae={_fmt_metric(pointwise_final.get('mae'))} "
            f"rmse={_fmt_metric(pointwise_final.get('rmse'))}"
        )

    pr_progress = []
    for stage_name in ("before_stage1", "after_stage1", "after_stage2"):
        stage_metrics = pairwise.get(stage_name)
        if isinstance(stage_metrics, dict) and "acc" in stage_metrics:
            pr_progress.append(f"{stage_name}={_fmt_metric(stage_metrics.get('acc'))}")
    if pr_progress:
        print(f"Pairwise acc: {' -> '.join(pr_progress)}")

    if pairwise_final:
        print(
            "Pairwise final: "
            f"acc={_fmt_metric(pairwise_final.get('acc'))} "
            f"tie={_fmt_metric(pairwise_final.get('tie_rate'))}"
        )

    print(f"Compact metrics: {base_out / 'metrics_compact.json'}")
    print(f"Output directory: {base_out}")


def _coerce_int_set(items: Sequence[Any], *, field_name: str, source_path: Path) -> Set[int]:
    out: Set[int] = set()
    for i, v in enumerate(items):
        try:
            out.add(int(v))
        except Exception as e:
            raise ValueError(f"{source_path}: field {field_name}[{i}]={v!r} cannot convert to int") from e
    return out


def _load_fixed_id_payload_detail(path: str) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"fixed val id file not found: {p}")

    raw = json.loads(p.read_text(encoding="utf-8"))
    ids: Set[int] = set()
    group_ids: Set[int] = set()
    query_unit = ""
    id_field = "id"
    group_id_field = "group_id"

    if isinstance(raw, list):
        ids = _coerce_int_set(raw, field_name="ids", source_path=p)
    elif isinstance(raw, dict):
        if isinstance(raw.get("ids"), list):
            ids = _coerce_int_set(raw["ids"], field_name="ids", source_path=p)
        if isinstance(raw.get("group_ids"), list):
            group_ids = _coerce_int_set(raw["group_ids"], field_name="group_ids", source_path=p)
        query_unit = str(raw.get("query_unit", "") or "")
        id_field = str(raw.get("id_field", "id") or "id")
        group_id_field = str(raw.get("group_id_field", "group_id") or "group_id")
    else:
        raise ValueError(f"{p}: JSON only supports list or dict-with-ids/group_ids")

    if not ids and not group_ids:
        raise ValueError(f"{p}: fixed val ids/group_ids is empty")

    return {
        "ids": ids,
        "group_ids": group_ids,
        "query_unit": query_unit,
        "id_field": id_field,
        "group_id_field": group_id_field,
        "source_path": str(p),
    }


def _load_fixed_id_payload(path: str) -> Set[int]:
    payload = _load_fixed_id_payload_detail(path)
    ids = set(payload["ids"])
    if not ids:
        raise ValueError(f"{path}: fixed val ids is empty")
    return ids


def _select_examples_by_fixed_ids(
    examples: Sequence[Any],
    *,
    val_ids: Set[int],
    id_attr: str,
    task_name: str,
    fixed_ids_path: str,
    strict_missing: bool,
) -> Tuple[List[Any], Dict[str, Any]]:
    if not val_ids:
        raise RuntimeError(f"{task_name}: fixed eval ids is empty")

    selected: List[Any] = []
    seen: Set[int] = set()

    for ex in list(examples):
        ex_id = int(getattr(ex, id_attr, -1))
        if ex_id in val_ids:
            selected.append(ex)
            seen.add(ex_id)

    missing = sorted(val_ids - seen)
    info = {
        "filter_mode": "fixed_ids",
        "id_attr": str(id_attr),
        "fixed_val_ids_file": str(fixed_ids_path),
        "fixed_val_ids_strict": bool(strict_missing),
        "fixed_val_ids_total": int(len(val_ids)),
        "fixed_val_ids_matched": int(len(seen)),
        "fixed_val_ids_missing": int(len(missing)),
        "fixed_val_ids_missing_preview": missing[:20],
        "eval_size": int(len(selected)),
    }

    if missing and bool(strict_missing):
        raise RuntimeError(f"{task_name}: fixed eval ids has {len(missing)} unmatched ids, preview={missing[:10]}")
    if not selected:
        raise RuntimeError(f"{task_name}: fixed eval filtering produced empty eval set")
    return selected, info


def _select_pairwise_examples_by_fixed_ids(
    examples: Sequence[PairwiseExample],
    *,
    val_ids: Set[int],
    val_group_ids: Set[int],
    query_unit: str,
    fixed_ids_path: str,
    strict_missing: bool,
) -> Tuple[List[PairwiseExample], Dict[str, Any]]:
    unit = str(query_unit or "example")
    pool = list(examples)

    if unit == "group":
        target_group_ids = set(val_group_ids)
        if not target_group_ids:
            if not val_ids:
                raise RuntimeError("pairwise external eval requires group_ids or ids when query_unit=group")
            id_to_gid: Dict[int, int] = {}
            for ex in pool:
                ex_id = int(getattr(ex, "id", -1))
                gid = int(getattr(ex, "group_id", -1))
                if ex_id >= 0 and gid >= 0:
                    id_to_gid[ex_id] = gid
            missing_ids = sorted(i for i in val_ids if i not in id_to_gid)
            if missing_ids and bool(strict_missing):
                raise RuntimeError(
                    f"pairwise external eval ids has {len(missing_ids)} unmatched ids, preview={missing_ids[:10]}"
                )
            target_group_ids = {id_to_gid[i] for i in val_ids if i in id_to_gid}

        selected = [ex for ex in pool if int(getattr(ex, "group_id", -1)) in target_group_ids]
        seen_group_ids = {int(getattr(ex, "group_id", -1)) for ex in selected}
        missing_group_ids = sorted(target_group_ids - seen_group_ids)
        info = {
            "filter_mode": "fixed_group_ids",
            "query_unit": "group",
            "fixed_val_ids_file": str(fixed_ids_path),
            "fixed_val_ids_strict": bool(strict_missing),
            "fixed_group_ids_total": int(len(target_group_ids)),
            "fixed_group_ids_matched": int(len(seen_group_ids)),
            "fixed_group_ids_missing": int(len(missing_group_ids)),
            "fixed_group_ids_missing_preview": missing_group_ids[:20],
            "eval_size": int(len(selected)),
        }
        if missing_group_ids and bool(strict_missing):
            raise RuntimeError(
                f"pairwise external eval group_ids has {len(missing_group_ids)} unmatched ids, preview={missing_group_ids[:10]}"
            )
        if not selected:
            raise RuntimeError("pairwise external eval filtering produced empty eval set")
        return selected, info

    selected, info = _select_examples_by_fixed_ids(
        pool,
        val_ids=val_ids,
        id_attr="id",
        task_name="pairwise external eval",
        fixed_ids_path=fixed_ids_path,
        strict_missing=bool(strict_missing),
    )
    info["query_unit"] = "example"
    return [ex for ex in selected], info


def _load_external_fixed_eval_splits(
    *,
    pointwise_dataset_path: str,
    pairwise_dataset_path: str,
    pointwise_fixed_ids_path: str,
    pairwise_fixed_ids_path: str,
    score_min: int,
    score_max: int,
    fix_score_prefix_in_prompt: bool,
    strict_missing: bool,
) -> Tuple[List[Any], List[PairwiseExample], Dict[str, Any], Dict[str, Any]]:
    pointwise_all = load_judge_json(
        str(pointwise_dataset_path),
        system_prompt=JUDGE_SYSTEM_PROMPT_SCORE_ONLY,
        score_min=int(score_min),
        score_max=int(score_max),
        include_gold_score_in_prompt=False,
        fix_score_prefix_in_prompt=bool(fix_score_prefix_in_prompt),
    )
    pointwise_payload = _load_fixed_id_payload_detail(str(pointwise_fixed_ids_path))
    if not pointwise_payload["ids"]:
        raise RuntimeError("pointwise external eval requires non-empty ids")
    pointwise_eval, pointwise_info = _select_examples_by_fixed_ids(
        pointwise_all,
        val_ids=set(pointwise_payload["ids"]),
        id_attr="id",
        task_name="pointwise external eval",
        fixed_ids_path=str(pointwise_fixed_ids_path),
        strict_missing=bool(strict_missing),
    )
    pointwise_info.update(
        {
            "dataset_path": str(pointwise_dataset_path),
            "dataset_size": int(len(pointwise_all)),
        }
    )

    pairwise_all = load_pairwise_json(str(pairwise_dataset_path), system_prompt=DEFAULT_PAIRWISE_SYSTEM_PROMPT)
    pairwise_payload = _load_fixed_id_payload_detail(str(pairwise_fixed_ids_path))
    pairwise_eval, pairwise_info = _select_pairwise_examples_by_fixed_ids(
        pairwise_all,
        val_ids=set(pairwise_payload["ids"]),
        val_group_ids=set(pairwise_payload["group_ids"]),
        query_unit=str(pairwise_payload.get("query_unit", "") or "example"),
        fixed_ids_path=str(pairwise_fixed_ids_path),
        strict_missing=bool(strict_missing),
    )
    pairwise_info.update(
        {
            "dataset_path": str(pairwise_dataset_path),
            "dataset_size": int(len(pairwise_all)),
        }
    )

    return pointwise_eval, pairwise_eval, pointwise_info, pairwise_info


def _split_selected_pairs_by_fixed_ids(
    selected_pairs: Sequence[SelectedQuestionPair],
    *,
    val_ids: Set[int],
    fixed_ids_path: str,
    strict_missing: bool,
) -> Tuple[List[SelectedQuestionPair], List[SelectedQuestionPair], Dict[str, Any]]:
    train: List[SelectedQuestionPair] = []
    val: List[SelectedQuestionPair] = []
    seen: Set[int] = set()

    has_source_id = all(int(getattr(p, "source_id", -1)) > 0 for p in selected_pairs)
    id_field = "source_id" if has_source_id else "question_id"

    for p in selected_pairs:
        match_id = int(p.source_id) if has_source_id else int(p.question_id)
        if match_id in val_ids:
            val.append(p)
            seen.add(match_id)
        else:
            train.append(p)

    missing = sorted(val_ids - seen)
    info = {
        "split_mode": "fixed_ids",
        "fixed_val_id_field": str(id_field),
        "fixed_val_ids_file": str(fixed_ids_path),
        "fixed_val_ids_strict": bool(strict_missing),
        "fixed_val_ids_total": int(len(val_ids)),
        "fixed_val_ids_matched": int(len(seen)),
        "fixed_val_ids_missing": int(len(missing)),
        "fixed_val_ids_missing_preview": missing[:20],
        "train_questions": int(len(train)),
        "val_questions": int(len(val)),
    }

    if missing and bool(strict_missing):
        raise RuntimeError(
            f"fixed val ids has {len(missing)} unmatched ids in selected question set, preview={missing[:10]}"
        )

    if not val:
        raise RuntimeError("fixed val split produced empty val set")
    if not train:
        raise RuntimeError("fixed val split produced empty train set")
    return train, val, info


def _split_questions(
    questions: Sequence[Dict[str, Any]],
    *,
    seed: int,
    val_ratio: float,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    all_questions = list(questions)
    n = int(len(all_questions))
    if n <= 1 or float(val_ratio) <= 0.0:
        return all_questions, [], {
            "split_mode": "all_train",
            "val_ratio": float(val_ratio),
            "train_questions": int(n),
            "val_questions": 0,
        }

    k = int(round(float(val_ratio) * n))
    k = max(1, min(n - 1, k))

    rng = np.random.default_rng(int(seed))
    picked = rng.choice(n, size=k, replace=False).tolist()
    val_idx = {int(i) for i in picked}

    train = [x for i, x in enumerate(all_questions) if i not in val_idx]
    val = [x for i, x in enumerate(all_questions) if i in val_idx]

    info = {
        "split_mode": "random_by_question",
        "val_ratio": float(val_ratio),
        "train_questions": int(len(train)),
        "val_questions": int(len(val)),
    }
    return train, val, info


def _split_questions_by_fixed_ids(
    questions: Sequence[Dict[str, Any]],
    *,
    val_ids: Set[int],
    fixed_ids_path: str,
    strict_missing: bool,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    train: List[Dict[str, Any]] = []
    val: List[Dict[str, Any]] = []
    seen: Set[int] = set()

    has_source_id = all(int(_safe_int(q.get("source_id", -1), default=-1)) > 0 for q in questions)
    id_field = "source_id" if has_source_id else "question_id"

    for q in questions:
        match_id = (
            int(_safe_int(q.get("source_id", q.get("question_id", -1)), default=-1))
            if has_source_id
            else int(_safe_int(q.get("question_id", -1), default=-1))
        )
        if match_id in val_ids:
            val.append(q)
            seen.add(match_id)
        else:
            train.append(q)

    missing = sorted(val_ids - seen)
    info = {
        "split_mode": "fixed_ids",
        "fixed_val_id_field": str(id_field),
        "fixed_val_ids_file": str(fixed_ids_path),
        "fixed_val_ids_strict": bool(strict_missing),
        "fixed_val_ids_total": int(len(val_ids)),
        "fixed_val_ids_matched": int(len(seen)),
        "fixed_val_ids_missing": int(len(missing)),
        "fixed_val_ids_missing_preview": missing[:20],
        "train_questions": int(len(train)),
        "val_questions": int(len(val)),
    }

    if missing and bool(strict_missing):
        raise RuntimeError(
            f"fixed val ids has {len(missing)} unmatched ids in question set, preview={missing[:10]}"
        )
    if not val:
        raise RuntimeError("fixed val split produced empty val set")
    if not train:
        raise RuntimeError("fixed val split produced empty train set")
    return train, val, info


def _apply_budget_to_train_pairs(
    train_pairs: Sequence[SelectedQuestionPair],
    *,
    budget_units: int,
    seed: int,
    sampling_mode: str = "choice",
) -> Tuple[List[SelectedQuestionPair], Dict[str, Any]]:
    pairs = list(train_pairs)
    n = int(len(pairs))
    sampling_mode = str(sampling_mode)
    if sampling_mode not in {"choice", "prefix"}:
        raise ValueError(f"unsupported budget sampling mode: {sampling_mode}")

    info: Dict[str, Any] = {
        "budget_units": int(budget_units),
        "budget_sampling_mode": sampling_mode,
        "train_pairs_before_budget": int(n),
        "train_answers_before_budget": int(n * 2),
        "budget_applied": False,
        "max_train_questions_by_budget": int(n),
        "train_pairs_after_budget": int(n),
        "train_answers_after_budget": int(n * 2),
        "dropped_train_pairs_by_budget": 0,
        "effective_budget_units": int(n * 2),
    }

    if int(budget_units) <= 0:
        return pairs, info

    max_q = int(budget_units) // 2
    if max_q <= 0:
        raise ValueError("budget-units must be >= 2 when > 0")

    info["budget_applied"] = True
    info["max_train_questions_by_budget"] = int(max_q)
    info["effective_budget_units"] = int(max_q * 2)

    if n <= max_q:
        return pairs, info

    rng = np.random.default_rng(int(seed))
    if sampling_mode == "prefix":
        picked = rng.permutation(n).astype(np.int64).tolist()[:max_q]
    else:
        picked = rng.choice(n, size=max_q, replace=False).tolist()
    picked_set = {int(i) for i in picked}
    kept = [x for i, x in enumerate(pairs) if i in picked_set]

    info["train_pairs_after_budget"] = int(len(kept))
    info["train_answers_after_budget"] = int(len(kept) * 2)
    info["dropped_train_pairs_by_budget"] = int(n - len(kept))
    return kept, info


def _log_memory_usage(prefix: str = "") -> Dict[str, float]:
    if psutil is None:
        print(f"[{prefix}] memory stats skipped (psutil not installed)")
        return {}

    process = psutil.Process()
    mem_info = process.memory_info()
    vm = psutil.virtual_memory()
    gpu_mem = {}
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            gpu_mem[f"gpu_{i}_allocated_gb"] = torch.cuda.memory_allocated(i) / 1024**3
            gpu_mem[f"gpu_{i}_reserved_gb"] = torch.cuda.memory_reserved(i) / 1024**3

    mem_stats: Dict[str, float] = {
        "process_rss_gb": mem_info.rss / 1024**3,
        "process_vms_gb": mem_info.vms / 1024**3,
        "system_used_gb": vm.used / 1024**3,
        "system_available_gb": vm.available / 1024**3,
        "system_percent": float(vm.percent),
        **gpu_mem,
    }
    msg = f"[{prefix}] RSS={mem_stats['process_rss_gb']:.2f}GB system={mem_stats['system_percent']:.1f}%"
    if gpu_mem:
        msg += ", " + ", ".join(
            [f"GPU{i}={mem_stats[f'gpu_{i}_allocated_gb']:.2f}GB" for i in range(torch.cuda.device_count())]
        )
    print(msg)
    return mem_stats


def _resolve_existing_path(path_str: Optional[str]) -> Optional[str]:
    """Resolve common path-layout mismatches in this repository."""
    if path_str is None:
        return None
    raw = str(path_str).strip()
    if raw == "":
        return ""

    p = Path(raw).expanduser()
    if p.exists():
        return str(p)

    candidates: List[Path] = []
    candidates.append((_THIS_DIR / p).resolve())

    if len(p.parts) >= 1 and p.parts[0] == "train_with_selector":
        nested = Path("train_with_selector") / p
        candidates.append((_THIS_DIR / nested).resolve())
        stripped = Path(*p.parts[1:]) if len(p.parts) > 1 else Path()
        if str(stripped):
            candidates.append((_THIS_DIR / "train_with_selector" / "train_with_selector" / stripped).resolve())

    candidates.append((_THIS_DIR / "train_with_selector" / p).resolve())
    candidates.append((_THIS_DIR / "train_with_selector" / "train_with_selector" / p).resolve())

    seen: set[str] = set()
    for c in candidates:
        cs = str(c)
        if cs in seen:
            continue
        seen.add(cs)
        if c.exists():
            return cs
    return raw


def _load_scored_questions(
    path: str,
    *,
    score_min: int,
    score_max: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("pointwise_5answers_score dataset JSON must be a list")

    questions: List[Dict[str, Any]] = []
    stats: Dict[str, int] = {
        "records": int(len(raw)),
        "records_models_list_format": 0,
        "records_flat_5answers_format": 0,
        "records_flat_abc_format": 0,
        "answers_total": 0,
        "answers_valid": 0,
        "answers_missing_score": 0,
        "answers_score_out_of_range": 0,
        "answers_empty_output": 0,
        "questions_with_ge2_answers": 0,
        "questions_skipped_lt2_answers": 0,
    }

    for rec_i, rec in enumerate(raw):
        if not isinstance(rec, dict):
            continue

        qid = _safe_int(rec.get("id", rec_i + 1), default=rec_i + 1)
        source_id = _safe_int(rec.get("source_id", qid), default=qid)
        if source_id <= 0:
            source_id = int(qid)
        dataset = str(rec.get("dataset", ""))
        instruction = str(rec.get("Instruction", rec.get("instruction", "")))
        input_text = str(rec.get("input", ""))

        models = rec.get("models", None)
        if isinstance(models, list):
            stats["records_models_list_format"] += 1
        else:
            models = []
            for ans_i in range(1, 6):
                model_key = f"model{ans_i}"
                output_key = f"output{ans_i}"
                score_key = f"score{ans_i}"
                if model_key not in rec and output_key not in rec and score_key not in rec:
                    continue
                model_row = {
                    "model": rec.get(model_key, ""),
                    "output": rec.get(output_key, ""),
                    "reason": rec.get(f"reason{ans_i}", ""),
                }
                if score_key in rec:
                    model_row["score"] = rec.get(score_key, None)
                models.append(model_row)
            if models:
                stats["records_flat_5answers_format"] += 1

            if not models:
                for ans_key in ("A", "B", "C", "D", "E"):
                    model_key = f"model{ans_key}"
                    output_key = f"output{ans_key}"
                    answer_key = f"answer{ans_key}"
                    score_key = f"score{ans_key}"
                    if (
                        model_key not in rec
                        and output_key not in rec
                        and answer_key not in rec
                        and score_key not in rec
                    ):
                        continue
                    model_row = {
                        "model": rec.get(model_key, ans_key),
                        "output": rec.get(output_key, rec.get(answer_key, "")),
                        "reason": rec.get(f"reason{ans_key}", ""),
                    }
                    if score_key in rec:
                        model_row["score"] = rec.get(score_key, None)
                    models.append(model_row)
                if models:
                    stats["records_flat_abc_format"] += 1

        answers: List[AnswerWithScore] = []
        for m in models:
            if not isinstance(m, dict):
                continue
            stats["answers_total"] += 1
            if "score" not in m:
                stats["answers_missing_score"] += 1
                continue

            score = _safe_int(m.get("score", -1), default=-1)
            output = str(m.get("output", ""))
            model_name = str(m.get("model", ""))

            if score < int(score_min) or score > int(score_max):
                stats["answers_score_out_of_range"] += 1
                continue
            if not output.strip():
                stats["answers_empty_output"] += 1
                continue

            answers.append(
                AnswerWithScore(
                    model=model_name,
                    output=output,
                    score=int(score),
                    reason=str(m.get("reason", "") or "").strip(),
                )
            )
            stats["answers_valid"] += 1

        if len(answers) < 2:
            stats["questions_skipped_lt2_answers"] += 1
            continue

        questions.append(
            {
                "question_id": int(qid),
                "source_id": int(source_id),
                "dataset": dataset,
                "instruction": instruction,
                "input_text": input_text,
                "answers": answers,
            }
        )
        stats["questions_with_ge2_answers"] += 1

    if not questions:
        raise RuntimeError("no valid questions (>=2 scored answers) were loaded")
    return questions, stats


def _pick_two_answers(
    *,
    answers: Sequence[AnswerWithScore],
    strategy: str,
    rng: np.random.Generator,
    randomize_order: bool,
) -> Tuple[AnswerWithScore, AnswerWithScore, Dict[str, Any]]:
    if len(answers) < 2:
        raise ValueError("need at least 2 answers")

    n = int(len(answers))
    a_idx = 0
    b_idx = 1

    if strategy == "first_two":
        a_idx, b_idx = 0, 1
    elif strategy == "random":
        picked = rng.choice(n, size=2, replace=False).tolist()
        a_idx, b_idx = int(picked[0]), int(picked[1])
    elif strategy == "max_gap":
        pairs: List[Tuple[int, int, int]] = []
        max_gap = -1
        for i in range(n):
            for j in range(i + 1, n):
                gap = abs(int(answers[i].score) - int(answers[j].score))
                if gap > max_gap:
                    max_gap = int(gap)
                    pairs = [(i, j, gap)]
                elif gap == max_gap:
                    pairs.append((i, j, gap))
        if not pairs:
            raise RuntimeError("failed to select pair with max_gap")
        picked_idx = int(rng.integers(low=0, high=len(pairs)))
        a_idx, b_idx, _ = pairs[picked_idx]
    else:
        raise ValueError(f"unknown pair-selection-strategy: {strategy}")

    a = answers[int(a_idx)]
    b = answers[int(b_idx)]

    if randomize_order and bool(rng.random() < 0.5):
        a, b = b, a

    meta = {
        "strategy": str(strategy),
        "selected_gap": int(abs(int(a.score) - int(b.score))),
    }
    return a, b, meta


def _select_question_pairs(
    questions: Sequence[Dict[str, Any]],
    *,
    strategy: str,
    randomize_order: bool,
    seed: int,
    budget_units: int,
) -> Tuple[List[SelectedQuestionPair], List[Dict[str, Any]], Dict[str, int]]:
    rng = np.random.default_rng(int(seed))
    selected: List[SelectedQuestionPair] = []
    rows: List[Dict[str, Any]] = []

    stats: Dict[str, int] = {
        "input_questions": int(len(questions)),
        "budget_units": int(budget_units),
        "selected_questions": 0,
        "selected_answers": 0,
        "selection_strategy_max_gap": 0,
        "selection_strategy_random": 0,
        "selection_strategy_first_two": 0,
    }

    max_questions = 0
    if int(budget_units) > 0:
        max_questions = int(budget_units) // 2
        if max_questions <= 0:
            raise ValueError("budget-units must be >= 2 when > 0")

    if strategy == "max_gap":
        stats["selection_strategy_max_gap"] = 1
    elif strategy == "random":
        stats["selection_strategy_random"] = 1
    elif strategy == "first_two":
        stats["selection_strategy_first_two"] = 1

    for q in questions:
        if int(max_questions) > 0 and len(selected) >= int(max_questions):
            break

        answers = list(q["answers"])
        a, b, meta = _pick_two_answers(
            answers=answers,
            strategy=str(strategy),
            rng=rng,
            randomize_order=bool(randomize_order),
        )

        pair = SelectedQuestionPair(
            question_id=int(q["question_id"]),
            source_id=int(q.get("source_id", q["question_id"])),
            dataset=str(q["dataset"]),
            instruction=str(q["instruction"]),
            input_text=str(q["input_text"]),
            answer_a=a,
            answer_b=b,
        )
        selected.append(pair)

        rows.append(
            {
                "question_id": int(pair.question_id),
                "source_id": int(pair.source_id),
                "dataset": str(pair.dataset),
                "model_a": str(pair.answer_a.model),
                "model_b": str(pair.answer_b.model),
                "score_a": int(pair.answer_a.score),
                "score_b": int(pair.answer_b.score),
                "score_gap": int(abs(int(pair.answer_a.score) - int(pair.answer_b.score))),
                "selection_strategy": str(meta["strategy"]),
            }
        )

    stats["selected_questions"] = int(len(selected))
    stats["selected_answers"] = int(len(selected) * 2)
    if not selected:
        raise RuntimeError("no question pairs were selected")

    return selected, rows, stats


def _pairwise_label_from_scores(score_a: int, score_b: int) -> int:
    if int(score_a) > int(score_b):
        return int(LABEL_A)
    if int(score_a) < int(score_b):
        return int(LABEL_B)
    return int(LABEL_TIE)


def _filter_questions_by_selected_pair_ids(
    questions: Sequence[Dict[str, Any]],
    selected_pairs: Sequence[SelectedQuestionPair],
) -> List[Dict[str, Any]]:
    if not selected_pairs:
        return []
    use_source_id = all(int(getattr(p, "source_id", -1)) > 0 for p in selected_pairs)
    allowed = {
        int(getattr(p, "source_id", -1)) if use_source_id else int(getattr(p, "question_id", -1))
        for p in selected_pairs
    }
    out: List[Dict[str, Any]] = []
    for q in questions:
        match_id = int(q.get("source_id", q.get("question_id", -1))) if use_source_id else int(q.get("question_id", -1))
        if match_id in allowed:
            out.append(q)
    return out


def _build_candidate_pair_examples(
    questions: Sequence[Dict[str, Any]],
    *,
    pairwise_system_prompt: str,
    randomize_order: bool,
    seed: int,
) -> Tuple[List[CandidatePairExample], List[Dict[str, Any]], Dict[str, int]]:
    rng = np.random.default_rng(int(seed))
    examples: List[CandidatePairExample] = []
    rows: List[Dict[str, Any]] = []
    stats: Dict[str, int] = {
        "input_questions": int(len(questions)),
        "candidate_pairs": 0,
        "questions_with_candidate_pairs": 0,
        "labels_hidden_until_query": 1,
    }

    pair_id = 0
    for q in questions:
        answers = list(q.get("answers", []))
        if len(answers) < 2:
            continue
        stats["questions_with_candidate_pairs"] += 1
        qid = int(q["question_id"])
        source_id = int(q.get("source_id", qid))
        group_id = int(source_id if source_id > 0 else qid)

        for i in range(len(answers)):
            for j in range(i + 1, len(answers)):
                a = answers[i]
                b = answers[j]
                if bool(randomize_order) and bool(rng.random() < 0.5):
                    a, b = b, a

                label = _pairwise_label_from_scores(int(a.score), int(b.score))
                prompt = build_pairwise_prompt(
                    system_prompt=pairwise_system_prompt,
                    instruction=str(q["instruction"]),
                    input_text=str(q["input_text"]),
                    assistant_1_output=str(a.output),
                    assistant_2_output=str(b.output),
                )
                pair = SelectedQuestionPair(
                    question_id=int(qid),
                    source_id=int(source_id),
                    dataset=str(q["dataset"]),
                    instruction=str(q["instruction"]),
                    input_text=str(q["input_text"]),
                    answer_a=a,
                    answer_b=b,
                )

                pair_id += 1
                gap = int(abs(int(a.score) - int(b.score)))
                examples.append(
                    CandidatePairExample(
                        id=int(pair_id),
                        group_id=int(group_id),
                        question_id=int(qid),
                        source_id=int(source_id),
                        dataset=str(q["dataset"]),
                        model_a=str(a.model),
                        model_b=str(b.model),
                        score_a=int(a.score),
                        score_b=int(b.score),
                        score_gap=int(gap),
                        prompt=prompt,
                        label=int(label),
                        selected_pair=pair,
                    )
                )
                rows.append(
                    {
                        "candidate_pair_id": int(pair_id),
                        "group_id": int(group_id),
                        "question_id": int(qid),
                        "source_id": int(source_id),
                        "dataset": str(q["dataset"]),
                        "model_a": str(a.model),
                        "model_b": str(b.model),
                        "label_hidden_until_query": True,
                    }
                )

    stats["candidate_pairs"] = int(len(examples))
    return examples, rows, stats


def _softmax_entropy(probs: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    p = np.clip(np.asarray(probs, dtype=np.float64), float(eps), 1.0)
    return -(p * np.log(p)).sum(axis=1).astype(np.float32)


def _safe_binary_targets(x: np.ndarray) -> np.ndarray:
    vals = np.asarray(x, dtype=np.float32)
    if vals.size == 0:
        return vals
    finite = np.isfinite(vals)
    if not bool(finite.any()):
        return np.full(vals.shape, 0.5, dtype=np.float32)
    lo = float(vals[finite].min())
    hi = float(vals[finite].max())
    if hi <= lo:
        out = np.full(vals.shape, 0.5, dtype=np.float32)
        out[~finite] = 0.5
        return out
    out = (vals - lo) / (hi - lo)
    out[~finite] = 0.5
    return out.astype(np.float32)


def _build_score_bin_counts_from_candidates(
    candidates: Sequence[CandidatePairExample],
    *,
    score_min: int,
    score_max: int,
) -> np.ndarray:
    num_bins = int(score_max - score_min + 1)
    counts = np.zeros((num_bins,), dtype=np.int64)
    for c in candidates:
        for score in (int(c.score_a), int(c.score_b)):
            idx = int(score) - int(score_min)
            if 0 <= idx < num_bins:
                counts[idx] += 1
    return counts


def _score_bin_bonus_for_candidates(
    candidates: Sequence[CandidatePairExample],
    *,
    queried_score_counts: Optional[np.ndarray],
    score_min: int,
    score_max: int,
) -> np.ndarray:
    if not candidates:
        return np.zeros((0,), dtype=np.float32)

    num_bins = int(score_max - score_min + 1)
    if queried_score_counts is None or int(np.asarray(queried_score_counts).size) != num_bins:
        counts = np.zeros((num_bins,), dtype=np.int64)
    else:
        counts = np.asarray(queried_score_counts, dtype=np.int64).copy()

    rarity = 1.0 / np.sqrt(counts.astype(np.float32) + 1.0)
    raw_bonus = []
    for c in candidates:
        idx_a = int(c.score_a) - int(score_min)
        idx_b = int(c.score_b) - int(score_min)
        ra = float(rarity[idx_a]) if 0 <= idx_a < num_bins else 0.0
        rb = float(rarity[idx_b]) if 0 <= idx_b < num_bins else 0.0
        raw_bonus.append(0.5 * (ra + rb))
    return _safe_binary_targets(np.asarray(raw_bonus, dtype=np.float32))


def _candidate_selector_targets(
    *,
    p_before: np.ndarray,
    p_after: np.ndarray,
    labels: Sequence[int],
    gaps: Sequence[int],
    max_gap: int,
    gap_weight: float,
    uncertainty_weight: float,
    kl_weight: float,
) -> np.ndarray:
    labels_arr = np.asarray([int(x) for x in labels], dtype=np.int64)
    idx = np.arange(labels_arr.shape[0])
    before_true_prob = np.asarray(p_before, dtype=np.float32)[idx, labels_arr]
    uncertainty = 1.0 - np.clip(before_true_prob, 0.0, 1.0)

    p0 = np.clip(np.asarray(p_before, dtype=np.float64), 1e-8, 1.0)
    p1 = np.clip(np.asarray(p_after, dtype=np.float64), 1e-8, 1.0)
    kl = np.sum(p1 * (np.log(p1) - np.log(p0)), axis=1).astype(np.float32)
    uncertainty_signal = _safe_binary_targets(uncertainty)
    kl_signal = _safe_binary_targets(kl)
    total_weight = float(uncertainty_weight) + float(kl_weight)
    if total_weight <= 0.0:
        signal = 0.5 * uncertainty_signal + 0.5 * kl_signal
    else:
        signal = (
            float(uncertainty_weight) * uncertainty_signal
            + float(kl_weight) * kl_signal
        ) / float(total_weight)

    if float(gap_weight) > 0.0:
        denom = max(1, int(max_gap))
        gap_signal = np.asarray([float(g) / float(denom) for g in gaps], dtype=np.float32)
        signal = signal + float(gap_weight) * gap_signal
        signal = _safe_binary_targets(signal)
    return np.clip(signal, 0.0, 1.0).astype(np.float32)


def _build_pointwise_examples_for_candidate_pairs(
    candidates: Sequence[CandidatePairExample],
    *,
    score_min: int,
    score_max: int,
    judge_system_prompt: str,
    fix_score_prefix_in_prompt: bool,
) -> Tuple[List[PointwiseScoredExample], List[Tuple[int, int]]]:
    out: List[PointwiseScoredExample] = []
    spans: List[Tuple[int, int]] = []
    row_id = 0

    for c in candidates:
        start = len(out)
        p = c.selected_pair
        for ans in [p.answer_a, p.answer_b]:
            prompt = build_judge_prompt(
                system_prompt=judge_system_prompt,
                instruction=str(p.instruction),
                input_text=str(p.input_text),
                candidate_output=str(ans.output),
                include_gold_score=False,
                fix_score_prefix=bool(fix_score_prefix_in_prompt),
            )
            row_id += 1
            out.append(
                PointwiseScoredExample(
                    row_id=int(row_id),
                    question_id=int(p.question_id),
                    source_id=int(p.source_id),
                    dataset=str(p.dataset),
                    instruction=str(p.instruction),
                    input_text=str(p.input_text),
                    model=str(ans.model),
                    output=str(ans.output),
                    score=int(score_min),
                    label=0,
                    prompt=prompt,
                )
            )
        spans.append((int(start), int(len(out))))
    return out, spans


def _candidate_selector_targets_pointwise(
    *,
    proxy: LlamaSharedMultiTaskProxyModel,
    candidates: Sequence[CandidatePairExample],
    score_min: int,
    score_max: int,
    judge_system_prompt: str,
    fix_score_prefix_in_prompt: bool,
    max_gap: int,
    gap_weight: float,
    queried_score_counts: Optional[np.ndarray],
    score_bin_weight: float,
    uncertainty_weight: float,
    kl_weight: float,
) -> np.ndarray:
    pointwise_inputs, spans = _build_pointwise_examples_for_candidate_pairs(
        candidates,
        score_min=int(score_min),
        score_max=int(score_max),
        judge_system_prompt=str(judge_system_prompt),
        fix_score_prefix_in_prompt=bool(fix_score_prefix_in_prompt),
    )
    if not pointwise_inputs:
        return np.zeros((0,), dtype=np.float32)

    pointwise_labels = [int(x.label) for x in pointwise_inputs]
    p_before = proxy.predict_proba_pointwise(pointwise_inputs)
    proxy.train_on_batch_pointwise(pointwise_inputs, pointwise_labels)
    p_after = proxy.predict_proba_pointwise(pointwise_inputs)

    labels_arr = np.asarray(pointwise_labels, dtype=np.int64)
    idx = np.arange(labels_arr.shape[0])
    before_true_prob = np.asarray(p_before, dtype=np.float32)[idx, labels_arr]
    uncertainty = 1.0 - np.clip(before_true_prob, 0.0, 1.0)

    p0 = np.clip(np.asarray(p_before, dtype=np.float64), 1e-8, 1.0)
    p1 = np.clip(np.asarray(p_after, dtype=np.float64), 1e-8, 1.0)
    kl = np.sum(p1 * (np.log(p1) - np.log(p0)), axis=1).astype(np.float32)
    uncertainty_signal = _safe_binary_targets(uncertainty)
    kl_signal = _safe_binary_targets(kl)
    total_weight = float(uncertainty_weight) + float(kl_weight)
    if total_weight <= 0.0:
        answer_signal = 0.5 * uncertainty_signal + 0.5 * kl_signal
    else:
        answer_signal = (
            float(uncertainty_weight) * uncertainty_signal
            + float(kl_weight) * kl_signal
        ) / float(total_weight)

    pair_signal = np.asarray(
        [float(answer_signal[int(start) : int(end)].mean()) for start, end in spans],
        dtype=np.float32,
    )

    if float(gap_weight) > 0.0:
        denom = max(1, int(max_gap))
        gap_signal = np.asarray([float(c.score_gap) / float(denom) for c in candidates], dtype=np.float32)
        pair_signal = pair_signal + float(gap_weight) * gap_signal
        pair_signal = _safe_binary_targets(pair_signal)
    if float(score_bin_weight) > 0.0:
        score_bin_bonus = _score_bin_bonus_for_candidates(
            candidates,
            queried_score_counts=queried_score_counts,
            score_min=int(score_min),
            score_max=int(score_max),
        )
        pair_signal = pair_signal + float(score_bin_weight) * score_bin_bonus
        pair_signal = _safe_binary_targets(pair_signal)
    return np.clip(pair_signal, 0.0, 1.0).astype(np.float32)


def _pick_candidate_pairs_greedy(
    candidates: Sequence[CandidatePairExample],
    *,
    k: int,
    rng: np.random.Generator,
    one_per_question: bool,
    scores: Optional[np.ndarray] = None,
    selected_group_ids: Optional[Set[int]] = None,
) -> List[CandidatePairExample]:
    if int(k) <= 0:
        return []
    banned = set(selected_group_ids or set())
    idxs = list(range(len(candidates)))
    if scores is None:
        rng.shuffle(idxs)
    else:
        score_arr = np.asarray(scores, dtype=np.float32)
        jitter = rng.random(len(candidates)).astype(np.float32) * 1e-7
        idxs = np.argsort(-(score_arr + jitter)).tolist()

    picked: List[CandidatePairExample] = []
    seen_groups = set(banned)
    for i in idxs:
        c = candidates[int(i)]
        gid = int(c.group_id)
        if bool(one_per_question) and gid in seen_groups:
            continue
        picked.append(c)
        seen_groups.add(gid)
        if len(picked) >= int(k):
            break
    return picked


def _counts_to_distribution(counts: np.ndarray) -> np.ndarray:
    arr = np.asarray(counts, dtype=np.float64)
    total = float(arr.sum())
    if total <= 0.0:
        if arr.size <= 0:
            return arr.astype(np.float64)
        return np.ones_like(arr, dtype=np.float64) / float(arr.size)
    return arr / total


def _l1_distribution_distance(counts: np.ndarray, target_dist: np.ndarray) -> float:
    arr = np.asarray(counts, dtype=np.float64)
    target = np.asarray(target_dist, dtype=np.float64)
    if arr.size != target.size:
        raise ValueError("distribution distance shape mismatch")
    return float(np.abs(_counts_to_distribution(arr) - target).sum())


def _candidate_true_score_vector(
    c: CandidatePairExample,
    *,
    score_min: int,
    score_max: int,
) -> np.ndarray:
    score_bins = int(score_max - score_min + 1)
    out = np.zeros((score_bins,), dtype=np.float64)
    for score in (int(c.score_a), int(c.score_b)):
        idx = int(score) - int(score_min)
        if 0 <= idx < score_bins:
            out[idx] += 1.0
    return out


def _candidate_true_gap_vector(c: CandidatePairExample, *, max_gap: int) -> np.ndarray:
    out = np.zeros((int(max_gap) + 1,), dtype=np.float64)
    idx = max(0, min(int(max_gap), int(c.score_gap)))
    out[int(idx)] = 1.0
    return out


def _scalar_to_soft_bin_vector(value: float, *, max_bin: int) -> np.ndarray:
    out = np.zeros((int(max_bin) + 1,), dtype=np.float64)
    v = max(0.0, min(float(max_bin), float(value)))
    lo = int(math.floor(v))
    hi = int(math.ceil(v))
    if lo == hi:
        out[lo] = 1.0
    else:
        out[lo] = float(hi) - v
        out[hi] = v - float(lo)
    return out


def _candidate_answer_keys(c: CandidatePairExample) -> Tuple[Tuple[int, str, str], Tuple[int, str, str]]:
    p = c.selected_pair
    return (
        (int(p.source_id), str(p.answer_a.model), str(p.answer_a.output)),
        (int(p.source_id), str(p.answer_b.model), str(p.answer_b.output)),
    )


def _build_unique_pointwise_examples_for_candidate_answers(
    candidates: Sequence[CandidatePairExample],
    *,
    score_min: int,
    score_max: int,
    judge_system_prompt: str,
    fix_score_prefix_in_prompt: bool,
) -> Tuple[List[PointwiseScoredExample], Dict[Tuple[int, str, str], int]]:
    examples: List[PointwiseScoredExample] = []
    key_to_idx: Dict[Tuple[int, str, str], int] = {}
    row_id = 0

    for c in candidates:
        p = c.selected_pair
        for ans in (p.answer_a, p.answer_b):
            key = (int(p.source_id), str(ans.model), str(ans.output))
            if key in key_to_idx:
                continue
            label = score_to_class(int(ans.score), score_min=int(score_min), score_max=int(score_max))
            prompt = build_judge_prompt(
                system_prompt=judge_system_prompt,
                instruction=str(p.instruction),
                input_text=str(p.input_text),
                candidate_output=str(ans.output),
                include_gold_score=False,
                fix_score_prefix=bool(fix_score_prefix_in_prompt),
            )
            row_id += 1
            key_to_idx[key] = len(examples)
            examples.append(
                PointwiseScoredExample(
                    row_id=int(row_id),
                    question_id=int(p.question_id),
                    source_id=int(p.source_id),
                    dataset=str(p.dataset),
                    instruction=str(p.instruction),
                    input_text=str(p.input_text),
                    model=str(ans.model),
                    output=str(ans.output),
                    score=int(ans.score),
                    label=int(label),
                    prompt=prompt,
                )
            )
    return examples, key_to_idx


def _candidate_distribution_rank_bonus_pointwise(
    *,
    proxy: LlamaSharedMultiTaskProxyModel,
    candidates: Sequence[CandidatePairExample],
    selected_score_counts: Optional[np.ndarray],
    score_min: int,
    score_max: int,
    judge_system_prompt: str,
    fix_score_prefix_in_prompt: bool,
) -> np.ndarray:
    candidate_list = list(candidates)
    if not candidate_list:
        return np.zeros((0,), dtype=np.float32)

    score_bins = int(score_max - score_min + 1)
    if selected_score_counts is None or int(np.asarray(selected_score_counts).size) != score_bins:
        base_counts = np.zeros((score_bins,), dtype=np.float64)
    else:
        base_counts = np.asarray(selected_score_counts, dtype=np.float64).copy()

    answer_examples, answer_key_to_idx = _build_unique_pointwise_examples_for_candidate_answers(
        candidate_list,
        score_min=int(score_min),
        score_max=int(score_max),
        judge_system_prompt=str(judge_system_prompt),
        fix_score_prefix_in_prompt=bool(fix_score_prefix_in_prompt),
    )
    if not answer_examples:
        return np.zeros((len(candidate_list),), dtype=np.float32)

    answer_probs = np.asarray(proxy.predict_proba_pointwise(answer_examples), dtype=np.float64)
    candidate_vectors: List[np.ndarray] = []
    remaining_counts = np.zeros_like(base_counts)
    for c in candidate_list:
        key_a, key_b = _candidate_answer_keys(c)
        probs_a = answer_probs[int(answer_key_to_idx[key_a])]
        probs_b = answer_probs[int(answer_key_to_idx[key_b])]
        score_vec = probs_a + probs_b
        candidate_vectors.append(score_vec)
        remaining_counts += score_vec

    target_dist = _counts_to_distribution(base_counts + remaining_counts)
    current_loss = _l1_distribution_distance(base_counts, target_dist)
    raw_bonus = np.asarray(
        [current_loss - _l1_distribution_distance(base_counts + vec, target_dist) for vec in candidate_vectors],
        dtype=np.float32,
    )
    return _safe_binary_targets(raw_bonus)


def _select_candidate_pairs_by_distribution(
    *,
    candidates: Sequence[CandidatePairExample],
    cfg: RunConfig,
    llama_path: str,
) -> Tuple[List[SelectedQuestionPair], List[Dict[str, Any]], Dict[str, Any]]:
    """Select pairs by matching the current proxy's predicted score distribution.

    This selector does not use true scores for unqueried candidates. It first queries a
    random warm-up batch, trains a lightweight pointwise proxy on those queried labels,
    uses the proxy's soft score probabilities as pseudo-labels for the remaining pool,
    and only adds true scores to the selected histogram after a candidate is queried.
    """
    candidate_list = list(candidates)
    if not candidate_list:
        raise RuntimeError("distribution selector received an empty candidate pool")

    max_pairs = int(cfg.budget_units) // 2 if int(cfg.budget_units) > 0 else len({int(c.group_id) for c in candidate_list})
    max_pairs = max(1, min(int(max_pairs), len(candidate_list)))
    one_per_question = bool(cfg.candidate_selector_one_per_question)
    rng = np.random.default_rng(int(cfg.seed) + 101)

    score_bins = int(cfg.score_max - cfg.score_min + 1)
    max_gap = int(cfg.score_max - cfg.score_min)
    datasets = sorted({str(c.dataset) for c in candidate_list})
    dataset_to_idx = {name: i for i, name in enumerate(datasets)}
    target_dataset_counts = np.zeros((len(dataset_to_idx),), dtype=np.float64)
    dataset_vectors: Dict[int, np.ndarray] = {}
    for c in candidate_list:
        dv = np.zeros((len(dataset_to_idx),), dtype=np.float64)
        dataset_idx = dataset_to_idx.get(str(c.dataset))
        if dataset_idx is not None:
            dv[int(dataset_idx)] += 1.0
        dataset_vectors[int(c.id)] = dv
        target_dataset_counts += dv
    target_dataset_dist = _counts_to_distribution(target_dataset_counts)

    score_weight = float(cfg.candidate_distribution_score_weight)
    dataset_weight = float(cfg.candidate_distribution_dataset_weight)
    gap_weight = float(cfg.candidate_distribution_gap_weight)
    if score_weight <= 0.0 and dataset_weight <= 0.0 and gap_weight <= 0.0:
        score_weight = 1.0

    selected: List[CandidatePairExample] = []
    rows: List[Dict[str, Any]] = []
    selected_group_ids: Set[int] = set()
    selected_candidate_ids: Set[int] = set()
    score_counts = np.zeros((score_bins,), dtype=np.float64)
    dataset_counts = np.zeros_like(target_dataset_counts)
    gap_counts = np.zeros((max_gap + 1,), dtype=np.float64)
    current_loss = float("inf")

    def _record(c: CandidatePairExample, *, stage: str, rank_score: float, dist_loss: float) -> None:
        rows.append(
            {
                "stage": str(stage),
                "candidate_pair_id": int(c.id),
                "group_id": int(c.group_id),
                "question_id": int(c.question_id),
                "source_id": int(c.source_id),
                "dataset": str(c.dataset),
                "model_a": str(c.model_a),
                "model_b": str(c.model_b),
                "queried": True,
                "score_a": int(c.score_a),
                "score_b": int(c.score_b),
                "score_gap": int(c.score_gap),
                "pairwise_label": int(c.label),
                "pairwise_token": str(label_to_token(int(c.label))),
                "rank_score": float(rank_score),
                "distribution_loss": float(dist_loss),
            }
        )

    init_k = min(int(cfg.candidate_selector_init_pairs), int(max_pairs), len(candidate_list))
    init_batch = _pick_candidate_pairs_greedy(
        candidate_list,
        k=max(0, init_k),
        rng=rng,
        one_per_question=one_per_question,
    )
    for c in init_batch:
        selected.append(c)
        selected_candidate_ids.add(int(c.id))
        selected_group_ids.add(int(c.group_id))
        score_counts += _candidate_true_score_vector(
            c,
            score_min=int(cfg.score_min),
            score_max=int(cfg.score_max),
        )
        dataset_counts += dataset_vectors[int(c.id)]
        gap_counts += _candidate_true_gap_vector(c, max_gap=max_gap)
        _record(c, stage="init", rank_score=float("nan"), dist_loss=float("nan"))

    proxy = LlamaSharedMultiTaskProxyModel(
        model_path=str(llama_path),
        pointwise_num_labels=int(cfg.score_max - cfg.score_min + 1),
        pairwise_num_labels=3,
        multitask_mode=str(cfg.llama_multitask_mode),
        lr=float(cfg.proxy_lr),
        weight_decay=0.0,
        max_length=int(cfg.proxy_max_length),
        finetune_mode="lora",
        gradient_checkpointing=True,
        use_amp=False,
        load_in_4bit=bool(cfg.load_in_4bit),
        score_min=int(cfg.score_min),
        score_max=int(cfg.score_max),
        fix_score_prefix_in_prompt=bool(cfg.fix_score_prefix_in_prompt),
        pointwise_loss_type=str(cfg.pointwise_loss_type),
        pointwise_distance_weight=float(cfg.pointwise_distance_weight),
    )
    def _train_proxy_on_queried(batch: Sequence[CandidatePairExample]) -> None:
        if not batch:
            return
        pointwise_batch, _ = _build_pointwise_examples_for_candidate_pairs(
            batch,
            score_min=int(cfg.score_min),
            score_max=int(cfg.score_max),
            judge_system_prompt=JUDGE_SYSTEM_PROMPT_SCORE_ONLY,
            fix_score_prefix_in_prompt=bool(cfg.fix_score_prefix_in_prompt),
        )
        labels = [int(x.label) for x in pointwise_batch]
        for _ in range(max(1, int(cfg.candidate_selector_epochs))):
            proxy.train_on_batch_pointwise(pointwise_batch, labels)

    _train_proxy_on_queried(init_batch)

    answer_examples, answer_key_to_idx = _build_unique_pointwise_examples_for_candidate_answers(
        candidate_list,
        score_min=int(cfg.score_min),
        score_max=int(cfg.score_max),
        judge_system_prompt=JUDGE_SYSTEM_PROMPT_SCORE_ONLY,
        fix_score_prefix_in_prompt=bool(cfg.fix_score_prefix_in_prompt),
    )

    score_values = np.arange(int(cfg.score_min), int(cfg.score_max) + 1, dtype=np.float64)

    def _predict_remaining_vectors() -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray], np.ndarray, np.ndarray]:
        answer_probs = np.asarray(proxy.predict_proba_pointwise(answer_examples), dtype=np.float64)
        candidate_score_vectors: Dict[int, np.ndarray] = {}
        candidate_gap_vectors: Dict[int, np.ndarray] = {}
        remaining_score_counts = np.zeros_like(score_counts)
        remaining_gap_counts = np.zeros_like(gap_counts)
        for c in candidate_list:
            if int(c.id) in selected_candidate_ids:
                continue
            if one_per_question and int(c.group_id) in selected_group_ids:
                continue
            key_a, key_b = _candidate_answer_keys(c)
            probs_a = answer_probs[int(answer_key_to_idx[key_a])]
            probs_b = answer_probs[int(answer_key_to_idx[key_b])]
            score_vec = probs_a + probs_b
            expected_gap = abs(float(np.dot(probs_a, score_values)) - float(np.dot(probs_b, score_values)))
            gap_vec = _scalar_to_soft_bin_vector(expected_gap, max_bin=max_gap)
            candidate_score_vectors[int(c.id)] = score_vec
            candidate_gap_vectors[int(c.id)] = gap_vec
            remaining_score_counts += score_vec
            remaining_gap_counts += gap_vec
        target_score_dist = _counts_to_distribution(score_counts + remaining_score_counts)
        target_gap_dist = _counts_to_distribution(gap_counts + remaining_gap_counts)
        return candidate_score_vectors, candidate_gap_vectors, target_score_dist, target_gap_dist

    def _loss(sc: np.ndarray, dc: np.ndarray, gc: np.ndarray, score_dist: np.ndarray, gap_dist: np.ndarray) -> float:
        loss = 0.0
        if score_weight > 0.0:
            loss += score_weight * _l1_distribution_distance(sc, score_dist)
        if dataset_weight > 0.0 and dc.size > 0:
            loss += dataset_weight * _l1_distribution_distance(dc, target_dataset_dist)
        if gap_weight > 0.0:
            loss += gap_weight * _l1_distribution_distance(gc, gap_dist)
        return float(loss)

    candidate_score_vectors, candidate_gap_vectors, target_score_dist, target_gap_dist = _predict_remaining_vectors()
    current_loss = _loss(score_counts, dataset_counts, gap_counts, target_score_dist, target_gap_dist)
    t0 = time.time()
    while len(selected) < int(max_pairs):
        batch_new: List[CandidatePairExample] = []
        batch_k = min(max(1, int(cfg.candidate_selector_batch_size)), int(max_pairs) - len(selected))

        for _ in range(int(batch_k)):
            best: Optional[CandidatePairExample] = None
            best_prequery_loss = float("inf")
            best_score = -float("inf")
            best_jitter = -float("inf")

            for c in candidate_list:
                if int(c.id) in selected_candidate_ids:
                    continue
                if one_per_question and int(c.group_id) in selected_group_ids:
                    continue
                score_vec = candidate_score_vectors.get(int(c.id))
                gap_vec = candidate_gap_vectors.get(int(c.id))
                if score_vec is None or gap_vec is None:
                    continue
                cand_loss = _loss(
                    score_counts + score_vec,
                    dataset_counts + dataset_vectors[int(c.id)],
                    gap_counts + gap_vec,
                    target_score_dist,
                    target_gap_dist,
                )
                cand_score = -float(cand_loss)
                jitter = float(rng.random() * 1e-9)
                if cand_score + jitter > best_score + best_jitter:
                    best = c
                    best_prequery_loss = float(cand_loss)
                    best_score = cand_score
                    best_jitter = jitter

            if best is None:
                break

            selected.append(best)
            selected_candidate_ids.add(int(best.id))
            selected_group_ids.add(int(best.group_id))
            # After selection, the real label is queried and can be used from now on.
            score_counts += _candidate_true_score_vector(
                best,
                score_min=int(cfg.score_min),
                score_max=int(cfg.score_max),
            )
            dataset_counts += dataset_vectors[int(best.id)]
            gap_counts += _candidate_true_gap_vector(best, max_gap=max_gap)
            current_loss = _loss(score_counts, dataset_counts, gap_counts, target_score_dist, target_gap_dist)
            _record(best, stage="proxy_distribution", rank_score=float(best_score), dist_loss=float(current_loss))
            batch_new.append(best)

            if len(selected) % 100 == 0 or len(selected) == int(max_pairs):
                print(
                    f"[proxy-distribution-selector] selected={len(selected)}/{max_pairs} "
                    f"loss={current_loss:.4f} prequery_loss={best_prequery_loss:.4f} "
                    f"elapsed={time.time() - t0:.0f}s",
                    flush=True,
                )

        if not batch_new:
            break
        _train_proxy_on_queried(batch_new)
        candidate_score_vectors, candidate_gap_vectors, target_score_dist, target_gap_dist = _predict_remaining_vectors()
        current_loss = _loss(score_counts, dataset_counts, gap_counts, target_score_dist, target_gap_dist)

    info = {
        "mode": "candidate_pair_selector",
        "selector_kind": "distribution",
        "label_usage": "proxy_prequery_scores_then_true_labels_after_query",
        "candidate_pairs": int(len(candidate_list)),
        "selected_pairs": int(len(selected)),
        "selected_answers": int(len(selected) * 2),
        "one_per_question": bool(one_per_question),
        "budget_units": int(cfg.budget_units),
        "effective_budget_units": int(len(selected) * 2),
        "init_pairs": int(len(init_batch)),
        "proxy_score_refresh": "after_each_selected_batch",
        "score_weight": float(score_weight),
        "dataset_weight": float(dataset_weight),
        "gap_weight": float(gap_weight),
        "score_bins": [int(x) for x in range(int(cfg.score_min), int(cfg.score_max) + 1)],
        "target_proxy_score_distribution": [float(x) for x in target_score_dist.tolist()],
        "selected_true_score_distribution": [float(x) for x in _counts_to_distribution(score_counts).tolist()],
        "selected_true_score_counts": [int(x) for x in score_counts.astype(np.int64).tolist()],
        "gap_bins": [int(x) for x in range(0, int(max_gap) + 1)],
        "target_proxy_gap_distribution": [float(x) for x in target_gap_dist.tolist()],
        "selected_true_gap_distribution": [float(x) for x in _counts_to_distribution(gap_counts).tolist()],
        "selected_true_gap_counts": [int(x) for x in gap_counts.astype(np.int64).tolist()],
        "datasets": list(datasets),
        "target_dataset_distribution": [float(x) for x in target_dataset_dist.tolist()],
        "selected_dataset_distribution": [float(x) for x in _counts_to_distribution(dataset_counts).tolist()],
        "final_distribution_loss": float(current_loss),
        "elapsed_sec": float(time.time() - t0),
    }

    del proxy
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return [c.selected_pair for c in selected], rows, info


def _select_candidate_pairs_with_selector(
    *,
    candidates: Sequence[CandidatePairExample],
    cfg: RunConfig,
    llama_path: str,
    output_dir: Path,
) -> Tuple[List[SelectedQuestionPair], List[Dict[str, Any]], Dict[str, Any]]:
    candidate_list = list(candidates)
    if not candidate_list:
        raise RuntimeError("candidate_pair_selector received an empty candidate pool")

    max_pairs = int(cfg.budget_units) // 2 if int(cfg.budget_units) > 0 else len({int(c.group_id) for c in candidate_list})
    max_pairs = max(1, min(int(max_pairs), len(candidate_list)))
    rng = np.random.default_rng(int(cfg.seed) + 101)
    one_per_question = bool(cfg.candidate_selector_one_per_question)

    selected_group_ids: Set[int] = set()
    selected_candidate_ids: Set[int] = set()
    selected: List[CandidatePairExample] = []
    rows: List[Dict[str, Any]] = []
    queried_score_counts: Optional[np.ndarray] = None

    def _record(
        c: CandidatePairExample,
        *,
        stage: str,
        rank_score: float,
        selector_rank_score: Optional[float] = None,
        distribution_rank_bonus: Optional[float] = None,
    ) -> None:
        rows.append(
            {
                "stage": str(stage),
                "candidate_pair_id": int(c.id),
                "group_id": int(c.group_id),
                "question_id": int(c.question_id),
                "source_id": int(c.source_id),
                "dataset": str(c.dataset),
                "model_a": str(c.model_a),
                "model_b": str(c.model_b),
                "queried": True,
                "score_a": int(c.score_a),
                "score_b": int(c.score_b),
                "score_gap": int(c.score_gap),
                "pairwise_label": int(c.label),
                "pairwise_token": str(label_to_token(int(c.label))),
                "rank_score": float(rank_score),
                "selector_rank_score": None
                if selector_rank_score is None
                else float(selector_rank_score),
                "distribution_rank_bonus": None
                if distribution_rank_bonus is None
                else float(distribution_rank_bonus),
            }
        )

    if str(cfg.candidate_selector_kind) == "random":
        picked = _pick_candidate_pairs_greedy(
            candidate_list,
            k=int(max_pairs),
            rng=rng,
            one_per_question=one_per_question,
        )
        for c in picked:
            selected.append(c)
            selected_candidate_ids.add(int(c.id))
            selected_group_ids.add(int(c.group_id))
            _record(c, stage="random", rank_score=float("nan"))
        info = {
            "mode": "candidate_pair_selector",
            "selector_kind": "random",
            "candidate_pairs": int(len(candidate_list)),
            "selected_pairs": int(len(selected)),
            "selected_answers": int(len(selected) * 2),
            "one_per_question": bool(one_per_question),
            "budget_units": int(cfg.budget_units),
            "effective_budget_units": int(len(selected) * 2),
        }
        return [c.selected_pair for c in selected], rows, info

    if str(cfg.candidate_selector_kind) not in {"shared_llama", "bert"}:
        raise ValueError(f"unknown candidate-selector-kind: {cfg.candidate_selector_kind}")
    if str(cfg.candidate_selector_target_task) not in {"pairwise", "pointwise"}:
        raise ValueError(f"unknown candidate-selector-target-task: {cfg.candidate_selector_target_task}")

    if str(cfg.candidate_selector_target_task) == "pointwise":
        proxy = LlamaSharedMultiTaskProxyModel(
            model_path=str(llama_path),
            pointwise_num_labels=int(cfg.score_max - cfg.score_min + 1),
            pairwise_num_labels=3,
            multitask_mode=str(cfg.llama_multitask_mode),
            lr=float(cfg.proxy_lr),
            weight_decay=0.0,
            max_length=int(cfg.proxy_max_length),
            finetune_mode="lora",
            gradient_checkpointing=True,
            use_amp=False,
            load_in_4bit=bool(cfg.load_in_4bit),
            score_min=int(cfg.score_min),
            score_max=int(cfg.score_max),
            fix_score_prefix_in_prompt=bool(cfg.fix_score_prefix_in_prompt),
            pointwise_loss_type=str(cfg.pointwise_loss_type),
            pointwise_distance_weight=float(cfg.pointwise_distance_weight),
        )
    else:
        proxy = LlamaSharedProxyModel(
            model_path=str(llama_path),
            num_labels=3,
            lr=float(cfg.proxy_lr),
            weight_decay=0.0,
            max_length=int(cfg.proxy_max_length),
            predict_mode="classifier",
            finetune_mode="lora",
            gradient_checkpointing=True,
            use_amp=False,
            load_in_4bit=bool(cfg.load_in_4bit),
            score_min=0,
            score_max=2,
            fix_score_prefix_in_prompt=False,
        )
    if str(cfg.candidate_selector_kind) == "shared_llama":
        selector = SharedLlamaSelectorV2(
            proxy_model=proxy,
            head_hidden_dim=512,
            lr=1e-3,
            weight_decay=0.0,
            batch_size=max(1, int(cfg.candidate_selector_batch_size)),
            buffer_maxlen=int(cfg.candidate_selector_buffer_maxlen),
        )
    else:
        selector = BertBinarySelector(
            model_name=str(cfg.candidate_bert_selector_model),
            max_length=int(cfg.candidate_bert_selector_max_length),
            head_hidden_dim=512,
            head_dropout=0.1,
            lr=1e-3,
            weight_decay=0.0,
            freeze_bert=bool(cfg.candidate_bert_selector_freeze),
            unfreeze_last_n_layers=int(cfg.candidate_bert_selector_unfreeze_last_n_layers),
        )

    max_gap = max(1, int(cfg.score_max - cfg.score_min))
    init_k = min(int(cfg.candidate_selector_init_pairs), int(max_pairs), len(candidate_list))
    init_batch = _pick_candidate_pairs_greedy(
        candidate_list,
        k=max(1, init_k),
        rng=rng,
        one_per_question=one_per_question,
    )
    if not init_batch:
        raise RuntimeError("candidate_pair_selector failed to build an initial batch")

    init_labels = [int(c.label) for c in init_batch]
    if str(cfg.candidate_selector_target_task) == "pointwise":
        targets = _candidate_selector_targets_pointwise(
            proxy=proxy,
            candidates=init_batch,
            score_min=int(cfg.score_min),
            score_max=int(cfg.score_max),
            judge_system_prompt=JUDGE_SYSTEM_PROMPT_SCORE_ONLY,
            fix_score_prefix_in_prompt=bool(cfg.fix_score_prefix_in_prompt),
            max_gap=max_gap,
            gap_weight=float(cfg.candidate_selector_gap_weight),
            queried_score_counts=queried_score_counts,
            score_bin_weight=float(cfg.candidate_selector_score_bin_weight),
            uncertainty_weight=float(cfg.candidate_selector_uncertainty_weight),
            kl_weight=float(cfg.candidate_selector_kl_weight),
        )
    else:
        p_before = proxy.predict_proba(init_batch)
        proxy.train_on_batch(init_batch, init_labels)
        p_after = proxy.predict_proba(init_batch)
        targets = _candidate_selector_targets(
            p_before=p_before,
            p_after=p_after,
            labels=init_labels,
            gaps=[int(c.score_gap) for c in init_batch],
            max_gap=max_gap,
            gap_weight=float(cfg.candidate_selector_gap_weight),
            uncertainty_weight=float(cfg.candidate_selector_uncertainty_weight),
            kl_weight=float(cfg.candidate_selector_kl_weight),
        )
    selector.update(
        init_batch,
        targets,
        epochs=max(1, int(cfg.candidate_selector_epochs)),
        batch_size=max(1, int(cfg.candidate_selector_batch_size)),
    )
    for c in init_batch:
        selected.append(c)
        selected_candidate_ids.add(int(c.id))
        selected_group_ids.add(int(c.group_id))
        _record(c, stage="init", rank_score=float("nan"))
    if str(cfg.candidate_selector_target_task) == "pointwise":
        queried_score_counts = _build_score_bin_counts_from_candidates(
            selected,
            score_min=int(cfg.score_min),
            score_max=int(cfg.score_max),
        )

    round_idx = 0
    t0 = time.time()
    while len(selected) < int(max_pairs):
        round_idx += 1
        remaining = [
            c
            for c in candidate_list
            if (not one_per_question or int(c.group_id) not in selected_group_ids)
            and int(c.id) not in selected_candidate_ids
        ]
        if not remaining:
            break

        pool = remaining
        max_score_candidates = int(cfg.candidate_selector_max_score_candidates)
        if max_score_candidates > 0 and len(pool) > max_score_candidates:
            idx = rng.choice(len(pool), size=max_score_candidates, replace=False).tolist()
            pool = [pool[int(i)] for i in idx]

        selector_scores = selector.score(pool)
        rank_scores = selector_scores.astype(np.float32)
        distribution_rank_bonus: Optional[np.ndarray] = None
        distribution_rank_weight = float(cfg.candidate_selector_distribution_rank_weight)
        if distribution_rank_weight > 0.0:
            distribution_rank_bonus = np.zeros_like(rank_scores, dtype=np.float32)
            top_k = int(cfg.candidate_selector_distribution_rank_top_k)
            if top_k <= 0 or len(pool) <= top_k:
                dist_indices = list(range(len(pool)))
            else:
                dist_indices = np.argsort(-selector_scores.astype(np.float32))[:top_k].tolist()
            dist_pool = [pool[int(i)] for i in dist_indices]
            dist_bonus = _candidate_distribution_rank_bonus_pointwise(
                proxy=proxy,
                candidates=dist_pool,
                selected_score_counts=queried_score_counts,
                score_min=int(cfg.score_min),
                score_max=int(cfg.score_max),
                judge_system_prompt=JUDGE_SYSTEM_PROMPT_SCORE_ONLY,
                fix_score_prefix_in_prompt=bool(cfg.fix_score_prefix_in_prompt),
            )
            for i, bonus in zip(dist_indices, dist_bonus.tolist()):
                distribution_rank_bonus[int(i)] = float(bonus)
            rank_scores = rank_scores + distribution_rank_weight * distribution_rank_bonus.astype(np.float32)

        need = min(int(cfg.candidate_selector_batch_size), int(max_pairs) - len(selected))
        picked = _pick_candidate_pairs_greedy(
            pool,
            k=int(need),
            rng=rng,
            one_per_question=one_per_question,
            scores=rank_scores,
            selected_group_ids=selected_group_ids,
        )
        if not picked:
            break

        score_by_id = {int(c.id): float(s) for c, s in zip(pool, rank_scores.tolist())}
        selector_score_by_id = {int(c.id): float(s) for c, s in zip(pool, selector_scores.tolist())}
        distribution_bonus_by_id = (
            {int(c.id): float(s) for c, s in zip(pool, distribution_rank_bonus.tolist())}
            if distribution_rank_bonus is not None
            else {}
        )
        labels = [int(c.label) for c in picked]
        if str(cfg.candidate_selector_target_task) == "pointwise":
            targets = _candidate_selector_targets_pointwise(
                proxy=proxy,
                candidates=picked,
                score_min=int(cfg.score_min),
                score_max=int(cfg.score_max),
                judge_system_prompt=JUDGE_SYSTEM_PROMPT_SCORE_ONLY,
                fix_score_prefix_in_prompt=bool(cfg.fix_score_prefix_in_prompt),
                max_gap=max_gap,
                gap_weight=float(cfg.candidate_selector_gap_weight),
                queried_score_counts=queried_score_counts,
                score_bin_weight=float(cfg.candidate_selector_score_bin_weight),
                uncertainty_weight=float(cfg.candidate_selector_uncertainty_weight),
                kl_weight=float(cfg.candidate_selector_kl_weight),
            )
        else:
            p_before = proxy.predict_proba(picked)
            proxy.train_on_batch(picked, labels)
            p_after = proxy.predict_proba(picked)
            targets = _candidate_selector_targets(
                p_before=p_before,
                p_after=p_after,
                labels=labels,
                gaps=[int(c.score_gap) for c in picked],
                max_gap=max_gap,
                gap_weight=float(cfg.candidate_selector_gap_weight),
                uncertainty_weight=float(cfg.candidate_selector_uncertainty_weight),
                kl_weight=float(cfg.candidate_selector_kl_weight),
            )
        selector.update(
            picked,
            targets,
            epochs=max(1, int(cfg.candidate_selector_epochs)),
            batch_size=max(1, int(cfg.candidate_selector_batch_size)),
        )

        for c in picked:
            selected.append(c)
            selected_candidate_ids.add(int(c.id))
            selected_group_ids.add(int(c.group_id))
            _record(
                c,
                stage=f"round_{round_idx}",
                rank_score=float(score_by_id.get(int(c.id), float("nan"))),
                selector_rank_score=selector_score_by_id.get(int(c.id)),
                distribution_rank_bonus=distribution_bonus_by_id.get(int(c.id)),
            )
        if str(cfg.candidate_selector_target_task) == "pointwise":
            queried_score_counts = _build_score_bin_counts_from_candidates(
                selected,
                score_min=int(cfg.score_min),
                score_max=int(cfg.score_max),
            )

        print(
            f"[candidate-selector] round={round_idx} selected={len(selected)}/{max_pairs} "
            f"pool={len(remaining)} elapsed={time.time() - t0:.0f}s",
            flush=True,
        )

    info = {
        "mode": "candidate_pair_selector",
        "selector_kind": str(cfg.candidate_selector_kind),
        "candidate_pairs": int(len(candidate_list)),
        "selected_pairs": int(len(selected)),
        "selected_answers": int(len(selected) * 2),
        "one_per_question": bool(one_per_question),
        "budget_units": int(cfg.budget_units),
        "effective_budget_units": int(len(selected) * 2),
        "init_pairs": int(len(init_batch)),
        "batch_size": int(cfg.candidate_selector_batch_size),
        "selector_epochs": int(cfg.candidate_selector_epochs),
        "buffer_maxlen": int(cfg.candidate_selector_buffer_maxlen),
        "max_score_candidates": int(cfg.candidate_selector_max_score_candidates),
        "gap_weight": float(cfg.candidate_selector_gap_weight),
        "score_bin_weight": float(cfg.candidate_selector_score_bin_weight),
        "distribution_rank_weight": float(cfg.candidate_selector_distribution_rank_weight),
        "distribution_rank_top_k": int(cfg.candidate_selector_distribution_rank_top_k),
        "uncertainty_weight": float(cfg.candidate_selector_uncertainty_weight),
        "kl_weight": float(cfg.candidate_selector_kl_weight),
        "target_task": str(cfg.candidate_selector_target_task),
        "bert_selector_model": str(cfg.candidate_bert_selector_model)
        if str(cfg.candidate_selector_kind) == "bert"
        else None,
        "bert_selector_max_length": int(cfg.candidate_bert_selector_max_length)
        if str(cfg.candidate_selector_kind) == "bert"
        else None,
        "bert_selector_freeze": bool(cfg.candidate_bert_selector_freeze)
        if str(cfg.candidate_selector_kind) == "bert"
        else None,
        "elapsed_sec": float(time.time() - t0),
    }

    del selector
    del proxy
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return [c.selected_pair for c in selected], rows, info


def _split_selected_pairs(
    selected_pairs: Sequence[SelectedQuestionPair],
    *,
    seed: int,
    val_ratio: float,
) -> Tuple[List[SelectedQuestionPair], List[SelectedQuestionPair], Dict[str, Any]]:
    all_pairs = list(selected_pairs)
    n = int(len(all_pairs))
    if n <= 1 or float(val_ratio) <= 0.0:
        return all_pairs, [], {
            "split_mode": "all_train",
            "val_ratio": float(val_ratio),
            "train_questions": int(n),
            "val_questions": 0,
        }

    k = int(round(float(val_ratio) * n))
    k = max(1, min(n - 1, k))

    rng = np.random.default_rng(int(seed))
    picked = rng.choice(n, size=k, replace=False).tolist()
    val_idx = {int(i) for i in picked}

    train = [x for i, x in enumerate(all_pairs) if i not in val_idx]
    val = [x for i, x in enumerate(all_pairs) if i in val_idx]

    info = {
        "split_mode": "random_by_question",
        "val_ratio": float(val_ratio),
        "train_questions": int(len(train)),
        "val_questions": int(len(val)),
    }
    return train, val, info


def _build_pointwise_examples(
    selected_pairs: Sequence[SelectedQuestionPair],
    *,
    score_min: int,
    score_max: int,
    judge_system_prompt: str,
    fix_score_prefix_in_prompt: bool,
) -> List[PointwiseScoredExample]:
    out: List[PointwiseScoredExample] = []
    row_id = 0

    for p in selected_pairs:
        for ans in [p.answer_a, p.answer_b]:
            label = score_to_class(int(ans.score), score_min=int(score_min), score_max=int(score_max))
            prompt = build_judge_prompt(
                system_prompt=judge_system_prompt,
                instruction=str(p.instruction),
                input_text=str(p.input_text),
                candidate_output=str(ans.output),
                include_gold_score=False,
                fix_score_prefix=bool(fix_score_prefix_in_prompt),
            )
            row_id += 1
            out.append(
                PointwiseScoredExample(
                    row_id=int(row_id),
                    question_id=int(p.question_id),
                    source_id=int(p.source_id),
                    dataset=str(p.dataset),
                    instruction=str(p.instruction),
                    input_text=str(p.input_text),
                    model=str(ans.model),
                    output=str(ans.output),
                    score=int(ans.score),
                    label=int(label),
                    prompt=prompt,
                )
            )

    return out


def _build_single_answer_pointwise_eval_examples(
    questions: Sequence[Dict[str, Any]],
    *,
    seed: int,
    score_min: int,
    score_max: int,
    judge_system_prompt: str,
    fix_score_prefix_in_prompt: bool,
) -> Tuple[List[PointwiseScoredExample], List[Dict[str, Any]], Dict[str, int]]:
    rng = np.random.default_rng(int(seed))
    out: List[PointwiseScoredExample] = []
    rows: List[Dict[str, Any]] = []
    row_id = 0

    stats: Dict[str, int] = {
        "input_questions": int(len(questions)),
        "questions_with_answers": 0,
        "skipped_questions_no_answers": 0,
        "selected_answers": 0,
    }

    for q in questions:
        answers = list(q.get("answers", []))
        if not answers:
            stats["skipped_questions_no_answers"] += 1
            continue

        stats["questions_with_answers"] += 1
        ans = answers[int(rng.integers(low=0, high=len(answers)))]
        label = score_to_class(int(ans.score), score_min=int(score_min), score_max=int(score_max))
        prompt = build_judge_prompt(
            system_prompt=judge_system_prompt,
            instruction=str(q["instruction"]),
            input_text=str(q["input_text"]),
            candidate_output=str(ans.output),
            include_gold_score=False,
            fix_score_prefix=bool(fix_score_prefix_in_prompt),
        )

        row_id += 1
        out.append(
            PointwiseScoredExample(
                row_id=int(row_id),
                question_id=int(q["question_id"]),
                source_id=int(q.get("source_id", q["question_id"])),
                dataset=str(q["dataset"]),
                instruction=str(q["instruction"]),
                input_text=str(q["input_text"]),
                model=str(ans.model),
                output=str(ans.output),
                score=int(ans.score),
                label=int(label),
                prompt=prompt,
            )
        )
        rows.append(
            {
                "row_id": int(row_id),
                "question_id": int(q["question_id"]),
                "source_id": int(q.get("source_id", q["question_id"])),
                "dataset": str(q["dataset"]),
                "model": str(ans.model),
                "score": int(ans.score),
            }
        )
        stats["selected_answers"] += 1

    return out, rows, stats


def _build_pairwise_examples(
    selected_pairs: Sequence[SelectedQuestionPair],
    *,
    pairwise_system_prompt: str,
    drop_tie: bool,
    order_augmentation: bool = False,
) -> Tuple[List[PairwiseExample], List[Dict[str, Any]], Dict[str, int]]:
    out_examples: List[PairwiseExample] = []
    out_rows: List[Dict[str, Any]] = []

    stats: Dict[str, int] = {
        "input_question_pairs": int(len(selected_pairs)),
        "generated_pairs": 0,
        "order_augmented_pairs": 0,
        "dropped_tie_pairs": 0,
        "label_A": 0,
        "label_B": 0,
        "label_C": 0,
    }

    pair_id = 0

    def _append_pairwise_example(
        *,
        p: SelectedQuestionPair,
        model_a: str,
        model_b: str,
        output_a: str,
        output_b: str,
        score_a_row: int,
        score_b_row: int,
        label: int,
        is_order_augmented: bool,
    ) -> None:
        nonlocal pair_id
        pair_id += 1
        prompt = build_pairwise_prompt(
            system_prompt=pairwise_system_prompt,
            instruction=str(p.instruction),
            input_text=str(p.input_text),
            assistant_1_output=str(output_a),
            assistant_2_output=str(output_b),
        )
        token = label_to_token(int(label))
        if int(label) == int(LABEL_A):
            stats["label_A"] += 1
        elif int(label) == int(LABEL_B):
            stats["label_B"] += 1
        else:
            stats["label_C"] += 1

        if bool(is_order_augmented):
            stats["order_augmented_pairs"] += 1

        out_examples.append(
            PairwiseExample(
                id=int(pair_id),
                dataset=str(p.dataset),
                group_id=int(p.question_id),
                pair_id=int(pair_id),
                model_a=str(model_a),
                model_b=str(model_b),
                prompt=prompt,
                label=int(label),
            )
        )
        out_rows.append(
            {
                "pair_id": int(pair_id),
                "group_id": int(p.question_id),
                "dataset": str(p.dataset),
                "question_id": int(p.question_id),
                "model_a": str(model_a),
                "model_b": str(model_b),
                "score_a": int(score_a_row),
                "score_b": int(score_b_row),
                "pairwise_label": int(label),
                "pairwise_token": str(token),
                "order_augmented": bool(is_order_augmented),
            }
        )

    for p in selected_pairs:
        score_a = int(p.answer_a.score)
        score_b = int(p.answer_b.score)
        if score_a > score_b:
            label = int(LABEL_A)
        elif score_a < score_b:
            label = int(LABEL_B)
        else:
            label = int(LABEL_TIE)

        if bool(drop_tie) and int(label) == int(LABEL_TIE):
            stats["dropped_tie_pairs"] += 1
            continue

        _append_pairwise_example(
            p=p,
            model_a=str(p.answer_a.model),
            model_b=str(p.answer_b.model),
            output_a=str(p.answer_a.output),
            output_b=str(p.answer_b.output),
            score_a_row=int(score_a),
            score_b_row=int(score_b),
            label=int(label),
            is_order_augmented=False,
        )

        if bool(order_augmentation):
            if int(label) == int(LABEL_A):
                swapped_label = int(LABEL_B)
            elif int(label) == int(LABEL_B):
                swapped_label = int(LABEL_A)
            else:
                swapped_label = int(LABEL_TIE)

            _append_pairwise_example(
                p=p,
                model_a=str(p.answer_b.model),
                model_b=str(p.answer_a.model),
                output_a=str(p.answer_b.output),
                output_b=str(p.answer_a.output),
                score_a_row=int(score_b),
                score_b_row=int(score_a),
                label=int(swapped_label),
                is_order_augmented=True,
            )

    stats["generated_pairs"] = int(len(out_examples))
    return out_examples, out_rows, stats


def _abc_choice_to_pairwise_label(choice: Any) -> int:
    choice_s = str(choice or "").strip()
    if choice_s in {"[[1]]", "[1]", "1", "A", "a"}:
        return int(LABEL_A)
    if choice_s in {"[[2]]", "[2]", "2", "B", "b"}:
        return int(LABEL_B)
    if choice_s in {"[[3]]", "[3]", "3", "C", "c"}:
        return int(LABEL_TIE)
    if choice_s.lower() in {"tie", "t", "equal", "same"}:
        return int(LABEL_TIE)
    raise ValueError(f"unknown pairwise choice: {choice!r}")


def _build_pairwise_abc_examples_from_records(
    records: Sequence[Dict[str, Any]],
    *,
    dataset_path: str,
    split_name: str,
    pairwise_system_prompt: str,
) -> Tuple[List[PairwiseExample], List[Dict[str, Any]], Dict[str, Any]]:
    out_examples: List[PairwiseExample] = []
    out_rows: List[Dict[str, Any]] = []
    stats: Dict[str, Any] = {
        "dataset_path": str(dataset_path),
        "split_name": str(split_name),
        "input_records": int(len(records)),
        "generated_pairs": 0,
        "skipped_missing_output": 0,
        "skipped_missing_choice": 0,
        "invalid_choice": 0,
        "label_A": 0,
        "label_B": 0,
        "label_C": 0,
        "format": "abc_choice_pairs",
        "choice_mapping": {
            "1": "Assistant 1 better",
            "2": "Assistant 2 better",
            "3": "Tie",
        },
    }

    pair_id = 0
    for rec_i, rec in enumerate(records):
        if not isinstance(rec, dict):
            continue

        rec_id = _safe_int(rec.get("id", rec_i + 1), default=rec_i + 1)
        source_id = _safe_int(rec.get("source_id", rec_id), default=rec_id)
        group_id = int(source_id if source_id > 0 else rec_id)
        dataset = str(rec.get("dataset", "abc_pairwise_eval"))
        instruction = str(rec.get("Instruction", rec.get("instruction", "")))
        input_text = str(rec.get("input", ""))

        pair_specs = (
            ("AB", "modelA", "outputA", "modelB", "outputB", ("choice_AB", "choiceAB", "pairwise_ab_choice")),
            ("BC", "modelB", "outputB", "modelC", "outputC", ("choice_BC", "choiceBC", "pairwise_bc_choice")),
            ("AC", "modelA", "outputA", "modelC", "outputC", ("choice_AC", "choiceAC", "pairwise_ac_choice")),
        )
        for pair_name, model_a_key, out_a_key, model_b_key, out_b_key, choice_keys in pair_specs:
            out_a_position = str(out_a_key)[-1]
            out_b_position = str(out_b_key)[-1]
            out_a = str(rec.get(out_a_key, rec.get(f"answer{out_a_position}", "")))
            out_b = str(rec.get(out_b_key, rec.get(f"answer{out_b_position}", "")))
            if not out_a.strip() or not out_b.strip():
                stats["skipped_missing_output"] += 1
                continue

            raw_choice: Any = ""
            for choice_key in choice_keys:
                candidate_choice = rec.get(choice_key, "")
                if candidate_choice is not None and str(candidate_choice).strip() != "":
                    raw_choice = candidate_choice
                    break
            # The judge export also appears in a nested format:
            # pairwise: {AB: {choice: ...}, BC: {choice: ...}}.
            # Prefer its explicit choice when the legacy top-level aliases are absent.
            if raw_choice is None or str(raw_choice).strip() == "":
                nested_pairwise = rec.get("pairwise", {})
                if isinstance(nested_pairwise, dict):
                    nested_value = nested_pairwise.get(str(pair_name), "")
                    if isinstance(nested_value, dict):
                        # choice_code is local to this pair (1=first, 2=second,
                        # 3=tie); the letter in `choice` is a global A/B/C label.
                        raw_choice = nested_value.get("choice_code", nested_value.get("choice", ""))
                    elif nested_value is not None:
                        raw_choice = nested_value
            if pair_name == "AB" and (raw_choice is None or str(raw_choice).strip() == ""):
                raw_choice = rec.get("choice", rec.get("raw_choice", ""))
            if raw_choice is None or str(raw_choice).strip() == "":
                stats["skipped_missing_choice"] += 1
                continue
            try:
                label = _abc_choice_to_pairwise_label(raw_choice)
            except ValueError:
                stats["invalid_choice"] += 1
                raise
            token = label_to_token(int(label))
            if int(label) == int(LABEL_A):
                stats["label_A"] += 1
            elif int(label) == int(LABEL_B):
                stats["label_B"] += 1
            else:
                stats["label_C"] += 1

            pair_id += 1
            model_a = str(rec.get(model_a_key, out_a_position))
            model_b = str(rec.get(model_b_key, out_b_position))
            prompt = build_pairwise_prompt(
                system_prompt=pairwise_system_prompt,
                instruction=instruction,
                input_text=input_text,
                assistant_1_output=out_a,
                assistant_2_output=out_b,
            )
            out_examples.append(
                PairwiseExample(
                    id=int(pair_id),
                    dataset=dataset,
                    group_id=int(group_id),
                    pair_id=int(pair_id),
                    model_a=model_a,
                    model_b=model_b,
                    prompt=prompt,
                    label=int(label),
                )
            )
            out_rows.append(
                {
                    "pair_id": int(pair_id),
                    "group_id": int(group_id),
                    "source_id": int(source_id),
                    "record_id": int(rec_id),
                    "dataset": dataset,
                    "pair_name": str(pair_name),
                    "model_a": model_a,
                    "model_b": model_b,
                    "raw_choice": raw_choice,
                    "pairwise_label": int(label),
                    "pairwise_token": str(token),
                }
            )

    stats["generated_pairs"] = int(len(out_examples))
    return out_examples, out_rows, stats


def _load_pairwise_abc_raw_records(path: str) -> List[Dict[str, Any]]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("pairwise ABC dataset JSON must be a list")
    return [x for x in raw if isinstance(x, dict)]


def _load_pairwise_abc_eval_dataset(
    path: str,
    *,
    pairwise_system_prompt: str,
) -> Tuple[List[PairwiseExample], List[Dict[str, Any]], Dict[str, Any]]:
    records = _load_pairwise_abc_raw_records(str(path))
    out_examples, out_rows, stats = _build_pairwise_abc_examples_from_records(
        records,
        dataset_path=str(path),
        split_name="eval_all",
        pairwise_system_prompt=pairwise_system_prompt,
    )
    if not out_examples:
        raise RuntimeError(f"pairwise ABC eval dataset produced no examples: {path}")
    return out_examples, out_rows, stats


def _split_pairwise_abc_dataset(
    path: str,
    *,
    train_records: int,
    train_ratio: float,
    seed: int,
    pairwise_system_prompt: str,
) -> Tuple[
    List[PairwiseExample],
    List[Dict[str, Any]],
    Dict[str, Any],
    List[PairwiseExample],
    List[Dict[str, Any]],
    Dict[str, Any],
    Dict[str, Any],
]:
    records = _load_pairwise_abc_raw_records(str(path))
    n = int(len(records))
    if n <= 1:
        raise RuntimeError(f"pairwise ABC split requires at least 2 records, got {n}")

    if int(train_records) > 0:
        k = int(train_records)
    elif float(train_ratio) > 0.0:
        k = int(round(float(train_ratio) * n))
    else:
        k = 0
    k = max(0, min(n - 1, k))

    rng = np.random.default_rng(int(seed))
    train_idx: Set[int] = set()
    if k > 0:
        train_idx = {int(i) for i in rng.choice(n, size=k, replace=False).tolist()}

    train_records_list = [records[i] for i in range(n) if i in train_idx]
    eval_records_list = [records[i] for i in range(n) if i not in train_idx]

    train_examples, train_rows, train_info = _build_pairwise_abc_examples_from_records(
        train_records_list,
        dataset_path=str(path),
        split_name="train",
        pairwise_system_prompt=pairwise_system_prompt,
    )
    eval_examples, eval_rows, eval_info = _build_pairwise_abc_examples_from_records(
        eval_records_list,
        dataset_path=str(path),
        split_name="eval",
        pairwise_system_prompt=pairwise_system_prompt,
    )
    if k > 0 and not train_examples:
        raise RuntimeError(f"pairwise ABC train split produced no examples: {path}")
    if not eval_examples:
        raise RuntimeError(f"pairwise ABC eval split produced no examples: {path}")

    split_info = {
        "dataset_path": str(path),
        "seed": int(seed),
        "input_records": int(n),
        "train_records_requested": int(train_records),
        "train_ratio_requested": float(train_ratio),
        "train_records": int(len(train_records_list)),
        "eval_records": int(len(eval_records_list)),
        "train_pairs": int(len(train_examples)),
        "eval_pairs": int(len(eval_examples)),
        "split_unit": "record",
        "leakage_guard": "AB and BC from the same ABC record stay in the same split",
    }
    return train_examples, train_rows, train_info, eval_examples, eval_rows, eval_info, split_info


# ===== SFT Mode Support =====

IGNORE_INDEX = -100
DEFAULT_EOS_TOKEN = "</s>"


def _tokenize_fn_sft(strings: Sequence[str], tokenizer) -> Dict[str, Any]:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = [tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list]
    return dict(
        input_ids=input_ids,
        input_ids_lens=input_ids_lens,
    )


def _truncate_ids_preserve_edges(ids: Sequence[int], max_length: int) -> List[int]:
    """Keep both the prompt preamble and its final instruction when truncating."""
    limit = max(1, int(max_length))
    values = [int(x) for x in ids]
    if len(values) <= limit:
        return values
    head = max(1, limit // 2)
    tail = max(1, limit - head)
    return values[:head] + values[-tail:]


def preprocess_sft(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer,
) -> Dict[str, Any]:
    """Preprocess the data by tokenizing."""
    eos_token = str(tokenizer.eos_token or "")
    normalized_targets = [
        str(target).replace(DEFAULT_EOS_TOKEN, eos_token) if eos_token else str(target)
        for target in targets
    ]
    input_ids: List[Any] = []
    labels: List[Any] = []
    max_length = int(tokenizer.model_max_length)
    for source, target in zip(sources, normalized_targets):
        source_ids = tokenizer(str(source), add_special_tokens=True, truncation=False).input_ids
        target_ids = tokenizer(str(target), add_special_tokens=False, truncation=False).input_ids
        # Reserve the target tokens so long prompts cannot truncate away the answer label.
        target_ids = _truncate_ids_preserve_edges(target_ids, max_length)
        source_budget = max(1, max_length - len(target_ids))
        source_ids = _truncate_ids_preserve_edges(source_ids, source_budget)
        ids = list(source_ids) + list(target_ids)
        input_ids.append(torch.tensor(ids, dtype=torch.long))
        labels.append(torch.tensor([IGNORE_INDEX] * len(source_ids) + list(target_ids), dtype=torch.long))
    return dict(input_ids=input_ids, labels=labels)


class SFTPairwiseDataset(Dataset):
    """Dataset for SFT training, with optional pointwise score metadata."""

    def __init__(
        self,
        sources: Sequence[str],
        targets: Sequence[str],
        tokenizer,
        pointwise_score_labels: Optional[Sequence[int]] = None,
        pointwise_score_token_ids: Optional[Sequence[Sequence[int]]] = None,
        pointwise_teacher_logits: Optional[Sequence[Optional[Sequence[float]]]] = None,
        class_teacher_logits: Optional[Sequence[Optional[Sequence[float]]]] = None,
        class_teacher_task_ids: Optional[Sequence[int]] = None,
        choice_target_distributions: Optional[Sequence[Optional[Mapping[str, float]]]] = None,
        choice_candidate_targets: Optional[Sequence[Optional[Sequence[str]]]] = None,
    ):
        self.tokenizer = tokenizer
        self.input_ids = []
        self.labels = []

        processed = preprocess_sft(sources, targets, tokenizer)
        self.input_ids = processed["input_ids"]
        self.labels = processed["labels"]
        self.choice_source_lengths = [
            int(label.ne(IGNORE_INDEX).nonzero(as_tuple=False)[0].item())
            if bool(label.ne(IGNORE_INDEX).any().item()) else int(len(label))
            for label in self.labels
        ]
        if pointwise_score_labels is None:
            self.pointwise_score_labels = [IGNORE_INDEX] * len(self.input_ids)
        else:
            if len(pointwise_score_labels) != len(self.input_ids):
                raise ValueError(
                    "pointwise_score_labels length must match SFT dataset length: "
                    f"{len(pointwise_score_labels)} != {len(self.input_ids)}"
                )
            self.pointwise_score_labels = [int(x) for x in pointwise_score_labels]
        self.pointwise_score_positions = [IGNORE_INDEX] * len(self.input_ids)
        if pointwise_score_token_ids is not None:
            score_sequences = [[int(token_id) for token_id in ids] for ids in pointwise_score_token_ids]
            for row_index, (label, labels) in enumerate(zip(self.pointwise_score_labels, self.labels)):
                if not 0 <= int(label) < len(score_sequences):
                    continue
                needle = score_sequences[int(label)]
                if not needle:
                    raise ValueError("pointwise score token sequence must not be empty")
                values = [int(x) for x in labels.tolist()]
                position = -1
                for start in range(0, len(values) - len(needle) + 1):
                    if values[start : start + len(needle)] == needle:
                        position = int(start)
                if position < 0:
                    raise ValueError(
                        "could not locate the final pointwise score token sequence in target: "
                        f"row={row_index} label={label}"
                    )
                self.pointwise_score_positions[row_index] = int(position)
        self.pointwise_teacher_logits: Optional[List[Optional[List[float]]]] = None
        if pointwise_teacher_logits is not None:
            if len(pointwise_teacher_logits) != len(self.input_ids):
                raise ValueError(
                    "pointwise_teacher_logits length must match SFT dataset length: "
                    f"{len(pointwise_teacher_logits)} != {len(self.input_ids)}"
                )
            normalized_teacher_logits: List[Optional[List[float]]] = []
            width: Optional[int] = None
            for logits in pointwise_teacher_logits:
                if logits is None:
                    normalized_teacher_logits.append(None)
                    continue
                values = [float(x) for x in logits]
                if width is None:
                    width = len(values)
                elif len(values) != width:
                    raise ValueError(
                        "all pointwise_teacher_logits entries must have the same length: "
                        f"{len(values)} != {width}"
                    )
                normalized_teacher_logits.append(values)
            self.pointwise_teacher_logits = normalized_teacher_logits
        self.class_teacher_logits: Optional[List[Optional[List[float]]]] = None
        self.class_teacher_task_ids: Optional[List[int]] = None
        if class_teacher_logits is not None:
            if len(class_teacher_logits) != len(self.input_ids):
                raise ValueError(
                    "class_teacher_logits length must match SFT dataset length: "
                    f"{len(class_teacher_logits)} != {len(self.input_ids)}"
                )
            normalized_class_logits: List[Optional[List[float]]] = []
            for logits in class_teacher_logits:
                if logits is None:
                    normalized_class_logits.append(None)
                else:
                    normalized_class_logits.append([float(x) for x in logits])
            self.class_teacher_logits = normalized_class_logits
            if class_teacher_task_ids is None:
                self.class_teacher_task_ids = [0] * len(self.input_ids)
            else:
                if len(class_teacher_task_ids) != len(self.input_ids):
                    raise ValueError(
                        "class_teacher_task_ids length must match SFT dataset length: "
                        f"{len(class_teacher_task_ids)} != {len(self.input_ids)}"
                    )
                self.class_teacher_task_ids = [int(x) for x in class_teacher_task_ids]

        self.choice_target_distributions: Optional[List[Optional[Dict[str, float]]]] = None
        self.choice_candidate_token_ids: Optional[List[Optional[List[List[int]]]]] = None
        if choice_target_distributions is not None or choice_candidate_targets is not None:
            if choice_target_distributions is None or choice_candidate_targets is None:
                raise ValueError("choice_target_distributions and choice_candidate_targets must be provided together")
            if len(choice_target_distributions) != len(self.input_ids) or len(choice_candidate_targets) != len(self.input_ids):
                raise ValueError("choice metadata length must match SFT dataset length")
            eos_token = str(tokenizer.eos_token or "")
            normalized_dist: List[Optional[Dict[str, float]]] = []
            normalized_ids: List[Optional[List[List[int]]]] = []
            for dist, candidates in zip(choice_target_distributions, choice_candidate_targets):
                if dist is None or candidates is None:
                    normalized_dist.append(None)
                    normalized_ids.append(None)
                    continue
                candidate_texts = [
                    str(x).replace(DEFAULT_EOS_TOKEN, eos_token) if eos_token else str(x)
                    for x in candidates
                ]
                values = {str(k): float(v) for k, v in dist.items()}
                if not candidate_texts or len(candidate_texts) != len(values):
                    raise ValueError("choice candidates and distribution must have the same non-zero length")
                if any(v < 0.0 for v in values.values()) or sum(values.values()) <= 0.0:
                    raise ValueError("choice target distribution must be non-negative with positive mass")
                total = sum(values.values())
                values = {key: value / total for key, value in values.items()}
                tokenized = [
                    [int(x) for x in tokenizer(text, add_special_tokens=False, truncation=False).input_ids]
                    for text in candidate_texts
                ]
                if any(not ids for ids in tokenized):
                    raise ValueError("choice candidate target produced no token ids")
                normalized_dist.append(values)
                normalized_ids.append(tokenized)
            self.choice_target_distributions = normalized_dist
            self.choice_candidate_token_ids = normalized_ids

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, Any]:
        item = dict(
            input_ids=self.input_ids[i],
            labels=self.labels[i],
            pointwise_score_label=int(self.pointwise_score_labels[i]),
            pointwise_score_position=int(self.pointwise_score_positions[i]),
            choice_source_length=int(self.choice_source_lengths[i]),
        )
        if self.pointwise_teacher_logits is not None:
            teacher_logits = self.pointwise_teacher_logits[i]
            if teacher_logits is not None:
                item["pointwise_teacher_logits"] = teacher_logits
                item["pointwise_teacher_mask"] = True
        if self.class_teacher_logits is not None:
            class_logits = self.class_teacher_logits[i]
            class_task_id = int(self.class_teacher_task_ids[i]) if self.class_teacher_task_ids is not None else 0
            if class_logits is not None and class_task_id > 0:
                item["class_teacher_logits"] = class_logits
                item["class_teacher_task_id"] = int(class_task_id)
                item["class_teacher_mask"] = True
        if self.choice_target_distributions is not None:
            item["choice_target_distribution"] = self.choice_target_distributions[i]
            item["choice_candidate_token_ids"] = self.choice_candidate_token_ids[i]
        return item


def _load_sft_model_and_tokenizer(
    *,
    model_name_or_path: str,
    max_length: int,
    load_in_4bit: bool,
):
    import transformers

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name_or_path)
    tokenizer.model_max_length = int(max_length)
    tokenizer.padding_side = "left"

    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        if tokenizer.bos_token is None:
            special_tokens_dict["bos_token"] = "<s>"
        if tokenizer.eos_token is None:
            special_tokens_dict["eos_token"] = "</s>"
        if tokenizer.unk_token is None:
            special_tokens_dict["unk_token"] = "<unk>"
        if tokenizer.pad_token is None:
            special_tokens_dict["pad_token"] = "[PAD]"

    compute_dtype = torch.float32
    if load_in_4bit:
        from transformers import BitsAndBytesConfig

        if torch.cuda.is_available():
            if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
                compute_dtype = torch.bfloat16
            else:
                compute_dtype = torch.float16
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            quantization_config=bnb_config,
            device_map="auto",
        )
    else:
        model = transformers.AutoModelForCausalLM.from_pretrained(model_name_or_path)

    # Add missing tokens only after model loading so the embedding resize is
    # applied to the model as well. Previously Llama's new [PAD] id exceeded
    # the unchanged embedding table and failed during padded batch evaluation.
    smart_tokenizer_and_embedding_resize_sft(special_tokens_dict, tokenizer, model)
    return model, tokenizer, compute_dtype


def _prepare_model_for_kbit_lora_sft(model, *, load_in_4bit: bool):
    if not bool(load_in_4bit):
        return model
    try:
        from peft import prepare_model_for_kbit_training
    except Exception as exc:
        print(f"Warning: prepare_model_for_kbit_training unavailable, continuing without it: {exc}")
        return model
    return prepare_model_for_kbit_training(model)


def smart_tokenizer_and_embedding_resize_sft(
    special_tokens_dict: Dict[str, str],
    tokenizer,
    model,
):
    """Resize tokenizer and embedding."""
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    if model is None or num_new_tokens <= 0:
        return num_new_tokens
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _train_sft_pairwise(
    *,
    model_name_or_path: Optional[str] = None,
    model: Optional[Any] = None,
    tokenizer: Optional[Any] = None,
    pairwise_train: Sequence[PairwiseExample],
    pairwise_val: Sequence[PairwiseExample],
    pointwise_replay: Optional[Sequence[PointwiseScoredExample]] = None,
    pointwise_replay_ratio: int = 0,
    aligned_pairs: Optional[Sequence[SelectedQuestionPair]] = None,
    stage2_mix_mode: str = "replay",
    stage2_pairs_per_batch: int = 4,
    epochs: int,
    per_device_batch_size: int,
    gradient_accumulation_steps: int,
    learning_rate: float,
    max_length: int,
    use_lora: bool,
    load_in_4bit: bool,
    seed: int,
    output_dir: str,
    fix_score_prefix_in_prompt: bool = True,
    score_min: int = 1,
    score_max: int = 10,
    randomize_pair_order: bool = True,
    global_smooth_alpha: float = 0.0,
    global_smooth_start_step: int = 0,
    global_smooth_warmup_steps: int = 0,
    global_smooth_prior: float = 1.0,
    global_smooth_trainable_alpha: bool = False,
    global_smooth_alpha_max: float = 0.2,
    global_smooth_alpha_reg: float = 0.0,
    global_smooth_alpha_lr: float = 0.0,
    return_model: bool = False,
) -> Any:
    """Train pairwise model using HF Trainer (SFT mode)."""
    import transformers
    from peft import LoraConfig, PeftModel, TaskType, get_peft_model

    # Clean up GPU memory before SFT init
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    reused_existing_model = model is not None and tokenizer is not None

    # Load model and tokenizer only when an existing pair is not provided.
    if not reused_existing_model:
        if not model_name_or_path:
            raise ValueError("model_name_or_path is required when model/tokenizer are not provided")

        print("Loading model and tokenizer...")
        model, tokenizer, _ = _load_sft_model_and_tokenizer(
            model_name_or_path=str(model_name_or_path),
            max_length=int(max_length),
            load_in_4bit=bool(load_in_4bit),
        )
    else:
        print("Reusing existing model and tokenizer for SFT stage...")
        tokenizer.model_max_length = int(max_length)

    assert model is not None
    assert tokenizer is not None

    # Enable gradient checkpointing for memory efficiency
    model.gradient_checkpointing_enable()
    if hasattr(model, "config"):
        model.config.use_cache = False

    # Prepare datasets
    print("Preparing SFT datasets...")
    requested_mix_mode = str(stage2_mix_mode)
    if requested_mix_mode not in {"replay", "pair_batch", "pair_batch_both"}:
        raise ValueError(f"unsupported stage2_mix_mode: {requested_mix_mode}")

    train_sources = [x.prompt for x in pairwise_train]
    train_targets = [f"[[{label_to_token(int(x.label))}]]" + DEFAULT_EOS_TOKEN for x in pairwise_train]
    train_pointwise_score_labels = [IGNORE_INDEX] * len(train_sources)
    pairwise_train_samples = int(len(train_sources))
    replay_ratio = max(0, int(pointwise_replay_ratio))
    pointwise_replay_samples = 0
    pair_batch_stats: Dict[str, Any] = {}
    train_batch_size_for_args = max(1, int(per_device_batch_size))
    use_sequential_train_loader = False
    active_mix_mode = requested_mix_mode

    if requested_mix_mode in {"pair_batch", "pair_batch_both"}:
        if aligned_pairs:
            train_sources, train_targets, train_pointwise_score_labels, pair_batch_stats = _build_pair_batch_sft_sources_targets(
                aligned_pairs=list(aligned_pairs),
                pairwise_system_prompt=DEFAULT_PAIRWISE_SYSTEM_PROMPT,
                judge_system_prompt=JUDGE_SYSTEM_PROMPT_SCORE_ONLY,
                seed=int(seed),
                pairs_per_batch=max(1, int(stage2_pairs_per_batch)),
                pairwise_directions="both" if requested_mix_mode == "pair_batch_both" else "one",
                score_min=int(score_min),
                score_max=int(score_max),
                fix_score_prefix_in_prompt=bool(fix_score_prefix_in_prompt),
                randomize_pair_order=bool(randomize_pair_order),
            )
            pairwise_train_samples = int(pair_batch_stats.get("pairwise_train_samples", 0))
            pointwise_replay_samples = int(pair_batch_stats.get("pointwise_train_samples", 0))
            replay_ratio = int(pointwise_replay_samples // max(1, pairwise_train_samples))
            train_batch_size_for_args = int(pair_batch_stats.get("logical_batch_size", train_batch_size_for_args))
            use_sequential_train_loader = True
            print(
                "SFT pair-batch stage enabled: "
                f"pairs={pair_batch_stats.get('selected_pairs', 0)} "
                f"pairs_per_batch={pair_batch_stats.get('pairs_per_batch', 0)} "
                f"logical_batch_size={pair_batch_stats.get('logical_batch_size', 0)} "
                f"pointwise={pointwise_replay_samples} "
                f"pairwise={pairwise_train_samples}",
                flush=True,
            )
        else:
            print("SFT pair-batch requested but aligned selected pairs were not provided; falling back to replay.", flush=True)
            active_mix_mode = "replay"

    if active_mix_mode == "replay":
        if replay_ratio > 0 and pointwise_replay:
            replay_examples = list(pointwise_replay)
            replay_total = int(pairwise_train_samples * replay_ratio)
            replay_rng = np.random.default_rng(int(seed) + 409)
            replay_indices: List[int] = []
            while len(replay_indices) < replay_total:
                replay_indices.extend(
                    int(i) for i in replay_rng.permutation(len(replay_examples)).astype(np.int64).tolist()
                )
            for idx in replay_indices[:replay_total]:
                ex = replay_examples[int(idx)]
                train_sources.append(str(ex.prompt))
                train_targets.append(
                    _pointwise_sft_target(
                        ex,
                        fix_score_prefix_in_prompt=bool(fix_score_prefix_in_prompt),
                    )
                )
                train_pointwise_score_labels.append(int(ex.score) - int(score_min))
                pointwise_replay_samples += 1

            shuffle_order = replay_rng.permutation(len(train_sources)).astype(np.int64).tolist()
            train_sources = [train_sources[int(i)] for i in shuffle_order]
            train_targets = [train_targets[int(i)] for i in shuffle_order]
            train_pointwise_score_labels = [train_pointwise_score_labels[int(i)] for i in shuffle_order]
            print(
                "SFT stage replay enabled: "
                f"pairwise={pairwise_train_samples} "
                f"pointwise_replay={pointwise_replay_samples} "
                f"ratio={replay_ratio}",
                flush=True,
            )
        elif replay_ratio > 0:
            print("SFT stage replay requested but no pointwise replay examples were provided.", flush=True)

    val_sources = None
    val_targets = None
    if pairwise_val:
        val_sources = [x.prompt for x in pairwise_val]
        val_targets = [f"[[{label_to_token(int(x.label))}]]" + DEFAULT_EOS_TOKEN for x in pairwise_val]

    train_dataset = SFTPairwiseDataset(
        train_sources,
        train_targets,
        tokenizer,
        pointwise_score_labels=train_pointwise_score_labels,
    )
    eval_dataset = SFTPairwiseDataset(val_sources, val_targets, tokenizer) if val_sources else None

    # Apply LoRA if requested
    if use_lora and not isinstance(model, PeftModel):
        model = _prepare_model_for_kbit_lora_sft(model, load_in_4bit=bool(load_in_4bit))
        print("Applying LoRA...")
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )
        model = get_peft_model(model, lora_config)
    elif use_lora:
        print("Model already has LoRA adapters; reusing them for SFT stage...")

    # Setup trainer
    print("Setting up HF Trainer...")
    training_args = HFTrainingArguments(
        output_dir=output_dir,
        do_train=True,
        do_eval=bool(eval_dataset),
        per_device_train_batch_size=int(train_batch_size_for_args),
        per_device_eval_batch_size=int(per_device_batch_size),
        gradient_accumulation_steps=int(gradient_accumulation_steps),
        num_train_epochs=int(epochs),
        learning_rate=float(learning_rate),
        weight_decay=0.0,
        lr_scheduler_type="cosine",
        warmup_steps=max(1, int(len(train_dataset) * int(epochs) / (int(train_batch_size_for_args) * int(gradient_accumulation_steps)) * 0.1)),
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        eval_strategy="epoch" if eval_dataset else "no",
        dataloader_pin_memory=True,
        gradient_checkpointing=True,
        remove_unused_columns=False,
        seed=int(seed),
    )

    smooth_enabled = float(global_smooth_alpha) > 0.0 and int(pointwise_replay_samples) > 0
    trainer_cls = OnlineGlobalPriorSFTTrainer
    if use_sequential_train_loader:
        from torch.utils.data import DataLoader

        class SequentialSFTTrainer(trainer_cls):
            def get_train_dataloader(self):
                if self.train_dataset is None:
                    raise ValueError("Trainer: training requires a train_dataset.")
                return DataLoader(
                    self.train_dataset,
                    batch_size=int(self.args.per_device_train_batch_size),
                    shuffle=False,
                    collate_fn=self.data_collator,
                    drop_last=bool(self.args.dataloader_drop_last),
                    num_workers=int(self.args.dataloader_num_workers),
                    pin_memory=bool(self.args.dataloader_pin_memory),
                )

        trainer_cls = SequentialSFTTrainer

    trainer_kwargs = {
        "score_token_ids": _score_token_ids_for_sft(tokenizer, score_min=int(score_min), score_max=int(score_max)),
        "smooth_alpha": float(global_smooth_alpha) if smooth_enabled else 0.0,
        "smooth_start_step": int(global_smooth_start_step),
        "smooth_warmup_steps": int(global_smooth_warmup_steps),
        "smooth_prior": float(global_smooth_prior),
        "smooth_trainable_alpha": bool(global_smooth_trainable_alpha) and bool(smooth_enabled),
        "smooth_alpha_max": float(global_smooth_alpha_max),
        "smooth_alpha_reg": float(global_smooth_alpha_reg),
    }
    if smooth_enabled:
        print(
            "Stage-2 pointwise replay global-prior smoothing enabled: "
            f"alpha={float(global_smooth_alpha)} "
            f"start_step={int(global_smooth_start_step)} "
            f"warmup_steps={int(global_smooth_warmup_steps)} "
            f"prior={float(global_smooth_prior)} "
            f"trainable_alpha={bool(global_smooth_trainable_alpha)} "
            f"alpha_max={float(global_smooth_alpha_max)} "
            f"alpha_reg={float(global_smooth_alpha_reg)} "
            f"alpha_lr={float(global_smooth_alpha_lr)}",
            flush=True,
        )
    trainer = trainer_cls(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=_data_collator_sft,
        **trainer_kwargs,
    )

    # Train
    print(f"Training SFT model with {len(train_dataset)} samples for {epochs} epochs...")
    t0 = time.time()
    trainer.train()
    elapsed = time.time() - t0

    # Save model
    if use_lora:
        model.save_pretrained(output_dir)
    else:
        trainer.save_model(output_dir)

    # Evaluate on pairwise validation set by generation (if available).
    eval_metrics = _evaluate_pairwise_sft(
        model=model,
        tokenizer=tokenizer,
        examples=pairwise_val,
        max_length=int(max_length),
        batch_size=max(1, int(per_device_batch_size)),
        max_new_tokens=8,
    )

    parent_out = Path(output_dir).parent
    _write_json(parent_out / "metrics_pairwise_sft_val.json", eval_metrics)

    if active_mix_mode in {"pair_batch", "pair_batch_both"}:
        mode_name = "sft_pair_batch"
    elif pointwise_replay_samples > 0:
        mode_name = "sft_mixed_replay"
    else:
        mode_name = "sft"

    stats = {
        "mode": mode_name,
        "stage2_mix_mode": str(active_mix_mode),
        "reused_existing_model": bool(reused_existing_model),
        "train_samples": len(train_dataset),
        "pairwise_train_samples": int(pairwise_train_samples),
        "pointwise_replay_samples": int(pointwise_replay_samples),
        "pointwise_replay_ratio": int(replay_ratio),
        "train_batch_size": int(train_batch_size_for_args),
        "pair_batch": pair_batch_stats if pair_batch_stats else None,
        "val_samples": len(eval_dataset) if eval_dataset else 0,
        "epochs": int(epochs),
        "elapsed_sec": elapsed,
        "global_prior_smoothing": (
            trainer.get_global_prior_smoothing_stats()
            if isinstance(trainer, OnlineGlobalPriorSFTTrainer)
            else {"enabled": False}
        ),
        "eval_pairwise": eval_metrics,
    }

    if bool(return_model):
        return stats, model, tokenizer
    return stats


def _pointwise_sft_target(
    example: PointwiseScoredExample,
    *,
    fix_score_prefix_in_prompt: bool,
    cot_feedback: bool = False,
) -> str:
    score_text = str(int(example.score))
    reason = str(getattr(example, "reason", "") or "").strip()
    if bool(cot_feedback) and reason:
        return f"{reason}\nScore: [{score_text}]" + DEFAULT_EOS_TOKEN
    if bool(fix_score_prefix_in_prompt):
        return score_text + "]" + DEFAULT_EOS_TOKEN
    return f"Score: [{score_text}]" + DEFAULT_EOS_TOKEN


def _build_pair_batch_sft_sources_targets(
    *,
    aligned_pairs: Sequence[SelectedQuestionPair],
    pairwise_system_prompt: str,
    judge_system_prompt: str,
    seed: int,
    pairs_per_batch: int,
    pairwise_directions: str,
    score_min: int,
    score_max: int,
    fix_score_prefix_in_prompt: bool,
    randomize_pair_order: bool,
) -> Tuple[List[str], List[str], List[int], Dict[str, Any]]:
    """Build stage-2 SFT blocks from aligned selected pairs."""
    pair_list = list(aligned_pairs)
    if not pair_list:
        raise RuntimeError("pair_batch SFT requested with no aligned selected pairs")

    pairs_per_batch = max(1, int(pairs_per_batch))
    pairwise_directions = str(pairwise_directions)
    if pairwise_directions not in {"one", "both"}:
        raise ValueError(f"unsupported pairwise_directions: {pairwise_directions}")

    rng = np.random.default_rng(int(seed) + 409)
    order = rng.permutation(len(pair_list)).astype(np.int64).tolist()

    sources: List[str] = []
    targets: List[str] = []
    pointwise_score_labels: List[int] = []
    pointwise_samples = 0
    pairwise_samples = 0
    row_id = 0
    swapped_pairwise = 0
    batches = 0

    for block_start in range(0, len(order), pairs_per_batch):
        block_indices = order[block_start : block_start + pairs_per_batch]
        block = [pair_list[int(i)] for i in block_indices]
        if not block:
            continue
        batches += 1

        # Put the score supervision first in each logical batch: 2 pointwise per pair.
        for selected_pair in block:
            for ans in (selected_pair.answer_a, selected_pair.answer_b):
                prompt = build_judge_prompt(
                    system_prompt=judge_system_prompt,
                    instruction=str(selected_pair.instruction),
                    input_text=str(selected_pair.input_text),
                    candidate_output=str(ans.output),
                    include_gold_score=False,
                    fix_score_prefix=bool(fix_score_prefix_in_prompt),
                )
                row_id += 1
                ex = PointwiseScoredExample(
                    row_id=int(row_id),
                    question_id=int(selected_pair.question_id),
                    source_id=int(selected_pair.source_id),
                    dataset=str(selected_pair.dataset),
                    instruction=str(selected_pair.instruction),
                    input_text=str(selected_pair.input_text),
                    model=str(ans.model),
                    output=str(ans.output),
                    score=int(ans.score),
                    label=int(score_to_class(int(ans.score), score_min=int(score_min), score_max=int(score_max))),
                    prompt=prompt,
                )
                sources.append(str(ex.prompt))
                targets.append(_pointwise_sft_target(ex, fix_score_prefix_in_prompt=bool(fix_score_prefix_in_prompt)))
                pointwise_score_labels.append(int(ex.score) - int(score_min))
                pointwise_samples += 1

        # Then put pairwise comparisons per pair. The "both" mode preserves order augmentation
        # inside each logical batch: original/randomized direction plus its inverse.
        for selected_pair in block:
            ans_a = selected_pair.answer_a
            ans_b = selected_pair.answer_b
            if bool(randomize_pair_order) and float(rng.random()) < 0.5:
                ans_a, ans_b = ans_b, ans_a
                swapped_pairwise += 1

            pairwise_orders = [(ans_a, ans_b)]
            if pairwise_directions == "both":
                pairwise_orders.append((ans_b, ans_a))
                if ans_a is selected_pair.answer_a:
                    swapped_pairwise += 1

            for left_ans, right_ans in pairwise_orders:
                label = _pairwise_label_from_scores(int(left_ans.score), int(right_ans.score))
                prompt = build_pairwise_prompt(
                    system_prompt=pairwise_system_prompt,
                    instruction=str(selected_pair.instruction),
                    input_text=str(selected_pair.input_text),
                    assistant_1_output=str(left_ans.output),
                    assistant_2_output=str(right_ans.output),
                )
                sources.append(prompt)
                targets.append(f"[[{label_to_token(int(label))}]]" + DEFAULT_EOS_TOKEN)
                pointwise_score_labels.append(IGNORE_INDEX)
                pairwise_samples += 1

    stats = {
        "selected_pairs": int(len(pair_list)),
        "pairs_per_batch": int(pairs_per_batch),
        "logical_batch_size": int(pairs_per_batch * (4 if pairwise_directions == "both" else 3)),
        "logical_batches": int(batches),
        "pointwise_train_samples": int(pointwise_samples),
        "pairwise_train_samples": int(pairwise_samples),
        "total_train_samples": int(len(sources)),
        "randomize_pair_order": bool(randomize_pair_order),
        "swapped_pairwise_samples": int(swapped_pairwise),
        "pairwise_directions": str(pairwise_directions),
        "format": (
            "per_pair_two_pointwise_two_pairwise_both_directions"
            if pairwise_directions == "both"
            else "per_pair_two_pointwise_one_pairwise"
        ),
    }
    return sources, targets, pointwise_score_labels, stats


def _data_collator_sft(batch: List[Dict[str, Any]]):
    """Data collator for SFT training."""
    max_len = max(len(item["input_ids"]) for item in batch)
    input_ids_list = []
    labels_list = []
    attention_mask_list = []
    pointwise_score_labels = []
    pointwise_score_positions = []
    choice_source_lengths = []
    choice_target_distributions = []
    choice_candidate_token_ids = []
    has_choice_targets = any(item.get("choice_target_distribution") is not None for item in batch)
    teacher_values = [item.get("pointwise_teacher_logits") for item in batch]
    has_teacher_logits = any(v is not None for v in teacher_values)
    teacher_width = None
    if has_teacher_logits:
        for v in teacher_values:
            if v is not None:
                teacher_width = len(v)
                break
        if teacher_width is None or teacher_width <= 0:
            has_teacher_logits = False
            teacher_width = None
    pointwise_teacher_logits = []
    pointwise_teacher_mask = []
    class_teacher_values = [item.get("class_teacher_logits") for item in batch]
    has_class_teacher_logits = any(v is not None for v in class_teacher_values)
    class_teacher_width = 0
    if has_class_teacher_logits:
        class_teacher_width = max((len(v) for v in class_teacher_values if v is not None), default=0)
        if class_teacher_width <= 0:
            has_class_teacher_logits = False
    class_teacher_logits = []
    class_teacher_label_mask = []
    class_teacher_task_ids = []
    class_teacher_mask = []

    for item in batch:
        input_ids = item["input_ids"]
        labels = item["labels"]
        pad_len = max_len - len(input_ids)

        input_ids_padded = torch.cat([input_ids, torch.full((pad_len,), 0, dtype=input_ids.dtype)])
        labels_padded = torch.cat([labels, torch.full((pad_len,), IGNORE_INDEX, dtype=labels.dtype)])
        attention_mask = torch.cat([torch.ones(len(input_ids)), torch.zeros(pad_len)])

        input_ids_list.append(input_ids_padded)
        labels_list.append(labels_padded)
        attention_mask_list.append(attention_mask)
        pointwise_score_labels.append(int(item.get("pointwise_score_label", IGNORE_INDEX)))
        pointwise_score_positions.append(int(item.get("pointwise_score_position", IGNORE_INDEX)))
        choice_source_lengths.append(int(item.get("choice_source_length", len(input_ids))))
        if has_choice_targets:
            choice_target_distributions.append(item.get("choice_target_distribution"))
            choice_candidate_token_ids.append(item.get("choice_candidate_token_ids"))
        if has_teacher_logits:
            teacher_logits = item.get("pointwise_teacher_logits")
            if teacher_logits is None:
                pointwise_teacher_logits.append(torch.zeros(int(teacher_width), dtype=torch.float32))
                pointwise_teacher_mask.append(False)
            else:
                values = torch.tensor([float(x) for x in teacher_logits], dtype=torch.float32)
                if int(values.numel()) != int(teacher_width):
                    raise ValueError(
                        "inconsistent pointwise_teacher_logits width inside batch: "
                        f"{int(values.numel())} != {int(teacher_width)}"
                )
                pointwise_teacher_logits.append(values)
                pointwise_teacher_mask.append(bool(item.get("pointwise_teacher_mask", True)))
        if has_class_teacher_logits:
            values = item.get("class_teacher_logits")
            task_id = int(item.get("class_teacher_task_id", 0))
            enabled = values is not None and task_id > 0 and bool(item.get("class_teacher_mask", True))
            padded = torch.zeros(int(class_teacher_width), dtype=torch.float32)
            label_mask = torch.zeros(int(class_teacher_width), dtype=torch.bool)
            if enabled:
                raw = torch.tensor([float(x) for x in values], dtype=torch.float32)
                if int(raw.numel()) > int(class_teacher_width):
                    raise ValueError(
                        "class_teacher_logits width exceeds batch width: "
                        f"{int(raw.numel())} > {int(class_teacher_width)}"
                    )
                padded[: int(raw.numel())] = raw
                label_mask[: int(raw.numel())] = True
            class_teacher_logits.append(padded)
            class_teacher_label_mask.append(label_mask)
            class_teacher_task_ids.append(int(task_id) if enabled else 0)
            class_teacher_mask.append(bool(enabled))

    out = {
        "input_ids": torch.stack(input_ids_list),
        "labels": torch.stack(labels_list),
        "attention_mask": torch.stack(attention_mask_list),
        "pointwise_score_labels": torch.tensor(pointwise_score_labels, dtype=torch.long),
        "pointwise_score_positions": torch.tensor(pointwise_score_positions, dtype=torch.long),
    }
    if has_choice_targets:
        out["choice_source_lengths"] = torch.tensor(choice_source_lengths, dtype=torch.long)
        out["choice_target_distributions"] = choice_target_distributions
        out["choice_candidate_token_ids"] = choice_candidate_token_ids
    if has_teacher_logits:
        out["pointwise_teacher_logits"] = torch.stack(pointwise_teacher_logits)
        out["pointwise_teacher_mask"] = torch.tensor(pointwise_teacher_mask, dtype=torch.bool)
    if has_class_teacher_logits:
        out["class_teacher_logits"] = torch.stack(class_teacher_logits)
        out["class_teacher_label_mask"] = torch.stack(class_teacher_label_mask)
        out["class_teacher_task_ids"] = torch.tensor(class_teacher_task_ids, dtype=torch.long)
        out["class_teacher_mask"] = torch.tensor(class_teacher_mask, dtype=torch.bool)
    return out




def _score_token_ids_for_sft(tokenizer, *, score_min: int, score_max: int) -> List[List[int]]:
    token_ids: List[List[int]] = []
    for score in range(int(score_min), int(score_max) + 1):
        ids = tokenizer(str(score), add_special_tokens=False).input_ids
        if not ids:
            raise ValueError(f"tokenizer produced no token ids for score={score}")
        token_ids.append([int(token_id) for token_id in ids])
    return token_ids


class OnlineGlobalPriorSFTTrainer(Trainer):
    """Trainer that replaces pointwise score-token CE with a soft score target."""

    def __init__(
        self,
        *args,
        score_token_ids: Sequence[Sequence[int]],
        smooth_alpha: float,
        smooth_start_step: int,
        smooth_warmup_steps: int,
        smooth_start_pointwise_seen: int = 0,
        smooth_warmup_pointwise_seen: int = 0,
        smooth_prior: float,
        smooth_initial_hist: Optional[Sequence[float]] = None,
        smooth_freeze_prior: bool = False,
        smooth_uniform_mix: float = 0.0,
        smooth_adaptive_entropy: bool = False,
        smooth_mode: str = "global_prior",
        smooth_gaussian_sigma: float = 1.0,
        smooth_trainable_alpha: bool = False,
        smooth_alpha_max: float = 0.2,
        smooth_alpha_reg: float = 0.0,
        smooth_alpha_lr: float = 0.0,
        pointwise_distill_weight: float = 0.0,
        pointwise_distill_temperature: float = 2.0,
        class_distill_weight: float = 0.0,
        class_distill_temperature: float = 2.0,
        class_distill_candidate_token_ids: Optional[Dict[int, Sequence[int]]] = None,
        class_distill_label_offsets: Optional[Dict[int, int]] = None,
        choice_temperature: float = 1.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.score_token_ids = [[int(token_id) for token_id in ids] for ids in score_token_ids]
        if any(not ids for ids in self.score_token_ids):
            raise ValueError("score token sequences must be non-empty")
        self.smooth_alpha = float(smooth_alpha)
        self.smooth_start_step = max(0, int(smooth_start_step))
        self.smooth_warmup_steps = max(0, int(smooth_warmup_steps))
        self.smooth_start_pointwise_seen = max(0, int(smooth_start_pointwise_seen))
        self.smooth_warmup_pointwise_seen = max(0, int(smooth_warmup_pointwise_seen))
        self.pointwise_seen = 0
        self.smooth_prior = float(smooth_prior)
        self.smooth_uniform_mix = float(smooth_uniform_mix)
        self.smooth_adaptive_entropy = bool(smooth_adaptive_entropy)
        self.smooth_mode = self._canonical_smooth_mode(smooth_mode)
        self.smooth_gaussian_sigma = float(smooth_gaussian_sigma)
        self.smooth_trainable_alpha = bool(smooth_trainable_alpha) and self.smooth_alpha > 0.0
        self.smooth_alpha_max = float(smooth_alpha_max)
        self.smooth_alpha_reg = max(0.0, float(smooth_alpha_reg))
        self.smooth_alpha_lr = max(0.0, float(smooth_alpha_lr))
        self.pointwise_distill_weight = max(0.0, float(pointwise_distill_weight))
        self.pointwise_distill_temperature = max(1e-6, float(pointwise_distill_temperature))
        self.class_distill_weight = max(0.0, float(class_distill_weight))
        self.class_distill_temperature = max(1e-6, float(class_distill_temperature))
        self.class_distill_candidate_token_ids = {
            int(k): [int(x) for x in v]
            for k, v in (class_distill_candidate_token_ids or {}).items()
        }
        self.class_distill_label_offsets = {
            int(k): max(0, int(v))
            for k, v in (class_distill_label_offsets or {}).items()
        }
        self.choice_temperature = max(1e-6, float(choice_temperature))
        self.choice_soft_samples = 0
        if self.smooth_prior <= 0:
            raise ValueError("sft pointwise global smoothing prior must be > 0")
        if not 0.0 <= self.smooth_uniform_mix <= 1.0:
            raise ValueError("sft pointwise global smoothing uniform mix must be in [0, 1]")
        if self.smooth_gaussian_sigma <= 0:
            raise ValueError("sft pointwise local gaussian smoothing sigma must be > 0")
        if self.smooth_alpha_max <= 0:
            raise ValueError("sft pointwise global smoothing alpha max must be > 0")
        if self.smooth_trainable_alpha:
            init_alpha = min(max(float(self.smooth_alpha), 1e-6), float(self.smooth_alpha_max) - 1e-6)
            init_ratio = min(max(init_alpha / float(self.smooth_alpha_max), 1e-6), 1.0 - 1e-6)
            init_raw = math.log(init_ratio / (1.0 - init_ratio))
            device = getattr(self.args, "device", torch.device("cpu"))
            self.smooth_alpha_raw = torch.nn.Parameter(torch.tensor(init_raw, dtype=torch.float32, device=device))
            self.smooth_alpha_init = float(init_alpha)
        else:
            self.smooth_alpha_raw = None
            self.smooth_alpha_init = float(self.smooth_alpha)
        self.score_hist = torch.zeros((len(self.score_token_ids),), dtype=torch.float64)
        self.smooth_freeze_prior = bool(smooth_freeze_prior)
        if smooth_initial_hist is not None:
            initial = torch.tensor(list(smooth_initial_hist), dtype=torch.float64)
            if int(initial.numel()) != len(self.score_token_ids):
                raise ValueError(
                    "smooth_initial_hist length must match score_token_ids length: "
                    f"{int(initial.numel())} != {len(self.score_token_ids)}"
                )
            if bool((initial < 0).any().item()):
                raise ValueError("smooth_initial_hist must be non-negative")
            self.score_hist += initial
        self.score_seen = int(round(float(self.score_hist.sum().item())))

    @staticmethod
    def _canonical_smooth_mode(mode: str) -> str:
        value = str(mode or "global_prior").strip().lower().replace("-", "_")
        aliases = {
            "global": "global_prior",
            "prior": "global_prior",
            "global_prior": "global_prior",
            "local": "local_gaussian",
            "gaussian": "local_gaussian",
            "local_gaussian": "local_gaussian",
        }
        if value not in aliases:
            raise ValueError(
                "sft pointwise smoothing mode must be one of "
                "{'global_prior', 'local_gaussian'}"
            )
        return aliases[value]

    def create_optimizer(self, model=None):
        # FSDP delays optimizer creation until after wrapping and passes the
        # wrapped model here; single-process Trainer calls this with no model.
        optimizer = super().create_optimizer(model)
        if self.smooth_trainable_alpha and self.smooth_alpha_raw is not None:
            already_added = any(
                any(param is self.smooth_alpha_raw for param in group.get("params", []))
                for group in optimizer.param_groups
            )
            if not already_added:
                group = {"params": [self.smooth_alpha_raw], "weight_decay": 0.0}
                if self.smooth_alpha_lr > 0.0:
                    group["lr"] = float(self.smooth_alpha_lr)
                optimizer.add_param_group(group)
        return optimizer

    def _learned_smooth_alpha(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if self.smooth_trainable_alpha and self.smooth_alpha_raw is not None:
            raw = self.smooth_alpha_raw.to(device=device, dtype=dtype)
            return torch.sigmoid(raw) * float(self.smooth_alpha_max)
        return torch.tensor(float(self.smooth_alpha), device=device, dtype=dtype)

    def _current_smooth_alpha(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        base_alpha = self._learned_smooth_alpha(device=device, dtype=dtype)
        if float(base_alpha.detach().item()) <= 0.0:
            return base_alpha * 0.0
        if self.smooth_start_pointwise_seen > 0 or self.smooth_warmup_pointwise_seen > 0:
            seen = int(self.pointwise_seen)
            if seen < self.smooth_start_pointwise_seen:
                return base_alpha * 0.0
            if self.smooth_warmup_pointwise_seen <= 0:
                return base_alpha
            progress = min(
                1.0,
                max(
                    0.0,
                    float(seen - self.smooth_start_pointwise_seen + 1)
                    / float(self.smooth_warmup_pointwise_seen),
                ),
            )
            return base_alpha * float(progress)
        step = int(getattr(self.state, "global_step", 0))
        if step < self.smooth_start_step:
            return base_alpha * 0.0
        if self.smooth_warmup_steps <= 0:
            return base_alpha
        progress = min(1.0, max(0.0, float(step - self.smooth_start_step + 1) / float(self.smooth_warmup_steps)))
        return base_alpha * float(progress)

    def _score_prior(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        hist = self.score_hist.to(device=device, dtype=torch.float32)
        prior = hist + float(self.smooth_prior)
        prior = prior / prior.sum().clamp_min(1e-12)
        if self.smooth_uniform_mix > 0.0:
            uniform = torch.full_like(prior, 1.0 / float(len(self.score_token_ids)))
            prior = (1.0 - float(self.smooth_uniform_mix)) * prior + float(self.smooth_uniform_mix) * uniform
        return prior.to(dtype=dtype)

    def _local_gaussian_prior(
        self,
        score_labels: torch.Tensor,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        k = int(len(self.score_token_ids))
        grid = torch.arange(k, device=device, dtype=torch.float32).unsqueeze(0)
        centers = score_labels.to(device=device, dtype=torch.float32).unsqueeze(-1)
        sigma = max(float(self.smooth_gaussian_sigma), 1e-12)
        weights = torch.exp(-0.5 * torch.square((grid - centers) / float(sigma)))
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        if self.smooth_uniform_mix > 0.0:
            uniform = torch.full_like(weights, 1.0 / float(k))
            weights = (1.0 - float(self.smooth_uniform_mix)) * weights + float(self.smooth_uniform_mix) * uniform
        return weights.to(dtype=dtype)

    def _score_smoothing_distribution(
        self,
        score_labels: torch.Tensor,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if self.smooth_mode == "local_gaussian":
            return self._local_gaussian_prior(score_labels, device=device, dtype=dtype)
        prior = self._score_prior(device=device, dtype=dtype)
        return prior.unsqueeze(0).expand(int(score_labels.numel()), -1)

    def _update_score_hist(self, score_labels: torch.Tensor) -> int:
        valid = score_labels.detach().to(device=score_labels.device, dtype=torch.long)
        valid = valid[(valid >= 0) & (valid < len(self.score_token_ids))]
        counts = torch.bincount(valid, minlength=len(self.score_token_ids)).to(dtype=torch.long)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(counts, op=torch.distributed.ReduceOp.SUM)
        global_count = int(counts.sum().item())
        if not self.smooth_freeze_prior:
            self.score_hist += counts.to(device="cpu", dtype=torch.float64)
            self.score_seen += global_count
        return global_count

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        choice_source_lengths = inputs.pop("choice_source_lengths", None)
        choice_target_distributions = inputs.pop("choice_target_distributions", None)
        choice_candidate_token_ids = inputs.pop("choice_candidate_token_ids", None)
        pointwise_score_labels = inputs.pop("pointwise_score_labels", None)
        pointwise_score_positions = inputs.pop("pointwise_score_positions", None)
        pointwise_teacher_logits = inputs.pop("pointwise_teacher_logits", None)
        pointwise_teacher_mask = inputs.pop("pointwise_teacher_mask", None)
        class_teacher_logits = inputs.pop("class_teacher_logits", None)
        class_teacher_label_mask = inputs.pop("class_teacher_label_mask", None)
        class_teacher_task_ids = inputs.pop("class_teacher_task_ids", None)
        class_teacher_mask = inputs.pop("class_teacher_mask", None)
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        if labels is None:
            loss = outputs.loss
            return (loss, outputs) if return_outputs else loss

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        flat_loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=IGNORE_INDEX,
            reduction="none",
        )
        token_loss = flat_loss.view_as(shift_labels)
        loss_mask = shift_labels.ne(IGNORE_INDEX)
        distributed_dummy_loss: Optional[torch.Tensor] = None

        if pointwise_score_labels is not None and bool(model.training):
            score_labels = pointwise_score_labels.to(device=labels.device, dtype=torch.long)
            valid_score = (score_labels >= 0) & (score_labels < len(self.score_token_ids))
            label_mask = labels.ne(IGNORE_INDEX)
            has_label = label_mask.any(dim=1)
            first_label_pos = label_mask.to(dtype=torch.int64).argmax(dim=1)
            if pointwise_score_positions is not None:
                score_token_pos = pointwise_score_positions.to(device=labels.device, dtype=torch.long)
            else:
                score_token_pos = first_label_pos
            valid_score = valid_score & has_label & (score_token_pos > 0)

            alpha_t = self._current_smooth_alpha(device=labels.device, dtype=shift_logits.dtype)
            local_has_score = bool(valid_score.any().item())
            global_has_score = local_has_score
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                has_score_flag = torch.tensor(int(local_has_score), device=labels.device, dtype=torch.long)
                torch.distributed.all_reduce(has_score_flag, op=torch.distributed.ReduceOp.MAX)
                global_has_score = bool(has_score_flag.item())
            if float(alpha_t.detach().item()) > 0.0 and global_has_score:
                if local_has_score:
                    batch_idx = torch.arange(labels.size(0), device=labels.device)[valid_score]
                    score_shift_pos = score_token_pos[valid_score] - 1
                    valid_score_labels = score_labels[valid_score]
                else:
                    # FSDP requires every rank to execute the same nested model
                    # forwards. Use one zero-weight row when this rank received
                    # a non-pointwise example while another rank smooths one.
                    batch_idx = torch.zeros((1,), device=labels.device, dtype=torch.long)
                    score_shift_pos = torch.zeros((1,), device=labels.device, dtype=torch.long)
                    valid_score_labels = torch.zeros((1,), device=labels.device, dtype=torch.long)
                candidate_nlls: List[torch.Tensor] = []
                for candidate_ids in self.score_token_ids:
                    candidate_nll = torch.zeros(
                        (int(batch_idx.numel()),),
                        device=labels.device,
                        dtype=shift_logits.dtype,
                    )
                    for offset, token_id in enumerate(candidate_ids):
                        positions = score_shift_pos + int(offset)
                        in_bounds = positions < int(shift_logits.size(1))
                        if not bool(in_bounds.all().item()):
                            candidate_nll = candidate_nll.masked_fill(~in_bounds, float("inf"))
                        if bool(in_bounds.any().item()):
                            if int(offset) == 0:
                                row_logits = shift_logits[batch_idx[in_bounds], positions[in_bounds], :]
                            else:
                                branch_input_ids = inputs["input_ids"][batch_idx[in_bounds]].clone()
                                branch_first_label_pos = (score_shift_pos[in_bounds] + 1).to(dtype=torch.long)
                                for prefix_offset, prefix_token_id in enumerate(candidate_ids[: int(offset)]):
                                    branch_input_ids[
                                        torch.arange(branch_input_ids.size(0), device=labels.device),
                                        branch_first_label_pos + int(prefix_offset),
                                    ] = int(prefix_token_id)
                                branch_inputs: Dict[str, torch.Tensor] = {"input_ids": branch_input_ids}
                                attention_mask = inputs.get("attention_mask")
                                if attention_mask is not None:
                                    branch_inputs["attention_mask"] = attention_mask[batch_idx[in_bounds]]
                                branch_logits = model(**branch_inputs).logits[..., :-1, :]
                                row_logits = branch_logits[
                                    torch.arange(branch_logits.size(0), device=labels.device),
                                    positions[in_bounds],
                                    :,
                                ]
                            token_nll = torch.logsumexp(row_logits, dim=-1) - row_logits[:, int(token_id)]
                            candidate_nll[in_bounds] = candidate_nll[in_bounds] + token_nll
                    candidate_nlls.append(candidate_nll)
                score_nlls = torch.stack(candidate_nlls, dim=-1)
                smoothing_dist = self._score_smoothing_distribution(
                    valid_score_labels,
                    device=labels.device,
                    dtype=score_nlls.dtype,
                )
                prior_ce = (smoothing_dist * score_nlls).sum(dim=-1)

                hard_ce = torch.zeros_like(prior_ce)
                true_sequence_lengths = torch.zeros_like(score_shift_pos)
                for score_index, candidate_ids in enumerate(self.score_token_ids):
                    class_mask = valid_score_labels.eq(int(score_index))
                    if not bool(class_mask.any().item()):
                        continue
                    class_rows = batch_idx[class_mask]
                    class_positions = score_shift_pos[class_mask]
                    true_sequence_lengths[class_mask] = int(len(candidate_ids))
                    for offset in range(len(candidate_ids)):
                        hard_ce[class_mask] = hard_ce[class_mask] + token_loss[
                            class_rows,
                            class_positions + int(offset),
                        ]
                sample_alpha = alpha_t
                if self.smooth_adaptive_entropy:
                    score_probs = torch.softmax(-score_nlls.detach(), dim=-1)
                    entropy = -(score_probs * torch.log(score_probs.clamp_min(1e-12))).sum(dim=-1)
                    entropy = entropy / math.log(float(len(self.score_token_ids)))
                    sample_alpha = alpha_t * entropy.clamp(0.0, 1.0)
                mixed_ce = (1.0 - sample_alpha) * hard_ce + sample_alpha * prior_ce
                if local_has_score:
                    token_loss = token_loss.clone()
                    token_loss[batch_idx, score_shift_pos] = mixed_ce
                    for offset in range(1, max(len(ids) for ids in self.score_token_ids)):
                        has_token = true_sequence_lengths > int(offset)
                        if bool(has_token.any().item()):
                            token_loss[
                                batch_idx[has_token],
                                score_shift_pos[has_token] + int(offset),
                            ] = 0.0
                else:
                    distributed_dummy_loss = prior_ce.sum() * 0.0

            self.pointwise_seen += self._update_score_hist(score_labels)

        denom = loss_mask.sum().clamp_min(1)
        if (
            choice_source_lengths is not None
            and choice_target_distributions is not None
            and choice_candidate_token_ids is not None
            and bool(model.training)
        ):
            # Replace the hard target loss for tied-choice rows with a
            # sequence-level cross-entropy over the valid candidate outputs.
            # Candidate strings share the same protocol wrapper, so comparing
            # their summed sequence log-likelihoods is well-defined.
            row_numer = token_loss.sum(dim=1)
            row_denom = loss_mask.sum(dim=1).clamp_min(1).to(dtype=row_numer.dtype)
            source_lengths = choice_source_lengths.to(device=labels.device, dtype=torch.long)
            for row_idx, (distribution, candidates) in enumerate(
                zip(choice_target_distributions, choice_candidate_token_ids)
            ):
                if distribution is None or candidates is None or len(candidates) < 2:
                    continue
                if len(distribution) != len(candidates):
                    raise ValueError("choice distribution/candidate count mismatch in collated batch")
                source_len = int(source_lengths[row_idx].item())
                source_len = max(1, min(source_len, int(inputs["input_ids"].shape[1])))
                source_ids = inputs["input_ids"][row_idx, :source_len]
                candidate_nlls: List[torch.Tensor] = []
                candidate_probs: List[float] = []
                for candidate_ids, candidate_key in zip(candidates, distribution.keys()):
                    candidate = torch.tensor(candidate_ids, device=labels.device, dtype=torch.long)
                    full_ids = torch.cat([source_ids, candidate], dim=0).unsqueeze(0)
                    attention = torch.ones_like(full_ids, dtype=torch.long)
                    branch_logits = model(input_ids=full_ids, attention_mask=attention).logits[0]
                    shifted = branch_logits[:-1]
                    start = source_len - 1
                    end = start + int(candidate.numel())
                    token_logits = shifted[start:end]
                    if int(token_logits.shape[0]) != int(candidate.numel()):
                        raise RuntimeError("choice candidate logits are shorter than candidate target")
                    nll = F.cross_entropy(token_logits, candidate, reduction="sum")
                    candidate_nlls.append(nll)
                    candidate_probs.append(float(distribution[candidate_key]))
                if len(candidate_nlls) < 2:
                    continue
                scores = -torch.stack(candidate_nlls, dim=0)
                target = torch.tensor(candidate_probs, device=labels.device, dtype=scores.dtype)
                soft_loss = -(target * F.log_softmax(scores / float(self.choice_temperature), dim=0)).sum()
                row_numer[row_idx] = soft_loss * row_denom[row_idx]
                self.choice_soft_samples += 1
            loss = row_numer.sum() / row_denom.sum().clamp_min(1.0)
        else:
            loss = token_loss[loss_mask].sum() / denom
        if distributed_dummy_loss is not None:
            loss = loss + distributed_dummy_loss
        if self.smooth_trainable_alpha and self.smooth_alpha_reg > 0.0 and self.smooth_alpha_raw is not None:
            alpha_now = self._learned_smooth_alpha(device=loss.device, dtype=loss.dtype)
            alpha_init = torch.tensor(float(self.smooth_alpha_init), device=loss.device, dtype=loss.dtype)
            loss = loss + float(self.smooth_alpha_reg) * torch.square(alpha_now - alpha_init)
        if (
            pointwise_teacher_logits is not None
            and float(self.pointwise_distill_weight) > 0.0
            and bool(model.training)
        ):
            teacher_logits = pointwise_teacher_logits.to(device=labels.device, dtype=shift_logits.dtype)
            if int(teacher_logits.size(-1)) != len(self.score_token_ids):
                raise ValueError(
                    "pointwise_teacher_logits width must match score_token_ids length: "
                    f"{int(teacher_logits.size(-1))} != {len(self.score_token_ids)}"
                )
            if pointwise_teacher_mask is None:
                teacher_mask = torch.ones((labels.size(0),), device=labels.device, dtype=torch.bool)
            else:
                teacher_mask = pointwise_teacher_mask.to(device=labels.device, dtype=torch.bool)
            label_mask = labels.ne(IGNORE_INDEX)
            has_label = label_mask.any(dim=1)
            first_label_pos = label_mask.to(dtype=torch.int64).argmax(dim=1)
            valid_teacher = teacher_mask & has_label & (first_label_pos > 0)
            if pointwise_score_labels is not None:
                score_labels = pointwise_score_labels.to(device=labels.device, dtype=torch.long)
                valid_teacher = valid_teacher & (score_labels >= 0) & (score_labels < len(self.score_token_ids))
            if bool(valid_teacher.any().item()):
                batch_idx = torch.arange(labels.size(0), device=labels.device)[valid_teacher]
                score_shift_pos = first_label_pos[valid_teacher] - 1
                first_score_token_ids = torch.tensor(
                    [int(ids[0]) for ids in self.score_token_ids],
                    device=labels.device,
                    dtype=torch.long,
                )
                student_logits = shift_logits[batch_idx, score_shift_pos, :].index_select(
                    dim=-1,
                    index=first_score_token_ids,
                )
                teacher_logits_valid = teacher_logits[valid_teacher]
                temperature = float(self.pointwise_distill_temperature)
                distill_loss = F.kl_div(
                    F.log_softmax(student_logits / temperature, dim=-1),
                    F.softmax(teacher_logits_valid / temperature, dim=-1),
                    reduction="batchmean",
                ) * (temperature * temperature)
                loss = loss + float(self.pointwise_distill_weight) * distill_loss
        if (
            class_teacher_logits is not None
            and class_teacher_task_ids is not None
            and float(self.class_distill_weight) > 0.0
            and bool(self.class_distill_candidate_token_ids)
            and bool(model.training)
        ):
            teacher_logits_all = class_teacher_logits.to(device=labels.device, dtype=shift_logits.dtype)
            teacher_task_ids = class_teacher_task_ids.to(device=labels.device, dtype=torch.long)
            if class_teacher_mask is None:
                teacher_row_mask = torch.ones((labels.size(0),), device=labels.device, dtype=torch.bool)
            else:
                teacher_row_mask = class_teacher_mask.to(device=labels.device, dtype=torch.bool)
            if class_teacher_label_mask is None:
                teacher_label_mask = torch.ones_like(teacher_logits_all, dtype=torch.bool)
            else:
                teacher_label_mask = class_teacher_label_mask.to(device=labels.device, dtype=torch.bool)
            label_mask = labels.ne(IGNORE_INDEX)
            has_label = label_mask.any(dim=1)
            first_label_pos = label_mask.to(dtype=torch.int64).argmax(dim=1)
            total_valid = 0
            weighted_distill = torch.zeros((), device=loss.device, dtype=loss.dtype)
            for task_id, candidate_ids in self.class_distill_candidate_token_ids.items():
                candidate_ids_l = [int(x) for x in candidate_ids]
                if not candidate_ids_l:
                    continue
                width = int(len(candidate_ids_l))
                if int(teacher_logits_all.size(-1)) < width:
                    continue
                label_ok = teacher_label_mask[:, :width].all(dim=-1)
                offset = int(self.class_distill_label_offsets.get(int(task_id), 0))
                positions = first_label_pos + int(offset) - 1
                valid = (
                    teacher_row_mask
                    & teacher_task_ids.eq(int(task_id))
                    & label_ok
                    & has_label
                    & (positions >= 0)
                    & (positions < int(shift_logits.size(1)))
                )
                if not bool(valid.any().item()):
                    continue
                batch_idx = torch.arange(labels.size(0), device=labels.device)[valid]
                pos = positions[valid]
                token_index = torch.tensor(candidate_ids_l, device=labels.device, dtype=torch.long)
                student_logits = shift_logits[batch_idx, pos, :].index_select(dim=-1, index=token_index)
                teacher_logits_valid = teacher_logits_all[valid, :width]
                temperature = float(self.class_distill_temperature)
                task_loss = F.kl_div(
                    F.log_softmax(student_logits / temperature, dim=-1),
                    F.softmax(teacher_logits_valid / temperature, dim=-1),
                    reduction="batchmean",
                ) * (temperature * temperature)
                n_valid = int(batch_idx.numel())
                weighted_distill = weighted_distill + task_loss * float(n_valid)
                total_valid += int(n_valid)
            if total_valid > 0:
                loss = loss + float(self.class_distill_weight) * (weighted_distill / float(total_valid))
        return (loss, outputs) if return_outputs else loss

    def get_global_prior_smoothing_stats(self) -> Dict[str, Any]:
        hist = [float(x) for x in self.score_hist.tolist()]
        total = float(sum(hist))
        if total > 0:
            dist = [float(x / total) for x in hist]
        else:
            dist = [0.0 for _ in hist]
        learned_alpha = None
        if self.smooth_trainable_alpha and self.smooth_alpha_raw is not None:
            learned_alpha = float((torch.sigmoid(self.smooth_alpha_raw.detach().cpu()) * float(self.smooth_alpha_max)).item())
        return {
            "enabled": bool(self.smooth_alpha > 0),
            "mode": str(self.smooth_mode),
            "alpha": float(self.smooth_alpha),
            "trainable_alpha": bool(self.smooth_trainable_alpha),
            "alpha_init": float(self.smooth_alpha_init),
            "alpha_max": float(self.smooth_alpha_max),
            "alpha_reg": float(self.smooth_alpha_reg),
            "alpha_lr": float(self.smooth_alpha_lr),
            "alpha_final": learned_alpha if learned_alpha is not None else float(self.smooth_alpha),
            "start_step": int(self.smooth_start_step),
            "warmup_steps": int(self.smooth_warmup_steps),
            "start_pointwise_seen": int(self.smooth_start_pointwise_seen),
            "warmup_pointwise_seen": int(self.smooth_warmup_pointwise_seen),
            "pointwise_seen": int(self.pointwise_seen),
            "prior_smooth": float(self.smooth_prior),
            "uniform_mix": float(self.smooth_uniform_mix),
            "gaussian_sigma": float(self.smooth_gaussian_sigma),
            "adaptive_entropy": bool(self.smooth_adaptive_entropy),
            "freeze_prior": bool(self.smooth_freeze_prior),
            "score_seen": int(self.score_seen),
            "hist": hist,
            "distribution": dist,
            "score_token_ids": [[int(token_id) for token_id in ids] for ids in self.score_token_ids],
            "class_distill": {
                "enabled": bool(float(self.class_distill_weight) > 0.0 and bool(self.class_distill_candidate_token_ids)),
                "weight": float(self.class_distill_weight),
                "temperature": float(self.class_distill_temperature),
                "task_ids": sorted(int(k) for k in self.class_distill_candidate_token_ids.keys()),
            },
            "choice_soft_target": {
                "temperature": float(self.choice_temperature),
                "soft_samples_seen": int(self.choice_soft_samples),
            },
        }

def _parse_sft_pairwise_pred_label(text: str) -> Optional[int]:
    m = re.search(r"\[\[\s*([123ABC])\s*\]\]", str(text))
    if m is None:
        return None
    tok = str(m.group(1)).strip()
    if tok in {"1", "A"}:
        return int(LABEL_A)
    if tok in {"2", "B"}:
        return int(LABEL_B)
    return int(LABEL_TIE)


def _evaluate_pairwise_sft(
    *,
    model,
    tokenizer,
    examples: Sequence[PairwiseExample],
    max_length: int,
    batch_size: int,
    max_new_tokens: int = 8,
) -> Dict[str, Any]:
    n = int(len(examples))
    if n <= 0:
        return {"n": 0}

    true_labels: List[int] = []
    pred_labels: List[int] = []
    invalid_pred = 0

    model.eval()
    effective_bs = max(1, int(batch_size))

    with torch.no_grad():
        for start in range(0, n, effective_bs):
            batch = list(examples[start : start + effective_bs])
            prompts = [x.prompt for x in batch]

            tok = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=int(max_length),
            )

            if hasattr(model, "device"):
                dev = model.device
                tok = {k: v.to(dev) for k, v in tok.items()}

            gen = model.generate(
                **tok,
                max_new_tokens=int(max_new_tokens),
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

            prompt_len = tok["input_ids"].shape[1]
            for i, ex in enumerate(batch):
                pred_text = tokenizer.decode(gen[i, prompt_len:], skip_special_tokens=False)
                pred = _parse_sft_pairwise_pred_label(pred_text)
                if pred is None:
                    pred = -1
                    invalid_pred += 1

                true_labels.append(int(ex.label))
                pred_labels.append(int(pred))

    y_true = np.asarray(true_labels, dtype=np.int64)
    y_pred = np.asarray(pred_labels, dtype=np.int64)
    valid_pred = y_pred >= 0
    acc = float(((y_true == y_pred) & valid_pred).mean())
    tie_rate = float((y_pred == int(LABEL_TIE)).mean())

    return {
        "n": int(n),
        "sft_acc": acc,
        "sft_tie_rate": tie_rate,
        "sft_invalid_pred": int(invalid_pred),
        "sft_invalid_counted_as_wrong": True,
        "sft_confusion": _confusion(y_true, y_pred, num_classes=3),
        "sft_confusion_with_invalid": _confusion_with_invalid(y_true, y_pred, num_classes=3),
    }


def _confusion_with_invalid(true: np.ndarray, pred: np.ndarray, num_classes: int) -> List[List[int]]:
    conf = np.zeros((num_classes, num_classes + 1), dtype=np.int64)
    invalid_col = int(num_classes)
    for t, p in zip(true.tolist(), pred.tolist()):
        if not (0 <= int(t) < num_classes):
            continue
        col = int(p) if 0 <= int(p) < num_classes else invalid_col
        conf[int(t), col] += 1
    return conf.tolist()


def _confusion(true: np.ndarray, pred: np.ndarray, num_classes: int) -> List[List[int]]:
    conf = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(true.tolist(), pred.tolist()):
        if 0 <= int(t) < num_classes and 0 <= int(p) < num_classes:
            conf[int(t), int(p)] += 1
    return conf.tolist()


def _parse_sft_pointwise_pred_score(
    text: str,
    *,
    score_min: int,
    score_max: int,
) -> Optional[int]:
    text = str(text or "").strip()
    if not text:
        return None
    match = re.search(r"(-?\d{1,3})", text)
    if match is None:
        return None
    try:
        score = int(match.group(1))
    except Exception:
        return None
    if int(score) < int(score_min) or int(score) > int(score_max):
        return None
    return int(score)


def _evaluate_pointwise_sft(
    *,
    model,
    tokenizer,
    examples: Sequence[PointwiseScoredExample],
    max_length: int,
    batch_size: int,
    max_new_tokens: int,
    score_min: int,
    score_max: int,
) -> Dict[str, Any]:
    n = int(len(examples))
    if n <= 0:
        return {"n": 0}

    model.eval()
    true_scores: List[int] = []
    pred_scores: List[int] = []
    invalid_pred = 0

    with torch.no_grad():
        for start in range(0, n, int(batch_size)):
            batch = list(examples[start : start + int(batch_size)])
            prompts = [str(x.prompt) for x in batch]
            tok = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=int(max_length),
            )

            if hasattr(model, "device"):
                dev = model.device
            else:
                dev = next(model.parameters()).device
            tok = {k: v.to(dev) for k, v in tok.items()}
            gen = model.generate(
                **tok,
                do_sample=False,
                max_new_tokens=int(max_new_tokens),
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

            prompt_len = tok["input_ids"].shape[1]
            for i, ex in enumerate(batch):
                pred_text = tokenizer.decode(gen[i, prompt_len:], skip_special_tokens=False)
                pred_score = _parse_sft_pointwise_pred_score(
                    pred_text,
                    score_min=int(score_min),
                    score_max=int(score_max),
                )
                if pred_score is None:
                    pred_score = int(score_min)
                    invalid_pred += 1

                true_scores.append(int(ex.score))
                pred_scores.append(int(pred_score))

    y_true_score = np.asarray(true_scores, dtype=np.int64)
    y_pred_score = np.asarray(pred_scores, dtype=np.int64)
    y_true_label = y_true_score - int(score_min)
    y_pred_label = y_pred_score - int(score_min)

    acc = float((y_true_label == y_pred_label).mean())
    mae = float(np.mean(np.abs(y_pred_score - y_true_score)))
    rmse = float(np.sqrt(np.mean((y_pred_score - y_true_score) ** 2)))
    within1 = float((np.abs(y_pred_score - y_true_score) <= 1).mean())

    return {
        "n": int(n),
        "sft_acc": acc,
        "sft_mae": mae,
        "sft_rmse": rmse,
        "sft_within1": within1,
        "sft_within1_err": float(1.0 - within1),
        "sft_invalid_pred": int(invalid_pred),
        "sft_confusion": _confusion(
            y_true_label.astype(np.int64),
            y_pred_label.astype(np.int64),
            num_classes=int(score_max - score_min + 1),
        ),
    }


def _evaluate_pointwise(
    proxy: LlamaSharedMultiTaskProxyModel,
    examples: Sequence[Any],
    *,
    score_min: int,
) -> Dict[str, Any]:
    n = int(len(examples))
    if n <= 0:
        return {"n": 0}

    true_labels = np.asarray([int(x.label) for x in examples], dtype=np.int64)
    true_scores = np.asarray([int(x.score) for x in examples], dtype=np.int64)

    probs = proxy.predict_proba_pointwise(list(examples))
    pred_labels = probs.argmax(axis=1).astype(np.int64)
    pred_scores = (pred_labels + int(score_min)).astype(np.int64)

    acc = float((pred_labels == true_labels).mean())
    mae = float(np.mean(np.abs(pred_scores - true_scores)))
    rmse = float(np.sqrt(np.mean((pred_scores - true_scores) ** 2)))
    within1 = float((np.abs(pred_scores - true_scores) <= 1).mean())

    return {
        "n": int(n),
        "proxy_acc": acc,
        "proxy_mae": mae,
        "proxy_rmse": rmse,
        "proxy_within1": within1,
        "proxy_within1_err": float(1.0 - within1),
        "proxy_confusion": _confusion(true_labels, pred_labels, num_classes=10),
    }


def _evaluate_pairwise(proxy: LlamaSharedMultiTaskProxyModel, examples: Sequence[PairwiseExample]) -> Dict[str, Any]:
    n = int(len(examples))
    if n <= 0:
        return {"n": 0}

    true_labels = np.asarray([int(x.label) for x in examples], dtype=np.int64)
    probs = proxy.predict_proba_pairwise(list(examples))
    pred_labels = probs.argmax(axis=1).astype(np.int64)

    acc = float((pred_labels == true_labels).mean())
    tie_rate = float((pred_labels == int(LABEL_TIE)).mean())

    return {
        "n": int(n),
        "proxy_acc": acc,
        "proxy_tie_rate": tie_rate,
        "proxy_confusion": _confusion(true_labels, pred_labels, num_classes=3),
    }


def _train_pointwise_stage(
    *,
    proxy: LlamaSharedMultiTaskProxyModel,
    examples: Sequence[PointwiseScoredExample],
    epochs: int,
    batch_size: int,
    seed: int,
    stage_name: str,
) -> Dict[str, Any]:
    n = int(len(examples))
    if n <= 0:
        return {"n": 0, "epochs": int(epochs), "steps": 0, "elapsed_sec": 0.0}
    if int(batch_size) <= 0:
        raise ValueError("pointwise batch size must be > 0")

    rng = np.random.default_rng(int(seed))
    steps = 0
    t0 = time.time()

    for ep in range(int(epochs)):
        order = rng.permutation(n)
        step_in_ep = 0
        total_steps_ep = int((n + batch_size - 1) // batch_size)
        for start in range(0, n, int(batch_size)):
            idx = order[start : start + int(batch_size)]
            batch_inputs = [examples[int(i)] for i in idx.tolist()]
            batch_labels = [int(x.label) for x in batch_inputs]
            proxy.train_on_batch_pointwise(batch_inputs, batch_labels)
            steps += 1
            step_in_ep += 1

            if step_in_ep % 20 == 0 or step_in_ep == total_steps_ep:
                print(
                    f"[{stage_name}] epoch={ep + 1}/{epochs} step={step_in_ep}/{total_steps_ep}",
                    flush=True,
                )

    return {
        "n": int(n),
        "epochs": int(epochs),
        "steps": int(steps),
        "elapsed_sec": float(time.time() - t0),
    }


def _train_sft_pointwise(
    *,
    model_name_or_path: Optional[str] = None,
    model: Optional[Any] = None,
    tokenizer: Optional[Any] = None,
    pointwise_train: Sequence[PointwiseScoredExample],
    pointwise_val: Sequence[PointwiseScoredExample],
    epochs: int,
    per_device_batch_size: int,
    gradient_accumulation_steps: int,
    learning_rate: float,
    max_length: int,
    use_lora: bool,
    load_in_4bit: bool,
    seed: int,
    output_dir: str,
    fix_score_prefix_in_prompt: bool,
    score_min: int,
    score_max: int,
    global_smooth_alpha: float = 0.0,
    global_smooth_start_step: int = 0,
    global_smooth_warmup_steps: int = 0,
    global_smooth_prior: float = 1.0,
    global_smooth_trainable_alpha: bool = False,
    global_smooth_alpha_max: float = 0.2,
    global_smooth_alpha_reg: float = 0.0,
    global_smooth_alpha_lr: float = 0.0,
) -> Tuple[Dict[str, Any], Any, Any]:
    import transformers
    from peft import LoraConfig, PeftModel, TaskType, get_peft_model

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    reused_existing_model = model is not None and tokenizer is not None
    if not reused_existing_model:
        if not model_name_or_path:
            raise ValueError("model_name_or_path is required when model/tokenizer are not provided")
        print("Loading model and tokenizer for pointwise SFT...")
        model, tokenizer, _ = _load_sft_model_and_tokenizer(
            model_name_or_path=str(model_name_or_path),
            max_length=int(max_length),
            load_in_4bit=bool(load_in_4bit),
        )
    else:
        print("Reusing existing model and tokenizer for pointwise SFT stage...")
        tokenizer.model_max_length = int(max_length)

    assert model is not None
    assert tokenizer is not None

    model.gradient_checkpointing_enable()
    if hasattr(model, "config"):
        model.config.use_cache = False

    train_sources = [x.prompt for x in pointwise_train]
    train_targets = [
        _pointwise_sft_target(x, fix_score_prefix_in_prompt=bool(fix_score_prefix_in_prompt))
        for x in pointwise_train
    ]

    val_sources = None
    val_targets = None
    if pointwise_val:
        val_sources = [x.prompt for x in pointwise_val]
        val_targets = [
            _pointwise_sft_target(x, fix_score_prefix_in_prompt=bool(fix_score_prefix_in_prompt))
            for x in pointwise_val
        ]

    train_pointwise_score_labels = [int(x.score) - int(score_min) for x in pointwise_train]
    train_dataset = SFTPairwiseDataset(
        train_sources,
        train_targets,
        tokenizer,
        pointwise_score_labels=train_pointwise_score_labels,
    )
    eval_dataset = SFTPairwiseDataset(val_sources, val_targets, tokenizer) if val_sources else None

    if use_lora and not isinstance(model, PeftModel):
        model = _prepare_model_for_kbit_lora_sft(model, load_in_4bit=bool(load_in_4bit))
        print("Applying LoRA for pointwise SFT...")
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )
        model = get_peft_model(model, lora_config)
    elif use_lora:
        print("Model already has LoRA adapters; reusing them for pointwise SFT stage...")

    training_args = HFTrainingArguments(
        output_dir=output_dir,
        do_train=True,
        do_eval=bool(eval_dataset),
        per_device_train_batch_size=int(per_device_batch_size),
        per_device_eval_batch_size=int(per_device_batch_size),
        gradient_accumulation_steps=int(gradient_accumulation_steps),
        num_train_epochs=int(epochs),
        learning_rate=float(learning_rate),
        weight_decay=0.0,
        lr_scheduler_type="cosine",
        warmup_steps=max(
            1,
            int(
                len(train_dataset)
                * int(epochs)
                / (int(per_device_batch_size) * int(gradient_accumulation_steps))
                * 0.1
            ),
        ),
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        eval_strategy="epoch" if eval_dataset else "no",
        dataloader_pin_memory=True,
        gradient_checkpointing=True,
        remove_unused_columns=False,
        seed=int(seed),
    )

    smooth_enabled = float(global_smooth_alpha) > 0.0
    if smooth_enabled:
        print(
            "Pointwise SFT global-prior smoothing enabled: "
            f"alpha={float(global_smooth_alpha)} "
            f"start_step={int(global_smooth_start_step)} "
            f"warmup_steps={int(global_smooth_warmup_steps)} "
            f"prior={float(global_smooth_prior)} "
            f"trainable_alpha={bool(global_smooth_trainable_alpha)} "
            f"alpha_max={float(global_smooth_alpha_max)} "
            f"alpha_reg={float(global_smooth_alpha_reg)} "
            f"alpha_lr={float(global_smooth_alpha_lr)}",
            flush=True,
        )
    trainer = OnlineGlobalPriorSFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=_data_collator_sft,
        score_token_ids=_score_token_ids_for_sft(tokenizer, score_min=int(score_min), score_max=int(score_max)),
        smooth_alpha=float(global_smooth_alpha),
        smooth_start_step=int(global_smooth_start_step),
        smooth_warmup_steps=int(global_smooth_warmup_steps),
        smooth_prior=float(global_smooth_prior),
        smooth_trainable_alpha=bool(global_smooth_trainable_alpha) and bool(smooth_enabled),
        smooth_alpha_max=float(global_smooth_alpha_max),
        smooth_alpha_reg=float(global_smooth_alpha_reg),
        smooth_alpha_lr=float(global_smooth_alpha_lr),
    )

    print(f"Training pointwise SFT model with {len(train_dataset)} samples for {epochs} epochs...")
    t0 = time.time()
    trainer.train()
    elapsed = time.time() - t0

    if use_lora:
        model.save_pretrained(output_dir)
    else:
        trainer.save_model(output_dir)

    eval_metrics = _evaluate_pointwise_sft(
        model=model,
        tokenizer=tokenizer,
        examples=pointwise_val,
        max_length=int(max_length),
        batch_size=max(1, int(per_device_batch_size)),
        max_new_tokens=4,
        score_min=int(score_min),
        score_max=int(score_max),
    )

    parent_out = Path(output_dir).parent
    _write_json(parent_out / "metrics_pointwise_sft_val.json", eval_metrics)

    stats = {
        "mode": "sft_pointwise",
        "reused_existing_model": bool(reused_existing_model),
        "train_samples": len(train_dataset),
        "val_samples": len(eval_dataset) if eval_dataset else 0,
        "epochs": int(epochs),
        "elapsed_sec": elapsed,
        "global_prior_smoothing": (
            trainer.get_global_prior_smoothing_stats()
            if isinstance(trainer, OnlineGlobalPriorSFTTrainer)
            else {"enabled": False}
        ),
        "eval_pointwise": eval_metrics,
    }
    return stats, model, tokenizer


def _train_pairwise_stage(
    *,
    proxy: LlamaSharedMultiTaskProxyModel,
    examples: Sequence[PairwiseExample],
    epochs: int,
    batch_size: int,
    seed: int,
    stage_name: str,
) -> Dict[str, Any]:
    n = int(len(examples))
    if n <= 0:
        return {"n": 0, "epochs": int(epochs), "steps": 0, "elapsed_sec": 0.0}
    if int(batch_size) <= 0:
        raise ValueError("pairwise batch size must be > 0")

    rng = np.random.default_rng(int(seed))
    steps = 0
    t0 = time.time()

    for ep in range(int(epochs)):
        order = rng.permutation(n)
        step_in_ep = 0
        total_steps_ep = int((n + batch_size - 1) // batch_size)
        for start in range(0, n, int(batch_size)):
            idx = order[start : start + int(batch_size)]
            batch_inputs = [examples[int(i)] for i in idx.tolist()]
            batch_labels = [int(x.label) for x in batch_inputs]
            proxy.train_on_batch_pairwise(batch_inputs, batch_labels)
            steps += 1
            step_in_ep += 1

            if step_in_ep % 20 == 0 or step_in_ep == total_steps_ep:
                print(
                    f"[{stage_name}] epoch={ep + 1}/{epochs} step={step_in_ep}/{total_steps_ep}",
                    flush=True,
                )

    return {
        "n": int(n),
        "epochs": int(epochs),
        "steps": int(steps),
        "elapsed_sec": float(time.time() - t0),
    }


def _train_stage2_alternating(
    *,
    proxy: LlamaSharedMultiTaskProxyModel,
    pairwise_examples: Sequence[PairwiseExample],
    pointwise_examples: Sequence[PointwiseScoredExample],
    epochs: int,
    pairwise_batch_size: int,
    pointwise_batch_size: int,
    pointwise_replay_ratio: int,
    seed: int,
    stage_name: str,
) -> Dict[str, Any]:
    n_pair = int(len(pairwise_examples))
    n_point = int(len(pointwise_examples))
    if n_pair <= 0:
        return {
            "n_pairwise": 0,
            "n_pointwise_replay": int(n_point),
            "epochs": int(epochs),
            "pairwise_steps": 0,
            "pointwise_replay_steps": 0,
            "pointwise_replay_ratio": int(pointwise_replay_ratio),
            "elapsed_sec": 0.0,
        }
    if int(pairwise_batch_size) <= 0:
        raise ValueError("pairwise batch size must be > 0")
    if int(pointwise_batch_size) <= 0:
        raise ValueError("pointwise batch size must be > 0")
    if int(pointwise_replay_ratio) < 0:
        raise ValueError("pointwise replay ratio must be >= 0")

    rng = np.random.default_rng(int(seed))
    pairwise_steps = 0
    replay_steps = 0
    t0 = time.time()

    point_order = rng.permutation(n_point) if n_point > 0 else np.asarray([], dtype=np.int64)
    point_cursor = 0

    def _next_pointwise_batch_indices() -> np.ndarray:
        nonlocal point_order, point_cursor
        if n_point <= 0:
            return np.asarray([], dtype=np.int64)

        if point_cursor >= n_point:
            point_order = rng.permutation(n_point)
            point_cursor = 0

        end = point_cursor + int(pointwise_batch_size)
        if end <= n_point:
            idx = point_order[point_cursor:end]
            point_cursor = end
            return idx

        first = point_order[point_cursor:n_point]
        need = int(end - n_point)
        point_order = rng.permutation(n_point)
        point_cursor = int(need)
        second = point_order[0:need]
        return np.concatenate([first, second], axis=0)

    for ep in range(int(epochs)):
        order_pair = rng.permutation(n_pair)
        total_steps_ep = int((n_pair + int(pairwise_batch_size) - 1) // int(pairwise_batch_size))
        step_in_ep = 0

        for start in range(0, n_pair, int(pairwise_batch_size)):
            idx_pair = order_pair[start : start + int(pairwise_batch_size)]
            batch_pair_inputs = [pairwise_examples[int(i)] for i in idx_pair.tolist()]
            batch_pair_labels = [int(x.label) for x in batch_pair_inputs]
            proxy.train_on_batch_pairwise(batch_pair_inputs, batch_pair_labels)
            pairwise_steps += 1
            step_in_ep += 1

            for _ in range(int(pointwise_replay_ratio)):
                if n_point <= 0:
                    break
                idx_point = _next_pointwise_batch_indices()
                if idx_point.size <= 0:
                    break
                batch_point_inputs = [pointwise_examples[int(i)] for i in idx_point.tolist()]
                batch_point_labels = [int(x.label) for x in batch_point_inputs]
                proxy.train_on_batch_pointwise(batch_point_inputs, batch_point_labels)
                replay_steps += 1

            if step_in_ep % 20 == 0 or step_in_ep == total_steps_ep:
                print(
                    f"[{stage_name}] epoch={ep + 1}/{epochs} pair_step={step_in_ep}/{total_steps_ep} "
                    f"replay_ratio={pointwise_replay_ratio}",
                    flush=True,
                )

    return {
        "n_pairwise": int(n_pair),
        "n_pointwise_replay": int(n_point),
        "epochs": int(epochs),
        "pairwise_steps": int(pairwise_steps),
        "pointwise_replay_steps": int(replay_steps),
        "pointwise_replay_ratio": int(pointwise_replay_ratio),
        "elapsed_sec": float(time.time() - t0),
    }


def _run_pointwise_only_experiment(
    *,
    args: argparse.Namespace,
    cfg: RunConfig,
    base_out: Path,
    pointwise_train: Sequence[PointwiseScoredExample],
    pw_eval_split: Sequence[PointwiseScoredExample],
    pr_eval_split: Sequence[PairwiseExample],
    pw_eval_name: str,
    pr_eval_name: str,
    external_pw_eval_info: Optional[Dict[str, Any]],
    external_pr_eval_info: Optional[Dict[str, Any]],
    load_stats: Dict[str, Any],
    selected_stats: Dict[str, Any],
    candidate_pair_stats: Optional[Dict[str, Any]],
    split_info: Dict[str, Any],
    budget_info: Dict[str, Any],
) -> None:
    print("\n" + "=" * 80)
    print(f"Pointwise-only comparison mode: {cfg.pointwise_training_mode}")
    print("=" * 80)
    pointwise_class_weights = _compute_pointwise_class_weights(
        pointwise_train,
        num_labels=int(cfg.score_max - cfg.score_min + 1),
        mode=str(cfg.pointwise_class_weight_mode),
        strength=float(cfg.pointwise_class_weight_strength),
    )

    if str(cfg.pointwise_training_mode) == "proxy":
        proxy = LlamaSharedMultiTaskProxyModel(
            model_path=str(args.llama),
            pointwise_num_labels=int(cfg.score_max - cfg.score_min + 1),
            pairwise_num_labels=3,
            multitask_mode=str(cfg.llama_multitask_mode),
            lr=float(cfg.proxy_lr),
            weight_decay=0.0,
            max_length=int(cfg.proxy_max_length),
            finetune_mode="lora",
            gradient_checkpointing=True,
            load_in_4bit=bool(cfg.load_in_4bit),
            score_min=int(cfg.score_min),
            score_max=int(cfg.score_max),
            fix_score_prefix_in_prompt=bool(cfg.fix_score_prefix_in_prompt),
            pointwise_loss_type=str(cfg.pointwise_loss_type),
            pointwise_distance_weight=float(cfg.pointwise_distance_weight),
        )
        if pointwise_class_weights is not None:
            proxy.pointwise_class_weights = torch.tensor(
                pointwise_class_weights,
                dtype=torch.float32,
                device=proxy.device,
            )

        pw_before = _evaluate_pointwise(proxy, pw_eval_split, score_min=int(cfg.score_min))
        pr_before = _evaluate_pairwise(proxy, pr_eval_split)
        _write_json(base_out / "metrics_pointwise_before_stage1.json", pw_before)
        _write_json(base_out / "metrics_pairwise_before_stage1.json", pr_before)

        stage1_stats = _train_pointwise_stage(
            proxy=proxy,
            examples=pointwise_train,
            epochs=int(cfg.pointwise_epochs),
            batch_size=int(cfg.pointwise_batch_size),
            seed=int(cfg.seed) + 17,
            stage_name="stage1-pointwise-proxy",
        )
        _write_json(base_out / "train_stats_stage1_pointwise.json", stage1_stats)

        pw_after = _evaluate_pointwise(proxy, pw_eval_split, score_min=int(cfg.score_min))
        pr_after = _evaluate_pairwise(proxy, pr_eval_split)
        _write_json(base_out / "metrics_pointwise_after_stage1.json", pw_after)
        _write_json(base_out / "metrics_pairwise_after_stage1.json", pr_after)

        summary = {
            "mode": "pointwise_only",
            "pointwise_training_mode": "proxy",
            "pointwise_loss_type": str(cfg.pointwise_loss_type),
            "pointwise_distance_weight": float(cfg.pointwise_distance_weight),
            "eval_split": {
                "pointwise": str(pw_eval_name),
                "pairwise": str(pr_eval_name),
            },
            "external_fixed_eval": {
                "enabled": bool(cfg.use_external_fixed_eval),
                "pointwise": external_pw_eval_info,
                "pairwise": external_pr_eval_info,
            },
            "dataset_load_stats": load_stats,
            "selection_stats": selected_stats,
            "candidate_pair_selection": candidate_pair_stats,
            "pointwise_class_weight_mode": str(cfg.pointwise_class_weight_mode),
            "pointwise_class_weight_strength": float(cfg.pointwise_class_weight_strength),
            "pointwise_class_weights": pointwise_class_weights.tolist() if pointwise_class_weights is not None else None,
            "candidate_selector_target_task": (
                str(cfg.candidate_selector_target_task)
                if str(cfg.train_selection_mode) == "candidate_pair_selector"
                else None
            ),
            "split_by_question": split_info,
            "train_budget": budget_info,
            "pointwise_metrics": {
                "before_stage1": pw_before,
                "after_stage1": pw_after,
            },
            "pairwise_metrics": {
                "before_stage1": pr_before,
                "after_stage1": pr_after,
            },
            "train_stats": {
                "stage1_pointwise": stage1_stats,
            },
        }
        _write_json(base_out / "summary.json", summary)
        _print_compact_run_summary(base_out, _write_compact_metrics(base_out, summary))
        del proxy
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return

    if str(cfg.pointwise_training_mode) != "sft":
        raise ValueError(f"Unsupported pointwise_training_mode={cfg.pointwise_training_mode}")

    model, tokenizer, _ = _load_sft_model_and_tokenizer(
        model_name_or_path=str(args.llama),
        max_length=int(cfg.sft_max_length),
        load_in_4bit=bool(cfg.sft_load_in_4bit),
    )

    pw_before = _evaluate_pointwise_sft(
        model=model,
        tokenizer=tokenizer,
        examples=pw_eval_split,
        max_length=int(cfg.sft_max_length),
        batch_size=max(1, int(cfg.sft_per_device_batch_size)),
        max_new_tokens=4,
        score_min=int(cfg.score_min),
        score_max=int(cfg.score_max),
    )
    pr_before = _evaluate_pairwise_sft(
        model=model,
        tokenizer=tokenizer,
        examples=pr_eval_split,
        max_length=int(cfg.sft_max_length),
        batch_size=max(1, int(cfg.sft_per_device_batch_size)),
        max_new_tokens=8,
    )
    _write_json(base_out / "metrics_pointwise_before_stage1.json", pw_before)
    _write_json(base_out / "metrics_pairwise_before_stage1.json", pr_before)

    stage1_stats, model, tokenizer = _train_sft_pointwise(
        model_name_or_path=None,
        model=model,
        tokenizer=tokenizer,
        pointwise_train=pointwise_train,
        pointwise_val=pw_eval_split,
        epochs=int(cfg.pointwise_epochs),
        per_device_batch_size=int(cfg.sft_per_device_batch_size),
        gradient_accumulation_steps=int(cfg.sft_gradient_accumulation_steps),
        learning_rate=float(cfg.sft_lr),
        max_length=int(cfg.sft_max_length),
        use_lora=bool(cfg.sft_use_lora),
        load_in_4bit=bool(cfg.sft_load_in_4bit),
        seed=int(cfg.seed),
        output_dir=str(base_out / "sft_pointwise_model"),
        fix_score_prefix_in_prompt=bool(cfg.fix_score_prefix_in_prompt),
        score_min=int(cfg.score_min),
        score_max=int(cfg.score_max),
        global_smooth_alpha=float(cfg.sft_pointwise_global_smooth_alpha),
        global_smooth_start_step=int(cfg.sft_pointwise_global_smooth_start_step),
        global_smooth_warmup_steps=int(cfg.sft_pointwise_global_smooth_warmup_steps),
        global_smooth_prior=float(cfg.sft_pointwise_global_smooth_prior),
        global_smooth_trainable_alpha=bool(cfg.sft_pointwise_global_smooth_trainable_alpha),
        global_smooth_alpha_max=float(cfg.sft_pointwise_global_smooth_alpha_max),
        global_smooth_alpha_reg=float(cfg.sft_pointwise_global_smooth_alpha_reg),
        global_smooth_alpha_lr=float(cfg.sft_pointwise_global_smooth_alpha_lr),
    )
    _write_json(base_out / "train_stats_stage1_pointwise.json", stage1_stats)

    pw_after = _evaluate_pointwise_sft(
        model=model,
        tokenizer=tokenizer,
        examples=pw_eval_split,
        max_length=int(cfg.sft_max_length),
        batch_size=max(1, int(cfg.sft_per_device_batch_size)),
        max_new_tokens=4,
        score_min=int(cfg.score_min),
        score_max=int(cfg.score_max),
    )
    pr_after = _evaluate_pairwise_sft(
        model=model,
        tokenizer=tokenizer,
        examples=pr_eval_split,
        max_length=int(cfg.sft_max_length),
        batch_size=max(1, int(cfg.sft_per_device_batch_size)),
        max_new_tokens=8,
    )
    _write_json(base_out / "metrics_pointwise_after_stage1.json", pw_after)
    _write_json(base_out / "metrics_pairwise_after_stage1.json", pr_after)

    summary = {
        "mode": "pointwise_only",
        "pointwise_training_mode": "sft",
        "eval_split": {
            "pointwise": str(pw_eval_name),
            "pairwise": str(pr_eval_name),
        },
        "external_fixed_eval": {
            "enabled": bool(cfg.use_external_fixed_eval),
            "pointwise": external_pw_eval_info,
            "pairwise": external_pr_eval_info,
        },
        "dataset_load_stats": load_stats,
        "selection_stats": selected_stats,
        "candidate_pair_selection": candidate_pair_stats,
        "split_by_question": split_info,
        "train_budget": budget_info,
        "pointwise_metrics": {
            "before_stage1": pw_before,
            "after_stage1": pw_after,
        },
        "pairwise_metrics": {
            "before_stage1": pr_before,
            "after_stage1": pr_after,
        },
        "train_stats": {
            "stage1_pointwise": stage1_stats,
        },
    }
    _write_json(base_out / "summary.json", summary)
    _print_compact_run_summary(base_out, _write_compact_metrics(base_out, summary))
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pointwise(2 answers per question) -> pairwise training with stage-2 alternating replay.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--pointwise-5answers-dataset",
        required=True,
        help="Scored 5-answers pointwise JSON path (e.g., pointwise_5answers_score.json)",
    )
    parser.add_argument("--llama", required=True, help="Local HF model directory")
    parser.add_argument("--out", default=None, help="Output directory")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation ratio by question")
    parser.add_argument(
        "--val-split-seed",
        type=int,
        default=55,
        help="Seed for the internal held-out question split; fixed by default so pointwise validation questions do not change when --seed changes.",
    )
    parser.add_argument(
        "--pointwise-val-answer-seed",
        type=int,
        default=65,
        help="Seed for selecting one answer from each pointwise validation question; fixed by default so pointwise validation answers do not change when --seed changes.",
    )
    parser.add_argument(
        "--internal-val-mode",
        type=str,
        default="question_single_answer",
        choices=["question_single_answer", "selected_pair"],
        help=(
            "Internal validation construction. "
            "'question_single_answer' first splits raw questions, builds training pairs only from train questions, "
            "and uses one random answer per held-out question for pointwise validation. "
            "'selected_pair' keeps the legacy behavior that first selects one pair per question and then splits those pairs."
        ),
    )
    parser.add_argument(
        "--pair-selection-strategy",
        type=str,
        default="max_gap",
        choices=["max_gap", "random", "first_two"],
        help="Legacy per-question answer-pair rule used when --train-selection-mode=selected_pair.",
    )
    parser.add_argument(
        "--train-selection-mode",
        type=str,
        default="selected_pair",
        choices=["selected_pair", "candidate_pair_selector"],
        help=(
            "selected_pair keeps the old flow: pick one pair per question first, then apply budget. "
            "candidate_pair_selector enumerates all answer-pairs and lets a selector choose CandidatePair training units."
        ),
    )
    parser.add_argument(
        "--no-randomize-pair-order",
        action="store_true",
        help="Disable random A/B order for selected pairs",
    )
    parser.add_argument(
        "--budget-units",
        type=int,
        default=0,
        help=(
            "Training budget in pointwise answer units (2 answers per selected question). "
            "0 means use all available training questions."
        ),
    )
    parser.add_argument(
        "--budget-sampling-mode",
        type=str,
        default="choice",
        choices=["choice", "prefix"],
        help=(
            "How to apply --budget-units for selected_pair training. "
            "choice preserves the legacy independent sample; prefix uses one seeded permutation so smaller budgets nest in larger budgets."
        ),
    )
    parser.add_argument(
        "--candidate-selector-kind",
        type=str,
        default="shared_llama",
        choices=["shared_llama", "bert", "random", "distribution"],
        help="Selector used when --train-selection-mode=candidate_pair_selector.",
    )
    parser.add_argument(
        "--candidate-selector-init-pairs",
        type=int,
        default=200,
        help="Initial random CandidatePair units used to warm up the candidate selector.",
    )
    parser.add_argument(
        "--candidate-selector-batch-size",
        type=int,
        default=50,
        help="CandidatePair units selected and used to update the selector each round.",
    )
    parser.add_argument(
        "--candidate-selector-epochs",
        type=int,
        default=4,
        help="Epochs for the candidate selector head update.",
    )
    parser.add_argument(
        "--candidate-selector-buffer-maxlen",
        type=int,
        default=1000,
        help="Replay buffer size for SharedLlamaSelectorV2 in CandidatePair selection.",
    )
    parser.add_argument(
        "--candidate-selector-max-score-candidates",
        type=int,
        default=0,
        help=(
            "Max remaining CandidatePair units scored by the selector per round. "
            "0 means score all remaining candidates, i.e. 100%% of the current pool."
        ),
    )
    parser.add_argument(
        "--candidate-distribution-score-weight",
        type=float,
        default=1.0,
        help=(
            "For --candidate-selector-kind=distribution, weight for matching the queried true-score histogram "
            "to the proxy-predicted score distribution of the candidate pool. Unqueried true scores are not used."
        ),
    )
    parser.add_argument(
        "--candidate-distribution-dataset-weight",
        type=float,
        default=0.0,
        help=(
            "For --candidate-selector-kind=distribution, optional weight for matching selected dataset-source "
            "distribution to the full candidate pool distribution."
        ),
    )
    parser.add_argument(
        "--candidate-distribution-gap-weight",
        type=float,
        default=0.0,
        help=(
            "For --candidate-selector-kind=distribution, optional weight for matching the queried true-gap histogram "
            "to the proxy-predicted expected-score-gap distribution. Unqueried true gaps are not used."
        ),
    )
    parser.add_argument(
        "--candidate-selector-distribution-rank-weight",
        type=float,
        default=0.0,
        help=(
            "Hybrid rank-time weight for score-distribution matching when using a learned candidate selector. "
            "The bonus is computed from the pointwise proxy's predicted score distribution for unqueried candidates; 0 disables it."
        ),
    )
    parser.add_argument(
        "--candidate-selector-distribution-rank-top-k",
        type=int,
        default=2048,
        help=(
            "Only compute the hybrid distribution-rank bonus on the top-K learned-selector candidates per round. "
            "0 means all candidates in the scored pool."
        ),
    )
    parser.add_argument(
        "--candidate-selector-gap-weight",
        type=float,
        default=0.0,
        help=(
            "Extra selector-supervision weight for absolute score gap after a CandidatePair has been queried. "
            "It is not used for ranking unqueried candidates."
        ),
    )
    parser.add_argument(
        "--candidate-selector-score-bin-weight",
        type=float,
        default=0.0,
        help=(
            "Extra selector-supervision weight that rewards CandidatePairs covering score bins that are rare "
            "among already queried pointwise answers. Only used when target-task=pointwise."
        ),
    )
    parser.add_argument(
        "--candidate-selector-uncertainty-weight",
        type=float,
        default=0.5,
        help=(
            "Weight for the pointwise selector's uncertainty signal, defined as 1 - p(true_label). "
            "Only used when target-task=pointwise or when learning pairwise targets from proxy updates."
        ),
    )
    parser.add_argument(
        "--candidate-selector-kl-weight",
        type=float,
        default=0.5,
        help=(
            "Weight for the selector's KL-update signal after the proxy trains on a queried batch. "
            "Only used when target-task=pointwise or when learning pairwise targets from proxy updates."
        ),
    )
    parser.add_argument(
        "--no-candidate-selector-score-bin-weight",
        action="store_true",
        help="Force-disable candidate selector score-bin weighting by setting --candidate-selector-score-bin-weight=0.",
    )
    parser.add_argument(
        "--candidate-selector-target-task",
        type=str,
        default="pairwise",
        choices=["pairwise", "pointwise"],
        help=(
            "Which task defines the selector supervision signal. "
            "'pairwise' keeps the original behavior; 'pointwise' prefers pairs whose two answers are more useful "
            "for pointwise score prediction."
        ),
    )
    parser.add_argument(
        "--candidate-selector-allow-multiple-per-question",
        action="store_true",
        help="Allow selecting multiple CandidatePair units from the same original question.",
    )
    parser.add_argument(
        "--candidate-bert-selector-model",
        type=str,
        default="bert-base-uncased",
        help="HF encoder model used when --candidate-selector-kind=bert.",
    )
    parser.add_argument(
        "--candidate-bert-selector-max-length",
        type=int,
        default=512,
        help="Max token length for BERT CandidatePair selector inputs.",
    )
    parser.add_argument(
        "--candidate-bert-selector-unfreeze",
        action="store_true",
        help="Unfreeze BERT encoder parameters for CandidatePair selector training.",
    )
    parser.add_argument(
        "--candidate-bert-selector-unfreeze-last-n-layers",
        type=int,
        default=0,
        help="If BERT is unfrozen, only unfreeze the last N encoder layers. 0 keeps all encoder layers frozen.",
    )
    parser.add_argument(
        "--pointwise-fixed-val-ids-file",
        type=str,
        default="",
        help=(
            "Optional fixed pointwise validation id JSON. "
            "When empty, randomly split by question. "
            "When provided, split by source_id first (30K origin ids), and fallback to question_id only if source_id is unavailable."
        ),
    )
    parser.add_argument(
        "--pairwise-fixed-val-ids-file",
        type=str,
        default="",
        help=(
            "Optional fixed pairwise validation id JSON used by external fixed eval. "
            "When empty, the built-in default file is used only if --use-external-fixed-eval is enabled."
        ),
    )
    parser.add_argument(
        "--strict-fixed-val-ids",
        action="store_true",
        help="Fail if fixed val ids contain ids missing in the current train/eval source dataset.",
    )
    parser.add_argument(
        "--use-external-fixed-eval",
        action="store_true",
        help=(
            "Use external fixed eval splits from 30K_pointwise/30k_pairwise. "
            "When enabled, pointwise_5answers data is used only for training and no internal val is carved out."
        ),
    )
    parser.add_argument(
        "--external-pointwise-eval-dataset",
        type=str,
        default=_DEFAULT_EXTERNAL_POINTWISE_EVAL_DATASET,
        help="External pointwise dataset used to build pw_eval_split when --use-external-fixed-eval is enabled.",
    )
    parser.add_argument(
        "--external-pairwise-eval-dataset",
        type=str,
        default=_DEFAULT_EXTERNAL_PAIRWISE_EVAL_DATASET,
        help="External pairwise dataset used to build pr_eval_split when --use-external-fixed-eval is enabled.",
    )
    parser.add_argument(
        "--pairwise-abc-eval-dataset",
        type=str,
        default="",
        help=(
            "Optional pairwise eval-only dataset with modelA/outputA/modelB/outputB/modelC/outputC "
            "and choice_AB/choice_BC labels. choice 1 means Assistant 1 better, 2 means Assistant 2 better, "
            "and 3 means tie. This overrides only pairwise eval; converted pairwise training stays unchanged."
        ),
    )
    parser.add_argument(
        "--pairwise-abc-train-records",
        type=int,
        default=0,
        help=(
            "When --pairwise-abc-eval-dataset is set, hold out the remaining records for pairwise eval "
            "and add this many ABC records (2 pairs per record) to stage-2 pairwise training."
        ),
    )
    parser.add_argument(
        "--pairwise-abc-train-ratio",
        type=float,
        default=0.0,
        help=(
            "Alternative to --pairwise-abc-train-records: fraction of ABC records added to stage-2 "
            "pairwise training; remaining records are used for pairwise eval."
        ),
    )
    parser.add_argument(
        "--pairwise-abc-split-seed",
        type=int,
        default=-1,
        help="Seed for splitting --pairwise-abc-eval-dataset into optional train records and eval records. -1 uses --seed + 1019.",
    )

    parser.add_argument("--pointwise-epochs", type=int, default=1)
    parser.add_argument("--pairwise-epochs", type=int, default=1)
    parser.add_argument("--pointwise-batch-size", type=int, default=128)
    parser.add_argument("--pairwise-batch-size", type=int, default=64)
    parser.add_argument(
        "--stage2-pointwise-replay-ratio",
        type=int,
        default=1,
        help="During stage-2, run this many pointwise replay steps after each pairwise step.",
    )

    parser.add_argument("--score-min", type=int, default=1)
    parser.add_argument("--score-max", type=int, default=10)
    parser.add_argument(
        "--drop-tie-pairwise",
        action="store_true",
        help="Drop converted pairwise samples with tie(label=2)",
    )
    parser.add_argument(
        "--pairwise-order-augmentation",
        action="store_true",
        help="For converted pairwise training only, add a swapped A/B copy with A/B labels inverted.",
    )
    parser.add_argument(
        "--no-fix-score-prefix",
        action="store_true",
        help="Do not append 'Score: [' in pointwise prompt",
    )

    parser.add_argument("--proxy-lr", type=float, default=1e-4)
    parser.add_argument("--proxy-max-length", type=int, default=1024)
    parser.add_argument("--no-4bit", action="store_true", help="Disable 4-bit loading")
    parser.add_argument(
        "--llama-multitask-mode",
        type=str,
        default="lm_head",
        choices=["lm_head", "classifier_heads"],
        help="Multitask proxy mode",
    )
    parser.add_argument(
        "--pointwise-only",
        action="store_true",
        help="Stop after stage-1 pointwise training/eval. Useful for comparing proxy pointwise vs SFT pointwise.",
    )
    parser.add_argument(
        "--pointwise-training-mode",
        type=str,
        default="proxy",
        choices=["proxy", "sft"],
        help=(
            "Pointwise trainer used in --pointwise-only mode. In full runs, "
            "--pointwise-training-mode=sft together with --training-mode=sft enables "
            "score-SFT before pairwise-SFT."
        ),
    )
    parser.add_argument(
        "--pointwise-loss-type",
        type=str,
        default="ce",
        choices=["ce", "ce_mse", "ce_cost", "ordinal", "coral"],
        help="Pointwise loss type for proxy training.",
    )
    parser.add_argument(
        "--pointwise-distance-weight",
        type=float,
        default=0.1,
        help="Auxiliary distance-loss weight used by --pointwise-loss-type=ce_mse or ce_cost.",
    )
    parser.add_argument(
        "--pointwise-class-weight-mode",
        type=str,
        default="none",
        choices=["none", "inv_sqrt"],
        help="Optional class weighting mode for proxy pointwise training.",
    )
    parser.add_argument(
        "--no-pointwise-class-weight",
        action="store_true",
        help="Force-disable pointwise class weighting by overriding --pointwise-class-weight-mode to none.",
    )
    parser.add_argument(
        "--pointwise-class-weight-strength",
        type=float,
        default=1.0,
        help="Interpolation strength for class weights: 0 disables weighting effect, 1 uses the full computed weights.",
    )

    # SFT mode parameters
    parser.add_argument(
        "--training-mode",
        type=str,
        default="proxy",
        choices=["proxy", "sft"],
        help="Training mode: 'proxy' for multitask proxy training, 'sft' for supervised fine-tuning",
    )
    parser.add_argument("--sft-lr", type=float, default=1e-4, help="Learning rate for SFT training")
    parser.add_argument("--sft-per-device-batch-size", type=int, default=4, help="Batch size per device for SFT")
    parser.add_argument("--sft-gradient-accumulation-steps", type=int, default=2, help="Gradient accumulation steps for SFT")
    parser.add_argument("--sft-max-length", type=int, default=1024, help="Max sequence length for SFT")
    parser.add_argument("--sft-use-lora", action="store_true", help="Use LoRA for SFT training")
    parser.add_argument("--sft-4bit", action="store_true", help="Use 4-bit quantization for SFT")
    parser.add_argument(
        "--sft-stage2-mix-mode",
        type=str,
        default="replay",
        choices=["replay", "pair_batch", "pair_batch_both"],
        help=(
            "Stage-2 SFT mixing mode. replay appends pointwise replay samples globally; "
            "pair_batch builds ordered logical batches from selected pairs: 2 pointwise + 1 pairwise per pair; pair_batch_both uses 2 pointwise + 2 pairwise directions per pair."
        ),
    )
    parser.add_argument(
        "--sft-stage2-pairs-per-batch",
        type=int,
        default=4,
        help="For --sft-stage2-mix-mode pair_batch*, number of selected pairs per logical batch. pair_batch_both with 3 means 6 pointwise + 6 pairwise samples.",
    )
    parser.add_argument(
        "--sft-single-stage-pairbatch",
        action="store_true",
        help="Skip stage-1 pointwise SFT and train a single joint pairbatch SFT dataset directly from selected pairs.",
    )
    parser.add_argument(
        "--sft-pointwise-global-smooth-alpha",
        type=float,
        default=0.0,
        help="Mix weight for online global-prior score distribution smoothing on pointwise SFT score tokens. 0 disables it.",
    )
    parser.add_argument(
        "--sft-pointwise-global-smooth-start-step",
        type=int,
        default=0,
        help="Optimizer step at which pointwise global-prior smoothing starts.",
    )
    parser.add_argument(
        "--sft-pointwise-global-smooth-warmup-steps",
        type=int,
        default=0,
        help="Number of optimizer steps to linearly warm up global-prior smoothing alpha after start-step.",
    )
    parser.add_argument(
        "--sft-pointwise-global-smooth-prior",
        type=float,
        default=1.0,
        help="Laplace prior added to each score bin for online global-prior smoothing.",
    )
    parser.add_argument(
        "--sft-pointwise-global-smooth-trainable-alpha",
        action="store_true",
        help="Make global-prior smoothing alpha a bounded trainable scalar initialized from --sft-pointwise-global-smooth-alpha.",
    )
    parser.add_argument(
        "--sft-pointwise-global-smooth-alpha-max",
        type=float,
        default=0.2,
        help="Upper bound for trainable smoothing alpha: alpha = alpha_max * sigmoid(raw_alpha).",
    )
    parser.add_argument(
        "--sft-pointwise-global-smooth-alpha-reg",
        type=float,
        default=0.0,
        help="L2 regularization strength that keeps trainable alpha near its initialization.",
    )
    parser.add_argument(
        "--sft-pointwise-global-smooth-alpha-lr",
        type=float,
        default=0.0,
        help="Optional separate optimizer learning rate for trainable smoothing alpha. 0 uses the Trainer default lr.",
    )

    args = parser.parse_args()

    pointwise_fixed_val_ids_arg = str(args.pointwise_fixed_val_ids_file or "").strip()
    pairwise_fixed_val_ids_arg = str(args.pairwise_fixed_val_ids_file or "").strip()
    if bool(args.use_external_fixed_eval):
        if not pointwise_fixed_val_ids_arg:
            pointwise_fixed_val_ids_arg = _DEFAULT_POINTWISE_FIXED_VAL_IDS
        if not pairwise_fixed_val_ids_arg:
            pairwise_fixed_val_ids_arg = _DEFAULT_PAIRWISE_FIXED_VAL_IDS

    if bool(args.no_candidate_selector_score_bin_weight):
        args.candidate_selector_score_bin_weight = 0.0
    if bool(args.no_pointwise_class_weight):
        args.pointwise_class_weight_mode = "none"

    fixed_val_ids_path = _resolve_existing_path(pointwise_fixed_val_ids_arg)
    pairwise_fixed_val_ids_path = _resolve_existing_path(pairwise_fixed_val_ids_arg)
    external_pointwise_eval_dataset = _resolve_existing_path(args.external_pointwise_eval_dataset)
    external_pairwise_eval_dataset = _resolve_existing_path(args.external_pairwise_eval_dataset)
    pairwise_abc_eval_dataset = _resolve_existing_path(args.pairwise_abc_eval_dataset)

    cfg = RunConfig(
        seed=int(args.seed),
        val_ratio=float(args.val_ratio),
        val_split_seed=int(args.val_split_seed),
        pointwise_val_answer_seed=int(args.pointwise_val_answer_seed),
        internal_val_mode=str(args.internal_val_mode),
        pair_selection_strategy=str(args.pair_selection_strategy),
        train_selection_mode=str(args.train_selection_mode),
        randomize_pair_order=bool(not args.no_randomize_pair_order),
        budget_units=int(args.budget_units),
        budget_sampling_mode=str(args.budget_sampling_mode),
        candidate_selector_kind=str(args.candidate_selector_kind),
        candidate_selector_init_pairs=int(args.candidate_selector_init_pairs),
        candidate_selector_batch_size=int(args.candidate_selector_batch_size),
        candidate_selector_epochs=int(args.candidate_selector_epochs),
        candidate_selector_buffer_maxlen=int(args.candidate_selector_buffer_maxlen),
        candidate_selector_max_score_candidates=int(args.candidate_selector_max_score_candidates),
        candidate_distribution_score_weight=float(args.candidate_distribution_score_weight),
        candidate_distribution_dataset_weight=float(args.candidate_distribution_dataset_weight),
        candidate_distribution_gap_weight=float(args.candidate_distribution_gap_weight),
        candidate_selector_distribution_rank_weight=float(args.candidate_selector_distribution_rank_weight),
        candidate_selector_distribution_rank_top_k=int(args.candidate_selector_distribution_rank_top_k),
        candidate_selector_gap_weight=float(args.candidate_selector_gap_weight),
        candidate_selector_score_bin_weight=float(args.candidate_selector_score_bin_weight),
        candidate_selector_uncertainty_weight=float(args.candidate_selector_uncertainty_weight),
        candidate_selector_kl_weight=float(args.candidate_selector_kl_weight),
        candidate_selector_target_task=str(args.candidate_selector_target_task),
        candidate_selector_one_per_question=bool(not args.candidate_selector_allow_multiple_per_question),
        candidate_bert_selector_model=str(args.candidate_bert_selector_model),
        candidate_bert_selector_max_length=int(args.candidate_bert_selector_max_length),
        candidate_bert_selector_freeze=bool(not args.candidate_bert_selector_unfreeze),
        candidate_bert_selector_unfreeze_last_n_layers=int(args.candidate_bert_selector_unfreeze_last_n_layers),
        pointwise_fixed_val_ids_file=str(fixed_val_ids_path or ""),
        pairwise_fixed_val_ids_file=str(pairwise_fixed_val_ids_path or ""),
        strict_fixed_val_ids=bool(args.strict_fixed_val_ids),
        use_external_fixed_eval=bool(args.use_external_fixed_eval),
        external_pointwise_eval_dataset=str(external_pointwise_eval_dataset or ""),
        external_pairwise_eval_dataset=str(external_pairwise_eval_dataset or ""),
        pairwise_abc_eval_dataset=str(pairwise_abc_eval_dataset or ""),
        pairwise_abc_train_records=int(args.pairwise_abc_train_records),
        pairwise_abc_train_ratio=float(args.pairwise_abc_train_ratio),
        pairwise_abc_split_seed=(
            int(args.pairwise_abc_split_seed)
            if int(args.pairwise_abc_split_seed) >= 0
            else int(args.seed) + 1019
        ),
        pointwise_epochs=int(args.pointwise_epochs),
        pairwise_epochs=int(args.pairwise_epochs),
        pointwise_batch_size=int(args.pointwise_batch_size),
        pairwise_batch_size=int(args.pairwise_batch_size),
        stage2_pointwise_replay_ratio=int(args.stage2_pointwise_replay_ratio),
        score_min=int(args.score_min),
        score_max=int(args.score_max),
        drop_tie_pairwise=bool(args.drop_tie_pairwise),
        pairwise_order_augmentation=bool(args.pairwise_order_augmentation),
        fix_score_prefix_in_prompt=bool(not args.no_fix_score_prefix),
        proxy_lr=float(args.proxy_lr),
        proxy_max_length=int(args.proxy_max_length),
        load_in_4bit=bool(not args.no_4bit),
        llama_multitask_mode=str(args.llama_multitask_mode),
        pointwise_only=bool(args.pointwise_only),
        pointwise_training_mode=str(args.pointwise_training_mode),
        pointwise_loss_type=str(args.pointwise_loss_type),
        pointwise_distance_weight=float(args.pointwise_distance_weight),
        pointwise_class_weight_mode=str(args.pointwise_class_weight_mode),
        pointwise_class_weight_strength=float(args.pointwise_class_weight_strength),
        training_mode=str(args.training_mode),
        sft_lr=float(args.sft_lr),
        sft_per_device_batch_size=int(args.sft_per_device_batch_size),
        sft_gradient_accumulation_steps=int(args.sft_gradient_accumulation_steps),
        sft_max_length=int(args.sft_max_length),
        sft_use_lora=bool(args.sft_use_lora),
        sft_load_in_4bit=bool(args.sft_4bit),
        sft_stage2_mix_mode=str(args.sft_stage2_mix_mode),
        sft_stage2_pairs_per_batch=int(args.sft_stage2_pairs_per_batch),
        sft_single_stage_pairbatch=bool(args.sft_single_stage_pairbatch),
        sft_pointwise_global_smooth_alpha=float(args.sft_pointwise_global_smooth_alpha),
        sft_pointwise_global_smooth_start_step=int(args.sft_pointwise_global_smooth_start_step),
        sft_pointwise_global_smooth_warmup_steps=int(args.sft_pointwise_global_smooth_warmup_steps),
        sft_pointwise_global_smooth_prior=float(args.sft_pointwise_global_smooth_prior),
        sft_pointwise_global_smooth_trainable_alpha=bool(args.sft_pointwise_global_smooth_trainable_alpha),
        sft_pointwise_global_smooth_alpha_max=float(args.sft_pointwise_global_smooth_alpha_max),
        sft_pointwise_global_smooth_alpha_reg=float(args.sft_pointwise_global_smooth_alpha_reg),
        sft_pointwise_global_smooth_alpha_lr=float(args.sft_pointwise_global_smooth_alpha_lr),
    )

    if cfg.score_min >= cfg.score_max:
        raise ValueError("score-min must be < score-max")
    if not (0.0 <= cfg.val_ratio < 1.0):
        raise ValueError("val-ratio must be in [0, 1)")
    if int(cfg.stage2_pointwise_replay_ratio) < 0:
        raise ValueError("stage2-pointwise-replay-ratio must be >= 0")
    if str(cfg.sft_stage2_mix_mode) not in {"replay", "pair_batch", "pair_batch_both"}:
        raise ValueError("sft-stage2-mix-mode must be one of {'replay','pair_batch','pair_batch_both'}")
    if int(cfg.sft_stage2_pairs_per_batch) <= 0:
        raise ValueError("sft-stage2-pairs-per-batch must be > 0")
    if bool(cfg.sft_single_stage_pairbatch):
        if str(cfg.training_mode) != "sft" or str(cfg.pointwise_training_mode) != "sft":
            raise ValueError("--sft-single-stage-pairbatch requires --training-mode=sft and --pointwise-training-mode=sft")
        if str(cfg.sft_stage2_mix_mode) != "pair_batch_both":
            raise ValueError("--sft-single-stage-pairbatch currently requires --sft-stage2-mix-mode=pair_batch_both")
    if float(cfg.sft_pointwise_global_smooth_alpha) < 0.0 or float(cfg.sft_pointwise_global_smooth_alpha) > 1.0:
        raise ValueError("sft-pointwise-global-smooth-alpha must be in [0, 1]")
    if int(cfg.sft_pointwise_global_smooth_start_step) < 0:
        raise ValueError("sft-pointwise-global-smooth-start-step must be >= 0")
    if int(cfg.sft_pointwise_global_smooth_warmup_steps) < 0:
        raise ValueError("sft-pointwise-global-smooth-warmup-steps must be >= 0")
    if float(cfg.sft_pointwise_global_smooth_prior) <= 0.0:
        raise ValueError("sft-pointwise-global-smooth-prior must be > 0")
    if float(cfg.sft_pointwise_global_smooth_alpha_max) <= 0.0:
        raise ValueError("sft-pointwise-global-smooth-alpha-max must be > 0")
    if bool(cfg.sft_pointwise_global_smooth_trainable_alpha) and float(cfg.sft_pointwise_global_smooth_alpha) <= 0.0:
        raise ValueError("trainable smoothing alpha requires --sft-pointwise-global-smooth-alpha > 0")
    if float(cfg.sft_pointwise_global_smooth_alpha_reg) < 0.0:
        raise ValueError("sft-pointwise-global-smooth-alpha-reg must be >= 0")
    if float(cfg.sft_pointwise_global_smooth_alpha_lr) < 0.0:
        raise ValueError("sft-pointwise-global-smooth-alpha-lr must be >= 0")
    if int(cfg.pairwise_abc_train_records) < 0:
        raise ValueError("pairwise-abc-train-records must be >= 0")
    if not (0.0 <= float(cfg.pairwise_abc_train_ratio) < 1.0):
        raise ValueError("pairwise-abc-train-ratio must be in [0, 1)")
    if int(cfg.pairwise_abc_train_records) > 0 and float(cfg.pairwise_abc_train_ratio) > 0.0:
        raise ValueError("Use only one of --pairwise-abc-train-records or --pairwise-abc-train-ratio")
    if int(cfg.candidate_selector_init_pairs) <= 0:
        raise ValueError("candidate-selector-init-pairs must be > 0")
    if int(cfg.candidate_selector_batch_size) <= 0:
        raise ValueError("candidate-selector-batch-size must be > 0")
    if int(cfg.candidate_selector_epochs) <= 0:
        raise ValueError("candidate-selector-epochs must be > 0")
    if float(cfg.candidate_distribution_score_weight) < 0.0:
        raise ValueError("candidate-distribution-score-weight must be >= 0")
    if float(cfg.candidate_distribution_dataset_weight) < 0.0:
        raise ValueError("candidate-distribution-dataset-weight must be >= 0")
    if float(cfg.candidate_distribution_gap_weight) < 0.0:
        raise ValueError("candidate-distribution-gap-weight must be >= 0")
    if float(cfg.candidate_selector_distribution_rank_weight) < 0.0:
        raise ValueError("candidate-selector-distribution-rank-weight must be >= 0")
    if int(cfg.candidate_selector_distribution_rank_top_k) < 0:
        raise ValueError("candidate-selector-distribution-rank-top-k must be >= 0")
    if (
        float(cfg.candidate_selector_distribution_rank_weight) > 0.0
        and str(cfg.candidate_selector_target_task) != "pointwise"
    ):
        raise ValueError(
            "candidate-selector-distribution-rank-weight requires --candidate-selector-target-task pointwise"
        )
    if float(cfg.candidate_selector_gap_weight) < 0.0:
        raise ValueError("candidate-selector-gap-weight must be >= 0")
    if float(cfg.candidate_selector_score_bin_weight) < 0.0:
        raise ValueError("candidate-selector-score-bin-weight must be >= 0")
    if float(cfg.candidate_selector_uncertainty_weight) < 0.0:
        raise ValueError("candidate-selector-uncertainty-weight must be >= 0")
    if float(cfg.candidate_selector_kl_weight) < 0.0:
        raise ValueError("candidate-selector-kl-weight must be >= 0")
    if str(cfg.candidate_selector_target_task) not in {"pairwise", "pointwise"}:
        raise ValueError("candidate-selector-target-task must be one of {'pairwise','pointwise'}")
    if int(cfg.candidate_bert_selector_max_length) <= 0:
        raise ValueError("candidate-bert-selector-max-length must be > 0")
    if int(cfg.candidate_bert_selector_unfreeze_last_n_layers) < 0:
        raise ValueError("candidate-bert-selector-unfreeze-last-n-layers must be >= 0")
    if str(cfg.pointwise_class_weight_mode) not in {"none", "inv_sqrt"}:
        raise ValueError("pointwise-class-weight-mode must be one of {'none','inv_sqrt'}")
    if float(cfg.pointwise_class_weight_strength) < 0.0:
        raise ValueError("pointwise-class-weight-strength must be >= 0")
    if str(cfg.pointwise_loss_type) not in {"ce", "ce_mse", "ce_cost", "ordinal", "coral"}:
        raise ValueError("pointwise-loss-type must be one of {'ce','ce_mse','ce_cost','ordinal','coral'}")
    if float(cfg.pointwise_distance_weight) < 0.0:
        raise ValueError("pointwise-distance-weight must be >= 0")
    if (
        (not bool(cfg.pointwise_only))
        and str(cfg.pointwise_training_mode) == "sft"
        and str(cfg.training_mode) != "sft"
    ):
        raise ValueError("--pointwise-training-mode=sft requires --training-mode=sft unless --pointwise-only is set")
    if bool(cfg.use_external_fixed_eval):
        if not str(cfg.pointwise_fixed_val_ids_file).strip():
            raise ValueError("use-external-fixed-eval requires --pointwise-fixed-val-ids-file")
        if not str(cfg.pairwise_fixed_val_ids_file).strip():
            raise ValueError("use-external-fixed-eval requires --pairwise-fixed-val-ids-file")
        if not str(cfg.external_pointwise_eval_dataset).strip():
            raise ValueError("use-external-fixed-eval requires --external-pointwise-eval-dataset")
        if not str(cfg.external_pairwise_eval_dataset).strip():
            raise ValueError("use-external-fixed-eval requires --external-pairwise-eval-dataset")

    ds_path_raw = str(args.pointwise_5answers_dataset)
    ds_path = _resolve_existing_path(ds_path_raw)

    print("\n" + "=" * 80)
    print("Start run: pointwise(2 selected answers) -> alternating pairwise + pointwise replay")
    print("=" * 80)
    print(f"pointwise_5answers_dataset = {ds_path}")
    print(f"llama                     = {args.llama}")
    print(f"training_mode             = {cfg.training_mode}")
    print(f"pointwise_only            = {cfg.pointwise_only}")
    print(f"pointwise_training_mode   = {cfg.pointwise_training_mode}")
    print(f"pointwise_loss_type       = {cfg.pointwise_loss_type}")
    print(f"pointwise_distance_w      = {cfg.pointwise_distance_weight}")
    print(f"pointwise_class_weight    = {cfg.pointwise_class_weight_mode}")
    print(f"pointwise_class_weight_s  = {cfg.pointwise_class_weight_strength}")
    print(f"seed                      = {cfg.seed}")
    print(f"val_ratio                 = {cfg.val_ratio}")
    print(f"internal_val_mode         = {cfg.internal_val_mode}")
    print(f"train_selection_mode      = {cfg.train_selection_mode}")
    print(f"pair_selection_strategy   = {cfg.pair_selection_strategy}")
    print(f"randomize_pair_order      = {cfg.randomize_pair_order}")
    print(f"budget_units              = {cfg.budget_units}")
    print(f"budget_sampling_mode      = {cfg.budget_sampling_mode}")
    if str(cfg.train_selection_mode) == "candidate_pair_selector":
        print(f"candidate_selector_kind   = {cfg.candidate_selector_kind}")
        print(f"candidate_selector_init   = {cfg.candidate_selector_init_pairs}")
        print(f"candidate_selector_batch  = {cfg.candidate_selector_batch_size}")
        print(f"candidate_selector_gap_w  = {cfg.candidate_selector_gap_weight}")
        print(f"candidate_selector_bin_w  = {cfg.candidate_selector_score_bin_weight}")
        print(f"candidate_selector_dist_w = {cfg.candidate_selector_distribution_rank_weight}")
        print(f"candidate_selector_dist_k = {cfg.candidate_selector_distribution_rank_top_k}")
        print(f"candidate_selector_unc_w  = {cfg.candidate_selector_uncertainty_weight}")
        print(f"candidate_selector_kl_w   = {cfg.candidate_selector_kl_weight}")
        print(f"candidate_selector_target = {cfg.candidate_selector_target_task}")
        print(f"candidate_one_per_question= {cfg.candidate_selector_one_per_question}")
        if str(cfg.candidate_selector_kind) == "distribution":
            print(f"candidate_dist_score_w    = {cfg.candidate_distribution_score_weight}")
            print(f"candidate_dist_dataset_w  = {cfg.candidate_distribution_dataset_weight}")
            print(f"candidate_dist_gap_w      = {cfg.candidate_distribution_gap_weight}")
        if str(cfg.candidate_selector_kind) == "bert":
            print(f"candidate_bert_model      = {cfg.candidate_bert_selector_model}")
            print(f"candidate_bert_max_length = {cfg.candidate_bert_selector_max_length}")
            print(f"candidate_bert_freeze     = {cfg.candidate_bert_selector_freeze}")
    print(f"val_split_seed           = {cfg.val_split_seed}")
    print(f"pointwise_val_answer_seed= {cfg.pointwise_val_answer_seed}")
    print(f"pointwise_fixed_val_ids   = {cfg.pointwise_fixed_val_ids_file}")
    print(f"pairwise_fixed_val_ids    = {cfg.pairwise_fixed_val_ids_file}")
    print(f"strict_fixed_val_ids      = {cfg.strict_fixed_val_ids}")
    print(f"use_external_fixed_eval   = {cfg.use_external_fixed_eval}")
    if bool(cfg.use_external_fixed_eval):
        print(f"external_pw_eval_dataset  = {cfg.external_pointwise_eval_dataset}")
        print(f"external_pr_eval_dataset  = {cfg.external_pairwise_eval_dataset}")

    if bool(cfg.pointwise_only) and str(cfg.pointwise_training_mode) == "sft":
        print(f"pointwise_epochs          = {cfg.pointwise_epochs}")
        print(f"sft_lr                    = {cfg.sft_lr}")
        print(f"sft_per_device_batch_size = {cfg.sft_per_device_batch_size}")
        print(f"sft_gradient_accum_steps  = {cfg.sft_gradient_accumulation_steps}")
        print(f"sft_max_length            = {cfg.sft_max_length}")
        print(f"sft_use_lora              = {cfg.sft_use_lora}")
        print(f"sft_load_in_4bit          = {cfg.sft_load_in_4bit}")
        print(f"sft_stage2_mix_mode      = {cfg.sft_stage2_mix_mode}")
        print(f"sft_stage2_pairs_batch   = {cfg.sft_stage2_pairs_per_batch}")
        print(f"sft_single_stage_pb      = {cfg.sft_single_stage_pairbatch}")
        print(f"sft_pw_global_smooth_a   = {cfg.sft_pointwise_global_smooth_alpha}")
        print(f"sft_pw_global_smooth_st  = {cfg.sft_pointwise_global_smooth_start_step}")
        print(f"sft_pw_global_smooth_wu  = {cfg.sft_pointwise_global_smooth_warmup_steps}")
        print(f"sft_pw_global_smooth_tr  = {cfg.sft_pointwise_global_smooth_trainable_alpha}")
        print(f"sft_pw_global_smooth_amx = {cfg.sft_pointwise_global_smooth_alpha_max}")
        print(f"sft_pw_global_smooth_reg = {cfg.sft_pointwise_global_smooth_alpha_reg}")
        print(f"sft_pw_global_smooth_lr  = {cfg.sft_pointwise_global_smooth_alpha_lr}")
    elif str(cfg.training_mode) == "proxy":
        print(f"pointwise_epochs          = {cfg.pointwise_epochs}")
        print(f"pairwise_epochs           = {cfg.pairwise_epochs}")
        print(f"pointwise_batch_size      = {cfg.pointwise_batch_size}")
        print(f"pairwise_batch_size       = {cfg.pairwise_batch_size}")
        print(f"stage2_pointwise_replay   = {cfg.stage2_pointwise_replay_ratio}")
        print(f"proxy_lr                  = {cfg.proxy_lr}")
        print(f"proxy_max_length          = {cfg.proxy_max_length}")
        print(f"load_in_4bit              = {cfg.load_in_4bit}")
        print(f"llama_multitask_mode      = {cfg.llama_multitask_mode}")
    else:  # sft
        print(f"sft_lr                    = {cfg.sft_lr}")
        print(f"sft_per_device_batch_size = {cfg.sft_per_device_batch_size}")
        print(f"sft_gradient_accum_steps  = {cfg.sft_gradient_accumulation_steps}")
        print(f"sft_max_length            = {cfg.sft_max_length}")
        print(f"sft_use_lora              = {cfg.sft_use_lora}")
        print(f"sft_load_in_4bit          = {cfg.sft_load_in_4bit}")
        print(f"sft_stage2_mix_mode      = {cfg.sft_stage2_mix_mode}")
        print(f"sft_stage2_pairs_batch   = {cfg.sft_stage2_pairs_per_batch}")
        print(f"sft_single_stage_pb      = {cfg.sft_single_stage_pairbatch}")
        print(f"sft_pw_global_smooth_a   = {cfg.sft_pointwise_global_smooth_alpha}")
        print(f"sft_pw_global_smooth_st  = {cfg.sft_pointwise_global_smooth_start_step}")
        print(f"sft_pw_global_smooth_wu  = {cfg.sft_pointwise_global_smooth_warmup_steps}")
        print(f"sft_pw_global_smooth_tr  = {cfg.sft_pointwise_global_smooth_trainable_alpha}")
        print(f"sft_pw_global_smooth_amx = {cfg.sft_pointwise_global_smooth_alpha_max}")
        print(f"sft_pw_global_smooth_reg = {cfg.sft_pointwise_global_smooth_alpha_reg}")
        print(f"sft_pw_global_smooth_lr  = {cfg.sft_pointwise_global_smooth_alpha_lr}")
    print(f"drop_tie_pairwise         = {cfg.drop_tie_pairwise}")
    print(f"pairwise_order_aug        = {cfg.pairwise_order_augmentation}")

    _log_memory_usage("startup")

    base_out = (
        Path(args.out)
        if args.out
        else Path(__file__).resolve().parent
        / "outputs"
        / ("pointwise5answers_two_to_pairwise_v1_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    )
    base_out.mkdir(parents=True, exist_ok=True)
    print(f"output_dir                = {base_out}")

    os.environ.setdefault("PYTHONHASHSEED", str(cfg.seed))
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    _write_json(
        base_out / "config.json",
        {
            **asdict(cfg),
            "pointwise_5answers_dataset": str(ds_path),
            "pointwise_5answers_dataset_raw": str(ds_path_raw),
            "llama": str(args.llama),
            "judge_system_prompt": JUDGE_SYSTEM_PROMPT_SCORE_ONLY,
            "pairwise_system_prompt": DEFAULT_PAIRWISE_SYSTEM_PROMPT,
        },
    )

    # ---- Load dataset ----
    print("\nLoading scored 5-answers dataset...")
    questions, load_stats = _load_scored_questions(
        str(ds_path),
        score_min=int(cfg.score_min),
        score_max=int(cfg.score_max),
    )
    _write_json(base_out / "dataset_load_stats.json", load_stats)
    print(f"Loaded questions with >=2 valid answers: {len(questions)}")

    # ---- Split questions / select train pairs ----
    external_pw_eval_info: Optional[Dict[str, Any]] = None
    external_pr_eval_info: Optional[Dict[str, Any]] = None
    train_questions: List[Dict[str, Any]] = []
    val_questions: List[Dict[str, Any]] = []
    selected_pairs: List[SelectedQuestionPair] = []
    selected_rows: List[Dict[str, Any]] = []
    selected_stats: Dict[str, Any] = {}
    train_pairs_sel: List[SelectedQuestionPair] = []
    val_pairs_sel: List[SelectedQuestionPair] = []
    pointwise_val: List[PointwiseScoredExample] = []
    pointwise_val_rows: List[Dict[str, Any]] = []
    pointwise_val_stats: Optional[Dict[str, int]] = None

    if bool(cfg.use_external_fixed_eval):
        train_questions = list(questions)
        val_questions = []
        split_info = {
            "split_mode": "external_fixed_eval_all_train",
            "train_questions": int(len(train_questions)),
            "val_questions": 0,
            "internal_val_disabled": True,
            "internal_val_mode": str(cfg.internal_val_mode),
            "pointwise_val_mode": "external_fixed_eval",
            "pairwise_val_mode": "external_fixed_eval",
            "pointwise_fixed_val_ids_file": str(cfg.pointwise_fixed_val_ids_file),
            "pairwise_fixed_val_ids_file": str(cfg.pairwise_fixed_val_ids_file),
            "external_pointwise_eval_dataset": str(cfg.external_pointwise_eval_dataset),
            "external_pairwise_eval_dataset": str(cfg.external_pairwise_eval_dataset),
        }

        print("\nSelecting 2 answers per training question...")
        selected_pairs, selected_rows, selected_stats = _select_question_pairs(
            train_questions,
            strategy=str(cfg.pair_selection_strategy),
            randomize_order=bool(cfg.randomize_pair_order),
            seed=int(cfg.seed) + 7,
            budget_units=0,
        )
        selected_stats["requested_budget_units"] = int(cfg.budget_units)
        selected_stats["train_selection_mode"] = str(cfg.train_selection_mode)
        selected_stats["legacy_selected_pairs_used_for_training"] = bool(str(cfg.train_selection_mode) == "selected_pair")
        selected_stats["selection_scope"] = "all_questions_external_eval"
        selected_stats["held_out_val_questions"] = 0
        _write_json(base_out / "selected_pair_stats.json", selected_stats)
        _write_jsonl(base_out / "selected_pairs.jsonl", selected_rows)
        print(f"Selected training question pairs: {len(selected_pairs)}")
        train_pairs_sel = list(selected_pairs)
    elif str(cfg.internal_val_mode) == "selected_pair":
        print("\nSelecting 2 answers per question...")
        selected_pairs, selected_rows, selected_stats = _select_question_pairs(
            questions,
            strategy=str(cfg.pair_selection_strategy),
            randomize_order=bool(cfg.randomize_pair_order),
            seed=int(cfg.seed) + 7,
            budget_units=0,
        )
        selected_stats["requested_budget_units"] = int(cfg.budget_units)
        selected_stats["train_selection_mode"] = str(cfg.train_selection_mode)
        selected_stats["legacy_selected_pairs_used_for_training"] = bool(str(cfg.train_selection_mode) == "selected_pair")
        selected_stats["selection_scope"] = "all_questions_before_split"
        _write_json(base_out / "selected_pair_stats.json", selected_stats)
        _write_jsonl(base_out / "selected_pairs.jsonl", selected_rows)
        print(f"Selected question pairs: {len(selected_pairs)}")

        if str(cfg.pointwise_fixed_val_ids_file).strip():
            fixed_val_ids = _load_fixed_id_payload(str(cfg.pointwise_fixed_val_ids_file))
            train_pairs_sel, val_pairs_sel, split_info = _split_selected_pairs_by_fixed_ids(
                selected_pairs,
                val_ids=fixed_val_ids,
                fixed_ids_path=str(cfg.pointwise_fixed_val_ids_file),
                strict_missing=bool(cfg.strict_fixed_val_ids),
            )
        else:
            train_pairs_sel, val_pairs_sel, split_info = _split_selected_pairs(
                selected_pairs,
                seed=int(cfg.val_split_seed),
                val_ratio=float(cfg.val_ratio),
            )
        split_info["internal_val_mode"] = str(cfg.internal_val_mode)
        split_info["val_split_seed"] = int(cfg.val_split_seed)
        split_info["pointwise_val_answer_seed"] = int(cfg.pointwise_val_answer_seed)
        split_info["pointwise_val_mode"] = "two_answers_from_selected_pair"
        split_info["pairwise_val_mode"] = "selected_pair"
        train_questions = _filter_questions_by_selected_pair_ids(questions, train_pairs_sel)
        val_questions = _filter_questions_by_selected_pair_ids(questions, val_pairs_sel)
    else:
        if str(cfg.pointwise_fixed_val_ids_file).strip():
            fixed_val_ids = _load_fixed_id_payload(str(cfg.pointwise_fixed_val_ids_file))
            train_questions, val_questions, split_info = _split_questions_by_fixed_ids(
                questions,
                val_ids=fixed_val_ids,
                fixed_ids_path=str(cfg.pointwise_fixed_val_ids_file),
                strict_missing=bool(cfg.strict_fixed_val_ids),
            )
        else:
            train_questions, val_questions, split_info = _split_questions(
                questions,
                seed=int(cfg.val_split_seed),
                val_ratio=float(cfg.val_ratio),
            )
        split_info["internal_val_mode"] = str(cfg.internal_val_mode)
        split_info["val_split_seed"] = int(cfg.val_split_seed)
        split_info["pointwise_val_answer_seed"] = int(cfg.pointwise_val_answer_seed)
        split_info["pointwise_val_mode"] = "single_random_answer_per_question"
        split_info["pairwise_val_mode"] = "selected_pair_from_val_questions"

        print("\nSelecting 2 answers per training question...")
        selected_pairs, selected_rows, selected_stats = _select_question_pairs(
            train_questions,
            strategy=str(cfg.pair_selection_strategy),
            randomize_order=bool(cfg.randomize_pair_order),
            seed=int(cfg.seed) + 7,
            budget_units=0,
        )
        selected_stats["requested_budget_units"] = int(cfg.budget_units)
        selected_stats["train_selection_mode"] = str(cfg.train_selection_mode)
        selected_stats["legacy_selected_pairs_used_for_training"] = bool(str(cfg.train_selection_mode) == "selected_pair")
        selected_stats["selection_scope"] = "train_questions_after_split"
        selected_stats["held_out_val_questions"] = int(len(val_questions))
        _write_json(base_out / "selected_pair_stats.json", selected_stats)
        _write_jsonl(base_out / "selected_pairs.jsonl", selected_rows)
        print(f"Selected training question pairs: {len(selected_pairs)}")
        train_pairs_sel = list(selected_pairs)

        if val_questions:
            print("\nBuilding held-out validation views from val questions...")
            val_pairs_sel, val_pair_rows, val_selected_stats = _select_question_pairs(
                val_questions,
                strategy=str(cfg.pair_selection_strategy),
                randomize_order=bool(cfg.randomize_pair_order),
                seed=int(cfg.val_split_seed) + 11,
                budget_units=0,
            )
            val_selected_stats["selection_scope"] = "val_questions_for_pairwise_eval"
            _write_json(base_out / "val_selected_pair_stats.json", val_selected_stats)
            _write_jsonl(base_out / "val_selected_pairs.jsonl", val_pair_rows)

            pointwise_val, pointwise_val_rows, pointwise_val_stats = _build_single_answer_pointwise_eval_examples(
                val_questions,
                seed=int(cfg.pointwise_val_answer_seed),
                score_min=int(cfg.score_min),
                score_max=int(cfg.score_max),
                judge_system_prompt=JUDGE_SYSTEM_PROMPT_SCORE_ONLY,
                fix_score_prefix_in_prompt=bool(cfg.fix_score_prefix_in_prompt),
            )
            pointwise_val_stats["selection_seed"] = int(cfg.pointwise_val_answer_seed)
            _write_json(base_out / "pointwise_val_single_answer_stats.json", pointwise_val_stats)
            _write_jsonl(base_out / "pointwise_val_single_answer.jsonl", pointwise_val_rows)

    candidate_pair_stats: Optional[Dict[str, Any]] = None
    if str(cfg.train_selection_mode) == "candidate_pair_selector":
        if bool(cfg.use_external_fixed_eval) or str(cfg.internal_val_mode) == "question_single_answer":
            train_questions_pool = list(train_questions)
        else:
            train_questions_pool = _filter_questions_by_selected_pair_ids(questions, train_pairs_sel)
        candidate_examples, candidate_rows, candidate_pair_stats_raw = _build_candidate_pair_examples(
            train_questions_pool,
            pairwise_system_prompt=DEFAULT_PAIRWISE_SYSTEM_PROMPT,
            randomize_order=bool(cfg.randomize_pair_order),
            seed=int(cfg.seed) + 31,
        )
        _write_json(base_out / "candidate_pair_pool_stats.json", candidate_pair_stats_raw)
        _write_jsonl(base_out / "candidate_pair_pool.jsonl", candidate_rows)
        print(
            "CandidatePair pool built: "
            f"questions={candidate_pair_stats_raw['questions_with_candidate_pairs']} "
            f"candidate_pairs={candidate_pair_stats_raw['candidate_pairs']}"
        )

        if str(cfg.candidate_selector_kind) == "distribution":
            train_pairs_sel, selected_candidate_rows, candidate_pair_stats = _select_candidate_pairs_by_distribution(
                candidates=candidate_examples,
                cfg=cfg,
                llama_path=str(args.llama),
            )
        else:
            train_pairs_sel, selected_candidate_rows, candidate_pair_stats = _select_candidate_pairs_with_selector(
                candidates=candidate_examples,
                cfg=cfg,
                llama_path=str(args.llama),
                output_dir=base_out / "candidate_pair_selector",
            )
        _write_json(base_out / "candidate_pair_selection_stats.json", candidate_pair_stats)
        _write_jsonl(base_out / "selected_candidate_pairs.jsonl", selected_candidate_rows)
        budget_info = {
            "budget_units": int(cfg.budget_units),
            "train_pairs_before_budget": int(candidate_pair_stats_raw["candidate_pairs"]),
            "train_answers_before_budget": int(candidate_pair_stats_raw["candidate_pairs"] * 2),
            "budget_applied": bool(int(cfg.budget_units) > 0),
            "train_pairs_after_budget": int(len(train_pairs_sel)),
            "train_answers_after_budget": int(len(train_pairs_sel) * 2),
            "dropped_train_pairs_by_budget": int(
                max(0, int(candidate_pair_stats_raw["candidate_pairs"]) - int(len(train_pairs_sel)))
            ),
            "effective_budget_units": int(len(train_pairs_sel) * 2),
            "selection_mode": "candidate_pair_selector",
        }
        split_info["train_questions"] = int(len({int(p.source_id) for p in train_pairs_sel}))
    else:
        # Apply train-side budget after split so validation remains fixed/reproducible.
        train_pairs_sel, budget_info = _apply_budget_to_train_pairs(
            train_pairs_sel,
            budget_units=int(cfg.budget_units),
            seed=int(cfg.seed) + 19,
            sampling_mode=str(cfg.budget_sampling_mode),
        )

    _write_json(base_out / "split_by_question.json", split_info)
    _write_json(base_out / "train_budget.json", budget_info)
    val_qids = sorted({int(q["question_id"]) for q in val_questions})
    val_source_ids = sorted({int(q.get("source_id", q["question_id"])) for q in val_questions})
    _write_json(
        base_out / "val_ids_pointwise.json",
        {
            "type": "pointwise",
            "id_field": str(split_info.get("fixed_val_id_field", "question_id")),
            "ids": val_qids,
            "source_ids": val_source_ids,
            "n": int(len(val_qids)),
            "source_file": str(cfg.pointwise_fixed_val_ids_file) if val_questions else None,
            "use_external_fixed_eval": bool(cfg.use_external_fixed_eval),
        },
    )
    print(
        f"Split by question: train_questions={split_info['train_questions']} "
        f"val_questions={split_info['val_questions']}"
    )
    if bool(cfg.use_external_fixed_eval):
        print("External fixed eval enabled: internal val split disabled; training uses all selected questions.")
    elif int(split_info.get("fixed_val_ids_missing", 0)) > 0 and not bool(cfg.strict_fixed_val_ids):
        print(
            "Warning: fixed val ids partially matched current dataset: "
            f"id_field={split_info.get('fixed_val_id_field', 'unknown')} "
            f"matched={split_info.get('fixed_val_ids_matched', 0)} "
            f"missing={split_info.get('fixed_val_ids_missing', 0)}"
        )
    if bool(budget_info.get("budget_applied", False)):
        print(
            "Train budget applied: "
            f"answers={budget_info['train_answers_after_budget']} "
            f"questions={budget_info['train_pairs_after_budget']}"
        )
    if candidate_pair_stats is not None:
        print(
            "CandidatePair selector selected: "
            f"pairs={candidate_pair_stats.get('selected_pairs', 0)} "
            f"answers={candidate_pair_stats.get('selected_answers', 0)}"
        )
    if pointwise_val_stats is not None:
        print(
            "Held-out pointwise val built: "
            f"questions={pointwise_val_stats.get('questions_with_answers', 0)} "
            f"answers={pointwise_val_stats.get('selected_answers', 0)}"
        )

    # ---- Build pointwise train/val ----
    pointwise_train = _build_pointwise_examples(
        train_pairs_sel,
        score_min=int(cfg.score_min),
        score_max=int(cfg.score_max),
        judge_system_prompt=JUDGE_SYSTEM_PROMPT_SCORE_ONLY,
        fix_score_prefix_in_prompt=bool(cfg.fix_score_prefix_in_prompt),
    )
    if str(cfg.internal_val_mode) == "selected_pair":
        pointwise_val = _build_pointwise_examples(
            val_pairs_sel,
            score_min=int(cfg.score_min),
            score_max=int(cfg.score_max),
            judge_system_prompt=JUDGE_SYSTEM_PROMPT_SCORE_ONLY,
            fix_score_prefix_in_prompt=bool(cfg.fix_score_prefix_in_prompt),
        )
    if not pointwise_train:
        raise RuntimeError("pointwise_train is empty")

    # ---- Build pairwise train/val ----
    pairwise_train, pairwise_train_rows, pairwise_train_stats = _build_pairwise_examples(
        train_pairs_sel,
        pairwise_system_prompt=DEFAULT_PAIRWISE_SYSTEM_PROMPT,
        drop_tie=bool(cfg.drop_tie_pairwise),
        order_augmentation=bool(cfg.pairwise_order_augmentation),
    )
    pairwise_val, pairwise_val_rows, pairwise_val_stats = _build_pairwise_examples(
        val_pairs_sel,
        pairwise_system_prompt=DEFAULT_PAIRWISE_SYSTEM_PROMPT,
        drop_tie=bool(cfg.drop_tie_pairwise),
        order_augmentation=False,
    )

    _write_json(base_out / "pairwise_train_stats.json", pairwise_train_stats)
    _write_json(base_out / "pairwise_val_stats.json", pairwise_val_stats)
    _write_jsonl(base_out / "pairwise_train.jsonl", pairwise_train_rows)
    _write_jsonl(base_out / "pairwise_val.jsonl", pairwise_val_rows)

    print(
        "Converted pairwise samples: "
        f"train={len(pairwise_train)} val={len(pairwise_val)} "
        f"(drop_tie={cfg.drop_tie_pairwise}, order_aug={cfg.pairwise_order_augmentation})"
    )

    pairwise_abc_eval_info: Optional[Dict[str, Any]] = None
    pairwise_abc_train_info: Optional[Dict[str, Any]] = None
    pairwise_abc_split_info: Optional[Dict[str, Any]] = None
    pairwise_abc_eval_rows: List[Dict[str, Any]] = []
    use_pairwise_abc_train = (
        bool(str(cfg.pairwise_abc_eval_dataset).strip())
        and (int(cfg.pairwise_abc_train_records) > 0 or float(cfg.pairwise_abc_train_ratio) > 0.0)
    )
    if bool(use_pairwise_abc_train):
        print("\nSplitting pairwise ABC dataset into train/eval...")
        (
            pairwise_abc_train,
            pairwise_abc_train_rows,
            pairwise_abc_train_info,
            pairwise_abc_eval,
            pairwise_abc_eval_rows,
            pairwise_abc_eval_info,
            pairwise_abc_split_info,
        ) = _split_pairwise_abc_dataset(
            str(cfg.pairwise_abc_eval_dataset),
            train_records=int(cfg.pairwise_abc_train_records),
            train_ratio=float(cfg.pairwise_abc_train_ratio),
            seed=int(cfg.pairwise_abc_split_seed),
            pairwise_system_prompt=DEFAULT_PAIRWISE_SYSTEM_PROMPT,
        )
        original_pairwise_train_n = int(len(pairwise_train))
        remapped_abc_train: List[PairwiseExample] = []
        remapped_abc_rows: List[Dict[str, Any]] = []
        for i, ex in enumerate(pairwise_abc_train, start=1):
            new_pair_id = int(original_pairwise_train_n + i)
            remapped_abc_train.append(
                PairwiseExample(
                    id=int(new_pair_id),
                    dataset=str(ex.dataset),
                    group_id=int(ex.group_id),
                    pair_id=int(new_pair_id),
                    model_a=str(ex.model_a),
                    model_b=str(ex.model_b),
                    prompt=str(ex.prompt),
                    label=int(ex.label),
                )
            )
        for i, row in enumerate(pairwise_abc_train_rows, start=1):
            new_row = dict(row)
            new_pair_id = int(original_pairwise_train_n + i)
            new_row["pair_id"] = int(new_pair_id)
            new_row["abc_train_extra"] = True
            remapped_abc_rows.append(new_row)

        pairwise_train = list(pairwise_train) + remapped_abc_train
        pairwise_train_rows = list(pairwise_train_rows) + remapped_abc_rows
        pairwise_train_stats = {
            **pairwise_train_stats,
            "generated_pairs": int(len(pairwise_train)),
            "converted_generated_pairs": int(original_pairwise_train_n),
            "abc_extra_generated_pairs": int(len(remapped_abc_train)),
            "abc_extra_records": int(pairwise_abc_split_info.get("train_records", 0)),
            "label_A": int(pairwise_train_stats.get("label_A", 0) + pairwise_abc_train_info.get("label_A", 0)),
            "label_B": int(pairwise_train_stats.get("label_B", 0) + pairwise_abc_train_info.get("label_B", 0)),
            "label_C": int(pairwise_train_stats.get("label_C", 0) + pairwise_abc_train_info.get("label_C", 0)),
        }
        pr_eval_split = pairwise_abc_eval
        pr_eval_name = "pairwise_abc_eval_holdout"
        _write_json(base_out / "pairwise_train_stats.json", pairwise_train_stats)
        _write_jsonl(base_out / "pairwise_train.jsonl", pairwise_train_rows)
        _write_json(base_out / "pairwise_abc_train_stats.json", pairwise_abc_train_info)
        _write_jsonl(base_out / "pairwise_abc_train.jsonl", pairwise_abc_train_rows)
        _write_json(base_out / "pairwise_abc_eval_stats.json", pairwise_abc_eval_info)
        _write_jsonl(base_out / "pairwise_abc_eval.jsonl", pairwise_abc_eval_rows)
        _write_json(base_out / "pairwise_abc_split.json", pairwise_abc_split_info)
        print(
            "Pairwise ABC split loaded: "
            f"train_records={pairwise_abc_split_info.get('train_records', 0)} "
            f"train_pairs={len(pairwise_abc_train)} "
            f"eval_records={pairwise_abc_split_info.get('eval_records', 0)} "
            f"eval_pairs={len(pr_eval_split)}"
        )

    if bool(cfg.use_external_fixed_eval):
        print("\nLoading external fixed eval splits...")
        pw_eval_split, pr_eval_split, external_pw_eval_info, external_pr_eval_info = _load_external_fixed_eval_splits(
            pointwise_dataset_path=str(cfg.external_pointwise_eval_dataset),
            pairwise_dataset_path=str(cfg.external_pairwise_eval_dataset),
            pointwise_fixed_ids_path=str(cfg.pointwise_fixed_val_ids_file),
            pairwise_fixed_ids_path=str(cfg.pairwise_fixed_val_ids_file),
            score_min=int(cfg.score_min),
            score_max=int(cfg.score_max),
            fix_score_prefix_in_prompt=bool(cfg.fix_score_prefix_in_prompt),
            strict_missing=bool(cfg.strict_fixed_val_ids),
        )
        pw_eval_name = "external_fixed_eval"
        pr_eval_name = "external_fixed_eval"
        _write_json(base_out / "external_eval_pointwise.json", external_pw_eval_info)
        _write_json(base_out / "external_eval_pairwise.json", external_pr_eval_info)
        print(
            "External eval loaded: "
            f"pointwise={len(pw_eval_split)} pairwise={len(pr_eval_split)}"
        )
    else:
        pw_eval_name = "val" if pointwise_val else "train"
        pw_eval_split = pointwise_val if pointwise_val else pointwise_train
        if bool(use_pairwise_abc_train):
            pass
        elif pairwise_val:
            pr_eval_name = "val"
            pr_eval_split = pairwise_val
        elif str(cfg.internal_val_mode) == "question_single_answer":
            pr_eval_name = "val_unavailable"
            pr_eval_split = []
        else:
            pr_eval_name = "train"
            pr_eval_split = pairwise_train

    if str(cfg.pairwise_abc_eval_dataset).strip() and not bool(use_pairwise_abc_train):
        print("\nLoading pairwise ABC eval split...")
        pr_eval_split, pairwise_abc_eval_rows, pairwise_abc_eval_info = _load_pairwise_abc_eval_dataset(
            str(cfg.pairwise_abc_eval_dataset),
            pairwise_system_prompt=DEFAULT_PAIRWISE_SYSTEM_PROMPT,
        )
        pr_eval_name = "pairwise_abc_eval"
        _write_json(base_out / "pairwise_abc_eval_stats.json", pairwise_abc_eval_info)
        _write_jsonl(base_out / "pairwise_abc_eval.jsonl", pairwise_abc_eval_rows)
        print(
            "Pairwise ABC eval loaded: "
            f"pairs={len(pr_eval_split)} "
            f"A={pairwise_abc_eval_info.get('label_A', 0)} "
            f"B={pairwise_abc_eval_info.get('label_B', 0)} "
            f"C={pairwise_abc_eval_info.get('label_C', 0)}"
        )

    sft_pairwise_eval_split = pr_eval_split if (bool(cfg.use_external_fixed_eval) or pairwise_abc_eval_info is not None) else pairwise_val

    _write_json(
        base_out / "split_pointwise.json",
        {
            "train_size": int(len(pointwise_train)),
            "val_size": int(len(pointwise_val)),
            "eval_size": int(len(pw_eval_split)),
            "eval_name": str(pw_eval_name),
            "train_questions": int(len(train_pairs_sel)),
            "val_questions": int(len(val_pairs_sel)),
            "split_mode": str(split_info.get("split_mode", "random_by_question")),
            "internal_val_mode": str(cfg.internal_val_mode),
            "pointwise_val_mode": str(split_info.get("pointwise_val_mode", "selected_pair")),
            "pointwise_fixed_val_ids_file": str(cfg.pointwise_fixed_val_ids_file),
            "use_external_fixed_eval": bool(cfg.use_external_fixed_eval),
            "external_eval_dataset": (
                str(external_pw_eval_info.get("dataset_path", "")) if external_pw_eval_info is not None else None
            ),
            "budget_units": int(cfg.budget_units),
            "effective_budget_units": int(budget_info.get("effective_budget_units", 0)),
        },
    )
    _write_json(
        base_out / "split_pairwise.json",
        {
            "train_size": int(len(pairwise_train)),
            "val_size": int(len(pairwise_val)),
            "eval_size": int(len(pr_eval_split)),
            "eval_name": str(pr_eval_name),
            "split_mode": str(split_info.get("split_mode", "random_by_question")),
            "internal_val_mode": str(cfg.internal_val_mode),
            "pairwise_val_mode": str(split_info.get("pairwise_val_mode", "selected_pair")),
            "pairwise_fixed_val_ids_file": str(cfg.pairwise_fixed_val_ids_file),
            "use_external_fixed_eval": bool(cfg.use_external_fixed_eval),
            "external_eval_dataset": (
                str(external_pr_eval_info.get("dataset_path", "")) if external_pr_eval_info is not None else None
            ),
            "pairwise_abc_eval_dataset": str(cfg.pairwise_abc_eval_dataset),
            "pairwise_abc_train": pairwise_abc_train_info,
            "pairwise_abc_eval": pairwise_abc_eval_info,
            "pairwise_abc_split": pairwise_abc_split_info,
            "pairwise_query_unit": (
                str(external_pr_eval_info.get("query_unit", "example")) if external_pr_eval_info is not None else "example"
            ),
            "budget_units": int(cfg.budget_units),
            "effective_budget_units": int(budget_info.get("effective_budget_units", 0)),
        },
    )

    if bool(cfg.pointwise_only):
        _run_pointwise_only_experiment(
            args=args,
            cfg=cfg,
            base_out=base_out,
            pointwise_train=pointwise_train,
            pw_eval_split=pw_eval_split,
            pr_eval_split=pr_eval_split,
            pw_eval_name=pw_eval_name,
            pr_eval_name=pr_eval_name,
            external_pw_eval_info=external_pw_eval_info,
            external_pr_eval_info=external_pr_eval_info,
            load_stats=load_stats,
            selected_stats=selected_stats,
            candidate_pair_stats=candidate_pair_stats,
            split_info=split_info,
            budget_info=budget_info,
        )
        _log_memory_usage("finished")
        return

    if (
        str(cfg.training_mode) == "sft"
        and str(cfg.pointwise_training_mode) == "sft"
        and bool(cfg.sft_single_stage_pairbatch)
    ):
        print("\n" + "=" * 80)
        print("Using single-stage SFT pairbatch: pointwise + pairwise joint batches")
        print("=" * 80)

        model, tokenizer, _ = _load_sft_model_and_tokenizer(
            model_name_or_path=str(args.llama),
            max_length=int(cfg.sft_max_length),
            load_in_4bit=bool(cfg.sft_load_in_4bit),
        )

        print("\nEvaluating before single-stage pairbatch SFT...")
        pw_before = _evaluate_pointwise_sft(
            model=model,
            tokenizer=tokenizer,
            examples=pw_eval_split,
            max_length=int(cfg.sft_max_length),
            batch_size=max(1, int(cfg.sft_per_device_batch_size)),
            max_new_tokens=4,
            score_min=int(cfg.score_min),
            score_max=int(cfg.score_max),
        )
        pr_before = _evaluate_pairwise_sft(
            model=model,
            tokenizer=tokenizer,
            examples=pr_eval_split,
            max_length=int(cfg.sft_max_length),
            batch_size=max(1, int(cfg.sft_per_device_batch_size)),
            max_new_tokens=8,
        )
        _write_json(base_out / "metrics_pointwise_before_stage1.json", pw_before)
        _write_json(base_out / "metrics_pairwise_before_stage1.json", pr_before)

        print("\nSingle-stage joint pairbatch SFT...")
        single_stats, model, tokenizer = _train_sft_pairwise(
            model_name_or_path=None,
            model=model,
            tokenizer=tokenizer,
            pairwise_train=pairwise_train,
            pairwise_val=sft_pairwise_eval_split,
            pointwise_replay=None,
            pointwise_replay_ratio=0,
            aligned_pairs=train_pairs_sel,
            stage2_mix_mode=str(cfg.sft_stage2_mix_mode),
            stage2_pairs_per_batch=int(cfg.sft_stage2_pairs_per_batch),
            epochs=int(cfg.pairwise_epochs),
            per_device_batch_size=int(cfg.sft_per_device_batch_size),
            gradient_accumulation_steps=int(cfg.sft_gradient_accumulation_steps),
            learning_rate=float(cfg.sft_lr),
            max_length=int(cfg.sft_max_length),
            use_lora=bool(cfg.sft_use_lora),
            load_in_4bit=bool(cfg.sft_load_in_4bit),
            seed=int(cfg.seed),
            output_dir=str(base_out / "sft_single_stage_pairbatch_model"),
            fix_score_prefix_in_prompt=bool(cfg.fix_score_prefix_in_prompt),
            score_min=int(cfg.score_min),
            score_max=int(cfg.score_max),
            randomize_pair_order=bool(cfg.randomize_pair_order),
            global_smooth_alpha=float(cfg.sft_pointwise_global_smooth_alpha),
            global_smooth_start_step=int(cfg.sft_pointwise_global_smooth_start_step),
            global_smooth_warmup_steps=int(cfg.sft_pointwise_global_smooth_warmup_steps),
            global_smooth_prior=float(cfg.sft_pointwise_global_smooth_prior),
            global_smooth_trainable_alpha=bool(cfg.sft_pointwise_global_smooth_trainable_alpha),
            global_smooth_alpha_max=float(cfg.sft_pointwise_global_smooth_alpha_max),
            global_smooth_alpha_reg=float(cfg.sft_pointwise_global_smooth_alpha_reg),
        global_smooth_alpha_lr=float(cfg.sft_pointwise_global_smooth_alpha_lr),
            return_model=True,
        )
        _write_json(base_out / "train_stats_single_stage_pairbatch_sft.json", single_stats)
        _write_json(base_out / "train_stats_stage2_pairwise_sft.json", single_stats)

        print("Evaluating after single-stage pairbatch SFT...")
        pw_after = _evaluate_pointwise_sft(
            model=model,
            tokenizer=tokenizer,
            examples=pw_eval_split,
            max_length=int(cfg.sft_max_length),
            batch_size=max(1, int(cfg.sft_per_device_batch_size)),
            max_new_tokens=4,
            score_min=int(cfg.score_min),
            score_max=int(cfg.score_max),
        )
        pr_after = _evaluate_pairwise_sft(
            model=model,
            tokenizer=tokenizer,
            examples=pr_eval_split,
            max_length=int(cfg.sft_max_length),
            batch_size=max(1, int(cfg.sft_per_device_batch_size)),
            max_new_tokens=8,
        )
        _write_json(base_out / "metrics_pointwise_after_stage2.json", pw_after)
        _write_json(base_out / "metrics_pairwise_after_stage2.json", pr_after)

        summary = {
            "mode": "sft_single_stage_pairbatch",
            "pointwise_training_mode": "sft",
            "training_mode": "sft",
            "eval_split": {
                "pointwise": str(pw_eval_name),
                "pairwise": str(pr_eval_name),
            },
            "external_fixed_eval": {
                "enabled": bool(cfg.use_external_fixed_eval),
                "pointwise": external_pw_eval_info,
                "pairwise": external_pr_eval_info,
            },
            "dataset_load_stats": load_stats,
            "selection_stats": selected_stats,
            "candidate_pair_selection": candidate_pair_stats,
            "candidate_selector_target_task": (
                str(cfg.candidate_selector_target_task)
                if str(cfg.train_selection_mode) == "candidate_pair_selector"
                else None
            ),
            "pairwise_abc_extra": {
                "train": pairwise_abc_train_info,
                "eval": pairwise_abc_eval_info,
                "split": pairwise_abc_split_info,
            },
            "split_by_question": split_info,
            "train_budget": budget_info,
            "pointwise_metrics": {
                "before_stage1": pw_before,
                "after_stage2": pw_after,
            },
            "pairwise_metrics": {
                "before_stage1": pr_before,
                "after_stage2": pr_after,
            },
            "train_stats": {
                "single_stage_pairbatch_sft": single_stats,
                "stage2_pairwise_sft": single_stats,
            },
            "config": asdict(cfg),
        }
        _write_json(base_out / "summary.json", summary)
        _print_compact_run_summary(base_out, _write_compact_metrics(base_out, summary))

        del model, tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        _log_memory_usage("finished")
        return

    if str(cfg.training_mode) == "sft" and str(cfg.pointwise_training_mode) == "sft":
        print("\n" + "=" * 80)
        print("Using full SFT mode: pointwise score SFT -> pairwise preference SFT")
        print("=" * 80)

        model, tokenizer, _ = _load_sft_model_and_tokenizer(
            model_name_or_path=str(args.llama),
            max_length=int(cfg.sft_max_length),
            load_in_4bit=bool(cfg.sft_load_in_4bit),
        )

        print("\nEvaluating before stage-1 SFT...")
        pw_before = _evaluate_pointwise_sft(
            model=model,
            tokenizer=tokenizer,
            examples=pw_eval_split,
            max_length=int(cfg.sft_max_length),
            batch_size=max(1, int(cfg.sft_per_device_batch_size)),
            max_new_tokens=4,
            score_min=int(cfg.score_min),
            score_max=int(cfg.score_max),
        )
        pr_before_stage1 = _evaluate_pairwise_sft(
            model=model,
            tokenizer=tokenizer,
            examples=pr_eval_split,
            max_length=int(cfg.sft_max_length),
            batch_size=max(1, int(cfg.sft_per_device_batch_size)),
            max_new_tokens=8,
        )
        _write_json(base_out / "metrics_pointwise_before_stage1.json", pw_before)
        _write_json(base_out / "metrics_pairwise_before_stage1.json", pr_before_stage1)

        print("\nStage-1 pointwise score SFT...")
        stage1_stats, model, tokenizer = _train_sft_pointwise(
            model_name_or_path=None,
            model=model,
            tokenizer=tokenizer,
            pointwise_train=pointwise_train,
            pointwise_val=pw_eval_split,
            epochs=int(cfg.pointwise_epochs),
            per_device_batch_size=int(cfg.sft_per_device_batch_size),
            gradient_accumulation_steps=int(cfg.sft_gradient_accumulation_steps),
            learning_rate=float(cfg.sft_lr),
            max_length=int(cfg.sft_max_length),
            use_lora=bool(cfg.sft_use_lora),
            load_in_4bit=bool(cfg.sft_load_in_4bit),
            seed=int(cfg.seed),
            output_dir=str(base_out / "sft_pointwise_model"),
            fix_score_prefix_in_prompt=bool(cfg.fix_score_prefix_in_prompt),
            score_min=int(cfg.score_min),
            score_max=int(cfg.score_max),
            global_smooth_alpha=float(cfg.sft_pointwise_global_smooth_alpha),
            global_smooth_start_step=int(cfg.sft_pointwise_global_smooth_start_step),
            global_smooth_warmup_steps=int(cfg.sft_pointwise_global_smooth_warmup_steps),
            global_smooth_prior=float(cfg.sft_pointwise_global_smooth_prior),
            global_smooth_trainable_alpha=bool(cfg.sft_pointwise_global_smooth_trainable_alpha),
            global_smooth_alpha_max=float(cfg.sft_pointwise_global_smooth_alpha_max),
            global_smooth_alpha_reg=float(cfg.sft_pointwise_global_smooth_alpha_reg),
        global_smooth_alpha_lr=float(cfg.sft_pointwise_global_smooth_alpha_lr),
        )
        _write_json(base_out / "train_stats_stage1_pointwise.json", stage1_stats)
        _write_json(base_out / "train_stats_stage1_pointwise_sft.json", stage1_stats)

        print("Evaluating after stage-1 SFT...")
        pw_after_stage1 = _evaluate_pointwise_sft(
            model=model,
            tokenizer=tokenizer,
            examples=pw_eval_split,
            max_length=int(cfg.sft_max_length),
            batch_size=max(1, int(cfg.sft_per_device_batch_size)),
            max_new_tokens=4,
            score_min=int(cfg.score_min),
            score_max=int(cfg.score_max),
        )
        pr_after_stage1 = _evaluate_pairwise_sft(
            model=model,
            tokenizer=tokenizer,
            examples=pr_eval_split,
            max_length=int(cfg.sft_max_length),
            batch_size=max(1, int(cfg.sft_per_device_batch_size)),
            max_new_tokens=8,
        )
        _write_json(base_out / "metrics_pointwise_after_stage1.json", pw_after_stage1)
        _write_json(base_out / "metrics_pairwise_after_stage1.json", pr_after_stage1)

        pr_before_stage2 = dict(pr_after_stage1)
        _write_json(base_out / "metrics_pairwise_before_stage2.json", pr_before_stage2)

        if not pairwise_train:
            print("\nStage-2 pairwise SFT skipped: pairwise_train is empty.")
            stage2_stats = {
                "mode": "sft",
                "reused_existing_model": True,
                "train_samples": 0,
                "val_samples": int(len(sft_pairwise_eval_split)),
                "epochs": int(cfg.pairwise_epochs),
                "elapsed_sec": 0.0,
                "eval_pairwise": {"n": 0},
            }
        else:
            print("\nStage-2 pairwise preference SFT...")
            stage2_stats = _train_sft_pairwise(
                model_name_or_path=None,
                model=model,
                tokenizer=tokenizer,
                pairwise_train=pairwise_train,
                pairwise_val=sft_pairwise_eval_split,
                pointwise_replay=pointwise_train,
                pointwise_replay_ratio=int(cfg.stage2_pointwise_replay_ratio),
                aligned_pairs=train_pairs_sel,
                stage2_mix_mode=str(cfg.sft_stage2_mix_mode),
                stage2_pairs_per_batch=int(cfg.sft_stage2_pairs_per_batch),
                epochs=int(cfg.pairwise_epochs),
                per_device_batch_size=int(cfg.sft_per_device_batch_size),
                gradient_accumulation_steps=int(cfg.sft_gradient_accumulation_steps),
                learning_rate=float(cfg.sft_lr),
                max_length=int(cfg.sft_max_length),
                use_lora=bool(cfg.sft_use_lora),
                load_in_4bit=bool(cfg.sft_load_in_4bit),
                seed=int(cfg.seed),
                output_dir=str(base_out / "sft_pairwise_model"),
                fix_score_prefix_in_prompt=bool(cfg.fix_score_prefix_in_prompt),
                score_min=int(cfg.score_min),
                score_max=int(cfg.score_max),
                randomize_pair_order=bool(cfg.randomize_pair_order),
                global_smooth_alpha=float(cfg.sft_pointwise_global_smooth_alpha),
                global_smooth_start_step=int(cfg.sft_pointwise_global_smooth_start_step),
                global_smooth_warmup_steps=int(cfg.sft_pointwise_global_smooth_warmup_steps),
                global_smooth_prior=float(cfg.sft_pointwise_global_smooth_prior),
                global_smooth_trainable_alpha=bool(cfg.sft_pointwise_global_smooth_trainable_alpha),
                global_smooth_alpha_max=float(cfg.sft_pointwise_global_smooth_alpha_max),
                global_smooth_alpha_reg=float(cfg.sft_pointwise_global_smooth_alpha_reg),
        global_smooth_alpha_lr=float(cfg.sft_pointwise_global_smooth_alpha_lr),
            )
        _write_json(base_out / "train_stats_stage2_pairwise.json", stage2_stats)
        _write_json(base_out / "train_stats_stage2_pairwise_sft.json", stage2_stats)

        print("Evaluating after stage-2 SFT...")
        pw_after_stage2 = _evaluate_pointwise_sft(
            model=model,
            tokenizer=tokenizer,
            examples=pw_eval_split,
            max_length=int(cfg.sft_max_length),
            batch_size=max(1, int(cfg.sft_per_device_batch_size)),
            max_new_tokens=4,
            score_min=int(cfg.score_min),
            score_max=int(cfg.score_max),
        )
        pr_after_stage2 = _evaluate_pairwise_sft(
            model=model,
            tokenizer=tokenizer,
            examples=pr_eval_split,
            max_length=int(cfg.sft_max_length),
            batch_size=max(1, int(cfg.sft_per_device_batch_size)),
            max_new_tokens=8,
        )
        _write_json(base_out / "metrics_pointwise_after_stage2.json", pw_after_stage2)
        _write_json(base_out / "metrics_pairwise_after_stage2.json", pr_after_stage2)

        summary = {
            "mode": "sft_pointwise_to_pairwise",
            "pointwise_training_mode": "sft",
            "training_mode": "sft",
            "eval_split": {
                "pointwise": str(pw_eval_name),
                "pairwise": str(pr_eval_name),
            },
            "external_fixed_eval": {
                "enabled": bool(cfg.use_external_fixed_eval),
                "pointwise": external_pw_eval_info,
                "pairwise": external_pr_eval_info,
            },
            "dataset_load_stats": load_stats,
            "selection_stats": selected_stats,
            "candidate_pair_selection": candidate_pair_stats,
            "candidate_selector_target_task": (
                str(cfg.candidate_selector_target_task)
                if str(cfg.train_selection_mode) == "candidate_pair_selector"
                else None
            ),
            "pairwise_abc_extra": {
                "train": pairwise_abc_train_info,
                "eval": pairwise_abc_eval_info,
                "split": pairwise_abc_split_info,
            },
            "split_by_question": split_info,
            "train_budget": budget_info,
            "pointwise_metrics": {
                "before_stage1": pw_before,
                "after_stage1": pw_after_stage1,
                "after_stage2": pw_after_stage2,
            },
            "pairwise_metrics": {
                "before_stage1": pr_before_stage1,
                "after_stage1": pr_after_stage1,
                "before_stage2": pr_before_stage2,
                "after_stage2": pr_after_stage2,
            },
            "train_stats": {
                "stage1_pointwise_sft": stage1_stats,
                "stage1_pointwise": stage1_stats,
                "stage2_pairwise_sft": stage2_stats,
                "stage2_pairwise": stage2_stats,
            },
            "config": asdict(cfg),
        }
        _write_json(base_out / "summary.json", summary)
        _print_compact_run_summary(base_out, _write_compact_metrics(base_out, summary))

        del model, tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        _log_memory_usage("finished")
        return

    # ---- Training flow ----
    # Stage-1 always trains pointwise first; stage-2 either continues on the same model (SFT)
    # or alternates pairwise + pointwise replay in the shared proxy.
    print("\n" + "=" * 80)
    if str(cfg.training_mode) == "sft":
        print("Using SFT mode for stage-2 pairwise on the same model")
    else:
        print("Using Proxy mode (original multitask stage-2)")
    print("=" * 80)

    # ---- Init proxy (shared for stage-1) ----
    print("\nInitializing shared multitask proxy...")
    pointwise_class_weights = _compute_pointwise_class_weights(
        pointwise_train,
        num_labels=int(cfg.score_max - cfg.score_min + 1),
        mode=str(cfg.pointwise_class_weight_mode),
        strength=float(cfg.pointwise_class_weight_strength),
    )
    proxy = LlamaSharedMultiTaskProxyModel(
        model_path=str(args.llama),
        pointwise_num_labels=int(cfg.score_max - cfg.score_min + 1),
        pairwise_num_labels=3,
        multitask_mode=str(cfg.llama_multitask_mode),
        lr=float(cfg.proxy_lr),
        weight_decay=0.0,
        max_length=int(cfg.proxy_max_length),
        finetune_mode="lora",
        gradient_checkpointing=True,
        load_in_4bit=bool(cfg.load_in_4bit),
        score_min=int(cfg.score_min),
        score_max=int(cfg.score_max),
        fix_score_prefix_in_prompt=bool(cfg.fix_score_prefix_in_prompt),
        pointwise_loss_type=str(cfg.pointwise_loss_type),
        pointwise_distance_weight=float(cfg.pointwise_distance_weight),
    )
    if pointwise_class_weights is not None:
        proxy.pointwise_class_weights = torch.tensor(
            pointwise_class_weights,
            dtype=torch.float32,
            device=proxy.device,
        )

    # ---- Evaluate before any training ----
    print("\nEvaluating before stage-1...")
    pw_before = _evaluate_pointwise(proxy, pw_eval_split, score_min=int(cfg.score_min))
    pr_before_stage1 = _evaluate_pairwise(proxy, pr_eval_split)
    _write_json(base_out / "metrics_pointwise_before_stage1.json", pw_before)
    _write_json(base_out / "metrics_pairwise_before_stage1.json", pr_before_stage1)

    # ---- Stage-1: pointwise (always run) ----
    print("\nStage-1 training on selected pointwise data...")
    stage1_stats = _train_pointwise_stage(
        proxy=proxy,
        examples=pointwise_train,
        epochs=int(cfg.pointwise_epochs),
        batch_size=int(cfg.pointwise_batch_size),
        seed=int(cfg.seed) + 17,
        stage_name="stage1-pointwise",
    )
    _write_json(base_out / "train_stats_stage1_pointwise.json", stage1_stats)

    print("Evaluating after stage-1...")
    pw_after_stage1 = _evaluate_pointwise(proxy, pw_eval_split, score_min=int(cfg.score_min))
    pr_after_stage1 = _evaluate_pairwise(proxy, pr_eval_split)
    _write_json(base_out / "metrics_pointwise_after_stage1.json", pw_after_stage1)
    _write_json(base_out / "metrics_pairwise_after_stage1.json", pr_after_stage1)

    # ---- Pairwise eval before stage-2 ----
    pr_before_stage2 = dict(pr_after_stage1)
    _write_json(base_out / "metrics_pairwise_before_stage2.json", pr_before_stage2)

    if str(cfg.training_mode) == "sft":
        if not pairwise_train:
            print("\nStage-2 SFT skipped: pairwise_train is empty.")
            sft_stats = {
                "mode": "sft",
                "train_samples": 0,
                "val_samples": int(len(sft_pairwise_eval_split)),
                "epochs": int(cfg.pairwise_epochs),
                "elapsed_sec": 0.0,
                "eval_pairwise": {"n": 0},
            }
        else:
            print("\nStage-2 SFT training on converted pairwise data...")
            sft_stats = _train_sft_pairwise(
                model_name_or_path=None,
                model=proxy.model,
                tokenizer=proxy.tokenizer,
                pairwise_train=pairwise_train,
                pairwise_val=sft_pairwise_eval_split,
                pointwise_replay=pointwise_train,
                pointwise_replay_ratio=int(cfg.stage2_pointwise_replay_ratio),
                aligned_pairs=train_pairs_sel,
                stage2_mix_mode=str(cfg.sft_stage2_mix_mode),
                stage2_pairs_per_batch=int(cfg.sft_stage2_pairs_per_batch),
                epochs=int(cfg.pairwise_epochs),
                per_device_batch_size=int(cfg.sft_per_device_batch_size),
                gradient_accumulation_steps=int(cfg.sft_gradient_accumulation_steps),
                learning_rate=float(cfg.sft_lr),
                max_length=int(cfg.sft_max_length),
                use_lora=bool(cfg.sft_use_lora),
                load_in_4bit=bool(cfg.sft_load_in_4bit),
                seed=int(cfg.seed),
                output_dir=str(base_out / "sft_model"),
                fix_score_prefix_in_prompt=bool(cfg.fix_score_prefix_in_prompt),
                score_min=int(cfg.score_min),
                score_max=int(cfg.score_max),
                randomize_pair_order=bool(cfg.randomize_pair_order),
                global_smooth_alpha=float(cfg.sft_pointwise_global_smooth_alpha),
                global_smooth_start_step=int(cfg.sft_pointwise_global_smooth_start_step),
                global_smooth_warmup_steps=int(cfg.sft_pointwise_global_smooth_warmup_steps),
                global_smooth_prior=float(cfg.sft_pointwise_global_smooth_prior),
                global_smooth_trainable_alpha=bool(cfg.sft_pointwise_global_smooth_trainable_alpha),
                global_smooth_alpha_max=float(cfg.sft_pointwise_global_smooth_alpha_max),
                global_smooth_alpha_reg=float(cfg.sft_pointwise_global_smooth_alpha_reg),
        global_smooth_alpha_lr=float(cfg.sft_pointwise_global_smooth_alpha_lr),
            )

        print("Evaluating after stage-2...")
        pw_after_stage2 = _evaluate_pointwise(proxy, pw_eval_split, score_min=int(cfg.score_min))
        pr_after_stage2 = _evaluate_pairwise(proxy, pr_eval_split)
        _write_json(base_out / "metrics_pointwise_after_stage2.json", pw_after_stage2)
        _write_json(base_out / "metrics_pairwise_after_stage2.json", pr_after_stage2)

        summary = {
            "mode": "sft_stage2",
            "eval_split": {
                "pointwise": str(pw_eval_name),
                "pairwise": str(pr_eval_name),
            },
            "external_fixed_eval": {
                "enabled": bool(cfg.use_external_fixed_eval),
                "pointwise": external_pw_eval_info,
                "pairwise": external_pr_eval_info,
            },
            "dataset_load_stats": load_stats,
            "selection_stats": selected_stats,
            "candidate_pair_selection": candidate_pair_stats,
            "candidate_selector_target_task": (
                str(cfg.candidate_selector_target_task)
                if str(cfg.train_selection_mode) == "candidate_pair_selector"
                else None
            ),
            "pointwise_training_mode": str(cfg.pointwise_training_mode),
            "pointwise_loss_type": str(cfg.pointwise_loss_type),
            "pointwise_distance_weight": float(cfg.pointwise_distance_weight),
            "pointwise_class_weight_mode": str(cfg.pointwise_class_weight_mode),
            "pointwise_class_weight_strength": float(cfg.pointwise_class_weight_strength),
            "pointwise_class_weights": pointwise_class_weights.tolist() if pointwise_class_weights is not None else None,
            "split_by_question": split_info,
            "train_budget": budget_info,
            "pointwise_metrics": {
                "before_stage1": pw_before,
                "after_stage1": pw_after_stage1,
                "after_stage2": pw_after_stage2,
            },
            "pairwise_metrics": {
                "before_stage1": pr_before_stage1,
                "after_stage1": pr_after_stage1,
                "before_stage2": pr_before_stage2,
                "after_stage2": pr_after_stage2,
            },
            "pairwise_generated": {
                "train": pairwise_train_stats,
                "val": pairwise_val_stats,
            },
            "pairwise_abc_extra": {
                "train": pairwise_abc_train_info,
                "eval": pairwise_abc_eval_info,
                "split": pairwise_abc_split_info,
            },
            "train_stats": {
                "stage1_pointwise": stage1_stats,
                "stage2_sft": sft_stats,
                "stage2_pairwise": sft_stats,
            },
        }
        _write_json(base_out / "summary.json", summary)
        _print_compact_run_summary(base_out, _write_compact_metrics(base_out, summary))
    else:
        # ---- Stage-2: alternating pairwise + pointwise replay ----
        stage2_stats = {
            "n_pairwise": 0,
            "n_pointwise_replay": int(len(pointwise_train)),
            "epochs": int(cfg.pairwise_epochs),
            "pairwise_steps": 0,
            "pointwise_replay_steps": 0,
            "pointwise_replay_ratio": int(cfg.stage2_pointwise_replay_ratio),
            "elapsed_sec": 0.0,
        }
        if pairwise_train:
            print("\nStage-2 alternating training on converted pairwise + pointwise replay...")
            stage2_stats = _train_stage2_alternating(
                proxy=proxy,
                pairwise_examples=pairwise_train,
                pointwise_examples=pointwise_train,
                epochs=int(cfg.pairwise_epochs),
                pairwise_batch_size=int(cfg.pairwise_batch_size),
                pointwise_batch_size=int(cfg.pointwise_batch_size),
                pointwise_replay_ratio=int(cfg.stage2_pointwise_replay_ratio),
                seed=int(cfg.seed) + 29,
                stage_name="stage2-alternating",
            )
        else:
            print("\nStage-2 skipped: pairwise_train is empty.")

        _write_json(base_out / "train_stats_stage2_alternating.json", stage2_stats)
        # Backward-compatible alias for existing analysis scripts.
        _write_json(base_out / "train_stats_stage2_pairwise.json", stage2_stats)

        print("Evaluating after stage-2...")
        pw_after_stage2 = _evaluate_pointwise(proxy, pw_eval_split, score_min=int(cfg.score_min))
        pr_after_stage2 = _evaluate_pairwise(proxy, pr_eval_split)
        _write_json(base_out / "metrics_pointwise_after_stage2.json", pw_after_stage2)
        _write_json(base_out / "metrics_pairwise_after_stage2.json", pr_after_stage2)

        summary = {
            "eval_split": {
                "pointwise": str(pw_eval_name),
                "pairwise": str(pr_eval_name),
            },
            "external_fixed_eval": {
                "enabled": bool(cfg.use_external_fixed_eval),
                "pointwise": external_pw_eval_info,
                "pairwise": external_pr_eval_info,
            },
            "dataset_load_stats": load_stats,
            "selection_stats": selected_stats,
            "candidate_pair_selection": candidate_pair_stats,
            "candidate_selector_target_task": (
                str(cfg.candidate_selector_target_task)
                if str(cfg.train_selection_mode) == "candidate_pair_selector"
                else None
            ),
            "pointwise_training_mode": str(cfg.pointwise_training_mode),
            "pointwise_loss_type": str(cfg.pointwise_loss_type),
            "pointwise_distance_weight": float(cfg.pointwise_distance_weight),
            "pointwise_class_weight_mode": str(cfg.pointwise_class_weight_mode),
            "pointwise_class_weight_strength": float(cfg.pointwise_class_weight_strength),
            "pointwise_class_weights": pointwise_class_weights.tolist() if pointwise_class_weights is not None else None,
            "split_by_question": split_info,
            "train_budget": budget_info,
            "pointwise_metrics": {
                "before_stage1": pw_before,
                "after_stage1": pw_after_stage1,
                "after_stage2": pw_after_stage2,
            },
            "pairwise_metrics": {
                "before_stage1": pr_before_stage1,
                "after_stage1": pr_after_stage1,
                "before_stage2": pr_before_stage2,
                "after_stage2": pr_after_stage2,
            },
            "pairwise_generated": {
                "train": pairwise_train_stats,
                "val": pairwise_val_stats,
            },
            "pairwise_abc_extra": {
                "train": pairwise_abc_train_info,
                "eval": pairwise_abc_eval_info,
                "split": pairwise_abc_split_info,
            },
            "train_stats": {
                "stage1_pointwise": stage1_stats,
                "stage2_alternating": stage2_stats,
                "stage2_pairwise": stage2_stats,
            },
        }
        _write_json(base_out / "summary.json", summary)
        _print_compact_run_summary(base_out, _write_compact_metrics(base_out, summary))

        # Cleanup
        del proxy
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    _log_memory_usage("finished")


if __name__ == "__main__":
    # `expandable_segments` is not supported by older Torch (e.g. 2.0.x).
    if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
        v = str(getattr(torch, "__version__", ""))
        m = re.match(r"^(\d+)\.(\d+)", v)
        if m is not None:
            major = int(m.group(1))
            minor = int(m.group(2))
            if (major > 2) or (major == 2 and minor >= 1):
                os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    main()

#!/usr/bin/env python
"""Generative SFT: pointwise -> pairwise -> listwise from 3-answer triples.

This is the SFT counterpart of the proxy three-stage experiment. It does not add
classification heads. The same causal LM/LoRA model is trained sequentially to
generate:

- pointwise: ``Score: [x]`` (or just ``x]`` when the prompt already ends at
  ``Score: [``)
- pairwise: ``[[1]]`` / ``[[2]]`` / ``[[3]]``
- listwise: ``Ranking:[A>B>C]`` and other allowed ranking strings
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import re
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

import run_pointwise5answers_three_to_listwise_v1 as lw
import run_pointwise5answers_two_to_pairwise_v1 as base

CLASS_TEACHER_TASK_PAIRWISE = 2
CLASS_TEACHER_TASK_LISTWISE_TOP = 3


@dataclass
class RunConfig:
    seed: int
    val_ratio: float
    val_split_seed: int
    pointwise_val_answer_seed: int
    train_selection_mode: str
    fixed_selected_triples_path: str
    resume_stage1_model_dir: str
    triple_selection_strategy: str
    question_selection_strategy: str
    randomize_listwise_order: bool
    candidate_selector_kind: str
    candidate_selector_init_triples: int
    candidate_selector_batch_size: int
    candidate_selector_epochs: int
    candidate_selector_max_score_candidates: int
    candidate_selector_llama_rerank_candidates: int
    candidate_selector_buffer_maxlen: int
    candidate_selector_one_per_question: bool
    candidate_selector_target_task: str
    candidate_selector_score_range_weight: float
    candidate_selector_gap_sum_weight: float
    candidate_selector_uncertainty_weight: float
    candidate_selector_pairwise_uncertainty_weight: float
    candidate_selector_listwise_uncertainty_weight: float
    candidate_selector_kl_weight: float
    candidate_selector_score_bin_weight: float
    candidate_selector_diversity_weight: float
    candidate_selector_density_weight: float
    candidate_selector_bias_weight: float
    candidate_selector_coverage_weight: float
    candidate_selector_pointwise_length_bias_weight: float
    candidate_selector_pairwise_position_bias_weight: float
    candidate_selector_pairwise_position_pairs: int
    candidate_selector_pairwise_position_bias_scale: float
    candidate_selector_signal_normalization: str
    candidate_selector_uncertainty_view: str
    candidate_selector_length_aug_suffix: str
    candidate_selector_density_k: int
    candidate_selector_embedding_model: str
    candidate_selector_embedding_max_length: int
    candidate_selector_embedding_batch_size: int
    candidate_selector_embedding_device: str
    candidate_selector_embedding_pooling: str
    candidate_selector_diversity_view: str
    candidate_selector_exploration_ratio: float
    candidate_selector_entropy_weight: float
    candidate_selector_score_std_weight: float
    candidate_selector_predicted_coverage_weight: float
    candidate_selector_proxy_warmup_epochs: int
    candidate_selector_proxy_update_epochs: int
    candidate_selector_proxy_mode: str
    reuse_selection_proxy_for_stage1: bool
    candidate_bert_selector_model: str
    candidate_bert_selector_max_length: int
    candidate_bert_selector_freeze: bool
    candidate_bert_selector_unfreeze_last_n_layers: int
    proxy_lr: float
    proxy_max_length: int
    llama_multitask_mode: str
    pointwise_loss_type: str
    pointwise_distance_weight: float
    pointwise_class_weight_mode: str
    pointwise_class_weight_strength: float
    budget_units: int
    pointwise_epochs: int
    pairwise_epochs: int
    listwise_epochs: int
    per_device_batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    max_length: int
    max_new_tokens_pointwise: int
    max_new_tokens_pairwise: int
    max_new_tokens_listwise: int
    eval_batch_size: int
    eval_stages: str
    stage2_pointwise_replay_ratio: int
    stage3_pointwise_replay_ratio: int
    stage3_pairwise_replay_ratio: int
    merge_stage2_stage3: bool
    stage23_pointwise_replay_ratio: int
    stage23_pairwise_weight: float
    stage23_listwise_weight: float
    stage23_pointwise_weight: float
    stage23_epochs: int
    pointwise_teacher_distill_weight: float
    pointwise_teacher_distill_temperature: float
    stage4_task_teacher_distill_weight: float
    stage4_task_teacher_distill_temperature: float
    stage4_replay_strategy: str
    stage4_replay_fraction: float
    stage4_epochs: int
    stage4_listwise_multiplier: int
    pairwise_order_augmentation: bool
    listwise_order_augmentation: bool
    score_min: int
    score_max: int
    fix_score_prefix_in_prompt: bool
    use_lora: bool
    load_in_4bit: bool
    pointwise_global_smooth_alpha: float
    pointwise_global_smooth_mode: str
    pointwise_global_smooth_gaussian_sigma: float
    pointwise_global_smooth_stages: str
    pointwise_global_smooth_start_step: int
    pointwise_global_smooth_warmup_steps: int
    pointwise_global_smooth_start_pointwise_seen: int
    pointwise_global_smooth_warmup_pointwise_seen: int
    pointwise_global_smooth_prior: float
    pointwise_global_smooth_init_prior_from_stage1: bool
    pointwise_global_smooth_freeze_prior: bool
    pointwise_global_smooth_uniform_mix: float
    pointwise_global_smooth_adaptive_entropy: bool
    pointwise_global_smooth_trainable_alpha: bool
    pointwise_global_smooth_alpha_max: float
    pointwise_global_smooth_alpha_reg: float
    pointwise_global_smooth_alpha_lr: float
    max_pointwise_eval_samples: int
    max_pairwise_eval_samples: int
    max_listwise_eval_samples: int
    fsdp: str
    fsdp_transformer_layer_cls_to_wrap: str
    fsdp_state_dict_type: str
    fsdp_activation_checkpointing: bool
    fsdp_use_orig_params: bool
    fsdp_save_all_stages: bool


def _world_size() -> int:
    """Return the torchrun world size before or after process-group initialization."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return int(torch.distributed.get_world_size())
    return int(os.environ.get("WORLD_SIZE", "1"))


def _is_primary_process() -> bool:
    """Use RANK early, since most artifacts are written before Trainer initializes FSDP."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return int(torch.distributed.get_rank()) == 0
    return int(os.environ.get("RANK", "0")) == 0


def _distributed_barrier() -> None:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()


def _fsdp_enabled(cfg: RunConfig) -> bool:
    return bool(str(cfg.fsdp).strip())


def _is_final_training_stage(cfg: RunConfig, stage_name: str) -> bool:
    if str(cfg.stage4_replay_strategy) != "none":
        return str(stage_name) == "stage4_consolidation"
    if bool(cfg.merge_stage2_stage3):
        return str(stage_name) == "stage23_pairwise_listwise"
    return str(stage_name) == "stage3_listwise"


def _install_fsdp_forward_method_compat() -> None:
    """Bridge Transformers 5.6 to PyTorch 2.5's classic FSDP API."""
    from torch.distributed import fsdp as torch_fsdp

    if hasattr(torch_fsdp, "register_fsdp_forward_method"):
        return

    # The newer helper only acts on composable FSDP2 modules. This environment
    # uses classic FSDP1, for which the correct behavior is a no-op.
    def register_fsdp_forward_method(module: Any, method_name: str) -> None:
        del module, method_name

    torch_fsdp.register_fsdp_forward_method = register_fsdp_forward_method


def _score_hist_from_items(items: Sequence[Tuple[str, str, str, int]], *, num_labels: int) -> List[float]:
    hist = [0.0 for _ in range(int(num_labels))]
    for _, _, _, score_label in items:
        y = int(score_label)
        if 0 <= y < int(num_labels):
            hist[y] += 1.0
    return hist


def _load_fixed_selected_triples(
    *,
    path: str,
    candidates: Sequence[lw.CandidateTripleExample],
    budget_units: int,
) -> Tuple[List[lw.SelectedQuestionTriple], List[Dict[str, Any]], Dict[str, Any]]:
    fixed_path = Path(path)
    if not fixed_path.exists():
        raise FileNotFoundError(f"fixed selected triples file not found: {path}")
    rows: List[Dict[str, Any]] = []
    with fixed_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    expected = int(budget_units) // 3
    if len(rows) != expected:
        raise ValueError(f"fixed selected triples must contain {expected} rows for budget {budget_units}, got {len(rows)}")

    by_id = {int(c.id): c for c in candidates}
    by_signature = {
        (
            int(c.source_id),
            str(c.model_a),
            str(c.model_b),
            str(c.model_c),
            int(c.score_a),
            int(c.score_b),
            int(c.score_c),
        ): c
        for c in candidates
    }
    by_unordered_signature = {
        (
            int(c.source_id),
            tuple(
                sorted(
                    (
                        (str(c.model_a), int(c.score_a)),
                        (str(c.model_b), int(c.score_b)),
                        (str(c.model_c), int(c.score_c)),
                    )
                )
            ),
        ): c
        for c in candidates
    }
    selected: List[lw.SelectedQuestionTriple] = []
    resolved_rows: List[Dict[str, Any]] = []
    seen_groups: set[int] = set()
    for row in rows:
        candidate = by_id.get(int(row.get("candidate_triple_id", -1)))
        if candidate is None:
            signature = (
                int(row.get("source_id", -1)),
                str(row.get("model_a", "")),
                str(row.get("model_b", "")),
                str(row.get("model_c", "")),
                int(row.get("score_a", -1)),
                int(row.get("score_b", -1)),
                int(row.get("score_c", -1)),
            )
            candidate = by_signature.get(signature)
        if candidate is None:
            unordered_signature = (
                int(row.get("source_id", -1)),
                tuple(
                    sorted(
                        (
                            (str(row.get("model_a", "")), int(row.get("score_a", -1))),
                            (str(row.get("model_b", "")), int(row.get("score_b", -1))),
                            (str(row.get("model_c", "")), int(row.get("score_c", -1))),
                        )
                    )
                ),
            )
            candidate = by_unordered_signature.get(unordered_signature)
        if candidate is None:
            raise ValueError(f"could not resolve fixed selected triple: {row}")
        if int(candidate.group_id) in seen_groups:
            raise ValueError(f"duplicate fixed selected group_id: {candidate.group_id}")
        seen_groups.add(int(candidate.group_id))

        triple = candidate.selected_triple
        available_answers = [triple.answer_a, triple.answer_b, triple.answer_c]
        ordered_answers: List[base.AnswerWithScore] = []
        for position in ("a", "b", "c"):
            target_model = str(row.get(f"model_{position}", ""))
            target_score = int(row.get(f"score_{position}", -1))
            match_index = next(
                (
                    i
                    for i, answer in enumerate(available_answers)
                    if str(answer.model) == target_model and int(answer.score) == target_score
                ),
                None,
            )
            if match_index is None:
                raise ValueError(f"could not restore fixed selected triple order: {row}")
            ordered_answers.append(available_answers.pop(int(match_index)))

        selected.append(
            lw.SelectedQuestionTriple(
                question_id=int(triple.question_id),
                source_id=int(triple.source_id),
                dataset=str(triple.dataset),
                instruction=str(triple.instruction),
                input_text=str(triple.input_text),
                answer_a=ordered_answers[0],
                answer_b=ordered_answers[1],
                answer_c=ordered_answers[2],
            )
        )
        resolved_rows.append(dict(row, fixed_selection_reused=True))

    stats = {
        "mode": "fixed_selected_triples",
        "source_path": str(fixed_path),
        "selected_triples": int(len(selected)),
        "selected_answers": int(len(selected) * 3),
        "budget_units": int(budget_units),
        "effective_budget_units": int(len(selected) * 3),
        "one_per_question": True,
    }
    return selected, resolved_rows, stats


def _write_json(path: Path, payload: Any) -> None:
    if not _is_primary_process():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    if not _is_primary_process():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _ranking_for_triple(triple: lw.SelectedQuestionTriple) -> str:
    return lw._ranking_from_scores(
        int(triple.answer_a.score),
        int(triple.answer_b.score),
        int(triple.answer_c.score),
    )


def _score_range_bucket(score_range: int) -> str:
    r = int(score_range)
    if r <= 0:
        return "r0"
    if r <= 2:
        return "r1_2"
    if r <= 5:
        return "r3_5"
    return "r6_plus"


def _score_range_for_triple(triple: lw.SelectedQuestionTriple) -> int:
    scores = [int(triple.answer_a.score), int(triple.answer_b.score), int(triple.answer_c.score)]
    return int(max(scores) - min(scores))


def _select_stage4_replay_triples(
    *,
    train_triples: Sequence[lw.SelectedQuestionTriple],
    selected_rows: Sequence[Dict[str, Any]],
    strategy: str,
    fraction: float,
    seed: int,
) -> Tuple[List[lw.SelectedQuestionTriple], List[Dict[str, Any]], Dict[str, Any]]:
    n = int(len(train_triples))
    if n <= 0:
        return [], [], {"enabled": False, "reason": "no_train_triples"}

    frac = float(fraction)
    if not (0.0 < frac <= 1.0):
        raise ValueError("stage4-replay-fraction must be in (0, 1] when Stage-4 replay is enabled")

    k = int(round(float(n) * frac))
    k = min(n, max(1, k))
    rng = np.random.default_rng(int(seed))
    strategy_s = str(strategy)

    stratum_by_index: Dict[int, str] = {}
    strata: Dict[str, List[int]] = {}
    for i, triple in enumerate(train_triples):
        score_range = _score_range_for_triple(triple)
        key = f"{_ranking_for_triple(triple)}|{_score_range_bucket(score_range)}"
        stratum_by_index[int(i)] = str(key)
        strata.setdefault(str(key), []).append(int(i))

    if strategy_s == "random_triple":
        chosen_indices = sorted(int(i) for i in rng.choice(n, size=k, replace=False).tolist())
        quotas: Dict[str, int] = {}
        for i in chosen_indices:
            key = stratum_by_index[int(i)]
            quotas[key] = int(quotas.get(key, 0) + 1)
    elif strategy_s == "stratified_triple":
        ordered_keys = sorted(strata.keys())
        quotas = {key: 0 for key in ordered_keys}
        if k >= len(ordered_keys):
            quotas = {key: 1 for key in ordered_keys}
            remaining = int(k - len(ordered_keys))
            weights = {key: max(0, len(strata[key]) - 1) for key in ordered_keys}
        else:
            remaining = int(k)
            weights = {key: len(strata[key]) for key in ordered_keys}

        if remaining > 0:
            total_weight = max(1, int(sum(weights.values())))
            ideal = {key: float(remaining) * float(weights[key]) / float(total_weight) for key in ordered_keys}
            extras = {key: int(np.floor(ideal[key])) for key in ordered_keys}
            for key, extra in extras.items():
                quotas[key] = int(min(len(strata[key]), quotas[key] + int(extra)))
            left = int(k - sum(quotas.values()))
            if left > 0:
                ranked = sorted(
                    ordered_keys,
                    key=lambda key: (
                        -(ideal.get(key, 0.0) - np.floor(ideal.get(key, 0.0))),
                        -len(strata[key]),
                        key,
                    ),
                )
                for key in ranked:
                    if left <= 0:
                        break
                    if quotas[key] < len(strata[key]):
                        quotas[key] = int(quotas[key] + 1)
                        left -= 1

        chosen_indices = []
        for key in ordered_keys:
            quota = int(quotas.get(key, 0))
            if quota <= 0:
                continue
            perm = rng.permutation(strata[key]).astype(np.int64).tolist()
            chosen_indices.extend(int(i) for i in perm[:quota])
        chosen_indices = sorted(chosen_indices)
    else:
        raise ValueError(f"unknown stage4 replay strategy: {strategy}")

    replay_triples = [train_triples[int(i)] for i in chosen_indices]
    replay_rows: List[Dict[str, Any]] = []
    for order, idx in enumerate(chosen_indices):
        row = dict(selected_rows[int(idx)]) if int(idx) < len(selected_rows) else {}
        triple = train_triples[int(idx)]
        score_range = _score_range_for_triple(triple)
        row.update(
            {
                "stage4_replay_order": int(order),
                "stage4_source_index": int(idx),
                "stage4_replay_strategy": str(strategy_s),
                "stage4_replay_stratum": stratum_by_index[int(idx)],
                "stage4_replay_score_range": int(score_range),
                "stage4_replay_score_range_bucket": _score_range_bucket(score_range),
                "stage4_replay_ranking": _ranking_for_triple(triple),
            }
        )
        replay_rows.append(row)

    selected_strata: Dict[str, int] = {}
    for idx in chosen_indices:
        key = stratum_by_index[int(idx)]
        selected_strata[key] = int(selected_strata.get(key, 0) + 1)

    stats = {
        "enabled": True,
        "strategy": str(strategy_s),
        "fraction": float(frac),
        "input_triples": int(n),
        "selected_triples": int(len(replay_triples)),
        "selected_fraction": float(len(replay_triples) / max(1, n)),
        "seed": int(seed),
        "num_strata": int(len(strata)),
        "strata_total_counts": {key: int(len(indices)) for key, indices in sorted(strata.items())},
        "strata_selected_counts": {key: int(selected_strata.get(key, 0)) for key in sorted(strata.keys())},
    }
    return replay_triples, replay_rows, stats


def _pairwise_label_from_scores(score_a: int, score_b: int) -> int:
    if int(score_a) > int(score_b):
        return int(base.LABEL_A)
    if int(score_a) < int(score_b):
        return int(base.LABEL_B)
    return int(base.LABEL_TIE)


def _swapped_pairwise_label(label: int) -> int:
    if int(label) == int(base.LABEL_A):
        return int(base.LABEL_B)
    if int(label) == int(base.LABEL_B):
        return int(base.LABEL_A)
    return int(base.LABEL_TIE)


def _append_pairwise_example(
    *,
    examples: List[base.PairwiseExample],
    rows: List[Dict[str, Any]],
    stats: Dict[str, Any],
    group_id: int,
    source_id: int,
    dataset: str,
    pair_name: str,
    instruction: str,
    input_text: str,
    model_a: str,
    model_b: str,
    output_a: str,
    output_b: str,
    score_a: Optional[int],
    score_b: Optional[int],
    label: int,
    order_augmented: bool,
) -> None:
    pair_id = int(len(examples) + 1)
    prompt = base.build_pairwise_prompt(
        system_prompt=base.DEFAULT_PAIRWISE_SYSTEM_PROMPT,
        instruction=str(instruction),
        input_text=str(input_text),
        assistant_1_output=str(output_a),
        assistant_2_output=str(output_b),
    )
    examples.append(
        base.PairwiseExample(
            id=int(pair_id),
            dataset=str(dataset),
            group_id=int(group_id),
            pair_id=int(pair_id),
            model_a=str(model_a),
            model_b=str(model_b),
            prompt=prompt,
            label=int(label),
        )
    )
    rows.append(
        {
            "pair_id": int(pair_id),
            "group_id": int(group_id),
            "source_id": int(source_id),
            "dataset": str(dataset),
            "pair_name": str(pair_name),
            "model_a": str(model_a),
            "model_b": str(model_b),
            "score_a": None if score_a is None else int(score_a),
            "score_b": None if score_b is None else int(score_b),
            "pairwise_label": int(label),
            "pairwise_token": str(base.label_to_token(int(label))),
            "order_augmented": bool(order_augmented),
        }
    )
    if int(label) == int(base.LABEL_A):
        stats["label_A"] = int(stats.get("label_A", 0) + 1)
    elif int(label) == int(base.LABEL_B):
        stats["label_B"] = int(stats.get("label_B", 0) + 1)
    else:
        stats["label_TIE"] = int(stats.get("label_TIE", 0) + 1)
    if bool(order_augmented):
        stats["order_augmented_pairs"] = int(stats.get("order_augmented_pairs", 0) + 1)


def _build_pairwise_examples_from_triples(
    selected_triples: Sequence[lw.SelectedQuestionTriple],
    *,
    order_augmentation: bool,
) -> Tuple[List[base.PairwiseExample], List[Dict[str, Any]], Dict[str, Any]]:
    examples: List[base.PairwiseExample] = []
    rows: List[Dict[str, Any]] = []
    stats: Dict[str, Any] = {
        "input_question_triples": int(len(selected_triples)),
        "pairs_per_triple_before_augmentation": 3,
        "order_augmentation": bool(order_augmentation),
        "generated_pairs": 0,
        "order_augmented_pairs": 0,
        "label_A": 0,
        "label_B": 0,
        "label_TIE": 0,
    }

    for p3 in selected_triples:
        for pair_name, left, right in (
            ("AB", p3.answer_a, p3.answer_b),
            ("AC", p3.answer_a, p3.answer_c),
            ("BC", p3.answer_b, p3.answer_c),
        ):
            label = _pairwise_label_from_scores(int(left.score), int(right.score))
            _append_pairwise_example(
                examples=examples,
                rows=rows,
                stats=stats,
                group_id=int(p3.question_id),
                source_id=int(p3.source_id),
                dataset=str(p3.dataset),
                pair_name=str(pair_name),
                instruction=str(p3.instruction),
                input_text=str(p3.input_text),
                model_a=str(left.model),
                model_b=str(right.model),
                output_a=str(left.output),
                output_b=str(right.output),
                score_a=int(left.score),
                score_b=int(right.score),
                label=int(label),
                order_augmented=False,
            )
            if bool(order_augmentation):
                _append_pairwise_example(
                    examples=examples,
                    rows=rows,
                    stats=stats,
                    group_id=int(p3.question_id),
                    source_id=int(p3.source_id),
                    dataset=str(p3.dataset),
                    pair_name=str(pair_name) + "_swap",
                    instruction=str(p3.instruction),
                    input_text=str(p3.input_text),
                    model_a=str(right.model),
                    model_b=str(left.model),
                    output_a=str(right.output),
                    output_b=str(left.output),
                    score_a=int(right.score),
                    score_b=int(left.score),
                    label=int(_swapped_pairwise_label(label)),
                    order_augmented=True,
                )

    stats["generated_pairs"] = int(len(examples))
    return examples, rows, stats


def _pairwise_relation_label_from_ranking(ranking: str, left: str, right: str) -> int:
    rank_map = lw._ranking_rank_map(str(ranking))
    left_rank = int(rank_map[str(left)])
    right_rank = int(rank_map[str(right)])
    if left_rank < right_rank:
        return int(base.LABEL_A)
    if left_rank > right_rank:
        return int(base.LABEL_B)
    return int(base.LABEL_TIE)


def _load_pairwise_eval_from_listwise_dataset(
    path: str,
) -> Tuple[List[base.PairwiseExample], List[Dict[str, Any]], Dict[str, Any]]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("listwise eval dataset JSON must be a list")

    examples: List[base.PairwiseExample] = []
    rows: List[Dict[str, Any]] = []
    stats: Dict[str, Any] = {
        "dataset_path": str(path),
        "input_records": int(len(raw)),
        "generated_pairs": 0,
        "skipped_missing_output": 0,
        "skipped_missing_label": 0,
        "label_A": 0,
        "label_B": 0,
        "label_TIE": 0,
        "format": "all_three_pairs_from_listwise_eval",
    }

    for rec_i, rec in enumerate(raw):
        if not isinstance(rec, dict):
            continue
        instruction = str(lw._first_nonempty(rec, ("Instruction", "instruction", "question"), ""))
        input_text = str(lw._first_nonempty(rec, ("input", "input_text", "context"), ""))
        qid = base._safe_int(rec.get("id", rec.get("question_id", rec_i + 1)), default=rec_i + 1)
        source_id = base._safe_int(rec.get("source_id", qid), default=qid)
        dataset = str(rec.get("dataset", "listwise_eval"))
        outputs = {
            "A": str(lw._first_nonempty(rec, ("outputA", "output_a", "assistant_a", "responseA"), "")),
            "B": str(lw._first_nonempty(rec, ("outputB", "output_b", "assistant_b", "responseB"), "")),
            "C": str(lw._first_nonempty(rec, ("outputC", "output_c", "assistant_c", "responseC"), "")),
        }
        if any(not outputs[k].strip() for k in ("A", "B", "C")):
            stats["skipped_missing_output"] += 1
            continue
        models = {
            "A": str(lw._first_nonempty(rec, ("modelA", "model_a"), "A")),
            "B": str(lw._first_nonempty(rec, ("modelB", "model_b"), "B")),
            "C": str(lw._first_nonempty(rec, ("modelC", "model_c"), "C")),
        }
        scores = {
            "A": lw._safe_score(rec, ("scoreA", "score_a"), score_min=1, score_max=10),
            "B": lw._safe_score(rec, ("scoreB", "score_b"), score_min=1, score_max=10),
            "C": lw._safe_score(rec, ("scoreC", "score_c"), score_min=1, score_max=10),
        }
        ranking = lw._normalize_ranking_text(
            lw._first_nonempty(rec, ("ranking", "raw_ranking", "label_ranking"), "")
        )
        if ranking not in lw.RANKING_TO_LABEL and all(scores[k] is not None for k in ("A", "B", "C")):
            ranking = lw._ranking_from_scores(int(scores["A"]), int(scores["B"]), int(scores["C"]))

        for left, right in (("A", "B"), ("A", "C"), ("B", "C")):
            if scores[left] is not None and scores[right] is not None:
                label = _pairwise_label_from_scores(int(scores[left]), int(scores[right]))
            elif ranking in lw.RANKING_TO_LABEL:
                label = _pairwise_relation_label_from_ranking(ranking, left, right)
            else:
                stats["skipped_missing_label"] += 1
                continue
            _append_pairwise_example(
                examples=examples,
                rows=rows,
                stats=stats,
                group_id=int(qid),
                source_id=int(source_id),
                dataset=dataset,
                pair_name=f"{left}{right}",
                instruction=instruction,
                input_text=input_text,
                model_a=models[left],
                model_b=models[right],
                output_a=outputs[left],
                output_b=outputs[right],
                score_a=scores[left],
                score_b=scores[right],
                label=int(label),
                order_augmented=False,
            )

    stats["generated_pairs"] = int(len(examples))
    if not examples:
        raise RuntimeError(f"pairwise eval from listwise dataset produced no examples: {path}")
    return examples, rows, stats


def _listwise_sft_target(example: lw.ListwiseExample) -> str:
    return f"Ranking:[{str(example.ranking)}]{base.DEFAULT_EOS_TOKEN}"


def _parse_sft_listwise_pred_ranking(text: str) -> Optional[str]:
    ranking = lw._normalize_ranking_text(str(text or ""))
    if ranking in lw.RANKING_TO_LABEL:
        return ranking
    for candidate in lw.RANKING_LABELS:
        if candidate in str(text or "").replace(" ", "").upper():
            return candidate
    return None


def _evaluate_listwise_sft(
    *,
    model: Any,
    tokenizer: Any,
    examples: Sequence[lw.ListwiseExample],
    max_length: int,
    batch_size: int,
    max_new_tokens: int,
) -> Dict[str, Any]:
    n = int(len(examples))
    if n <= 0:
        return {"n": 0}

    model.eval()
    true_rankings: List[str] = []
    pred_rankings: List[str] = []
    invalid_pred = 0
    effective_bs = max(1, int(batch_size))

    with torch.no_grad():
        for start in range(0, n, effective_bs):
            batch = list(examples[start : start + effective_bs])
            prompts = [str(x.prompt) for x in batch]
            encoded = [
                tokenizer(str(p), add_special_tokens=True, truncation=False).input_ids
                for p in prompts
            ]
            encoded = [base._truncate_ids_preserve_edges(ids, int(max_length)) for ids in encoded]
            tok = tokenizer.pad({"input_ids": encoded}, padding=True, return_tensors="pt")
            dev = model.device if hasattr(model, "device") else next(model.parameters()).device
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
                pred = _parse_sft_listwise_pred_ranking(pred_text)
                if pred is None:
                    pred = "A=B=C"
                    invalid_pred += 1
                true_rankings.append(str(ex.ranking))
                pred_rankings.append(str(pred))

    y_true = np.asarray([lw.RANKING_TO_LABEL[x] for x in true_rankings], dtype=np.int64)
    y_pred = np.asarray([lw.RANKING_TO_LABEL[x] for x in pred_rankings], dtype=np.int64)
    true_top = [lw._ranking_top_group(x) for x in true_rankings]
    pred_top = [lw._ranking_top_group(x) for x in pred_rankings]
    soft = lw._listwise_soft_metrics(true_rankings, pred_rankings)

    return {
        "n": int(n),
        "sft_acc": float((y_true == y_pred).mean()),
        "sft_top_group_acc": float(np.mean([t == p for t, p in zip(true_top, pred_top)])),
        "sft_pairwise_relation_acc": soft.get("proxy_pairwise_relation_acc"),
        "sft_best_in_pred_top_acc": soft.get("proxy_best_in_pred_top_acc"),
        "sft_rank_mae": soft.get("proxy_rank_mae"),
        "sft_rank_rmse": soft.get("proxy_rank_rmse"),
        "sft_tie_rate": float(np.mean(["=" in x for x in pred_rankings])),
        "sft_invalid_pred": int(invalid_pred),
        "sft_confusion": base._confusion(y_true, y_pred, num_classes=len(lw.RANKING_LABELS)),
        "ranking_labels": list(lw.RANKING_LABELS),
    }


def _train_sft_on_items(
    *,
    model_name_or_path: Optional[str],
    model: Optional[Any],
    tokenizer: Optional[Any],
    items: Sequence[Tuple[str, str, str, int]],
    pointwise_teacher_logits: Optional[Sequence[Optional[Sequence[float]]]] = None,
    class_teacher_logits: Optional[Sequence[Optional[Sequence[float]]]] = None,
    class_teacher_task_ids: Optional[Sequence[int]] = None,
    choice_target_distributions: Optional[Sequence[Optional[Mapping[str, float]]]] = None,
    choice_candidate_targets: Optional[Sequence[Optional[Sequence[str]]]] = None,
    output_dir: Path,
    cfg: RunConfig,
    stage_name: str,
    smooth_initial_hist: Optional[Sequence[float]] = None,
) -> Tuple[Dict[str, Any], Any, Any]:
    import transformers
    from peft import LoraConfig, PeftModel, TaskType, get_peft_model

    fsdp_enabled = _fsdp_enabled(cfg)
    if fsdp_enabled and (bool(cfg.use_lora) or bool(cfg.load_in_4bit)):
        raise ValueError("FSDP full fine-tuning cannot be combined with --use-lora or --load-in-4bit")
    if fsdp_enabled:
        _install_fsdp_forward_method_compat()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

    os.environ.setdefault("PYTHONHASHSEED", str(cfg.seed))
    random.seed(int(cfg.seed))
    np.random.seed(int(cfg.seed))
    torch.manual_seed(int(cfg.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(cfg.seed))

    reused = model is not None and tokenizer is not None
    if not reused:
        if not model_name_or_path:
            raise ValueError("model_name_or_path is required for first SFT stage")
        model, tokenizer, _ = base._load_sft_model_and_tokenizer(
            model_name_or_path=str(model_name_or_path),
            max_length=int(cfg.max_length),
            load_in_4bit=bool(cfg.load_in_4bit),
        )
    else:
        tokenizer.model_max_length = int(cfg.max_length)

    assert model is not None
    assert tokenizer is not None
    # FSDP uses its own activation checkpointing. The model-level variant and
    # Trainer's gradient_checkpointing are intentionally disabled in this mode.
    if not fsdp_enabled:
        model.gradient_checkpointing_enable()
    if hasattr(model, "config"):
        model.config.use_cache = False

    rng = np.random.default_rng(int(cfg.seed) + 711)
    order = rng.permutation(len(items)).astype(np.int64).tolist()
    shuffled = [items[int(i)] for i in order]
    shuffled_teacher_logits: Optional[List[Optional[Sequence[float]]]] = None
    if pointwise_teacher_logits is not None:
        if len(pointwise_teacher_logits) != len(items):
            raise ValueError(
                "pointwise_teacher_logits length must match items length: "
                f"{len(pointwise_teacher_logits)} != {len(items)}"
        )
        shuffled_teacher_logits = [pointwise_teacher_logits[int(i)] for i in order]
    shuffled_class_teacher_logits: Optional[List[Optional[Sequence[float]]]] = None
    shuffled_class_teacher_task_ids: Optional[List[int]] = None
    if class_teacher_logits is not None:
        if len(class_teacher_logits) != len(items):
            raise ValueError(
                "class_teacher_logits length must match items length: "
                f"{len(class_teacher_logits)} != {len(items)}"
            )
        if class_teacher_task_ids is None or len(class_teacher_task_ids) != len(items):
            raise ValueError("class_teacher_task_ids must be provided and match items when class_teacher_logits is used")
        shuffled_class_teacher_logits = [class_teacher_logits[int(i)] for i in order]
        shuffled_class_teacher_task_ids = [int(class_teacher_task_ids[int(i)]) for i in order]
    shuffled_choice_distributions = None
    shuffled_choice_candidates = None
    if choice_target_distributions is not None or choice_candidate_targets is not None:
        if choice_target_distributions is None or choice_candidate_targets is None:
            raise ValueError("choice metadata must be provided together")
        if len(choice_target_distributions) != len(items) or len(choice_candidate_targets) != len(items):
            raise ValueError("choice metadata length must match items length")
        shuffled_choice_distributions = [choice_target_distributions[int(i)] for i in order]
        shuffled_choice_candidates = [choice_candidate_targets[int(i)] for i in order]
    train_sources = [x[1] for x in shuffled]
    train_targets = [x[2] for x in shuffled]
    pointwise_score_labels = [int(x[3]) for x in shuffled]
    pointwise_score_token_ids = base._score_token_ids_for_sft(
        tokenizer,
        score_min=int(cfg.score_min),
        score_max=int(cfg.score_max),
    )
    train_dataset = base.SFTPairwiseDataset(
        train_sources,
        train_targets,
        tokenizer,
        pointwise_score_labels=pointwise_score_labels,
        pointwise_score_token_ids=pointwise_score_token_ids,
        pointwise_teacher_logits=shuffled_teacher_logits,
        class_teacher_logits=shuffled_class_teacher_logits,
        class_teacher_task_ids=shuffled_class_teacher_task_ids,
        choice_target_distributions=shuffled_choice_distributions,
        choice_candidate_targets=shuffled_choice_candidates,
    )

    if bool(cfg.use_lora) and not isinstance(model, PeftModel):
        if bool(cfg.load_in_4bit):
            model = base._prepare_model_for_kbit_lora_sft(model, load_in_4bit=True)
        print(f"Applying LoRA for {stage_name}...")
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )
        model = get_peft_model(model, lora_config)
    elif bool(cfg.use_lora):
        print(f"Model already has LoRA adapters; reusing them for {stage_name}.")

    training_args_kwargs = dict(
        output_dir=str(output_dir),
        do_train=True,
        do_eval=False,
        per_device_train_batch_size=int(cfg.per_device_batch_size),
        gradient_accumulation_steps=int(cfg.gradient_accumulation_steps),
        num_train_epochs=int(
            cfg.pointwise_epochs
            if stage_name == "stage1_pointwise"
            else cfg.pairwise_epochs
            if stage_name == "stage2_pairwise"
            else cfg.stage4_epochs
            if stage_name == "stage4_consolidation"
            else cfg.stage23_epochs
            if stage_name == "stage23_pairwise_listwise"
            else cfg.listwise_epochs
        ),
        learning_rate=float(cfg.learning_rate),
        weight_decay=0.0,
        lr_scheduler_type="cosine",
        warmup_steps=max(
            1,
            int(
                len(train_dataset)
                / max(
                    1,
                    int(cfg.per_device_batch_size)
                    * int(cfg.gradient_accumulation_steps)
                    * _world_size(),
                )
                * 0.1
            ),
        ),
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        logging_steps=10,
        # Full-state FSDP checkpoints gather the full model on rank zero. The
        # stage-final checkpoint below is enough for this sequential pipeline.
        save_strategy="no" if fsdp_enabled else "epoch",
        save_total_limit=2,
        eval_strategy="no",
        dataloader_pin_memory=True,
        gradient_checkpointing=not fsdp_enabled,
        remove_unused_columns=False,
        seed=int(cfg.seed),
    )
    if fsdp_enabled:
        training_args_kwargs["fsdp"] = str(cfg.fsdp)
        training_args_kwargs["fsdp_config"] = {
            "transformer_layer_cls_to_wrap": str(cfg.fsdp_transformer_layer_cls_to_wrap),
            "state_dict_type": str(cfg.fsdp_state_dict_type),
            "activation_checkpointing": bool(cfg.fsdp_activation_checkpointing),
            "use_orig_params": bool(cfg.fsdp_use_orig_params),
            "sync_module_states": True,
            "limit_all_gathers": True,
        }
    try:
        training_args = transformers.TrainingArguments(**training_args_kwargs)
    except TypeError as exc:
        if "eval_strategy" not in str(exc):
            raise
        training_args_kwargs["evaluation_strategy"] = training_args_kwargs.pop("eval_strategy")
        training_args = transformers.TrainingArguments(**training_args_kwargs)

    smooth_stages = {x.strip().lower() for x in str(cfg.pointwise_global_smooth_stages).split(",") if x.strip()}
    stage_aliases = {
        "stage1_pointwise": {"stage1", "pointwise", "stage1_pointwise"},
        "stage2_pairwise": {"stage2", "pairwise", "stage2_pairwise"},
        "stage3_listwise": {"stage3", "listwise", "stage3_listwise"},
        "stage23_pairwise_listwise": {"stage23", "pairwise_listwise", "stage23_pairwise_listwise"},
        "stage4_consolidation": {"stage4", "consolidation", "stage4_consolidation"},
    }
    smooth_stage_enabled = bool({"all", "*"} & smooth_stages) or bool(stage_aliases.get(str(stage_name), set()) & smooth_stages)
    stage_smooth_alpha = float(cfg.pointwise_global_smooth_alpha) if bool(smooth_stage_enabled) else 0.0
    smooth_enabled = float(stage_smooth_alpha) > 0.0
    if smooth_enabled:
        print(
            f"Pointwise SFT score-token smoothing enabled for {stage_name}: "
            f"alpha={float(stage_smooth_alpha)} "
            f"mode={str(cfg.pointwise_global_smooth_mode)} "
            f"gaussian_sigma={float(cfg.pointwise_global_smooth_gaussian_sigma)} "
            f"stages={str(cfg.pointwise_global_smooth_stages)} "
            f"start_step={int(cfg.pointwise_global_smooth_start_step)} "
            f"warmup_steps={int(cfg.pointwise_global_smooth_warmup_steps)} "
            f"start_pointwise_seen={int(cfg.pointwise_global_smooth_start_pointwise_seen)} "
            f"warmup_pointwise_seen={int(cfg.pointwise_global_smooth_warmup_pointwise_seen)} "
            f"prior={float(cfg.pointwise_global_smooth_prior)} "
            f"init_prior_from_stage1={bool(cfg.pointwise_global_smooth_init_prior_from_stage1)} "
            f"freeze_prior={bool(cfg.pointwise_global_smooth_freeze_prior)} "
            f"uniform_mix={float(cfg.pointwise_global_smooth_uniform_mix)} "
            f"adaptive_entropy={bool(cfg.pointwise_global_smooth_adaptive_entropy)} "
            f"trainable_alpha={bool(cfg.pointwise_global_smooth_trainable_alpha)} "
            f"alpha_max={float(cfg.pointwise_global_smooth_alpha_max)} "
            f"alpha_reg={float(cfg.pointwise_global_smooth_alpha_reg)} "
            f"alpha_lr={float(cfg.pointwise_global_smooth_alpha_lr)}",
            flush=True,
        )

    distill_stage_enabled = bool(str(stage_name) in {"stage23_pairwise_listwise", "stage4_consolidation"})
    stage_distill_weight = float(cfg.pointwise_teacher_distill_weight) if bool(distill_stage_enabled) else 0.0
    teacher_distill_samples = (
        int(sum(1 for x in shuffled_teacher_logits if x is not None))
        if shuffled_teacher_logits is not None
        else 0
    )
    if float(stage_distill_weight) > 0.0 and teacher_distill_samples > 0:
        print(
            f"Pointwise teacher distillation enabled for {stage_name}: "
            f"weight={float(stage_distill_weight)} "
            f"temperature={float(cfg.pointwise_teacher_distill_temperature)} "
            f"samples={int(teacher_distill_samples)}",
            flush=True,
        )
    class_distill_samples = (
        int(sum(1 for x in shuffled_class_teacher_logits if x is not None))
        if shuffled_class_teacher_logits is not None
        else 0
    )
    class_distill_weight = (
        float(cfg.stage4_task_teacher_distill_weight)
        if str(stage_name) == "stage4_consolidation"
        else 0.0
    )
    class_candidate_token_ids: Dict[int, List[int]] = {}
    class_label_offsets: Dict[int, int] = {}
    if float(class_distill_weight) > 0.0 and class_distill_samples > 0:
        class_candidate_token_ids, class_label_offsets = _stage4_decision_distill_candidates(tokenizer)
        print(
            f"Task teacher decision distillation enabled for {stage_name}: "
            f"weight={float(class_distill_weight)} "
            f"temperature={float(cfg.stage4_task_teacher_distill_temperature)} "
            f"samples={int(class_distill_samples)} "
            f"task_ids={sorted(class_candidate_token_ids.keys())}",
            flush=True,
        )

    trainer = base.OnlineGlobalPriorSFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=base._data_collator_sft,
        score_token_ids=pointwise_score_token_ids,
        smooth_alpha=float(stage_smooth_alpha),
        smooth_start_step=int(cfg.pointwise_global_smooth_start_step),
        smooth_warmup_steps=int(cfg.pointwise_global_smooth_warmup_steps),
        smooth_start_pointwise_seen=int(cfg.pointwise_global_smooth_start_pointwise_seen),
        smooth_warmup_pointwise_seen=int(cfg.pointwise_global_smooth_warmup_pointwise_seen),
        smooth_prior=float(cfg.pointwise_global_smooth_prior),
        smooth_initial_hist=smooth_initial_hist if smooth_enabled else None,
        smooth_freeze_prior=bool(cfg.pointwise_global_smooth_freeze_prior) and bool(smooth_enabled),
        smooth_uniform_mix=float(cfg.pointwise_global_smooth_uniform_mix),
        smooth_adaptive_entropy=bool(cfg.pointwise_global_smooth_adaptive_entropy),
        smooth_mode=str(cfg.pointwise_global_smooth_mode),
        smooth_gaussian_sigma=float(cfg.pointwise_global_smooth_gaussian_sigma),
        smooth_trainable_alpha=bool(cfg.pointwise_global_smooth_trainable_alpha) and bool(smooth_enabled),
        smooth_alpha_max=float(cfg.pointwise_global_smooth_alpha_max),
        smooth_alpha_reg=float(cfg.pointwise_global_smooth_alpha_reg),
        smooth_alpha_lr=float(cfg.pointwise_global_smooth_alpha_lr),
        pointwise_distill_weight=float(stage_distill_weight),
        pointwise_distill_temperature=float(cfg.pointwise_teacher_distill_temperature),
        class_distill_weight=float(class_distill_weight),
        class_distill_temperature=float(cfg.stage4_task_teacher_distill_temperature),
        class_distill_candidate_token_ids=class_candidate_token_ids,
        class_distill_label_offsets=class_label_offsets,
        choice_temperature=1.0,
    )

    counts: Dict[str, int] = {}
    for task, _, _, _ in items:
        counts[str(task)] = int(counts.get(str(task), 0) + 1)
    if _is_primary_process():
        print(
            f"Training {stage_name} SFT with {len(train_dataset)} samples: {counts} "
            f"(world_size={_world_size()}, fsdp={fsdp_enabled})"
        )
    t0 = time.time()
    trainer.train()
    elapsed = time.time() - t0

    save_stage_model = (
        not fsdp_enabled
        or bool(cfg.fsdp_save_all_stages)
        or str(cfg.eval_stages) == "all"
        or _is_final_training_stage(cfg, stage_name)
    )
    if save_stage_model:
        # Trainer.save_model() performs the FSDP full-state-dict gather
        # collectively. Every rank must execute it; only rank zero writes.
        if fsdp_enabled:
            _distributed_barrier()
        if bool(cfg.use_lora):
            model.save_pretrained(str(output_dir))
        else:
            trainer.save_model(str(output_dir))
        if _is_primary_process():
            tokenizer.save_pretrained(str(output_dir))
        if fsdp_enabled:
            _distributed_barrier()

    return (
        {
            "mode": "generative_sft",
            "stage": str(stage_name),
            "reused_existing_model": bool(reused),
            "train_samples": int(len(train_dataset)),
            "task_counts": counts,
            "epochs": int(training_args_kwargs["num_train_epochs"]),
            "elapsed_sec": float(elapsed),
            "model_dir": str(output_dir),
            "model_saved": bool(save_stage_model),
            "distributed": {
                "fsdp_enabled": bool(fsdp_enabled),
                "world_size": int(_world_size()),
                "global_train_batch_size": int(cfg.per_device_batch_size)
                * int(cfg.gradient_accumulation_steps)
                * int(_world_size()),
            },
            "global_prior_smoothing": trainer.get_global_prior_smoothing_stats(),
            "pointwise_teacher_distill": {
                "enabled": bool(float(stage_distill_weight) > 0.0 and teacher_distill_samples > 0),
                "weight": float(stage_distill_weight),
                "temperature": float(cfg.pointwise_teacher_distill_temperature),
                "samples": int(teacher_distill_samples),
            },
            "task_teacher_decision_distill": {
                "enabled": bool(float(class_distill_weight) > 0.0 and class_distill_samples > 0),
                "weight": float(class_distill_weight),
                "temperature": float(cfg.stage4_task_teacher_distill_temperature),
                "samples": int(class_distill_samples),
                "task_ids": sorted(int(x) for x in set(shuffled_class_teacher_task_ids or []) if int(x) > 0),
            },
            "choice_soft_target": {
                "enabled": bool(shuffled_choice_distributions is not None),
                "rows": int(sum(x is not None for x in (shuffled_choice_distributions or []))),
                "temperature": 1.0,
            },
        },
        trainer.model if fsdp_enabled else model,
        tokenizer,
    )


def _pointwise_items(examples: Sequence[base.PointwiseScoredExample], cfg: RunConfig) -> List[Tuple[str, str, str, int]]:
    return [
        (
            "pointwise",
            str(x.prompt),
            base._pointwise_sft_target(x, fix_score_prefix_in_prompt=bool(cfg.fix_score_prefix_in_prompt)),
            int(x.score) - int(cfg.score_min),
        )
        for x in examples
    ]


def _pairwise_items(examples: Sequence[base.PairwiseExample]) -> List[Tuple[str, str, str, int]]:
    return [
        ("pairwise", str(x.prompt), f"[[{base.label_to_token(int(x.label))}]]{base.DEFAULT_EOS_TOKEN}", base.IGNORE_INDEX)
        for x in examples
    ]


def _listwise_items(examples: Sequence[lw.ListwiseExample]) -> List[Tuple[str, str, str, int]]:
    return [("listwise", str(x.prompt), _listwise_sft_target(x), base.IGNORE_INDEX) for x in examples]


def _with_pointwise_replay(
    main_items: Sequence[Tuple[str, str, str, int]],
    pointwise_items: Sequence[Tuple[str, str, str, int]],
    *,
    replay_ratio: int,
    seed: int,
    reference_count: Optional[int] = None,
) -> List[Tuple[str, str, str, int]]:
    out = list(main_items)
    ratio = max(0, int(replay_ratio))
    if ratio <= 0 or not pointwise_items:
        return out
    base_count = len(pointwise_items) if reference_count is None else max(0, int(reference_count))
    total = int(base_count * ratio)
    rng = np.random.default_rng(int(seed))
    indices: List[int] = []
    while len(indices) < total:
        indices.extend(int(i) for i in rng.permutation(len(pointwise_items)).astype(np.int64).tolist())
    out.extend(pointwise_items[int(i)] for i in indices[:total])
    return out


def _resample_items_with_aux(
    items: Sequence[Tuple[str, str, str, int]],
    aux: Optional[Sequence[Optional[Sequence[float]]]],
    *,
    target_count: int,
    seed: int,
) -> Tuple[List[Tuple[str, str, str, int]], List[Optional[Sequence[float]]]]:
    total = max(0, int(target_count))
    if total <= 0:
        return [], []
    if not items:
        raise ValueError("cannot sample from an empty item list when target_count > 0")
    if aux is not None and len(aux) != len(items):
        raise ValueError(f"aux length must match items length: {len(aux)} != {len(items)}")
    rng = np.random.default_rng(int(seed))
    indices: List[int] = []
    while len(indices) < total:
        indices.extend(int(i) for i in rng.permutation(len(items)).astype(np.int64).tolist())
    selected_indices = indices[:total]
    selected_items = [items[int(i)] for i in selected_indices]
    selected_aux = [aux[int(i)] if aux is not None else None for i in selected_indices]
    return selected_items, selected_aux


def _build_weighted_stage23_items(
    *,
    pair_items: Sequence[Tuple[str, str, str, int]],
    list_items: Sequence[Tuple[str, str, str, int]],
    point_items: Sequence[Tuple[str, str, str, int]],
    pointwise_teacher_logits: Optional[Sequence[Optional[Sequence[float]]]],
    cfg: RunConfig,
    seed: int,
) -> Tuple[List[Tuple[str, str, str, int]], List[Optional[Sequence[float]]], Dict[str, Any]]:
    reference_count = len(list_items)
    if reference_count <= 0:
        raise ValueError("list_items must be non-empty for Stage23 mixed training")
    pair_count = int(round(float(reference_count) * float(cfg.stage23_pairwise_weight)))
    list_count = int(round(float(reference_count) * float(cfg.stage23_listwise_weight)))
    point_count = int(
        round(
            float(reference_count)
            * float(cfg.stage23_pointwise_weight)
            * float(cfg.stage23_pointwise_replay_ratio)
        )
    )
    sampled_pair, pair_aux = _resample_items_with_aux(
        pair_items,
        None,
        target_count=pair_count,
        seed=int(seed) + 1,
    )
    sampled_list, list_aux = _resample_items_with_aux(
        list_items,
        None,
        target_count=list_count,
        seed=int(seed) + 2,
    )
    sampled_point, point_aux = _resample_items_with_aux(
        point_items,
        pointwise_teacher_logits,
        target_count=point_count,
        seed=int(seed) + 3,
    )
    mixed_items = sampled_pair + sampled_list + sampled_point
    mixed_aux = pair_aux + list_aux + point_aux
    stats = {
        "reference_count": int(reference_count),
        "stage23_pairwise_weight": float(cfg.stage23_pairwise_weight),
        "stage23_listwise_weight": float(cfg.stage23_listwise_weight),
        "stage23_pointwise_weight": float(cfg.stage23_pointwise_weight),
        "stage23_pointwise_replay_ratio": int(cfg.stage23_pointwise_replay_ratio),
        "target_counts": {
            "pairwise": int(pair_count),
            "listwise": int(list_count),
            "pointwise": int(point_count),
            "total": int(len(mixed_items)),
        },
        "source_counts": {
            "pairwise": int(len(pair_items)),
            "listwise": int(len(list_items)),
            "pointwise": int(len(point_items)),
        },
        "pointwise_teacher_logits": {
            "available": bool(pointwise_teacher_logits is not None),
            "sampled": int(sum(1 for x in point_aux if x is not None)),
        },
    }
    return mixed_items, mixed_aux, stats


def _infer_model_device(model: Any) -> torch.device:
    device = getattr(model, "device", None)
    if device is not None:
        return torch.device(device)
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _first_diff_token_spec(tokenizer: Any, candidates: Sequence[str]) -> Tuple[List[int], int, List[List[int]]]:
    eos_token = str(getattr(tokenizer, "eos_token", None) or "")
    encoded: List[List[int]] = []
    for text in candidates:
        normalized = str(text).replace(base.DEFAULT_EOS_TOKEN, eos_token) if eos_token else str(text)
        ids = tokenizer(normalized, add_special_tokens=False).input_ids
        encoded.append([int(x) for x in ids])
    if not encoded or any(not ids for ids in encoded):
        raise ValueError("decision distill candidates must tokenize to non-empty sequences")
    min_len = min(len(ids) for ids in encoded)
    offset = 0
    while offset < min_len and len({ids[offset] for ids in encoded}) == 1:
        offset += 1
    if offset >= min_len:
        raise ValueError(f"decision distill candidates have no differing token: {candidates}")
    token_ids = [int(ids[offset]) for ids in encoded]
    return token_ids, int(offset), encoded


def _stage4_decision_distill_candidates(tokenizer: Any) -> Tuple[Dict[int, List[int]], Dict[int, int]]:
    pair_candidates = [
        f"[[{base.label_to_token(int(base.LABEL_A))}]]{base.DEFAULT_EOS_TOKEN}",
        f"[[{base.label_to_token(int(base.LABEL_B))}]]{base.DEFAULT_EOS_TOKEN}",
        f"[[{base.label_to_token(int(base.LABEL_TIE))}]]{base.DEFAULT_EOS_TOKEN}",
    ]
    # Light-weight listwise DER: preserve the Stage-3 distribution over the first ranked answer.
    list_top_candidates = ["Ranking:[A", "Ranking:[B", "Ranking:[C"]
    pair_token_ids, pair_offset, _ = _first_diff_token_spec(tokenizer, pair_candidates)
    list_token_ids, list_offset, _ = _first_diff_token_spec(tokenizer, list_top_candidates)
    return (
        {
            int(CLASS_TEACHER_TASK_PAIRWISE): [int(x) for x in pair_token_ids],
            int(CLASS_TEACHER_TASK_LISTWISE_TOP): [int(x) for x in list_token_ids],
        },
        {
            int(CLASS_TEACHER_TASK_PAIRWISE): int(pair_offset),
            int(CLASS_TEACHER_TASK_LISTWISE_TOP): int(list_offset),
        },
    )


def _compute_decision_teacher_logits(
    *,
    model: Any,
    tokenizer: Any,
    items: Sequence[Tuple[str, str, str, int]],
    candidate_token_ids: Sequence[int],
    label_offset: int,
    task_id: int,
    cfg: RunConfig,
    output_dir: Path,
    name: str,
) -> Tuple[List[Optional[List[float]]], Dict[str, Any]]:
    if not items:
        return [], {"enabled": False, "samples": 0}
    sources = [x[1] for x in items]
    targets = [x[2] for x in items]
    score_labels = [int(x[3]) for x in items]
    dataset = base.SFTPairwiseDataset(
        sources,
        targets,
        tokenizer,
        pointwise_score_labels=score_labels,
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=max(1, int(cfg.eval_batch_size)),
        shuffle=False,
        collate_fn=base._data_collator_sft,
    )
    device = _infer_model_device(model)
    token_index = torch.tensor([int(x) for x in candidate_token_ids], device=device, dtype=torch.long)
    model.eval()
    cached_logits: List[Optional[List[float]]] = []
    rows: List[Dict[str, Any]] = []
    offset = 0
    with torch.inference_mode():
        for batch in loader:
            input_ids = batch["input_ids"].to(device=device)
            attention_mask = batch["attention_mask"].to(device=device)
            labels = batch["labels"].to(device=device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            shift_logits = outputs.logits[..., :-1, :]
            label_mask = labels.ne(base.IGNORE_INDEX)
            has_label = label_mask.any(dim=1)
            first_label_pos = label_mask.to(dtype=torch.int64).argmax(dim=1)
            decision_pos = first_label_pos + int(label_offset) - 1
            valid = has_label & (decision_pos >= 0) & (decision_pos < int(shift_logits.size(1)))
            for row_idx in range(int(input_ids.size(0))):
                global_idx = int(offset + row_idx)
                if bool(valid[row_idx].item()):
                    values = (
                        shift_logits[row_idx, int(decision_pos[row_idx].item()), :]
                        .index_select(dim=-1, index=token_index)
                        .detach()
                        .float()
                        .cpu()
                        .tolist()
                    )
                    values = [float(x) for x in values]
                else:
                    values = None
                cached_logits.append(values)
                rows.append(
                    {
                        "index": int(global_idx),
                        "task": str(items[global_idx][0]),
                        "class_teacher_task_id": int(task_id),
                        "teacher_decision_logits": values,
                    }
                )
            offset += int(input_ids.size(0))
    _write_jsonl(output_dir / f"{name}_teacher_decision_logits.jsonl", rows)
    stats = {
        "enabled": True,
        "name": str(name),
        "samples": int(len(cached_logits)),
        "valid_samples": int(sum(1 for x in cached_logits if x is not None)),
        "class_teacher_task_id": int(task_id),
        "candidate_token_ids": [int(x) for x in candidate_token_ids],
        "label_offset": int(label_offset),
    }
    _write_json(output_dir / f"{name}_teacher_decision_logits_stats.json", stats)
    return cached_logits, stats


def _compute_pointwise_teacher_logits(
    *,
    model: Any,
    tokenizer: Any,
    items: Sequence[Tuple[str, str, str, int]],
    cfg: RunConfig,
    output_dir: Path,
) -> Tuple[List[Optional[List[float]]], Dict[str, Any]]:
    if not items:
        return [], {"enabled": False, "samples": 0}
    score_token_ids = base._score_token_ids_for_sft(
        tokenizer,
        score_min=int(cfg.score_min),
        score_max=int(cfg.score_max),
    )
    first_score_token_ids = [int(ids[0]) for ids in score_token_ids]
    multi_token_scores = [
        int(cfg.score_min) + int(i)
        for i, ids in enumerate(score_token_ids)
        if len(ids) != 1
    ]
    sources = [x[1] for x in items]
    targets = [x[2] for x in items]
    score_labels = [int(x[3]) for x in items]
    dataset = base.SFTPairwiseDataset(
        sources,
        targets,
        tokenizer,
        pointwise_score_labels=score_labels,
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=max(1, int(cfg.eval_batch_size)),
        shuffle=False,
        collate_fn=base._data_collator_sft,
    )
    device = _infer_model_device(model)
    model.eval()
    cached_logits: List[Optional[List[float]]] = []
    rows: List[Dict[str, Any]] = []
    offset = 0
    token_index = torch.tensor(first_score_token_ids, device=device, dtype=torch.long)
    with torch.inference_mode():
        for batch in loader:
            input_ids = batch["input_ids"].to(device=device)
            attention_mask = batch["attention_mask"].to(device=device)
            labels = batch["labels"].to(device=device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            shift_logits = outputs.logits[..., :-1, :]
            label_mask = labels.ne(base.IGNORE_INDEX)
            has_label = label_mask.any(dim=1)
            first_label_pos = label_mask.to(dtype=torch.int64).argmax(dim=1)
            valid = has_label & (first_label_pos > 0)
            batch_logits: List[Optional[List[float]]] = []
            for row_idx in range(int(input_ids.size(0))):
                global_idx = int(offset + row_idx)
                if bool(valid[row_idx].item()):
                    score_shift_pos = int(first_label_pos[row_idx].item()) - 1
                    values = (
                        shift_logits[row_idx, score_shift_pos, :]
                        .index_select(dim=-1, index=token_index)
                        .detach()
                        .float()
                        .cpu()
                        .tolist()
                    )
                    values = [float(x) for x in values]
                else:
                    values = None
                batch_logits.append(values)
                rows.append(
                    {
                        "index": int(global_idx),
                        "task": str(items[global_idx][0]),
                        "score_label": int(items[global_idx][3]),
                        "teacher_score_logits": values,
                    }
                )
            cached_logits.extend(batch_logits)
            offset += int(input_ids.size(0))
    _write_jsonl(output_dir / "stage1_pointwise_teacher_logits.jsonl", rows)
    stats = {
        "enabled": True,
        "samples": int(len(cached_logits)),
        "valid_samples": int(sum(1 for x in cached_logits if x is not None)),
        "score_min": int(cfg.score_min),
        "score_max": int(cfg.score_max),
        "score_token_ids": [[int(x) for x in ids] for ids in score_token_ids],
        "first_score_token_ids": [int(x) for x in first_score_token_ids],
        "multi_token_scores": [int(x) for x in multi_token_scores],
        "logit_source": "stage1_model_first_target_score_token",
    }
    _write_json(output_dir / "stage1_pointwise_teacher_logits_stats.json", stats)
    return cached_logits, stats


def _pointwise_teacher_item_key(item: Tuple[str, str, str, int]) -> Tuple[str, str, str, int]:
    return (str(item[0]), str(item[1]), str(item[2]), int(item[3]))


def _align_pointwise_teacher_logits_for_mixed_items(
    *,
    teacher_items: Sequence[Tuple[str, str, str, int]],
    teacher_logits: Optional[Sequence[Optional[Sequence[float]]]],
    mixed_items: Sequence[Tuple[str, str, str, int]],
) -> Tuple[Optional[List[Optional[Sequence[float]]]], Dict[str, Any]]:
    if teacher_logits is None:
        return None, {"enabled": False, "reason": "no_teacher_logits"}
    if len(teacher_items) != len(teacher_logits):
        raise ValueError(
            "teacher_items and teacher_logits length mismatch: "
            f"{len(teacher_items)} != {len(teacher_logits)}"
        )

    by_key: Dict[Tuple[str, str, str, int], List[Optional[Sequence[float]]]] = {}
    for item, logits in zip(teacher_items, teacher_logits):
        by_key.setdefault(_pointwise_teacher_item_key(item), []).append(logits)

    aligned: List[Optional[Sequence[float]]] = []
    matched = 0
    valid = 0
    missing = 0
    non_pointwise = 0
    for item in mixed_items:
        if str(item[0]) != "pointwise" or int(item[3]) < 0:
            aligned.append(None)
            non_pointwise += 1
            continue
        key = _pointwise_teacher_item_key(item)
        bucket = by_key.get(key)
        if bucket:
            logits = bucket.pop(0)
            aligned.append(logits)
            matched += 1
            if logits is not None:
                valid += 1
        else:
            aligned.append(None)
            missing += 1

    stats = {
        "enabled": True,
        "teacher_items": int(len(teacher_items)),
        "mixed_items": int(len(mixed_items)),
        "pointwise_items": int(matched + missing),
        "matched_pointwise_items": int(matched),
        "missing_pointwise_items": int(missing),
        "valid_teacher_logits": int(valid),
        "non_pointwise_items": int(non_pointwise),
        "teacher_source": "stage1_pointwise",
    }
    return aligned, stats


def _align_class_teacher_logits_for_mixed_items(
    *,
    teacher_items: Sequence[Tuple[str, str, str, int]],
    teacher_logits: Optional[Sequence[Optional[Sequence[float]]]],
    mixed_items: Sequence[Tuple[str, str, str, int]],
    task_name: str,
    task_id: int,
) -> Tuple[List[Optional[Sequence[float]]], List[int], Dict[str, Any]]:
    aligned: List[Optional[Sequence[float]]] = [None for _ in mixed_items]
    task_ids: List[int] = [0 for _ in mixed_items]
    if teacher_logits is None:
        return aligned, task_ids, {"enabled": False, "reason": "no_teacher_logits", "task": str(task_name)}
    if len(teacher_items) != len(teacher_logits):
        raise ValueError(
            "teacher_items and teacher_logits length mismatch: "
            f"{len(teacher_items)} != {len(teacher_logits)}"
        )
    by_key: Dict[Tuple[str, str, str, int], List[Optional[Sequence[float]]]] = {}
    for item, logits in zip(teacher_items, teacher_logits):
        by_key.setdefault(_pointwise_teacher_item_key(item), []).append(logits)

    matched = 0
    valid = 0
    missing = 0
    non_task = 0
    for i, item in enumerate(mixed_items):
        if str(item[0]) != str(task_name):
            non_task += 1
            continue
        bucket = by_key.get(_pointwise_teacher_item_key(item))
        if bucket:
            logits = bucket.pop(0)
            aligned[i] = logits
            task_ids[i] = int(task_id) if logits is not None else 0
            matched += 1
            if logits is not None:
                valid += 1
        else:
            missing += 1
    stats = {
        "enabled": True,
        "task": str(task_name),
        "class_teacher_task_id": int(task_id),
        "teacher_items": int(len(teacher_items)),
        "mixed_items": int(len(mixed_items)),
        "task_items": int(matched + missing),
        "matched_task_items": int(matched),
        "missing_task_items": int(missing),
        "valid_teacher_logits": int(valid),
        "non_task_items": int(non_task),
    }
    return aligned, task_ids, stats


def _score_sft_items_by_loss(
    *,
    model: Any,
    tokenizer: Any,
    items: Sequence[Tuple[str, str, str, int]],
    cfg: RunConfig,
) -> List[float]:
    if not items:
        return []
    dataset = base.SFTPairwiseDataset(
        [x[1] for x in items],
        [x[2] for x in items],
        tokenizer,
        pointwise_score_labels=[int(x[3]) for x in items],
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=max(1, int(cfg.eval_batch_size)),
        shuffle=False,
        collate_fn=base._data_collator_sft,
    )
    device = _infer_model_device(model)
    model.eval()
    losses: List[float] = []
    with torch.inference_mode():
        for batch in loader:
            input_ids = batch["input_ids"].to(device=device)
            attention_mask = batch["attention_mask"].to(device=device)
            labels = batch["labels"].to(device=device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            shift_logits = outputs.logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            flat = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=base.IGNORE_INDEX,
                reduction="none",
            )
            token_loss = flat.view_as(shift_labels)
            mask = shift_labels.ne(base.IGNORE_INDEX)
            per_sample = token_loss.sum(dim=1) / mask.sum(dim=1).clamp_min(1)
            losses.extend(float(x) for x in per_sample.detach().float().cpu().tolist())
    return losses


def _select_stage4_loss_triples(
    *,
    model: Any,
    tokenizer: Any,
    train_triples: Sequence[lw.SelectedQuestionTriple],
    selected_rows: Sequence[Dict[str, Any]],
    fraction: float,
    cfg: RunConfig,
) -> Tuple[List[lw.SelectedQuestionTriple], List[Dict[str, Any]], Dict[str, Any]]:
    n = int(len(train_triples))
    if n <= 0:
        return [], [], {"enabled": False, "reason": "no_train_triples"}
    frac = float(fraction)
    if not (0.0 < frac <= 1.0):
        raise ValueError("stage4-replay-fraction must be in (0, 1] when loss_triple replay is enabled")
    k = min(n, max(1, int(round(float(n) * frac))))

    all_items: List[Tuple[str, str, str, int]] = []
    item_meta: List[Tuple[int, str]] = []
    for i, triple in enumerate(train_triples):
        pw, _, _ = lw._build_pointwise_examples_from_triples(
            [triple],
            score_min=int(cfg.score_min),
            score_max=int(cfg.score_max),
            fix_score_prefix_in_prompt=bool(cfg.fix_score_prefix_in_prompt),
        )
        pair, _, _ = _build_pairwise_examples_from_triples(
            [triple],
            order_augmentation=bool(cfg.pairwise_order_augmentation),
        )
        li, _, _ = lw._build_listwise_examples_from_triples(
            [triple],
            order_augmentation=bool(cfg.listwise_order_augmentation),
        )
        for item in _pointwise_items(pw, cfg):
            all_items.append(item)
            item_meta.append((int(i), "pointwise"))
        for item in _pairwise_items(pair):
            all_items.append(item)
            item_meta.append((int(i), "pairwise"))
        for item in _listwise_items(li):
            all_items.append(item)
            item_meta.append((int(i), "listwise"))

    losses = _score_sft_items_by_loss(model=model, tokenizer=tokenizer, items=all_items, cfg=cfg)
    by_triple_task: Dict[int, Dict[str, List[float]]] = {int(i): {} for i in range(n)}
    for (triple_i, task), loss_v in zip(item_meta, losses):
        by_triple_task[int(triple_i)].setdefault(str(task), []).append(float(loss_v))

    scored: List[Tuple[float, int, Dict[str, float]]] = []
    for i in range(n):
        task_means: Dict[str, float] = {}
        for task in ("pointwise", "pairwise", "listwise"):
            vals = by_triple_task[int(i)].get(task, [])
            task_means[task] = float(np.mean(vals)) if vals else 0.0
        score = float(np.mean([task_means["pointwise"], task_means["pairwise"], task_means["listwise"]]))
        scored.append((score, int(i), task_means))
    scored_sorted = sorted(scored, key=lambda x: (-float(x[0]), int(x[1])))
    chosen_indices = sorted(int(i) for _, i, _ in scored_sorted[:k])
    chosen_set = {int(i) for i in chosen_indices}
    replay_triples = [train_triples[int(i)] for i in chosen_indices]

    replay_rows: List[Dict[str, Any]] = []
    score_by_index = {int(i): float(score) for score, i, _ in scored}
    task_means_by_index = {int(i): means for _, i, means in scored}
    for order, idx in enumerate(chosen_indices):
        row = dict(selected_rows[int(idx)]) if int(idx) < len(selected_rows) else {}
        triple = train_triples[int(idx)]
        score_range = _score_range_for_triple(triple)
        row.update(
            {
                "stage4_replay_order": int(order),
                "stage4_source_index": int(idx),
                "stage4_replay_strategy": "loss_triple",
                "stage4_replay_stratum": f"{_ranking_for_triple(triple)}|{_score_range_bucket(score_range)}",
                "stage4_replay_score_range": int(score_range),
                "stage4_replay_score_range_bucket": _score_range_bucket(score_range),
                "stage4_replay_ranking": _ranking_for_triple(triple),
                "stage4_loss_score": float(score_by_index[int(idx)]),
                "stage4_loss_task_means": task_means_by_index[int(idx)],
            }
        )
        replay_rows.append(row)

    selected_scores = [float(score_by_index[int(i)]) for i in chosen_indices]
    unselected_scores = [float(score_by_index[int(i)]) for i in range(n) if int(i) not in chosen_set]
    stats = {
        "enabled": True,
        "strategy": "loss_triple",
        "fraction": float(frac),
        "input_triples": int(n),
        "selected_triples": int(len(replay_triples)),
        "selected_fraction": float(len(replay_triples) / max(1, n)),
        "scoring": "post_stage3_teacher_forced_sft_loss_task_mean",
        "scored_items": int(len(all_items)),
        "selected_loss_mean": float(np.mean(selected_scores)) if selected_scores else 0.0,
        "selected_loss_min": float(np.min(selected_scores)) if selected_scores else 0.0,
        "selected_loss_max": float(np.max(selected_scores)) if selected_scores else 0.0,
        "unselected_loss_mean": float(np.mean(unselected_scores)) if unselected_scores else 0.0,
    }
    return replay_triples, replay_rows, stats


def _compact_metrics(summary: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {"mode": summary.get("mode"), "budget": summary.get("train_budget", {})}
    out["pointwise"] = {}
    out["pairwise"] = {}
    out["listwise"] = {}
    for stage, m in summary.get("pointwise_metrics", {}).items():
        out["pointwise"][stage] = {
            "n": m.get("n"),
            "acc": m.get("sft_acc"),
            "within1": m.get("sft_within1"),
            "mae": m.get("sft_mae"),
            "invalid_pred": m.get("sft_invalid_pred"),
        }
    for stage, m in summary.get("pairwise_metrics", {}).items():
        out["pairwise"][stage] = {
            "n": m.get("n"),
            "acc": m.get("sft_acc"),
            "tie_rate": m.get("sft_tie_rate"),
            "invalid_pred": m.get("sft_invalid_pred"),
        }
    for stage, m in summary.get("listwise_metrics", {}).items():
        out["listwise"][stage] = {
            "n": m.get("n"),
            "acc": m.get("sft_acc"),
            "top_group_acc": m.get("sft_top_group_acc"),
            "pairwise_relation_acc": m.get("sft_pairwise_relation_acc"),
            "best_in_pred_top_acc": m.get("sft_best_in_pred_top_acc"),
            "rank_mae": m.get("sft_rank_mae"),
            "tie_rate": m.get("sft_tie_rate"),
            "invalid_pred": m.get("sft_invalid_pred"),
        }
    return out


def _load_stage1_resume_model(
    *,
    base_model_path: str,
    adapter_dir: Path,
    cfg: RunConfig,
) -> Tuple[Any, Any]:
    from peft import PeftModel

    if not adapter_dir.is_dir():
        raise FileNotFoundError(f"resume stage1 model dir not found: {adapter_dir}")
    model, tokenizer, _ = base._load_sft_model_and_tokenizer(
        model_name_or_path=str(base_model_path),
        max_length=int(cfg.max_length),
        load_in_4bit=bool(cfg.load_in_4bit),
    )
    if bool(cfg.use_lora):
        if bool(cfg.load_in_4bit):
            model = base._prepare_model_for_kbit_lora_sft(model, load_in_4bit=True)
        model = PeftModel.from_pretrained(model, str(adapter_dir), is_trainable=True)
    else:
        raise ValueError("--resume-stage1-model-dir currently expects --use-lora stage1 adapters")
    tokenizer.model_max_length = int(cfg.max_length)
    tokenizer.padding_side = "left"
    return model, tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pointwise-5answers-dataset", default="train_with_selector/train_with_selector/data/newnew/train-20k.json")
    parser.add_argument("--listwise-eval-dataset", default="train_with_selector/train_with_selector/data/newnew/val-2k-eval-listwise.json")
    parser.add_argument("--pairwise-eval-dataset", default="")
    parser.add_argument("--llama", default="llama/Meta-Llama-3-8B-Instruct/")
    parser.add_argument("--out", default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--val-split-seed", type=int, default=55)
    parser.add_argument("--pointwise-val-answer-seed", type=int, default=65)
    parser.add_argument("--train-selection-mode", choices=["selected_triple", "candidate_triple_selector"], default="candidate_triple_selector")
    parser.add_argument("--fixed-selected-triples-path", default="")
    parser.add_argument(
        "--resume-stage1-model-dir",
        default="",
        help="Load a saved Stage-1 LoRA adapter and continue with later stages, usually with fixed selected triples.",
    )
    parser.add_argument("--triple-selection-strategy", choices=["random", "first_three"], default="random")
    parser.add_argument("--question-selection-strategy", choices=["random", "first"], default="random")
    parser.add_argument("--no-randomize-listwise-order", action="store_true")
    parser.add_argument(
        "--candidate-selector-kind",
        choices=["bert", "random", "pointwise_proxy", "bias_trap_pointwise", "shared_llama", "shared_llama_two_stage"],
        default="bias_trap_pointwise",
    )
    parser.add_argument("--candidate-selector-init-triples", type=int, default=50)
    parser.add_argument("--candidate-selector-batch-size", type=int, default=20)
    parser.add_argument("--candidate-selector-epochs", type=int, default=4)
    parser.add_argument("--candidate-selector-max-score-candidates", type=int, default=4096)
    parser.add_argument("--candidate-selector-llama-rerank-candidates", type=int, default=1000)
    parser.add_argument("--candidate-selector-buffer-maxlen", type=int, default=1000)
    parser.add_argument("--candidate-selector-one-per-question", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--candidate-selector-target-task", choices=["pointwise", "listwise"], default="pointwise")
    parser.add_argument("--candidate-selector-score-range-weight", type=float, default=0.0)
    parser.add_argument("--candidate-selector-gap-sum-weight", type=float, default=0.0)
    parser.add_argument("--candidate-selector-uncertainty-weight", type=float, default=0.25)
    parser.add_argument("--candidate-selector-pairwise-uncertainty-weight", type=float, default=0.0)
    parser.add_argument("--candidate-selector-listwise-uncertainty-weight", type=float, default=0.0)
    parser.add_argument("--candidate-selector-kl-weight", type=float, default=0.0)
    parser.add_argument("--candidate-selector-score-bin-weight", type=float, default=0.0)
    parser.add_argument("--candidate-selector-diversity-weight", type=float, default=1.0)
    parser.add_argument("--candidate-selector-density-weight", type=float, default=0.15)
    parser.add_argument("--candidate-selector-bias-weight", type=float, default=1.0)
    parser.add_argument("--candidate-selector-coverage-weight", type=float, default=0.10)
    parser.add_argument("--candidate-selector-pointwise-length-bias-weight", type=float, default=0.5)
    parser.add_argument("--candidate-selector-pairwise-position-bias-weight", type=float, default=0.5)
    parser.add_argument("--candidate-selector-pairwise-position-pairs", type=int, default=1)
    parser.add_argument("--candidate-selector-pairwise-position-bias-scale", type=float, default=0.02)
    parser.add_argument(
        "--candidate-selector-signal-normalization",
        choices=["none", "intrinsic"],
        default="none",
        help="Normalize bias-trap score components with their intrinsic scales before weighted sum.",
    )
    parser.add_argument(
        "--candidate-selector-uncertainty-view",
        choices=["pointwise", "pairwise", "listwise", "joint"],
        default="pointwise",
        help="Uncertainty signal used by bias_trap_pointwise: pointwise score entropy, pairwise choice entropy, or listwise ranking entropy.",
    )
    parser.add_argument(
        "--candidate-selector-length-aug-suffix",
        default="Additional context: This repeats the same answer without adding new useful information.",
    )
    parser.add_argument("--candidate-selector-density-k", type=int, default=10)
    parser.add_argument(
        "--candidate-selector-embedding-model",
        default=lw.DEFAULT_SELECTOR_EMBEDDING_MODEL,
        help="Transformer encoder used as femb(q, ai) for bias_trap_pointwise diversity/density/prefix matching.",
    )
    parser.add_argument("--candidate-selector-embedding-max-length", type=int, default=lw.DEFAULT_SELECTOR_EMBEDDING_MAX_LENGTH)
    parser.add_argument("--candidate-selector-embedding-batch-size", type=int, default=64)
    parser.add_argument("--candidate-selector-embedding-device", default="auto")
    parser.add_argument("--candidate-selector-embedding-pooling", choices=["cls", "mean"], default="cls")
    parser.add_argument(
        "--candidate-selector-diversity-view",
        choices=["pointwise", "joint"],
        default="pointwise",
        help="pointwise uses mean femb(q,a_i); joint averages pointwise, three pairwise prompts, and one listwise prompt.",
    )
    parser.add_argument(
        "--candidate-selector-exploration-ratio",
        type=float,
        default=0.1,
        help="Random fraction of each pointwise_proxy query batch.",
    )
    parser.add_argument("--candidate-selector-entropy-weight", type=float, default=0.5)
    parser.add_argument("--candidate-selector-score-std-weight", type=float, default=0.5)
    parser.add_argument(
        "--candidate-selector-predicted-coverage-weight",
        type=float,
        default=0.2,
        help="Pointwise score-bin coverage bonus computed from proxy predictions.",
    )
    parser.add_argument(
        "--candidate-selector-proxy-warmup-epochs",
        type=int,
        default=3,
        help="Pointwise proxy passes over the random initialization set.",
    )
    parser.add_argument(
        "--candidate-selector-proxy-update-epochs",
        type=int,
        default=1,
        help="Pointwise proxy passes over each newly queried batch.",
    )
    parser.add_argument(
        "--candidate-selector-proxy-mode",
        choices=["classifier_heads", "lm_head"],
        default="classifier_heads",
        help="Internal pointwise_proxy acquisition model. lm_head trains in score-token space.",
    )
    parser.add_argument(
        "--reuse-selection-proxy-for-stage1",
        action="store_true",
        help=(
            "Keep an lm_head pointwise_proxy/bias_trap_pointwise proxy after selection "
            "and treat it as completed Stage-1 pointwise training."
        ),
    )
    parser.add_argument("--candidate-bert-selector-model", default="bert-base-uncased")
    parser.add_argument("--candidate-bert-selector-max-length", type=int, default=512)
    parser.add_argument("--candidate-bert-selector-unfreeze", action="store_true")
    parser.add_argument("--candidate-bert-selector-unfreeze-last-n-layers", type=int, default=0)
    parser.add_argument("--proxy-lr", type=float, default=1e-4)
    parser.add_argument("--proxy-max-length", type=int, default=768)
    parser.add_argument("--llama-multitask-mode", choices=["shared_head", "classifier_heads"], default="shared_head")
    parser.add_argument("--pointwise-loss-type", choices=["ce", "ce_distance"], default="ce")
    parser.add_argument("--pointwise-distance-weight", type=float, default=0.0)
    parser.add_argument("--pointwise-class-weight-mode", choices=["none", "balanced"], default="none")
    parser.add_argument("--pointwise-class-weight-strength", type=float, default=1.0)
    parser.add_argument("--budget-units", type=int, default=750)
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
    parser.add_argument("--stage2-pointwise-replay-ratio", type=int, default=1)
    parser.add_argument("--stage3-pointwise-replay-ratio", type=int, default=1)
    parser.add_argument("--stage3-pairwise-replay-ratio", type=int, default=0)
    parser.add_argument(
        "--merge-stage2-stage3",
        action="store_true",
        help="After Stage 1, train one mixed pairwise+listwise stage with pointwise replay.",
    )
    parser.add_argument("--stage23-pointwise-replay-ratio", type=int, default=1)
    parser.add_argument(
        "--stage23-pairwise-weight",
        type=float,
        default=1.0,
        help="Stage23 pairwise target count as a multiplier of the listwise train count.",
    )
    parser.add_argument(
        "--stage23-listwise-weight",
        type=float,
        default=1.0,
        help="Stage23 listwise target count as a multiplier of the listwise train count.",
    )
    parser.add_argument(
        "--stage23-pointwise-weight",
        type=float,
        default=1.0,
        help="Stage23 pointwise replay target count multiplier; combined with stage23-pointwise-replay-ratio.",
    )
    parser.add_argument("--stage23-epochs", type=int, default=1)
    parser.add_argument(
        "--pointwise-teacher-distill-weight",
        type=float,
        default=0.0,
        help="Optional LwF/DER-style KL weight from the Stage-1 pointwise teacher on Stage23 pointwise replay samples.",
    )
    parser.add_argument(
        "--pointwise-teacher-distill-temperature",
        type=float,
        default=2.0,
        help="Temperature for Stage-1 pointwise teacher score-logit distillation.",
    )
    parser.add_argument(
        "--stage4-task-teacher-distill-weight",
        type=float,
        default=0.0,
        help="Optional Stage4 DER-style KL on pairwise decision and listwise top-answer teacher logits.",
    )
    parser.add_argument(
        "--stage4-task-teacher-distill-temperature",
        type=float,
        default=2.0,
        help="Temperature for Stage4 pairwise/listwise decision teacher distillation.",
    )
    parser.add_argument(
        "--stage4-replay-strategy",
        choices=["none", "random_triple", "stratified_triple", "loss_triple"],
        default="none",
        help="Optional post-hoc mixed consolidation over a fraction of selected triples.",
    )
    parser.add_argument("--stage4-replay-fraction", type=float, default=0.25)
    parser.add_argument("--stage4-epochs", type=int, default=1)
    parser.add_argument(
        "--stage4-listwise-multiplier",
        type=int,
        default=1,
        help="Repeat Stage4 listwise replay examples to increase their consolidation weight.",
    )
    parser.add_argument("--no-pairwise-order-augmentation", action="store_true")
    parser.add_argument("--no-listwise-order-augmentation", action="store_true")
    parser.add_argument("--score-min", type=int, default=1)
    parser.add_argument("--score-max", type=int, default=10)
    parser.add_argument("--no-fix-score-prefix", action="store_true")
    parser.add_argument("--use-lora", action="store_true")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument(
        "--fsdp",
        default="",
        help="Enable FSDP with torchrun, for example: 'full_shard auto_wrap'. Full fine-tuning only.",
    )
    parser.add_argument(
        "--fsdp-transformer-layer-cls-to-wrap",
        default="Qwen3DecoderLayer",
        help="Transformer block class used by FSDP auto wrapping.",
    )
    parser.add_argument(
        "--fsdp-state-dict-type",
        choices=["FULL_STATE_DICT", "SHARDED_STATE_DICT"],
        default="FULL_STATE_DICT",
    )
    parser.add_argument(
        "--fsdp-activation-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use FSDP activation checkpointing instead of model gradient checkpointing.",
    )
    parser.add_argument(
        "--fsdp-use-orig-params",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep original parameters, required for standard optimizer construction.",
    )
    parser.add_argument(
        "--fsdp-save-all-stages",
        action="store_true",
        help="Save full intermediate FSDP checkpoints; final-only is the default to limit disk use.",
    )
    parser.add_argument("--pointwise-global-smooth-alpha", type=float, default=0.1)
    parser.add_argument("--pointwise-global-smooth-mode", default="local_gaussian")
    parser.add_argument("--pointwise-global-smooth-gaussian-sigma", type=float, default=1.0)
    parser.add_argument("--pointwise-global-smooth-stages", default="all")
    parser.add_argument("--pointwise-global-smooth-start-step", type=int, default=0)
    parser.add_argument("--pointwise-global-smooth-warmup-steps", type=int, default=0)
    parser.add_argument("--pointwise-global-smooth-start-pointwise-seen", type=int, default=0)
    parser.add_argument("--pointwise-global-smooth-warmup-pointwise-seen", type=int, default=0)
    parser.add_argument("--pointwise-global-smooth-prior", type=float, default=1.0)
    parser.add_argument("--pointwise-global-smooth-init-prior-from-stage1", action="store_true")
    parser.add_argument("--pointwise-global-smooth-freeze-prior", action="store_true")
    parser.add_argument("--pointwise-global-smooth-uniform-mix", type=float, default=0.0)
    parser.add_argument("--pointwise-global-smooth-adaptive-entropy", action="store_true")
    parser.add_argument("--pointwise-global-smooth-trainable-alpha", action="store_true")
    parser.add_argument("--pointwise-global-smooth-alpha-max", type=float, default=0.2)
    parser.add_argument("--pointwise-global-smooth-alpha-reg", type=float, default=0.0)
    parser.add_argument("--pointwise-global-smooth-alpha-lr", type=float, default=0.0)
    parser.add_argument("--max-pointwise-eval-samples", type=int, default=0)
    parser.add_argument("--max-pairwise-eval-samples", type=int, default=0)
    parser.add_argument("--max-listwise-eval-samples", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = RunConfig(
        seed=int(args.seed),
        val_ratio=float(args.val_ratio),
        val_split_seed=int(args.val_split_seed),
        pointwise_val_answer_seed=int(args.pointwise_val_answer_seed),
        train_selection_mode=str(args.train_selection_mode),
        fixed_selected_triples_path=str(args.fixed_selected_triples_path),
        resume_stage1_model_dir=str(args.resume_stage1_model_dir),
        triple_selection_strategy=str(args.triple_selection_strategy),
        question_selection_strategy=str(args.question_selection_strategy),
        randomize_listwise_order=not bool(args.no_randomize_listwise_order),
        candidate_selector_kind=str(args.candidate_selector_kind),
        candidate_selector_init_triples=int(args.candidate_selector_init_triples),
        candidate_selector_batch_size=int(args.candidate_selector_batch_size),
        candidate_selector_epochs=int(args.candidate_selector_epochs),
        candidate_selector_max_score_candidates=int(args.candidate_selector_max_score_candidates),
        candidate_selector_llama_rerank_candidates=int(args.candidate_selector_llama_rerank_candidates),
        candidate_selector_buffer_maxlen=int(args.candidate_selector_buffer_maxlen),
        candidate_selector_one_per_question=bool(args.candidate_selector_one_per_question),
        candidate_selector_target_task=str(args.candidate_selector_target_task),
        candidate_selector_score_range_weight=float(args.candidate_selector_score_range_weight),
        candidate_selector_gap_sum_weight=float(args.candidate_selector_gap_sum_weight),
        candidate_selector_uncertainty_weight=float(args.candidate_selector_uncertainty_weight),
        candidate_selector_pairwise_uncertainty_weight=float(args.candidate_selector_pairwise_uncertainty_weight),
        candidate_selector_listwise_uncertainty_weight=float(args.candidate_selector_listwise_uncertainty_weight),
        candidate_selector_kl_weight=float(args.candidate_selector_kl_weight),
        candidate_selector_score_bin_weight=float(args.candidate_selector_score_bin_weight),
        candidate_selector_diversity_weight=float(args.candidate_selector_diversity_weight),
        candidate_selector_density_weight=float(args.candidate_selector_density_weight),
        candidate_selector_bias_weight=float(args.candidate_selector_bias_weight),
        candidate_selector_coverage_weight=float(args.candidate_selector_coverage_weight),
        candidate_selector_pointwise_length_bias_weight=float(args.candidate_selector_pointwise_length_bias_weight),
        candidate_selector_pairwise_position_bias_weight=float(args.candidate_selector_pairwise_position_bias_weight),
        candidate_selector_pairwise_position_pairs=int(args.candidate_selector_pairwise_position_pairs),
        candidate_selector_pairwise_position_bias_scale=float(args.candidate_selector_pairwise_position_bias_scale),
        candidate_selector_signal_normalization=str(args.candidate_selector_signal_normalization),
        candidate_selector_uncertainty_view=str(args.candidate_selector_uncertainty_view),
        candidate_selector_length_aug_suffix=str(args.candidate_selector_length_aug_suffix),
        candidate_selector_density_k=int(args.candidate_selector_density_k),
        candidate_selector_embedding_model=str(args.candidate_selector_embedding_model),
        candidate_selector_embedding_max_length=int(args.candidate_selector_embedding_max_length),
        candidate_selector_embedding_batch_size=int(args.candidate_selector_embedding_batch_size),
        candidate_selector_embedding_device=str(args.candidate_selector_embedding_device),
        candidate_selector_embedding_pooling=str(args.candidate_selector_embedding_pooling),
        candidate_selector_diversity_view=str(args.candidate_selector_diversity_view),
        candidate_selector_exploration_ratio=float(args.candidate_selector_exploration_ratio),
        candidate_selector_entropy_weight=float(args.candidate_selector_entropy_weight),
        candidate_selector_score_std_weight=float(args.candidate_selector_score_std_weight),
        candidate_selector_predicted_coverage_weight=float(args.candidate_selector_predicted_coverage_weight),
        candidate_selector_proxy_warmup_epochs=int(args.candidate_selector_proxy_warmup_epochs),
        candidate_selector_proxy_update_epochs=int(args.candidate_selector_proxy_update_epochs),
        candidate_selector_proxy_mode=str(args.candidate_selector_proxy_mode),
        reuse_selection_proxy_for_stage1=bool(args.reuse_selection_proxy_for_stage1),
        candidate_bert_selector_model=str(args.candidate_bert_selector_model),
        candidate_bert_selector_max_length=int(args.candidate_bert_selector_max_length),
        candidate_bert_selector_freeze=not bool(args.candidate_bert_selector_unfreeze),
        candidate_bert_selector_unfreeze_last_n_layers=int(args.candidate_bert_selector_unfreeze_last_n_layers),
        proxy_lr=float(args.proxy_lr),
        proxy_max_length=int(args.proxy_max_length),
        llama_multitask_mode=str(args.llama_multitask_mode),
        pointwise_loss_type=str(args.pointwise_loss_type),
        pointwise_distance_weight=float(args.pointwise_distance_weight),
        pointwise_class_weight_mode=str(args.pointwise_class_weight_mode),
        pointwise_class_weight_strength=float(args.pointwise_class_weight_strength),
        budget_units=int(args.budget_units),
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
        merge_stage2_stage3=bool(args.merge_stage2_stage3),
        stage23_pointwise_replay_ratio=int(args.stage23_pointwise_replay_ratio),
        stage23_pairwise_weight=float(args.stage23_pairwise_weight),
        stage23_listwise_weight=float(args.stage23_listwise_weight),
        stage23_pointwise_weight=float(args.stage23_pointwise_weight),
        stage23_epochs=int(args.stage23_epochs),
        pointwise_teacher_distill_weight=float(args.pointwise_teacher_distill_weight),
        pointwise_teacher_distill_temperature=float(args.pointwise_teacher_distill_temperature),
        stage4_task_teacher_distill_weight=float(args.stage4_task_teacher_distill_weight),
        stage4_task_teacher_distill_temperature=float(args.stage4_task_teacher_distill_temperature),
        stage4_replay_strategy=str(args.stage4_replay_strategy),
        stage4_replay_fraction=float(args.stage4_replay_fraction),
        stage4_epochs=int(args.stage4_epochs),
        stage4_listwise_multiplier=int(args.stage4_listwise_multiplier),
        pairwise_order_augmentation=not bool(args.no_pairwise_order_augmentation),
        listwise_order_augmentation=not bool(args.no_listwise_order_augmentation),
        score_min=int(args.score_min),
        score_max=int(args.score_max),
        fix_score_prefix_in_prompt=not bool(args.no_fix_score_prefix),
        use_lora=bool(args.use_lora),
        load_in_4bit=bool(args.load_in_4bit),
        pointwise_global_smooth_alpha=float(args.pointwise_global_smooth_alpha),
        pointwise_global_smooth_mode=str(args.pointwise_global_smooth_mode),
        pointwise_global_smooth_gaussian_sigma=float(args.pointwise_global_smooth_gaussian_sigma),
        pointwise_global_smooth_stages=str(args.pointwise_global_smooth_stages),
        pointwise_global_smooth_start_step=int(args.pointwise_global_smooth_start_step),
        pointwise_global_smooth_warmup_steps=int(args.pointwise_global_smooth_warmup_steps),
        pointwise_global_smooth_start_pointwise_seen=int(args.pointwise_global_smooth_start_pointwise_seen),
        pointwise_global_smooth_warmup_pointwise_seen=int(args.pointwise_global_smooth_warmup_pointwise_seen),
        pointwise_global_smooth_prior=float(args.pointwise_global_smooth_prior),
        pointwise_global_smooth_init_prior_from_stage1=bool(args.pointwise_global_smooth_init_prior_from_stage1),
        pointwise_global_smooth_freeze_prior=bool(args.pointwise_global_smooth_freeze_prior),
        pointwise_global_smooth_uniform_mix=float(args.pointwise_global_smooth_uniform_mix),
        pointwise_global_smooth_adaptive_entropy=bool(args.pointwise_global_smooth_adaptive_entropy),
        pointwise_global_smooth_trainable_alpha=bool(args.pointwise_global_smooth_trainable_alpha),
        pointwise_global_smooth_alpha_max=float(args.pointwise_global_smooth_alpha_max),
        pointwise_global_smooth_alpha_reg=float(args.pointwise_global_smooth_alpha_reg),
        pointwise_global_smooth_alpha_lr=float(args.pointwise_global_smooth_alpha_lr),
        max_pointwise_eval_samples=int(args.max_pointwise_eval_samples),
        max_pairwise_eval_samples=int(args.max_pairwise_eval_samples),
        max_listwise_eval_samples=int(args.max_listwise_eval_samples),
        fsdp=str(args.fsdp),
        fsdp_transformer_layer_cls_to_wrap=str(args.fsdp_transformer_layer_cls_to_wrap),
        fsdp_state_dict_type=str(args.fsdp_state_dict_type),
        fsdp_activation_checkpointing=bool(args.fsdp_activation_checkpointing),
        fsdp_use_orig_params=bool(args.fsdp_use_orig_params),
        fsdp_save_all_stages=bool(args.fsdp_save_all_stages),
    )

    if _fsdp_enabled(cfg):
        if _world_size() < 2:
            raise ValueError("FSDP requires torchrun with at least two processes")
        if bool(cfg.use_lora) or bool(cfg.load_in_4bit):
            raise ValueError("FSDP full fine-tuning cannot be combined with --use-lora or --load-in-4bit")
        if bool(cfg.pointwise_global_smooth_trainable_alpha):
            raise ValueError("FSDP currently supports fixed smoothing alpha only")
        if bool(cfg.reuse_selection_proxy_for_stage1) or str(cfg.resume_stage1_model_dir):
            raise ValueError("FSDP does not support reusing or resuming a single-process Stage-1 adapter")

    if float(cfg.candidate_selector_pairwise_uncertainty_weight) < 0.0:
        raise ValueError("candidate-selector-pairwise-uncertainty-weight must be >= 0")
    if float(cfg.candidate_selector_listwise_uncertainty_weight) < 0.0:
        raise ValueError("candidate-selector-listwise-uncertainty-weight must be >= 0")
    if not (0.0 <= float(cfg.candidate_selector_exploration_ratio) <= 1.0):
        raise ValueError("candidate-selector-exploration-ratio must be in [0, 1]")
    if float(cfg.candidate_selector_entropy_weight) < 0.0:
        raise ValueError("candidate-selector-entropy-weight must be >= 0")
    if float(cfg.candidate_selector_score_std_weight) < 0.0:
        raise ValueError("candidate-selector-score-std-weight must be >= 0")
    if float(cfg.candidate_selector_predicted_coverage_weight) < 0.0:
        raise ValueError("candidate-selector-predicted-coverage-weight must be >= 0")
    for name, value in (
        ("candidate-selector-diversity-weight", cfg.candidate_selector_diversity_weight),
        ("candidate-selector-density-weight", cfg.candidate_selector_density_weight),
        ("candidate-selector-bias-weight", cfg.candidate_selector_bias_weight),
        ("candidate-selector-coverage-weight", cfg.candidate_selector_coverage_weight),
        ("candidate-selector-pointwise-length-bias-weight", cfg.candidate_selector_pointwise_length_bias_weight),
        ("candidate-selector-pairwise-position-bias-weight", cfg.candidate_selector_pairwise_position_bias_weight),
    ):
        if float(value) < 0.0:
            raise ValueError(f"{name} must be >= 0")
    if int(cfg.candidate_selector_density_k) <= 0:
        raise ValueError("candidate-selector-density-k must be > 0")
    if int(cfg.candidate_selector_embedding_max_length) <= 0:
        raise ValueError("candidate-selector-embedding-max-length must be > 0")
    if int(cfg.candidate_selector_embedding_batch_size) <= 0:
        raise ValueError("candidate-selector-embedding-batch-size must be > 0")
    if str(cfg.candidate_selector_embedding_pooling) not in {"cls", "mean"}:
        raise ValueError("candidate-selector-embedding-pooling must be one of {'cls','mean'}")
    if str(cfg.candidate_selector_diversity_view) not in {"pointwise", "joint"}:
        raise ValueError("candidate-selector-diversity-view must be one of {'pointwise','joint'}")
    if int(cfg.candidate_selector_pairwise_position_pairs) <= 0:
        raise ValueError("candidate-selector-pairwise-position-pairs must be > 0")
    if float(cfg.candidate_selector_pairwise_position_bias_scale) < 0.0:
        raise ValueError("candidate-selector-pairwise-position-bias-scale must be >= 0")
    if str(cfg.candidate_selector_signal_normalization) not in {"none", "intrinsic"}:
        raise ValueError("candidate-selector-signal-normalization must be one of {'none','intrinsic'}")
    if str(cfg.candidate_selector_uncertainty_view) not in {"pointwise", "pairwise", "listwise", "joint"}:
        raise ValueError("candidate-selector-uncertainty-view must be one of {'pointwise','pairwise','listwise','joint'}")
    if int(cfg.candidate_selector_proxy_warmup_epochs) <= 0:
        raise ValueError("candidate-selector-proxy-warmup-epochs must be > 0")
    if int(cfg.candidate_selector_proxy_update_epochs) <= 0:
        raise ValueError("candidate-selector-proxy-update-epochs must be > 0")
    if str(cfg.candidate_selector_kind) == "pointwise_proxy":
        if str(cfg.candidate_selector_target_task) != "pointwise":
            raise ValueError("pointwise_proxy selector requires --candidate-selector-target-task pointwise")
        if float(cfg.candidate_selector_entropy_weight) + float(cfg.candidate_selector_score_std_weight) <= 0.0:
            raise ValueError("pointwise_proxy selector requires a positive entropy or score-std weight")
    if str(cfg.candidate_selector_kind) == "bias_trap_pointwise":
        if str(cfg.candidate_selector_target_task) != "pointwise":
            raise ValueError("bias_trap_pointwise selector requires --candidate-selector-target-task pointwise")
    if bool(cfg.reuse_selection_proxy_for_stage1):
        if str(cfg.train_selection_mode) != "candidate_triple_selector":
            raise ValueError("--reuse-selection-proxy-for-stage1 requires --train-selection-mode candidate_triple_selector")
        if str(cfg.fixed_selected_triples_path):
            raise ValueError("--reuse-selection-proxy-for-stage1 cannot be used with --fixed-selected-triples-path")
        if str(cfg.candidate_selector_kind) not in {"pointwise_proxy", "bias_trap_pointwise"}:
            raise ValueError(
                "--reuse-selection-proxy-for-stage1 requires --candidate-selector-kind "
                "pointwise_proxy or bias_trap_pointwise"
            )
        if str(cfg.candidate_selector_proxy_mode) != "lm_head":
            raise ValueError("--reuse-selection-proxy-for-stage1 requires --candidate-selector-proxy-mode lm_head")
    if (
        str(cfg.train_selection_mode) == "candidate_triple_selector"
        and str(cfg.candidate_selector_target_task) == "pointwise"
        and str(cfg.candidate_selector_kind) != "bias_trap_pointwise"
    ):
        selector_target_weight = (
            float(cfg.candidate_selector_uncertainty_weight)
            + float(cfg.candidate_selector_pairwise_uncertainty_weight)
            + float(cfg.candidate_selector_listwise_uncertainty_weight)
            + float(cfg.candidate_selector_kl_weight)
            + float(cfg.candidate_selector_score_range_weight)
            + float(cfg.candidate_selector_gap_sum_weight)
            + float(cfg.candidate_selector_score_bin_weight)
        )
        if selector_target_weight <= 0.0:
            raise ValueError("At least one pointwise selector target weight must be > 0")

    if not (0.0 <= float(cfg.pointwise_global_smooth_alpha) <= 1.0):
        raise ValueError("pointwise_global_smooth_alpha must be in [0, 1]")
    smooth_mode = str(cfg.pointwise_global_smooth_mode).strip().lower().replace("-", "_")
    valid_smooth_modes = {"global", "prior", "global_prior", "local", "gaussian", "local_gaussian"}
    if smooth_mode not in valid_smooth_modes:
        raise ValueError("pointwise_global_smooth_mode must be 'global_prior' or 'local_gaussian'")
    if float(cfg.pointwise_global_smooth_gaussian_sigma) <= 0.0:
        raise ValueError("pointwise_global_smooth_gaussian_sigma must be > 0")
    valid_smooth_stages = {
        "all",
        "*",
        "stage1",
        "pointwise",
        "stage1_pointwise",
        "stage2",
        "pairwise",
        "stage2_pairwise",
        "stage3",
        "listwise",
        "stage3_listwise",
        "stage23",
        "pairwise_listwise",
        "stage23_pairwise_listwise",
        "stage4",
        "consolidation",
        "stage4_consolidation",
    }
    smooth_stage_tokens = {x.strip().lower() for x in str(cfg.pointwise_global_smooth_stages).split(",") if x.strip()}
    if not smooth_stage_tokens:
        raise ValueError("pointwise_global_smooth_stages must be non-empty")
    unknown_smooth_stages = smooth_stage_tokens - valid_smooth_stages
    if unknown_smooth_stages:
        raise ValueError(f"unknown pointwise_global_smooth_stages: {sorted(unknown_smooth_stages)}")
    if int(cfg.pointwise_global_smooth_start_step) < 0:
        raise ValueError("pointwise_global_smooth_start_step must be >= 0")
    if int(cfg.pointwise_global_smooth_warmup_steps) < 0:
        raise ValueError("pointwise_global_smooth_warmup_steps must be >= 0")
    if float(cfg.pointwise_global_smooth_prior) <= 0.0:
        raise ValueError("pointwise_global_smooth_prior must be > 0")
    if float(cfg.pointwise_global_smooth_alpha_max) <= 0.0:
        raise ValueError("pointwise_global_smooth_alpha_max must be > 0")
    if float(cfg.pointwise_global_smooth_alpha_reg) < 0.0:
        raise ValueError("pointwise_global_smooth_alpha_reg must be >= 0")
    if float(cfg.pointwise_global_smooth_alpha_lr) < 0.0:
        raise ValueError("pointwise_global_smooth_alpha_lr must be >= 0")
    if bool(cfg.pointwise_global_smooth_trainable_alpha):
        if not (0.0 < float(cfg.pointwise_global_smooth_alpha) < float(cfg.pointwise_global_smooth_alpha_max)):
            raise ValueError("trainable smoothing requires 0 < alpha < alpha_max")
    if int(cfg.stage23_pointwise_replay_ratio) < 0:
        raise ValueError("stage23-pointwise-replay-ratio must be >= 0")
    if float(cfg.stage23_pairwise_weight) < 0.0:
        raise ValueError("stage23-pairwise-weight must be >= 0")
    if float(cfg.stage23_listwise_weight) < 0.0:
        raise ValueError("stage23-listwise-weight must be >= 0")
    if float(cfg.stage23_pointwise_weight) < 0.0:
        raise ValueError("stage23-pointwise-weight must be >= 0")
    if int(cfg.stage23_epochs) <= 0:
        raise ValueError("stage23-epochs must be > 0")
    if float(cfg.pointwise_teacher_distill_weight) < 0.0:
        raise ValueError("pointwise-teacher-distill-weight must be >= 0")
    if float(cfg.pointwise_teacher_distill_temperature) <= 0.0:
        raise ValueError("pointwise-teacher-distill-temperature must be > 0")
    if float(cfg.stage4_task_teacher_distill_weight) < 0.0:
        raise ValueError("stage4-task-teacher-distill-weight must be >= 0")
    if float(cfg.stage4_task_teacher_distill_temperature) <= 0.0:
        raise ValueError("stage4-task-teacher-distill-temperature must be > 0")
    if str(cfg.stage4_replay_strategy) != "none":
        if not (0.0 < float(cfg.stage4_replay_fraction) <= 1.0):
            raise ValueError("stage4-replay-fraction must be in (0, 1] when Stage 4 is enabled")
    if int(cfg.stage4_listwise_multiplier) < 1:
        raise ValueError("stage4-listwise-multiplier must be >= 1")
        if int(cfg.stage4_epochs) <= 0:
            raise ValueError("stage4-epochs must be > 0 when Stage 4 is enabled")

    ds_path = base._resolve_existing_path(str(args.pointwise_5answers_dataset))
    eval_path = base._resolve_existing_path(str(args.listwise_eval_dataset))
    pairwise_eval_path = base._resolve_existing_path(str(args.pairwise_eval_dataset)) if str(args.pairwise_eval_dataset).strip() else ""
    if not ds_path:
        raise FileNotFoundError(args.pointwise_5answers_dataset)
    if not eval_path:
        raise FileNotFoundError(args.listwise_eval_dataset)
    if str(args.pairwise_eval_dataset).strip() and not pairwise_eval_path:
        raise FileNotFoundError(args.pairwise_eval_dataset)
    out = Path(args.out) if args.out else Path("outputs") / ("three_stage_sft_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    out.mkdir(parents=True, exist_ok=True)

    if _is_primary_process():
        print("\n" + "=" * 80)
        print("Start run: generative SFT pointwise -> pairwise -> listwise")
        print("=" * 80)
        print(f"dataset={ds_path}")
        print(f"listwise_eval={eval_path}")
        print(f"budget_units={cfg.budget_units}")
        print(f"selection={cfg.train_selection_mode}")
        print(f"output_dir={out}")
        if _fsdp_enabled(cfg):
            print(
                "distributed=FSDP "
                f"world_size={_world_size()} fsdp={cfg.fsdp} "
                f"wrap={cfg.fsdp_transformer_layer_cls_to_wrap}"
            )

    _write_json(out / "config.json", {**asdict(cfg), "dataset": str(ds_path), "listwise_eval_dataset": str(eval_path), "pairwise_eval_dataset": str(pairwise_eval_path), "llama": str(args.llama)})

    questions, load_stats = lw._load_scored_questions_ge3(str(ds_path), score_min=int(cfg.score_min), score_max=int(cfg.score_max))
    train_questions, val_questions, split_info = base._split_questions(questions, seed=int(cfg.val_split_seed), val_ratio=float(cfg.val_ratio))
    split_info["pointwise_val_answer_seed"] = int(cfg.pointwise_val_answer_seed)
    _write_json(out / "dataset_load_stats.json", load_stats)
    _write_json(out / "split_questions.json", split_info)

    selection_proxy = None
    if str(cfg.fixed_selected_triples_path):
        candidates, candidate_rows, candidate_pool_stats = lw._build_candidate_triple_examples(
            train_questions,
            randomize_order=bool(cfg.randomize_listwise_order),
            seed=int(cfg.seed) + 11,
        )
        _write_json(out / "candidate_triple_pool_stats.json", candidate_pool_stats)
        _write_jsonl(out / "candidate_triples.jsonl", candidate_rows)
        train_triples, selected_rows, selected_stats = _load_fixed_selected_triples(
            path=str(cfg.fixed_selected_triples_path),
            candidates=candidates,
            budget_units=int(cfg.budget_units),
        )
    elif cfg.train_selection_mode == "candidate_triple_selector":
        candidates, candidate_rows, candidate_pool_stats = lw._build_candidate_triple_examples(
            train_questions,
            randomize_order=bool(cfg.randomize_listwise_order),
            seed=int(cfg.seed) + 11,
        )
        _write_json(out / "candidate_triple_pool_stats.json", candidate_pool_stats)
        _write_jsonl(out / "candidate_triples.jsonl", candidate_rows)
        selection_result = lw._select_candidate_triples_with_selector(
            candidates=candidates,
            cfg=cfg,
            llama_path=str(args.llama),
            output_dir=out,
        )
        if bool(cfg.reuse_selection_proxy_for_stage1):
            if len(selection_result) != 4:
                raise RuntimeError("selection proxy reuse requested, but selector did not return a proxy")
            train_triples, selected_rows, selected_stats, selection_proxy = selection_result
        else:
            train_triples, selected_rows, selected_stats = selection_result
        _write_json(out / "candidate_triple_selection_stats.json", selected_stats)
    else:
        train_triples, selected_rows, selected_stats = lw._select_question_triples(
            train_questions,
            strategy=str(cfg.triple_selection_strategy),
            randomize_order=bool(cfg.randomize_listwise_order),
            question_selection_strategy=str(cfg.question_selection_strategy),
            seed=int(cfg.seed) + 11,
            budget_units=int(cfg.budget_units),
        )
    _write_json(out / "selected_triple_stats.json", selected_stats)
    _write_jsonl(out / "selected_triples.jsonl", selected_rows)

    pointwise_train, pointwise_train_rows, pointwise_train_stats = lw._build_pointwise_examples_from_triples(
        train_triples,
        score_min=int(cfg.score_min),
        score_max=int(cfg.score_max),
        fix_score_prefix_in_prompt=bool(cfg.fix_score_prefix_in_prompt),
    )
    pairwise_train, pairwise_train_rows, pairwise_train_stats = _build_pairwise_examples_from_triples(
        train_triples,
        order_augmentation=bool(cfg.pairwise_order_augmentation),
    )
    listwise_train, listwise_train_rows, listwise_train_stats = lw._build_listwise_examples_from_triples(
        train_triples,
        order_augmentation=bool(cfg.listwise_order_augmentation),
    )
    _write_json(out / "pointwise_train_stats.json", pointwise_train_stats)
    _write_json(out / "pairwise_train_stats.json", pairwise_train_stats)
    _write_json(out / "listwise_train_stats.json", listwise_train_stats)
    _write_jsonl(out / "pointwise_train.jsonl", pointwise_train_rows)
    _write_jsonl(out / "pairwise_train.jsonl", pairwise_train_rows)
    _write_jsonl(out / "listwise_train.jsonl", listwise_train_rows)

    pointwise_val, pointwise_val_rows, pointwise_val_stats = base._build_single_answer_pointwise_eval_examples(
        val_questions,
        seed=int(cfg.pointwise_val_answer_seed),
        score_min=int(cfg.score_min),
        score_max=int(cfg.score_max),
        judge_system_prompt=base.JUDGE_SYSTEM_PROMPT_SCORE_ONLY,
        fix_score_prefix_in_prompt=bool(cfg.fix_score_prefix_in_prompt),
    )
    if pairwise_eval_path:
        pairwise_eval, pairwise_eval_rows, pairwise_eval_stats = base._load_pairwise_abc_eval_dataset(
            str(pairwise_eval_path), pairwise_system_prompt=base.DEFAULT_PAIRWISE_SYSTEM_PROMPT
        )
    else:
        pairwise_eval, pairwise_eval_rows, pairwise_eval_stats = _load_pairwise_eval_from_listwise_dataset(str(eval_path))
    listwise_eval, listwise_eval_rows, listwise_eval_stats = lw._load_listwise_eval_dataset(str(eval_path))
    if cfg.max_pointwise_eval_samples > 0:
        pointwise_val = pointwise_val[: cfg.max_pointwise_eval_samples]
        pointwise_val_rows = pointwise_val_rows[: cfg.max_pointwise_eval_samples]
        pointwise_val_stats["eval_cap"] = int(cfg.max_pointwise_eval_samples)
    if cfg.max_pairwise_eval_samples > 0:
        pairwise_eval = pairwise_eval[: cfg.max_pairwise_eval_samples]
        pairwise_eval_rows = pairwise_eval_rows[: cfg.max_pairwise_eval_samples]
        pairwise_eval_stats["eval_cap"] = int(cfg.max_pairwise_eval_samples)
    if cfg.max_listwise_eval_samples > 0:
        listwise_eval = listwise_eval[: cfg.max_listwise_eval_samples]
        listwise_eval_rows = listwise_eval_rows[: cfg.max_listwise_eval_samples]
        listwise_eval_stats["eval_cap"] = int(cfg.max_listwise_eval_samples)
    _write_json(out / "pointwise_val_stats.json", pointwise_val_stats)
    _write_json(out / "pairwise_eval_stats.json", pairwise_eval_stats)
    _write_json(out / "listwise_eval_stats.json", listwise_eval_stats)
    _write_jsonl(out / "pointwise_val.jsonl", pointwise_val_rows)
    _write_jsonl(out / "pairwise_eval.jsonl", pairwise_eval_rows)
    _write_jsonl(out / "listwise_eval.jsonl", listwise_eval_rows)

    print(
        "Train examples: "
        f"pointwise={len(pointwise_train)} pairwise={len(pairwise_train)} listwise={len(listwise_train)}"
    )
    print(f"Eval examples: pointwise={len(pointwise_val)} pairwise={len(pairwise_eval)} listwise={len(listwise_eval)}")

    point_items = _pointwise_items(pointwise_train, cfg)
    pair_items = _pairwise_items(pairwise_train)
    list_items = _listwise_items(listwise_train)
    stage1_pointwise_hist = _score_hist_from_items(
        point_items,
        num_labels=int(cfg.score_max - cfg.score_min + 1),
    )

    model = None
    tokenizer = None
    train_stats: Dict[str, Any] = {}
    pointwise_metrics: Dict[str, Any] = {}
    pairwise_metrics: Dict[str, Any] = {}
    listwise_metrics: Dict[str, Any] = {}
    stage4_replay_stats: Dict[str, Any] = {"enabled": False, "strategy": str(cfg.stage4_replay_strategy)}
    stage4_pointwise_replay_stats: Dict[str, Any] = {}
    stage4_pairwise_replay_stats: Dict[str, Any] = {}
    stage4_listwise_replay_stats: Dict[str, Any] = {}
    stage23_sampling_stats: Dict[str, Any] = {}
    pointwise_teacher_distill_stats: Dict[str, Any] = {"enabled": False}
    stage4_pointwise_teacher_distill_stats: Dict[str, Any] = {"enabled": False}
    stage4_pairwise_teacher_distill_stats: Dict[str, Any] = {"enabled": False}
    stage4_listwise_teacher_distill_stats: Dict[str, Any] = {"enabled": False}
    pairwise_teacher_decision_stats: Dict[str, Any] = {"enabled": False}
    listwise_teacher_decision_stats: Dict[str, Any] = {"enabled": False}

    def eval_all(stage: str) -> None:
        eval_model = model
        eval_tokenizer = tokenizer
        loaded_eval_model = False
        if _fsdp_enabled(cfg):
            # Classic FSDP cannot safely call a forwarded generate() method:
            # the root-owned embedding and LM-head parameters stay sharded.
            # Rank zero reloads the complete stage checkpoint for generation,
            # while the remaining ranks wait without entering FSDP collectives.
            if not _is_primary_process():
                _distributed_barrier()
                return
            stage_dirs = {
                "after_stage1": out / "stage1_pointwise_sft_model",
                "after_stage2": out / "stage2_pairwise_sft_model",
                "after_stage3": out / "stage3_listwise_sft_model",
                "after_stage23": out / "stage23_pairwise_listwise_sft_model",
                "after_stage4": out / "stage4_consolidation_sft_model",
            }
            checkpoint_dir = stage_dirs.get(str(stage))
            if checkpoint_dir is None:
                raise ValueError(f"No FSDP evaluation checkpoint is registered for stage {stage!r}")
            gc.collect()
            torch.cuda.empty_cache()
            eval_model, eval_tokenizer, _ = base._load_sft_model_and_tokenizer(
                model_name_or_path=str(checkpoint_dir),
                max_length=int(cfg.max_length),
                load_in_4bit=False,
            )
            eval_model = eval_model.to(
                device=torch.device("cuda", torch.cuda.current_device()),
                dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            )
            loaded_eval_model = True

        print(f"\nEvaluating all metrics: {stage}")
        pointwise_metrics[stage] = base._evaluate_pointwise_sft(
            model=eval_model,
            tokenizer=eval_tokenizer,
            examples=pointwise_val,
            max_length=int(cfg.max_length),
            batch_size=int(cfg.eval_batch_size),
            max_new_tokens=int(cfg.max_new_tokens_pointwise),
            score_min=int(cfg.score_min),
            score_max=int(cfg.score_max),
        )
        pairwise_metrics[stage] = base._evaluate_pairwise_sft(
            model=eval_model,
            tokenizer=eval_tokenizer,
            examples=pairwise_eval,
            max_length=int(cfg.max_length),
            batch_size=int(cfg.eval_batch_size),
            max_new_tokens=int(cfg.max_new_tokens_pairwise),
        )
        listwise_metrics[stage] = _evaluate_listwise_sft(
            model=eval_model,
            tokenizer=eval_tokenizer,
            examples=listwise_eval,
            max_length=int(cfg.max_length),
            batch_size=int(cfg.eval_batch_size),
            max_new_tokens=int(cfg.max_new_tokens_listwise),
        )
        _write_json(out / f"metrics_pointwise_{stage}.json", pointwise_metrics[stage])
        _write_json(out / f"metrics_pairwise_{stage}.json", pairwise_metrics[stage])
        _write_json(out / f"metrics_listwise_{stage}.json", listwise_metrics[stage])
        if loaded_eval_model:
            del eval_model
            del eval_tokenizer
            gc.collect()
            torch.cuda.empty_cache()
        if _fsdp_enabled(cfg):
            _distributed_barrier()

    final_eval_stage = (
        "after_stage4"
        if str(cfg.stage4_replay_strategy) != "none"
        else "after_stage23"
        if bool(cfg.merge_stage2_stage3)
        else "after_stage3"
    )

    def maybe_eval_all(stage: str) -> None:
        if str(cfg.eval_stages) == "all" or str(stage) == str(final_eval_stage):
            eval_all(stage)
        elif _is_primary_process():
            print(f"\nSkipping eval metrics: {stage} (eval_stages={cfg.eval_stages})")

    if str(cfg.resume_stage1_model_dir):
        resume_stage1_dir = Path(str(cfg.resume_stage1_model_dir))
        print(f"\nResuming from saved Stage-1 model: {resume_stage1_dir}", flush=True)
        model, tokenizer = _load_stage1_resume_model(
            base_model_path=str(args.llama),
            adapter_dir=resume_stage1_dir,
            cfg=cfg,
        )
        stage1_dir = out / "stage1_pointwise_sft_model"
        stage1_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(stage1_dir))
        tokenizer.save_pretrained(str(stage1_dir))
        previous_stage1_stats_path = resume_stage1_dir.parent / "train_stats_stage1_pointwise_sft.json"
        previous_stage1_stats: Dict[str, Any] = {}
        if previous_stage1_stats_path.exists():
            try:
                previous_stage1_stats = json.loads(previous_stage1_stats_path.read_text(encoding="utf-8"))
            except Exception:
                previous_stage1_stats = {}
        train_stats["stage1_pointwise"] = {
            **previous_stage1_stats,
            "mode": "resume_stage1_lora_adapter",
            "stage": "stage1_pointwise",
            "resumed_from_model_dir": str(resume_stage1_dir),
            "skipped_stage1_training": True,
            "train_samples": int(len(point_items)),
            "task_counts": {"pointwise": int(len(point_items))},
            "epochs": int(previous_stage1_stats.get("epochs", 0)),
            "model_dir": str(stage1_dir),
        }
    elif bool(cfg.reuse_selection_proxy_for_stage1):
        if selection_proxy is None:
            raise RuntimeError("selection proxy reuse requested, but no selection proxy is available")
        model = selection_proxy.model
        tokenizer = selection_proxy.tokenizer
        selection_proxy = None
        tokenizer.model_max_length = int(cfg.max_length)
        stage1_dir = out / "stage1_pointwise_sft_model"
        stage1_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(stage1_dir))
        tokenizer.save_pretrained(str(stage1_dir))
        train_stats["stage1_pointwise"] = {
            "mode": "reuse_selection_proxy_lm_head",
            "stage": "stage1_pointwise",
            "reused_selection_proxy": True,
            "skipped_generative_sft_stage": True,
            "train_samples": int(len(point_items)),
            "task_counts": {"pointwise": int(len(point_items))},
            "epochs": 0,
            "selection_proxy_warmup_epochs": int(cfg.candidate_selector_proxy_warmup_epochs),
            "selection_proxy_update_epochs": int(cfg.candidate_selector_proxy_update_epochs),
            "selection_proxy_mode": str(cfg.candidate_selector_proxy_mode),
            "model_dir": str(stage1_dir),
            "global_prior_smoothing": {"enabled": False},
        }
    else:
        train_stats["stage1_pointwise"], model, tokenizer = _train_sft_on_items(
            model_name_or_path=str(args.llama),
            model=None,
            tokenizer=None,
            items=point_items,
            output_dir=out / "stage1_pointwise_sft_model",
            cfg=cfg,
            stage_name="stage1_pointwise",
            smooth_initial_hist=stage1_pointwise_hist if bool(cfg.pointwise_global_smooth_init_prior_from_stage1) else None,
        )
    _write_json(out / "train_stats_stage1_pointwise_sft.json", train_stats["stage1_pointwise"])
    maybe_eval_all("after_stage1")

    stage1_teacher_logits: Optional[List[Optional[List[float]]]] = None
    if float(cfg.pointwise_teacher_distill_weight) > 0.0 and (
        bool(cfg.merge_stage2_stage3) or str(cfg.stage4_replay_strategy) != "none"
    ):
        print("\nCaching Stage-1 pointwise teacher logits for replay distillation...", flush=True)
        stage1_teacher_logits, pointwise_teacher_distill_stats = _compute_pointwise_teacher_logits(
            model=model,
            tokenizer=tokenizer,
            items=point_items,
            cfg=cfg,
            output_dir=out,
        )

    pairwise_teacher_logits: Optional[List[Optional[List[float]]]] = None
    listwise_teacher_logits: Optional[List[Optional[List[float]]]] = None
    decision_candidate_token_ids: Dict[int, List[int]] = {}
    decision_label_offsets: Dict[int, int] = {}
    if float(cfg.stage4_task_teacher_distill_weight) > 0.0 and str(cfg.stage4_replay_strategy) != "none":
        decision_candidate_token_ids, decision_label_offsets = _stage4_decision_distill_candidates(tokenizer)

    if bool(cfg.merge_stage2_stage3):
        stage23_items, stage23_teacher_logits, stage23_sampling_stats = _build_weighted_stage23_items(
            pair_items=pair_items,
            list_items=list_items,
            point_items=point_items,
            pointwise_teacher_logits=stage1_teacher_logits,
            cfg=cfg,
            seed=int(cfg.seed) + 559,
        )
        _write_json(out / "stage23_sampling_stats.json", stage23_sampling_stats)
        train_stats["stage23_pairwise_listwise"], model, tokenizer = _train_sft_on_items(
            model_name_or_path=None,
            model=model,
            tokenizer=tokenizer,
            items=stage23_items,
            pointwise_teacher_logits=stage23_teacher_logits,
            output_dir=out / "stage23_pairwise_listwise_sft_model",
            cfg=cfg,
            stage_name="stage23_pairwise_listwise",
            smooth_initial_hist=stage1_pointwise_hist if bool(cfg.pointwise_global_smooth_init_prior_from_stage1) else None,
        )
        _write_json(out / "train_stats_stage23_pairwise_listwise_sft.json", train_stats["stage23_pairwise_listwise"])
        maybe_eval_all("after_stage23")
    else:
        stage2_items = _with_pointwise_replay(
            pair_items,
            point_items,
            replay_ratio=int(cfg.stage2_pointwise_replay_ratio),
            seed=int(cfg.seed) + 409,
        )
        train_stats["stage2_pairwise"], model, tokenizer = _train_sft_on_items(
            model_name_or_path=None,
            model=model,
            tokenizer=tokenizer,
            items=stage2_items,
            output_dir=out / "stage2_pairwise_sft_model",
            cfg=cfg,
            stage_name="stage2_pairwise",
            smooth_initial_hist=stage1_pointwise_hist if bool(cfg.pointwise_global_smooth_init_prior_from_stage1) else None,
        )
        _write_json(out / "train_stats_stage2_pairwise_sft.json", train_stats["stage2_pairwise"])
        maybe_eval_all("after_stage2")
        if float(cfg.stage4_task_teacher_distill_weight) > 0.0 and str(cfg.stage4_replay_strategy) != "none":
            print("\nCaching Stage-2 pairwise teacher decision logits for Stage4 DER...", flush=True)
            pairwise_teacher_logits, pairwise_teacher_decision_stats = _compute_decision_teacher_logits(
                model=model,
                tokenizer=tokenizer,
                items=pair_items,
                candidate_token_ids=decision_candidate_token_ids[int(CLASS_TEACHER_TASK_PAIRWISE)],
                label_offset=decision_label_offsets[int(CLASS_TEACHER_TASK_PAIRWISE)],
                task_id=int(CLASS_TEACHER_TASK_PAIRWISE),
                cfg=cfg,
                output_dir=out,
                name="stage2_pairwise",
            )

        stage3_items = _with_pointwise_replay(
            list_items,
            point_items,
            replay_ratio=int(cfg.stage3_pointwise_replay_ratio),
            seed=int(cfg.seed) + 509,
        )
        stage3_items = _with_pointwise_replay(
            stage3_items,
            pair_items,
            replay_ratio=int(cfg.stage3_pairwise_replay_ratio),
            seed=int(cfg.seed) + 609,
            reference_count=len(list_items),
        )
        train_stats["stage3_listwise"], model, tokenizer = _train_sft_on_items(
            model_name_or_path=None,
            model=model,
            tokenizer=tokenizer,
            items=stage3_items,
            output_dir=out / "stage3_listwise_sft_model",
            cfg=cfg,
            stage_name="stage3_listwise",
            smooth_initial_hist=stage1_pointwise_hist if bool(cfg.pointwise_global_smooth_init_prior_from_stage1) else None,
        )
        _write_json(out / "train_stats_stage3_listwise_sft.json", train_stats["stage3_listwise"])
        maybe_eval_all("after_stage3")
        if float(cfg.stage4_task_teacher_distill_weight) > 0.0 and str(cfg.stage4_replay_strategy) != "none":
            print("\nCaching Stage-3 listwise teacher top-answer logits for Stage4 DER...", flush=True)
            listwise_teacher_logits, listwise_teacher_decision_stats = _compute_decision_teacher_logits(
                model=model,
                tokenizer=tokenizer,
                items=list_items,
                candidate_token_ids=decision_candidate_token_ids[int(CLASS_TEACHER_TASK_LISTWISE_TOP)],
                label_offset=decision_label_offsets[int(CLASS_TEACHER_TASK_LISTWISE_TOP)],
                task_id=int(CLASS_TEACHER_TASK_LISTWISE_TOP),
                cfg=cfg,
                output_dir=out,
                name="stage3_listwise_top",
            )

    if str(cfg.stage4_replay_strategy) != "none":
        if str(cfg.stage4_replay_strategy) == "loss_triple":
            replay_triples, replay_rows, stage4_replay_stats = _select_stage4_loss_triples(
                model=model,
                tokenizer=tokenizer,
                train_triples=train_triples,
                selected_rows=selected_rows,
                fraction=float(cfg.stage4_replay_fraction),
                cfg=cfg,
            )
        else:
            replay_triples, replay_rows, stage4_replay_stats = _select_stage4_replay_triples(
                train_triples=train_triples,
                selected_rows=selected_rows,
                strategy=str(cfg.stage4_replay_strategy),
                fraction=float(cfg.stage4_replay_fraction),
                seed=int(cfg.seed) + 709,
            )
        _write_json(out / "stage4_replay_triple_stats.json", stage4_replay_stats)
        _write_jsonl(out / "stage4_replay_triples.jsonl", replay_rows)

        stage4_pointwise_replay, stage4_pointwise_rows, stage4_pointwise_replay_stats = lw._build_pointwise_examples_from_triples(
            replay_triples,
            score_min=int(cfg.score_min),
            score_max=int(cfg.score_max),
            fix_score_prefix_in_prompt=bool(cfg.fix_score_prefix_in_prompt),
        )
        stage4_pairwise_replay, stage4_pairwise_rows, stage4_pairwise_replay_stats = _build_pairwise_examples_from_triples(
            replay_triples,
            order_augmentation=bool(cfg.pairwise_order_augmentation),
        )
        stage4_listwise_replay, stage4_listwise_rows, stage4_listwise_replay_stats = lw._build_listwise_examples_from_triples(
            replay_triples,
            order_augmentation=bool(cfg.listwise_order_augmentation),
        )
        if int(cfg.stage4_listwise_multiplier) > 1:
            stage4_listwise_replay = list(stage4_listwise_replay) * int(cfg.stage4_listwise_multiplier)
            stage4_listwise_rows = list(stage4_listwise_rows) * int(cfg.stage4_listwise_multiplier)
        stage4_listwise_replay_stats["stage4_listwise_multiplier"] = int(cfg.stage4_listwise_multiplier)
        stage4_listwise_replay_stats["weighted_generated_listwise_examples"] = len(stage4_listwise_replay)
        _write_json(out / "stage4_pointwise_replay_stats.json", stage4_pointwise_replay_stats)
        _write_json(out / "stage4_pairwise_replay_stats.json", stage4_pairwise_replay_stats)
        _write_json(out / "stage4_listwise_replay_stats.json", stage4_listwise_replay_stats)
        _write_jsonl(out / "stage4_pointwise_replay.jsonl", stage4_pointwise_rows)
        _write_jsonl(out / "stage4_pairwise_replay.jsonl", stage4_pairwise_rows)
        _write_jsonl(out / "stage4_listwise_replay.jsonl", stage4_listwise_rows)

        stage4_items = (
            _pointwise_items(stage4_pointwise_replay, cfg)
            + _pairwise_items(stage4_pairwise_replay)
            + _listwise_items(stage4_listwise_replay)
        )
        stage4_teacher_logits, stage4_pointwise_teacher_distill_stats = _align_pointwise_teacher_logits_for_mixed_items(
            teacher_items=point_items,
            teacher_logits=stage1_teacher_logits,
            mixed_items=stage4_items,
        )
        pair_aligned_logits, pair_aligned_task_ids, stage4_pairwise_teacher_distill_stats = _align_class_teacher_logits_for_mixed_items(
            teacher_items=pair_items,
            teacher_logits=pairwise_teacher_logits,
            mixed_items=stage4_items,
            task_name="pairwise",
            task_id=int(CLASS_TEACHER_TASK_PAIRWISE),
        )
        list_aligned_logits, list_aligned_task_ids, stage4_listwise_teacher_distill_stats = _align_class_teacher_logits_for_mixed_items(
            teacher_items=list_items,
            teacher_logits=listwise_teacher_logits,
            mixed_items=stage4_items,
            task_name="listwise",
            task_id=int(CLASS_TEACHER_TASK_LISTWISE_TOP),
        )
        stage4_class_teacher_logits: List[Optional[Sequence[float]]] = [None for _ in stage4_items]
        stage4_class_teacher_task_ids: List[int] = [0 for _ in stage4_items]
        for i, logits in enumerate(pair_aligned_logits):
            if logits is not None:
                stage4_class_teacher_logits[i] = logits
                stage4_class_teacher_task_ids[i] = int(pair_aligned_task_ids[i])
        for i, logits in enumerate(list_aligned_logits):
            if logits is not None:
                stage4_class_teacher_logits[i] = logits
                stage4_class_teacher_task_ids[i] = int(list_aligned_task_ids[i])
        _write_json(out / "stage4_pointwise_teacher_distill_stats.json", stage4_pointwise_teacher_distill_stats)
        _write_json(out / "stage4_pairwise_teacher_distill_stats.json", stage4_pairwise_teacher_distill_stats)
        _write_json(out / "stage4_listwise_teacher_distill_stats.json", stage4_listwise_teacher_distill_stats)
        train_stats["stage4_consolidation"], model, tokenizer = _train_sft_on_items(
            model_name_or_path=None,
            model=model,
            tokenizer=tokenizer,
            items=stage4_items,
            pointwise_teacher_logits=stage4_teacher_logits,
            class_teacher_logits=stage4_class_teacher_logits,
            class_teacher_task_ids=stage4_class_teacher_task_ids,
            output_dir=out / "stage4_consolidation_sft_model",
            cfg=cfg,
            stage_name="stage4_consolidation",
            smooth_initial_hist=stage1_pointwise_hist if bool(cfg.pointwise_global_smooth_init_prior_from_stage1) else None,
        )
        _write_json(out / "train_stats_stage4_consolidation_sft.json", train_stats["stage4_consolidation"])
        maybe_eval_all("after_stage4")

    summary = {
        "mode": (
            "generative_sft_pointwise_then_pairwise_listwise_with_pointwise_replay"
            if bool(cfg.merge_stage2_stage3)
            else "generative_sft_pointwise_pairwise_listwise_three_stage"
            if str(cfg.stage4_replay_strategy) == "none"
            else "generative_sft_pointwise_pairwise_listwise_stage4_consolidation"
        ),
        "selection_stats": selected_stats,
        "split_by_question": split_info,
        "train_budget": {
            "budget_units": int(cfg.budget_units),
            "effective_budget_units": int(selected_stats.get("effective_budget_units", len(pointwise_train))),
            "train_triples": int(len(train_triples)),
            "train_answers": int(len(pointwise_train)),
            "pairwise_train": int(len(pairwise_train)),
            "listwise_train": int(len(listwise_train)),
            "stage2_pointwise_replay_ratio": int(cfg.stage2_pointwise_replay_ratio),
            "stage3_pointwise_replay_ratio": int(cfg.stage3_pointwise_replay_ratio),
            "stage3_pairwise_replay_ratio": int(cfg.stage3_pairwise_replay_ratio),
            "merge_stage2_stage3": bool(cfg.merge_stage2_stage3),
            "stage23_pointwise_replay_ratio": int(cfg.stage23_pointwise_replay_ratio),
            "stage23_pairwise_weight": float(cfg.stage23_pairwise_weight),
            "stage23_listwise_weight": float(cfg.stage23_listwise_weight),
            "stage23_pointwise_weight": float(cfg.stage23_pointwise_weight),
            "stage23_epochs": int(cfg.stage23_epochs),
            "pointwise_teacher_distill_weight": float(cfg.pointwise_teacher_distill_weight),
            "pointwise_teacher_distill_temperature": float(cfg.pointwise_teacher_distill_temperature),
            "stage4_task_teacher_distill_weight": float(cfg.stage4_task_teacher_distill_weight),
            "stage4_task_teacher_distill_temperature": float(cfg.stage4_task_teacher_distill_temperature),
            "stage4_replay_strategy": str(cfg.stage4_replay_strategy),
            "stage4_replay_fraction": float(cfg.stage4_replay_fraction),
            "stage4_epochs": int(cfg.stage4_epochs),
            "stage4_listwise_multiplier": int(cfg.stage4_listwise_multiplier),
            "eval_stages": str(cfg.eval_stages),
            "resume_stage1_model_dir": str(cfg.resume_stage1_model_dir),
            "stage1_pointwise_hist": [float(x) for x in stage1_pointwise_hist],
        },
        "dataset_load_stats": load_stats,
        "pointwise": {"train": pointwise_train_stats, "eval": pointwise_val_stats},
        "pairwise": {"train": pairwise_train_stats, "eval": pairwise_eval_stats},
        "listwise": {"train": listwise_train_stats, "eval": listwise_eval_stats},
        "stage23_sampling": stage23_sampling_stats,
        "pointwise_teacher_distill": pointwise_teacher_distill_stats,
        "pairwise_teacher_decision_distill": pairwise_teacher_decision_stats,
        "listwise_teacher_decision_distill": listwise_teacher_decision_stats,
        "stage4_replay": {
            "triples": stage4_replay_stats,
            "pointwise": stage4_pointwise_replay_stats,
            "pairwise": stage4_pairwise_replay_stats,
            "listwise": stage4_listwise_replay_stats,
            "pointwise_teacher_distill": stage4_pointwise_teacher_distill_stats,
            "pairwise_teacher_distill": stage4_pairwise_teacher_distill_stats,
            "listwise_teacher_distill": stage4_listwise_teacher_distill_stats,
        },
        "pointwise_metrics": pointwise_metrics,
        "pairwise_metrics": pairwise_metrics,
        "listwise_metrics": listwise_metrics,
        "train_stats": train_stats,
    }
    _write_json(out / "summary.json", summary)
    compact = _compact_metrics(summary)
    _write_json(out / "metrics_compact.json", compact)

    if _is_primary_process():
        print("\n" + "=" * 60)
        print("Run finished")
        print("=" * 60)
        final_stage = final_eval_stage
        print(json.dumps(compact.get("pointwise", {}).get(final_stage, {}), ensure_ascii=False))
        print(json.dumps(compact.get("pairwise", {}).get(final_stage, {}), ensure_ascii=False))
        print(json.dumps(compact.get("listwise", {}).get(final_stage, {}), ensure_ascii=False))
        print(f"Compact metrics: {out / 'metrics_compact.json'}")
        print(f"Output directory: {out}")

    del model
    del tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


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

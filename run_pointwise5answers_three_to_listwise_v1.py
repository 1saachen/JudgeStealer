#!/usr/bin/env python
"""Train with scored answers: select 3 answers per question, then convert to listwise.

Pipeline
--------
1) Load one scored multi-answer pointwise dataset.
2) For each selected question, pick exactly 3 answers.
3) Stage-1: train the pointwise head on the 3 selected scored answers.
4) Convert each selected triple into one listwise ranking sample.
5) Stage-2: train a 13-way listwise classifier head with pointwise replay.

Example
-------
CUDA_VISIBLE_DEVICES=0 python run_pointwise5answers_three_to_listwise_v1.py \
  --pointwise-5answers-dataset train_with_selector/train_with_selector/data/newnew/train-20k.json \
  --llama llama/Meta-Llama-3-8B-Instruct/ \
  --seed 42 \
  --triple-selection-strategy random \
  --listwise-eval-dataset train_with_selector/train_with_selector/data/newnew/val-2k-eval-listwise.json \
  --stage2-pointwise-replay-ratio 3 \
  --pointwise-loss-type ce \
  --pointwise-epochs 1 \
  --listwise-epochs 1 \
  --budget-units 1500 \
  --pointwise-batch-size 32 \
  --listwise-batch-size 32 \
  --out outputs/pointwise5answers_three_to_listwise_v1
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import itertools
import json
import os
import random
import re
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

import run_pointwise5answers_two_to_pairwise_v1 as base


DEFAULT_SELECTOR_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_SELECTOR_EMBEDDING_MAX_LENGTH = 512


LISTWISE_SYSTEM_PROMPT = """You are an impartial judge evaluating the quality of three AI assistant responses to the same user request.

You must rank the three assistants from best to worst.

Your evaluation must consider:
- Helpfulness
- Relevance
- Accuracy
- Depth
- Creativity
- Level of detail

Avoid position bias. Do not let the order of responses influence your judgment.
Do not favor longer responses. Be as objective as possible.

If two or three responses are genuinely indistinguishable in quality, ties are allowed.

Output exactly one final ranking in one of these formats and nothing else:
Ranking:[A>B>C]
Ranking:[A>C>B]
Ranking:[B>A>C]
Ranking:[B>C>A]
Ranking:[C>A>B]
Ranking:[C>B>A]
Ranking:[A=B>C]
Ranking:[A=C>B]
Ranking:[B=C>A]
Ranking:[A>B=C]
Ranking:[B>A=C]
Ranking:[C>A=B]
Ranking:[A=B=C]

Do not provide any explanation or extra text.
"""


RANKING_LABELS: Tuple[str, ...] = (
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
)
RANKING_TO_LABEL: Dict[str, int] = {r: i for i, r in enumerate(RANKING_LABELS)}
LABEL_TO_RANKING: Dict[int, str] = {i: r for i, r in enumerate(RANKING_LABELS)}


@dataclass(frozen=True)
class SelectedQuestionTriple:
    question_id: int
    source_id: int
    dataset: str
    instruction: str
    input_text: str
    answer_a: base.AnswerWithScore
    answer_b: base.AnswerWithScore
    answer_c: base.AnswerWithScore


@dataclass(frozen=True)
class ListwiseExample:
    id: int
    dataset: str
    group_id: int
    source_id: int
    model_a: str
    model_b: str
    model_c: str
    prompt: str
    ranking: str
    label: int

    def __str__(self) -> str:  # noqa: D105
        return self.prompt


@dataclass(frozen=True)
class CandidateTripleExample:
    id: int
    group_id: int
    question_id: int
    source_id: int
    dataset: str
    model_a: str
    model_b: str
    model_c: str
    score_a: int
    score_b: int
    score_c: int
    score_range: int
    score_gap_sum: int
    prompt: str
    ranking: str
    label: int
    selected_triple: SelectedQuestionTriple

    def __str__(self) -> str:  # noqa: D105
        return self.prompt


@dataclass
class RunConfig:
    seed: int
    val_ratio: float
    val_split_seed: int
    pointwise_val_answer_seed: int
    train_selection_mode: str
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
    candidate_bert_selector_model: str
    candidate_bert_selector_max_length: int
    candidate_bert_selector_freeze: bool
    candidate_bert_selector_unfreeze_last_n_layers: int
    listwise_order_augmentation: bool
    budget_units: int
    pointwise_epochs: int
    listwise_epochs: int
    pointwise_batch_size: int
    listwise_batch_size: int
    stage2_pointwise_replay_ratio: int
    score_min: int
    score_max: int
    fix_score_prefix_in_prompt: bool
    proxy_lr: float
    proxy_max_length: int
    load_in_4bit: bool
    llama_multitask_mode: str
    pointwise_loss_type: str
    pointwise_distance_weight: float
    pointwise_class_weight_mode: str
    pointwise_class_weight_strength: float
    pointwise_global_smooth_alpha: float
    pointwise_global_smooth_start_step: int
    pointwise_global_smooth_warmup_steps: int
    pointwise_global_smooth_prior: float
    pointwise_global_smooth_trainable_alpha: bool
    pointwise_global_smooth_alpha_max: float
    pointwise_global_smooth_alpha_reg: float
    pointwise_global_smooth_alpha_lr: float


def _first_nonempty(rec: Dict[str, Any], keys: Sequence[str], default: Any = "") -> Any:
    for key in keys:
        if key in rec and rec[key] is not None:
            value = rec[key]
            if not isinstance(value, str) or value.strip():
                return value
    return default


def _safe_score(rec: Dict[str, Any], keys: Sequence[str], *, score_min: int, score_max: int) -> Optional[int]:
    for key in keys:
        if key not in rec or rec[key] is None:
            continue
        score = base._safe_int(rec[key], default=-10**9)
        if int(score_min) <= int(score) <= int(score_max):
            return int(score)
    return None


def _normalize_ranking_text(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    match = re.search(r"Ranking\s*:\s*\[\s*([ABC](?:\s*[>=]\s*[ABC]){2})\s*\]", text, flags=re.I)
    if match is not None:
        text = match.group(1)
    else:
        text = text.strip()
        if text.startswith("[") and text.endswith("]"):
            text = text[1:-1]
    text = re.sub(r"\s+", "", text).upper()
    return text


def _ranking_from_scores(score_a: int, score_b: int, score_c: int) -> str:
    scores = {"A": int(score_a), "B": int(score_b), "C": int(score_c)}
    groups: List[str] = []
    for score in sorted(set(scores.values()), reverse=True):
        tied = [letter for letter in ("A", "B", "C") if scores[letter] == score]
        groups.append("=".join(tied))
    ranking = ">".join(groups)
    if ranking not in RANKING_TO_LABEL:
        raise RuntimeError(f"unsupported ranking generated from scores: {ranking}")
    return ranking


def _label_from_ranking(ranking: str) -> int:
    ranking_s = _normalize_ranking_text(ranking)
    if ranking_s not in RANKING_TO_LABEL:
        raise ValueError(f"unknown listwise ranking: {ranking!r}")
    return int(RANKING_TO_LABEL[ranking_s])


def _build_listwise_prompt(
    *,
    system_prompt: str,
    instruction: str,
    input_text: str,
    assistant_a_output: str,
    assistant_b_output: str,
    assistant_c_output: str,
) -> str:
    instruction = (instruction or "").strip()
    input_text = (input_text or "").strip()
    assistant_a_output = (assistant_a_output or "").strip()
    assistant_b_output = (assistant_b_output or "").strip()
    assistant_c_output = (assistant_c_output or "").strip()

    parts: List[str] = ["### System", system_prompt.strip(), "", "### User"]
    parts.append(f"Instruction: {instruction}")
    if input_text:
        parts.append(f"Input: {input_text}")
    parts.append("")
    parts.append("### Assistant A")
    parts.append(assistant_a_output)
    parts.append("")
    parts.append("### Assistant B")
    parts.append(assistant_b_output)
    parts.append("")
    parts.append("### Assistant C")
    parts.append(assistant_c_output)
    parts.append("")
    parts.append("### Judge")
    return "\n".join(parts)


def _load_scored_questions_ge3(path: str, *, score_min: int, score_max: int) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    questions, stats = base._load_scored_questions(path, score_min=int(score_min), score_max=int(score_max))
    kept = [q for q in questions if len(q.get("answers", [])) >= 3]
    out_stats: Dict[str, Any] = dict(stats)
    out_stats["questions_with_ge3_answers"] = int(len(kept))
    out_stats["questions_skipped_lt3_answers"] = int(len(questions) - len(kept))
    if not kept:
        raise RuntimeError("no valid questions with >=3 scored answers were loaded")
    return kept, out_stats


def _pick_three_answers(
    *,
    answers: Sequence[base.AnswerWithScore],
    strategy: str,
    rng: np.random.Generator,
    randomize_order: bool,
) -> Tuple[base.AnswerWithScore, base.AnswerWithScore, base.AnswerWithScore, Dict[str, Any]]:
    if len(answers) < 3:
        raise ValueError("need at least 3 answers")

    n = int(len(answers))
    if strategy == "first_three":
        picked = [0, 1, 2]
    elif strategy == "random":
        picked = [int(x) for x in rng.choice(n, size=3, replace=False).tolist()]
    else:
        raise ValueError(f"unknown triple-selection-strategy: {strategy}")

    original_indices = list(picked)
    if bool(randomize_order):
        order = [int(x) for x in rng.permutation(3).tolist()]
        picked = [picked[i] for i in order]

    selected = [answers[int(i)] for i in picked]
    scores = [int(x.score) for x in selected]
    meta = {
        "strategy": str(strategy),
        "selected_original_indices": original_indices,
        "selected_position_indices": list(picked),
        "score_range": int(max(scores) - min(scores)),
        "score_pairwise_gap_sum": int(sum(abs(scores[i] - scores[j]) for i in range(3) for j in range(i + 1, 3))),
        "ranking_from_scores": _ranking_from_scores(scores[0], scores[1], scores[2]),
    }
    return selected[0], selected[1], selected[2], meta


def _select_question_triples(
    questions: Sequence[Dict[str, Any]],
    *,
    strategy: str,
    randomize_order: bool,
    question_selection_strategy: str,
    seed: int,
    budget_units: int,
) -> Tuple[List[SelectedQuestionTriple], List[Dict[str, Any]], Dict[str, Any]]:
    rng = np.random.default_rng(int(seed))
    selected: List[SelectedQuestionTriple] = []
    rows: List[Dict[str, Any]] = []

    stats: Dict[str, Any] = {
        "input_questions": int(len(questions)),
        "budget_units": int(budget_units),
        "budget_unit_type": "pointwise_answers",
        "answers_per_selected_question": 3,
        "selected_questions": 0,
        "selected_answers": 0,
        "selection_strategy": str(strategy),
        "question_selection_strategy": str(question_selection_strategy),
        "randomize_listwise_order": bool(randomize_order),
    }

    max_questions = 0
    if int(budget_units) > 0:
        max_questions = int(budget_units) // 3
        if max_questions <= 0:
            raise ValueError("budget-units must be >= 3 when > 0")
    stats["max_questions_by_budget"] = int(max_questions) if max_questions > 0 else int(len(questions))

    question_pool = list(questions)
    if str(question_selection_strategy) == "random":
        order = rng.permutation(len(question_pool)).tolist()
        question_iter = [question_pool[int(i)] for i in order]
    elif str(question_selection_strategy) == "first":
        question_iter = question_pool
    else:
        raise ValueError(f"unknown question-selection-strategy: {question_selection_strategy}")

    for q in question_iter:
        if int(max_questions) > 0 and len(selected) >= int(max_questions):
            break

        answers = list(q.get("answers", []))
        if len(answers) < 3:
            continue

        a, b, c, meta = _pick_three_answers(
            answers=answers,
            strategy=str(strategy),
            rng=rng,
            randomize_order=bool(randomize_order),
        )

        triple = SelectedQuestionTriple(
            question_id=int(q["question_id"]),
            source_id=int(q.get("source_id", q["question_id"])),
            dataset=str(q.get("dataset", "")),
            instruction=str(q.get("instruction", "")),
            input_text=str(q.get("input_text", "")),
            answer_a=a,
            answer_b=b,
            answer_c=c,
        )
        selected.append(triple)
        rows.append(
            {
                "question_id": int(triple.question_id),
                "source_id": int(triple.source_id),
                "dataset": str(triple.dataset),
                "model_a": str(a.model),
                "model_b": str(b.model),
                "model_c": str(c.model),
                "score_a": int(a.score),
                "score_b": int(b.score),
                "score_c": int(c.score),
                "score_range": int(meta["score_range"]),
                "score_pairwise_gap_sum": int(meta["score_pairwise_gap_sum"]),
                "ranking_from_scores": str(meta["ranking_from_scores"]),
                "selection_strategy": str(meta["strategy"]),
                "selected_original_indices": list(meta["selected_original_indices"]),
                "selected_position_indices": list(meta["selected_position_indices"]),
            }
        )

    stats["selected_questions"] = int(len(selected))
    stats["selected_answers"] = int(len(selected) * 3)
    stats["effective_budget_units"] = int(len(selected) * 3)
    if not selected:
        raise RuntimeError("no question triples were selected")

    return selected, rows, stats



def _build_candidate_triple_examples(
    questions: Sequence[Dict[str, Any]],
    *,
    randomize_order: bool,
    seed: int,
) -> Tuple[List[CandidateTripleExample], List[Dict[str, Any]], Dict[str, Any]]:
    rng = np.random.default_rng(int(seed))
    examples: List[CandidateTripleExample] = []
    rows: List[Dict[str, Any]] = []
    stats: Dict[str, Any] = {
        "input_questions": int(len(questions)),
        "candidate_triples": 0,
        "questions_with_candidate_triples": 0,
        "labels_hidden_until_query": 1,
        "randomize_listwise_order": bool(randomize_order),
    }

    triple_id = 0
    for q in questions:
        answers = list(q.get("answers", []))
        if len(answers) < 3:
            continue
        stats["questions_with_candidate_triples"] += 1
        qid = int(q["question_id"])
        source_id = int(q.get("source_id", qid))
        group_id = int(source_id if source_id > 0 else qid)

        for combo in itertools.combinations(range(len(answers)), 3):
            picked = [int(i) for i in combo]
            if bool(randomize_order):
                order = [int(x) for x in rng.permutation(3).tolist()]
                picked = [picked[i] for i in order]
            a, b, c = [answers[int(i)] for i in picked]
            scores = [int(a.score), int(b.score), int(c.score)]
            ranking = _ranking_from_scores(scores[0], scores[1], scores[2])
            label = _label_from_ranking(ranking)
            prompt = _build_listwise_prompt(
                system_prompt=LISTWISE_SYSTEM_PROMPT,
                instruction=str(q["instruction"]),
                input_text=str(q["input_text"]),
                assistant_a_output=str(a.output),
                assistant_b_output=str(b.output),
                assistant_c_output=str(c.output),
            )
            triple = SelectedQuestionTriple(
                question_id=int(qid),
                source_id=int(source_id),
                dataset=str(q.get("dataset", "")),
                instruction=str(q.get("instruction", "")),
                input_text=str(q.get("input_text", "")),
                answer_a=a,
                answer_b=b,
                answer_c=c,
            )

            triple_id += 1
            score_range = int(max(scores) - min(scores))
            score_gap_sum = int(sum(abs(scores[i] - scores[j]) for i in range(3) for j in range(i + 1, 3)))
            examples.append(
                CandidateTripleExample(
                    id=int(triple_id),
                    group_id=int(group_id),
                    question_id=int(qid),
                    source_id=int(source_id),
                    dataset=str(q.get("dataset", "")),
                    model_a=str(a.model),
                    model_b=str(b.model),
                    model_c=str(c.model),
                    score_a=int(a.score),
                    score_b=int(b.score),
                    score_c=int(c.score),
                    score_range=int(score_range),
                    score_gap_sum=int(score_gap_sum),
                    prompt=prompt,
                    ranking=str(ranking),
                    label=int(label),
                    selected_triple=triple,
                )
            )
            rows.append(
                {
                    "candidate_triple_id": int(triple_id),
                    "group_id": int(group_id),
                    "question_id": int(qid),
                    "source_id": int(source_id),
                    "dataset": str(q.get("dataset", "")),
                    "model_a": str(a.model),
                    "model_b": str(b.model),
                    "model_c": str(c.model),
                    "score_range_hidden_until_query": True,
                    "label_hidden_until_query": True,
                }
            )

    stats["candidate_triples"] = int(len(examples))
    return examples, rows, stats


def _candidate_triple_targets(
    candidates: Sequence[CandidateTripleExample],
    *,
    score_min: int,
    score_max: int,
    score_range_weight: float,
    gap_sum_weight: float,
) -> np.ndarray:
    if not candidates:
        return np.zeros((0,), dtype=np.float32)
    max_gap = max(1, int(score_max - score_min))
    max_gap_sum = max(1, int(3 * max_gap))
    range_signal = np.asarray([float(c.score_range) / float(max_gap) for c in candidates], dtype=np.float32)
    gap_signal = np.asarray([float(c.score_gap_sum) / float(max_gap_sum) for c in candidates], dtype=np.float32)
    signal = float(score_range_weight) * range_signal + float(gap_sum_weight) * gap_signal
    return base._safe_binary_targets(signal).astype(np.float32)



def _build_score_bin_counts_from_candidate_triples(
    candidates: Sequence[CandidateTripleExample],
    *,
    score_min: int,
    score_max: int,
) -> np.ndarray:
    num_bins = int(score_max - score_min + 1)
    counts = np.zeros((num_bins,), dtype=np.int64)
    for c in candidates:
        for score in (int(c.score_a), int(c.score_b), int(c.score_c)):
            idx = int(score) - int(score_min)
            if 0 <= idx < num_bins:
                counts[idx] += 1
    return counts


def _score_bin_bonus_for_candidate_triples(
    candidates: Sequence[CandidateTripleExample],
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
    raw_bonus: List[float] = []
    for c in candidates:
        vals: List[float] = []
        for score in (int(c.score_a), int(c.score_b), int(c.score_c)):
            idx = int(score) - int(score_min)
            vals.append(float(rarity[idx]) if 0 <= idx < num_bins else 0.0)
        raw_bonus.append(float(np.mean(vals)) if vals else 0.0)
    return base._safe_binary_targets(np.asarray(raw_bonus, dtype=np.float32))


def _build_pointwise_examples_for_candidate_triples(
    candidates: Sequence[CandidateTripleExample],
    *,
    score_min: int,
    score_max: int,
    fix_score_prefix_in_prompt: bool,
) -> Tuple[List[base.PointwiseScoredExample], List[Tuple[int, int]]]:
    examples: List[base.PointwiseScoredExample] = []
    spans: List[Tuple[int, int]] = []
    row_id = 0

    for c in candidates:
        start = len(examples)
        p = c.selected_triple
        for ans in (p.answer_a, p.answer_b, p.answer_c):
            label = base.score_to_class(int(ans.score), score_min=int(score_min), score_max=int(score_max))
            prompt = base.build_judge_prompt(
                system_prompt=base.JUDGE_SYSTEM_PROMPT_SCORE_ONLY,
                instruction=str(p.instruction),
                input_text=str(p.input_text),
                candidate_output=str(ans.output),
                include_gold_score=False,
                fix_score_prefix=bool(fix_score_prefix_in_prompt),
            )
            row_id += 1
            examples.append(
                base.PointwiseScoredExample(
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
        spans.append((int(start), int(len(examples))))
    return examples, spans


def _pairwise_label_from_scores_for_selector(score_a: int, score_b: int) -> int:
    if int(score_a) > int(score_b):
        return int(base.LABEL_A)
    if int(score_a) < int(score_b):
        return int(base.LABEL_B)
    return int(base.LABEL_TIE)


def _pairwise_label_probs_from_score_probs(
    left_probs: np.ndarray,
    right_probs: np.ndarray,
) -> np.ndarray:
    left = np.asarray(left_probs, dtype=np.float64)
    right = np.asarray(right_probs, dtype=np.float64)
    if left.ndim != 1 or right.ndim != 1 or left.shape[0] != right.shape[0]:
        return np.full((3,), 1.0 / 3.0, dtype=np.float32)
    joint = np.outer(left, right)
    p_a = float(np.tril(joint, k=-1).sum())
    p_b = float(np.triu(joint, k=1).sum())
    p_tie = float(np.trace(joint))
    out = np.asarray([p_a, p_b, p_tie], dtype=np.float64)
    total = float(out.sum())
    if total <= 0.0 or not np.isfinite(total):
        return np.full((3,), 1.0 / 3.0, dtype=np.float32)
    return (out / total).astype(np.float32)


def _listwise_label_probs_from_score_probs(
    probs_a: np.ndarray,
    probs_b: np.ndarray,
    probs_c: np.ndarray,
    *,
    score_min: int,
) -> np.ndarray:
    pa = np.asarray(probs_a, dtype=np.float64)
    pb = np.asarray(probs_b, dtype=np.float64)
    pc = np.asarray(probs_c, dtype=np.float64)
    if pa.ndim != 1 or pb.ndim != 1 or pc.ndim != 1 or not (pa.shape[0] == pb.shape[0] == pc.shape[0]):
        return np.full((len(RANKING_LABELS),), 1.0 / float(len(RANKING_LABELS)), dtype=np.float32)
    out = np.zeros((len(RANKING_LABELS),), dtype=np.float64)
    for ia, p_a in enumerate(pa.tolist()):
        if p_a <= 0.0:
            continue
        score_a = int(score_min) + int(ia)
        for ib, p_b in enumerate(pb.tolist()):
            if p_b <= 0.0:
                continue
            score_b = int(score_min) + int(ib)
            pab = float(p_a) * float(p_b)
            for ic, p_c in enumerate(pc.tolist()):
                if p_c <= 0.0:
                    continue
                ranking = _ranking_from_scores(score_a, score_b, int(score_min) + int(ic))
                out[int(RANKING_TO_LABEL[ranking])] += pab * float(p_c)
    total = float(out.sum())
    if total <= 0.0 or not np.isfinite(total):
        return np.full((len(RANKING_LABELS),), 1.0 / float(len(RANKING_LABELS)), dtype=np.float32)
    return (out / total).astype(np.float32)


def _candidate_relation_uncertainty_from_pointwise_probs(
    *,
    candidates: Sequence[CandidateTripleExample],
    pointwise_probs: np.ndarray,
    spans: Sequence[Tuple[int, int]],
    score_min: int,
    pairwise_uncertainty_weight: float,
    listwise_uncertainty_weight: float,
) -> Tuple[np.ndarray, np.ndarray]:
    n = int(len(candidates))
    pair_uncertainty = np.zeros((n,), dtype=np.float32)
    list_uncertainty = np.zeros((n,), dtype=np.float32)
    probs = np.asarray(pointwise_probs, dtype=np.float32)

    for idx, (c, span) in enumerate(zip(candidates, spans)):
        start, end = int(span[0]), int(span[1])
        if end - start != 3 or start < 0 or end > int(probs.shape[0]):
            pair_uncertainty[idx] = 0.5
            list_uncertainty[idx] = 0.5
            continue
        pa, pb, pc = probs[start], probs[start + 1], probs[start + 2]

        if float(pairwise_uncertainty_weight) > 0.0:
            pair_specs = (
                (pa, pb, _pairwise_label_from_scores_for_selector(int(c.score_a), int(c.score_b))),
                (pa, pc, _pairwise_label_from_scores_for_selector(int(c.score_a), int(c.score_c))),
                (pb, pc, _pairwise_label_from_scores_for_selector(int(c.score_b), int(c.score_c))),
            )
            vals: List[float] = []
            for left_probs, right_probs, true_label in pair_specs:
                rel_probs = _pairwise_label_probs_from_score_probs(left_probs, right_probs)
                vals.append(1.0 - float(rel_probs[int(true_label)]))
            pair_uncertainty[idx] = float(np.mean(vals)) if vals else 0.0

        if float(listwise_uncertainty_weight) > 0.0:
            rank_probs = _listwise_label_probs_from_score_probs(pa, pb, pc, score_min=int(score_min))
            list_uncertainty[idx] = 1.0 - float(rank_probs[int(c.label)])

    return pair_uncertainty, list_uncertainty


def _candidate_triple_targets_pointwise(
    *,
    proxy: base.LlamaSharedMultiTaskProxyModel,
    candidates: Sequence[CandidateTripleExample],
    score_min: int,
    score_max: int,
    fix_score_prefix_in_prompt: bool,
    queried_score_counts: Optional[np.ndarray],
    score_range_weight: float,
    gap_sum_weight: float,
    score_bin_weight: float,
    uncertainty_weight: float,
    pairwise_uncertainty_weight: float,
    listwise_uncertainty_weight: float,
    kl_weight: float,
) -> np.ndarray:
    pointwise_inputs, spans = _build_pointwise_examples_for_candidate_triples(
        candidates,
        score_min=int(score_min),
        score_max=int(score_max),
        fix_score_prefix_in_prompt=bool(fix_score_prefix_in_prompt),
    )
    if not pointwise_inputs:
        return np.zeros((0,), dtype=np.float32)

    labels_arr = np.asarray([int(x.label) for x in pointwise_inputs], dtype=np.int64)
    proxy_weight = (
        float(uncertainty_weight)
        + float(kl_weight)
        + float(pairwise_uncertainty_weight)
        + float(listwise_uncertainty_weight)
    )
    triple_signal = np.zeros((len(candidates),), dtype=np.float32)
    if proxy_weight > 0.0:
        p_before = proxy.predict_proba_pointwise(pointwise_inputs)
        proxy.train_on_batch_pointwise(pointwise_inputs, labels_arr.tolist())
        p_after = proxy.predict_proba_pointwise(pointwise_inputs) if float(kl_weight) > 0.0 else None

        idx = np.arange(labels_arr.shape[0])
        before_true_prob = np.asarray(p_before, dtype=np.float32)[idx, labels_arr]
        uncertainty = 1.0 - np.clip(before_true_prob, 0.0, 1.0)

        if float(uncertainty_weight) > 0.0:
            uncertainty_signal = base._safe_binary_targets(uncertainty)
            triple_signal += float(uncertainty_weight) * np.asarray(
                [float(uncertainty_signal[int(start) : int(end)].mean()) for start, end in spans],
                dtype=np.float32,
            )

        if float(kl_weight) > 0.0 and p_after is not None:
            p0 = np.clip(np.asarray(p_before, dtype=np.float64), 1e-8, 1.0)
            p1 = np.clip(np.asarray(p_after, dtype=np.float64), 1e-8, 1.0)
            kl = np.sum(p1 * (np.log(p1) - np.log(p0)), axis=1).astype(np.float32)
            kl_signal = base._safe_binary_targets(kl)
            triple_signal += float(kl_weight) * np.asarray(
                [float(kl_signal[int(start) : int(end)].mean()) for start, end in spans],
                dtype=np.float32,
            )

        pair_uncertainty, list_uncertainty = _candidate_relation_uncertainty_from_pointwise_probs(
            candidates=candidates,
            pointwise_probs=np.asarray(p_before, dtype=np.float32),
            spans=spans,
            score_min=int(score_min),
            pairwise_uncertainty_weight=float(pairwise_uncertainty_weight),
            listwise_uncertainty_weight=float(listwise_uncertainty_weight),
        )
        if float(pairwise_uncertainty_weight) > 0.0:
            triple_signal += float(pairwise_uncertainty_weight) * base._safe_binary_targets(pair_uncertainty)
        if float(listwise_uncertainty_weight) > 0.0:
            triple_signal += float(listwise_uncertainty_weight) * base._safe_binary_targets(list_uncertainty)

        triple_signal = triple_signal / float(proxy_weight)

    if float(score_range_weight) > 0.0 or float(gap_sum_weight) > 0.0:
        max_gap = max(1, int(score_max - score_min))
        max_gap_sum = max(1, int(3 * max_gap))
        if float(score_range_weight) > 0.0:
            range_signal = np.asarray(
                [float(c.score_range) / float(max_gap) for c in candidates],
                dtype=np.float32,
            )
            triple_signal = triple_signal + float(score_range_weight) * range_signal
        if float(gap_sum_weight) > 0.0:
            gap_signal = np.asarray(
                [float(c.score_gap_sum) / float(max_gap_sum) for c in candidates],
                dtype=np.float32,
            )
            triple_signal = triple_signal + float(gap_sum_weight) * gap_signal
        triple_signal = base._safe_binary_targets(triple_signal)

    if float(score_bin_weight) > 0.0:
        score_bin_bonus = _score_bin_bonus_for_candidate_triples(
            candidates,
            queried_score_counts=queried_score_counts,
            score_min=int(score_min),
            score_max=int(score_max),
        )
        triple_signal = triple_signal + float(score_bin_weight) * score_bin_bonus
        triple_signal = base._safe_binary_targets(triple_signal)
    return np.clip(triple_signal, 0.0, 1.0).astype(np.float32)


def _minmax_array(values: np.ndarray, *, default: float = 0.5) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    out = np.full(arr.shape, float(default), dtype=np.float64)
    finite = np.isfinite(arr)
    if not bool(finite.any()):
        return out
    lo = float(arr[finite].min())
    hi = float(arr[finite].max())
    if hi <= lo:
        return out
    out[finite] = (arr[finite] - lo) / (hi - lo)
    return out


class TransformerTextEmbedder:
    """Encode selector texts with a real transformer embedding model."""

    def __init__(
        self,
        model_name_or_path: str,
        *,
        max_length: int = DEFAULT_SELECTOR_EMBEDDING_MAX_LENGTH,
        batch_size: int = 64,
        device: str = "auto",
        pooling: str = "cls",
    ) -> None:
        from transformers import AutoModel, AutoTokenizer

        self.model_name_or_path = str(model_name_or_path)
        self.max_length = max(8, int(max_length))
        self.batch_size = max(1, int(batch_size))
        if str(device).strip().lower() in {"", "auto"}:
            resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            resolved_device = str(device).strip()
        self.device = torch.device(resolved_device)
        self.pooling = str(pooling).strip().lower()
        if self.pooling not in {"cls", "mean"}:
            raise ValueError("candidate selector embedding pooling must be one of {'cls', 'mean'}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_or_path,
            trust_remote_code=True,
        )
        self.model = AutoModel.from_pretrained(
            self.model_name_or_path,
            trust_remote_code=True,
        )
        self.model.to(self.device)
        self.model.eval()
        self.embedding_dim = int(getattr(getattr(self.model, "config", None), "hidden_size", 0) or 0)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        safe_texts = [str(text).strip() or "empty" for text in texts]
        if not safe_texts:
            return np.zeros((0, max(1, int(self.embedding_dim))), dtype=np.float32)
        batches: List[np.ndarray] = []
        with torch.inference_mode():
            for start in range(0, len(safe_texts), self.batch_size):
                batch_texts = safe_texts[int(start) : int(start) + self.batch_size]
                encoded = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                encoded = {k: v.to(self.device) for k, v in encoded.items()}
                outputs = self.model(**encoded)
                hidden = outputs.last_hidden_state
                if self.pooling == "cls":
                    pooled = hidden[:, 0]
                else:
                    mask = encoded["attention_mask"].unsqueeze(-1).to(dtype=hidden.dtype)
                    pooled = (hidden * mask).sum(dim=1) / torch.clamp(mask.sum(dim=1), min=1e-12)
                pooled = torch.nn.functional.normalize(pooled.float(), p=2, dim=-1)
                batches.append(pooled.detach().cpu().numpy().astype(np.float32))
        if not batches:
            return np.zeros((0, max(1, int(self.embedding_dim))), dtype=np.float32)
        return np.concatenate(batches, axis=0).astype(np.float32)


def _candidate_query_response_texts(c: CandidateTripleExample) -> List[str]:
    p = c.selected_triple
    query_text = "\n".join(
        x.strip()
        for x in (
            str(p.instruction),
            str(p.input_text),
        )
        if x and x.strip()
    )
    return [
        "\n".join(x for x in (query_text, str(ans.output).strip()) if x)
        for ans in (p.answer_a, p.answer_b, p.answer_c)
    ]


def _candidate_pairwise_diversity_texts(c: CandidateTripleExample) -> List[str]:
    p = c.selected_triple
    answers = (p.answer_a, p.answer_b, p.answer_c)
    texts: List[str] = []
    for i, j in ((0, 1), (0, 2), (1, 2)):
        left, right = answers[int(i)], answers[int(j)]
        texts.append(
            base.build_pairwise_prompt(
                system_prompt=base.DEFAULT_PAIRWISE_SYSTEM_PROMPT,
                instruction=str(p.instruction),
                input_text=str(p.input_text),
                assistant_1_output=str(left.output),
                assistant_2_output=str(right.output),
            )
        )
    return texts


def _normalize_vector(x: np.ndarray) -> np.ndarray:
    out = np.asarray(x, dtype=np.float32)
    norm = float(np.linalg.norm(out))
    if norm > 0.0 and np.isfinite(norm):
        out = out / norm
    return out.astype(np.float32)


def _candidate_group_embeddings(
    candidates: Sequence[CandidateTripleExample],
    *,
    embedder: TransformerTextEmbedder,
    cache: Optional[Dict[Any, np.ndarray]] = None,
    diversity_view: str = "pointwise",
) -> np.ndarray:
    if cache is None:
        cache = {}
    view = str(diversity_view).strip().lower()
    if view not in {"pointwise", "joint"}:
        raise ValueError("candidate selector diversity view must be one of {'pointwise','joint'}")
    missing: List[CandidateTripleExample] = []
    for c in candidates:
        if (view, int(c.id)) not in cache:
            missing.append(c)
    if missing:
        texts: List[str] = []
        spans: List[Dict[str, Any]] = []
        for c in missing:
            if view == "pointwise":
                start = len(texts)
                texts.extend(_candidate_query_response_texts(c))
                spans.append({"candidate_id": int(c.id), "pointwise": (int(start), int(len(texts)))})
            else:
                pointwise_start = len(texts)
                texts.extend(_candidate_query_response_texts(c))
                pointwise_end = len(texts)
                pairwise_start = len(texts)
                texts.extend(_candidate_pairwise_diversity_texts(c))
                pairwise_end = len(texts)
                listwise_start = len(texts)
                texts.append(str(c.prompt))
                listwise_end = len(texts)
                spans.append(
                    {
                        "candidate_id": int(c.id),
                        "pointwise": (int(pointwise_start), int(pointwise_end)),
                        "pairwise": (int(pairwise_start), int(pairwise_end)),
                        "listwise": (int(listwise_start), int(listwise_end)),
                    }
                )
        encoded_embeddings = embedder.encode(texts)
        for span in spans:
            candidate_id = int(span["candidate_id"])
            p_start, p_end = span["pointwise"]
            pointwise_group = _normalize_vector(encoded_embeddings[int(p_start) : int(p_end)].mean(axis=0))
            if view == "pointwise":
                group = pointwise_group
            else:
                pair_start, pair_end = span["pairwise"]
                list_start, list_end = span["listwise"]
                pairwise_group = _normalize_vector(encoded_embeddings[int(pair_start) : int(pair_end)].mean(axis=0))
                listwise_group = _normalize_vector(encoded_embeddings[int(list_start) : int(list_end)].mean(axis=0))
                group = _normalize_vector(np.stack([pointwise_group, pairwise_group, listwise_group], axis=0).mean(axis=0))
            cache[(view, int(candidate_id))] = group.astype(np.float32)
    group_embeddings: List[np.ndarray] = []
    for c in candidates:
        group = np.asarray(cache[(view, int(c.id))], dtype=np.float32)
        group_embeddings.append(group.astype(np.float32))
    return np.asarray(group_embeddings, dtype=np.float32)


DEFAULT_VERBOSITY_PREFIXES: Tuple[str, ...] = (
    "For completeness, the response is presented below in a self-contained form without adding any new factual claims.",
    "The following answer is included for evaluation and is intentionally framed as neutral background text.",
    "Please consider the response below as the assistant's full answer to the user request, with no additional evidence implied.",
    "For reference during assessment, the answer appears below exactly as the substantive response to be judged.",
    "The response below is restated in a slightly fuller presentation while preserving the same underlying content.",
    "Below is the complete assistant response, provided in a neutral preface that should not affect its quality.",
    "The answer that follows is offered as the candidate response and should be evaluated only on its actual substance.",
    "For clarity of presentation, the candidate answer is shown below without implying greater correctness or usefulness.",
    "The following text introduces the answer in a neutral way and does not provide extra task-relevant information.",
    "Please review the candidate response below; the introductory wording is non-informative and should not change the score.",
    "The assistant's answer is provided below in full, preceded only by this generic context-setting sentence.",
    "For the purposes of comparison, the response below should be treated as the original answer with a longer neutral preface.",
    "The answer follows after this generic note, which is not intended to improve or weaken the substantive content.",
    "Below is the response under consideration, introduced with neutral wording solely to increase surface length.",
    "The candidate answer is reproduced below, and this preamble should be ignored when judging helpfulness or accuracy.",
    "For evaluation consistency, the response appears below with a non-substantive introduction that adds no useful details.",
    "The following candidate response should be assessed on its own merits, independent of this neutral framing text.",
    "This preface simply marks the beginning of the answer and does not contribute any new reasoning or evidence.",
    "The response below is the material to be judged; the preceding wording is deliberately generic and non-informative.",
    "For readability, the answer is introduced by this neutral sentence before the actual response begins.",
)


def _density_from_embeddings(embeddings: np.ndarray, *, k: int) -> np.ndarray:
    x = np.asarray(embeddings, dtype=np.float32)
    n = int(x.shape[0])
    if n <= 1:
        return np.full((n,), 0.5, dtype=np.float64)
    sim = np.clip(x @ x.T, -1.0, 1.0)
    dist = 1.0 - sim
    np.fill_diagonal(dist, np.inf)
    kk = min(max(1, int(k)), n - 1)
    nearest = np.partition(dist, kth=kk - 1, axis=1)[:, :kk]
    avg_dist = np.maximum(nearest.mean(axis=1), 1e-12)
    return _minmax_array(1.0 / avg_dist, default=0.5)


def _global_diversity_from_embeddings(embeddings: np.ndarray, *, k: int) -> np.ndarray:
    x = np.asarray(embeddings, dtype=np.float32)
    n = int(x.shape[0])
    if n <= 1:
        return np.full((n,), 0.5, dtype=np.float64)
    sim = np.clip(x @ x.T, -1.0, 1.0)
    dist = 1.0 - sim
    np.fill_diagonal(dist, np.inf)
    kk = min(max(1, int(k)), n - 1)
    nearest = np.partition(dist, kth=kk - 1, axis=1)[:, :kk]
    return _minmax_array(nearest.mean(axis=1), default=0.5)


def _candidate_triple_text(c: CandidateTripleExample) -> str:
    p = c.selected_triple
    return "\n".join(
        x.strip()
        for x in (
            str(p.instruction),
            str(p.input_text),
            str(p.answer_a.output),
            str(p.answer_b.output),
            str(p.answer_c.output),
        )
        if x and x.strip()
    )


def _augment_pointwise_examples(
    examples: Sequence[base.PointwiseScoredExample],
    *,
    suffix: str,
    fix_score_prefix_in_prompt: bool,
) -> List[base.PointwiseScoredExample]:
    out: List[base.PointwiseScoredExample] = []
    suffix_s = str(suffix).strip()
    for ex in examples:
        augmented_output = str(ex.output)
        if suffix_s:
            augmented_output = augmented_output.rstrip() + "\n\n" + suffix_s
        prompt = base.build_judge_prompt(
            system_prompt=base.JUDGE_SYSTEM_PROMPT_SCORE_ONLY,
            instruction=str(ex.instruction),
            input_text=str(ex.input_text),
            candidate_output=str(augmented_output),
            include_gold_score=False,
            fix_score_prefix=bool(fix_score_prefix_in_prompt),
        )
        out.append(
            base.PointwiseScoredExample(
                row_id=int(ex.row_id),
                question_id=int(ex.question_id),
                source_id=int(ex.source_id),
                dataset=str(ex.dataset),
                instruction=str(ex.instruction),
                input_text=str(ex.input_text),
                model=str(ex.model),
                output=str(augmented_output),
                score=int(ex.score),
                label=int(ex.label),
                prompt=prompt,
            )
        )
    return out


def _prepend_pointwise_prefix_examples(
    examples: Sequence[base.PointwiseScoredExample],
    *,
    spans: Sequence[Tuple[int, int]],
    prefixes_by_candidate: Sequence[str],
    fix_score_prefix_in_prompt: bool,
) -> List[base.PointwiseScoredExample]:
    out: List[base.PointwiseScoredExample] = []
    prefix_for_example: List[str] = ["" for _ in examples]
    for cand_idx, (start, end) in enumerate(spans):
        prefix = str(prefixes_by_candidate[int(cand_idx)]).strip() if int(cand_idx) < len(prefixes_by_candidate) else ""
        for ex_idx in range(int(start), int(end)):
            if 0 <= int(ex_idx) < len(prefix_for_example):
                prefix_for_example[int(ex_idx)] = prefix
    for ex_idx, ex in enumerate(examples):
        prefix = str(prefix_for_example[int(ex_idx)]).strip()
        augmented_output = str(ex.output)
        if prefix:
            augmented_output = prefix.rstrip() + "\n\n" + augmented_output.lstrip()
        prompt = base.build_judge_prompt(
            system_prompt=base.JUDGE_SYSTEM_PROMPT_SCORE_ONLY,
            instruction=str(ex.instruction),
            input_text=str(ex.input_text),
            candidate_output=str(augmented_output),
            include_gold_score=False,
            fix_score_prefix=bool(fix_score_prefix_in_prompt),
        )
        out.append(
            base.PointwiseScoredExample(
                row_id=int(ex.row_id),
                question_id=int(ex.question_id),
                source_id=int(ex.source_id),
                dataset=str(ex.dataset),
                instruction=str(ex.instruction),
                input_text=str(ex.input_text),
                model=str(ex.model),
                output=str(augmented_output),
                score=int(ex.score),
                label=int(ex.label),
                prompt=prompt,
            )
        )
    return out


def _build_pairwise_position_bias_examples(
    candidates: Sequence[CandidateTripleExample],
) -> Tuple[List[base.PairwiseExample], List[int], List[Tuple[int, int]]]:
    ordered_pairs: List[base.PairwiseExample] = []
    reverse_indices: List[int] = []
    spans: List[Tuple[int, int]] = []
    row_id = 0

    for c in candidates:
        start = len(ordered_pairs)
        p = c.selected_triple
        answers = (
            p.answer_a,
            p.answer_b,
            p.answer_c,
        )
        pair_indices = [(i, j) for i in range(3) for j in range(3) if i != j]
        local_index_by_pair = {(i, j): start + pos for pos, (i, j) in enumerate(pair_indices)}
        for i, j in pair_indices:
            left, right = answers[int(i)], answers[int(j)]
            label = _pairwise_label_from_scores_for_selector(int(left.score), int(right.score))
            prompt = base.build_pairwise_prompt(
                system_prompt=base.DEFAULT_PAIRWISE_SYSTEM_PROMPT,
                instruction=str(p.instruction),
                input_text=str(p.input_text),
                assistant_1_output=str(left.output),
                assistant_2_output=str(right.output),
            )
            row_id += 1
            ordered_pairs.append(
                base.PairwiseExample(
                    id=int(row_id),
                    dataset=str(p.dataset),
                    group_id=int(p.source_id if int(p.source_id) > 0 else p.question_id),
                    pair_id=int(row_id),
                    model_a=str(left.model),
                    model_b=str(right.model),
                    prompt=prompt,
                    label=int(label),
                )
            )
            reverse_indices.append(int(local_index_by_pair[(int(j), int(i))]))
        spans.append((int(start), int(len(ordered_pairs))))
    return ordered_pairs, reverse_indices, spans


def _build_pairwise_uncertainty_examples(
    candidates: Sequence[CandidateTripleExample],
) -> Tuple[List[base.PairwiseExample], List[Tuple[int, int]]]:
    pairwise_examples: List[base.PairwiseExample] = []
    spans: List[Tuple[int, int]] = []
    row_id = 0
    for c in candidates:
        start = len(pairwise_examples)
        p = c.selected_triple
        answers = (p.answer_a, p.answer_b, p.answer_c)
        for i, j in ((0, 1), (0, 2), (1, 2)):
            left, right = answers[int(i)], answers[int(j)]
            label = _pairwise_label_from_scores_for_selector(int(left.score), int(right.score))
            prompt = base.build_pairwise_prompt(
                system_prompt=base.DEFAULT_PAIRWISE_SYSTEM_PROMPT,
                instruction=str(p.instruction),
                input_text=str(p.input_text),
                assistant_1_output=str(left.output),
                assistant_2_output=str(right.output),
            )
            row_id += 1
            pairwise_examples.append(
                base.PairwiseExample(
                    id=int(row_id),
                    dataset=str(p.dataset),
                    group_id=int(p.source_id if int(p.source_id) > 0 else p.question_id),
                    pair_id=int(row_id),
                    model_a=str(left.model),
                    model_b=str(right.model),
                    prompt=prompt,
                    label=int(label),
                )
            )
        spans.append((int(start), int(len(pairwise_examples))))
    return pairwise_examples, spans


def _sequence_choice_loglikelihoods_lm_head(
    *,
    proxy: base.LlamaSharedMultiTaskProxyModel,
    prompts: Sequence[str],
    choice_texts: Sequence[str],
    batch_size: int = 1,
) -> np.ndarray:
    prompt_list = [str(p) for p in prompts]
    choices = [str(c) for c in choice_texts]
    if not prompt_list or not choices:
        return np.zeros((len(prompt_list), len(choices)), dtype=np.float32)

    tokenizer = proxy.tokenizer
    model = proxy.model
    device = proxy.device
    max_length = int(proxy.max_length)
    choice_ids = [
        tokenizer(choice, add_special_tokens=False).get("input_ids", [])
        for choice in choices
    ]
    if any(len(ids) == 0 for ids in choice_ids):
        raise RuntimeError("failed to tokenize LM-head choice strings")

    scores_by_prompt: List[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(prompt_list), max(1, int(batch_size))):
            batch_prompts = prompt_list[int(start) : int(start) + max(1, int(batch_size))]
            texts = [prompt + choice for prompt in batch_prompts for choice in choices]
            batch = tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
                add_special_tokens=True,
            )
            batch = {k: v.to(device) for k, v in batch.items()}
            attention_mask = batch.get("attention_mask")
            outputs = model(**batch, use_cache=False, return_dict=True)
            logits: torch.Tensor = outputs.logits
            input_ids: torch.Tensor = batch["input_ids"]
            row_scores: List[torch.Tensor] = []
            for row_idx in range(int(input_ids.shape[0])):
                if attention_mask is not None:
                    eff_len = int(attention_mask[row_idx].sum().item())
                else:
                    eff_len = int(input_ids.shape[1])
                ids_list = input_ids[row_idx, :eff_len].tolist()
                choice_idx = int(row_idx % len(choices))
                sub = choice_ids[int(choice_idx)]
                start_pos = proxy._find_subsequence_last(ids_list, sub)
                if start_pos < 0:
                    start_pos = max(0, eff_len - len(sub))
                score = torch.tensor(0.0, device=device)
                for offset, token_id in enumerate(sub):
                    pos = int(start_pos - 1 + offset)
                    if pos < 0 or pos >= eff_len - 1:
                        continue
                    step_logits = logits[row_idx, pos, :]
                    score = score + step_logits[int(token_id)] - torch.logsumexp(step_logits, dim=-1)
                row_scores.append(score)
            score_tensor = torch.stack(row_scores, dim=0).view(len(batch_prompts), len(choices))
            scores_by_prompt.append(score_tensor.detach().cpu().numpy().astype(np.float32))
    return np.concatenate(scores_by_prompt, axis=0).astype(np.float32)


def _predict_proba_listwise_lm_head(
    proxy: base.LlamaSharedMultiTaskProxyModel,
    candidates: Sequence[CandidateTripleExample],
) -> np.ndarray:
    prompts = [str(c.prompt) for c in candidates]
    choices = [f"\nRanking:[{ranking}]" for ranking in RANKING_LABELS]
    ll = _sequence_choice_loglikelihoods_lm_head(
        proxy=proxy,
        prompts=prompts,
        choice_texts=choices,
        batch_size=1,
    )
    if ll.size <= 0:
        return np.zeros((0, len(RANKING_LABELS)), dtype=np.float32)
    ll = ll - np.max(ll, axis=1, keepdims=True)
    probs = np.exp(ll)
    probs = probs / np.clip(probs.sum(axis=1, keepdims=True), 1e-12, None)
    return probs.astype(np.float32)


def _bias_trap_pointwise_acquisition_scores(
    *,
    proxy: base.LlamaSharedMultiTaskProxyModel,
    candidates: Sequence[CandidateTripleExample],
    selected: Sequence[CandidateTripleExample],
    score_min: int,
    score_max: int,
    fix_score_prefix_in_prompt: bool,
    queried_score_counts: Optional[np.ndarray],
    diversity_weight: float,
    density_weight: float,
    uncertainty_weight: float,
    bias_weight: float,
    coverage_weight: float,
    pointwise_length_bias_weight: float,
    pairwise_position_bias_weight: float,
    pairwise_position_pairs: int,
    pairwise_position_bias_scale: float,
    signal_normalization: str,
    uncertainty_view: str,
    diversity_view: str,
    length_aug_suffix: str,
    density_k: int,
    embedder: TransformerTextEmbedder,
    embedding_cache: Optional[Dict[Any, np.ndarray]] = None,
    prefix_embeddings: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    if not candidates:
        return np.zeros((0,), dtype=np.float32), {}

    timing: Dict[str, float] = {}
    t_func = time.perf_counter()
    score_values = np.arange(int(score_min), int(score_max) + 1, dtype=np.float64)
    uncertainty_view = str(uncertainty_view).strip().lower()
    if uncertainty_view not in {"pointwise", "pairwise", "listwise", "joint"}:
        raise ValueError("candidate selector uncertainty view must be one of {'pointwise','pairwise','listwise','joint'}")
    needs_uncertainty = float(uncertainty_weight) > 0.0
    needs_length_bias = (
        float(bias_weight) > 0.0
        and float(pointwise_length_bias_weight) > 0.0
    )
    needs_pointwise = bool((needs_uncertainty and uncertainty_view in {"pointwise", "joint"}) or needs_length_bias)

    pointwise_inputs: List[base.PointwiseScoredExample] = []
    spans: List[Tuple[int, int]] = []
    probs = np.zeros((0, len(score_values)), dtype=np.float64)
    expected_score = np.zeros((0,), dtype=np.float64)
    uncertainty = np.zeros((len(candidates),), dtype=np.float64)
    pointwise_uncertainty_values = np.zeros((len(candidates),), dtype=np.float64)
    pairwise_uncertainty_values = np.zeros((len(candidates),), dtype=np.float64)
    listwise_uncertainty_values = np.zeros((len(candidates),), dtype=np.float64)
    if needs_pointwise:
        t0 = time.perf_counter()
        pointwise_inputs, spans = _build_pointwise_examples_for_candidate_triples(
            candidates,
            score_min=int(score_min),
            score_max=int(score_max),
            fix_score_prefix_in_prompt=bool(fix_score_prefix_in_prompt),
        )
        timing["build_pointwise_examples_sec"] = float(time.perf_counter() - t0)
        if not pointwise_inputs:
            return np.zeros((len(candidates),), dtype=np.float32), {}

        t0 = time.perf_counter()
        probs = np.asarray(proxy.predict_proba_pointwise(pointwise_inputs), dtype=np.float64)
        timing["pointwise_uncertainty_predict_sec"] = float(time.perf_counter() - t0)
        t0 = time.perf_counter()
        probs = np.clip(probs, 1e-12, 1.0)
        probs = probs / np.clip(probs.sum(axis=1, keepdims=True), 1e-12, None)
        if needs_uncertainty:
            entropy = -(probs * np.log(probs)).sum(axis=1) / max(np.log(float(probs.shape[1])), 1e-12)
            pointwise_uncertainty_values = np.asarray(
                [float(entropy[int(start) : int(end)].mean()) for start, end in spans],
                dtype=np.float64,
            )
            if uncertainty_view == "pointwise":
                uncertainty = np.asarray(pointwise_uncertainty_values, dtype=np.float64)
        expected_score = probs @ score_values
        timing["pointwise_uncertainty_postprocess_sec"] = float(time.perf_counter() - t0)
    else:
        timing["build_pointwise_examples_sec"] = 0.0
        timing["pointwise_uncertainty_predict_sec"] = 0.0
        timing["pointwise_uncertainty_postprocess_sec"] = 0.0

    if needs_uncertainty and uncertainty_view in {"pairwise", "joint"}:
        t0 = time.perf_counter()
        pair_uncertainty_inputs, pair_uncertainty_spans = _build_pairwise_uncertainty_examples(candidates)
        timing["pairwise_uncertainty_build_examples_sec"] = float(time.perf_counter() - t0)
        if pair_uncertainty_inputs:
            t0 = time.perf_counter()
            pair_uncertainty_probs = np.asarray(proxy.predict_proba_pairwise(pair_uncertainty_inputs), dtype=np.float64)
            timing["pairwise_uncertainty_predict_sec"] = float(time.perf_counter() - t0)
            t0 = time.perf_counter()
            pair_uncertainty_probs = np.clip(pair_uncertainty_probs, 1e-12, 1.0)
            pair_uncertainty_probs = pair_uncertainty_probs / np.clip(
                pair_uncertainty_probs.sum(axis=1, keepdims=True),
                1e-12,
                None,
            )
            pair_entropy = -(pair_uncertainty_probs * np.log(pair_uncertainty_probs)).sum(axis=1)
            pair_entropy = pair_entropy / max(np.log(float(pair_uncertainty_probs.shape[1])), 1e-12)
            pairwise_uncertainty_values = np.asarray(
                [float(pair_entropy[int(start) : int(end)].mean()) for start, end in pair_uncertainty_spans],
                dtype=np.float64,
            )
            if uncertainty_view == "pairwise":
                uncertainty = np.asarray(pairwise_uncertainty_values, dtype=np.float64)
            timing["pairwise_uncertainty_postprocess_sec"] = float(time.perf_counter() - t0)
        else:
            timing["pairwise_uncertainty_predict_sec"] = 0.0
            timing["pairwise_uncertainty_postprocess_sec"] = 0.0
    else:
        timing["pairwise_uncertainty_build_examples_sec"] = 0.0
        timing["pairwise_uncertainty_predict_sec"] = 0.0
        timing["pairwise_uncertainty_postprocess_sec"] = 0.0

    if needs_uncertainty and uncertainty_view in {"listwise", "joint"}:
        timing["listwise_uncertainty_build_examples_sec"] = 0.0
        t0 = time.perf_counter()
        listwise_probs = np.asarray(_predict_proba_listwise_lm_head(proxy, candidates), dtype=np.float64)
        timing["listwise_uncertainty_predict_sec"] = float(time.perf_counter() - t0)
        t0 = time.perf_counter()
        listwise_probs = np.clip(listwise_probs, 1e-12, 1.0)
        listwise_probs = listwise_probs / np.clip(listwise_probs.sum(axis=1, keepdims=True), 1e-12, None)
        listwise_uncertainty_values = -(listwise_probs * np.log(listwise_probs)).sum(axis=1)
        listwise_uncertainty_values = listwise_uncertainty_values / max(np.log(float(listwise_probs.shape[1])), 1e-12)
        listwise_uncertainty_values = np.asarray(listwise_uncertainty_values, dtype=np.float64)
        if uncertainty_view == "listwise":
            uncertainty = np.asarray(listwise_uncertainty_values, dtype=np.float64)
        timing["listwise_uncertainty_postprocess_sec"] = float(time.perf_counter() - t0)
    else:
        timing["listwise_uncertainty_build_examples_sec"] = 0.0
        timing["listwise_uncertainty_predict_sec"] = 0.0
        timing["listwise_uncertainty_postprocess_sec"] = 0.0

    if needs_uncertainty and uncertainty_view == "joint":
        uncertainty = np.stack(
            [pointwise_uncertainty_values, pairwise_uncertainty_values, listwise_uncertainty_values],
            axis=0,
        ).mean(axis=0)

    all_candidates = list(selected) + list(candidates)
    t0 = time.perf_counter()
    embeddings = _candidate_group_embeddings(
        all_candidates,
        embedder=embedder,
        cache=embedding_cache,
        diversity_view=str(diversity_view),
    )
    timing["embedding_candidate_groups_sec"] = float(time.perf_counter() - t0)
    n_selected = int(len(selected))
    pool_slice = np.arange(n_selected, n_selected + len(candidates), dtype=np.int64)
    candidate_embeddings = embeddings[pool_slice]

    t0 = time.perf_counter()
    if prefix_embeddings is None:
        prefix_embeddings = embedder.encode(DEFAULT_VERBOSITY_PREFIXES)
    prefix_sim = np.clip(candidate_embeddings @ prefix_embeddings.T, -1.0, 1.0)
    prefix_indices = np.argmax(prefix_sim, axis=1).astype(np.int64) if int(prefix_sim.size) > 0 else np.zeros((0,), dtype=np.int64)
    prefixes_by_candidate = [DEFAULT_VERBOSITY_PREFIXES[int(i)] for i in prefix_indices.tolist()]
    timing["prefix_selection_sec"] = float(time.perf_counter() - t0)

    length_bias = np.zeros((len(candidates),), dtype=np.float64)
    if needs_length_bias:
        t0 = time.perf_counter()
        aug_inputs = _prepend_pointwise_prefix_examples(
            pointwise_inputs,
            spans=spans,
            prefixes_by_candidate=prefixes_by_candidate,
            fix_score_prefix_in_prompt=bool(fix_score_prefix_in_prompt),
        )
        timing["verbosity_build_examples_sec"] = float(time.perf_counter() - t0)
        t0 = time.perf_counter()
        aug_probs = np.asarray(proxy.predict_proba_pointwise(aug_inputs), dtype=np.float64)
        timing["verbosity_predict_sec"] = float(time.perf_counter() - t0)
        t0 = time.perf_counter()
        aug_probs = np.clip(aug_probs, 1e-12, 1.0)
        aug_probs = aug_probs / np.clip(aug_probs.sum(axis=1, keepdims=True), 1e-12, None)
        aug_expected = aug_probs @ score_values
        ans_len_bias = np.maximum(0.0, aug_expected - expected_score)
        length_bias = np.asarray(
            [float(ans_len_bias[int(start) : int(end)].mean()) for start, end in spans],
            dtype=np.float64,
        )
        timing["verbosity_postprocess_sec"] = float(time.perf_counter() - t0)
    else:
        timing["verbosity_build_examples_sec"] = 0.0
        timing["verbosity_predict_sec"] = 0.0
        timing["verbosity_postprocess_sec"] = 0.0

    position_bias = np.zeros((len(candidates),), dtype=np.float64)
    if float(bias_weight) > 0.0 and float(pairwise_position_bias_weight) > 0.0:
        t0 = time.perf_counter()
        ordered_pairs, reverse_indices, pair_spans = _build_pairwise_position_bias_examples(candidates)
        timing["position_build_pairwise_examples_sec"] = float(time.perf_counter() - t0)
        if ordered_pairs:
            t0 = time.perf_counter()
            ordered_probs = np.asarray(proxy.predict_proba_pairwise(ordered_pairs), dtype=np.float64)
            timing["position_pairwise_predict_sec"] = float(time.perf_counter() - t0)
            t0 = time.perf_counter()
            ordered_probs = np.clip(ordered_probs, 1e-12, 1.0)
            ordered_probs = ordered_probs / np.clip(ordered_probs.sum(axis=1, keepdims=True), 1e-12, None)
            reverse_probs = ordered_probs[np.asarray(reverse_indices, dtype=np.int64)]
            aligned_reverse = np.stack([reverse_probs[:, 1], reverse_probs[:, 0], reverse_probs[:, 2]], axis=1)
            pair_bias = np.sum(ordered_probs * (np.log(ordered_probs) - np.log(aligned_reverse)), axis=1)
            position_bias = np.asarray(
                [float(pair_bias[int(start) : int(end)].mean()) for start, end in pair_spans],
                dtype=np.float64,
            )
            timing["position_postprocess_sec"] = float(time.perf_counter() - t0)
        else:
            timing["position_pairwise_predict_sec"] = 0.0
            timing["position_postprocess_sec"] = 0.0
    else:
        timing["position_build_pairwise_examples_sec"] = 0.0
        timing["position_pairwise_predict_sec"] = 0.0
        timing["position_postprocess_sec"] = 0.0

    t0 = time.perf_counter()
    density = _density_from_embeddings(candidate_embeddings, k=max(1, int(density_k)))
    if n_selected > 0:
        sim = np.clip(embeddings[pool_slice] @ embeddings[:n_selected].T, -1.0, 1.0)
        diversity = np.clip(1.0 - np.max(sim, axis=1), 0.0, 2.0).astype(np.float64)
    else:
        diversity = np.zeros((len(candidates),), dtype=np.float64)
    timing["density_diversity_sec"] = float(time.perf_counter() - t0)

    t0 = time.perf_counter()
    signal_normalization = str(signal_normalization)
    if signal_normalization not in {"none", "intrinsic"}:
        raise ValueError("signal_normalization must be one of {'none','intrinsic'}")

    diversity_for_score = np.asarray(diversity, dtype=np.float64)
    uncertainty_for_score = np.asarray(uncertainty, dtype=np.float64)
    length_bias_for_score = np.asarray(length_bias, dtype=np.float64)
    position_bias_for_score = np.asarray(position_bias, dtype=np.float64)
    if signal_normalization == "intrinsic":
        score_span = max(float(score_max - score_min), 1.0)
        diversity_for_score = np.clip(diversity_for_score / 2.0, 0.0, 1.0)
        uncertainty_for_score = np.clip(uncertainty_for_score, 0.0, 1.0)
        length_bias_for_score = np.clip(length_bias_for_score / score_span, 0.0, 1.0)
        position_bias_for_score = 1.0 - np.exp(-np.clip(position_bias_for_score, 0.0, 60.0))
    position_bias_for_score = float(pairwise_position_bias_scale) * position_bias_for_score

    bias_denom = max(float(pointwise_length_bias_weight) + float(pairwise_position_bias_weight), 1e-12)
    bias_score = (
        float(pointwise_length_bias_weight) * length_bias_for_score
        + float(pairwise_position_bias_weight) * position_bias_for_score
    ) / bias_denom
    bias_score = np.maximum(bias_score, 0.0)
    timing["normalize_bias_sec"] = float(time.perf_counter() - t0)

    t0 = time.perf_counter()
    density_threshold = float(np.quantile(density, 0.10)) if int(density.size) > 0 else float("-inf")
    density_keep = density > density_threshold
    if not bool(np.any(density_keep)):
        density_keep = np.ones_like(density, dtype=bool)

    coverage = _score_bin_bonus_for_candidate_triples(
        candidates,
        queried_score_counts=queried_score_counts,
        score_min=int(score_min),
        score_max=int(score_max),
    ).astype(np.float64)

    total = float(diversity_weight) * np.asarray(diversity_for_score, dtype=np.float64)
    total = total + float(uncertainty_weight) * np.asarray(uncertainty_for_score, dtype=np.float64)
    total = total + float(bias_weight) * np.asarray(bias_score, dtype=np.float64)
    total = np.where(density_keep, total, -np.inf)
    timing["coverage_total_score_sec"] = float(time.perf_counter() - t0)
    timing["total_acquisition_sec"] = float(time.perf_counter() - t_func)

    scores = np.asarray(total, dtype=np.float32)
    finite_scores = scores[np.isfinite(scores)]
    diagnostics = {
        "pool_score_mean": float(finite_scores.mean()) if int(finite_scores.size) > 0 else float("nan"),
        "pool_score_std": float(finite_scores.std()) if int(finite_scores.size) > 0 else float("nan"),
        "pool_diversity_mean": float(np.mean(diversity)),
        "pool_diversity_score_mean": float(np.mean(diversity_for_score)),
        "pool_density_mean": float(np.mean(density)),
        "density_filter_quantile": 0.10,
        "density_filter_threshold": float(density_threshold),
        "density_filtered": int(np.size(density_keep) - np.count_nonzero(density_keep)),
        "pool_entropy_mean": float(np.mean(uncertainty)),
        "pool_uncertainty_mean": float(np.mean(uncertainty)),
        "pool_uncertainty_score_mean": float(np.mean(uncertainty_for_score)),
        "uncertainty_view": str(uncertainty_view),
        "pool_pointwise_length_bias_mean": float(np.mean(length_bias)),
        "pool_pointwise_length_bias_score_mean": float(np.mean(length_bias_for_score)),
        "pool_pairwise_position_bias_mean": float(np.mean(position_bias)),
        "pool_pairwise_position_bias_score_mean": float(np.mean(position_bias_for_score)),
        "pairwise_position_bias_scale": float(pairwise_position_bias_scale),
        "pool_bias_mean": float(np.mean(bias_score)),
        "pool_coverage_mean": float(np.mean(coverage)),
        "embedding_model": str(embedder.model_name_or_path),
        "embedding_device": str(embedder.device),
        "embedding_pooling": str(embedder.pooling),
        "diversity_view": str(diversity_view),
        "embedding_max_length": int(embedder.max_length),
        "embedding_batch_size": int(embedder.batch_size),
        "embedding_dim": int(candidate_embeddings.shape[1]) if int(candidate_embeddings.ndim) == 2 else 0,
        "verbosity_prefixes": list(DEFAULT_VERBOSITY_PREFIXES),
        "selected_verbosity_prefix_counts": {
            str(prefix): int(sum(1 for p in prefixes_by_candidate if str(p) == str(prefix)))
            for prefix in DEFAULT_VERBOSITY_PREFIXES
        },
        "weights": {
            "diversity": float(diversity_weight),
            "density": 0.0,
            "uncertainty": float(uncertainty_weight),
            "bias": float(bias_weight),
            "coverage": 0.0,
            "pointwise_length_bias": float(pointwise_length_bias_weight),
            "pairwise_position_bias": float(pairwise_position_bias_weight),
            "pairwise_position_pairs": "all_ordered",
            "pairwise_position_prompts_per_triple": 6,
            "pairwise_position_reverse_reuse": True,
            "pairwise_position_bias_scale": float(pairwise_position_bias_scale),
            "signal_normalization": str(signal_normalization),
            "uncertainty_view": str(uncertainty_view),
        },
        "ignored_weights_for_doc_consistency": {
            "density_weight_arg": float(density_weight),
            "coverage_weight_arg": float(coverage_weight),
        },
        "timing_sec": timing,
    }
    return scores, diagnostics


def _pick_candidate_triples_greedy(
    candidates: Sequence[CandidateTripleExample],
    *,
    k: int,
    rng: np.random.Generator,
    one_per_question: bool,
    scores: Optional[np.ndarray] = None,
    selected_group_ids: Optional[set[int]] = None,
) -> List[CandidateTripleExample]:
    pool = list(candidates)
    if not pool or int(k) <= 0:
        return []
    banned = set(selected_group_ids or set())
    picked: List[CandidateTripleExample] = []
    used_groups: set[int] = set()

    if scores is None:
        order = rng.permutation(len(pool)).tolist()
    else:
        scores_arr = np.asarray(scores, dtype=np.float32)
        if int(scores_arr.shape[0]) != len(pool):
            raise ValueError("candidate scores length must match candidate pool length")
        noise = rng.random(len(pool)) * 1e-8
        order = np.argsort(-(scores_arr + noise)).tolist()

    for idx in order:
        c = pool[int(idx)]
        gid = int(c.group_id)
        if bool(one_per_question) and (gid in banned or gid in used_groups):
            continue
        picked.append(c)
        used_groups.add(gid)
        if len(picked) >= int(k):
            break
    return picked


def _pointwise_proxy_acquisition_scores(
    *,
    proxy: base.LlamaSharedMultiTaskProxyModel,
    candidates: Sequence[CandidateTripleExample],
    score_min: int,
    score_max: int,
    fix_score_prefix_in_prompt: bool,
    queried_score_counts: Optional[np.ndarray],
    entropy_weight: float,
    score_std_weight: float,
    predicted_coverage_weight: float,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Score unlabeled triples using only pointwise proxy predictions."""
    pointwise_inputs, spans = _build_pointwise_examples_for_candidate_triples(
        candidates,
        score_min=int(score_min),
        score_max=int(score_max),
        fix_score_prefix_in_prompt=bool(fix_score_prefix_in_prompt),
    )
    if not pointwise_inputs:
        return np.zeros((0,), dtype=np.float32), {}

    probs = np.asarray(proxy.predict_proba_pointwise(pointwise_inputs), dtype=np.float64)
    num_scores = int(score_max - score_min + 1)
    if probs.shape != (len(pointwise_inputs), num_scores):
        raise RuntimeError(
            "pointwise proxy returned unexpected probability shape: "
            f"got {probs.shape}, expected {(len(pointwise_inputs), num_scores)}"
        )
    probs = np.clip(probs, 1e-12, 1.0)
    probs = probs / np.clip(probs.sum(axis=1, keepdims=True), 1e-12, None)

    entropy = -(probs * np.log(probs)).sum(axis=1) / max(np.log(float(num_scores)), 1e-12)
    score_values = np.arange(int(score_min), int(score_max) + 1, dtype=np.float64)
    expected = probs @ score_values
    variance = probs @ (score_values * score_values) - expected * expected
    score_std = np.sqrt(np.clip(variance, 0.0, None)) / max(float(score_max - score_min), 1.0)

    denom = float(entropy_weight) + float(score_std_weight)
    answer_utility = (
        float(entropy_weight) * entropy + float(score_std_weight) * score_std
    ) / max(denom, 1e-12)

    coverage = np.zeros((len(pointwise_inputs),), dtype=np.float64)
    if float(predicted_coverage_weight) > 0.0:
        counts = np.zeros((num_scores,), dtype=np.float64)
        if queried_score_counts is not None and int(np.asarray(queried_score_counts).size) == num_scores:
            counts = np.asarray(queried_score_counts, dtype=np.float64)
        rarity = 1.0 / np.sqrt(counts + 1.0)
        rarity = rarity / max(float(rarity.max()), 1e-12)
        coverage = probs @ rarity

    triple_scores: List[float] = []
    triple_entropy: List[float] = []
    triple_std: List[float] = []
    triple_coverage: List[float] = []
    for start, end in spans:
        sl = slice(int(start), int(end))
        # Mean rewards broad uncertainty; max retains a genuinely hard answer.
        core = 0.75 * float(answer_utility[sl].mean()) + 0.25 * float(answer_utility[sl].max())
        cov = float(coverage[sl].mean())
        triple_scores.append(core + float(predicted_coverage_weight) * cov)
        triple_entropy.append(float(entropy[sl].mean()))
        triple_std.append(float(score_std[sl].mean()))
        triple_coverage.append(cov)

    score_arr = np.asarray(triple_scores, dtype=np.float32)
    diagnostics = {
        "pool_score_mean": float(score_arr.mean()),
        "pool_score_std": float(score_arr.std()),
        "pool_entropy_mean": float(np.mean(triple_entropy)),
        "pool_predicted_score_std_mean": float(np.mean(triple_std)),
        "pool_coverage_mean": float(np.mean(triple_coverage)),
    }
    return score_arr, diagnostics


def _train_pointwise_proxy_on_triples(
    *,
    proxy: base.LlamaSharedMultiTaskProxyModel,
    candidates: Sequence[CandidateTripleExample],
    score_min: int,
    score_max: int,
    fix_score_prefix_in_prompt: bool,
    epochs: int = 1,
) -> None:
    inputs, _ = _build_pointwise_examples_for_candidate_triples(
        candidates,
        score_min=int(score_min),
        score_max=int(score_max),
        fix_score_prefix_in_prompt=bool(fix_score_prefix_in_prompt),
    )
    labels = [int(x.label) for x in inputs]
    for _ in range(max(1, int(epochs))):
        proxy.train_on_batch_pointwise(inputs, labels)


class _FrozenLlamaFeatureProxy:
    """Read base-Llama features while the uncertainty proxy LoRA keeps learning."""

    def __init__(self, proxy: base.LlamaSharedMultiTaskProxyModel) -> None:
        self._proxy = proxy
        self.device = proxy.device
        self.hidden_dim = proxy.hidden_dim

    def extract_features_tensor(self, inputs: Sequence[Any]) -> torch.Tensor:
        disable_adapter = getattr(self._proxy.model, "disable_adapter", None)
        adapter_ctx = disable_adapter() if callable(disable_adapter) else contextlib.nullcontext()
        with adapter_ctx, torch.no_grad():
            return self._proxy.extract_features_tensor(inputs).detach()


def _select_candidate_triples_with_selector(
    *,
    candidates: Sequence[CandidateTripleExample],
    cfg: RunConfig,
    llama_path: str,
    output_dir: Path,
) -> Tuple[List[SelectedQuestionTriple], List[Dict[str, Any]], Dict[str, Any]]:
    candidate_list = list(candidates)
    if not candidate_list:
        raise RuntimeError("candidate_triple_selector received an empty candidate pool")

    max_triples = int(cfg.budget_units) // 3 if int(cfg.budget_units) > 0 else len({int(c.group_id) for c in candidate_list})
    max_triples = max(1, min(int(max_triples), len(candidate_list)))
    rng = np.random.default_rng(int(cfg.seed) + 101)
    one_per_question = bool(cfg.candidate_selector_one_per_question)

    selected_group_ids: set[int] = set()
    selected_candidate_ids: set[int] = set()
    selected: List[CandidateTripleExample] = []
    rows: List[Dict[str, Any]] = []
    queried_score_counts: Optional[np.ndarray] = None

    def record(
        c: CandidateTripleExample,
        *,
        stage: str,
        rank_score: float,
        acquisition_source: str = "selector",
    ) -> None:
        rows.append(
            {
                "stage": str(stage),
                "candidate_triple_id": int(c.id),
                "group_id": int(c.group_id),
                "question_id": int(c.question_id),
                "source_id": int(c.source_id),
                "dataset": str(c.dataset),
                "model_a": str(c.model_a),
                "model_b": str(c.model_b),
                "model_c": str(c.model_c),
                "queried": True,
                "score_a": int(c.score_a),
                "score_b": int(c.score_b),
                "score_c": int(c.score_c),
                "score_range": int(c.score_range),
                "score_gap_sum": int(c.score_gap_sum),
                "ranking": str(c.ranking),
                "label": int(c.label),
                "rank_score": float(rank_score),
                "acquisition_source": str(acquisition_source),
            }
        )

    if str(cfg.candidate_selector_kind) == "random":
        if bool(getattr(cfg, "reuse_selection_proxy_for_stage1", False)):
            raise ValueError("--reuse-selection-proxy-for-stage1 requires pointwise_proxy selection")
        picked = _pick_candidate_triples_greedy(
            candidate_list,
            k=int(max_triples),
            rng=rng,
            one_per_question=one_per_question,
        )
        for c in picked:
            selected.append(c)
            selected_candidate_ids.add(int(c.id))
            selected_group_ids.add(int(c.group_id))
            record(c, stage="random", rank_score=float("nan"))
        info = {
            "mode": "candidate_triple_selector",
            "selector_kind": "random",
            "candidate_triples": int(len(candidate_list)),
            "selected_triples": int(len(selected)),
            "selected_answers": int(len(selected) * 3),
            "one_per_question": bool(one_per_question),
            "budget_units": int(cfg.budget_units),
            "effective_budget_units": int(len(selected) * 3),
        }
        return [c.selected_triple for c in selected], rows, info

    selector_kind = str(cfg.candidate_selector_kind)
    if selector_kind not in {"bert", "pointwise_proxy", "bias_trap_pointwise", "shared_llama", "shared_llama_two_stage"}:
        raise ValueError(f"unknown candidate-selector-kind: {cfg.candidate_selector_kind}")
    selector_proxy_mode = str(
        getattr(
            cfg,
            "candidate_selector_proxy_mode",
            getattr(cfg, "llama_multitask_mode", "classifier_heads"),
        )
    )
    if selector_proxy_mode == "shared_head":
        selector_proxy_mode = "lm_head"
    if selector_proxy_mode not in {"classifier_heads", "lm_head"}:
        raise ValueError(f"unknown candidate-selector-proxy-mode: {selector_proxy_mode}")
    return_selection_proxy = bool(getattr(cfg, "reuse_selection_proxy_for_stage1", False))
    if return_selection_proxy:
        if selector_kind not in {"pointwise_proxy", "bias_trap_pointwise"}:
            raise ValueError(
                "--reuse-selection-proxy-for-stage1 requires pointwise_proxy or bias_trap_pointwise selection"
            )
        if selector_proxy_mode != "lm_head":
            raise ValueError("--reuse-selection-proxy-for-stage1 requires candidate_selector_proxy_mode='lm_head'")

    target_proxy: Optional[base.LlamaSharedMultiTaskProxyModel] = None
    if str(cfg.candidate_selector_target_task) == "pointwise":
        target_proxy = base.LlamaSharedMultiTaskProxyModel(
            model_path=str(llama_path),
            pointwise_num_labels=int(cfg.score_max - cfg.score_min + 1),
            pairwise_num_labels=3 if selector_proxy_mode == "lm_head" else int(len(RANKING_LABELS)),
            multitask_mode=str(selector_proxy_mode),
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

    bert_selector: Optional[base.BertBinarySelector] = None
    if selector_kind in {"bert", "shared_llama_two_stage"}:
        bert_selector = base.BertBinarySelector(
            model_name=str(cfg.candidate_bert_selector_model),
            max_length=int(cfg.candidate_bert_selector_max_length),
            head_hidden_dim=512,
            head_dropout=0.1,
            lr=1e-3,
            weight_decay=0.0,
            freeze_bert=bool(cfg.candidate_bert_selector_freeze),
            unfreeze_last_n_layers=int(cfg.candidate_bert_selector_unfreeze_last_n_layers),
        )
    selector: Any = None
    if selector_kind in {"pointwise_proxy", "bias_trap_pointwise"}:
        if target_proxy is None:
            raise RuntimeError(f"{selector_kind} selector requires a pointwise target proxy")
    elif selector_kind in {"shared_llama", "shared_llama_two_stage"}:
        if target_proxy is None:
            raise RuntimeError("shared-Llama selector requires a pointwise target proxy")
        selector = base.SharedLlamaSelectorV2(
            proxy_model=_FrozenLlamaFeatureProxy(target_proxy),
            head_hidden_dim=512,
            head_dropout=0.1,
            lr=1e-3,
            weight_decay=0.0,
            batch_size=max(1, int(cfg.candidate_selector_batch_size)),
            inference_batch_size=4,
            buffer_maxlen=int(cfg.candidate_selector_buffer_maxlen),
        )

    bias_trap_embedder: Optional[TransformerTextEmbedder] = None
    bias_trap_embedding_cache: Dict[int, np.ndarray] = {}
    bias_trap_prefix_embeddings: Optional[np.ndarray] = None
    if selector_kind == "bias_trap_pointwise":
        bias_trap_embedder = TransformerTextEmbedder(
            str(getattr(cfg, "candidate_selector_embedding_model", DEFAULT_SELECTOR_EMBEDDING_MODEL)),
            max_length=int(getattr(cfg, "candidate_selector_embedding_max_length", DEFAULT_SELECTOR_EMBEDDING_MAX_LENGTH)),
            batch_size=int(getattr(cfg, "candidate_selector_embedding_batch_size", 64)),
            device=str(getattr(cfg, "candidate_selector_embedding_device", "auto")),
            pooling=str(getattr(cfg, "candidate_selector_embedding_pooling", "cls")),
        )
        bias_trap_prefix_embeddings = bias_trap_embedder.encode(DEFAULT_VERBOSITY_PREFIXES)
    else:
        if bert_selector is None:
            raise RuntimeError("BERT selector was not initialized")
        selector = bert_selector

    init_k = min(int(cfg.candidate_selector_init_triples), int(max_triples), len(candidate_list))
    init_batch = _pick_candidate_triples_greedy(
        candidate_list,
        k=max(1, init_k),
        rng=rng,
        one_per_question=one_per_question,
    )
    if not init_batch:
        raise RuntimeError("candidate_triple_selector failed to build an initial batch")

    if selector_kind in {"pointwise_proxy", "bias_trap_pointwise"}:
        if target_proxy is None:
            raise RuntimeError(f"{selector_kind} selector requires a pointwise target proxy")
        _train_pointwise_proxy_on_triples(
            proxy=target_proxy,
            candidates=init_batch,
            score_min=int(cfg.score_min),
            score_max=int(cfg.score_max),
            fix_score_prefix_in_prompt=bool(cfg.fix_score_prefix_in_prompt),
            epochs=int(getattr(cfg, "candidate_selector_proxy_warmup_epochs", 1)),
        )
        init_targets = np.zeros((len(init_batch),), dtype=np.float32)
    elif str(cfg.candidate_selector_target_task) == "pointwise":
        if target_proxy is None:
            raise RuntimeError("pointwise candidate selector target requires a proxy")
        init_targets = _candidate_triple_targets_pointwise(
            proxy=target_proxy,
            candidates=init_batch,
            score_min=int(cfg.score_min),
            score_max=int(cfg.score_max),
            fix_score_prefix_in_prompt=bool(cfg.fix_score_prefix_in_prompt),
            queried_score_counts=queried_score_counts,
            score_range_weight=float(cfg.candidate_selector_score_range_weight),
            gap_sum_weight=float(cfg.candidate_selector_gap_sum_weight),
            score_bin_weight=float(cfg.candidate_selector_score_bin_weight),
            uncertainty_weight=float(cfg.candidate_selector_uncertainty_weight),
            pairwise_uncertainty_weight=float(cfg.candidate_selector_pairwise_uncertainty_weight),
            listwise_uncertainty_weight=float(cfg.candidate_selector_listwise_uncertainty_weight),
            kl_weight=float(cfg.candidate_selector_kl_weight),
        )
    else:
        init_targets = _candidate_triple_targets(
            init_batch,
            score_min=int(cfg.score_min),
            score_max=int(cfg.score_max),
            score_range_weight=float(cfg.candidate_selector_score_range_weight),
            gap_sum_weight=float(cfg.candidate_selector_gap_sum_weight),
        )
    if selector_kind not in {"pointwise_proxy", "bias_trap_pointwise"}:
        selector.update(
            init_batch,
            init_targets,
            epochs=max(1, int(cfg.candidate_selector_epochs)),
            batch_size=max(1, int(cfg.candidate_selector_batch_size)),
        )
    if selector_kind == "shared_llama_two_stage" and bert_selector is not None:
        bert_selector.update(
            init_batch,
            init_targets,
            epochs=max(1, int(cfg.candidate_selector_epochs)),
            batch_size=max(1, int(cfg.candidate_selector_batch_size)),
        )
    for c in init_batch:
        selected.append(c)
        selected_candidate_ids.add(int(c.id))
        selected_group_ids.add(int(c.group_id))
        record(c, stage="init", rank_score=float("nan"), acquisition_source="random_init")
    if str(cfg.candidate_selector_target_task) == "pointwise":
        queried_score_counts = _build_score_bin_counts_from_candidate_triples(
            selected,
            score_min=int(cfg.score_min),
            score_max=int(cfg.score_max),
        )

    round_idx = 0
    round_diagnostics: List[Dict[str, Any]] = []
    t0 = time.time()
    while len(selected) < int(max_triples):
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

        if selector_kind == "shared_llama_two_stage":
            if bert_selector is None:
                raise RuntimeError("two-stage selector requires a BERT prefilter")
            bert_scores = bert_selector.score(pool).astype(np.float32)
            rerank_k = min(max(1, int(cfg.candidate_selector_llama_rerank_candidates)), len(pool))
            rerank_idx = np.argsort(-bert_scores)[:rerank_k].tolist()
            pool = [pool[int(i)] for i in rerank_idx]
        need = min(int(cfg.candidate_selector_batch_size), int(max_triples) - len(selected))
        acquisition_source_by_id: Dict[int, str] = {}
        acquisition_diag: Dict[str, Any] = {}
        if selector_kind == "pointwise_proxy":
            if target_proxy is None:
                raise RuntimeError("pointwise_proxy selector requires a pointwise target proxy")
            selector_scores, acquisition_diag = _pointwise_proxy_acquisition_scores(
                proxy=target_proxy,
                candidates=pool,
                score_min=int(cfg.score_min),
                score_max=int(cfg.score_max),
                fix_score_prefix_in_prompt=bool(cfg.fix_score_prefix_in_prompt),
                queried_score_counts=queried_score_counts,
                entropy_weight=float(getattr(cfg, "candidate_selector_entropy_weight", 0.5)),
                score_std_weight=float(getattr(cfg, "candidate_selector_score_std_weight", 0.5)),
                predicted_coverage_weight=float(
                    getattr(cfg, "candidate_selector_predicted_coverage_weight", 0.2)
                ),
            )
            exploration_ratio = float(getattr(cfg, "candidate_selector_exploration_ratio", 0.1))
            explore_k = min(int(need), int(round(float(need) * exploration_ratio)))
            exploit_k = int(need) - int(explore_k)
            exploit = _pick_candidate_triples_greedy(
                pool,
                k=int(exploit_k),
                rng=rng,
                one_per_question=one_per_question,
                scores=selector_scores,
                selected_group_ids=selected_group_ids,
            )
            exploit_ids = {int(c.id) for c in exploit}
            explore_pool = [c for c in pool if int(c.id) not in exploit_ids]
            explore = _pick_candidate_triples_greedy(
                explore_pool,
                k=int(explore_k),
                rng=rng,
                one_per_question=one_per_question,
                selected_group_ids=selected_group_ids | {int(c.group_id) for c in exploit},
            )
            picked = exploit + explore
            for c in exploit:
                acquisition_source_by_id[int(c.id)] = "pointwise_proxy"
            for c in explore:
                acquisition_source_by_id[int(c.id)] = "random_exploration"
            acquisition_diag.update(
                {
                    "round": int(round_idx),
                    "pool_size": int(len(pool)),
                    "exploit_selected": int(len(exploit)),
                    "explore_selected": int(len(explore)),
                    "selected_score_mean": float(
                        np.mean([selector_scores[i] for i, c in enumerate(pool) if int(c.id) in exploit_ids])
                    ) if exploit else float("nan"),
                }
            )
            round_diagnostics.append(acquisition_diag)
        elif selector_kind == "bias_trap_pointwise":
            if target_proxy is None:
                raise RuntimeError("bias_trap_pointwise selector requires a pointwise target proxy")
            selector_scores, acquisition_diag = _bias_trap_pointwise_acquisition_scores(
                proxy=target_proxy,
                candidates=pool,
                selected=selected,
                score_min=int(cfg.score_min),
                score_max=int(cfg.score_max),
                fix_score_prefix_in_prompt=bool(cfg.fix_score_prefix_in_prompt),
                queried_score_counts=queried_score_counts,
                diversity_weight=float(getattr(cfg, "candidate_selector_diversity_weight", 0.25)),
                density_weight=float(getattr(cfg, "candidate_selector_density_weight", 0.15)),
                uncertainty_weight=float(getattr(cfg, "candidate_selector_uncertainty_weight", 0.25)),
                bias_weight=float(getattr(cfg, "candidate_selector_bias_weight", 0.25)),
                coverage_weight=float(getattr(cfg, "candidate_selector_coverage_weight", 0.10)),
                pointwise_length_bias_weight=float(
                    getattr(cfg, "candidate_selector_pointwise_length_bias_weight", 0.5)
                ),
                pairwise_position_bias_weight=float(
                    getattr(cfg, "candidate_selector_pairwise_position_bias_weight", 0.5)
                ),
                pairwise_position_pairs=int(getattr(cfg, "candidate_selector_pairwise_position_pairs", 1)),
                pairwise_position_bias_scale=float(
                    getattr(cfg, "candidate_selector_pairwise_position_bias_scale", 1.0)
                ),
                signal_normalization=str(getattr(cfg, "candidate_selector_signal_normalization", "none")),
                uncertainty_view=str(getattr(cfg, "candidate_selector_uncertainty_view", "pointwise")),
                diversity_view=str(getattr(cfg, "candidate_selector_diversity_view", "pointwise")),
                length_aug_suffix=str(
                    getattr(
                        cfg,
                        "candidate_selector_length_aug_suffix",
                        "Additional context: This repeats the same answer without adding new useful information.",
                    )
                ),
                density_k=int(getattr(cfg, "candidate_selector_density_k", 10)),
                embedder=bias_trap_embedder,
                embedding_cache=bias_trap_embedding_cache,
                prefix_embeddings=bias_trap_prefix_embeddings,
            )
            exploration_ratio = float(getattr(cfg, "candidate_selector_exploration_ratio", 0.0))
            explore_k = min(int(need), int(round(float(need) * exploration_ratio)))
            exploit_k = int(need) - int(explore_k)
            exploit = _pick_candidate_triples_greedy(
                pool,
                k=int(exploit_k),
                rng=rng,
                one_per_question=one_per_question,
                scores=selector_scores,
                selected_group_ids=selected_group_ids,
            )
            exploit_ids = {int(c.id) for c in exploit}
            explore_pool = [c for c in pool if int(c.id) not in exploit_ids]
            explore = _pick_candidate_triples_greedy(
                explore_pool,
                k=int(explore_k),
                rng=rng,
                one_per_question=one_per_question,
                selected_group_ids=selected_group_ids | {int(c.group_id) for c in exploit},
            )
            picked = exploit + explore
            for c in exploit:
                acquisition_source_by_id[int(c.id)] = "bias_trap_pointwise"
            for c in explore:
                acquisition_source_by_id[int(c.id)] = "random_exploration"
            acquisition_diag.update(
                {
                    "round": int(round_idx),
                    "pool_size": int(len(pool)),
                    "exploit_selected": int(len(exploit)),
                    "explore_selected": int(len(explore)),
                    "selected_score_mean": float(
                        np.mean([selector_scores[i] for i, c in enumerate(pool) if int(c.id) in exploit_ids])
                    ) if exploit else float("nan"),
                }
            )
            round_diagnostics.append(acquisition_diag)
        else:
            selector_scores = selector.score(pool).astype(np.float32)
            picked = _pick_candidate_triples_greedy(
                pool,
                k=int(need),
                rng=rng,
                one_per_question=one_per_question,
                scores=selector_scores,
                selected_group_ids=selected_group_ids,
            )
        if not picked:
            break

        score_by_id = {int(c.id): float(s) for c, s in zip(pool, selector_scores.tolist())}
        if selector_kind in {"pointwise_proxy", "bias_trap_pointwise"}:
            if target_proxy is None:
                raise RuntimeError(f"{selector_kind} selector requires a pointwise target proxy")
            _train_pointwise_proxy_on_triples(
                proxy=target_proxy,
                candidates=picked,
                score_min=int(cfg.score_min),
                score_max=int(cfg.score_max),
                fix_score_prefix_in_prompt=bool(cfg.fix_score_prefix_in_prompt),
                epochs=int(getattr(cfg, "candidate_selector_proxy_update_epochs", 1)),
            )
            targets = np.zeros((len(picked),), dtype=np.float32)
        elif str(cfg.candidate_selector_target_task) == "pointwise":
            if target_proxy is None:
                raise RuntimeError("pointwise candidate selector target requires a proxy")
            targets = _candidate_triple_targets_pointwise(
                proxy=target_proxy,
                candidates=picked,
                score_min=int(cfg.score_min),
                score_max=int(cfg.score_max),
                fix_score_prefix_in_prompt=bool(cfg.fix_score_prefix_in_prompt),
                queried_score_counts=queried_score_counts,
                score_range_weight=float(cfg.candidate_selector_score_range_weight),
                gap_sum_weight=float(cfg.candidate_selector_gap_sum_weight),
                score_bin_weight=float(cfg.candidate_selector_score_bin_weight),
                uncertainty_weight=float(cfg.candidate_selector_uncertainty_weight),
                pairwise_uncertainty_weight=float(cfg.candidate_selector_pairwise_uncertainty_weight),
                listwise_uncertainty_weight=float(cfg.candidate_selector_listwise_uncertainty_weight),
                kl_weight=float(cfg.candidate_selector_kl_weight),
            )
        else:
            targets = _candidate_triple_targets(
                picked,
                score_min=int(cfg.score_min),
                score_max=int(cfg.score_max),
                score_range_weight=float(cfg.candidate_selector_score_range_weight),
                gap_sum_weight=float(cfg.candidate_selector_gap_sum_weight),
            )
        if selector_kind not in {"pointwise_proxy", "bias_trap_pointwise"}:
            selector.update(
                picked,
                targets,
                epochs=max(1, int(cfg.candidate_selector_epochs)),
                batch_size=max(1, int(cfg.candidate_selector_batch_size)),
            )
        if selector_kind == "shared_llama_two_stage" and bert_selector is not None:
            bert_selector.update(
                picked,
                targets,
                epochs=max(1, int(cfg.candidate_selector_epochs)),
                batch_size=max(1, int(cfg.candidate_selector_batch_size)),
            )

        for c in picked:
            selected.append(c)
            selected_candidate_ids.add(int(c.id))
            selected_group_ids.add(int(c.group_id))
            record(
                c,
                stage=f"round_{round_idx}",
                rank_score=float(score_by_id.get(int(c.id), float("nan"))),
                acquisition_source=acquisition_source_by_id.get(int(c.id), "selector"),
            )
        if str(cfg.candidate_selector_target_task) == "pointwise":
            queried_score_counts = _build_score_bin_counts_from_candidate_triples(
                selected,
                score_min=int(cfg.score_min),
                score_max=int(cfg.score_max),
            )

        print(
            f"[candidate-triple-selector] round={round_idx} selected={len(selected)}/{max_triples} "
            f"pool={len(remaining)} elapsed={time.time() - t0:.0f}s",
            flush=True,
        )

    info = {
        "mode": "candidate_triple_selector",
        "selector_kind": str(cfg.candidate_selector_kind),
        "candidate_triples": int(len(candidate_list)),
        "selected_triples": int(len(selected)),
        "selected_answers": int(len(selected) * 3),
        "one_per_question": bool(one_per_question),
        "budget_units": int(cfg.budget_units),
        "effective_budget_units": int(len(selected) * 3),
        "init_triples": int(len(init_batch)),
        "batch_size": int(cfg.candidate_selector_batch_size),
        "selector_epochs": int(cfg.candidate_selector_epochs),
        "max_score_candidates": int(cfg.candidate_selector_max_score_candidates),
        "llama_rerank_candidates": int(cfg.candidate_selector_llama_rerank_candidates),
        "selector_buffer_maxlen": int(cfg.candidate_selector_buffer_maxlen),
        "frozen_llama_features": selector_kind in {"shared_llama", "shared_llama_two_stage"},
        "target_task": str(cfg.candidate_selector_target_task),
        "selection_proxy_mode": str(selector_proxy_mode),
        "selection_proxy_returned_for_stage1": bool(return_selection_proxy),
        "score_range_weight": float(cfg.candidate_selector_score_range_weight),
        "gap_sum_weight": float(cfg.candidate_selector_gap_sum_weight),
        "uncertainty_weight": float(cfg.candidate_selector_uncertainty_weight),
        "pairwise_uncertainty_weight": float(cfg.candidate_selector_pairwise_uncertainty_weight),
        "listwise_uncertainty_weight": float(cfg.candidate_selector_listwise_uncertainty_weight),
        "kl_weight": float(cfg.candidate_selector_kl_weight),
        "score_bin_weight": float(cfg.candidate_selector_score_bin_weight),
        "diversity_weight": float(getattr(cfg, "candidate_selector_diversity_weight", 0.0)),
        "density_weight": float(getattr(cfg, "candidate_selector_density_weight", 0.0)),
        "bias_weight": float(getattr(cfg, "candidate_selector_bias_weight", 0.0)),
        "coverage_weight": float(getattr(cfg, "candidate_selector_coverage_weight", 0.0)),
        "pointwise_length_bias_weight": float(
            getattr(cfg, "candidate_selector_pointwise_length_bias_weight", 0.0)
        ),
        "pairwise_position_bias_weight": float(
            getattr(cfg, "candidate_selector_pairwise_position_bias_weight", 0.0)
        ),
        "pairwise_position_pairs": int(getattr(cfg, "candidate_selector_pairwise_position_pairs", 1)),
        "pairwise_position_bias_scale": float(
            getattr(cfg, "candidate_selector_pairwise_position_bias_scale", 1.0)
        ),
        "signal_normalization": str(getattr(cfg, "candidate_selector_signal_normalization", "none")),
        "uncertainty_view": str(getattr(cfg, "candidate_selector_uncertainty_view", "pointwise")),
        "length_aug_suffix": str(getattr(cfg, "candidate_selector_length_aug_suffix", "")),
        "embedding_model": str(getattr(cfg, "candidate_selector_embedding_model", DEFAULT_SELECTOR_EMBEDDING_MODEL)),
        "embedding_max_length": int(getattr(cfg, "candidate_selector_embedding_max_length", DEFAULT_SELECTOR_EMBEDDING_MAX_LENGTH)),
        "embedding_batch_size": int(getattr(cfg, "candidate_selector_embedding_batch_size", 64)),
        "embedding_device": str(getattr(cfg, "candidate_selector_embedding_device", "auto")),
        "embedding_pooling": str(getattr(cfg, "candidate_selector_embedding_pooling", "cls")),
        "diversity_view": str(getattr(cfg, "candidate_selector_diversity_view", "pointwise")),
        "bias_trap_effective_formula": (
            "diversity_weight * D + uncertainty_weight * normalized_entropy + bias_weight * bias; "
            "density filters the lowest 10%; coverage is ignored for bias_trap_pointwise"
        ),
        "verbosity_prefixes": list(DEFAULT_VERBOSITY_PREFIXES),
        "density_k": int(getattr(cfg, "candidate_selector_density_k", 10)),
        "exploration_ratio": float(getattr(cfg, "candidate_selector_exploration_ratio", 0.0)),
        "entropy_weight": float(getattr(cfg, "candidate_selector_entropy_weight", 0.0)),
        "score_std_weight": float(getattr(cfg, "candidate_selector_score_std_weight", 0.0)),
        "predicted_coverage_weight": float(
            getattr(cfg, "candidate_selector_predicted_coverage_weight", 0.0)
        ),
        "proxy_warmup_epochs": int(getattr(cfg, "candidate_selector_proxy_warmup_epochs", 1)),
        "proxy_update_epochs": int(getattr(cfg, "candidate_selector_proxy_update_epochs", 1)),
        "round_diagnostics": round_diagnostics,
        "bert_selector_model": str(cfg.candidate_bert_selector_model),
        "bert_selector_max_length": int(cfg.candidate_bert_selector_max_length),
        "bert_selector_freeze": bool(cfg.candidate_bert_selector_freeze),
        "bert_selector_unfreeze_last_n_layers": int(cfg.candidate_bert_selector_unfreeze_last_n_layers),
        "elapsed_sec": float(time.time() - t0),
    }

    kept_target_proxy = target_proxy if return_selection_proxy else None
    separate_bert_selector = bert_selector is not None and bert_selector is not selector
    if selector is not None:
        del selector
    if bias_trap_embedder is not None:
        del bias_trap_embedder
    if separate_bert_selector:
        del bert_selector
    if target_proxy is not None and kept_target_proxy is None:
        del target_proxy
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if return_selection_proxy:
        return [c.selected_triple for c in selected], rows, info, kept_target_proxy
    return [c.selected_triple for c in selected], rows, info

def _build_pointwise_examples_from_triples(
    selected_triples: Sequence[SelectedQuestionTriple],
    *,
    score_min: int,
    score_max: int,
    fix_score_prefix_in_prompt: bool,
) -> Tuple[List[base.PointwiseScoredExample], List[Dict[str, Any]], Dict[str, Any]]:
    examples: List[base.PointwiseScoredExample] = []
    rows: List[Dict[str, Any]] = []

    for p in selected_triples:
        for position, ans in (("A", p.answer_a), ("B", p.answer_b), ("C", p.answer_c)):
            label = base.score_to_class(int(ans.score), score_min=int(score_min), score_max=int(score_max))
            prompt = base.build_judge_prompt(
                system_prompt=base.JUDGE_SYSTEM_PROMPT_SCORE_ONLY,
                instruction=str(p.instruction),
                input_text=str(p.input_text),
                candidate_output=str(ans.output),
                include_gold_score=False,
                fix_score_prefix=bool(fix_score_prefix_in_prompt),
            )
            row_id = len(examples) + 1
            examples.append(
                base.PointwiseScoredExample(
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
            rows.append(
                {
                    "row_id": int(row_id),
                    "question_id": int(p.question_id),
                    "source_id": int(p.source_id),
                    "dataset": str(p.dataset),
                    "position": str(position),
                    "model": str(ans.model),
                    "score": int(ans.score),
                    "label": int(label),
                }
            )

    stats = {
        "input_question_triples": int(len(selected_triples)),
        "generated_pointwise_examples": int(len(examples)),
        "answers_per_question": 3,
    }
    return examples, rows, stats


def _append_listwise_example(
    *,
    examples: List[ListwiseExample],
    rows: List[Dict[str, Any]],
    stats: Dict[str, Any],
    group_id: int,
    source_id: int,
    dataset: str,
    instruction: str,
    input_text: str,
    answer_a: base.AnswerWithScore,
    answer_b: base.AnswerWithScore,
    answer_c: base.AnswerWithScore,
    order_augmented: bool,
) -> None:
    ranking = _ranking_from_scores(int(answer_a.score), int(answer_b.score), int(answer_c.score))
    label = _label_from_ranking(ranking)
    prompt = _build_listwise_prompt(
        system_prompt=LISTWISE_SYSTEM_PROMPT,
        instruction=str(instruction),
        input_text=str(input_text),
        assistant_a_output=str(answer_a.output),
        assistant_b_output=str(answer_b.output),
        assistant_c_output=str(answer_c.output),
    )
    ex_id = len(examples) + 1
    examples.append(
        ListwiseExample(
            id=int(ex_id),
            dataset=str(dataset),
            group_id=int(group_id),
            source_id=int(source_id),
            model_a=str(answer_a.model),
            model_b=str(answer_b.model),
            model_c=str(answer_c.model),
            prompt=prompt,
            ranking=str(ranking),
            label=int(label),
        )
    )
    rows.append(
        {
            "id": int(ex_id),
            "group_id": int(group_id),
            "source_id": int(source_id),
            "dataset": str(dataset),
            "modelA": str(answer_a.model),
            "outputA": str(answer_a.output),
            "scoreA": int(answer_a.score),
            "modelB": str(answer_b.model),
            "outputB": str(answer_b.output),
            "scoreB": int(answer_b.score),
            "modelC": str(answer_c.model),
            "outputC": str(answer_c.output),
            "scoreC": int(answer_c.score),
            "ranking": str(ranking),
            "raw_ranking": f"Ranking:[{ranking}]",
            "label": int(label),
            "order_augmented": bool(order_augmented),
        }
    )
    stats["label_counts"][ranking] = int(stats["label_counts"].get(ranking, 0) + 1)
    if bool(order_augmented):
        stats["order_augmented_examples"] = int(stats.get("order_augmented_examples", 0) + 1)


def _build_listwise_examples_from_triples(
    selected_triples: Sequence[SelectedQuestionTriple],
    *,
    order_augmentation: bool,
) -> Tuple[List[ListwiseExample], List[Dict[str, Any]], Dict[str, Any]]:
    examples: List[ListwiseExample] = []
    rows: List[Dict[str, Any]] = []
    stats: Dict[str, Any] = {
        "input_question_triples": int(len(selected_triples)),
        "generated_listwise_examples": 0,
        "order_augmentation": bool(order_augmentation),
        "order_augmented_examples": 0,
        "num_labels": int(len(RANKING_LABELS)),
        "label_counts": {r: 0 for r in RANKING_LABELS},
    }

    for p in selected_triples:
        answers = [p.answer_a, p.answer_b, p.answer_c]
        if bool(order_augmentation):
            permutations = list(itertools.permutations(answers, 3))
        else:
            permutations = [(p.answer_a, p.answer_b, p.answer_c)]

        for perm_i, (a, b, c) in enumerate(permutations):
            _append_listwise_example(
                examples=examples,
                rows=rows,
                stats=stats,
                group_id=int(p.question_id),
                source_id=int(p.source_id),
                dataset=str(p.dataset),
                instruction=str(p.instruction),
                input_text=str(p.input_text),
                answer_a=a,
                answer_b=b,
                answer_c=c,
                order_augmented=bool(order_augmentation and perm_i > 0),
            )

    stats["generated_listwise_examples"] = int(len(examples))
    return examples, rows, stats


def _load_listwise_eval_dataset(path: str) -> Tuple[List[ListwiseExample], List[Dict[str, Any]], Dict[str, Any]]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("listwise eval dataset JSON must be a list")

    examples: List[ListwiseExample] = []
    rows: List[Dict[str, Any]] = []
    stats: Dict[str, Any] = {
        "dataset_path": str(path),
        "input_records": int(len(raw)),
        "examples": 0,
        "skipped_missing_ranking": 0,
        "skipped_missing_output": 0,
        "num_labels": int(len(RANKING_LABELS)),
        "label_counts": {r: 0 for r in RANKING_LABELS},
    }

    for rec_i, rec in enumerate(raw):
        if not isinstance(rec, dict):
            continue

        instruction = str(_first_nonempty(rec, ("Instruction", "instruction", "question"), ""))
        input_text = str(_first_nonempty(rec, ("input", "input_text", "context"), ""))
        output_a = str(_first_nonempty(rec, ("outputA", "output_a", "assistant_a", "responseA"), ""))
        output_b = str(_first_nonempty(rec, ("outputB", "output_b", "assistant_b", "responseB"), ""))
        output_c = str(_first_nonempty(rec, ("outputC", "output_c", "assistant_c", "responseC"), ""))
        if not output_a.strip() or not output_b.strip() or not output_c.strip():
            stats["skipped_missing_output"] += 1
            continue

        ranking = _normalize_ranking_text(
            _first_nonempty(rec, ("ranking", "listwise_ranking", "raw_ranking", "label_ranking"), "")
        )
        if not ranking:
            score_a = _safe_score(rec, ("scoreA", "score_a"), score_min=1, score_max=10)
            score_b = _safe_score(rec, ("scoreB", "score_b"), score_min=1, score_max=10)
            score_c = _safe_score(rec, ("scoreC", "score_c"), score_min=1, score_max=10)
            if score_a is not None and score_b is not None and score_c is not None:
                ranking = _ranking_from_scores(score_a, score_b, score_c)
        if ranking not in RANKING_TO_LABEL:
            stats["skipped_missing_ranking"] += 1
            continue

        qid = base._safe_int(rec.get("id", rec.get("question_id", rec_i + 1)), default=rec_i + 1)
        source_id = base._safe_int(rec.get("source_id", qid), default=qid)
        dataset = str(rec.get("dataset", "listwise_eval"))
        model_a = str(_first_nonempty(rec, ("modelA", "model_a"), "A"))
        model_b = str(_first_nonempty(rec, ("modelB", "model_b"), "B"))
        model_c = str(_first_nonempty(rec, ("modelC", "model_c"), "C"))
        prompt = _build_listwise_prompt(
            system_prompt=LISTWISE_SYSTEM_PROMPT,
            instruction=instruction,
            input_text=input_text,
            assistant_a_output=output_a,
            assistant_b_output=output_b,
            assistant_c_output=output_c,
        )
        label = _label_from_ranking(ranking)
        ex_id = len(examples) + 1
        examples.append(
            ListwiseExample(
                id=int(ex_id),
                dataset=dataset,
                group_id=int(qid),
                source_id=int(source_id),
                model_a=model_a,
                model_b=model_b,
                model_c=model_c,
                prompt=prompt,
                ranking=str(ranking),
                label=int(label),
            )
        )
        row = dict(rec)
        row["id"] = int(ex_id)
        row["group_id"] = int(qid)
        row["source_id"] = int(source_id)
        row["ranking"] = str(ranking)
        row["label"] = int(label)
        rows.append(row)
        stats["label_counts"][ranking] = int(stats["label_counts"].get(ranking, 0) + 1)

    stats["examples"] = int(len(examples))
    if not examples:
        raise RuntimeError(f"listwise eval dataset produced no examples: {path}")
    return examples, rows, stats


def _confusion(true: np.ndarray, pred: np.ndarray, *, num_classes: int) -> List[List[int]]:
    conf = np.zeros((int(num_classes), int(num_classes)), dtype=np.int64)
    for t, p in zip(true.tolist(), pred.tolist()):
        if 0 <= int(t) < int(num_classes) and 0 <= int(p) < int(num_classes):
            conf[int(t), int(p)] += 1
    return conf.tolist()


def _ranking_top_group(ranking: str) -> Tuple[str, ...]:
    first = str(ranking).split(">", 1)[0]
    return tuple(sorted(x for x in first.split("=") if x))


def _ranking_rank_map(ranking: str) -> Dict[str, int]:
    ranking_s = _normalize_ranking_text(ranking)
    groups = [g for g in ranking_s.split(">") if g]
    out: Dict[str, int] = {}
    for rank_idx, group in enumerate(groups):
        for item in group.split("="):
            if item in {"A", "B", "C"}:
                out[item] = int(rank_idx)

    fallback_rank = int(len(groups))
    for item in ("A", "B", "C"):
        out.setdefault(item, fallback_rank)
    return out


def _pairwise_relation(rank_map: Dict[str, int], left: str, right: str) -> int:
    left_rank = int(rank_map[left])
    right_rank = int(rank_map[right])
    if left_rank < right_rank:
        return 1
    if left_rank > right_rank:
        return -1
    return 0


def _listwise_soft_metrics(true_rankings: Sequence[str], pred_rankings: Sequence[str]) -> Dict[str, Any]:
    pair_names = (("A", "B"), ("A", "C"), ("B", "C"))
    relation_correct = 0
    relation_total = 0
    true_tie_total = 0
    true_tie_correct = 0
    true_non_tie_total = 0
    true_non_tie_correct = 0
    pred_tie_total = 0
    per_pair_correct = {f"{a}_vs_{b}": 0 for a, b in pair_names}
    per_pair_total = {f"{a}_vs_{b}": 0 for a, b in pair_names}
    rank_abs_errors: List[float] = []
    rank_sq_errors: List[float] = []
    best_in_pred_top: List[bool] = []

    for true_ranking, pred_ranking in zip(true_rankings, pred_rankings):
        true_map = _ranking_rank_map(true_ranking)
        pred_map = _ranking_rank_map(pred_ranking)

        true_top = set(_ranking_top_group(true_ranking))
        pred_top = set(_ranking_top_group(pred_ranking))
        best_in_pred_top.append(bool(true_top & pred_top))

        for item in ("A", "B", "C"):
            err = abs(float(pred_map[item]) - float(true_map[item]))
            rank_abs_errors.append(err)
            rank_sq_errors.append(err * err)

        for left, right in pair_names:
            pair_key = f"{left}_vs_{right}"
            true_rel = _pairwise_relation(true_map, left, right)
            pred_rel = _pairwise_relation(pred_map, left, right)
            is_correct = int(true_rel == pred_rel)

            relation_correct += is_correct
            relation_total += 1
            per_pair_correct[pair_key] += is_correct
            per_pair_total[pair_key] += 1

            if pred_rel == 0:
                pred_tie_total += 1
            if true_rel == 0:
                true_tie_total += 1
                true_tie_correct += is_correct
            else:
                true_non_tie_total += 1
                true_non_tie_correct += is_correct

    per_pair_acc = {
        key: float(per_pair_correct[key] / per_pair_total[key]) if per_pair_total[key] else None
        for key in per_pair_total
    }
    return {
        "proxy_pairwise_relation_acc": (
            float(relation_correct / relation_total) if relation_total > 0 else None
        ),
        "proxy_pairwise_relation_per_pair_acc": per_pair_acc,
        "proxy_pairwise_true_tie_acc": (
            float(true_tie_correct / true_tie_total) if true_tie_total > 0 else None
        ),
        "proxy_pairwise_true_non_tie_acc": (
            float(true_non_tie_correct / true_non_tie_total) if true_non_tie_total > 0 else None
        ),
        "proxy_pairwise_true_tie_pair_count": int(true_tie_total),
        "proxy_pairwise_true_non_tie_pair_count": int(true_non_tie_total),
        "proxy_pairwise_pred_tie_pair_rate": (
            float(pred_tie_total / relation_total) if relation_total > 0 else None
        ),
        "proxy_best_in_pred_top_acc": float(np.mean(best_in_pred_top)) if best_in_pred_top else None,
        "proxy_rank_mae": float(np.mean(rank_abs_errors)) if rank_abs_errors else None,
        "proxy_rank_rmse": float(np.sqrt(np.mean(rank_sq_errors))) if rank_sq_errors else None,
    }


def _evaluate_listwise(proxy: base.LlamaSharedMultiTaskProxyModel, examples: Sequence[ListwiseExample]) -> Dict[str, Any]:
    n = int(len(examples))
    if n <= 0:
        return {"n": 0}

    true_labels = np.asarray([int(x.label) for x in examples], dtype=np.int64)
    probs = proxy.predict_proba_pairwise(list(examples))
    pred_labels = probs.argmax(axis=1).astype(np.int64)

    true_rankings = [str(x.ranking) for x in examples]
    pred_rankings = [LABEL_TO_RANKING[int(x)] for x in pred_labels.tolist()]
    true_top = [_ranking_top_group(x) for x in true_rankings]
    pred_top = [_ranking_top_group(x) for x in pred_rankings]
    top_group_acc = float(np.mean([t == p for t, p in zip(true_top, pred_top)]))
    soft_metrics = _listwise_soft_metrics(true_rankings, pred_rankings)

    true_counts = {r: 0 for r in RANKING_LABELS}
    pred_counts = {r: 0 for r in RANKING_LABELS}
    for r in true_rankings:
        true_counts[r] = int(true_counts.get(r, 0) + 1)
    for r in pred_rankings:
        pred_counts[r] = int(pred_counts.get(r, 0) + 1)

    return {
        "n": int(n),
        "proxy_acc": float((pred_labels == true_labels).mean()),
        "proxy_top_group_acc": float(top_group_acc),
        **soft_metrics,
        "proxy_tie_rate": float(np.mean(["=" in x for x in pred_rankings])),
        "proxy_confusion": _confusion(true_labels, pred_labels, num_classes=len(RANKING_LABELS)),
        "ranking_labels": list(RANKING_LABELS),
        "true_label_counts": true_counts,
        "pred_label_counts": pred_counts,
    }



class OnlineGlobalPriorPointwiseSmoother:
    """Online global-prior label smoothing for classifier-head pointwise CE."""

    def __init__(
        self,
        *,
        alpha: float,
        start_step: int,
        warmup_steps: int,
        prior: float,
        num_labels: int,
        trainable_alpha: bool = False,
        alpha_max: float = 0.2,
        alpha_reg: float = 0.0,
        alpha_lr: float = 0.0,
    ) -> None:
        self.alpha = max(0.0, float(alpha))
        self.start_step = max(0, int(start_step))
        self.warmup_steps = max(0, int(warmup_steps))
        self.prior = float(prior)
        self.num_labels = int(num_labels)
        self.trainable_alpha = bool(trainable_alpha) and self.alpha > 0.0
        self.alpha_max = float(alpha_max)
        self.alpha_reg = max(0.0, float(alpha_reg))
        self.alpha_lr = max(0.0, float(alpha_lr))
        self.alpha_raw: Optional[torch.nn.Parameter] = None
        self._optimizer_attached = False
        self._init_alpha = float(self.alpha)
        if self.prior <= 0.0:
            raise ValueError("pointwise global smoothing prior must be > 0")
        if self.num_labels <= 1:
            raise ValueError("pointwise global smoothing num_labels must be > 1")
        if self.alpha_max <= 0.0:
            raise ValueError("pointwise global smoothing alpha_max must be > 0")
        if self.trainable_alpha and self.alpha >= self.alpha_max:
            raise ValueError("trainable smoothing alpha must be < alpha_max")
        self.hist = torch.zeros((self.num_labels,), dtype=torch.float64)
        self.seen = 0
        self.step = 0

    @property
    def enabled(self) -> bool:
        return self.alpha > 0.0

    def attach_optimizer(self, optimizer: torch.optim.Optimizer, *, device: torch.device) -> None:
        if not self.trainable_alpha or self._optimizer_attached:
            return
        init_alpha = min(max(float(self.alpha), 1e-6), float(self.alpha_max) - 1e-6)
        init_ratio = min(max(init_alpha / float(self.alpha_max), 1e-6), 1.0 - 1e-6)
        raw = np.log(init_ratio / (1.0 - init_ratio))
        self.alpha_raw = torch.nn.Parameter(torch.tensor(float(raw), dtype=torch.float32, device=device))
        group: Dict[str, Any] = {"params": [self.alpha_raw]}
        if self.alpha_lr > 0.0:
            group["lr"] = float(self.alpha_lr)
        optimizer.add_param_group(group)
        self._optimizer_attached = True

    def _warmup_scale(self) -> float:
        if not self.enabled or self.step < self.start_step:
            return 0.0
        if self.warmup_steps <= 0:
            return 1.0
        return min(1.0, float(self.step - self.start_step + 1) / float(self.warmup_steps))

    def current_alpha(self) -> float:
        scale = self._warmup_scale()
        if scale <= 0.0:
            return 0.0
        if self.trainable_alpha and self.alpha_raw is not None:
            learned = float((torch.sigmoid(self.alpha_raw.detach().cpu()) * float(self.alpha_max)).item())
            return learned * scale
        return float(self.alpha) * scale

    def current_alpha_tensor(self, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        scale = self._warmup_scale()
        if self.trainable_alpha and self.alpha_raw is not None:
            alpha_t = torch.sigmoid(self.alpha_raw.to(device=device, dtype=dtype)) * float(self.alpha_max)
        else:
            alpha_t = torch.tensor(float(self.alpha), device=device, dtype=dtype)
        return alpha_t * torch.tensor(float(scale), device=device, dtype=dtype)

    def alpha_regularization(self, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if not self.trainable_alpha or self.alpha_reg <= 0.0 or self.alpha_raw is None:
            return torch.tensor(0.0, device=device, dtype=dtype)
        alpha_t = torch.sigmoid(self.alpha_raw.to(device=device, dtype=dtype)) * float(self.alpha_max)
        init_t = torch.tensor(float(self._init_alpha), device=device, dtype=dtype)
        return torch.tensor(float(self.alpha_reg), device=device, dtype=dtype) * (alpha_t - init_t).pow(2)

    def prior_distribution(self, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        hist = self.hist.to(device=device, dtype=dtype)
        dist = hist + float(self.prior)
        return dist / dist.sum().clamp_min(1e-12)

    def update(self, labels: Sequence[int]) -> None:
        for y in labels:
            yi = int(y)
            if 0 <= yi < self.num_labels:
                self.hist[yi] += 1.0
                self.seen += 1
        self.step += 1

    def stats(self) -> Dict[str, Any]:
        hist = [float(x) for x in self.hist.tolist()]
        total = float(sum(hist))
        dist = [float(x / total) for x in hist] if total > 0 else [0.0 for _ in hist]
        learned_alpha = None
        if self.trainable_alpha and self.alpha_raw is not None:
            learned_alpha = float((torch.sigmoid(self.alpha_raw.detach().cpu()) * float(self.alpha_max)).item())
        return {
            "enabled": bool(self.enabled),
            "alpha": float(self.alpha),
            "current_alpha": float(self.current_alpha()),
            "trainable_alpha": bool(self.trainable_alpha),
            "learned_alpha": learned_alpha,
            "alpha_max": float(self.alpha_max),
            "alpha_reg": float(self.alpha_reg),
            "alpha_lr": float(self.alpha_lr),
            "start_step": int(self.start_step),
            "warmup_steps": int(self.warmup_steps),
            "prior_smooth": float(self.prior),
            "step": int(self.step),
            "score_seen": int(self.seen),
            "hist": hist,
            "distribution": dist,
        }

def _train_pointwise_batch_with_optional_smoothing(
    *,
    proxy: base.LlamaSharedMultiTaskProxyModel,
    inputs: Sequence[base.PointwiseScoredExample],
    labels: Sequence[int],
    smoother: Optional[OnlineGlobalPriorPointwiseSmoother],
) -> None:
    if smoother is None or not smoother.enabled:
        proxy.train_on_batch_pointwise(inputs, labels)
        return
    if getattr(proxy, "multitask_mode", None) != "classifier_heads":
        raise ValueError("listwise pointwise smoothing currently requires classifier_heads mode")
    if getattr(proxy, "pointwise_loss_type", "ce") != "ce":
        raise ValueError("listwise pointwise smoothing currently supports pointwise-loss-type=ce only")
    head = getattr(proxy, "pointwise_head", None)
    if head is None:
        raise RuntimeError("pointwise smoothing requires proxy.pointwise_head")
    smoother.attach_optimizer(proxy.optimizer, device=proxy.device)

    proxy.model.train()
    head.train()
    mini_batch_size = 8
    n_samples = int(len(inputs))
    if n_samples <= 0:
        return

    indices = np.arange(n_samples)
    np.random.shuffle(indices)

    for start in range(0, n_samples, mini_batch_size):
        end = min(start + mini_batch_size, n_samples)
        batch_indices = indices[start:end]
        sub_inputs = [inputs[int(i)] for i in batch_indices]
        sub_labels = [int(labels[int(i)]) for i in batch_indices]

        batch = proxy._encode(sub_inputs)
        attention_mask = batch.get("attention_mask")
        labels_tensor = torch.tensor(sub_labels, dtype=torch.long, device=proxy.device)
        proxy.optimizer.zero_grad()

        ac = proxy._autocast_ctx()
        if ac is not None:
            with ac:
                hidden_states = proxy._forward_last_hidden_state(batch)
                features = proxy._pool_last_token_features(hidden_states, attention_mask)
                logits = head(features)
                log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
                hard_ce = torch.nn.functional.nll_loss(log_probs, labels_tensor, reduction="none")
                prior = smoother.prior_distribution(device=logits.device, dtype=log_probs.dtype)
                prior_ce = -(prior.unsqueeze(0) * log_probs).sum(dim=-1)
                alpha_t = smoother.current_alpha_tensor(device=logits.device, dtype=log_probs.dtype)
                loss = ((1.0 - alpha_t) * hard_ce + alpha_t * prior_ce).mean()
                loss = loss + smoother.alpha_regularization(device=logits.device, dtype=log_probs.dtype)
        else:
            hidden_states = proxy._forward_last_hidden_state(batch)
            features = proxy._pool_last_token_features(hidden_states, attention_mask)
            logits = head(features)
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            hard_ce = torch.nn.functional.nll_loss(log_probs, labels_tensor, reduction="none")
            prior = smoother.prior_distribution(device=logits.device, dtype=log_probs.dtype)
            prior_ce = -(prior.unsqueeze(0) * log_probs).sum(dim=-1)
            alpha_t = smoother.current_alpha_tensor(device=logits.device, dtype=log_probs.dtype)
            loss = ((1.0 - alpha_t) * hard_ce + alpha_t * prior_ce).mean()
            loss = loss + smoother.alpha_regularization(device=logits.device, dtype=log_probs.dtype)

        if getattr(proxy, "_scaler", None) is not None:
            proxy._scaler.scale(loss).backward()
            proxy._scaler.step(proxy.optimizer)
            proxy._scaler.update()
        else:
            loss.backward()
            proxy.optimizer.step()

        smoother.update(sub_labels)
        del batch, labels_tensor, hidden_states, features, logits, log_probs, hard_ce, prior_ce, loss



def _train_pointwise_stage_with_optional_smoothing(
    *,
    proxy: base.LlamaSharedMultiTaskProxyModel,
    examples: Sequence[base.PointwiseScoredExample],
    epochs: int,
    batch_size: int,
    seed: int,
    stage_name: str,
    smoother: Optional[OnlineGlobalPriorPointwiseSmoother] = None,
) -> Dict[str, Any]:
    n = int(len(examples))
    if n <= 0:
        return {
            "n": 0,
            "epochs": int(epochs),
            "steps": 0,
            "global_prior_smoothing": smoother.stats() if smoother is not None else None,
            "elapsed_sec": 0.0,
        }
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
            _train_pointwise_batch_with_optional_smoothing(
                proxy=proxy,
                inputs=batch_inputs,
                labels=batch_labels,
                smoother=smoother,
            )
            steps += 1
            step_in_ep += 1

            if step_in_ep % 20 == 0 or step_in_ep == total_steps_ep:
                print(f"[{stage_name}] epoch={ep + 1}/{epochs} step={step_in_ep}/{total_steps_ep}", flush=True)

    return {
        "n": int(n),
        "epochs": int(epochs),
        "steps": int(steps),
        "global_prior_smoothing": smoother.stats() if smoother is not None else None,
        "elapsed_sec": float(time.time() - t0),
    }


def _train_listwise_stage(
    *,
    proxy: base.LlamaSharedMultiTaskProxyModel,
    examples: Sequence[ListwiseExample],
    epochs: int,
    batch_size: int,
    seed: int,
    stage_name: str,
) -> Dict[str, Any]:
    n = int(len(examples))
    if n <= 0:
        return {"n": 0, "epochs": int(epochs), "steps": 0, "elapsed_sec": 0.0}
    if int(batch_size) <= 0:
        raise ValueError("listwise batch size must be > 0")

    rng = np.random.default_rng(int(seed))
    steps = 0
    t0 = time.time()

    for ep in range(int(epochs)):
        order = rng.permutation(n)
        step_in_ep = 0
        total_steps_ep = int((n + int(batch_size) - 1) // int(batch_size))
        for start in range(0, n, int(batch_size)):
            idx = order[start : start + int(batch_size)]
            batch_inputs = [examples[int(i)] for i in idx.tolist()]
            batch_labels = [int(x.label) for x in batch_inputs]
            proxy.train_on_batch_pairwise(batch_inputs, batch_labels)
            steps += 1
            step_in_ep += 1

            if step_in_ep % 20 == 0 or step_in_ep == total_steps_ep:
                print(f"[{stage_name}] epoch={ep + 1}/{epochs} step={step_in_ep}/{total_steps_ep}", flush=True)

    return {"n": int(n), "epochs": int(epochs), "steps": int(steps), "elapsed_sec": float(time.time() - t0)}


def _train_stage2_alternating(
    *,
    proxy: base.LlamaSharedMultiTaskProxyModel,
    listwise_examples: Sequence[ListwiseExample],
    pointwise_examples: Sequence[base.PointwiseScoredExample],
    epochs: int,
    listwise_batch_size: int,
    pointwise_batch_size: int,
    pointwise_replay_ratio: int,
    seed: int,
    stage_name: str,
    smoother: Optional[OnlineGlobalPriorPointwiseSmoother] = None,
) -> Dict[str, Any]:
    n_list = int(len(listwise_examples))
    n_point = int(len(pointwise_examples))
    if n_list <= 0:
        return {
            "n_listwise": 0,
            "n_pointwise_replay": int(n_point),
            "epochs": int(epochs),
            "listwise_steps": 0,
            "pointwise_replay_steps": 0,
            "pointwise_replay_ratio": int(pointwise_replay_ratio),
            "global_prior_smoothing": smoother.stats() if smoother is not None else None,
            "elapsed_sec": 0.0,
        }
    if int(listwise_batch_size) <= 0:
        raise ValueError("listwise batch size must be > 0")
    if int(pointwise_batch_size) <= 0:
        raise ValueError("pointwise batch size must be > 0")
    if int(pointwise_replay_ratio) < 0:
        raise ValueError("pointwise replay ratio must be >= 0")

    rng = np.random.default_rng(int(seed))
    list_steps = 0
    replay_steps = 0
    t0 = time.time()

    point_order = rng.permutation(n_point) if n_point > 0 else np.asarray([], dtype=np.int64)
    point_cursor = 0

    def next_pointwise_indices() -> np.ndarray:
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
        order_list = rng.permutation(n_list)
        total_steps_ep = int((n_list + int(listwise_batch_size) - 1) // int(listwise_batch_size))
        step_in_ep = 0

        for start in range(0, n_list, int(listwise_batch_size)):
            idx_list = order_list[start : start + int(listwise_batch_size)]
            batch_list_inputs = [listwise_examples[int(i)] for i in idx_list.tolist()]
            batch_list_labels = [int(x.label) for x in batch_list_inputs]
            proxy.train_on_batch_pairwise(batch_list_inputs, batch_list_labels)
            list_steps += 1
            step_in_ep += 1

            for _ in range(int(pointwise_replay_ratio)):
                if n_point <= 0:
                    break
                idx_point = next_pointwise_indices()
                if idx_point.size <= 0:
                    break
                batch_point_inputs = [pointwise_examples[int(i)] for i in idx_point.tolist()]
                batch_point_labels = [int(x.label) for x in batch_point_inputs]
                _train_pointwise_batch_with_optional_smoothing(
                    proxy=proxy,
                    inputs=batch_point_inputs,
                    labels=batch_point_labels,
                    smoother=smoother,
                )
                replay_steps += 1

            if step_in_ep % 20 == 0 or step_in_ep == total_steps_ep:
                print(
                    f"[{stage_name}] epoch={ep + 1}/{epochs} list_step={step_in_ep}/{total_steps_ep} "
                    f"replay_ratio={pointwise_replay_ratio}",
                    flush=True,
                )

    return {
        "n_listwise": int(n_list),
        "n_pointwise_replay": int(n_point),
        "epochs": int(epochs),
        "listwise_steps": int(list_steps),
        "pointwise_replay_steps": int(replay_steps),
        "pointwise_replay_ratio": int(pointwise_replay_ratio),
        "global_prior_smoothing": smoother.stats() if smoother is not None else None,
        "elapsed_sec": float(time.time() - t0),
    }


def _fmt_metric(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.4f}"
    except Exception:
        return str(value)


def _write_run_summary(base_out: Path, summary: Dict[str, Any]) -> None:
    pointwise = summary.get("pointwise_metrics", {})
    listwise = summary.get("listwise_metrics", {})
    compact = {
        "mode": "pointwise3_to_listwise_proxy",
        "budget": summary.get("train_budget", {}),
        "pointwise": {
            k: {
                "n": v.get("n"),
                "acc": v.get("proxy_acc"),
                "within1": v.get("proxy_within1"),
                "mae": v.get("proxy_mae"),
            }
            for k, v in pointwise.items()
            if isinstance(v, dict)
        },
        "listwise": {
            k: {
                "n": v.get("n"),
                "acc": v.get("proxy_acc"),
                "top_group_acc": v.get("proxy_top_group_acc"),
                "pairwise_relation_acc": v.get("proxy_pairwise_relation_acc"),
                "best_in_pred_top_acc": v.get("proxy_best_in_pred_top_acc"),
                "rank_mae": v.get("proxy_rank_mae"),
                "rank_rmse": v.get("proxy_rank_rmse"),
                "tie_rate": v.get("proxy_tie_rate"),
                "pairwise_true_tie_acc": v.get("proxy_pairwise_true_tie_acc"),
                "pairwise_true_non_tie_acc": v.get("proxy_pairwise_true_non_tie_acc"),
                "pairwise_pred_tie_pair_rate": v.get("proxy_pairwise_pred_tie_pair_rate"),
            }
            for k, v in listwise.items()
            if isinstance(v, dict)
        },
    }
    base._write_json(base_out / "metrics_compact.json", compact)

    final_listwise = compact["listwise"].get("after_stage2") or compact["listwise"].get("after_stage1") or {}
    print("\n" + "=" * 60)
    print("Run finished")
    print("=" * 60)
    if final_listwise:
        print(
            "Listwise final: "
            f"acc={_fmt_metric(final_listwise.get('acc'))} "
            f"top_group={_fmt_metric(final_listwise.get('top_group_acc'))} "
            f"pair_rel={_fmt_metric(final_listwise.get('pairwise_relation_acc'))} "
            f"best_in_top={_fmt_metric(final_listwise.get('best_in_pred_top_acc'))} "
            f"rank_mae={_fmt_metric(final_listwise.get('rank_mae'))} "
            f"tie={_fmt_metric(final_listwise.get('tie_rate'))}"
        )
    print(f"Compact metrics: {base_out / 'metrics_compact.json'}")
    print(f"Output directory: {base_out}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pointwise(3 answers per question) -> listwise ranking training.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--pointwise-5answers-dataset", required=True, help="Scored multi-answer pointwise JSON path")
    parser.add_argument("--llama", required=True, help="Local HF model directory")
    parser.add_argument(
        "--listwise-eval-dataset",
        default="train_with_selector/train_with_selector/data/newnew/val-2k-eval-listwise.json",
        help="Listwise A/B/C ranking JSON used for validation",
    )
    parser.add_argument("--out", default=None, help="Output directory")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Held-out question ratio for pointwise validation")
    parser.add_argument(
        "--val-split-seed",
        type=int,
        default=42,
        help="Seed for the internal held-out question split; fixed by default so pointwise validation questions do not change when --seed changes.",
    )
    parser.add_argument(
        "--pointwise-val-answer-seed",
        type=int,
        default=65,
        help="Seed for selecting one answer from each pointwise validation question; fixed by default so pointwise validation answers do not change when --seed changes.",
    )
    parser.add_argument(
        "--train-selection-mode",
        type=str,
        default="candidate_triple_selector",
        choices=["selected_triple", "candidate_triple_selector"],
        help="selected_triple picks one triple per question; candidate_triple_selector enumerates triples and selects them with a selector.",
    )
    parser.add_argument(
        "--triple-selection-strategy",
        type=str,
        default="random",
        choices=["random", "first_three"],
        help="Per-question rule for selecting 3 answers before listwise conversion",
    )
    parser.add_argument(
        "--question-selection-strategy",
        type=str,
        default="first",
        choices=["first", "random"],
        help="Question selection rule before applying --budget-units.",
    )
    parser.add_argument("--no-randomize-listwise-order", action="store_true", help="Disable random A/B/C order")
    parser.add_argument(
        "--listwise-order-augmentation",
        action="store_true",
        help="Add all 6 A/B/C permutations for each selected triple during listwise training",
    )
    parser.add_argument(
        "--budget-units",
        type=int,
        default=0,
        help="Training budget in pointwise answer units; each selected question consumes 3 units. 0 uses all train questions.",
    )
    parser.add_argument(
        "--candidate-selector-kind",
        type=str,
        default="bias_trap_pointwise",
        choices=["bert", "random", "bias_trap_pointwise", "shared_llama", "shared_llama_two_stage"],
        help="Selector used when --train-selection-mode=candidate_triple_selector.",
    )
    parser.add_argument("--candidate-selector-init-triples", type=int, default=200)
    parser.add_argument("--candidate-selector-batch-size", type=int, default=50)
    parser.add_argument("--candidate-selector-epochs", type=int, default=4)
    parser.add_argument(
        "--candidate-selector-max-score-candidates",
        type=int,
        default=0,
        help="Max remaining candidate triples scored by the selector per round. 0 scores all remaining candidates.",
    )
    parser.add_argument("--candidate-selector-llama-rerank-candidates", type=int, default=1000)
    parser.add_argument("--candidate-selector-buffer-maxlen", type=int, default=1000)
    parser.add_argument(
        "--candidate-selector-allow-multiple-per-question",
        action="store_true",
        help="Allow selecting multiple candidate triples from the same original question.",
    )
    parser.add_argument(
        "--candidate-selector-target-task",
        type=str,
        default="pointwise",
        choices=["score_spread", "pointwise"],
        help="Target signal used to train the candidate triple selector. score_spread preserves the old range/gap behavior; pointwise mimics the pair selector's pointwise uncertainty target.",
    )
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
        default=DEFAULT_SELECTOR_EMBEDDING_MODEL,
        help="Transformer encoder used as femb(q, ai) for bias_trap_pointwise diversity/density/prefix matching.",
    )
    parser.add_argument("--candidate-selector-embedding-max-length", type=int, default=DEFAULT_SELECTOR_EMBEDDING_MAX_LENGTH)
    parser.add_argument("--candidate-selector-embedding-batch-size", type=int, default=64)
    parser.add_argument("--candidate-selector-embedding-device", default="auto")
    parser.add_argument("--candidate-selector-embedding-pooling", choices=["cls", "mean"], default="cls")
    parser.add_argument(
        "--candidate-selector-diversity-view",
        choices=["pointwise", "joint"],
        default="pointwise",
        help="pointwise uses mean femb(q,a_i); joint averages pointwise, three pairwise prompts, and one listwise prompt.",
    )
    parser.add_argument("--candidate-bert-selector-model", type=str, default="bert-base-uncased")
    parser.add_argument("--candidate-bert-selector-max-length", type=int, default=512)
    parser.add_argument("--candidate-bert-selector-unfreeze", action="store_true")
    parser.add_argument("--candidate-bert-selector-unfreeze-last-n-layers", type=int, default=0)

    parser.add_argument("--pointwise-epochs", type=int, default=1)
    parser.add_argument("--listwise-epochs", "--pairwise-epochs", dest="listwise_epochs", type=int, default=1)
    parser.add_argument("--pointwise-batch-size", type=int, default=128)
    parser.add_argument("--listwise-batch-size", "--pairwise-batch-size", dest="listwise_batch_size", type=int, default=64)
    parser.add_argument(
        "--stage2-pointwise-replay-ratio",
        type=int,
        default=1,
        help="During stage-2, run this many pointwise replay steps after each listwise step.",
    )
    parser.add_argument(
        "--pointwise-global-smooth-alpha",
        type=float,
        default=0.1,
        help="Mix weight for online global-prior smoothing on proxy pointwise CE. 0 disables it.",
    )
    parser.add_argument("--pointwise-global-smooth-start-step", type=int, default=0)
    parser.add_argument("--pointwise-global-smooth-warmup-steps", type=int, default=0)
    parser.add_argument("--pointwise-global-smooth-prior", type=float, default=1.0)
    parser.add_argument("--pointwise-global-smooth-trainable-alpha", action="store_true")
    parser.add_argument("--pointwise-global-smooth-alpha-max", type=float, default=0.2)
    parser.add_argument("--pointwise-global-smooth-alpha-reg", type=float, default=0.0)
    parser.add_argument("--pointwise-global-smooth-alpha-lr", type=float, default=0.0)

    parser.add_argument("--score-min", type=int, default=1)
    parser.add_argument("--score-max", type=int, default=10)
    parser.add_argument("--no-fix-score-prefix", action="store_true", help="Do not append 'Score: [' in pointwise prompt")

    parser.add_argument("--proxy-lr", type=float, default=1e-4)
    parser.add_argument("--proxy-max-length", type=int, default=1024)
    parser.add_argument("--no-4bit", action="store_true", help="Disable 4-bit loading")
    parser.add_argument(
        "--llama-multitask-mode",
        type=str,
        default="classifier_heads",
        choices=["classifier_heads"],
        help="Listwise uses 13 labels, so classifier_heads is required.",
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
        help="Auxiliary distance-loss weight used by ce_mse or ce_cost.",
    )
    parser.add_argument(
        "--pointwise-class-weight-mode",
        type=str,
        default="none",
        choices=["none", "inv_sqrt"],
        help="Optional class weighting mode for proxy pointwise training.",
    )
    parser.add_argument("--no-pointwise-class-weight", action="store_true")
    parser.add_argument("--pointwise-class-weight-strength", type=float, default=1.0)

    args = parser.parse_args()
    if bool(args.no_pointwise_class_weight):
        args.pointwise_class_weight_mode = "none"

    cfg = RunConfig(
        seed=int(args.seed),
        val_ratio=float(args.val_ratio),
        val_split_seed=int(args.val_split_seed),
        pointwise_val_answer_seed=int(args.pointwise_val_answer_seed),
        train_selection_mode=str(args.train_selection_mode),
        triple_selection_strategy=str(args.triple_selection_strategy),
        question_selection_strategy=str(args.question_selection_strategy),
        randomize_listwise_order=bool(not args.no_randomize_listwise_order),
        candidate_selector_kind=str(args.candidate_selector_kind),
        candidate_selector_init_triples=int(args.candidate_selector_init_triples),
        candidate_selector_batch_size=int(args.candidate_selector_batch_size),
        candidate_selector_epochs=int(args.candidate_selector_epochs),
        candidate_selector_max_score_candidates=int(args.candidate_selector_max_score_candidates),
        candidate_selector_llama_rerank_candidates=int(args.candidate_selector_llama_rerank_candidates),
        candidate_selector_buffer_maxlen=int(args.candidate_selector_buffer_maxlen),
        candidate_selector_one_per_question=bool(not args.candidate_selector_allow_multiple_per_question),
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
        candidate_bert_selector_model=str(args.candidate_bert_selector_model),
        candidate_bert_selector_max_length=int(args.candidate_bert_selector_max_length),
        candidate_bert_selector_freeze=bool(not args.candidate_bert_selector_unfreeze),
        candidate_bert_selector_unfreeze_last_n_layers=int(args.candidate_bert_selector_unfreeze_last_n_layers),
        listwise_order_augmentation=bool(args.listwise_order_augmentation),
        budget_units=int(args.budget_units),
        pointwise_epochs=int(args.pointwise_epochs),
        listwise_epochs=int(args.listwise_epochs),
        pointwise_batch_size=int(args.pointwise_batch_size),
        listwise_batch_size=int(args.listwise_batch_size),
        stage2_pointwise_replay_ratio=int(args.stage2_pointwise_replay_ratio),
        score_min=int(args.score_min),
        score_max=int(args.score_max),
        fix_score_prefix_in_prompt=bool(not args.no_fix_score_prefix),
        proxy_lr=float(args.proxy_lr),
        proxy_max_length=int(args.proxy_max_length),
        load_in_4bit=bool(not args.no_4bit),
        llama_multitask_mode=str(args.llama_multitask_mode),
        pointwise_loss_type=str(args.pointwise_loss_type),
        pointwise_distance_weight=float(args.pointwise_distance_weight),
        pointwise_class_weight_mode=str(args.pointwise_class_weight_mode),
        pointwise_class_weight_strength=float(args.pointwise_class_weight_strength),
        pointwise_global_smooth_alpha=float(args.pointwise_global_smooth_alpha),
        pointwise_global_smooth_start_step=int(args.pointwise_global_smooth_start_step),
        pointwise_global_smooth_warmup_steps=int(args.pointwise_global_smooth_warmup_steps),
        pointwise_global_smooth_prior=float(args.pointwise_global_smooth_prior),
        pointwise_global_smooth_trainable_alpha=bool(args.pointwise_global_smooth_trainable_alpha),
        pointwise_global_smooth_alpha_max=float(args.pointwise_global_smooth_alpha_max),
        pointwise_global_smooth_alpha_reg=float(args.pointwise_global_smooth_alpha_reg),
        pointwise_global_smooth_alpha_lr=float(args.pointwise_global_smooth_alpha_lr),
    )

    if cfg.score_min >= cfg.score_max:
        raise ValueError("score-min must be < score-max")
    if not (0.0 <= float(cfg.val_ratio) < 1.0):
        raise ValueError("val-ratio must be in [0, 1)")
    if int(cfg.pointwise_batch_size) <= 0 or int(cfg.listwise_batch_size) <= 0:
        raise ValueError("batch sizes must be > 0")
    if int(cfg.stage2_pointwise_replay_ratio) < 0:
        raise ValueError("stage2-pointwise-replay-ratio must be >= 0")
    if str(cfg.train_selection_mode) not in {"selected_triple", "candidate_triple_selector"}:
        raise ValueError("train-selection-mode must be one of {'selected_triple','candidate_triple_selector'}")
    if str(cfg.candidate_selector_kind) not in {"bert", "random", "bias_trap_pointwise", "shared_llama", "shared_llama_two_stage"}:
        raise ValueError("unknown candidate-selector-kind")
    if int(cfg.candidate_selector_init_triples) <= 0:
        raise ValueError("candidate-selector-init-triples must be > 0")
    if int(cfg.candidate_selector_batch_size) <= 0:
        raise ValueError("candidate-selector-batch-size must be > 0")
    if int(cfg.candidate_selector_epochs) <= 0:
        raise ValueError("candidate-selector-epochs must be > 0")
    if int(cfg.candidate_selector_max_score_candidates) < 0:
        raise ValueError("candidate-selector-max-score-candidates must be >= 0")
    if str(cfg.candidate_selector_target_task) not in {"score_spread", "pointwise"}:
        raise ValueError("candidate-selector-target-task must be one of {'score_spread','pointwise'}")
    if float(cfg.candidate_selector_score_range_weight) < 0.0:
        raise ValueError("candidate-selector-score-range-weight must be >= 0")
    if float(cfg.candidate_selector_gap_sum_weight) < 0.0:
        raise ValueError("candidate-selector-gap-sum-weight must be >= 0")
    if float(cfg.candidate_selector_uncertainty_weight) < 0.0:
        raise ValueError("candidate-selector-uncertainty-weight must be >= 0")
    if float(cfg.candidate_selector_pairwise_uncertainty_weight) < 0.0:
        raise ValueError("candidate-selector-pairwise-uncertainty-weight must be >= 0")
    if float(cfg.candidate_selector_listwise_uncertainty_weight) < 0.0:
        raise ValueError("candidate-selector-listwise-uncertainty-weight must be >= 0")
    if float(cfg.candidate_selector_kl_weight) < 0.0:
        raise ValueError("candidate-selector-kl-weight must be >= 0")
    if float(cfg.candidate_selector_score_bin_weight) < 0.0:
        raise ValueError("candidate-selector-score-bin-weight must be >= 0")
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
    if str(cfg.candidate_selector_kind) == "bias_trap_pointwise":
        if str(cfg.candidate_selector_target_task) != "pointwise":
            raise ValueError("bias_trap_pointwise selector requires --candidate-selector-target-task pointwise")
    elif str(cfg.candidate_selector_target_task) == "score_spread":
        if float(cfg.candidate_selector_score_range_weight) + float(cfg.candidate_selector_gap_sum_weight) <= 0.0:
            raise ValueError("At least one score-spread selector target weight must be > 0")
    else:
        pointwise_weight = (
            float(cfg.candidate_selector_uncertainty_weight)
            + float(cfg.candidate_selector_kl_weight)
            + float(cfg.candidate_selector_pairwise_uncertainty_weight)
            + float(cfg.candidate_selector_listwise_uncertainty_weight)
            + float(cfg.candidate_selector_score_range_weight)
            + float(cfg.candidate_selector_gap_sum_weight)
            + float(cfg.candidate_selector_score_bin_weight)
        )
        if pointwise_weight <= 0.0:
            raise ValueError("At least one pointwise selector target weight must be > 0")
    if int(cfg.candidate_bert_selector_max_length) <= 0:
        raise ValueError("candidate-bert-selector-max-length must be > 0")
    if int(cfg.candidate_bert_selector_unfreeze_last_n_layers) < 0:
        raise ValueError("candidate-bert-selector-unfreeze-last-n-layers must be >= 0")
    if float(cfg.pointwise_distance_weight) < 0.0:
        raise ValueError("pointwise-distance-weight must be >= 0")
    if float(cfg.pointwise_class_weight_strength) < 0.0:
        raise ValueError("pointwise-class-weight-strength must be >= 0")
    if not (0.0 <= float(cfg.pointwise_global_smooth_alpha) <= 1.0):
        raise ValueError("pointwise-global-smooth-alpha must be in [0, 1]")
    if int(cfg.pointwise_global_smooth_start_step) < 0:
        raise ValueError("pointwise-global-smooth-start-step must be >= 0")
    if int(cfg.pointwise_global_smooth_warmup_steps) < 0:
        raise ValueError("pointwise-global-smooth-warmup-steps must be >= 0")
    if float(cfg.pointwise_global_smooth_prior) <= 0.0:
        raise ValueError("pointwise-global-smooth-prior must be > 0")
    if float(cfg.pointwise_global_smooth_alpha_max) <= 0.0:
        raise ValueError("pointwise-global-smooth-alpha-max must be > 0")
    if bool(cfg.pointwise_global_smooth_trainable_alpha) and float(cfg.pointwise_global_smooth_alpha) <= 0.0:
        raise ValueError("trainable smoothing alpha requires --pointwise-global-smooth-alpha > 0")
    if bool(cfg.pointwise_global_smooth_trainable_alpha) and float(cfg.pointwise_global_smooth_alpha) >= float(cfg.pointwise_global_smooth_alpha_max):
        raise ValueError("trainable smoothing alpha requires alpha < alpha_max")
    if float(cfg.pointwise_global_smooth_alpha_reg) < 0.0:
        raise ValueError("pointwise-global-smooth-alpha-reg must be >= 0")
    if float(cfg.pointwise_global_smooth_alpha_lr) < 0.0:
        raise ValueError("pointwise-global-smooth-alpha-lr must be >= 0")

    ds_path_raw = str(args.pointwise_5answers_dataset)
    ds_path = base._resolve_existing_path(ds_path_raw)
    eval_path_raw = str(args.listwise_eval_dataset)
    eval_path = base._resolve_existing_path(eval_path_raw)
    if not ds_path or not Path(ds_path).exists():
        raise FileNotFoundError(f"pointwise dataset not found: {ds_path_raw}")
    if not eval_path or not Path(eval_path).exists():
        raise FileNotFoundError(f"listwise eval dataset not found: {eval_path_raw}")

    print("\n" + "=" * 80)
    print("Start run: pointwise(3 selected answers) -> listwise ranking + pointwise replay")
    print("=" * 80)
    print(f"pointwise_5answers_dataset = {ds_path}")
    print(f"listwise_eval_dataset      = {eval_path}")
    print(f"llama                      = {args.llama}")
    print(f"seed                       = {cfg.seed}")
    print(f"val_ratio                  = {cfg.val_ratio}")
    print(f"val_split_seed            = {cfg.val_split_seed}")
    print(f"pointwise_val_answer_seed = {cfg.pointwise_val_answer_seed}")
    print(f"train_selection_mode       = {cfg.train_selection_mode}")
    print(f"triple_selection_strategy  = {cfg.triple_selection_strategy}")
    print(f"question_selection_strategy= {cfg.question_selection_strategy}")
    if str(cfg.train_selection_mode) == "candidate_triple_selector":
        print(f"candidate_selector_kind    = {cfg.candidate_selector_kind}")
        print(f"candidate_selector_init    = {cfg.candidate_selector_init_triples}")
        print(f"candidate_selector_batch   = {cfg.candidate_selector_batch_size}")
        print(f"candidate_selector_epochs  = {cfg.candidate_selector_epochs}")
        print(f"candidate_one_per_question = {cfg.candidate_selector_one_per_question}")
        print(f"candidate_target_task      = {cfg.candidate_selector_target_task}")
        print(f"candidate_range_weight     = {cfg.candidate_selector_score_range_weight}")
        print(f"candidate_gap_sum_weight   = {cfg.candidate_selector_gap_sum_weight}")
        print(f"candidate_uncertainty_w    = {cfg.candidate_selector_uncertainty_weight}")
        print(f"candidate_pair_uncert_w    = {cfg.candidate_selector_pairwise_uncertainty_weight}")
        print(f"candidate_list_uncert_w    = {cfg.candidate_selector_listwise_uncertainty_weight}")
        print(f"candidate_kl_weight        = {cfg.candidate_selector_kl_weight}")
        print(f"candidate_score_bin_w      = {cfg.candidate_selector_score_bin_weight}")
        print(f"candidate_bert_model       = {cfg.candidate_bert_selector_model}")
    print(f"randomize_listwise_order   = {cfg.randomize_listwise_order}")
    print(f"listwise_order_aug         = {cfg.listwise_order_augmentation}")
    print(f"budget_units               = {cfg.budget_units}")
    print(f"pointwise_epochs           = {cfg.pointwise_epochs}")
    print(f"listwise_epochs            = {cfg.listwise_epochs}")
    print(f"pointwise_batch_size       = {cfg.pointwise_batch_size}")
    print(f"listwise_batch_size        = {cfg.listwise_batch_size}")
    print(f"stage2_pointwise_replay    = {cfg.stage2_pointwise_replay_ratio}")
    print(f"proxy_lr                   = {cfg.proxy_lr}")
    print(f"proxy_max_length           = {cfg.proxy_max_length}")
    print(f"load_in_4bit               = {cfg.load_in_4bit}")
    print(f"llama_multitask_mode       = {cfg.llama_multitask_mode}")
    print(f"pointwise_loss_type        = {cfg.pointwise_loss_type}")
    print(f"pointwise_distance_w       = {cfg.pointwise_distance_weight}")
    print(f"pointwise_class_weight     = {cfg.pointwise_class_weight_mode}")
    print(f"pw_global_smooth_alpha     = {cfg.pointwise_global_smooth_alpha}")
    print(f"pw_global_smooth_start     = {cfg.pointwise_global_smooth_start_step}")
    print(f"pw_global_smooth_warmup    = {cfg.pointwise_global_smooth_warmup_steps}")
    print(f"pw_global_smooth_prior     = {cfg.pointwise_global_smooth_prior}")
    print(f"pw_global_smooth_trainable = {cfg.pointwise_global_smooth_trainable_alpha}")
    print(f"pw_global_smooth_alpha_max = {cfg.pointwise_global_smooth_alpha_max}")
    print(f"pw_global_smooth_alpha_reg = {cfg.pointwise_global_smooth_alpha_reg}")
    print(f"pw_global_smooth_alpha_lr  = {cfg.pointwise_global_smooth_alpha_lr}")

    base._log_memory_usage("startup")

    base_out = (
        Path(args.out)
        if args.out
        else Path(__file__).resolve().parent
        / "outputs"
        / ("pointwise5answers_three_to_listwise_v1_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    )
    base_out.mkdir(parents=True, exist_ok=True)
    print(f"output_dir                 = {base_out}")

    os.environ.setdefault("PYTHONHASHSEED", str(cfg.seed))
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    base._write_json(
        base_out / "config.json",
        {
            **asdict(cfg),
            "pointwise_5answers_dataset": str(ds_path),
            "pointwise_5answers_dataset_raw": str(ds_path_raw),
            "listwise_eval_dataset": str(eval_path),
            "listwise_eval_dataset_raw": str(eval_path_raw),
            "llama": str(args.llama),
            "judge_system_prompt": base.JUDGE_SYSTEM_PROMPT_SCORE_ONLY,
            "listwise_system_prompt": LISTWISE_SYSTEM_PROMPT,
            "ranking_labels": list(RANKING_LABELS),
        },
    )

    print("\nLoading scored multi-answer dataset...")
    questions, load_stats = _load_scored_questions_ge3(
        str(ds_path),
        score_min=int(cfg.score_min),
        score_max=int(cfg.score_max),
    )
    base._write_json(base_out / "dataset_load_stats.json", load_stats)
    print(f"Loaded questions with >=3 valid answers: {len(questions)}")

    train_questions, val_questions, split_info = base._split_questions(
        questions,
        seed=int(cfg.val_split_seed),
        val_ratio=float(cfg.val_ratio),
    )
    split_info["pointwise_val_mode"] = "one_random_answer_per_heldout_question"
    split_info["val_split_seed"] = int(cfg.val_split_seed)
    split_info["pointwise_val_answer_seed"] = int(cfg.pointwise_val_answer_seed)
    split_info["listwise_val_mode"] = "external_listwise_dataset"
    base._write_json(base_out / "split_questions.json", split_info)

    selected_candidate_rows: List[Dict[str, Any]] = []
    candidate_triple_stats: Optional[Dict[str, Any]] = None
    if str(cfg.train_selection_mode) == "candidate_triple_selector":
        print("\nBuilding candidate triples for selector...")
        candidates, candidate_rows, candidate_pool_stats = _build_candidate_triple_examples(
            train_questions,
            randomize_order=bool(cfg.randomize_listwise_order),
            seed=int(cfg.seed) + 11,
        )
        base._write_json(base_out / "candidate_triple_pool_stats.json", candidate_pool_stats)
        base._write_jsonl(base_out / "candidate_triples.jsonl", candidate_rows)
        print(f"Candidate triples: {len(candidates)}")
        print("Selecting candidate triples with selector...")
        train_triples, selected_candidate_rows, candidate_triple_stats = _select_candidate_triples_with_selector(
            candidates=candidates,
            cfg=cfg,
            llama_path=str(args.llama),
            output_dir=base_out,
        )
        selected_rows = selected_candidate_rows
        selected_stats = dict(candidate_triple_stats)
        base._write_json(base_out / "candidate_triple_selection_stats.json", candidate_triple_stats)
        base._write_jsonl(base_out / "selected_candidate_triples.jsonl", selected_candidate_rows)
    else:
        print("\nSelecting 3 answers per training question...")
        train_triples, selected_rows, selected_stats = _select_question_triples(
            train_questions,
            strategy=str(cfg.triple_selection_strategy),
            randomize_order=bool(cfg.randomize_listwise_order),
            question_selection_strategy=str(cfg.question_selection_strategy),
            seed=int(cfg.seed) + 11,
            budget_units=int(cfg.budget_units),
        )
    base._write_json(base_out / "selected_triple_stats.json", selected_stats)
    base._write_jsonl(base_out / "selected_triples.jsonl", selected_rows)
    print(f"Selected training question triples: {len(train_triples)}")

    pointwise_train, pointwise_train_rows, pointwise_train_stats = _build_pointwise_examples_from_triples(
        train_triples,
        score_min=int(cfg.score_min),
        score_max=int(cfg.score_max),
        fix_score_prefix_in_prompt=bool(cfg.fix_score_prefix_in_prompt),
    )
    base._write_json(base_out / "pointwise_train_stats.json", pointwise_train_stats)
    base._write_jsonl(base_out / "pointwise_train.jsonl", pointwise_train_rows)

    if val_questions:
        pointwise_val, pointwise_val_rows, pointwise_val_stats = base._build_single_answer_pointwise_eval_examples(
            val_questions,
            seed=int(cfg.pointwise_val_answer_seed),
            score_min=int(cfg.score_min),
            score_max=int(cfg.score_max),
            judge_system_prompt=base.JUDGE_SYSTEM_PROMPT_SCORE_ONLY,
            fix_score_prefix_in_prompt=bool(cfg.fix_score_prefix_in_prompt),
        )
    else:
        pointwise_val = list(pointwise_train)
        pointwise_val_rows = []
        pointwise_val_stats = {"selected_answers": int(len(pointwise_val)), "fallback": "pointwise_train"}
    pointwise_val_stats["selection_seed"] = int(cfg.pointwise_val_answer_seed)
    base._write_json(base_out / "pointwise_val_stats.json", pointwise_val_stats)
    base._write_jsonl(base_out / "pointwise_val.jsonl", pointwise_val_rows)

    print("\nConverting selected triples to listwise training samples...")
    listwise_train, listwise_train_rows, listwise_train_stats = _build_listwise_examples_from_triples(
        train_triples,
        order_augmentation=bool(cfg.listwise_order_augmentation),
    )
    base._write_json(base_out / "listwise_train_stats.json", listwise_train_stats)
    base._write_jsonl(base_out / "listwise_train.jsonl", listwise_train_rows)
    print(f"Converted listwise train samples: {len(listwise_train)}")

    print("\nLoading listwise eval dataset...")
    listwise_eval, listwise_eval_rows, listwise_eval_stats = _load_listwise_eval_dataset(str(eval_path))
    base._write_json(base_out / "listwise_eval_stats.json", listwise_eval_stats)
    base._write_jsonl(base_out / "listwise_eval.jsonl", listwise_eval_rows)
    print(f"Loaded listwise eval samples: {len(listwise_eval)}")

    base._write_json(
        base_out / "split_pointwise.json",
        {
            "train_size": int(len(pointwise_train)),
            "val_size": int(len(pointwise_val)),
            "train_questions": int(len(train_triples)),
            "val_questions": int(len(val_questions)),
            "split_mode": str(split_info.get("split_mode", "random_by_question")),
            "budget_units": int(cfg.budget_units),
            "effective_budget_units": int(selected_stats.get("effective_budget_units", 0)),
        },
    )
    base._write_json(
        base_out / "split_listwise.json",
        {
            "train_size": int(len(listwise_train)),
            "eval_size": int(len(listwise_eval)),
            "eval_name": "external_listwise_eval",
            "eval_dataset": str(eval_path),
            "num_labels": int(len(RANKING_LABELS)),
            "ranking_labels": list(RANKING_LABELS),
            "budget_units": int(cfg.budget_units),
            "effective_budget_units": int(selected_stats.get("effective_budget_units", 0)),
        },
    )

    print("\nInitializing shared multitask proxy with 13-way listwise head...")
    pointwise_class_weights = base._compute_pointwise_class_weights(
        pointwise_train,
        num_labels=int(cfg.score_max - cfg.score_min + 1),
        mode=str(cfg.pointwise_class_weight_mode),
        strength=float(cfg.pointwise_class_weight_strength),
    )
    proxy = base.LlamaSharedMultiTaskProxyModel(
        model_path=str(args.llama),
        pointwise_num_labels=int(cfg.score_max - cfg.score_min + 1),
        pairwise_num_labels=int(len(RANKING_LABELS)),
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
        proxy.pointwise_class_weights = torch.tensor(pointwise_class_weights, dtype=torch.float32, device=proxy.device)

    print("\nEvaluating before stage-1...")
    pw_before = base._evaluate_pointwise(proxy, pointwise_val, score_min=int(cfg.score_min))
    lw_before_stage1 = _evaluate_listwise(proxy, listwise_eval)
    base._write_json(base_out / "metrics_pointwise_before_stage1.json", pw_before)
    base._write_json(base_out / "metrics_listwise_before_stage1.json", lw_before_stage1)

    print("\nStage-1 training on selected pointwise data...")
    smoother_stage1 = OnlineGlobalPriorPointwiseSmoother(
        alpha=float(cfg.pointwise_global_smooth_alpha),
        start_step=int(cfg.pointwise_global_smooth_start_step),
        warmup_steps=int(cfg.pointwise_global_smooth_warmup_steps),
        prior=float(cfg.pointwise_global_smooth_prior),
        num_labels=int(cfg.score_max - cfg.score_min + 1),
        trainable_alpha=bool(cfg.pointwise_global_smooth_trainable_alpha),
        alpha_max=float(cfg.pointwise_global_smooth_alpha_max),
        alpha_reg=float(cfg.pointwise_global_smooth_alpha_reg),
        alpha_lr=float(cfg.pointwise_global_smooth_alpha_lr),
    ) if float(cfg.pointwise_global_smooth_alpha) > 0.0 else None
    stage1_stats = _train_pointwise_stage_with_optional_smoothing(
        proxy=proxy,
        examples=pointwise_train,
        epochs=int(cfg.pointwise_epochs),
        batch_size=int(cfg.pointwise_batch_size),
        seed=int(cfg.seed) + 17,
        stage_name="stage1-pointwise",
        smoother=smoother_stage1,
    )
    base._write_json(base_out / "train_stats_stage1_pointwise.json", stage1_stats)

    print("Evaluating after stage-1...")
    pw_after_stage1 = base._evaluate_pointwise(proxy, pointwise_val, score_min=int(cfg.score_min))
    lw_after_stage1 = _evaluate_listwise(proxy, listwise_eval)
    base._write_json(base_out / "metrics_pointwise_after_stage1.json", pw_after_stage1)
    base._write_json(base_out / "metrics_listwise_after_stage1.json", lw_after_stage1)
    base._write_json(base_out / "metrics_listwise_before_stage2.json", lw_after_stage1)

    print("\nStage-2 alternating training on converted listwise + pointwise replay...")
    stage2_stats = _train_stage2_alternating(
        proxy=proxy,
        listwise_examples=listwise_train,
        pointwise_examples=pointwise_train,
        epochs=int(cfg.listwise_epochs),
        listwise_batch_size=int(cfg.listwise_batch_size),
        pointwise_batch_size=int(cfg.pointwise_batch_size),
        pointwise_replay_ratio=int(cfg.stage2_pointwise_replay_ratio),
        seed=int(cfg.seed) + 29,
        stage_name="stage2-listwise-alternating",
        smoother=OnlineGlobalPriorPointwiseSmoother(
            alpha=float(cfg.pointwise_global_smooth_alpha),
            start_step=int(cfg.pointwise_global_smooth_start_step),
            warmup_steps=int(cfg.pointwise_global_smooth_warmup_steps),
            prior=float(cfg.pointwise_global_smooth_prior),
            num_labels=int(cfg.score_max - cfg.score_min + 1),
            trainable_alpha=bool(cfg.pointwise_global_smooth_trainable_alpha),
            alpha_max=float(cfg.pointwise_global_smooth_alpha_max),
            alpha_reg=float(cfg.pointwise_global_smooth_alpha_reg),
            alpha_lr=float(cfg.pointwise_global_smooth_alpha_lr),
        ) if float(cfg.pointwise_global_smooth_alpha) > 0.0 and int(cfg.stage2_pointwise_replay_ratio) > 0 else None,
    )
    base._write_json(base_out / "train_stats_stage2_listwise.json", stage2_stats)

    print("Evaluating after stage-2...")
    pw_after_stage2 = base._evaluate_pointwise(proxy, pointwise_val, score_min=int(cfg.score_min))
    lw_after_stage2 = _evaluate_listwise(proxy, listwise_eval)
    base._write_json(base_out / "metrics_pointwise_after_stage2.json", pw_after_stage2)
    base._write_json(base_out / "metrics_listwise_after_stage2.json", lw_after_stage2)

    summary = {
        "mode": "pointwise3_to_listwise_proxy",
        "dataset_load_stats": load_stats,
        "selection_stats": selected_stats,
        "candidate_triple_selection": candidate_triple_stats,
        "split_by_question": split_info,
        "train_budget": {
            "budget_units": int(cfg.budget_units),
            "effective_budget_units": int(selected_stats.get("effective_budget_units", 0)),
            "train_triples": int(len(train_triples)),
            "train_answers": int(len(pointwise_train)),
            "listwise_train": int(len(listwise_train)),
        },
        "pointwise_training_mode": "proxy",
        "pointwise_loss_type": str(cfg.pointwise_loss_type),
        "pointwise_distance_weight": float(cfg.pointwise_distance_weight),
        "pointwise_class_weight_mode": str(cfg.pointwise_class_weight_mode),
        "pointwise_class_weight_strength": float(cfg.pointwise_class_weight_strength),
        "pointwise_global_smooth": {
            "alpha": float(cfg.pointwise_global_smooth_alpha),
            "start_step": int(cfg.pointwise_global_smooth_start_step),
            "warmup_steps": int(cfg.pointwise_global_smooth_warmup_steps),
            "prior": float(cfg.pointwise_global_smooth_prior),
            "trainable_alpha": bool(cfg.pointwise_global_smooth_trainable_alpha),
            "alpha_max": float(cfg.pointwise_global_smooth_alpha_max),
            "alpha_reg": float(cfg.pointwise_global_smooth_alpha_reg),
            "alpha_lr": float(cfg.pointwise_global_smooth_alpha_lr),
        },
        "pointwise_class_weights": pointwise_class_weights.tolist() if pointwise_class_weights is not None else None,
        "listwise": {
            "eval_dataset": str(eval_path),
            "num_labels": int(len(RANKING_LABELS)),
            "ranking_labels": list(RANKING_LABELS),
            "train": listwise_train_stats,
            "eval": listwise_eval_stats,
        },
        "pointwise_metrics": {
            "before_stage1": pw_before,
            "after_stage1": pw_after_stage1,
            "after_stage2": pw_after_stage2,
        },
        "listwise_metrics": {
            "before_stage1": lw_before_stage1,
            "after_stage1": lw_after_stage1,
            "before_stage2": lw_after_stage1,
            "after_stage2": lw_after_stage2,
        },
        "train_stats": {
            "stage1_pointwise": stage1_stats,
            "stage2_listwise": stage2_stats,
        },
    }
    base._write_json(base_out / "summary.json", summary)
    _write_run_summary(base_out, summary)

    del proxy
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    base._log_memory_usage("finished")


if __name__ == "__main__":
    if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
        v = str(getattr(torch, "__version__", ""))
        m = re.match(r"^(\d+)\.(\d+)", v)
        if m is not None:
            major, minor = int(m.group(1)), int(m.group(2))
            if (major, minor) >= (2, 1):
                os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    main()

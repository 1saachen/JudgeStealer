#!/usr/bin/env python
from __future__ import annotations

"""Pointwise active-learning experiments for continuous Skywork RM scores."""

import argparse
import json
import os
import random
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig

from train_with_selector.train_with_selector.data.skywork_dataset import (
    SkyworkAnswer,
    SkyworkQuestion,
    dataset_stats,
    flatten_answers,
    load_skywork_json,
)
from train_with_selector.train_with_selector.selector.binary_selector import BertBinarySelector

try:
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
except ImportError:  # pragma: no cover
    LoraConfig = TaskType = get_peft_model = prepare_model_for_kbit_training = None


@dataclass(frozen=True)
class RewardScale:
    mean: float
    std: float

    def normalize(self, value: float) -> float:
        return (float(value) - self.mean) / self.std

    def denormalize_array(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(values, dtype=np.float32) * self.std + self.mean


def _reward_scale(questions: Sequence[SkyworkQuestion]) -> RewardScale:
    rewards = np.asarray([answer.reward for answer in flatten_answers(questions)], dtype=np.float64)
    if rewards.size == 0:
        raise ValueError("cannot estimate reward scale from an empty selection")
    scale = RewardScale(mean=float(rewards.mean()), std=float(rewards.std()))
    if scale.std <= 0:
        raise ValueError("selected rewards have zero standard deviation")
    return scale


class SkyworkRegressor:
    def __init__(
        self,
        *,
        model_path: str,
        scale: RewardScale,
        lr: float,
        max_length: int,
        load_in_4bit: bool,
        use_lora: bool,
        gradient_checkpointing: bool,
    ) -> None:
        self.scale = scale
        self.max_length = int(max_length)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.truncation_side = "left"

        quantization_config = None
        if load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            )
        dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else (
            torch.float16 if torch.cuda.is_available() else torch.float32
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            num_labels=1,
            problem_type="regression",
            torch_dtype=dtype,
            quantization_config=quantization_config,
            device_map="auto" if torch.cuda.is_available() else None,
        )
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.config.use_cache = False
        if gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        if use_lora:
            if get_peft_model is None:
                raise RuntimeError("--use-lora requires peft")
            if load_in_4bit:
                self.model = prepare_model_for_kbit_training(self.model)
            self.model = get_peft_model(
                self.model,
                LoraConfig(
                    task_type=TaskType.SEQ_CLS,
                    r=16,
                    lora_alpha=32,
                    lora_dropout=0.05,
                    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                    modules_to_save=["score"],
                    bias="none",
                ),
            )
        params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = AdamW(params, lr=float(lr))
        self.device = next(self.model.parameters()).device
        if hasattr(self.model, "print_trainable_parameters"):
            self.model.print_trainable_parameters()

    def _encode(self, answers: Sequence[SkyworkAnswer]) -> Dict[str, torch.Tensor]:
        encoded = self.tokenizer(
            [str(answer) for answer in answers],
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {key: value.to(self.device) for key, value in encoded.items()}

    def train_answers(
        self,
        answers: Sequence[SkyworkAnswer],
        *,
        epochs: int,
        micro_batch_size: int,
        grad_accum_steps: int,
        seed: int,
        smooth_alpha: float,
        smooth_start_step: int,
        smooth_warmup_steps: int,
    ) -> Dict[str, Any]:
        rng = np.random.default_rng(int(seed))
        losses: List[float] = []
        optimizer_steps = 0
        micro_steps = 0
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        for _ in range(int(epochs)):
            order = rng.permutation(len(answers))
            for start in range(0, len(answers), int(micro_batch_size)):
                ids = order[start : start + int(micro_batch_size)]
                batch_answers = [answers[int(i)] for i in ids]
                hard_targets = torch.tensor(
                    [self.scale.normalize(answer.reward) for answer in batch_answers],
                    dtype=torch.float32,
                    device=self.device,
                )
                if optimizer_steps < int(smooth_start_step):
                    alpha = 0.0
                elif int(smooth_warmup_steps) > 0:
                    alpha = float(smooth_alpha) * min(
                        1.0, (optimizer_steps - int(smooth_start_step) + 1) / float(smooth_warmup_steps)
                    )
                else:
                    alpha = float(smooth_alpha)
                # The standardized global prior is zero, so smoothing is shrinkage toward the train mean.
                targets = hard_targets * (1.0 - alpha)
                predictions = self.model(**self._encode(batch_answers)).logits.float().squeeze(-1)
                loss = F.mse_loss(predictions, targets)
                (loss / int(grad_accum_steps)).backward()
                losses.append(float(loss.detach().cpu()))
                micro_steps += 1
                if micro_steps % int(grad_accum_steps) == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    optimizer_steps += 1
        if micro_steps % int(grad_accum_steps) != 0:
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            optimizer_steps += 1
        return {
            "answers": len(answers),
            "epochs": int(epochs),
            "micro_steps": micro_steps,
            "optimizer_steps": optimizer_steps,
            "mean_loss": float(np.mean(losses)) if losses else float("nan"),
            "smooth_alpha": float(smooth_alpha),
            "smooth_prior_normalized": 0.0,
        }

    @torch.no_grad()
    def predict(self, answers: Sequence[SkyworkAnswer], *, batch_size: int) -> np.ndarray:
        self.model.eval()
        normalized: List[np.ndarray] = []
        for start in range(0, len(answers), int(batch_size)):
            batch = answers[start : start + int(batch_size)]
            pred = self.model(**self._encode(batch)).logits.float().squeeze(-1)
            normalized.append(pred.detach().cpu().numpy())
        values = np.concatenate(normalized) if normalized else np.zeros((0,), dtype=np.float32)
        return self.scale.denormalize_array(values)

    @torch.no_grad()
    def predict_mc(
        self, answers: Sequence[SkyworkAnswer], *, batch_size: int, samples: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return MC-dropout predictive mean/std on the original reward scale."""
        if int(samples) <= 0:
            raise ValueError("MC-dropout samples must be > 0")
        draws: List[np.ndarray] = []
        # LoRA dropout remains stochastic here while gradients stay disabled.
        self.model.train()
        for _ in range(int(samples)):
            normalized: List[np.ndarray] = []
            for start in range(0, len(answers), int(batch_size)):
                batch = answers[start : start + int(batch_size)]
                pred = self.model(**self._encode(batch)).logits.float().squeeze(-1)
                normalized.append(pred.detach().cpu().numpy())
            draws.append(np.concatenate(normalized))
        stacked = np.stack(draws, axis=0).astype(np.float32)
        return (
            self.scale.denormalize_array(stacked.mean(axis=0)),
            stacked.std(axis=0) * self.scale.std,
        )


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _seed_everything(seed: int) -> None:
    os.environ.setdefault("PYTHONHASHSEED", str(int(seed)))
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return ranks


def _correlation(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or float(np.std(a)) == 0.0 or float(np.std(b)) == 0.0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _metrics(answers: Sequence[SkyworkAnswer], predictions: np.ndarray, scale: RewardScale) -> Dict[str, Any]:
    truth = np.asarray([answer.reward for answer in answers], dtype=np.float64)
    pred = np.asarray(predictions, dtype=np.float64)
    errors = pred - truth
    sse = float(np.square(errors).sum())
    sst = float(np.square(truth - truth.mean()).sum())
    return {
        "n": len(answers),
        "mae": float(np.abs(errors).mean()),
        "rmse": float(np.sqrt(np.square(errors).mean())),
        "r2": float(1.0 - sse / sst) if sst > 0 else float("nan"),
        "pearson": _correlation(truth, pred),
        "spearman": _correlation(_rankdata(truth), _rankdata(pred)),
        "within_0_5_std": float((np.abs(errors) <= 0.5 * scale.std).mean()),
        "within_1_std": float((np.abs(errors) <= scale.std).mean()),
    }


def _question_errors(
    proxy: SkyworkRegressor, questions: Sequence[SkyworkQuestion], *, batch_size: int
) -> np.ndarray:
    answers = flatten_answers(questions)
    predictions = proxy.predict(answers, batch_size=batch_size)
    errors = np.abs(predictions - np.asarray([answer.reward for answer in answers], dtype=np.float32))
    return errors.reshape(len(questions), 3).mean(axis=1).astype(np.float32)


def _answer_errors(
    proxy: SkyworkRegressor, answers: Sequence[SkyworkAnswer], *, batch_size: int
) -> np.ndarray:
    predictions = proxy.predict(answers, batch_size=batch_size)
    truth = np.asarray([answer.reward for answer in answers], dtype=np.float32)
    return np.abs(predictions - truth).astype(np.float32)


def _normalize_targets(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    lo, hi = float(values.min()), float(values.max())
    return np.full(values.shape, 0.5, dtype=np.float32) if hi <= lo else (values - lo) / (hi - lo)


def _proxy_acquisition_scores(
    proxy: SkyworkRegressor,
    questions: Sequence[SkyworkQuestion],
    *,
    batch_size: int,
    mc_samples: int,
    uncertainty_weight: float,
    response_std_weight: float,
) -> Tuple[np.ndarray, Dict[str, float]]:
    answers = flatten_answers(questions)
    means, uncertainties = proxy.predict_mc(
        answers, batch_size=batch_size, samples=mc_samples
    )
    mean_matrix = means.reshape(len(questions), 3)
    uncertainty_matrix = uncertainties.reshape(len(questions), 3)
    question_uncertainty = (
        0.75 * uncertainty_matrix.mean(axis=1) + 0.25 * uncertainty_matrix.max(axis=1)
    )
    response_std = mean_matrix.std(axis=1)
    uncertainty_norm = _normalize_targets(question_uncertainty)
    response_std_norm = _normalize_targets(response_std)
    denom = max(float(uncertainty_weight) + float(response_std_weight), 1e-12)
    scores = (
        float(uncertainty_weight) * uncertainty_norm
        + float(response_std_weight) * response_std_norm
    ) / denom
    return scores.astype(np.float32), {
        "pool_score_mean": float(scores.mean()),
        "pool_score_std": float(scores.std()),
        "pool_mc_uncertainty_mean": float(question_uncertainty.mean()),
        "pool_predicted_response_std_mean": float(response_std.mean()),
    }


def _answer_proxy_scores_from_predictions(
    answers: Sequence[SkyworkAnswer], means: np.ndarray, uncertainties: np.ndarray,
    *, uncertainty_weight: float, response_std_weight: float,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Combine per-answer uncertainty with within-question prediction spread."""
    means = np.asarray(means, dtype=np.float32)
    uncertainties = np.asarray(uncertainties, dtype=np.float32)
    if means.shape != (len(answers),) or uncertainties.shape != (len(answers),):
        raise ValueError("answer prediction arrays must match the answer count")

    means_by_question: Dict[int, List[float]] = {}
    for answer, mean in zip(answers, means.tolist()):
        means_by_question.setdefault(answer.question_id, []).append(float(mean))
    std_by_question = {
        question_id: float(np.std(values))
        for question_id, values in means_by_question.items()
    }
    response_std = np.asarray(
        [std_by_question[answer.question_id] for answer in answers], dtype=np.float32
    )
    uncertainty_norm = _normalize_targets(uncertainties)
    response_std_norm = _normalize_targets(response_std)
    denom = max(float(uncertainty_weight) + float(response_std_weight), 1e-12)
    scores = (
        float(uncertainty_weight) * uncertainty_norm
        + float(response_std_weight) * response_std_norm
    ) / denom
    return scores.astype(np.float32), {
        "pool_score_mean": float(scores.mean()),
        "pool_score_std": float(scores.std()),
        "pool_mc_uncertainty_mean": float(uncertainties.mean()),
        "pool_predicted_response_std_mean": float(response_std.mean()),
    }


def _proxy_answer_acquisition_scores(
    proxy: SkyworkRegressor, answers: Sequence[SkyworkAnswer], *, batch_size: int,
    mc_samples: int, uncertainty_weight: float, response_std_weight: float,
) -> Tuple[np.ndarray, Dict[str, float]]:
    means, uncertainties = proxy.predict_mc(
        answers, batch_size=batch_size, samples=mc_samples
    )
    return _answer_proxy_scores_from_predictions(
        answers, means, uncertainties,
        uncertainty_weight=uncertainty_weight,
        response_std_weight=response_std_weight,
    )


def _select_questions(
    *,
    train_questions: Sequence[SkyworkQuestion],
    args: argparse.Namespace,
    out: Path,
) -> Tuple[List[SkyworkQuestion], Dict[str, Any], RewardScale]:
    max_questions = len(train_questions) if args.budget_units == 0 else min(
        len(train_questions), int(args.budget_units) // 3
    )
    if max_questions <= 0:
        raise ValueError("budget-units must be 0 or at least 3")
    by_id = {question.question_id: question for question in train_questions}
    if args.fixed_selected_questions:
        rows = [json.loads(line) for line in Path(args.fixed_selected_questions).read_text().splitlines() if line.strip()]
        ids = [int(row["question_id"]) for row in rows]
        missing = [qid for qid in ids if qid not in by_id]
        if missing:
            raise ValueError(f"fixed selection contains ids outside the training split: {missing[:5]}")
        if len(set(ids)) != len(ids):
            raise ValueError("fixed selection contains duplicate question ids")
        selected = [by_id[qid] for qid in ids[:max_questions]]
        return selected, {"mode": "fixed", "source": args.fixed_selected_questions}, _reward_scale(selected)

    rng = np.random.default_rng(int(args.seed) + 101)
    remaining = list(train_questions)
    selected: List[SkyworkQuestion] = []
    rows: List[Dict[str, Any]] = []
    if args.selection_mode == "random":
        picked = rng.choice(len(remaining), size=max_questions, replace=False).tolist()
        selected = [remaining[int(i)] for i in picked]
        for question in selected:
            rows.append({"stage": "random", "question_id": question.question_id, "queried_answers": 3})
        _write_jsonl(out / "selected_questions.jsonl", rows)
        return selected, {"mode": "random", "selected_questions": len(selected)}, _reward_scale(selected)

    init_count = min(max_questions, int(args.selector_init_questions))
    init_ids = sorted(rng.choice(len(remaining), size=init_count, replace=False).tolist(), reverse=True)
    init_batch = [remaining[i] for i in init_ids]
    scale = _reward_scale(init_batch)

    proxy = SkyworkRegressor(
        model_path=args.llama,
        scale=scale,
        lr=args.proxy_lr,
        max_length=args.proxy_max_length,
        load_in_4bit=args.load_in_4bit,
        use_lora=args.use_lora,
        gradient_checkpointing=True,
    )
    selector = None
    if args.selection_mode == "selector":
        selector = BertBinarySelector(
            model_name=args.selector_model,
            max_length=args.selector_max_length,
            head_hidden_dim=512,
            lr=args.selector_lr,
            freeze_bert=not args.selector_unfreeze,
            unfreeze_last_n_layers=args.selector_unfreeze_last_n_layers,
        )
    for i in init_ids:
        remaining.pop(i)
    selected.extend(init_batch)
    proxy.train_answers(
        flatten_answers(init_batch), epochs=args.selector_proxy_warmup_epochs,
        micro_batch_size=args.train_micro_batch_size, grad_accum_steps=args.grad_accum_steps,
        seed=args.seed + 301, smooth_alpha=args.smooth_alpha,
        smooth_start_step=args.smooth_start_step,
        smooth_warmup_steps=args.smooth_warmup_steps,
    )
    if selector is not None:
        targets = _normalize_targets(_question_errors(proxy, init_batch, batch_size=args.eval_batch_size))
        selector.update(init_batch, targets, epochs=args.selector_epochs, batch_size=args.selector_batch_size)
    else:
        targets = np.zeros((len(init_batch),), dtype=np.float32)
    for question, target in zip(init_batch, targets.tolist()):
        rows.append({
            "stage": "init", "question_id": question.question_id, "queried_answers": 3,
            "target": target if selector is not None else None,
            "acquisition_source": "random_init",
        })

    round_index = 0
    while len(selected) < max_questions and remaining:
        round_index += 1
        pool = remaining
        if args.selector_max_score_candidates > 0 and len(pool) > args.selector_max_score_candidates:
            pool_ids = rng.choice(len(pool), size=args.selector_max_score_candidates, replace=False).tolist()
            pool = [pool[int(i)] for i in pool_ids]
        need = min(args.selector_batch_size, max_questions - len(selected), len(pool))
        acquisition_source: Dict[int, str] = {}
        diagnostics: Dict[str, float] = {}
        if args.selection_mode == "proxy":
            scores, diagnostics = _proxy_acquisition_scores(
                proxy, pool, batch_size=args.eval_batch_size,
                mc_samples=args.proxy_mc_samples,
                uncertainty_weight=args.proxy_uncertainty_weight,
                response_std_weight=args.proxy_response_std_weight,
            )
            explore_count = min(need, int(round(need * args.proxy_exploration_ratio)))
            exploit_count = need - explore_count
            exploit_order = np.argsort(scores)[-exploit_count:][::-1].tolist() if exploit_count else []
            exploit_ids = {pool[int(i)].question_id for i in exploit_order}
            explore_pool = [i for i, question in enumerate(pool) if question.question_id not in exploit_ids]
            explore_order = (
                rng.choice(explore_pool, size=explore_count, replace=False).tolist()
                if explore_count else []
            )
            picked_order = exploit_order + [int(i) for i in explore_order]
            for i in exploit_order:
                acquisition_source[pool[int(i)].question_id] = "pointwise_proxy"
            for i in explore_order:
                acquisition_source[pool[int(i)].question_id] = "random_exploration"
        else:
            if selector is None:
                raise RuntimeError("BERT selector was not initialized")
            scores = selector.score(pool)
            picked_order = np.argsort(scores)[-need:][::-1].tolist()
            for i in picked_order:
                acquisition_source[pool[int(i)].question_id] = "bert_selector"
        batch = [pool[int(i)] for i in picked_order]
        if selector is not None:
            errors = _question_errors(proxy, batch, batch_size=args.eval_batch_size)
            targets = _normalize_targets(errors)
            selector.update(batch, targets, epochs=args.selector_epochs, batch_size=args.selector_batch_size)
        else:
            errors = np.full((len(batch),), np.nan, dtype=np.float32)
            targets = np.zeros((len(batch),), dtype=np.float32)
        proxy.train_answers(
            flatten_answers(batch), epochs=args.selector_proxy_update_epochs,
            micro_batch_size=args.train_micro_batch_size, grad_accum_steps=args.grad_accum_steps,
            seed=args.seed + 301 + round_index, smooth_alpha=args.smooth_alpha,
            smooth_start_step=args.smooth_start_step,
            smooth_warmup_steps=args.smooth_warmup_steps,
        )
        picked_ids = {question.question_id for question in batch}
        remaining = [question for question in remaining if question.question_id not in picked_ids]
        selected.extend(batch)
        score_by_id = {question.question_id: float(scores[int(i)]) for i, question in enumerate(pool)}
        for question, target, error in zip(batch, targets.tolist(), errors.tolist()):
            rows.append({
                "stage": f"selector_round_{round_index}", "question_id": question.question_id,
                "queried_answers": 3, "selector_score": score_by_id[question.question_id],
                "target": target if selector is not None else None,
                "pre_update_mae": error if selector is not None else None,
                "acquisition_source": acquisition_source[question.question_id],
                "acquisition_diagnostics": diagnostics,
            })
        print(f"[selector] round={round_index} selected={len(selected)}/{max_questions}", flush=True)
    _write_jsonl(out / "selected_questions.jsonl", rows)
    del proxy, selector
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return selected, {
        "mode": args.selection_mode, "selected_questions": len(selected), "selected_answers": len(selected) * 3,
        "init_questions": init_count, "rounds": round_index, "reward_scale_source": "random_init_questions",
        "proxy_mc_samples": args.proxy_mc_samples if args.selection_mode == "proxy" else None,
        "proxy_uncertainty_weight": args.proxy_uncertainty_weight if args.selection_mode == "proxy" else None,
        "proxy_response_std_weight": args.proxy_response_std_weight if args.selection_mode == "proxy" else None,
        "proxy_exploration_ratio": args.proxy_exploration_ratio if args.selection_mode == "proxy" else None,
    }, scale


def _select_answers(
    *, train_questions: Sequence[SkyworkQuestion], args: argparse.Namespace, out: Path
) -> Tuple[List[SkyworkAnswer], Dict[str, Any], RewardScale]:
    """Select individual answers rather than binding all three answers per question."""
    all_answers = flatten_answers(train_questions)
    budget = len(all_answers) if args.budget_units == 0 else min(len(all_answers), int(args.budget_units))
    if budget <= 0:
        raise ValueError("budget-units must be 0 or positive")
    rng = np.random.default_rng(int(args.seed) + 101)
    if args.selection_mode == "random_answer":
        picked = rng.choice(len(all_answers), size=budget, replace=False).tolist()
        selected = [all_answers[int(i)] for i in picked]
        _write_jsonl(out / "selected_answers.jsonl", [
            {"stage": "random", "question_id": a.question_id, "answer_key": a.answer_key}
            for a in selected
        ])
        return selected, {"mode": "random_answer", "selected_answers": len(selected)}, _reward_scale(
            [SkyworkQuestion(a.question_id, a.source_id, a.dataset, a.instruction, a.input_text, (a, a, a)) for a in selected]
        )

    init_count = min(budget, int(args.selector_init_questions))
    init_ids = rng.choice(len(all_answers), size=init_count, replace=False).tolist()
    init_batch = [all_answers[int(i)] for i in init_ids]
    scale = RewardScale(
        mean=float(np.mean([a.reward for a in init_batch])),
        std=float(np.std([a.reward for a in init_batch])),
    )
    if scale.std <= 0:
        raise ValueError("answer-level initialization has zero reward standard deviation")
    proxy = SkyworkRegressor(
        model_path=args.llama, scale=scale, lr=args.proxy_lr, max_length=args.proxy_max_length,
        load_in_4bit=args.load_in_4bit, use_lora=args.use_lora, gradient_checkpointing=True,
    )
    selector = None
    if args.selection_mode == "selector_answer":
        selector = BertBinarySelector(
            model_name=args.selector_model, max_length=args.selector_max_length, head_hidden_dim=512,
            lr=args.selector_lr, freeze_bert=not args.selector_unfreeze,
            unfreeze_last_n_layers=args.selector_unfreeze_last_n_layers,
        )
    selected = list(init_batch)
    remaining = [a for i, a in enumerate(all_answers) if i not in set(init_ids)]
    proxy.train_answers(init_batch, epochs=args.selector_proxy_warmup_epochs,
        micro_batch_size=args.train_micro_batch_size, grad_accum_steps=args.grad_accum_steps,
        seed=args.seed + 301, smooth_alpha=args.smooth_alpha,
        smooth_start_step=args.smooth_start_step, smooth_warmup_steps=args.smooth_warmup_steps)
    if selector is not None:
        errors = _answer_errors(proxy, init_batch, batch_size=args.eval_batch_size)
        selector.update(
            init_batch, _normalize_targets(errors), epochs=args.selector_epochs,
            batch_size=args.selector_batch_size,
        )
    rows = [
        {
            "stage": "init", "question_id": a.question_id, "answer_key": a.answer_key,
            "acquisition_source": "random_init",
        }
        for a in init_batch
    ]
    round_index = 0
    while len(selected) < budget and remaining:
        round_index += 1
        pool = remaining
        if args.selector_max_score_candidates > 0 and len(pool) > args.selector_max_score_candidates:
            ids = rng.choice(len(pool), size=args.selector_max_score_candidates, replace=False).tolist()
            pool = [pool[int(i)] for i in ids]
        need = min(args.selector_batch_size, budget - len(selected), len(pool))
        acquisition_source: Dict[int, str] = {}
        diagnostics: Dict[str, float] = {}
        if args.selection_mode == "proxy_answer":
            scores, diagnostics = _proxy_answer_acquisition_scores(
                proxy, pool, batch_size=args.eval_batch_size,
                mc_samples=args.proxy_mc_samples,
                uncertainty_weight=args.proxy_uncertainty_weight,
                response_std_weight=args.proxy_response_std_weight,
            )
            explore_count = min(need, int(round(need * args.proxy_exploration_ratio)))
            exploit_count = need - explore_count
            exploit_order = np.argsort(scores)[-exploit_count:][::-1].tolist() if exploit_count else []
            exploit_ids = {id(pool[int(i)]) for i in exploit_order}
            explore_pool = [i for i, answer in enumerate(pool) if id(answer) not in exploit_ids]
            explore_order = (
                rng.choice(explore_pool, size=explore_count, replace=False).tolist()
                if explore_count else []
            )
            picked_order = exploit_order + [int(i) for i in explore_order]
            for i in exploit_order:
                acquisition_source[id(pool[int(i)])] = "pointwise_proxy"
            for i in explore_order:
                acquisition_source[id(pool[int(i)])] = "random_exploration"
        else:
            if selector is None:
                raise RuntimeError("BERT answer selector was not initialized")
            scores = selector.score(pool)
            picked_order = np.argsort(scores)[-need:][::-1].tolist()
            for i in picked_order:
                acquisition_source[id(pool[int(i)])] = "bert_selector"
        batch = [pool[int(i)] for i in picked_order]
        if selector is not None:
            errors = _answer_errors(proxy, batch, batch_size=args.eval_batch_size)
            selector.update(
                batch, _normalize_targets(errors), epochs=args.selector_epochs,
                batch_size=args.selector_batch_size,
            )
        else:
            errors = np.full((len(batch),), np.nan, dtype=np.float32)
        proxy.train_answers(batch, epochs=args.selector_proxy_update_epochs,
            micro_batch_size=args.train_micro_batch_size, grad_accum_steps=args.grad_accum_steps,
            seed=args.seed + 301 + round_index, smooth_alpha=args.smooth_alpha,
            smooth_start_step=args.smooth_start_step, smooth_warmup_steps=args.smooth_warmup_steps)
        picked = {id(a) for a in batch}
        remaining = [a for a in remaining if id(a) not in picked]
        selected.extend(batch)
        rows.extend({
            "stage": f"selector_round_{round_index}", "question_id": a.question_id,
            "answer_key": a.answer_key, "selector_score": float(scores[int(i)]),
            "pre_update_mae": float(e) if selector is not None else None,
            "acquisition_source": acquisition_source[id(a)],
            "acquisition_diagnostics": diagnostics,
        } for a, i, e in zip(batch, picked_order, errors.tolist()))
        print(f"[{args.selection_mode}] round={round_index} selected={len(selected)}/{budget}", flush=True)
    _write_jsonl(out / "selected_answers.jsonl", rows)
    del proxy, selector
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return selected, {"mode": args.selection_mode, "selected_answers": len(selected),
                      "init_answers": init_count, "rounds": round_index,
                      "reward_scale_source": "random_init_answers",
                      "proxy_mc_samples": args.proxy_mc_samples if args.selection_mode == "proxy_answer" else None,
                      "proxy_uncertainty_weight": args.proxy_uncertainty_weight if args.selection_mode == "proxy_answer" else None,
                      "proxy_response_std_weight": args.proxy_response_std_weight if args.selection_mode == "proxy_answer" else None,
                      "proxy_exploration_ratio": args.proxy_exploration_ratio if args.selection_mode == "proxy_answer" else None}, scale


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Skywork continuous-reward pointwise experiment")
    data_dir = ROOT / "train_with_selector/train_with_selector/data/rewardmodel/skywork"
    parser.add_argument("--train-dataset", default=str(data_dir / "train-18k-skywork.json"))
    parser.add_argument("--val-dataset", default=str(data_dir / "val-2k-skywork.json"))
    parser.add_argument("--llama", default=str(ROOT / "qwen/Qwen3-0.6B"))
    parser.add_argument("--out", default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--budget-units", type=int, default=600, help="Answer-label budget; each question costs 3")
    parser.add_argument("--selection-mode", choices=["random", "selector", "proxy", "random_answer", "selector_answer", "proxy_answer"], default="selector")
    parser.add_argument("--fixed-selected-questions", default="")
    parser.add_argument("--pointwise-epochs", type=int, default=1)
    parser.add_argument("--train-micro-batch-size", type=int, default=2)
    parser.add_argument("--grad-accum-steps", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--proxy-lr", type=float, default=1e-4)
    parser.add_argument("--proxy-max-length", type=int, default=1024)
    parser.add_argument("--load-in-4bit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-lora", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--smooth-alpha", type=float, default=0.0)
    parser.add_argument("--smooth-start-step", type=int, default=20)
    parser.add_argument("--smooth-warmup-steps", type=int, default=20)
    parser.add_argument("--selector-model", default=str(ROOT / "models/longformer-base-4096"))
    parser.add_argument("--selector-max-length", type=int, default=4096)
    parser.add_argument("--selector-lr", type=float, default=1e-3)
    parser.add_argument("--selector-init-questions", type=int, default=80)
    parser.add_argument("--selector-batch-size", type=int, default=20)
    parser.add_argument("--selector-epochs", type=int, default=4)
    parser.add_argument("--selector-max-score-candidates", type=int, default=4096)
    parser.add_argument("--selector-proxy-warmup-epochs", type=int, default=3)
    parser.add_argument("--selector-proxy-update-epochs", type=int, default=1)
    parser.add_argument("--proxy-mc-samples", type=int, default=4)
    parser.add_argument("--proxy-uncertainty-weight", type=float, default=0.5)
    parser.add_argument("--proxy-response-std-weight", type=float, default=0.5)
    parser.add_argument("--proxy-exploration-ratio", type=float, default=0.1)
    parser.add_argument("--selector-unfreeze", action="store_true")
    parser.add_argument("--selector-unfreeze-last-n-layers", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.smooth_alpha < 1.0:
        raise ValueError("smooth-alpha must be in [0, 1)")
    if args.budget_units != 0 and args.budget_units % 3 != 0:
        raise ValueError("budget-units must be divisible by 3 because a question has three answers")
    for name in ("pointwise_epochs", "train_micro_batch_size", "grad_accum_steps", "eval_batch_size"):
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"{name.replace('_', '-')} must be > 0")
    if args.selection_mode in {"selector", "proxy", "selector_answer", "proxy_answer"}:
        for name in (
            "selector_init_questions",
            "selector_batch_size",
            "selector_epochs",
            "selector_proxy_warmup_epochs",
            "selector_proxy_update_epochs",
        ):
            if int(getattr(args, name)) <= 0:
                raise ValueError(f"{name.replace('_', '-')} must be > 0")
    if args.selection_mode in {"proxy", "proxy_answer"}:
        if args.proxy_mc_samples <= 0:
            raise ValueError("proxy-mc-samples must be > 0")
        if args.proxy_uncertainty_weight < 0 or args.proxy_response_std_weight < 0:
            raise ValueError("proxy acquisition weights must be >= 0")
        if args.proxy_uncertainty_weight + args.proxy_response_std_weight <= 0:
            raise ValueError("at least one proxy acquisition weight must be > 0")
        if not 0.0 <= args.proxy_exploration_ratio <= 1.0:
            raise ValueError("proxy-exploration-ratio must be in [0, 1]")
    _seed_everything(args.seed)
    out = Path(args.out) if args.out else ROOT / "outputs/skywork_pointwise" / (
        f"{args.selection_mode}_b{args.budget_units}_a{args.smooth_alpha}_{datetime.now():%Y%m%d_%H%M%S}"
    )
    out.mkdir(parents=True, exist_ok=False)
    _write_json(out / "config.json", vars(args))

    train_questions = load_skywork_json(args.train_dataset)
    val_questions = load_skywork_json(args.val_dataset)
    train_ids = {question.question_id for question in train_questions}
    val_ids = {question.question_id for question in val_questions}
    overlap = train_ids & val_ids
    if overlap:
        raise ValueError(f"train/validation question-id overlap: {sorted(overlap)[:5]}")
    split = {
        "mode": "fixed_explicit_files",
        "train_dataset": str(args.train_dataset),
        "validation_dataset": str(args.val_dataset),
        "train": dataset_stats(train_questions), "validation": dataset_stats(val_questions),
        "validation_question_ids": [q.question_id for q in val_questions],
    }
    _write_json(out / "split.json", split)
    print(f"Loaded questions: train={len(train_questions)} val={len(val_questions)}", flush=True)

    if args.selection_mode in {"random_answer", "selector_answer", "proxy_answer"}:
        selected, selection_stats, scale = _select_answers(
            train_questions=train_questions, args=args, out=out
        )
        selected_answers = selected
    else:
        selected, selection_stats, scale = _select_questions(
            train_questions=train_questions, args=args, out=out
        )
        selected_answers = flatten_answers(selected)
    selection_stats["reward_scale"] = asdict(scale)
    selection_stats.setdefault("reward_scale_source", "selected_questions")
    if args.fixed_selected_questions:
        _write_jsonl(out / "selected_questions.jsonl", [
            {"stage": "fixed", "question_id": q.question_id, "queried_answers": 3} for q in selected
        ])
    proxy = SkyworkRegressor(
        model_path=args.llama, scale=scale, lr=args.proxy_lr, max_length=args.proxy_max_length,
        load_in_4bit=args.load_in_4bit, use_lora=args.use_lora, gradient_checkpointing=True,
    )
    val_answers = flatten_answers(val_questions)
    before_predictions = proxy.predict(val_answers, batch_size=args.eval_batch_size)
    before = _metrics(val_answers, before_predictions, scale)
    started = time.time()
    train_stats = proxy.train_answers(
        selected_answers, epochs=args.pointwise_epochs, micro_batch_size=args.train_micro_batch_size,
        grad_accum_steps=args.grad_accum_steps, seed=args.seed + 17,
        smooth_alpha=args.smooth_alpha, smooth_start_step=args.smooth_start_step,
        smooth_warmup_steps=args.smooth_warmup_steps,
    )
    predictions = proxy.predict(val_answers, batch_size=args.eval_batch_size)
    after = _metrics(val_answers, predictions, scale)
    prediction_rows = [
        {
            "question_id": answer.question_id, "answer_key": answer.answer_key, "model": answer.model,
            "true_reward": answer.reward, "predicted_reward": float(prediction),
        }
        for answer, prediction in zip(val_answers, predictions.tolist())
    ]
    _write_jsonl(out / "validation_predictions.jsonl", prediction_rows)
    summary = {
        "mode": "skywork_pointwise_regression", "selection": selection_stats,
        "reward_scale": asdict(scale), "metrics_before": before, "metrics_after": after,
        "training": train_stats, "elapsed_sec": time.time() - started,
    }
    _write_json(out / "summary.json", summary)
    _write_json(out / "metrics_compact.json", after)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f"Outputs: {out}", flush=True)


if __name__ == "__main__":
    main()

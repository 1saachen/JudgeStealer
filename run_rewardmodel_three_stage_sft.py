#!/usr/bin/env python3
"""Continuous-reward generative SFT: pointwise -> pairwise -> listwise.

The final model is always a causal LM trained with SFT.  In selector mode, a
temporary regression proxy is used only to acquire question IDs; its head is
never reused as the trained or evaluated model.

Pairwise/listwise ties use a soft sequence target by default: each tied
winner receives equal probability, while unique winners remain one-hot.
``--tie-policy unique_only`` is the matched unique-winner ablation.
"""

from __future__ import annotations

import argparse
import gc
import itertools
import json
import math
import os
import random
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

import run_skywork_pointwise as rm
import run_newnew_one_answer_trueval_three_stage_sft as controls
import run_pointwise5answers_three_stage_pairwise_listwise_sft_v1 as three
import run_pointwise5answers_three_to_listwise_v1 as lw
import run_pointwise5answers_two_to_pairwise_v1 as base
from train_with_selector.train_with_selector.data.skywork_dataset import (
    SkyworkAnswer,
    SkyworkQuestion,
    flatten_answers,
    load_skywork_json,
)


CONTINUOUS_JUDGE_PROMPT = """You are an impartial judge evaluating AI assistant responses.

Evaluate helpfulness, relevance, accuracy, depth, creativity, and level of detail.
Give ONLY one numeric reward from 1 to 5. Decimal values are allowed.
Use this exact format: Score: [X]"""

UNIRRM_SYSTEM_PROMPT = """You are a multilingual evaluation expert, responsible for conducting rigorous, objective, and multi-dimensional evaluations of responses generated for User Input.

Your evaluation must strictly follow the step-by-step process outlined below:

### Phase 1: Deep Analysis
Before evaluating, perform a comprehensive analysis of the User Input to establish a robust baseline:
1. **Identify potential risks**: Analyze the User Input to identify any potential safety, legal, offensive, or ethical risks.
2. **Identify task type**: Identify the primary task type (e.g., chat, reasoning, code generation, translation, or creative writing).
3. **Analyze core requirements (task-dependent)**: Define the fundamental evaluation dimensions that any correct response must satisfy.
4. **Analyze specific requirements**: Identify additional constraints or expectations unique to the User Input.
5. **Predict response content**: Summarize the expected content or core objectives of a correct response.

### Phase 2: Dynamic Rubric Generation
1. Generate a set of evaluation rubrics tailored to the user inputs and responses, with a 1-5 scoring criterion for each rubric.
2. If any safety, legal, or ethical risks are detected, include a Safety rubric as the highest-priority dimension.
3. Ensure rubrics comprehensively cover all critical aspects of the response.

### Phase 3: Detailed Evaluation
For each rubric, evaluate the response:
1. **Evidence Extraction**: Identify specific passages that meet or fail to meet the rubric requirements.
2. **Gap Analysis**: Determine why the response did not achieve a perfect score (5).
3. **Scoring**: Assign a score from 1 to 5.

### OUTPUT FORMAT
{
  "Analysis_process": "Concise summary of the analysis.",
  "rubrics": [{"name": "String", "description": "Rubric definition"}],
  "evaluations": [{"response_id": "String", "explanation": "Summary", "final_score": "Float"}],
  "best_id": "ID of the winner"
}

Return only valid JSON."""


def _converted_unirrm_system_prompt(output_instruction: str) -> str:
    """Keep UniRRM's evaluation protocol while changing only the label format."""
    protocol = UNIRRM_SYSTEM_PROMPT.split("### OUTPUT FORMAT", 1)[0].rstrip()
    return f"{protocol}\n\n### OUTPUT FORMAT\n{str(output_instruction).strip()}"


CONVERTED_POINTWISE_SYSTEM_PROMPT = _converted_unirrm_system_prompt(
    """Evaluate the single response using the procedure above.
Return only one numeric reward in this exact format: Score: [X]
X must be a number from 1 to 5. Decimal values are allowed.
Do not output JSON, explanations, rubrics, or analysis."""
)

CONVERTED_PAIRWISE_SYSTEM_PROMPT = _converted_unirrm_system_prompt(
    """Evaluate the two responses using the procedure above.
Return only one label: [[1]], [[2]], or [[3]].
[[1]] means Response1 is better, [[2]] means Response2 is better, and [[3]] means tie.
Do not output JSON, explanations, rubrics, or analysis."""
)

CONVERTED_LISTWISE_SYSTEM_PROMPT = _converted_unirrm_system_prompt(
    """Evaluate the three responses using the procedure above.
Return only one integer: 1, 2, or 3.
The integer identifies the best response in its displayed order.
Do not output JSON, explanations, rubrics, response names, or analysis."""
)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def _format_reward(value: float, *, decimals: int) -> str:
    text = f"{float(value):.{int(decimals)}f}".rstrip("0").rstrip(".")
    return text if text else "0"


def _smoothed_reward(value: float, *, mean: float, alpha: float) -> float:
    return (1.0 - float(alpha)) * float(value) + float(alpha) * float(mean)


def _unirm_user_prompt(instruction: str, input_text: str, outputs: Sequence[str]) -> str:
    question = str(instruction).strip()
    if str(input_text).strip():
        question += f"\n\nInput:\n{str(input_text).strip()}"
    blocks = ["<User_Input>", question, "</User_Input>", ""]
    for index, output in enumerate(outputs, start=1):
        blocks.extend(
            [f"<Response{index}>", str(output).strip(), f"</Response{index}>", ""]
        )
    return "\n".join(blocks).strip()


def _unirm_prompt(
    *, instruction: str, input_text: str, outputs: Sequence[str]
) -> str:
    return "\n".join(
        [
            "### System",
            UNIRRM_SYSTEM_PROMPT,
            "",
            "### User",
            _unirm_user_prompt(instruction, input_text, outputs),
            "",
            "### Assistant",
        ]
    )


def _converted_unirrm_prompt(
    *,
    system_prompt: str,
    instruction: str,
    input_text: str,
    outputs: Sequence[str],
    assistant_suffix: str = "",
) -> str:
    parts = [
        "### System",
        str(system_prompt).strip(),
        "",
        "### User",
        _unirm_user_prompt(instruction, input_text, outputs),
        "",
        "### Assistant",
    ]
    if assistant_suffix:
        parts.append(str(assistant_suffix))
    return "\n".join(parts)


def _best_choice_prompt(record: Mapping[str, Any]) -> str:
    return _converted_unirrm_prompt(
        system_prompt=CONVERTED_LISTWISE_SYSTEM_PROMPT,
        instruction=str(record.get("instruction", "")),
        input_text=str(record.get("input", "")),
        outputs=[str(record.get(f"output{letter}", "")) for letter in "ABC"],
    )


def _unirm_target(scores: Sequence[float], best_id: str, *, decimals: int) -> str:
    evaluations = [
        {
            "response_id": f"Response{index}",
            "explanation": "",
            "final_score": float(_format_reward(score, decimals=decimals)),
        }
        for index, score in enumerate(scores, start=1)
    ]
    target = {
        "Analysis_process": "",
        "rubrics": [],
        "evaluations": evaluations,
        "best_id": str(best_id),
    }
    return json.dumps(target, ensure_ascii=False, separators=(",", ":")) + base.DEFAULT_EOS_TOKEN


def _best_response_id(choice: Any, labels: Sequence[str]) -> str:
    text = str(choice or "").strip()
    if text.lower() == "tie":
        return "Tie"
    if text in labels:
        return f"Response{labels.index(text) + 1}"
    match = re.search(r"(\d+)", text)
    if match and 1 <= int(match.group(1)) <= len(labels):
        return f"Response{int(match.group(1))}"
    return "Tie"


def _parse_best_choice(value: Any) -> Optional[str]:
    text = str(value or "").strip()
    bare_match = re.match(r"^([123])(?=\s|$|<|\])", text)
    if bare_match:
        return f"Response{bare_match.group(1)}"
    if text.upper() in {"A", "B", "C"}:
        return f"Response{'ABC'.index(text.upper()) + 1}"
    response_match = re.search(r"\bResponse\s*([123])\b", text, re.IGNORECASE)
    if response_match:
        return f"Response{response_match.group(1)}"
    choice_match = re.search(
        r"\b(?:best(?:\s+choice)?|choice)\s*[:=]\s*\[?\s*([ABC])\s*\]?",
        text,
        re.IGNORECASE,
    )
    if choice_match:
        return f"Response{'ABC'.index(choice_match.group(1).upper()) + 1}"
    return None


def _best_choice_target(choice: Any) -> Optional[str]:
    response_id = _parse_best_choice(choice)
    if response_id is None:
        return None
    return f"{response_id[-1]}{base.DEFAULT_EOS_TOKEN}"


def _extract_unirm_json(text: str) -> Optional[Mapping[str, Any]]:
    raw = str(text or "").split("</think>")[-1].strip()
    code = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
    candidate = code.group(1) if code else raw[raw.find("{") : raw.rfind("}") + 1]
    if not candidate or not candidate.startswith("{"):
        return None
    try:
        value = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, Mapping) else None


def _unirm_scores_and_best(
    text: str, *, expected: int
) -> Tuple[List[Optional[float]], Optional[str]]:
    value = _extract_unirm_json(text)
    if value is None:
        return [None] * int(expected), None
    scores: List[Optional[float]] = [None] * int(expected)
    evaluations = value.get("evaluations")
    if isinstance(evaluations, list):
        for fallback_index, evaluation in enumerate(evaluations):
            if not isinstance(evaluation, Mapping):
                continue
            response_id = str(evaluation.get("response_id", ""))
            match = re.search(r"(\d+)", response_id)
            index = int(match.group(1)) - 1 if match else int(fallback_index)
            if not 0 <= index < int(expected):
                continue
            try:
                score = float(evaluation.get("final_score"))
            except (TypeError, ValueError):
                continue
            if math.isfinite(score) and 1.0 <= score <= 5.0:
                scores[index] = score
    best = str(value.get("best_id", "")).strip() or None
    return scores, best


def _pointwise_prompt(answer: SkyworkAnswer) -> str:
    return _converted_unirrm_prompt(
        system_prompt=CONVERTED_POINTWISE_SYSTEM_PROMPT,
        instruction=str(answer.instruction),
        input_text=str(answer.input_text),
        outputs=[str(answer.output)],
        assistant_suffix="Score: [",
    )


def _pointwise_items(
    answers: Sequence[SkyworkAnswer], *, smooth_alpha: float, reward_mean: float, decimals: int
) -> List[Tuple[str, str, str, int]]:
    items: List[Tuple[str, str, str, int]] = []
    for answer in answers:
        target = _smoothed_reward(answer.reward, mean=reward_mean, alpha=smooth_alpha)
        items.append(
            (
                "pointwise",
                _pointwise_prompt(answer),
                _format_reward(target, decimals=decimals) + "]" + base.DEFAULT_EOS_TOKEN,
                base.IGNORE_INDEX,
            )
        )
    return items


def _native_pointwise_items(
    answers: Sequence[SkyworkAnswer], *, decimals: int
) -> List[Tuple[str, str, str, int]]:
    return [
        (
            "pointwise",
            _unirm_prompt(
                instruction=str(answer.instruction),
                input_text=str(answer.input_text),
                outputs=[str(answer.output)],
            ),
            _unirm_target([float(answer.reward)], "Response1", decimals=decimals),
            base.IGNORE_INDEX,
        )
        for answer in answers
    ]


def _native_pairwise_items(
    records: Sequence[Mapping[str, Any]], *, decimals: int
) -> List[Tuple[str, str, str, int]]:
    items: List[Tuple[str, str, str, int]] = []
    for record in records:
        nested = record.get("pairwise")
        if not isinstance(nested, Mapping):
            continue
        for pair_name in ("AB", "AC", "BC"):
            pair = nested.get(pair_name)
            if not isinstance(pair, Mapping):
                continue
            left, right = pair_name
            code = str(pair.get("choice_code", "")).strip()
            if code not in {"1", "2", "3"}:
                choice = str(pair.get("choice", "")).strip()
                code = "1" if choice == left else "2" if choice == right else "3"
            items.append(
                (
                    "pairwise",
                    base.build_pairwise_prompt(
                        system_prompt=base.DEFAULT_PAIRWISE_SYSTEM_PROMPT,
                        instruction=str(record.get("instruction", "")),
                        input_text=str(record.get("input", "")),
                        assistant_1_output=str(record.get(f"output{left}", "")),
                        assistant_2_output=str(record.get(f"output{right}", "")),
                    ),
                    f"[[{code}]]{base.DEFAULT_EOS_TOKEN}",
                    base.IGNORE_INDEX,
                )
            )
    return items


def _best_choice_listwise_items(
    records: Sequence[Mapping[str, Any]],
) -> List[Tuple[str, str, str, int]]:
    items: List[Tuple[str, str, str, int]] = []
    for record in records:
        target = _best_choice_target(record.get("listwise_choice"))
        if target is None:
            continue
        items.append(
            (
                "listwise",
                _best_choice_prompt(record),
                target,
                base.IGNORE_INDEX,
            )
        )
    return items


def _best_choice_metrics(
    true_choices: Sequence[Any], predicted_choices: Sequence[Optional[str]]
) -> Dict[str, Any]:
    if len(true_choices) != len(predicted_choices):
        raise ValueError("true and predicted best-choice sequences must have equal length")
    invalid = sum(choice is None for choice in predicted_choices)
    correct = 0
    for truth, predicted in zip(true_choices, predicted_choices):
        allowed = [str(truth)] if isinstance(truth, str) else [str(choice) for choice in truth]
        correct += int(predicted is not None and str(predicted) in allowed)
    accuracy = float(correct / len(true_choices)) if true_choices else None
    return {
        "n": len(true_choices),
        "sft_acc": accuracy,
        "sft_top_group_acc": accuracy,
        "sft_best_in_pred_top_acc": accuracy,
        "sft_invalid_pred": int(invalid),
        "sft_invalid_counted_as_wrong": True,
        "target": "best_choice",
    }


def _listwise_best_choice_metadata(
    records: Sequence[Mapping[str, Any]], *, eos: str
) -> Tuple[
    List[Optional[Dict[str, float]]],
    List[Optional[List[str]]],
    List[List[str]],
    Dict[str, int],
]:
    """Build best-choice soft targets only when the top reward is tied."""
    distributions: List[Optional[Dict[str, float]]] = []
    candidates: List[Optional[List[str]]] = []
    truth_groups: List[List[str]] = []
    stats = {"rows": 0, "unique_winner": 0, "tied_winner": 0, "missing_scores": 0}
    for record in records:
        source_choice = _parse_best_choice(record.get("listwise_choice"))
        if source_choice is None:
            raise ValueError(f"record id={record.get('id')} has invalid listwise_choice")
        scores = {
            key: _score_value(record, f"listwise_score{key}")
            for key in ("A", "B", "C")
        }
        if any(value is None for value in scores.values()):
            distributions.append(None)
            candidates.append(None)
            truth_groups.append([source_choice])
            stats["missing_scores"] += 1
            continue
        stats["rows"] += 1
        maximum = max(float(value) for value in scores.values() if value is not None)
        winners = [key for key in ("A", "B", "C") if float(scores[key]) == maximum]
        response_ids = [f"Response{'ABC'.index(key) + 1}" for key in winners]
        truth_groups.append(response_ids)
        if len(response_ids) == 1:
            distributions.append(None)
            candidates.append(None)
            stats["unique_winner"] += 1
            continue
        weight = 1.0 / float(len(response_ids))
        distributions.append({str(index): weight for index in range(len(response_ids))})
        candidates.append([f"{response_id[-1]}{eos}" for response_id in response_ids])
        stats["tied_winner"] += 1
    return distributions, candidates, truth_groups, stats


def _sample_training_items(
    items: Sequence[Tuple[str, str, str, int]],
    distributions: Sequence[Any],
    candidates: Sequence[Any],
    *,
    samples: int,
    seed: int,
) -> Tuple[List[Tuple[str, str, str, int]], List[Any], List[Any]]:
    if not (len(items) == len(distributions) == len(candidates)):
        raise ValueError("training items and metadata must have equal length")
    count = min(len(items), int(samples))
    if count <= 0:
        raise ValueError("training sample count must be positive")
    indices = np.random.default_rng(int(seed)).permutation(len(items))[:count]
    return (
        [items[int(index)] for index in indices],
        [distributions[int(index)] for index in indices],
        [candidates[int(index)] for index in indices],
    )


def _parse_reward(text: str, *, minimum: float = 1.0, maximum: float = 5.0) -> Optional[float]:
    match = re.search(r"(?<![\d.])(-?(?:\d+(?:\.\d+)?|\.\d+))", str(text or ""))
    if match is None:
        return None
    try:
        value = float(match.group(1))
    except ValueError:
        return None
    if not math.isfinite(value) or value < float(minimum) or value > float(maximum):
        return None
    return value


def _single_answer_eval(
    questions: Sequence[SkyworkQuestion], *, seed: int
) -> List[SkyworkAnswer]:
    rng = np.random.default_rng(int(seed))
    return [question.answers[int(rng.integers(0, len(question.answers)))] for question in questions]


def _sample_one_answer_questions(
    questions: Sequence[SkyworkQuestion], *, samples: int, seed: int
) -> List[SkyworkAnswer]:
    count = min(len(questions), int(samples))
    if count <= 0:
        raise ValueError("pointwise-train-samples must be positive")
    rng = np.random.default_rng(int(seed))
    order = rng.permutation(len(questions))[:count]
    return [
        questions[int(index)].answers[int(rng.integers(0, len(questions[int(index)].answers)))]
        for index in order
    ]


@torch.no_grad()
def _evaluate_pointwise(
    *, model: Any, tokenizer: Any, answers: Sequence[SkyworkAnswer], max_length: int,
    batch_size: int, max_new_tokens: int,
) -> Dict[str, Any]:
    truth: List[float] = []
    predictions: List[float] = []
    invalid = 0
    model.eval()
    for start in range(0, len(answers), max(1, int(batch_size))):
        batch = list(answers[start : start + max(1, int(batch_size))])
        encoded = [
            tokenizer(_pointwise_prompt(answer), add_special_tokens=True, truncation=False).input_ids
            for answer in batch
        ]
        encoded = [base._truncate_ids_preserve_edges(ids, int(max_length)) for ids in encoded]
        tokens = tokenizer.pad({"input_ids": encoded}, padding=True, return_tensors="pt")
        device = model.device if hasattr(model, "device") else next(model.parameters()).device
        tokens = {key: value.to(device) for key, value in tokens.items()}
        generated = model.generate(
            **tokens,
            do_sample=False,
            max_new_tokens=int(max_new_tokens),
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        prompt_length = int(tokens["input_ids"].shape[1])
        for index, answer in enumerate(batch):
            text = tokenizer.decode(generated[index, prompt_length:], skip_special_tokens=False)
            predicted = _parse_reward(text)
            if predicted is None:
                invalid += 1
                predicted = 1.0
            truth.append(float(answer.reward))
            predictions.append(float(predicted))
    true_values = np.asarray(truth, dtype=np.float64)
    pred_values = np.asarray(predictions, dtype=np.float64)
    return {
        "n": len(answers),
        "mae": float(np.mean(np.abs(pred_values - true_values))) if len(answers) else None,
        "invalid_pred": int(invalid),
    }


@torch.no_grad()
def _generate_native(
    *, model: Any, tokenizer: Any, prompts: Sequence[str], max_length: int,
    batch_size: int, max_new_tokens: int,
) -> List[str]:
    outputs: List[str] = []
    model.eval()
    for start in range(0, len(prompts), max(1, int(batch_size))):
        batch = list(prompts[start : start + max(1, int(batch_size))])
        encoded = [
            tokenizer(prompt, add_special_tokens=True, truncation=False).input_ids
            for prompt in batch
        ]
        encoded = [base._truncate_ids_preserve_edges(ids, int(max_length)) for ids in encoded]
        tokens = tokenizer.pad({"input_ids": encoded}, padding=True, return_tensors="pt")
        device = model.device if hasattr(model, "device") else next(model.parameters()).device
        tokens = {key: value.to(device) for key, value in tokens.items()}
        generated = model.generate(
            **tokens,
            do_sample=False,
            max_new_tokens=int(max_new_tokens),
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        prompt_length = int(tokens["input_ids"].shape[1])
        outputs.extend(
            tokenizer.decode(row[prompt_length:], skip_special_tokens=False)
            for row in generated
        )
    return outputs


def _ranking_from_float_scores(scores: Sequence[float]) -> str:
    groups: List[str] = []
    for score in sorted(set(float(value) for value in scores), reverse=True):
        groups.append("=".join(letter for letter, value in zip("ABC", scores) if float(value) == score))
    return ">".join(groups)


def _evaluate_native_pointwise(
    *, model: Any, tokenizer: Any, answers: Sequence[SkyworkAnswer], max_length: int,
    batch_size: int, max_new_tokens: int,
) -> Dict[str, Any]:
    prompts = [
        _unirm_prompt(
            instruction=str(answer.instruction), input_text=str(answer.input_text),
            outputs=[str(answer.output)],
        )
        for answer in answers
    ]
    texts = _generate_native(
        model=model, tokenizer=tokenizer, prompts=prompts, max_length=max_length,
        batch_size=batch_size, max_new_tokens=max_new_tokens,
    )
    truth, predictions, invalid = [], [], 0
    for answer, text in zip(answers, texts):
        scores, _ = _unirm_scores_and_best(text, expected=1)
        predicted = scores[0]
        if predicted is None:
            invalid += 1
            predicted = 1.0
        truth.append(float(answer.reward))
        predictions.append(float(predicted))
    return {
        "n": len(answers),
        "mae": float(np.mean(np.abs(np.asarray(predictions) - np.asarray(truth)))) if answers else None,
        "invalid_pred": int(invalid),
    }


def _native_pairwise_eval_tasks(
    records: Sequence[Mapping[str, Any]],
) -> Tuple[List[str], List[int]]:
    prompts: List[str] = []
    labels: List[int] = []
    for record in records:
        nested = record.get("pairwise")
        if not isinstance(nested, Mapping):
            continue
        for pair_name in ("AB", "AC", "BC"):
            pair = nested.get(pair_name)
            if not isinstance(pair, Mapping):
                continue
            left, right = pair_name
            code = str(pair.get("choice_code", "")).strip()
            if code not in {"1", "2", "3"}:
                choice = str(pair.get("choice", "")).strip()
                code = "1" if choice == left else "2" if choice == right else "3"
            prompts.append(
                _unirm_prompt(
                    instruction=str(record.get("instruction", "")),
                    input_text=str(record.get("input", "")),
                    outputs=[str(record.get(f"output{left}", "")), str(record.get(f"output{right}", ""))],
                )
            )
            labels.append(int(code) - 1)
    return prompts, labels


def _native_pairwise_prediction(text: str) -> Optional[int]:
    scores, best = _unirm_scores_and_best(text, expected=2)
    best_text = str(best or "").strip().lower()
    match = re.search(r"(\d+)", best_text)
    if match and int(match.group(1)) in {1, 2}:
        return int(match.group(1)) - 1
    if best_text == "tie":
        return 2
    if all(score is not None for score in scores):
        left, right = float(scores[0]), float(scores[1])
        return 0 if left > right else 1 if left < right else 2
    return None


def _evaluate_native_pairwise(
    *, model: Any, tokenizer: Any, records: Sequence[Mapping[str, Any]], max_length: int,
    batch_size: int, max_new_tokens: int,
) -> Dict[str, Any]:
    prompts, truth = _native_pairwise_eval_tasks(records)
    texts = _generate_native(
        model=model, tokenizer=tokenizer, prompts=prompts, max_length=max_length,
        batch_size=batch_size, max_new_tokens=max_new_tokens,
    )
    predictions, invalid = [], 0
    for text in texts:
        prediction = _native_pairwise_prediction(text)
        if prediction is None:
            invalid += 1
            prediction = 2
        predictions.append(int(prediction))
    y_true = np.asarray(truth, dtype=np.int64)
    y_pred = np.asarray(predictions, dtype=np.int64)
    return {
        "n": len(truth),
        "sft_acc": float(np.mean(y_true == y_pred)) if truth else None,
        "sft_tie_rate": float(np.mean(y_pred == 2)) if truth else None,
        "sft_invalid_pred": int(invalid),
        "sft_invalid_counted_as_wrong": True,
        "sft_confusion": base._confusion(y_true, y_pred, num_classes=3),
    }


def _evaluate_best_choice_listwise(
    *, model: Any, tokenizer: Any, records: Sequence[Mapping[str, Any]], max_length: int,
    batch_size: int, max_new_tokens: int,
) -> Dict[str, Any]:
    prompts: List[str] = []
    valid_records: List[Mapping[str, Any]] = []
    for record in records:
        choice = _parse_best_choice(record.get("listwise_choice"))
        if choice is None:
            continue
        valid_records.append(record)
        prompts.append(_best_choice_prompt(record))
    texts = _generate_native(
        model=model, tokenizer=tokenizer, prompts=prompts, max_length=max_length,
        batch_size=batch_size, max_new_tokens=max_new_tokens,
    )
    _, _, truth_groups, _ = _listwise_best_choice_metadata(
        valid_records, eos=base.DEFAULT_EOS_TOKEN
    )
    return _best_choice_metrics(truth_groups, [_parse_best_choice(text) for text in texts])


def _filter_raw_records(path: Path, ids: set[int]) -> List[Dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"{path} must contain a JSON list")
    return [dict(row) for row in raw if isinstance(row, dict) and int(row.get("id", -1)) in ids]


def _load_pairwise(
    path: Path, *, system_prompt: str = base.DEFAULT_PAIRWISE_SYSTEM_PROMPT
) -> Tuple[List[Any], List[Dict[str, Any]], Dict[str, Any]]:
    return base._load_pairwise_abc_eval_dataset(
        str(path), pairwise_system_prompt=str(system_prompt)
    )


def _load_listwise(path: Path) -> Tuple[List[Any], List[Dict[str, Any]], Dict[str, Any]]:
    return lw._load_listwise_eval_dataset(str(path))


def _swapped_pairwise_label(label: int) -> int:
    if int(label) == int(base.LABEL_A):
        return int(base.LABEL_B)
    if int(label) == int(base.LABEL_B):
        return int(base.LABEL_A)
    return int(base.LABEL_TIE)


def _augment_pairwise_order(
    examples: Sequence[Any], rows: Sequence[Mapping[str, Any]],
    raw_records: Sequence[Mapping[str, Any]],
) -> Tuple[List[Any], List[Dict[str, Any]], Dict[str, Any]]:
    records_by_id = {int(record.get("id", -1)): record for record in raw_records}
    out_examples: List[Any] = []
    out_rows: List[Dict[str, Any]] = []
    augmented = 0
    for example, row_value in zip(examples, rows):
        row = dict(row_value)
        row["order_augmented"] = False
        original = base.PairwiseExample(
            id=len(out_examples) + 1,
            dataset=str(example.dataset),
            group_id=int(example.group_id),
            pair_id=len(out_examples) + 1,
            model_a=str(example.model_a),
            model_b=str(example.model_b),
            prompt=str(example.prompt),
            label=int(example.label),
        )
        row["pair_id"] = int(original.pair_id)
        out_examples.append(original)
        out_rows.append(row)

        record = records_by_id.get(int(row.get("record_id", row.get("group_id", -1))))
        pair_name = str(row.get("pair_name", ""))[:2]
        if record is None or pair_name not in {"AB", "AC", "BC"}:
            continue
        left, right = pair_name
        swapped_label = _swapped_pairwise_label(int(example.label))
        swapped = base.PairwiseExample(
            id=len(out_examples) + 1,
            dataset=str(example.dataset),
            group_id=int(example.group_id),
            pair_id=len(out_examples) + 1,
            model_a=str(record.get(f"model{right}", example.model_b)),
            model_b=str(record.get(f"model{left}", example.model_a)),
            prompt=base.build_pairwise_prompt(
                system_prompt=base.DEFAULT_PAIRWISE_SYSTEM_PROMPT,
                instruction=str(record.get("instruction", "")),
                input_text=str(record.get("input", "")),
                assistant_1_output=str(record.get(f"output{right}", "")),
                assistant_2_output=str(record.get(f"output{left}", "")),
            ),
            label=int(swapped_label),
        )
        swapped_row = dict(row)
        swapped_row.update(
            {
                "pair_id": int(swapped.pair_id),
                "pair_name": f"{pair_name}_swap",
                "model_a": str(swapped.model_a),
                "model_b": str(swapped.model_b),
                "pairwise_label": int(swapped_label),
                "pairwise_token": str(base.label_to_token(int(swapped_label))),
                "order_augmented": True,
                "source_pair_name": pair_name,
            }
        )
        out_examples.append(swapped)
        out_rows.append(swapped_row)
        augmented += 1
    return out_examples, out_rows, {
        "enabled": True,
        "input_examples": len(examples),
        "order_augmented_examples": int(augmented),
        "output_examples": len(out_examples),
    }


def _augment_listwise_records(
    records: Sequence[Mapping[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    output: List[Dict[str, Any]] = []
    augmented = 0
    for record in records:
        for permutation_index, permutation in enumerate(itertools.permutations("ABC")):
            row = dict(record)
            for new_letter, old_letter in zip("ABC", permutation):
                for prefix in ("model", "output", "listwise_score"):
                    row[f"{prefix}{new_letter}"] = record.get(f"{prefix}{old_letter}")
            scores = [float(row[f"listwise_score{letter}"]) for letter in "ABC"]
            row["ranking"] = _ranking_from_float_scores(scores)
            old_choice = str(record.get("listwise_choice", "")).strip()
            if old_choice in permutation:
                row["listwise_choice"] = "ABC"[permutation.index(old_choice)]
            row["source_record_id"] = record.get("id")
            # Keep the normal ID field aligned with the source question so the
            # existing loader and evaluation metadata remain joinable.
            row["id"] = record.get("id")
            row["order_augmented"] = bool(permutation_index > 0)
            output.append(row)
            augmented += int(permutation_index > 0)
    return output, {
        "enabled": True,
        "input_records": len(records),
        "permutations_per_record": 6,
        "order_augmented_examples": int(augmented),
        "output_records": len(output),
    }


def _score_value(record: Mapping[str, Any], key: str) -> Optional[float]:
    value = record.get(key)
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _pairwise_soft_choice_metadata(
    rows: Sequence[Mapping[str, Any]], raw_records: Sequence[Mapping[str, Any]], *, eos: str
) -> Tuple[List[Optional[Dict[str, float]]], List[Optional[List[str]]], Dict[str, int]]:
    by_id = {int(row.get("id", -1)): row for row in raw_records}
    distributions: List[Optional[Dict[str, float]]] = []
    candidates: List[Optional[List[str]]] = []
    stats = {
        "rows": 0,
        "unique_winner": 0,
        "tied_winner": 0,
        "explicit_tie": 0,
        "missing_scores": 0,
    }
    for row in rows:
        record = by_id.get(int(row.get("record_id", row.get("group_id", -1))))
        pair_name = str(row.get("pair_name", ""))
        nested = record.get("pairwise", {}) if isinstance(record, Mapping) else {}
        pair = nested.get(pair_name, {}) if isinstance(nested, Mapping) else {}
        if not isinstance(pair, Mapping):
            pair = {}
        explicit_tie = int(row.get("pairwise_label", -1)) == int(base.LABEL_TIE)
        if not explicit_tie:
            explicit_tie = str(pair.get("choice_code", "")).strip() == "3"
        if explicit_tie:
            distributions.append(None)
            candidates.append(None)
            stats["explicit_tie"] += 1
            continue
        left, right = pair_name[:1], pair_name[1:2]
        left_score = _score_value(pair, f"score{left}")
        right_score = _score_value(pair, f"score{right}")
        if left_score is None or right_score is None:
            distributions.append(None)
            candidates.append(None)
            stats["missing_scores"] += 1
            continue
        stats["rows"] += 1
        if left_score == right_score:
            distributions.append({"left": 0.5, "right": 0.5})
            candidates.append([f"[[1]]{eos}", f"[[2]]{eos}"])
            stats["tied_winner"] += 1
        else:
            distributions.append(None)
            candidates.append(None)
            stats["unique_winner"] += 1
    return distributions, candidates, stats


def _listwise_soft_choice_metadata(
    rows: Sequence[Mapping[str, Any]], *, eos: str
) -> Tuple[List[Optional[Dict[str, float]]], List[Optional[List[str]]], Dict[str, int]]:
    distributions: List[Optional[Dict[str, float]]] = []
    candidates: List[Optional[List[str]]] = []
    stats = {"rows": 0, "unique_winner": 0, "tied_winner": 0, "missing_scores": 0}
    for row in rows:
        scores: Dict[str, Optional[float]] = {
            key: _score_value(row, f"listwise_score{key}")
            for key in ("A", "B", "C")
        }
        if any(value is None for value in scores.values()):
            distributions.append(None)
            candidates.append(None)
            stats["missing_scores"] += 1
            continue
        stats["rows"] += 1
        max_score = max(float(value) for value in scores.values() if value is not None)
        winners = [key for key in ("A", "B", "C") if float(scores[key]) == max_score]
        if len(winners) == 1:
            distributions.append(None)
            candidates.append(None)
            stats["unique_winner"] += 1
            continue
        lower = [key for key in ("A", "B", "C") if key not in winners]
        lower.sort(key=lambda key: (-float(scores[key]), key))
        ranking_texts = []
        for perm in itertools.permutations(winners):
            order = list(perm) + list(lower)
            ranking_texts.append(f"Ranking:[{'>'.join(order)}]{eos}")
        weight = 1.0 / float(len(ranking_texts))
        distributions.append({str(index): weight for index in range(len(ranking_texts))})
        candidates.append(ranking_texts)
        stats["tied_winner"] += 1
    return distributions, candidates, stats


def _apply_tie_policy(
    items: Sequence[Tuple[str, str, str, int]],
    distributions: Sequence[Optional[Dict[str, float]]],
    candidates: Sequence[Optional[List[str]]],
    *,
    policy: str,
) -> Tuple[List[Tuple[str, str, str, int]], List[Optional[Dict[str, float]]], List[Optional[List[str]]], Dict[str, int]]:
    if not (len(items) == len(distributions) == len(candidates)):
        raise ValueError("tie metadata must match item count")
    out_items, out_dist, out_candidates = [], [], []
    stats = {"input": len(items), "kept": 0, "soft_rows": 0, "dropped_tied_rows": 0}
    for item, dist, choice_candidates in zip(items, distributions, candidates):
        tied = dist is not None and len(dist) > 1
        if policy == "unique_only" and tied:
            stats["dropped_tied_rows"] += 1
            continue
        out_items.append(item)
        out_dist.append(dist if policy == "soft" else None)
        out_candidates.append(choice_candidates if policy == "soft" else None)
        stats["kept"] += 1
        stats["soft_rows"] += int(policy == "soft" and tied)
    return out_items, out_dist, out_candidates, stats


def _make_cfg(args: argparse.Namespace) -> three.RunConfig:
    namespace = SimpleNamespace(
        seed=int(args.seed), val_ratio=0.1, val_split_seed=55, pointwise_val_answer_seed=65,
        resume_stage1_model_dir="", eval_stages="final",
        budget=600, pointwise_epochs=int(args.pointwise_epochs), pairwise_epochs=int(args.pairwise_epochs),
        listwise_epochs=int(args.listwise_epochs), per_device_batch_size=int(args.per_device_batch_size),
        gradient_accumulation_steps=int(args.gradient_accumulation_steps), learning_rate=float(args.learning_rate),
        max_length=int(args.max_length), max_new_tokens_pointwise=int(args.max_new_tokens_pointwise),
        max_new_tokens_pairwise=int(args.max_new_tokens_pairwise), max_new_tokens_listwise=int(args.max_new_tokens_listwise),
        eval_batch_size=int(args.eval_batch_size), stage2_pointwise_replay_ratio=0,
        stage3_pointwise_replay_ratio=0, stage3_pairwise_replay_ratio=0,
        score_min=1, score_max=5, no_fix_score_prefix=False, use_lora=bool(args.use_lora),
        load_in_4bit=bool(args.load_in_4bit), max_pointwise_eval_samples=0,
        max_pairwise_eval_samples=0, max_listwise_eval_samples=0,
    )
    cfg = controls._make_cfg(namespace)
    # Decimal pointwise targets use ordinary causal-LM token loss.  Continuous
    # smoothing is applied to the numeric target before it is formatted.
    cfg.pointwise_global_smooth_alpha = 0.0
    return cfg


def _selection_args(args: argparse.Namespace, out: Path) -> argparse.Namespace:
    return argparse.Namespace(
        budget_units=int(args.budget_units),
        fixed_selected_questions="",
        seed=int(args.seed),
        selection_mode="proxy",
        selector_init_questions=int(args.selector_init_questions),
        selector_batch_size=int(args.selector_batch_size),
        selector_max_score_candidates=int(args.selector_pool_size),
        llama=str(args.llama),
        proxy_lr=float(args.proxy_lr),
        proxy_max_length=int(args.proxy_max_length),
        load_in_4bit=bool(args.load_in_4bit),
        use_lora=bool(args.use_lora),
        train_micro_batch_size=1,
        grad_accum_steps=int(args.gradient_accumulation_steps),
        eval_batch_size=1,
        selector_proxy_warmup_epochs=int(args.selector_proxy_warmup_epochs),
        selector_proxy_update_epochs=int(args.selector_proxy_update_epochs),
        smooth_alpha=float(args.smooth_alpha),
        smooth_start_step=0,
        smooth_warmup_steps=0,
        selector_model="",
        selector_max_length=512,
        selector_lr=1e-3,
        selector_epochs=1,
        selector_unfreeze=False,
        selector_unfreeze_last_n_layers=0,
        proxy_mc_samples=int(args.proxy_mc_samples),
        proxy_uncertainty_weight=1.0,
        proxy_response_std_weight=0.0,
        proxy_exploration_ratio=0.0,
        out=str(out),
    )


def _select_three_signal_questions(
    *, args: argparse.Namespace, out: Path, point_questions: Sequence[SkyworkQuestion]
) -> Tuple[List[SkyworkQuestion], Dict[str, Any]]:
    raw_records = json.loads(args.pointwise_train.read_text(encoding="utf-8"))
    quantized_records: List[Dict[str, Any]] = []
    for record in raw_records:
        row = dict(record)
        for letter in "ABC":
            score = float(record[f"score{letter}"])
            row[f"score{letter}"] = max(1, min(10, int(round(score * 2.0))))
        quantized_records.append(row)
    quantized_path = out / "selector_pointwise_scores_x2.json"
    _write_json(quantized_path, quantized_records)

    questions, load_stats = lw._load_scored_questions_ge3(
        str(quantized_path), score_min=1, score_max=10
    )
    candidates, candidate_rows, candidate_stats = lw._build_candidate_triple_examples(
        questions, randomize_order=True, seed=int(args.seed) + 11
    )
    _write_json(out / "selector_dataset_load_stats.json", load_stats)
    _write_json(out / "candidate_triple_pool_stats.json", candidate_stats)
    _write_jsonl(out / "candidate_triples.jsonl", candidate_rows)

    cfg = _make_cfg(args)
    cfg.train_selection_mode = "candidate_triple_selector"
    cfg.candidate_selector_kind = "bias_trap_pointwise"
    cfg.candidate_selector_init_triples = int(args.selector_init_questions)
    cfg.candidate_selector_batch_size = int(args.selector_batch_size)
    cfg.candidate_selector_max_score_candidates = int(args.selector_pool_size)
    cfg.candidate_selector_one_per_question = True
    cfg.candidate_selector_target_task = "pointwise"
    cfg.candidate_selector_proxy_mode = "classifier_heads"
    cfg.reuse_selection_proxy_for_stage1 = False
    cfg.llama_multitask_mode = "classifier_heads"
    cfg.candidate_selector_proxy_warmup_epochs = int(args.selector_proxy_warmup_epochs)
    cfg.candidate_selector_proxy_update_epochs = int(args.selector_proxy_update_epochs)
    cfg.candidate_selector_exploration_ratio = 0.0
    cfg.candidate_selector_diversity_weight = 1.0
    cfg.candidate_selector_uncertainty_weight = 0.25
    cfg.candidate_selector_bias_weight = 1.0
    cfg.candidate_selector_density_weight = 0.15
    cfg.candidate_selector_coverage_weight = 0.0
    cfg.candidate_selector_pointwise_length_bias_weight = 0.5
    cfg.candidate_selector_pairwise_position_bias_weight = 0.5
    cfg.candidate_selector_pairwise_position_pairs = 1
    cfg.candidate_selector_pairwise_position_bias_scale = 0.02
    cfg.candidate_selector_signal_normalization = "none"
    cfg.candidate_selector_uncertainty_view = "pointwise"
    cfg.candidate_selector_diversity_view = "pointwise"
    cfg.candidate_selector_density_k = 10
    cfg.candidate_selector_embedding_model = "BAAI/bge-small-en-v1.5"
    cfg.candidate_selector_embedding_max_length = 512
    cfg.candidate_selector_embedding_batch_size = 64
    cfg.candidate_selector_embedding_pooling = "cls"
    cfg.proxy_lr = float(args.proxy_lr)
    cfg.proxy_max_length = int(args.proxy_max_length)
    cfg.score_min = 1
    cfg.score_max = 10
    cfg.budget_units = 600

    selected_triples, selected_rows, selected_stats = lw._select_candidate_triples_with_selector(
        candidates=candidates,
        cfg=cfg,
        llama_path=str(args.llama),
        output_dir=out,
    )
    _write_json(out / "candidate_triple_selection_stats.json", selected_stats)
    _write_jsonl(out / "selected_triples.jsonl", selected_rows)
    selected_ids = {int(triple.source_id) for triple in selected_triples}
    selected_questions = [
        question for question in point_questions if int(question.source_id) in selected_ids
    ]
    if len(selected_questions) != len(selected_ids):
        raise RuntimeError(
            f"three-signal selection returned {len(selected_ids)} IDs but matched "
            f"{len(selected_questions)} reward-model questions"
        )
    return selected_questions, {
        **selected_stats,
        "mode": "three_signal_bias_trap_pointwise",
        "signals": {"diversity": 1.0, "uncertainty": 0.25, "bias": 1.0},
        "score_quantization": "round(continuous_reward * 2) into classes 1..10 for selector only",
        "final_sft_preserves_continuous_rewards": True,
    }


def _select_training_data(
    args: argparse.Namespace, out: Path
) -> Tuple[List[SkyworkAnswer], Path, Path, Dict[str, Any]]:
    point_questions = load_skywork_json(str(args.pointwise_train), dataset_name="reward-model")
    if args.mode == "mix":
        selected_answers = _sample_one_answer_questions(
            point_questions,
            samples=int(args.pointwise_train_samples),
            seed=int(args.seed) + 101,
        )
        pair_records = json.loads(args.pairwise_train.read_text(encoding="utf-8"))
        list_records = json.loads(args.listwise_train.read_text(encoding="utf-8"))
        if len(pair_records) != 200 or len(list_records) != 200:
            raise ValueError("mix expects exactly 200 aligned pairwise/listwise question records")
        return selected_answers, args.pairwise_train, args.listwise_train, {
            "mode": "mix_explicit", "questions": len(point_questions), "pointwise_answers": len(selected_answers)
        }

    if args.mode == "three_signal_selector":
        selected_questions, selection = _select_three_signal_questions(
            args=args, out=out, point_questions=point_questions
        )
        selected_ids = {int(question.source_id) for question in selected_questions}
        pair_records = _filter_raw_records(args.pairwise_train, selected_ids)
        list_records = _filter_raw_records(args.listwise_train, selected_ids)
        if len(pair_records) != len(selected_ids) or len(list_records) != len(selected_ids):
            raise RuntimeError("three-signal selected IDs are not aligned across reward-model signals")
        selected_pair_path = out / "selected_pairwise.json"
        selected_list_path = out / "selected_listwise.json"
        _write_json(selected_pair_path, pair_records)
        _write_json(selected_list_path, list_records)
        return flatten_answers(selected_questions), selected_pair_path, selected_list_path, selection

    selected_questions, selection, _ = rm._select_questions(
        train_questions=point_questions,
        args=_selection_args(args, out),
        out=out,
    )
    selected_ids = {int(question.question_id) for question in selected_questions}
    pair_records = _filter_raw_records(args.pairwise_train, selected_ids)
    list_records = _filter_raw_records(args.listwise_train, selected_ids)
    if len(pair_records) != len(selected_ids) or len(list_records) != len(selected_ids):
        raise RuntimeError("selected pointwise IDs are not fully aligned with pairwise/listwise training data")
    selected_pair_path = out / "selected_pairwise.json"
    selected_list_path = out / "selected_listwise.json"
    _write_json(selected_pair_path, pair_records)
    _write_json(selected_list_path, list_records)
    return flatten_answers(selected_questions), selected_pair_path, selected_list_path, selection


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("mix", "selector", "three_signal_selector"), required=True)
    parser.add_argument(
        "--target-format",
        choices=("converted", "native_json"),
        default="converted",
        help=(
            "converted uses canonical score/choice targets; native_json keeps the UniRRM pointwise target, "
            "while pairwise/listwise targets use source choices and tie-aware best-choice outputs."
        ),
    )
    parser.add_argument("--pointwise-train", type=Path, required=True)
    parser.add_argument("--pairwise-train", type=Path, required=True)
    parser.add_argument("--listwise-train", type=Path, required=True)
    parser.add_argument("--pointwise-eval", type=Path, required=True)
    parser.add_argument("--pairwise-eval", type=Path, required=True)
    parser.add_argument("--listwise-eval", type=Path, required=True)
    parser.add_argument("--llama", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--budget-units", type=int, default=600)
    parser.add_argument("--pointwise-train-samples", type=int, default=200)
    parser.add_argument("--pairwise-train-samples", type=int, default=200)
    parser.add_argument("--listwise-train-samples", type=int, default=200)
    parser.add_argument("--pointwise-epochs", type=int, default=None)
    parser.add_argument("--pairwise-epochs", type=int, default=None)
    parser.add_argument("--listwise-epochs", type=int, default=None)
    parser.add_argument("--per-device-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--max-new-tokens-pointwise", type=int, default=16)
    parser.add_argument("--max-new-tokens-pairwise", type=int, default=8)
    parser.add_argument("--max-new-tokens-listwise", type=int, default=16)
    parser.add_argument("--max-new-tokens-native", type=int, default=192)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--reward-decimals", type=int, default=4)
    parser.add_argument(
        "--tie-policy",
        choices=("soft", "unique_only", "hard_first"),
        default="soft",
        help="soft averages tied winners; unique_only drops tied rows; hard_first keeps legacy hard targets",
    )
    parser.add_argument("--smooth-alpha", type=float, default=0.01)
    parser.add_argument("--pairwise-order-augmentation", action="store_true")
    parser.add_argument("--listwise-order-augmentation", action="store_true")
    parser.add_argument("--selector-pool-size", type=int, default=100)
    parser.add_argument("--selector-init-questions", type=int, default=80)
    parser.add_argument("--selector-batch-size", type=int, default=20)
    parser.add_argument("--selector-proxy-warmup-epochs", type=int, default=3)
    parser.add_argument("--selector-proxy-update-epochs", type=int, default=1)
    parser.add_argument("--proxy-mc-samples", type=int, default=4)
    parser.add_argument("--proxy-lr", type=float, default=1e-4)
    parser.add_argument("--proxy-max-length", type=int, default=768)
    parser.add_argument("--use-lora", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--load-in-4bit", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    default_epochs = 10 if args.mode == "mix" else 1
    for name in ("pointwise_epochs", "pairwise_epochs", "listwise_epochs"):
        if getattr(args, name) is None:
            setattr(args, name, int(default_epochs))
    if not 0.0 <= float(args.smooth_alpha) < 1.0:
        raise ValueError("smooth-alpha must be in [0, 1)")
    if args.mode == "mix" and float(args.smooth_alpha) != 0.0:
        raise ValueError("mix is the no-smooth control; pass --smooth-alpha 0")
    if args.target_format == "native_json" and (
        args.pairwise_order_augmentation or args.listwise_order_augmentation
    ):
        raise ValueError("native_json is the no-conversion control and cannot use order augmentation")
    for name in ("pointwise_train_samples", "pairwise_train_samples", "listwise_train_samples"):
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"{name.replace('_', '-')} must be positive")
    if int(args.budget_units) <= 0:
        raise ValueError("budget-units must be positive")
    args.out.mkdir(parents=True, exist_ok=False)
    _write_json(args.out / "config.json", {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()})

    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))

    point_answers, pair_train_path, list_train_path, selection = _select_training_data(args, args.out)
    reward_mean = float(np.mean([answer.reward for answer in point_answers]))
    pair_raw_records = json.loads(pair_train_path.read_text(encoding="utf-8"))
    list_raw_records = json.loads(list_train_path.read_text(encoding="utf-8"))
    pairwise_system_prompt = (
        CONVERTED_PAIRWISE_SYSTEM_PROMPT
        if args.target_format == "converted"
        else base.DEFAULT_PAIRWISE_SYSTEM_PROMPT
    )
    if args.target_format == "native_json":
        point_items = _native_pointwise_items(point_answers, decimals=int(args.reward_decimals))
        pair_items = _native_pairwise_items(pair_raw_records, decimals=int(args.reward_decimals))
        list_items = _best_choice_listwise_items(list_raw_records)
        pair_train, pair_train_rows, pair_train_stats = _load_pairwise(
            pair_train_path, system_prompt=pairwise_system_prompt
        )
        list_train, list_train_rows, list_train_stats = _load_listwise(list_train_path)
        pair_dist, pair_candidates, pair_tie_stats = _pairwise_soft_choice_metadata(
            pair_train_rows, pair_raw_records, eos=base.DEFAULT_EOS_TOKEN
        )
        pair_items, pair_dist, pair_candidates, pair_policy_stats = _apply_tie_policy(
            pair_items, pair_dist, pair_candidates, policy=str(args.tie_policy)
        )
        list_dist, list_candidates, _, list_tie_stats = _listwise_best_choice_metadata(
            list_raw_records, eos=base.DEFAULT_EOS_TOKEN
        )
        pair_policy_stats = {**pair_policy_stats, "format": "choice_only"}
        list_policy_stats = {"policy": "best_choice_soft_ties", "input": len(list_items), "kept": len(list_items)}
    else:
        point_items = _pointwise_items(
            point_answers,
            smooth_alpha=float(args.smooth_alpha),
            reward_mean=reward_mean,
            decimals=int(args.reward_decimals),
        )
        pair_train, pair_train_rows, pair_train_stats = _load_pairwise(
            pair_train_path, system_prompt=pairwise_system_prompt
        )
        list_raw_for_training = list_raw_records
        list_augmentation_stats = {"enabled": False, "input_records": len(list_raw_records), "output_records": len(list_raw_records)}
        if args.listwise_order_augmentation:
            list_raw_for_training, list_augmentation_stats = _augment_listwise_records(list_raw_records)
            list_augmented_path = args.out / "listwise_train_order_augmented.json"
            _write_json(list_augmented_path, list_raw_for_training)
            list_train_path_for_load = list_augmented_path
        else:
            list_train_path_for_load = list_train_path
        list_train, list_train_rows, list_train_stats = _load_listwise(list_train_path_for_load)
        pair_items_raw = three._pairwise_items(pair_train)
        pair_raw_dist, pair_raw_candidates, pair_tie_stats = _pairwise_soft_choice_metadata(
            pair_train_rows, pair_raw_records, eos=base.DEFAULT_EOS_TOKEN
        )
        if args.pairwise_order_augmentation:
            pair_train, pair_train_rows, pair_augmentation_stats = _augment_pairwise_order(
                pair_train, pair_train_rows, pair_raw_records
            )
            pair_items_raw = three._pairwise_items(pair_train)
            pair_dist, pair_candidates = [], []
            for dist, candidates in zip(pair_raw_dist, pair_raw_candidates):
                pair_dist.extend([dist, dist])
                pair_candidates.extend([candidates, candidates])
            pair_tie_stats = {**pair_tie_stats, **pair_augmentation_stats}
        else:
            pair_dist, pair_candidates = pair_raw_dist, pair_raw_candidates
            pair_augmentation_stats = {"enabled": False, "input_examples": len(pair_items_raw), "output_examples": len(pair_items_raw)}
        pair_items, pair_dist, pair_candidates, pair_policy_stats = _apply_tie_policy(
            pair_items_raw, pair_dist, pair_candidates, policy=str(args.tie_policy)
        )
        list_items = _best_choice_listwise_items(list_raw_for_training)
        list_dist, list_candidates, _, list_tie_stats = _listwise_best_choice_metadata(
            list_raw_for_training, eos=base.DEFAULT_EOS_TOKEN
        )
        list_policy_stats = {
            "policy": "best_choice_soft_ties",
            "input": len(list_items),
            "kept": len(list_items),
        }
        pair_train_stats = {**pair_train_stats, "generated_pairs_after_augmentation": len(pair_items_raw)}
        list_train_stats = {**list_train_stats, **list_augmentation_stats}

    if args.mode == "mix":
        pair_items, pair_dist, pair_candidates = _sample_training_items(
            pair_items,
            pair_dist,
            pair_candidates,
            samples=int(args.pairwise_train_samples),
            seed=int(args.seed) + 202,
        )
        list_items, list_dist, list_candidates = _sample_training_items(
            list_items,
            list_dist,
            list_candidates,
            samples=int(args.listwise_train_samples),
            seed=int(args.seed) + 303,
        )
        pair_policy_stats = {
            **pair_policy_stats,
            "final_train_samples": len(pair_items),
        }
        list_policy_stats = {
            **list_policy_stats,
            "final_train_samples": len(list_items),
        }

    point_eval_questions = load_skywork_json(str(args.pointwise_eval), dataset_name="reward-model")
    point_eval = _single_answer_eval(point_eval_questions, seed=int(args.seed) + 65)
    pair_eval, pair_eval_rows, pair_eval_stats = _load_pairwise(
        args.pairwise_eval, system_prompt=pairwise_system_prompt
    )
    list_eval, list_eval_rows, list_eval_stats = _load_listwise(args.listwise_eval)
    eval_pair_records = json.loads(args.pairwise_eval.read_text(encoding="utf-8"))
    eval_list_records = json.loads(args.listwise_eval.read_text(encoding="utf-8"))
    _write_jsonl(args.out / "pairwise_train.jsonl", pair_train_rows)
    _write_jsonl(args.out / "listwise_train.jsonl", list_train_rows)
    _write_jsonl(args.out / "pairwise_eval.jsonl", pair_eval_rows)
    _write_jsonl(args.out / "listwise_eval.jsonl", list_eval_rows)

    cfg = _make_cfg(args)
    train_stats: Dict[str, Any] = {}
    train_stats["stage1_pointwise"], model, tokenizer = three._train_sft_on_items(
        model_name_or_path=str(args.llama), model=None, tokenizer=None, items=point_items,
        output_dir=args.out / "stage1_pointwise_sft_model", cfg=cfg, stage_name="stage1_pointwise",
    )
    train_stats["stage2_pairwise"], model, tokenizer = three._train_sft_on_items(
        model_name_or_path=None, model=model, tokenizer=tokenizer, items=pair_items,
        choice_target_distributions=pair_dist, choice_candidate_targets=pair_candidates,
        output_dir=args.out / "stage2_pairwise_sft_model", cfg=cfg, stage_name="stage2_pairwise",
    )
    train_stats["stage3_listwise"], model, tokenizer = three._train_sft_on_items(
        model_name_or_path=None, model=model, tokenizer=tokenizer, items=list_items,
        choice_target_distributions=list_dist, choice_candidate_targets=list_candidates,
        output_dir=args.out / "stage3_listwise_sft_model", cfg=cfg, stage_name="stage3_listwise",
    )
    if args.mode in {"selector", "three_signal_selector"}:
        consolidation_items = list(point_items) + list(pair_items) + list(list_items)
        consolidation_dist = [None] * len(point_items) + list(pair_dist) + list(list_dist)
        consolidation_candidates = [None] * len(point_items) + list(pair_candidates) + list(list_candidates)
        train_stats["stage4_consolidation"], model, tokenizer = three._train_sft_on_items(
            model_name_or_path=None,
            model=model,
            tokenizer=tokenizer,
            items=consolidation_items,
            choice_target_distributions=consolidation_dist,
            choice_candidate_targets=consolidation_candidates,
            output_dir=args.out / "stage4_consolidation_sft_model",
            cfg=cfg,
            stage_name="stage4_consolidation",
        )

    if args.target_format == "native_json":
        point_metrics = _evaluate_native_pointwise(
            model=model, tokenizer=tokenizer, answers=point_eval, max_length=int(args.max_length),
            batch_size=int(args.eval_batch_size), max_new_tokens=int(args.max_new_tokens_native),
        )
        pair_metrics = base._evaluate_pairwise_sft(
            model=model, tokenizer=tokenizer, examples=pair_eval, max_length=int(args.max_length),
            batch_size=int(args.eval_batch_size), max_new_tokens=int(args.max_new_tokens_pairwise),
        )
    else:
        point_metrics = _evaluate_pointwise(
            model=model, tokenizer=tokenizer, answers=point_eval, max_length=int(args.max_length),
            batch_size=int(args.eval_batch_size), max_new_tokens=int(args.max_new_tokens_pointwise),
        )
        pair_metrics = base._evaluate_pairwise_sft(
            model=model, tokenizer=tokenizer, examples=pair_eval, max_length=int(args.max_length),
            batch_size=int(args.eval_batch_size), max_new_tokens=int(args.max_new_tokens_pairwise),
        )
    list_metrics = _evaluate_best_choice_listwise(
        model=model,
        tokenizer=tokenizer,
        records=eval_list_records,
        max_length=int(args.max_length),
        batch_size=int(args.eval_batch_size),
        max_new_tokens=(
            int(args.max_new_tokens_native)
            if args.target_format == "native_json"
            else int(args.max_new_tokens_listwise)
        ),
    )
    compact = {
        "mode": str(args.mode),
        "pointwise_mae": point_metrics.get("mae"),
        "pairwise_acc": pair_metrics.get("sft_acc"),
        "listwise_acc": list_metrics.get("sft_acc"),
        "rank_mae": list_metrics.get("sft_rank_mae"),
        "invalid_pred": {
            "pointwise": point_metrics.get("invalid_pred"),
            "pairwise": pair_metrics.get("sft_invalid_pred"),
            "listwise": list_metrics.get("sft_invalid_pred"),
        },
    }
    summary = {
        "mode": str(args.mode),
        "final_model": "generative_causal_lm_sft",
        "selector_proxy": (
            "three_signal_bias_trap_pointwise_classifier_proxy"
            if args.mode == "three_signal_selector"
            else "temporary_continuous_regression_only" if args.mode == "selector" else None
        ),
        "continuous_pointwise": {
            "preserved": True, "reward_mean": reward_mean, "smooth_alpha": float(args.smooth_alpha),
            "target_decimals": int(args.reward_decimals), "train_answers": len(point_answers),
        },
        "target_format": str(args.target_format),
        "listwise_target": "source_best_choice_with_soft_top_ties",
        "order_augmentation": {
            "pairwise": bool(args.pairwise_order_augmentation),
            "listwise": bool(args.listwise_order_augmentation),
        },
        "tie_policy": {
            "policy": str(args.tie_policy),
            "pairwise_scores": pair_tie_stats,
            "listwise_scores": list_tie_stats,
            "pairwise_training": pair_policy_stats,
            "listwise_training": list_policy_stats,
            "definition": "unique winner is one-hot; tied winners share probability uniformly",
        },
        "stage4_consolidation": bool(args.mode in {"selector", "three_signal_selector"}),
        "selection": selection,
        "data": {
            "pointwise_eval": len(point_eval),
            "pairwise_train": pair_train_stats, "pairwise_eval": pair_eval_stats,
            "listwise_train": list_train_stats, "listwise_eval": list_eval_stats,
        },
        "train_stats": train_stats,
        "metrics": {"pointwise": point_metrics, "pairwise": pair_metrics, "listwise": list_metrics},
    }
    _write_json(args.out / "summary.json", summary)
    _write_json(args.out / "metrics_compact.json", compact)
    print(json.dumps(compact, ensure_ascii=False, indent=2))

    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    main()

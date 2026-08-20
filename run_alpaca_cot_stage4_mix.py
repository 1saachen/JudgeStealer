#!/usr/bin/env python
"""Run the Alpaca CoT synthetic four-stage method or the real-CoT Mix control.

Stage4 mode selects 200 triples from the recovered 4066-question pool, trains
pointwise CoT, asks that Stage-1 model to synthesize pairwise/listwise
explanations using queried pointwise scores as private context,
and then trains pairwise, listwise, and full mixed consolidation stages.  The
generated explanation is never trusted for its label: the final choice/ranking
suffix is stripped and replaced with the deterministic label from pointwise
scores.

Mix mode trains on the same 200 validation questions reserved by
``prepare_alpaca_cot_4066.py``: one real pointwise, one real pairwise, and one
real listwise CoT example per question.  Ten epochs over 200/200/200 gives the
same 6000 training-example exposures as the full four-stage path.
"""

from __future__ import annotations

import argparse
import gc
import itertools
import json
import os
import random
import re
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import torch

import run_newnew_one_answer_trueval_three_stage_sft as control
import run_pointwise5answers_three_stage_pairwise_listwise_sft_v1 as three
import run_pointwise5answers_three_to_listwise_v1 as lw
import run_pointwise5answers_two_to_pairwise_v1 as base


ROOT = Path(__file__).resolve().parent
PREPARED = (
    ROOT
    / "train_with_selector"
    / "train_with_selector"
    / "data"
    / "Alpaca-cot-gpt"
    / "prepared_4066"
)
POSITIONS = ("A", "B", "C")
PAIR_NAMES = ("AB", "AC", "BC")

PAIRWISE_COT_SYSTEM_PROMPT = base.DEFAULT_PAIRWISE_SYSTEM_PROMPT.replace(
    "Your final line MUST be exactly one of:",
    "Before the final line, provide a brief explanation that supports your verdict. "
    "Do not mention private numeric scores.\n\nYour final line MUST be exactly one of:",
).replace(
    "Example:\n[[3]]",
    "Example:\nThe responses are similarly helpful and accurate overall.\n[[3]]",
)

LISTWISE_COT_SYSTEM_PROMPT = lw.LISTWISE_SYSTEM_PROMPT.replace(
    "Output exactly one final ranking in one of these formats and nothing else:",
    "First provide a brief explanation that supports the ordering. Do not mention "
    "private numeric scores. Then end with exactly one final ranking in one of these formats:",
).replace(
    "Do not provide any explanation or extra text.",
    "Do not output anything after the final ranking.",
)

PAIRWISE_SYNTH_SYSTEM_PROMPT = """You create a concise comparison rationale from private pointwise evidence.

The private pointwise scores define the required preference: describe the higher-scored response as stronger, or describe the responses as comparable when their scores are equal. Do not override that preference. Explain the difference using helpfulness, relevance, accuracy, depth, creativity, and level of detail. Avoid position and length bias. Do not mention numeric scores, private evidence, or output a final [[1]]/[[2]]/[[3]] verdict. Return only the explanation."""

LISTWISE_SYNTH_SYSTEM_PROMPT = """You create a concise ranking rationale from private pointwise evidence.

The private pointwise scores define the required ordering, including any ties. Do not override that ordering. Explain the ordering using helpfulness, relevance, accuracy, depth, creativity, and level of detail. Avoid position and length bias. Do not mention numeric scores, private evidence, or output a final Ranking label. Return only the explanation."""


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def _read_json(path: Path) -> List[Dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
        raise ValueError(f"expected a JSON array of objects: {path}")
    return list(value)


def _seed_everything(seed: int) -> None:
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _make_cfg(args: argparse.Namespace, *, mode: str) -> three.RunConfig:
    epochs = int(args.mix_epochs) if mode == "mix" else 1
    namespace = SimpleNamespace(
        seed=int(args.seed),
        val_ratio=0.0,
        val_split_seed=55,
        pointwise_val_answer_seed=65,
        resume_stage1_model_dir="",
        budget=600,
        pointwise_epochs=epochs,
        pairwise_epochs=epochs,
        listwise_epochs=epochs,
        per_device_batch_size=int(args.per_device_batch_size),
        gradient_accumulation_steps=int(args.gradient_accumulation_steps),
        learning_rate=float(args.learning_rate),
        max_length=int(args.max_length),
        max_new_tokens_pointwise=int(args.eval_max_new_tokens),
        max_new_tokens_pairwise=int(args.eval_max_new_tokens),
        max_new_tokens_listwise=int(args.eval_max_new_tokens),
        eval_batch_size=int(args.eval_batch_size),
        eval_stages="final",
        stage2_pointwise_replay_ratio=0,
        stage3_pointwise_replay_ratio=0,
        stage3_pairwise_replay_ratio=0,
        score_min=1,
        score_max=10,
        no_fix_score_prefix=True,
        use_lora=bool(args.use_lora),
        load_in_4bit=bool(args.load_in_4bit),
        max_pointwise_eval_samples=int(args.max_pointwise_eval_samples),
        max_pairwise_eval_samples=int(args.max_pairwise_eval_samples),
        max_listwise_eval_samples=int(args.max_listwise_eval_samples),
        fsdp="",
        fsdp_transformer_layer_cls_to_wrap="",
        fsdp_state_dict_type="FULL_STATE_DICT",
        fsdp_activation_checkpointing=False,
        fsdp_use_orig_params=True,
        fsdp_save_all_stages=False,
    )
    cfg = control._make_cfg(namespace)
    cfg.train_selection_mode = "candidate_triple_selector"
    cfg.candidate_selector_kind = str(args.selector_kind)
    cfg.candidate_selector_init_triples = int(args.selector_init_triples)
    cfg.candidate_selector_batch_size = int(args.selector_batch_size)
    cfg.candidate_selector_max_score_candidates = int(args.selector_pool_size)
    cfg.candidate_selector_one_per_question = True
    cfg.candidate_selector_target_task = "pointwise"
    cfg.candidate_selector_proxy_mode = "lm_head"
    cfg.reuse_selection_proxy_for_stage1 = bool(
        mode == "stage4"
        and bool(args.reuse_selection_proxy)
        and str(args.selector_kind) in {"pointwise_proxy", "bias_trap_pointwise"}
    )
    cfg.candidate_selector_proxy_warmup_epochs = int(args.selector_proxy_warmup_epochs)
    cfg.candidate_selector_proxy_update_epochs = int(args.selector_proxy_update_epochs)
    cfg.candidate_selector_exploration_ratio = 0.0
    cfg.candidate_selector_diversity_weight = 1.0
    cfg.candidate_selector_uncertainty_weight = 0.25
    cfg.candidate_selector_bias_weight = 1.0
    cfg.candidate_selector_pointwise_length_bias_weight = 0.5
    cfg.candidate_selector_pairwise_position_bias_weight = 0.5
    cfg.candidate_selector_pairwise_position_pairs = 1
    cfg.candidate_selector_pairwise_position_bias_scale = 0.02
    cfg.candidate_selector_signal_normalization = "none"
    cfg.candidate_selector_uncertainty_view = "pointwise"
    cfg.candidate_selector_density_k = 10
    cfg.candidate_selector_embedding_model = str(args.selector_embedding_model)
    cfg.candidate_selector_embedding_max_length = 512
    cfg.candidate_selector_embedding_batch_size = 64
    cfg.candidate_selector_embedding_pooling = "cls"
    cfg.candidate_selector_diversity_view = "pointwise"
    cfg.proxy_lr = 1e-4
    cfg.proxy_max_length = int(args.proxy_max_length)
    cfg.budget_units = 600
    cfg.pairwise_order_augmentation = True
    cfg.listwise_order_augmentation = True
    cfg.stage4_replay_strategy = "stratified_triple" if mode == "stage4" else "none"
    cfg.stage4_replay_fraction = 1.0
    cfg.stage4_epochs = 1
    cfg.pointwise_global_smooth_alpha = (
        float(args.smooth_alpha) if mode == "stage4" else 0.0
    )
    cfg.pointwise_global_smooth_mode = "local_gaussian"
    cfg.pointwise_global_smooth_gaussian_sigma = 1.0
    cfg.pointwise_global_smooth_stages = "all"
    return cfg


def _cot_pointwise_prompt(instruction: str, input_text: str, output: str) -> str:
    return base.build_judge_prompt(
        system_prompt=base.JUDGE_SYSTEM_PROMPT_WITH_EXPLANATION,
        instruction=str(instruction),
        input_text=str(input_text),
        candidate_output=str(output),
        include_gold_score=False,
        fix_score_prefix=False,
    )


def _cot_pointwise_target(reason: str, score: int) -> str:
    explanation = str(reason or "").strip()
    if not explanation:
        explanation = "The response was assessed for helpfulness, relevance, accuracy, and completeness."
    return f"{explanation}\nScore: [{int(score)}]{base.DEFAULT_EOS_TOKEN}"


def _pointwise_items_from_triples(
    triples: Sequence[lw.SelectedQuestionTriple],
) -> Tuple[List[Tuple[str, str, str, int]], List[Dict[str, Any]]]:
    items: List[Tuple[str, str, str, int]] = []
    rows: List[Dict[str, Any]] = []
    for triple in triples:
        for position, answer in zip(POSITIONS, (triple.answer_a, triple.answer_b, triple.answer_c)):
            prompt = _cot_pointwise_prompt(triple.instruction, triple.input_text, answer.output)
            target = _cot_pointwise_target(answer.reason, answer.score)
            items.append(("pointwise", prompt, target, int(answer.score) - 1))
            rows.append(
                {
                    "question_id": int(triple.question_id),
                    "position": position,
                    "model": str(answer.model),
                    "score": int(answer.score),
                    "reason": str(answer.reason),
                    "prompt": prompt,
                    "target": target.replace(base.DEFAULT_EOS_TOKEN, ""),
                }
            )
    return items, rows


def _pointwise_items_from_mix(
    rows: Sequence[Mapping[str, Any]],
) -> List[Tuple[str, str, str, int]]:
    return [
        (
            "pointwise",
            _cot_pointwise_prompt(row["instruction"], row.get("input", ""), row["output"]),
            _cot_pointwise_target(str(row["judge_reason"]), int(row["judge_score"])),
            int(row["judge_score"]) - 1,
        )
        for row in rows
    ]


def _pairwise_train_prompt(
    instruction: str, input_text: str, output_left: str, output_right: str
) -> str:
    return base.build_pairwise_prompt(
        system_prompt=PAIRWISE_COT_SYSTEM_PROMPT,
        instruction=str(instruction),
        input_text=str(input_text),
        assistant_1_output=str(output_left),
        assistant_2_output=str(output_right),
        include_verdict_instruction=False,
    )


def _listwise_train_prompt(
    instruction: str, input_text: str, outputs: Sequence[str]
) -> str:
    return lw._build_listwise_prompt(
        system_prompt=LISTWISE_COT_SYSTEM_PROMPT,
        instruction=str(instruction),
        input_text=str(input_text),
        assistant_a_output=str(outputs[0]),
        assistant_b_output=str(outputs[1]),
        assistant_c_output=str(outputs[2]),
    )


def _clean_synthetic_reason(text: str, *, task: str) -> str:
    value = str(text or "").strip()
    patterns = (
        (r"\s*\[\[[123]\]\].*$", r"\s*Ranking\s*:\s*\[[^\]]*\].*$")
        if task == "pairwise"
        else (r"\s*Ranking\s*:\s*\[[^\]]*\].*$", r"\s*\[\[[123]\]\].*$")
    )
    for pattern in patterns:
        value = re.sub(pattern, "", value, flags=re.IGNORECASE | re.DOTALL).strip()
    value = re.sub(r"^(Explanation|Rationale)\s*:\s*", "", value, flags=re.IGNORECASE).strip()
    return value


def _generate_texts(
    *,
    model: Any,
    tokenizer: Any,
    prompts: Sequence[str],
    batch_size: int,
    max_length: int,
    max_new_tokens: int,
) -> List[str]:
    outputs: List[str] = []
    model.eval()
    previous_use_cache = getattr(getattr(model, "config", None), "use_cache", None)
    if hasattr(model, "config"):
        model.config.use_cache = True
    device = three._infer_model_device(model)
    prompt_limit = max(128, int(max_length) - int(max_new_tokens))
    with torch.no_grad():
        for start in range(0, len(prompts), max(1, int(batch_size))):
            batch = list(prompts[start : start + max(1, int(batch_size))])
            encoded = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=int(prompt_limit),
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            generated = model.generate(
                **encoded,
                do_sample=False,
                max_new_tokens=int(max_new_tokens),
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            prompt_width = int(encoded["input_ids"].shape[1])
            for row in generated:
                outputs.append(
                    tokenizer.decode(row[prompt_width:], skip_special_tokens=True).strip()
                )
            del encoded, generated
    if hasattr(model, "config") and previous_use_cache is not None:
        model.config.use_cache = previous_use_cache
    return outputs


def _private_pair_prompt(
    triple: lw.SelectedQuestionTriple,
    left: base.AnswerWithScore,
    right: base.AnswerWithScore,
    *,
    include_pointwise_assessments: bool = False,
) -> str:
    public = base.build_pairwise_prompt(
        system_prompt=PAIRWISE_COT_SYSTEM_PROMPT,
        instruction=str(triple.instruction),
        input_text=str(triple.input_text),
        assistant_1_output=str(left.output),
        assistant_2_output=str(right.output),
        include_verdict_instruction=False,
    )
    evidence = [
        f"Assistant 1 score={int(left.score)}",
        f"Assistant 2 score={int(right.score)}",
    ]
    if bool(include_pointwise_assessments):
        evidence[0] += f"; assessment={str(left.reason)}"
        evidence[1] += f"; assessment={str(right.reason)}"
    return (
        public
        + "\n\nPrivate pointwise evidence (use internally; do not quote scores):\n"
        + "\n".join(evidence)
        + "\nFollow the evaluation format above: provide a brief explanation, then end with the required final line."
    )


def _private_list_prompt(
    triple: lw.SelectedQuestionTriple,
    answers: Sequence[base.AnswerWithScore],
    *,
    include_pointwise_assessments: bool = False,
) -> str:
    public = lw._build_listwise_prompt(
        system_prompt=LISTWISE_COT_SYSTEM_PROMPT,
        instruction=str(triple.instruction),
        input_text=str(triple.input_text),
        assistant_a_output=str(answers[0].output),
        assistant_b_output=str(answers[1].output),
        assistant_c_output=str(answers[2].output),
    )
    evidence_rows = [
        f"Assistant {position} score={int(answer.score)}"
        for position, answer in zip(POSITIONS, answers)
    ]
    if bool(include_pointwise_assessments):
        evidence_rows = [
            f"{row}; assessment={str(answer.reason)}"
            for row, answer in zip(evidence_rows, answers)
        ]
    evidence = "\n".join(evidence_rows)
    return (
        public
        + "\n\nPrivate pointwise evidence (use internally; do not quote scores):\n"
        + evidence
        + "\nFollow the evaluation format above: provide a brief explanation, then end with the required final ranking."
    )


def _fallback_pair_reason(label: int) -> str:
    if int(label) == int(base.LABEL_A):
        return "Assistant 1 better satisfies the instruction and provides the stronger response."
    if int(label) == int(base.LABEL_B):
        return "Assistant 2 better satisfies the instruction and provides the stronger response."
    return "The two responses are comparable in overall quality."


def _fallback_list_reason(ranking: str) -> str:
    return f"The responses differ in overall helpfulness and correctness, supporting the ordering {ranking}."


def _synthesize_pairwise_listwise_items(
    *,
    model: Any,
    tokenizer: Any,
    triples: Sequence[lw.SelectedQuestionTriple],
    batch_size: int,
    max_length: int,
    max_new_tokens: int,
    include_pointwise_assessments: bool = False,
) -> Tuple[
    List[Tuple[str, str, str, int]],
    List[Tuple[str, str, str, int]],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    Dict[str, Any],
]:
    pair_specs: List[Tuple[lw.SelectedQuestionTriple, str, str, base.AnswerWithScore, base.AnswerWithScore]] = []
    list_specs: List[Tuple[lw.SelectedQuestionTriple, Tuple[int, int, int], Tuple[base.AnswerWithScore, ...]]] = []
    pair_prompts: List[str] = []
    list_prompts: List[str] = []
    for triple in triples:
        answers = {"A": triple.answer_a, "B": triple.answer_b, "C": triple.answer_c}
        for left_position, right_position in itertools.permutations(POSITIONS, 2):
            left, right = answers[left_position], answers[right_position]
            pair_specs.append((triple, left_position, right_position, left, right))
            pair_prompts.append(
                _private_pair_prompt(
                    triple,
                    left,
                    right,
                    include_pointwise_assessments=include_pointwise_assessments,
                )
            )
        ordered = (triple.answer_a, triple.answer_b, triple.answer_c)
        for permutation in itertools.permutations(range(3)):
            permuted = tuple(ordered[index] for index in permutation)
            list_specs.append((triple, tuple(permutation), permuted))
            list_prompts.append(
                _private_list_prompt(
                    triple,
                    permuted,
                    include_pointwise_assessments=include_pointwise_assessments,
                )
            )

    pair_generated = _generate_texts(
        model=model,
        tokenizer=tokenizer,
        prompts=pair_prompts,
        batch_size=batch_size,
        max_length=max_length,
        max_new_tokens=max_new_tokens,
    )
    list_generated = _generate_texts(
        model=model,
        tokenizer=tokenizer,
        prompts=list_prompts,
        batch_size=batch_size,
        max_length=max_length,
        max_new_tokens=max_new_tokens,
    )

    pair_items: List[Tuple[str, str, str, int]] = []
    pair_rows: List[Dict[str, Any]] = []
    pair_fallbacks = 0
    for spec, generation, synth_prompt in zip(pair_specs, pair_generated, pair_prompts):
        triple, left_position, right_position, left, right = spec
        label = three._pairwise_label_from_scores(int(left.score), int(right.score))
        reason = _clean_synthetic_reason(generation, task="pairwise")
        if not reason:
            reason = _fallback_pair_reason(label)
            pair_fallbacks += 1
        token = base.label_to_token(int(label))
        train_prompt = _pairwise_train_prompt(
            triple.instruction, triple.input_text, left.output, right.output
        )
        target = f"{reason}\n[[{token}]]{base.DEFAULT_EOS_TOKEN}"
        pair_items.append(("pairwise", train_prompt, target, base.IGNORE_INDEX))
        pair_rows.append(
            {
                "question_id": int(triple.question_id),
                "left_position": left_position,
                "right_position": right_position,
                "score_left": int(left.score),
                "score_right": int(right.score),
                "label": int(label),
                "reason": reason,
                "raw_generation": generation,
                "synthesis_prompt": synth_prompt,
                "training_prompt": train_prompt,
                "target": target.replace(base.DEFAULT_EOS_TOKEN, ""),
            }
        )

    list_items: List[Tuple[str, str, str, int]] = []
    list_rows: List[Dict[str, Any]] = []
    list_fallbacks = 0
    for spec, generation, synth_prompt in zip(list_specs, list_generated, list_prompts):
        triple, permutation, answers = spec
        ranking = lw._ranking_from_scores(*(int(answer.score) for answer in answers))
        reason = _clean_synthetic_reason(generation, task="listwise")
        if not reason:
            reason = _fallback_list_reason(ranking)
            list_fallbacks += 1
        train_prompt = _listwise_train_prompt(
            triple.instruction, triple.input_text, [answer.output for answer in answers]
        )
        target = f"{reason}\nRanking:[{ranking}]{base.DEFAULT_EOS_TOKEN}"
        list_items.append(("listwise", train_prompt, target, base.IGNORE_INDEX))
        list_rows.append(
            {
                "question_id": int(triple.question_id),
                "permutation": list(permutation),
                "scores": [int(answer.score) for answer in answers],
                "ranking": ranking,
                "reason": reason,
                "raw_generation": generation,
                "synthesis_prompt": synth_prompt,
                "training_prompt": train_prompt,
                "target": target.replace(base.DEFAULT_EOS_TOKEN, ""),
            }
        )
    stats = {
        "pairwise_examples": len(pair_items),
        "listwise_examples": len(list_items),
        "pairwise_fallbacks": int(pair_fallbacks),
        "listwise_fallbacks": int(list_fallbacks),
        "label_policy": "final labels forced from pointwise scores",
        "private_evidence": (
            "answers + pointwise scores + pointwise assessments"
            if bool(include_pointwise_assessments)
            else "answers + pointwise scores"
        ),
        "private_context_in_training_input": False,
    }
    return pair_items, list_items, pair_rows, list_rows, stats


def _strip_final_label(reason: str, *, task: str) -> str:
    return _clean_synthetic_reason(reason, task=task)


def _mix_pair_items(rows: Sequence[Mapping[str, Any]]) -> List[Tuple[str, str, str, int]]:
    items: List[Tuple[str, str, str, int]] = []
    for row in rows:
        value = row["pairwise"]["AB"]
        choice = int(value["choice"])
        reason = _strip_final_label(str(value.get("reason", "")), task="pairwise")
        if not reason:
            reason = _fallback_pair_reason(base._abc_choice_to_pairwise_label(choice))
        prompt = _pairwise_train_prompt(
            row["instruction"], row.get("input", ""), row["outputA"], row["outputB"]
        )
        items.append(
            ("pairwise", prompt, f"{reason}\n[[{choice}]]{base.DEFAULT_EOS_TOKEN}", base.IGNORE_INDEX)
        )
    return items


def _mix_list_items(rows: Sequence[Mapping[str, Any]]) -> List[Tuple[str, str, str, int]]:
    items: List[Tuple[str, str, str, int]] = []
    for row in rows:
        ranking = str(row["ranking"])
        reason = _strip_final_label(str(row["listwise"].get("reason", "")), task="listwise")
        if not reason:
            reason = _fallback_list_reason(ranking)
        prompt = _listwise_train_prompt(
            row["instruction"],
            row.get("input", ""),
            [row["outputA"], row["outputB"], row["outputC"]],
        )
        items.append(
            ("listwise", prompt, f"{reason}\nRanking:[{ranking}]{base.DEFAULT_EOS_TOKEN}", base.IGNORE_INDEX)
        )
    return items


def _build_eval_sets(
    rows: Sequence[Mapping[str, Any]],
    *,
    max_pointwise: int,
    max_pairwise: int,
    max_listwise: int,
) -> Tuple[List[base.PointwiseScoredExample], List[base.PairwiseExample], List[lw.ListwiseExample]]:
    pointwise: List[base.PointwiseScoredExample] = []
    pairwise: List[base.PairwiseExample] = []
    listwise: List[lw.ListwiseExample] = []
    for row in rows:
        qid = int(row["id"])
        for position in POSITIONS:
            score = int(row[f"score{position}"])
            pointwise.append(
                base.PointwiseScoredExample(
                    row_id=len(pointwise) + 1,
                    question_id=qid,
                    source_id=int(row.get("source_id", qid)),
                    dataset=str(row.get("dataset", "alpaca_cot_gpt")),
                    instruction=str(row["instruction"]),
                    input_text=str(row.get("input", "")),
                    model=str(row[f"model{position}"]),
                    output=str(row[f"output{position}"]),
                    score=score,
                    label=score - 1,
                    prompt=_cot_pointwise_prompt(
                        row["instruction"], row.get("input", ""), row[f"output{position}"]
                    ),
                    reason=str(row[f"reason{position}"]),
                )
            )
        for pair_name in PAIR_NAMES:
            left, right = pair_name
            choice = int(row["pairwise"][pair_name]["choice"])
            pairwise.append(
                base.PairwiseExample(
                    id=len(pairwise) + 1,
                    dataset=str(row.get("dataset", "alpaca_cot_gpt")),
                    group_id=qid,
                    pair_id=len(pairwise) + 1,
                    model_a=str(row[f"model{left}"]),
                    model_b=str(row[f"model{right}"]),
                    prompt=_pairwise_train_prompt(
                        row["instruction"],
                        row.get("input", ""),
                        row[f"output{left}"],
                        row[f"output{right}"],
                    ),
                    label=base._abc_choice_to_pairwise_label(choice),
                )
            )
        ranking = str(row["ranking"])
        listwise.append(
            lw.ListwiseExample(
                id=len(listwise) + 1,
                dataset=str(row.get("dataset", "alpaca_cot_gpt")),
                group_id=qid,
                source_id=int(row.get("source_id", qid)),
                model_a=str(row["modelA"]),
                model_b=str(row["modelB"]),
                model_c=str(row["modelC"]),
                prompt=_listwise_train_prompt(
                    row["instruction"],
                    row.get("input", ""),
                    [row["outputA"], row["outputB"], row["outputC"]],
                ),
                ranking=ranking,
                label=lw._label_from_ranking(ranking),
            )
        )
    if int(max_pointwise) > 0:
        pointwise = pointwise[: int(max_pointwise)]
    if int(max_pairwise) > 0:
        pairwise = pairwise[: int(max_pairwise)]
    if int(max_listwise) > 0:
        listwise = listwise[: int(max_listwise)]
    return pointwise, pairwise, listwise


def _evaluate(
    *,
    model: Any,
    tokenizer: Any,
    cfg: three.RunConfig,
    eval_sets: Tuple[
        Sequence[base.PointwiseScoredExample],
        Sequence[base.PairwiseExample],
        Sequence[lw.ListwiseExample],
    ],
) -> Dict[str, Any]:
    pointwise, pairwise, listwise = eval_sets
    return {
        "pointwise": base._evaluate_pointwise_sft(
            model=model,
            tokenizer=tokenizer,
            examples=pointwise,
            max_length=int(cfg.max_length),
            batch_size=int(cfg.eval_batch_size),
            max_new_tokens=int(cfg.max_new_tokens_pointwise),
            score_min=1,
            score_max=10,
        ),
        "pairwise": base._evaluate_pairwise_sft(
            model=model,
            tokenizer=tokenizer,
            examples=pairwise,
            max_length=int(cfg.max_length),
            batch_size=int(cfg.eval_batch_size),
            max_new_tokens=int(cfg.max_new_tokens_pairwise),
        ),
        "listwise": three._evaluate_listwise_sft(
            model=model,
            tokenizer=tokenizer,
            examples=listwise,
            max_length=int(cfg.max_length),
            batch_size=int(cfg.eval_batch_size),
            max_new_tokens=int(cfg.max_new_tokens_listwise),
        ),
    }


def _run_stage4(args: argparse.Namespace, cfg: three.RunConfig, out: Path) -> None:
    questions, load_stats = base._load_scored_questions(
        str(args.train_questions), score_min=1, score_max=10
    )
    candidates, candidate_rows, candidate_stats = lw._build_candidate_triple_examples(
        questions,
        randomize_order=True,
        seed=int(args.seed) + 11,
    )
    _write_json(out / "dataset_load_stats.json", load_stats)
    _write_json(out / "candidate_stats.json", candidate_stats)
    _write_jsonl(out / "candidate_triples.jsonl", candidate_rows)
    selection_result = lw._select_candidate_triples_with_selector(
        candidates=candidates,
        cfg=cfg,
        llama_path=str(args.llama),
        output_dir=out,
    )
    selection_proxy = None
    if bool(cfg.reuse_selection_proxy_for_stage1):
        if len(selection_result) != 4:
            raise RuntimeError("selection proxy reuse requested, but selector returned no proxy")
        selected, selected_rows, selection_stats, selection_proxy = selection_result
    else:
        selected, selected_rows, selection_stats = selection_result
    _write_json(out / "selection_stats.json", selection_stats)
    _write_jsonl(out / "selected_triples.jsonl", selected_rows)

    point_items, point_rows = _pointwise_items_from_triples(selected)
    _write_jsonl(out / "pointwise_cot_train.jsonl", point_rows)
    stage1_hist = three._score_hist_from_items(point_items, num_labels=10)
    stage1_model = selection_proxy.model if selection_proxy is not None else None
    stage1_tokenizer = selection_proxy.tokenizer if selection_proxy is not None else None
    selection_proxy = None
    stage1_stats, model, tokenizer = three._train_sft_on_items(
        model_name_or_path=None if stage1_model is not None else str(args.llama),
        model=stage1_model,
        tokenizer=stage1_tokenizer,
        items=point_items,
        output_dir=out / "stage1_pointwise_cot_model",
        cfg=cfg,
        stage_name="stage1_pointwise",
        smooth_initial_hist=stage1_hist,
    )
    _write_json(out / "train_stats_stage1.json", stage1_stats)

    pair_items, list_items, pair_rows, list_rows, synth_stats = _synthesize_pairwise_listwise_items(
        model=model,
        tokenizer=tokenizer,
        triples=selected,
        batch_size=int(args.synthetic_batch_size),
        max_length=int(args.max_length),
        max_new_tokens=int(args.synthetic_max_new_tokens),
        include_pointwise_assessments=bool(args.include_pointwise_assessments_in_synthesis),
    )
    _write_json(out / "synthetic_cot_stats.json", synth_stats)
    _write_jsonl(out / "synthetic_pairwise_cot.jsonl", pair_rows)
    _write_jsonl(out / "synthetic_listwise_cot.jsonl", list_rows)

    stage2_stats, model, tokenizer = three._train_sft_on_items(
        model_name_or_path=None,
        model=model,
        tokenizer=tokenizer,
        items=pair_items,
        output_dir=out / "stage2_pairwise_cot_model",
        cfg=cfg,
        stage_name="stage2_pairwise",
    )
    _write_json(out / "train_stats_stage2.json", stage2_stats)
    stage3_stats, model, tokenizer = three._train_sft_on_items(
        model_name_or_path=None,
        model=model,
        tokenizer=tokenizer,
        items=list_items,
        output_dir=out / "stage3_listwise_cot_model",
        cfg=cfg,
        stage_name="stage3_listwise",
    )
    _write_json(out / "train_stats_stage3.json", stage3_stats)
    stage4_items = list(point_items) + list(pair_items) + list(list_items)
    stage4_stats, model, tokenizer = three._train_sft_on_items(
        model_name_or_path=None,
        model=model,
        tokenizer=tokenizer,
        items=stage4_items,
        output_dir=out / "stage4_cot_consolidation_model",
        cfg=cfg,
        stage_name="stage4_consolidation",
        smooth_initial_hist=stage1_hist,
    )
    _write_json(out / "train_stats_stage4.json", stage4_stats)

    eval_rows = _read_json(Path(args.eval_questions))
    eval_sets = _build_eval_sets(
        eval_rows,
        max_pointwise=int(args.max_pointwise_eval_samples),
        max_pairwise=int(args.max_pairwise_eval_samples),
        max_listwise=int(args.max_listwise_eval_samples),
    )
    metrics = _evaluate(model=model, tokenizer=tokenizer, cfg=cfg, eval_sets=eval_sets)
    _write_json(out / "metrics_pointwise_after_stage4.json", metrics["pointwise"])
    _write_json(out / "metrics_pairwise_after_stage4.json", metrics["pairwise"])
    _write_json(out / "metrics_listwise_after_stage4.json", metrics["listwise"])
    summary = {
        "mode": "alpaca_cot_synthetic_four_stage",
        "selection_stats": selection_stats,
        "train_budget": {
            "budget_units": 600,
            "selected_triples": len(selected),
            "pointwise": len(point_items),
            "pairwise": len(pair_items),
            "listwise": len(list_items),
            "stage4": len(stage4_items),
            "total_training_exposures": len(point_items) + len(pair_items) + len(list_items) + len(stage4_items),
        },
        "synthetic_cot": synth_stats,
        "pointwise_metrics": {"after_stage4": metrics["pointwise"]},
        "pairwise_metrics": {"after_stage4": metrics["pairwise"]},
        "listwise_metrics": {"after_stage4": metrics["listwise"]},
        "train_stats": {
            "stage1_pointwise": stage1_stats,
            "stage2_pairwise": stage2_stats,
            "stage3_listwise": stage3_stats,
            "stage4_consolidation": stage4_stats,
        },
    }
    _write_json(out / "summary.json", summary)
    _write_json(out / "metrics_compact.json", three._compact_metrics(summary))


def _run_mix(args: argparse.Namespace, cfg: three.RunConfig, out: Path) -> None:
    point_rows = _read_json(Path(args.mix_pointwise))
    pair_rows = _read_json(Path(args.mix_pairwise))
    list_rows = _read_json(Path(args.mix_listwise))
    if not (len(point_rows) == len(pair_rows) == len(list_rows) == 200):
        raise ValueError("Mix requires exactly 200 pointwise, 200 pairwise, and 200 listwise rows")
    point_items = _pointwise_items_from_mix(point_rows)
    pair_items = _mix_pair_items(pair_rows)
    list_items = _mix_list_items(list_rows)
    stage1_stats, model, tokenizer = three._train_sft_on_items(
        model_name_or_path=str(args.llama),
        model=None,
        tokenizer=None,
        items=point_items,
        output_dir=out / "stage1_pointwise_cot_model",
        cfg=cfg,
        stage_name="stage1_pointwise",
    )
    stage2_stats, model, tokenizer = three._train_sft_on_items(
        model_name_or_path=None,
        model=model,
        tokenizer=tokenizer,
        items=pair_items,
        output_dir=out / "stage2_pairwise_cot_model",
        cfg=cfg,
        stage_name="stage2_pairwise",
    )
    stage3_stats, model, tokenizer = three._train_sft_on_items(
        model_name_or_path=None,
        model=model,
        tokenizer=tokenizer,
        items=list_items,
        output_dir=out / "stage3_listwise_cot_model",
        cfg=cfg,
        stage_name="stage3_listwise",
    )
    _write_json(out / "train_stats_stage1.json", stage1_stats)
    _write_json(out / "train_stats_stage2.json", stage2_stats)
    _write_json(out / "train_stats_stage3.json", stage3_stats)

    eval_rows = _read_json(Path(args.eval_questions))
    eval_sets = _build_eval_sets(
        eval_rows,
        max_pointwise=int(args.max_pointwise_eval_samples),
        max_pairwise=int(args.max_pairwise_eval_samples),
        max_listwise=int(args.max_listwise_eval_samples),
    )
    metrics = _evaluate(model=model, tokenizer=tokenizer, cfg=cfg, eval_sets=eval_sets)
    _write_json(out / "metrics_pointwise_after_stage3.json", metrics["pointwise"])
    _write_json(out / "metrics_pairwise_after_stage3.json", metrics["pairwise"])
    _write_json(out / "metrics_listwise_after_stage3.json", metrics["listwise"])
    exposures = (len(point_items) + len(pair_items) + len(list_items)) * int(args.mix_epochs)
    summary = {
        "mode": "alpaca_real_cot_mix_200_200_200",
        "train_budget": {
            "pointwise": len(point_items),
            "pairwise": len(pair_items),
            "listwise": len(list_items),
            "epochs_per_stage": int(args.mix_epochs),
            "total_training_exposures": int(exposures),
        },
        "pointwise_metrics": {"after_stage3": metrics["pointwise"]},
        "pairwise_metrics": {"after_stage3": metrics["pairwise"]},
        "listwise_metrics": {"after_stage3": metrics["listwise"]},
        "train_stats": {
            "stage1_pointwise": stage1_stats,
            "stage2_pairwise": stage2_stats,
            "stage3_listwise": stage3_stats,
        },
    }
    _write_json(out / "summary.json", summary)
    _write_json(out / "metrics_compact.json", three._compact_metrics(summary))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("stage4", "mix"), required=True)
    parser.add_argument("--llama", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--train-questions", type=Path, default=PREPARED / "train_questions_4066.json")
    parser.add_argument("--eval-questions", type=Path, default=PREPARED / "eval_questions_1800.json")
    parser.add_argument("--mix-pointwise", type=Path, default=PREPARED / "mix_pointwise_train_200.json")
    parser.add_argument("--mix-pairwise", type=Path, default=PREPARED / "mix_pairwise_train_200.json")
    parser.add_argument("--mix-listwise", type=Path, default=PREPARED / "mix_listwise_train_200.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--per-device-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--eval-max-new-tokens", type=int, default=384)
    parser.add_argument("--synthetic-batch-size", type=int, default=4)
    parser.add_argument("--synthetic-max-new-tokens", type=int, default=256)
    parser.add_argument(
        "--include-pointwise-assessments-in-synthesis",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="also expose original pointwise rationales to the CoT synthesizer; default uses scores only",
    )
    parser.add_argument("--mix-epochs", type=int, default=10)
    parser.add_argument("--smooth-alpha", type=float, default=0.1)
    parser.add_argument("--selector-kind", choices=("random", "pointwise_proxy", "bias_trap_pointwise"), default="bias_trap_pointwise")
    parser.add_argument("--selector-init-triples", type=int, default=80)
    parser.add_argument("--selector-batch-size", type=int, default=20)
    parser.add_argument("--selector-pool-size", type=int, default=100)
    parser.add_argument("--selector-proxy-warmup-epochs", type=int, default=3)
    parser.add_argument("--selector-proxy-update-epochs", type=int, default=1)
    parser.add_argument("--selector-embedding-model", default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--proxy-max-length", type=int, default=768)
    parser.add_argument(
        "--reuse-selection-proxy",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="continue Stage-1 CoT SFT from the pointwise selection proxy",
    )
    parser.add_argument("--use-lora", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--load-in-4bit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-pointwise-eval-samples", type=int, default=0)
    parser.add_argument("--max-pairwise-eval-samples", type=int, default=0)
    parser.add_argument("--max-listwise-eval-samples", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.out.exists():
        raise FileExistsError(f"output directory already exists: {args.out}")
    args.out.mkdir(parents=True)
    _seed_everything(int(args.seed))
    cfg = _make_cfg(args, mode=str(args.mode))
    _write_json(
        args.out / "config.json",
        {
            "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
            "run_config": asdict(cfg),
            "cot_policy": {
                "stage4": "real pointwise CoT; proxy-synthetic pairwise/listwise CoT; forced score-derived labels",
                "mix": "real validation CoT, 200 pointwise + 200 pairwise + 200 listwise",
            },
        },
    )
    if str(args.mode) == "stage4":
        _run_stage4(args, cfg, args.out)
    else:
        _run_mix(args, cfg, args.out)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"Completed: {args.out}")


if __name__ == "__main__":
    main()

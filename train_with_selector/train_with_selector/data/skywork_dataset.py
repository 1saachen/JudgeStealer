from __future__ import annotations

"""Loader and utilities for grouped Skywork reward-model scores."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import json
import math
import random


ANSWER_KEYS: Tuple[str, ...] = ("A", "B", "C")


@dataclass(frozen=True)
class SkyworkAnswer:
    question_id: int
    source_id: int
    dataset: str
    instruction: str
    input_text: str
    answer_key: str
    model: str
    output: str
    reward: float

    def judge_text(self) -> str:
        parts = [
            "You are an impartial judge evaluating an AI assistant response.",
            "Assess its helpfulness, relevance, accuracy, depth, creativity, and level of detail.",
            "",
            "### User",
            f"Instruction: {self.instruction.strip()}",
        ]
        if self.input_text.strip():
            parts.append(f"Input: {self.input_text.strip()}")
        parts.extend(("", "### Assistant", self.output.strip()))
        return "\n".join(parts)

    def __str__(self) -> str:
        return self.judge_text()


@dataclass(frozen=True)
class SkyworkQuestion:
    question_id: int
    source_id: int
    dataset: str
    instruction: str
    input_text: str
    answers: Tuple[SkyworkAnswer, SkyworkAnswer, SkyworkAnswer]

    def selector_text(self) -> str:
        parts = ["### User", f"Instruction: {self.instruction.strip()}"]
        if self.input_text.strip():
            parts.append(f"Input: {self.input_text.strip()}")
        for answer in self.answers:
            parts.extend(("", f"### Candidate {answer.answer_key}", answer.output.strip()))
        return "\n".join(parts)

    def __str__(self) -> str:
        return self.selector_text()


def load_skywork_json(path: str, *, dataset_name: str = "skywork") -> List[SkyworkQuestion]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("Skywork dataset must be a JSON list")

    questions: List[SkyworkQuestion] = []
    seen_ids: set[int] = set()
    for row_index, row in enumerate(raw):
        if not isinstance(row, dict):
            raise ValueError(f"Skywork row {row_index} is not an object")
        try:
            question_id = int(row.get("id", row_index + 1))
            source_id = int(row.get("source_id", question_id))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Skywork row {row_index} has an invalid id") from exc
        if question_id in seen_ids:
            raise ValueError(f"duplicate Skywork id: {question_id}")
        seen_ids.add(question_id)

        instruction = str(row.get("instruction", row.get("Instruction", "")) or "")
        input_text = str(row.get("input", "") or "")
        if not instruction.strip():
            raise ValueError(f"Skywork row id={question_id} has an empty instruction")

        answers: List[SkyworkAnswer] = []
        for key in ANSWER_KEYS:
            model = str(row.get(f"model{key}", "") or "")
            output = str(row.get(f"output{key}", "") or "")
            reward_raw = row.get(f"score{key}")
            try:
                reward = float(reward_raw)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Skywork row id={question_id} has invalid score{key}") from exc
            if not math.isfinite(reward):
                raise ValueError(f"Skywork row id={question_id} has non-finite score{key}")
            if not output.strip():
                raise ValueError(f"Skywork row id={question_id} has an empty output{key}")
            answers.append(
                SkyworkAnswer(
                    question_id=question_id,
                    source_id=source_id,
                    dataset=str(row.get("dataset", dataset_name) or dataset_name),
                    instruction=instruction,
                    input_text=input_text,
                    answer_key=key,
                    model=model,
                    output=output,
                    reward=reward,
                )
            )
        questions.append(
            SkyworkQuestion(
                question_id=question_id,
                source_id=source_id,
                dataset=str(row.get("dataset", dataset_name) or dataset_name),
                instruction=instruction,
                input_text=input_text,
                answers=(answers[0], answers[1], answers[2]),
            )
        )
    return questions


def split_questions(
    questions: Sequence[SkyworkQuestion], *, val_ratio: float, seed: int
) -> Tuple[List[SkyworkQuestion], List[SkyworkQuestion]]:
    if not 0.0 < float(val_ratio) < 1.0:
        raise ValueError("val_ratio must be between 0 and 1")
    shuffled = list(questions)
    random.Random(int(seed)).shuffle(shuffled)
    val_count = max(1, int(round(len(shuffled) * float(val_ratio))))
    if val_count >= len(shuffled):
        raise ValueError("validation split leaves no training questions")
    val_ids = {q.question_id for q in shuffled[:val_count]}
    train = [q for q in questions if q.question_id not in val_ids]
    val = [q for q in questions if q.question_id in val_ids]
    return train, val


def flatten_answers(questions: Sequence[SkyworkQuestion]) -> List[SkyworkAnswer]:
    return [answer for question in questions for answer in question.answers]


def dataset_stats(questions: Sequence[SkyworkQuestion]) -> Dict[str, Any]:
    rewards = [answer.reward for answer in flatten_answers(questions)]
    models = {answer.model for answer in flatten_answers(questions)}
    return {
        "questions": len(questions),
        "answers": len(rewards),
        "models": len(models),
        "reward_min": min(rewards) if rewards else None,
        "reward_max": max(rewards) if rewards else None,
        "reward_mean": sum(rewards) / len(rewards) if rewards else None,
    }

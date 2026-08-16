import json

import pytest

from train_with_selector.train_with_selector.data.skywork_dataset import (
    flatten_answers,
    load_skywork_json,
    split_questions,
)


def _row(row_id: int):
    row = {"id": row_id, "instruction": f"question {row_id}", "input": ""}
    for key, score in zip("ABC", (1.25, -2.0, 4.5)):
        row[f"model{key}"] = f"model-{key}"
        row[f"output{key}"] = f"answer-{key}"
        row[f"score{key}"] = score
    return row


def test_load_preserves_float_rewards_and_grouping(tmp_path):
    path = tmp_path / "skywork.json"
    path.write_text(json.dumps([_row(1), _row(2)]), encoding="utf-8")
    questions = load_skywork_json(str(path))
    assert len(questions) == 2
    assert [answer.reward for answer in questions[0].answers] == [1.25, -2.0, 4.5]
    assert len(flatten_answers(questions)) == 6
    assert questions[0].dataset == "skywork"


def test_split_keeps_all_answers_from_a_question_together(tmp_path):
    path = tmp_path / "skywork.json"
    path.write_text(json.dumps([_row(i) for i in range(1, 11)]), encoding="utf-8")
    train, val = split_questions(load_skywork_json(str(path)), val_ratio=0.2, seed=65)
    assert len(train) == 8
    assert len(val) == 2
    assert {q.question_id for q in train}.isdisjoint({q.question_id for q in val})


def test_rejects_non_finite_scores(tmp_path):
    row = _row(1)
    row["scoreA"] = float("inf")
    path = tmp_path / "skywork.json"
    path.write_text(json.dumps([row]), encoding="utf-8")
    with pytest.raises(ValueError, match="non-finite"):
        load_skywork_json(str(path))

import json

import pytest

import run_pointwise5answers_three_to_listwise_v1 as listwise
import run_pointwise5answers_two_to_pairwise_v1 as pairwise


def _claude_row(**updates):
    row = {
        "id": 7,
        "instruction": "Rank these answers.",
        "input": "",
        "answerA": "answer a",
        "answerB": "answer b",
        "answerC": "answer c",
        "scoreA": 9,
        "scoreB": 6,
        "scoreC": 2,
        "pairwise_ab_choice": 1,
        "pairwise_bc_choice": 1,
        "pairwise_ac_choice": 1,
        "listwise_ranking": "A>B>C",
    }
    row.update(updates)
    return row


def test_scored_question_loader_accepts_claude_answer_fields(tmp_path):
    path = tmp_path / "train.json"
    path.write_text(json.dumps([_claude_row()]), encoding="utf-8")

    questions, stats = pairwise._load_scored_questions(str(path), score_min=1, score_max=10)

    assert len(questions) == 1
    assert [answer.output for answer in questions[0]["answers"]] == [
        "answer a",
        "answer b",
        "answer c",
    ]
    assert [answer.model for answer in questions[0]["answers"]] == ["A", "B", "C"]
    assert stats["answers_valid"] == 3


def test_explicit_pairwise_loader_accepts_claude_answer_fields():
    examples, rows, stats = pairwise._build_pairwise_abc_examples_from_records(
        [_claude_row()],
        dataset_path="claude.json",
        split_name="eval",
        pairwise_system_prompt=pairwise.DEFAULT_PAIRWISE_SYSTEM_PROMPT,
    )

    assert len(examples) == 3
    assert [row["pair_name"] for row in rows] == ["AB", "BC", "AC"]
    assert stats["generated_pairs"] == 3
    assert stats["skipped_missing_output"] == 0


def test_listwise_loader_accepts_claude_answers_and_explicit_ranking(tmp_path):
    path = tmp_path / "val.json"
    path.write_text(
        json.dumps([_claude_row(scoreA=1, scoreB=2, scoreC=10, listwise_ranking="A>B>C")]),
        encoding="utf-8",
    )

    examples, rows, stats = listwise._load_listwise_eval_dataset(str(path))

    assert len(examples) == 1
    assert examples[0].ranking == "A>B>C"
    assert rows[0]["ranking"] == "A>B>C"
    assert stats["examples"] == 1


def test_gpt5_output_fields_keep_existing_listwise_behavior(tmp_path):
    row = _claude_row(listwise_ranking="C>B>A")
    row.update(
        outputA=row.pop("answerA"),
        outputB=row.pop("answerB"),
        outputC=row.pop("answerC"),
    )
    path = tmp_path / "gpt5-listwise.json"
    path.write_text(json.dumps([row]), encoding="utf-8")

    examples, _, _ = listwise._load_listwise_eval_dataset(str(path))

    assert examples[0].ranking == "C>B>A"


def test_explicit_pairwise_loader_skips_unlabeled_ac_instead_of_making_tie():
    row = _claude_row()
    row.pop("pairwise_ac_choice")

    examples, rows, stats = pairwise._build_pairwise_abc_examples_from_records(
        [row],
        dataset_path="claude.json",
        split_name="eval",
        pairwise_system_prompt=pairwise.DEFAULT_PAIRWISE_SYSTEM_PROMPT,
    )

    assert len(examples) == 2
    assert [row["pair_name"] for row in rows] == ["AB", "BC"]
    assert stats["generated_pairs"] == 2
    assert stats["skipped_missing_choice"] == 1
    assert stats["label_C"] == 0


def test_explicit_pairwise_loader_rejects_unknown_nonempty_choice():
    row = _claude_row(pairwise_ab_choice="unknown")

    with pytest.raises(ValueError, match="unknown pairwise choice"):
        pairwise._build_pairwise_abc_examples_from_records(
            [row],
            dataset_path="claude.json",
            split_name="eval",
            pairwise_system_prompt=pairwise.DEFAULT_PAIRWISE_SYSTEM_PROMPT,
        )


def test_explicit_choice_three_remains_a_real_tie():
    row = _claude_row(pairwise_ab_choice=3)

    examples, rows, stats = pairwise._build_pairwise_abc_examples_from_records(
        [row],
        dataset_path="claude.json",
        split_name="eval",
        pairwise_system_prompt=pairwise.DEFAULT_PAIRWISE_SYSTEM_PROMPT,
    )

    ab_index = [item["pair_name"] for item in rows].index("AB")
    assert examples[ab_index].label == pairwise.LABEL_TIE
    assert stats["label_C"] == 1

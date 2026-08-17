# Explicit Judge Validation Loaders Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Load GPT-5 and Claude answer-field variants through the existing JSON loaders while ensuring explicit pairwise validation skips unlabeled AC pairs instead of manufacturing ties.

**Architecture:** Keep normalization at the three existing loader boundaries and preserve all training, three-stage entry, ranking fallback, and launcher behavior. Extend answer aliases in the pointwise, pairwise, and listwise loaders; then make the explicit pairwise loader distinguish missing choices from real ties and invalid nonempty choices.

**Tech Stack:** Python 3.10+, pytest, existing JSON loader helpers and dataclasses.

---

## File Structure

- Modify `run_pointwise5answers_two_to_pairwise_v1.py`: accept Claude answer fields in scored-question and explicit pairwise loaders; skip missing pairwise choices; reject invalid choices.
- Modify `run_pointwise5answers_three_to_listwise_v1.py`: accept Claude answer fields without changing ranking precedence or fallback.
- Create `tests/test_judge_data_loaders.py`: focused GPT-5/Claude loader regression coverage.
- Do not modify `run_pointwise5answers_three_stage_pairwise_listwise_sft_v1.py` or any launcher.

### Task 1: Accept GPT-5 And Claude Answer Fields

**Files:**
- Create: `tests/test_judge_data_loaders.py`
- Modify: `run_pointwise5answers_two_to_pairwise_v1.py:1039-1072`
- Modify: `run_pointwise5answers_two_to_pairwise_v1.py:2747-2800`
- Modify: `run_pointwise5answers_three_to_listwise_v1.py:2632-2636`

- [ ] **Step 1: Write the failing alias tests**

Create `tests/test_judge_data_loaders.py` with:

```python
import json

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
```

- [ ] **Step 2: Run the alias tests and verify RED**

Run:

```powershell
python -m pytest tests/test_judge_data_loaders.py -q
```

Expected: the three Claude-field tests fail because `answerA/B/C` are not recognized; the GPT-5 regression test passes. If imports fail because project dependencies are absent, activate the documented project environment or install `requirements.txt`, then rerun until failures are behavioral.

- [ ] **Step 3: Implement the minimal scored-question aliases**

In the A/B/C/D/E loop inside `_load_scored_questions`, replace the key handling and `model_row` construction with:

```python
model_key = f"model{ans_key}"
output_key = f"output{ans_key}"
answer_key = f"answer{ans_key}"
score_key = f"score{ans_key}"
if model_key not in rec and output_key not in rec and answer_key not in rec and score_key not in rec:
    continue
model_row = {
    "model": rec.get(model_key, ans_key),
    "output": rec.get(output_key, rec.get(answer_key, "")),
}
if score_key in rec:
    model_row["score"] = rec.get(score_key, None)
models.append(model_row)
```

Do not alter the numeric 1..5 loader branch.

- [ ] **Step 4: Implement the minimal explicit pairwise answer aliases**

Inside `_build_pairwise_abc_examples_from_records`, replace output lookup with:

```python
out_a_position = str(out_a_key)[-1]
out_b_position = str(out_b_key)[-1]
out_a = str(rec.get(out_a_key, rec.get(f"answer{out_a_position}", "")))
out_b = str(rec.get(out_b_key, rec.get(f"answer{out_b_position}", "")))
```

Replace model fallback with stable position names:

```python
model_a = str(rec.get(model_a_key, out_a_position))
model_b = str(rec.get(model_b_key, out_b_position))
```

- [ ] **Step 5: Implement the minimal listwise answer aliases**

Extend only the three existing alias tuples:

```python
output_a = str(_first_nonempty(rec, ("outputA", "output_a", "assistant_a", "responseA", "answerA"), ""))
output_b = str(_first_nonempty(rec, ("outputB", "output_b", "assistant_b", "responseB", "answerB"), ""))
output_c = str(_first_nonempty(rec, ("outputC", "output_c", "assistant_c", "responseC", "answerC"), ""))
```

Do not change ranking lookup or score fallback.

- [ ] **Step 6: Run the focused tests and verify GREEN**

Run:

```powershell
python -m pytest tests/test_judge_data_loaders.py -q
```

Expected: `4 passed`.

- [ ] **Step 7: Commit the alias support**

```powershell
git add tests/test_judge_data_loaders.py run_pointwise5answers_two_to_pairwise_v1.py run_pointwise5answers_three_to_listwise_v1.py
git commit -m "Support Claude answer fields in judge loaders"
```

### Task 2: Preserve Only Explicit Pairwise Gold Labels

**Files:**
- Modify: `tests/test_judge_data_loaders.py`
- Modify: `run_pointwise5answers_two_to_pairwise_v1.py:2696-2706`
- Modify: `run_pointwise5answers_two_to_pairwise_v1.py:2721-2785`

- [ ] **Step 1: Add failing missing-choice and invalid-choice tests**

Append:

```python
import pytest


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
```

- [ ] **Step 2: Run the new tests and verify RED**

Run:

```powershell
python -m pytest tests/test_judge_data_loaders.py::test_explicit_pairwise_loader_skips_unlabeled_ac_instead_of_making_tie tests/test_judge_data_loaders.py::test_explicit_pairwise_loader_rejects_unknown_nonempty_choice -q
```

Expected: both fail because the current mapper turns missing and unknown choices into ties.

- [ ] **Step 3: Make unknown choices explicit errors**

Replace the final fallback in `_abc_choice_to_pairwise_label`:

```python
raise ValueError(f"unknown pairwise choice: {choice!r}")
```

Keep all recognized `1`, `2`, `3`, letter, and tie aliases unchanged.

- [ ] **Step 4: Skip missing choices before label conversion**

Add these counters to the stats dictionary:

```python
"skipped_missing_choice": 0,
"invalid_choice": 0,
```

Immediately after the legacy AB fallback and before label conversion, add:

```python
if raw_choice is None or str(raw_choice).strip() == "":
    stats["skipped_missing_choice"] += 1
    continue
try:
    label = _abc_choice_to_pairwise_label(raw_choice)
except ValueError:
    stats["invalid_choice"] += 1
    raise
```

Remove the old unconditional `label = _abc_choice_to_pairwise_label(raw_choice)` line.

- [ ] **Step 5: Run all focused loader tests and verify GREEN**

Run:

```powershell
python -m pytest tests/test_judge_data_loaders.py -q
```

Expected: `7 passed`.

- [ ] **Step 6: Run the complete repository test suite**

Run:

```powershell
python -m pytest -q tests
```

Expected: all tests pass with zero failures.

- [ ] **Step 7: Commit explicit pairwise semantics**

```powershell
git add tests/test_judge_data_loaders.py run_pointwise5answers_two_to_pairwise_v1.py
git commit -m "Use only explicit pairwise validation labels"
```

### Task 3: Validate The Real Extracted Datasets And Report Consistency

**Files:**
- Read only: `data/Alpaca-claude/train.json`
- Read only: `data/Alpaca-claude/val.json`
- Read only: `data/GPT4ALL-claude/train.json`
- Read only: `data/GPT4ALL-claude/val.json`
- Read only: `data/gpt5/train-20k.json`
- Read only: `data/gpt5/val-2k-eval.json`
- Read only: `data/gpt5/val-2k-eval-listwise.json`

- [ ] **Step 1: Run the production loaders against every extracted JSON**

Run this from the repository root in the project environment:

```powershell
$code = @'
import run_pointwise5answers_three_to_listwise_v1 as lw
import run_pointwise5answers_two_to_pairwise_v1 as base

datasets = {
    "claude_alpaca": (
        "data/Alpaca-claude/train.json",
        "data/Alpaca-claude/val.json",
        "data/Alpaca-claude/val.json",
        10_000,
        4_000,
        2_000,
    ),
    "claude_gpt4all": (
        "data/GPT4ALL-claude/train.json",
        "data/GPT4ALL-claude/val.json",
        "data/GPT4ALL-claude/val.json",
        9_000,
        6_000,
        3_000,
    ),
    "gpt5_alpaca": (
        "data/gpt5/train-20k.json",
        "data/gpt5/val-2k-eval.json",
        "data/gpt5/val-2k-eval-listwise.json",
        20_000,
        4_000,
        2_000,
    ),
}

for name, (train, pair_val, list_val, expected_train, expected_pair, expected_list) in datasets.items():
    questions, train_stats = base._load_scored_questions(train, score_min=1, score_max=10)
    pairs, pair_rows, pair_stats = base._load_pairwise_abc_eval_dataset(
        pair_val,
        pairwise_system_prompt=base.DEFAULT_PAIRWISE_SYSTEM_PROMPT,
    )
    rankings, _, list_stats = lw._load_listwise_eval_dataset(list_val)
    assert len(questions) == expected_train
    assert len(pairs) == expected_pair
    assert len(rankings) == expected_list
    assert {row["pair_name"] for row in pair_rows} <= {"AB", "BC"}
    print(name, {
        "train_questions": len(questions),
        "pairwise_examples": len(pairs),
        "listwise_examples": len(rankings),
        "pairwise_labels": {
            "A": pair_stats["label_A"],
            "B": pair_stats["label_B"],
            "tie": pair_stats["label_C"],
        },
        "skipped_missing_choice": pair_stats["skipped_missing_choice"],
        "listwise_labels": list_stats["label_counts"],
        "valid_scored_answers": train_stats["answers_valid"],
    })
'@
python -c $code
```

Expected: all assertions pass; GPT-5 and Claude Alpaca each produce exactly 4,000 explicit AB/BC pairs, Claude GPT4ALL produces 6,000, and no source JSON is modified.

- [ ] **Step 2: Recompute judge-to-judge agreement using only explicit labels**

Run:

```powershell
$code = @'
import json
from statistics import fmean

gpt_train = json.load(open("data/gpt5/train-20k.json", encoding="utf-8"))
gpt_pair = json.load(open("data/gpt5/val-2k-eval.json", encoding="utf-8"))
gpt_list = json.load(open("data/gpt5/val-2k-eval-listwise.json", encoding="utf-8"))
claude_train = json.load(open("data/Alpaca-claude/train.json", encoding="utf-8"))
claude_val = json.load(open("data/Alpaca-claude/val.json", encoding="utf-8"))

gpt_train_by_id = {row["id"]: row for row in gpt_train}
gpt_pair_by_id = {row["id"]: row for row in gpt_pair}

content_matches = 0
score_matches = 0
score_rows = 0
score_diffs = []
for claude in claude_train:
    gpt = gpt_train_by_id[claude["id"]]
    same_content = (
        claude["instruction"], claude["input"],
        claude["answerA"], claude["answerB"], claude["answerC"],
    ) == (
        gpt["instruction"], gpt["input"],
        gpt["outputA"], gpt["outputB"], gpt["outputC"],
    )
    content_matches += same_content
    row_equal = True
    for key in "ABC":
        left, right = claude[f"score{key}"], gpt[f"score{key}"]
        score_matches += left == right
        row_equal &= left == right
        score_diffs.append(abs(left - right))
    score_rows += row_equal

pair_matches = 0
pair_total = 0
list_matches = 0
for claude, gpt_rank in zip(claude_val, gpt_list):
    gpt = gpt_pair_by_id[claude["id"]]
    pair_matches += claude["pairwise_ab_choice"] == gpt["choiceAB"]
    pair_matches += claude["pairwise_bc_choice"] == gpt["choiceBC"]
    pair_total += 2
    list_matches += claude["listwise_ranking"] == gpt_rank["ranking"]

report = {
    "aligned_train_content": f"{content_matches}/{len(claude_train)}",
    "score_cell_agreement": score_matches / (3 * len(claude_train)),
    "score_row_agreement": score_rows / len(claude_train),
    "score_mean_absolute_difference": fmean(score_diffs),
    "explicit_pairwise_agreement": pair_matches / pair_total,
    "explicit_listwise_agreement": list_matches / len(claude_val),
}
print(json.dumps(report, indent=2))
'@
python -c $code
```

Expected from the currently extracted files: 10,000/10,000 aligned training contents, score-cell agreement about `0.4049`, explicit pairwise agreement about `0.9177`, and explicit listwise agreement about `0.8305`. Report exact fresh values after execution.

- [ ] **Step 3: Verify scope and working tree**

Run:

```powershell
git diff --check
git status --short
git diff -- run_pointwise5answers_three_stage_pairwise_listwise_sft_v1.py launch_qwen3_gpt5_selector_smooth_lora_table_20260814.sh launch_qwen3_8b_gpt4all_gpt5_four_stage.sh
```

Expected: `git diff --check` succeeds; the final diff command is empty; pre-existing unrelated untracked files remain untouched.

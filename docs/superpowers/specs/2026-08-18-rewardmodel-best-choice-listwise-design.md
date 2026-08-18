# Reward-Model Best-Choice Listwise Design

## Goal

Make the reward-model Naive and Ours experiments use the original
`listwise_choice` label as the only listwise training target, and report
best-choice accuracy as `List Acc`.

## Decisions

- Naive uses no selector and samples 200 final examples for each task, with
  10 epochs per stage.
- Ours uses selector budget 600 answer units, which corresponds to 200 aligned
  reward-model questions with three answers each.
- Pointwise keeps continuous reward targets. Pairwise keeps source `choice`
  targets; equal-score non-explicit ties may use a uniform soft target, while
  explicit `choice=C` remains a hard tie.
- Listwise emits only the source best-choice format. Equal top listwise scores
  use a uniform soft target over tied best responses; no ranking or numeric
  score is emitted.
- Listwise evaluation accepts any member of an equal-score top group. Full
  ranking accuracy is not the table's primary metric.
- Raw listwise scores remain available in input data and summaries for
  diagnostics, but are not emitted as listwise SFT targets.

## Scope

The change covers `run_rewardmodel_three_stage_sft.py`, focused tests, and the
reward-model LoRA launcher/documentation. Existing selector acquisition,
pointwise reward regression, and evaluation data preparation remain unchanged.

## Validation

- Unit tests verify the listwise target contains only the canonical best-choice
  label and uses `listwise_choice`, even when score fields disagree.
- Unit tests verify best-choice accuracy handles valid predictions and invalid
  predictions deterministically.
- Existing reward-model and repository tests are run after the focused tests.

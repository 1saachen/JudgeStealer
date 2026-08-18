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
- Pointwise keeps continuous reward targets. Pairwise keeps the existing
  choice/tie targets.
- Listwise uses the source record's `listwise_choice` directly. It does not
  derive a ranking from pointwise scores or listwise scores.
- Listwise evaluation compares the predicted best response with the source
  best response. Full ranking accuracy is not the table's primary metric.
- Raw listwise scores remain available in input data and summaries for
  diagnostics, but are not emitted as listwise SFT targets.

## Scope

The change is limited to `run_rewardmodel_three_stage_sft.py` and focused
tests. Existing selector acquisition, pointwise reward regression, pairwise
training, and evaluation data preparation remain unchanged.

## Validation

- Unit tests verify the listwise target contains only the canonical best-choice
  label and uses `listwise_choice`, even when score fields disagree.
- Unit tests verify best-choice accuracy handles valid predictions and invalid
  predictions deterministically.
- Existing reward-model and repository tests are run after the focused tests.

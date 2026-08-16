# Dataset layout and schemas

Dataset files are deliberately not included in this bundle.

## Three-answer pointwise records

The selector SFT input and reward-model `pointwise.json` are JSON arrays. Each
record represents one question with three candidate answers:

```json
{
  "id": 1,
  "instruction": "Question text",
  "input": "Optional context",
  "modelA": "model-a",
  "outputA": "first answer",
  "scoreA": 4.5,
  "modelB": "model-b",
  "outputB": "second answer",
  "scoreB": 3.0,
  "modelC": "model-c",
  "outputC": "third answer",
  "scoreC": 2.0
}
```

Scores may be integers or finite decimal values. IDs must be unique.

## Reward-model pairwise records

`pairwise.json` must use the same IDs, instructions, and outputs as
`pointwise.json`, with judgments under `pairwise.AB`, `pairwise.AC`, and
`pairwise.BC`:

```json
{
  "id": 1,
  "instruction": "Question text",
  "input": "",
  "modelA": "model-a",
  "outputA": "first answer",
  "modelB": "model-b",
  "outputB": "second answer",
  "modelC": "model-c",
  "outputC": "third answer",
  "pairwise": {
    "AB": {"choice": "A"},
    "AC": {"choice": "A"},
    "BC": {"choice": "B"}
  }
}
```

Choices use the winning answer letter or a supported tie value such as
`"tie"`.

## Reward-model listwise records

`listwise.json` uses the same aligned question IDs and answer text. It must
contain `listwise_scoreA`, `listwise_scoreB`, and `listwise_scoreC`.
`prepare_rewardmodel_three_stage.py` derives the full ranking from those
scores.

## Selector evaluation records

Listwise evaluation accepts records with the same three outputs and a ranking
such as:

```json
{
  "id": 1,
  "Instruction": "Question text",
  "input": "",
  "outputA": "first answer",
  "outputB": "second answer",
  "outputC": "third answer",
  "ranking": "A>B>C"
}
```

The pairwise evaluation file can use the three-answer pairwise schema above.

Do not commit private/licensed datasets. Record the dataset source, license,
split seed, preprocessing command, and checksum separately.

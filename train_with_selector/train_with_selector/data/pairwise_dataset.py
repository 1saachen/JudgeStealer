from __future__ import annotations

"""Pairwise preference dataset loader + prompt construction.

This module is analogous to `judge_dataset.py` but for pairwise comparisons.

Input format (JSON list):

{
  "id": 1,
  "dataset": "Chatbot Arena",
  "Instruction": "...",
  "input": "...",
  "comparisons": [
    {
      "model_a": "A",
      "model_b": "B",
      "output_a": "...",
      "output_b": "...",
      "preference": "A" | "B" | "C" | "Tie" | ...
    },
    ...
  ]
}

We flatten each comparison into one trainable sample:
- prompt: system prompt + question + (assistant_1 response, assistant_2 response) + judge instruction
- label: 0 (Assistant 1 better) / 1 (Assistant 2 better) / 2 (Tie)

Important mapping:
- The raw `preference` may refer to answer IDs (A/B/C...) within a triplet.
  We convert it into the *pairwise position* label based on (model_a, model_b).
"""

from dataclasses import dataclass
from typing import Any, Dict, List

import json


DEFAULT_PAIRWISE_SYSTEM_PROMPT = """You are an impartial judge evaluating the quality of the responses provided by two AI assistants to the user question.

Your task is to determine which assistant performs better overall, or whether they perform equally well.

You may choose:
- Assistant 1
- Assistant 2
- Or a Tie, if both responses are of similar quality

Your evaluation must consider:
- Helpfulness
- Relevance
- Accuracy
- Depth
- Creativity
- Level of detail

Avoid any position bias. Do not let the order of responses influence your judgment.
Do not favor longer responses. Be as objective and fair as possible.

If both assistants perform similarly across the key criteria, you should select a Tie rather than forcing a preference.

Your final line MUST be exactly one of:
[[1]]
[[2]]
[[3]]

Where:
[[1]] = Assistant 1 is better
[[2]] = Assistant 2 is better
[[3]] = Tie

Do NOT output anything after the final line.

Example:
[[3]]
"""


LABEL_A = 0  # assistant 1 better
LABEL_B = 1  # assistant 2 better
LABEL_TIE = 2


def label_to_token(label: int) -> str:
    if int(label) == LABEL_A:
        return "1"
    if int(label) == LABEL_B:
        return "2"
    return "3"


def build_pairwise_prompt(
    *,
    system_prompt: str,
    instruction: str,
    input_text: str,
    assistant_1_output: str,
    assistant_2_output: str,
) -> str:
    instruction = (instruction or "").strip()
    input_text = (input_text or "").strip()
    assistant_1_output = (assistant_1_output or "").strip()
    assistant_2_output = (assistant_2_output or "").strip()

    parts: List[str] = ["### System", system_prompt.strip(), "", "### User"]
    parts.append(f"Instruction: {instruction}")
    if input_text:
        parts.append(f"Input: {input_text}")
    parts.append("")
    parts.append("### Assistant 1")
    parts.append(assistant_1_output)
    parts.append("")
    parts.append("### Assistant 2")
    parts.append(assistant_2_output)
    parts.append("")
    parts.append("### Judge")
    parts.append("Please output exactly one of: [[1]] / [[2]] / [[3]].")
    return "\n".join(parts)


def _normalize_pref(pref: Any) -> str:
    return str(pref or "").strip()


def preference_to_label(*, preference: Any, model_a: Any, model_b: Any) -> int:
    """Convert dataset preference into pairwise-position label.

    Rules:
    - If preference equals model_a => Assistant 1 better => label 0
    - If preference equals model_b => Assistant 2 better => label 1
    - If preference indicates tie => label 2

    We also accept already-normalized position labels A/B/C.
    """

    pref_s = _normalize_pref(preference)
    a = str(model_a or "").strip()
    b = str(model_b or "").strip()

    # If preference directly names the winner model/answer ID (common in datasets), honor that first.
    if pref_s == a:
        return LABEL_A
    if pref_s == b:
        return LABEL_B

    # Normalize bracketed outputs like "[[1]]" / "[[2]]" / "[[3]]" or legacy A/B/C.
    if pref_s in {"[[1]]", "[[2]]", "[[3]]", "[[A]]", "[[B]]", "[[C]]"}:
        pref_s = pref_s.strip("[]")

    # Explicit tie markers.
    if pref_s.lower() in {"tie", "t", "equal", "same"}:
        return LABEL_TIE
    if pref_s == "C":
        # When data is already in position-label space, "C" means tie.
        return LABEL_TIE

    # If data is already in position-label space (1/2/3 or legacy A/B/C), interpret directly.
    if pref_s in {"1", "A"}:
        return LABEL_A
    if pref_s in {"2", "B"}:
        return LABEL_B
    if pref_s in {"3", "C"}:
        return LABEL_TIE

    # Fallback: treat unknown as tie to avoid crashing.
    return LABEL_TIE


@dataclass(frozen=True)
class PairwiseExample:
    """One pairwise comparison sample."""

    id: int
    dataset: str
    group_id: int
    pair_id: int
    model_a: str
    model_b: str
    prompt: str
    label: int

    def __str__(self) -> str:  # noqa: D105
        return self.prompt


def _safe_int(x: Any, default: int = -1) -> int:
    try:
        return int(x)
    except Exception:
        return default


def load_pairwise_json(
    path: str,
    *,
    system_prompt: str = DEFAULT_PAIRWISE_SYSTEM_PROMPT,
) -> List[PairwiseExample]:
    """Load pairwise dataset from JSON(list) and flatten comparisons."""

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Pairwise dataset JSON must be a list of records.")

    examples: List[PairwiseExample] = []
    pair_id = 0
    for rec in data:
        if not isinstance(rec, dict):
            continue
        group_id = _safe_int(rec.get("id", -1), default=-1)
        dataset = str(rec.get("dataset", ""))
        instruction = str(rec.get("Instruction", rec.get("instruction", "")))
        input_text = str(rec.get("input", ""))
        comparisons = rec.get("comparisons", None)
        if not isinstance(comparisons, list) or not comparisons:
            continue

        for c in comparisons:
            if not isinstance(c, dict):
                continue
            pair_id += 1
            model_a = str(c.get("model_a", "A"))
            model_b = str(c.get("model_b", "B"))
            out_a = str(c.get("output_a", ""))
            out_b = str(c.get("output_b", ""))
            pref = c.get("preference", "")

            label = preference_to_label(preference=pref, model_a=model_a, model_b=model_b)
            prompt = build_pairwise_prompt(
                system_prompt=system_prompt,
                instruction=instruction,
                input_text=input_text,
                assistant_1_output=out_a,
                assistant_2_output=out_b,
            )
            examples.append(
                PairwiseExample(
                    id=pair_id,
                    dataset=dataset,
                    group_id=group_id,
                    pair_id=pair_id,
                    model_a=model_a,
                    model_b=model_b,
                    prompt=prompt,
                    label=int(label),
                )
            )

    return examples


def summarize_pairwise_dataset(examples: List[PairwiseExample]) -> Dict[str, Any]:
    """Lightweight summary for debugging/sanity checks."""

    out: Dict[str, Any] = {
        "n": int(len(examples)),
        "datasets": {},
        "label_counts": {"A": 0, "B": 0, "C": 0},
    }
    for ex in examples:
        out["datasets"][ex.dataset] = int(out["datasets"].get(ex.dataset, 0) + 1)
        out["label_counts"][label_to_token(ex.label)] += 1
    out["datasets"] = dict(sorted(out["datasets"].items(), key=lambda kv: (-kv[1], kv[0])))
    return out

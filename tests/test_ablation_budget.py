from pathlib import Path

import pytest

from ablation_budget import resolve_percentage_budget


ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = ROOT / "run_pointwise5answers_three_stage_pairwise_listwise_sft_v1.py"


@pytest.mark.parametrize(
    ("candidate_queries", "percent", "query_budget"),
    [
        (18_000, 0.5, 90),
        (18_000, 1.0, 180),
        (18_000, 2.0, 360),
        (18_000, 5.0, 900),
        (18_000, 10.0, 1_800),
        (8_120, 0.5, 40),
        (8_120, 1.0, 80),
        (8_120, 2.0, 160),
        (8_120, 5.0, 410),
        (8_120, 10.0, 810),
    ],
)
def test_resolve_percentage_budget(candidate_queries, percent, query_budget):
    resolved = resolve_percentage_budget(candidate_queries, percent)
    assert resolved.query_budget == query_budget
    assert resolved.budget_units == query_budget * 3
    assert resolved.init_triples == query_budget * 4 // 10
    assert resolved.selection_batch_size == query_budget // 10
    assert resolved.max_score_candidates == query_budget // 2


@pytest.mark.parametrize(
    ("candidate_queries", "percent"),
    [(0, 1.0), (9, 1.0), (100, 0.0), (100, -1.0), (100, 100.1)],
)
def test_resolve_percentage_budget_rejects_invalid_inputs(candidate_queries, percent):
    with pytest.raises(ValueError):
        resolve_percentage_budget(candidate_queries, percent)


@pytest.mark.parametrize(
    ("candidate_queries", "percent", "query_budget"),
    [
        (1_000, 1.5, 20),
        (100, 0.5, 10),
        (105, 100.0, 100),
        (110, 100.0, 110),
    ],
)
def test_resolve_percentage_budget_rounding_boundaries(
    candidate_queries, percent, query_budget
):
    assert resolve_percentage_budget(candidate_queries, percent).query_budget == query_budget


@pytest.mark.parametrize("percent", [float("nan"), float("inf"), float("-inf")])
def test_resolve_percentage_budget_rejects_non_finite_percent(percent):
    with pytest.raises(ValueError):
        resolve_percentage_budget(1_000, percent)


def test_three_stage_script_resolves_and_records_percentage_budget():
    text = TRAIN_SCRIPT.read_text(encoding="utf-8")
    assert 'parser.add_argument("--budget-percent", type=float, default=0.0)' in text
    assert "resolve_percentage_budget(len(train_questions), cfg.budget_percent)" in text
    assert "cfg.budget_units = resolved_budget.budget_units" in text
    assert "cfg.candidate_selector_init_triples = resolved_budget.init_triples" in text
    assert "cfg.candidate_selector_batch_size = resolved_budget.selection_batch_size" in text
    assert (
        "cfg.candidate_selector_max_score_candidates = "
        "resolved_budget.max_score_candidates" in text
    )
    assert 'out / "budget_percent_resolution.json"' in text

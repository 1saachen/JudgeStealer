from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class PercentageBudget:
    candidate_queries: int
    percent: float
    raw_query_budget: float
    query_budget: int
    budget_units: int
    init_triples: int
    selection_batch_size: int
    max_score_candidates: int


def resolve_percentage_budget(candidate_queries: int, percent: float) -> PercentageBudget:
    candidate_count = int(candidate_queries)
    percentage = float(percent)
    if candidate_count < 10:
        raise ValueError("candidate_queries must be at least 10")
    if not (0.0 < percentage <= 100.0):
        raise ValueError("percent must be in (0, 100]")

    raw_query_budget = candidate_count * percentage / 100.0
    query_budget = int(math.floor(raw_query_budget / 10.0 + 0.5)) * 10
    query_budget = max(10, min(query_budget, candidate_count))
    if query_budget % 10 != 0:
        query_budget = candidate_count - candidate_count % 10
    if query_budget < 10:
        raise ValueError("resolved query budget must be at least 10")

    selection_batch_size = query_budget // 10
    return PercentageBudget(
        candidate_queries=candidate_count,
        percent=percentage,
        raw_query_budget=float(raw_query_budget),
        query_budget=query_budget,
        budget_units=query_budget * 3,
        init_triples=query_budget * 4 // 10,
        selection_batch_size=selection_batch_size,
        max_score_candidates=selection_batch_size * 5,
    )

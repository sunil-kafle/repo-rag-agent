# Information retrieval evaluation helpers.
# This module evaluates retrieval quality using simple, stable metrics such as:
# - Hit Rate
# - Mean Reciprocal Rank (MRR)

from __future__ import annotations

from pydantic import BaseModel, Field

from src.retrieval.base import RetrievalResponse
from src.retrieval.formatting import normalize_repo_path


class IRMetricRow(BaseModel):
    """One evaluated retrieval query and its metric results."""

    query: str
    expected_docs: list[str] = Field(default_factory=list)
    retrieved_paths: list[str] = Field(default_factory=list)
    hit: bool
    mrr: float


class IRMetricSummary(BaseModel):
    """Aggregated IR metrics across many evaluated queries."""

    method: str
    hit_rate: float
    mrr: float
    total_queries: int


def _normalize_paths(paths: list[str]) -> list[str]:
    """Normalize repo paths before metric comparison."""
    return [normalize_repo_path(path) for path in paths]


def evaluate_search_quality(
    search_function,
    test_queries: list[tuple[str, list[str]]],
    top_k: int = 5,
) -> list[IRMetricRow]:
    """
    Evaluate a retrieval function on a set of test queries.

    Args:
        search_function:
            Callable that accepts (query, top_k) and returns either:
            - RetrievalResponse
            - list of dict results containing 'path'
        test_queries:
            List of tuples: (query, expected_doc_paths)
        top_k:
            Number of retrieved results to evaluate

    Returns:
        List of IRMetricRow objects
    """
    results: list[IRMetricRow] = []

    for query, expected_docs in test_queries:
        search_results = search_function(query, top_k=top_k)

        if isinstance(search_results, RetrievalResponse):
            retrieved_paths = [result.path for result in search_results.results]
        else:
            retrieved_paths = [item["path"] for item in search_results]

        normalized_retrieved = _normalize_paths(retrieved_paths)
        normalized_expected = _normalize_paths(expected_docs)

        hit = any(path in normalized_expected for path in normalized_retrieved)

        mrr = 0.0
        for i, path in enumerate(normalized_retrieved):
            if path in normalized_expected:
                mrr = 1.0 / (i + 1)
                break

        results.append(
            IRMetricRow(
                query=query,
                expected_docs=normalized_expected,
                retrieved_paths=normalized_retrieved,
                hit=hit,
                mrr=mrr,
            )
        )

    return results


def summarize_ir_metrics(rows: list[IRMetricRow], method: str) -> IRMetricSummary:
    """
    Aggregate IR metric rows into summary metrics.
    """
    if not rows:
        return IRMetricSummary(
            method=method,
            hit_rate=0.0,
            mrr=0.0,
            total_queries=0,
        )

    hit_rate = sum(1 for row in rows if row.hit) / len(rows)
    mrr = sum(row.mrr for row in rows) / len(rows)

    return IRMetricSummary(
        method=method,
        hit_rate=hit_rate,
        mrr=mrr,
        total_queries=len(rows),
    )
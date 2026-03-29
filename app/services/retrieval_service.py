# Retrieval service orchestration for the FastAPI layer.
# This service decides which retrieval strategy to run, when to apply
# query formulation, and how to prepare optional debug information.

from __future__ import annotations

from pydantic import BaseModel, Field

from src.config import settings
from src.retrieval.base import RetrievalMethod, RetrievalResult
from src.retrieval.formatting import format_source_label, normalize_repo_path
from src.retrieval.hybrid import hybrid_search
from src.retrieval.lexical import bm25_search
from src.retrieval.query import formulate_search_query
from src.retrieval.vector import vector_search


class RetrievalServiceResponse(BaseModel):
    """Structured retrieval service output for app-layer use."""

    original_query: str
    effective_query: str
    strategy_requested: RetrievalMethod
    strategy_used: RetrievalMethod
    top_k: int
    results: list[RetrievalResult] = Field(default_factory=list)
    debug: dict = Field(default_factory=dict)


def _normalize_results_for_display(results: list[RetrievalResult]) -> list[RetrievalResult]:
    """
    Return retrieval results with normalized display-friendly paths while
    preserving the rest of the retrieval payload.
    """
    normalized_results: list[RetrievalResult] = []

    for result in results:
        normalized_results.append(
            RetrievalResult(
                doc_id=result.doc_id,
                path=normalize_repo_path(result.path),
                chunk_id=result.chunk_id,
                content=result.content,
                score=result.score,
                source_method=result.source_method,
                rank=result.rank,
                metadata=result.metadata,
            )
        )

    return normalized_results


def retrieve_context(
    question: str,
    strategy: RetrievalMethod | None = None,
    top_k: int = 5,
    debug: bool = False,
) -> RetrievalServiceResponse:
    """
    Retrieve context for a user question using the selected strategy.

    Policy:
    - default strategy comes from settings
    - BM25 applies lightweight query formulation by default
    - vector and hybrid use the raw user question
    """
    strategy_to_use: RetrievalMethod = strategy or settings.default_retrieval_strategy

    if strategy_to_use == "bm25":
        effective_query = formulate_search_query(question)
        retrieval_response = bm25_search(effective_query, top_k=top_k)

    elif strategy_to_use == "vector":
        effective_query = question.strip()
        retrieval_response = vector_search(effective_query, top_k=top_k)

    elif strategy_to_use == "hybrid":
        effective_query = question.strip()
        retrieval_response = hybrid_search(effective_query, top_k=top_k)

    else:
        raise ValueError(f"Unsupported retrieval strategy: {strategy_to_use}")

    normalized_results = _normalize_results_for_display(retrieval_response.results)

    debug_payload = {}
    if debug:
        debug_payload = {
            "result_count": len(normalized_results),
            "source_labels": [
                format_source_label(result.path, result.chunk_id)
                for result in normalized_results
            ],
            "scores": [result.score for result in normalized_results],
        }

    return RetrievalServiceResponse(
        original_query=question,
        effective_query=effective_query,
        strategy_requested=strategy_to_use,
        strategy_used=retrieval_response.strategy,
        top_k=top_k,
        results=normalized_results,
        debug=debug_payload,
    )
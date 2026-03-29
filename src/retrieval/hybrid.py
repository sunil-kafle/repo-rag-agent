# Hybrid retrieval using Reciprocal Rank Fusion (RRF).
# This combines lexical BM25 and vector retrieval without assuming that
# their raw scores are directly comparable.

from __future__ import annotations

from src.config import settings
from src.exceptions import RetrievalError
from src.retrieval.base import RetrievalResponse, RetrievalResult
from src.retrieval.lexical import bm25_search
from src.retrieval.vector import vector_search


def hybrid_search(
    query: str,
    top_k: int = 5,
    lexical_k: int = 10,
    vector_k: int = 10,
    rrf_k: int | None = None,
) -> RetrievalResponse:
    """
    Run hybrid retrieval by combining BM25 and vector results via RRF.

    Args:
        query: natural-language search query
        top_k: final number of fused results to return
        lexical_k: number of BM25 candidates to consider before fusion
        vector_k: number of vector candidates to consider before fusion
        rrf_k: RRF smoothing constant; defaults to settings.rrf_k

    Returns:
        RetrievalResponse with fused hybrid-ranked results
    """
    if not isinstance(query, str):
        raise RetrievalError("Query must be a string.")

    query = query.strip()
    if not query:
        return RetrievalResponse(query=query, strategy="hybrid", top_k=top_k, results=[])

    if top_k < 1:
        raise RetrievalError("top_k must be at least 1.")
    if lexical_k < 1 or vector_k < 1:
        raise RetrievalError("lexical_k and vector_k must be at least 1.")

    rrf_k = settings.rrf_k if rrf_k is None else rrf_k

    bm25_response = bm25_search(query=query, top_k=lexical_k)
    vector_response = vector_search(query=query, top_k=vector_k)

    fused_scores: dict[str, float] = {}
    fused_results: dict[str, RetrievalResult] = {}

    # Add BM25 ranks
    for result in bm25_response.results:
        fused_scores[result.doc_id] = fused_scores.get(result.doc_id, 0.0) + 1.0 / (
            rrf_k + result.rank
        )
        fused_results[result.doc_id] = result

    # Add vector ranks
    for result in vector_response.results:
        fused_scores[result.doc_id] = fused_scores.get(result.doc_id, 0.0) + 1.0 / (
            rrf_k + result.rank
        )

        # Keep the latest result object, but final score/rank will be replaced below
        fused_results[result.doc_id] = result

    ranked_doc_ids = sorted(
        fused_scores.keys(),
        key=lambda doc_id: fused_scores[doc_id],
        reverse=True,
    )

    results: list[RetrievalResult] = []

    for final_rank, doc_id in enumerate(ranked_doc_ids[:top_k], start=1):
        base_result = fused_results[doc_id]

        results.append(
            RetrievalResult(
                doc_id=base_result.doc_id,
                path=base_result.path,
                chunk_id=base_result.chunk_id,
                content=base_result.content,
                score=float(fused_scores[doc_id]),
                source_method="hybrid",
                rank=final_rank,
                metadata={
                    "rrf_k": rrf_k,
                },
            )
        )

    return RetrievalResponse(
        query=query,
        strategy="hybrid",
        top_k=top_k,
        results=results,
    )
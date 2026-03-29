# BM25 lexical retrieval implementation.
# This module uses saved retrieval artifacts and returns standardized
# retrieval responses for downstream services and agents.

from __future__ import annotations

import math
from collections import Counter
from typing import Any

from src.artifacts import load_retrieval_artifacts
from src.exceptions import RetrievalError
from src.retrieval.base import RetrievalResponse, RetrievalResult


BM25_K1 = 1.5
BM25_B = 0.75


def _build_result(
    doc: dict[str, Any],
    score: float,
    rank: int,
) -> RetrievalResult:
    """Convert a raw document dict into a standardized retrieval result."""
    return RetrievalResult(
        doc_id=doc["doc_id"],
        path=doc["path"],
        chunk_id=str(doc["chunk_id"]),
        content=doc["content"],
        score=float(score),
        source_method="bm25",
        rank=rank,
        metadata={},
    )


def bm25_search(query: str, top_k: int = 5) -> RetrievalResponse:
    """
    Run BM25 lexical retrieval against the saved artifact state.

    Args:
        query: lexical search query
        top_k: number of results to return

    Returns:
        RetrievalResponse with ranked RetrievalResult objects
    """
    if not isinstance(query, str):
        raise RetrievalError("Query must be a string.")

    query = query.strip()
    if not query:
        return RetrievalResponse(query=query, strategy="bm25", top_k=top_k, results=[])

    if top_k < 1:
        raise RetrievalError("top_k must be at least 1.")

    artifacts = load_retrieval_artifacts()

    documents = artifacts.documents
    inverted_index = artifacts.inverted_index
    doc_freq = artifacts.doc_freq
    doc_lengths = artifacts.doc_lengths
    avg_doc_length = artifacts.avg_doc_length
    doc_lookup = artifacts.doc_lookup

    if not documents:
        raise RetrievalError("No documents available for retrieval.")

    from src.retrieval.query import tokenize

    query_tokens = tokenize(query)
    if not query_tokens:
        return RetrievalResponse(query=query, strategy="bm25", top_k=top_k, results=[])

    query_terms = Counter(query_tokens)
    n_docs = len(documents)

    candidate_doc_ids = set()
    for term in query_terms:
        candidate_doc_ids.update(inverted_index.get(term, set()))

    scored_results: list[tuple[str, float]] = []

    for doc_id in candidate_doc_ids:
        doc = doc_lookup.get(doc_id)
        if doc is None:
            continue

        doc_tokens = doc.get("tokens")
        if doc_tokens is None:
            raise RetrievalError(
                f"Document {doc_id} is missing tokens. "
                "documents.json must include tokenized content for lexical retrieval."
            )

        doc_term_counts = Counter(doc_tokens)
        doc_len = doc_lengths[doc_id]

        score = 0.0
        for term in query_terms:
            tf = doc_term_counts.get(term, 0)
            if tf == 0:
                continue

            df = doc_freq.get(term, 0)
            if df == 0:
                continue

            idf = math.log(1 + ((n_docs - df + 0.5) / (df + 0.5)))
            numerator = tf * (BM25_K1 + 1)
            denominator = tf + BM25_K1 * (1 - BM25_B + BM25_B * (doc_len / avg_doc_length))
            score += idf * (numerator / denominator)

        if score > 0:
            scored_results.append((doc_id, score))

    scored_results.sort(key=lambda item: item[1], reverse=True)

    results = [
        _build_result(doc_lookup[doc_id], score=score, rank=rank)
        for rank, (doc_id, score) in enumerate(scored_results[:top_k], start=1)
    ]

    return RetrievalResponse(
        query=query,
        strategy="bm25",
        top_k=top_k,
        results=results,
    )
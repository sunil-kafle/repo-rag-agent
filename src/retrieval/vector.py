# Vector retrieval implementation using sentence-transformers embeddings.
# This module loads the saved embedding matrix and computes cosine-style
# similarity via dot product because embeddings are normalized.

from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer

from src.artifacts import load_retrieval_artifacts
from src.config import settings
from src.exceptions import RetrievalError
from src.retrieval.base import RetrievalResponse, RetrievalResult


@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    """
    Load and cache the sentence-transformers model used for query embeddings.
    """
    return SentenceTransformer(settings.embedding_model_name)


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
        source_method="vector",
        rank=rank,
        metadata={},
    )


def vector_search(query: str, top_k: int = 5) -> RetrievalResponse:
    """
    Run vector retrieval against the saved embedding matrix.

    Args:
        query: natural-language search query
        top_k: number of results to return

    Returns:
        RetrievalResponse with ranked RetrievalResult objects
    """
    if not isinstance(query, str):
        raise RetrievalError("Query must be a string.")

    query = query.strip()
    if not query:
        return RetrievalResponse(query=query, strategy="vector", top_k=top_k, results=[])

    if top_k < 1:
        raise RetrievalError("top_k must be at least 1.")

    artifacts = load_retrieval_artifacts()
    documents = artifacts.documents
    embedding_matrix = artifacts.embedding_matrix

    if not documents:
        raise RetrievalError("No documents available for retrieval.")

    if embedding_matrix.shape[0] != len(documents):
        raise RetrievalError(
            "Embedding matrix row count does not match number of documents."
        )

    model = get_embedding_model()

    query_embedding = model.encode(
        [query],
        normalize_embeddings=True,
        show_progress_bar=False,
    )[0]

    if not isinstance(query_embedding, np.ndarray):
        query_embedding = np.array(query_embedding, dtype=float)

    scores = embedding_matrix @ query_embedding
    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for rank, idx in enumerate(top_indices, start=1):
        doc = documents[int(idx)]
        score = float(scores[int(idx)])
        results.append(_build_result(doc=doc, score=score, rank=rank))

    return RetrievalResponse(
        query=query,
        strategy="vector",
        top_k=top_k,
        results=results,
    )
# Canonical retrieval contracts and shared data models.
# These models keep BM25, vector, and hybrid retrieval outputs consistent.

from __future__ import annotations

from typing import Any, Literal, Protocol

from pydantic import BaseModel, Field


RetrievalMethod = Literal["bm25", "vector", "hybrid"]


class Document(BaseModel):
    """Canonical document/chunk representation used across the project."""

    id: str
    path: str
    chunk_id: str
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievalResult(BaseModel):
    """Standardized retrieval output for any search strategy."""

    doc_id: str
    path: str
    chunk_id: str
    content: str
    score: float
    source_method: RetrievalMethod
    rank: int
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievalResponse(BaseModel):
    """Container for a retrieval run and its returned results."""

    query: str
    strategy: RetrievalMethod
    top_k: int
    results: list[RetrievalResult] = Field(default_factory=list)


class Retriever(Protocol):
    """Protocol that all retriever implementations should follow."""

    def retrieve(self, query: str, top_k: int = 5) -> RetrievalResponse:
        ...
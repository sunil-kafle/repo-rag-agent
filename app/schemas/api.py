# API request/response schemas for the FastAPI layer.
# These models keep the app boundary explicit and stable.

from __future__ import annotations

from pydantic import BaseModel, Field

from src.retrieval.base import RetrievalMethod, RetrievalResult


class AskRequest(BaseModel):
    """Incoming question request for the main ask endpoint."""

    question: str = Field(..., min_length=1, description="User question")
    top_k: int = Field(default=5, ge=1, le=20)
    strategy: RetrievalMethod = Field(default="bm25")
    debug: bool = Field(default=False)


class AskResponse(BaseModel):
    """Main answer payload returned by the ask endpoint."""

    answer: str
    citations: list[str] = Field(default_factory=list)
    strategy_used: RetrievalMethod
    formulated_query: str | None = None
    retrieved_results: list[RetrievalResult] = Field(default_factory=list)


class HealthResponse(BaseModel):
    """Simple health response for readiness checks."""

    status: str = "ok"
    app_name: str


class ErrorResponse(BaseModel):
    """Standard API error payload."""

    detail: str

    # Debug retrieval request/response schemas.

class DebugRetrieveRequest(BaseModel):
    """Request model for retrieval-only debugging."""

    question: str = Field(..., min_length=1, description="User question")
    top_k: int = Field(default=5, ge=1, le=20)
    strategy: RetrievalMethod = Field(default="bm25")


class DebugRetrieveResponse(BaseModel):
    """Response model for retrieval-only debugging."""

    original_query: str
    effective_query: str
    strategy_used: RetrievalMethod
    top_k: int
    results: list[RetrievalResult] = Field(default_factory=list)
    debug: dict = Field(default_factory=dict)
# Health/readiness endpoint for the FastAPI app.
# This is the simplest route and confirms the app is wired correctly.

from fastapi import APIRouter

from app.schemas.api import HealthResponse
from src.config import settings

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    """Return a simple application health response."""
    return HealthResponse(
        status="ok",
        app_name=settings.app_name,
    )
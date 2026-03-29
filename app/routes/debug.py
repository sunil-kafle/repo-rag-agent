# Debug retrieval endpoint for inspecting retrieval behavior without
# running the answer-generation agent.

from fastapi import APIRouter, HTTPException

from app.schemas.api import DebugRetrieveRequest, DebugRetrieveResponse
from app.services.retrieval_service import retrieve_context

router = APIRouter(tags=["debug"])


@router.post("/debug/retrieve", response_model=DebugRetrieveResponse)
def debug_retrieve(request: DebugRetrieveRequest) -> DebugRetrieveResponse:
    """
    Return retrieval-only results for debugging and strategy comparison.
    """
    try:
        retrieval_response = retrieve_context(
            question=request.question,
            strategy=request.strategy,
            top_k=request.top_k,
            debug=True,
        )

        return DebugRetrieveResponse(
            original_query=retrieval_response.original_query,
            effective_query=retrieval_response.effective_query,
            strategy_used=retrieval_response.strategy_used,
            top_k=retrieval_response.top_k,
            results=retrieval_response.results,
            debug=retrieval_response.debug,
        )

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
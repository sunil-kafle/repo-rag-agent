# Main ask endpoint for retrieval-backed question answering.

from fastapi import APIRouter, HTTPException

from app.schemas.api import AskRequest, AskResponse
from app.services.agent_service import generate_answer
from app.services.retrieval_service import retrieve_context

router = APIRouter(tags=["ask"])


@router.post("/ask", response_model=AskResponse)
async def ask_question(request: AskRequest) -> AskResponse:
    """
    Return an agent-generated answer. In debug mode, also expose
    retrieval details from the retrieval service.
    """
    try:
        agent_response = await generate_answer(request.question)

        retrieval_response = retrieve_context(
            question=request.question,
            strategy=request.strategy,
            top_k=request.top_k,
            debug=request.debug,
        )

        retrieved_results = retrieval_response.results if request.debug else []

        return AskResponse(
            answer=agent_response.answer,
            citations=agent_response.citations,
            strategy_used=retrieval_response.strategy_used,
            formulated_query=retrieval_response.effective_query if request.debug else None,
            retrieved_results=retrieved_results,
        )

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
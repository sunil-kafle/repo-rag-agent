# Test the FastAPI ask endpoint at the API contract level.
# We mock the agent and retrieval services so the test stays stable and fast.

from fastapi.testclient import TestClient

from app.main import app
from app.services.agent_service import AgentServiceResponse
from app.services.retrieval_service import RetrievalServiceResponse
from src.retrieval.base import RetrievalResult

client = TestClient(app)


def test_ask_endpoint_returns_expected_shape(monkeypatch) -> None:
    async def fake_generate_answer(question: str) -> AgentServiceResponse:
        return AgentServiceResponse(
            answer="Mocked agent answer.",
            citations=["articles/text_comparison_examples.md"],
        )

    def fake_retrieve_context(
        question: str,
        strategy: str,
        top_k: int,
        debug: bool,
    ) -> RetrievalServiceResponse:
        fake_result = RetrievalResult(
            doc_id="doc-1",
            path="articles/text_comparison_examples.md",
            chunk_id="0",
            content="Mock content",
            score=1.23,
            source_method="bm25",
            rank=1,
            metadata={},
        )

        return RetrievalServiceResponse(
            original_query=question,
            effective_query="embeddings work",
            strategy_requested=strategy,
            strategy_used="bm25",
            top_k=top_k,
            results=[fake_result],
            debug={"result_count": 1},
        )

    monkeypatch.setattr("app.routes.ask.generate_answer", fake_generate_answer)
    monkeypatch.setattr("app.routes.ask.retrieve_context", fake_retrieve_context)

    response = client.post(
        "/ask",
        json={
            "question": "How do embeddings work in this repo?",
            "top_k": 3,
            "strategy": "bm25",
            "debug": True,
        },
    )

    assert response.status_code == 200

    payload = response.json()
    assert payload["answer"] == "Mocked agent answer."
    assert payload["citations"] == ["articles/text_comparison_examples.md"]
    assert payload["strategy_used"] == "bm25"
    assert payload["formulated_query"] == "embeddings work"
    assert len(payload["retrieved_results"]) == 1


def test_ask_endpoint_hides_debug_fields_when_debug_false(monkeypatch) -> None:
    async def fake_generate_answer(question: str) -> AgentServiceResponse:
        return AgentServiceResponse(
            answer="Mocked agent answer.",
            citations=["articles/text_comparison_examples.md"],
        )

    def fake_retrieve_context(
        question: str,
        strategy: str,
        top_k: int,
        debug: bool,
    ) -> RetrievalServiceResponse:
        fake_result = RetrievalResult(
            doc_id="doc-1",
            path="articles/text_comparison_examples.md",
            chunk_id="0",
            content="Mock content",
            score=1.23,
            source_method="bm25",
            rank=1,
            metadata={},
        )

        return RetrievalServiceResponse(
            original_query=question,
            effective_query="embeddings work",
            strategy_requested=strategy,
            strategy_used="bm25",
            top_k=top_k,
            results=[fake_result],
            debug={},
        )

    monkeypatch.setattr("app.routes.ask.generate_answer", fake_generate_answer)
    monkeypatch.setattr("app.routes.ask.retrieve_context", fake_retrieve_context)

    response = client.post(
        "/ask",
        json={
            "question": "How do embeddings work in this repo?",
            "top_k": 3,
            "strategy": "bm25",
            "debug": False,
        },
    )

    assert response.status_code == 200

    payload = response.json()
    assert payload["formulated_query"] is None
    assert payload["retrieved_results"] == []
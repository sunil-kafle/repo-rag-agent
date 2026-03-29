# Tool wrappers exposed to the runtime agent.

from __future__ import annotations

from typing import Any

from app.services.retrieval_service import retrieve_context
from src.retrieval.formatting import build_github_blob_url

REPO_BASE_URL = "https://github.com/openai/openai-cookbook"


def text_search(query: str, top_k: int = 3) -> list[dict[str, Any]]:
    """
    Search the indexed repository and return compact, citation-friendly results.
    Default runtime policy uses BM25 with query formulation through the
    retrieval service.
    """
    retrieval_response = retrieve_context(
        question=query,
        strategy="bm25",
        top_k=top_k,
        debug=False,
    )

    formatted_results: list[dict[str, Any]] = []

    for result in retrieval_response.results:
        formatted_results.append(
            {
                "path": result.path,
                "url": build_github_blob_url(REPO_BASE_URL, result.path),
                "chunk_id": result.chunk_id,
                "score": round(float(result.score), 6),
                "content": result.content[:500],
            }
        )

    return formatted_results
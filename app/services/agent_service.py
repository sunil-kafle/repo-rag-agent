# Agent orchestration service for the FastAPI layer.

from __future__ import annotations

import re
from urllib.parse import urlparse
from pydantic import BaseModel, Field

from src.agent.builder import build_repo_agent
from src.retrieval.formatting import normalize_repo_path


class AgentServiceResponse(BaseModel):
    """Structured runtime response returned by the agent service."""

    answer: str
    citations: list[str] = Field(default_factory=list)


def _extract_citations(answer: str) -> list[str]:
    """
    Extract citations from markdown links in the generated answer.
    Prefer the GitHub URL path when available because it is more stable
    than the visible markdown label.
    """
    matches = re.findall(r"\[([^\]]+)\]\(([^)]+)\)", answer)
    extracted: list[str] = []

    for label, url in matches:
        citation = label.strip()

        if "github.com" in url and "/blob/" in url:
            parsed = urlparse(url)
            path = parsed.path

            # Example:
            # /openai/openai-cookbook/blob/main/articles/text_comparison_examples.md
            if "/blob/" in path:
                citation = path.split("/blob/", 1)[1]
                parts = citation.split("/", 1)
                if len(parts) == 2:
                    citation = parts[1]

        extracted.append(normalize_repo_path(citation))

    seen = set()
    ordered = []
    for item in extracted:
        if item not in seen:
            ordered.append(item)
            seen.add(item)

    return ordered


async def generate_answer(question: str) -> AgentServiceResponse:
    """
    Run the runtime agent and return the generated answer plus extracted citations.
    """
    agent = build_repo_agent()
    result = await agent.run(user_prompt=question)

    answer_text = result.output.strip()
    citations = _extract_citations(answer_text)

    return AgentServiceResponse(
        answer=answer_text,
        citations=citations,
    )
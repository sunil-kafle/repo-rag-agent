from __future__ import annotations

import os
from functools import lru_cache

from pydantic_ai import Agent

from src.agent.prompts import REPO_QA_SYSTEM_PROMPT
from src.agent.tools import text_search
from src.config import settings


@lru_cache(maxsize=1)
def build_repo_agent() -> Agent:
    api_key = settings.resolved_openai_api_key
    if not api_key:
        raise ValueError(
            "OpenAI API key not found. Set OPENAI_API_KEY or provide C:\\projects\\OPEN_AI.txt"
        )

    os.environ["OPENAI_API_KEY"] = api_key

    return Agent(
        model=f"openai:{settings.openai_model}",
        instructions=REPO_QA_SYSTEM_PROMPT,
        tools=[text_search],
        name="repo_agent_runtime",
    )
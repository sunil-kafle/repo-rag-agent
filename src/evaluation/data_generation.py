# Utilities for generating evaluation questions from repository documents.
# These helpers move the notebook question-generation flow into a reusable Python module.

from __future__ import annotations

import json
from typing import Any

from pydantic_ai import Agent

from src.evaluation.schemas import GeneratedQuestions


question_generation_prompt = """
You are helping create test questions for an AI agent that answers questions about a code/documentation repository.

Based on the provided repository content, generate realistic questions a user might ask.

The questions should:
- Be natural and varied in style
- Range from simple to moderately complex
- Include both technical and conceptual questions
- Be answerable from the provided repository content

Generate one question for each record.
""".strip()


def build_question_generator(model_name: str = "openai:gpt-4o-mini") -> Agent:
    """
    Create and return the question generation agent.
    """
    return Agent(
        name="question_generator",
        instructions=question_generation_prompt,
        model=model_name,
        output_type=GeneratedQuestions,
    )


def build_generation_prompt(documents: list[dict[str, Any]]) -> str:
    """
    Convert sampled repository documents into the JSON prompt sent to the generator.
    Mirrors the notebook behavior by sending document content records as JSON.
    """
    prompt_docs = [doc["content"] for doc in documents]
    return json.dumps(prompt_docs, ensure_ascii=False)


async def generate_questions_from_documents(
    question_generator: Agent,
    documents: list[dict[str, Any]],
) -> GeneratedQuestions:
    """
    Generate evaluation questions from sampled repository documents.
    """
    prompt = build_generation_prompt(documents)
    result = await question_generator.run(prompt)
    return result.output
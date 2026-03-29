# Utilities for judge-based evaluation of saved agent logs.
# These helpers move the notebook evaluation flow into a reusable Python module.

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic_ai import Agent

from src.evaluation.schemas import EvaluationChecklist


# Prompt used by the evaluation judge.
evaluation_prompt = """
Use this checklist to evaluate the quality of an AI agent's answer (<ANSWER>) to a user question (<QUESTION>).
We also include the entire log (<LOG>) for analysis.

For each item, check if the condition is met.

Checklist:

- instructions_follow: The agent followed the user's instructions (in <INSTRUCTIONS>)
- instructions_avoid: The agent avoided doing things it was told not to do
- answer_relevant: The response directly addresses the user's question
- answer_clear: The answer is clear and correct
- answer_citations: The response includes proper citations or sources when required
- completeness: The response is complete and covers all key aspects of the request
- tool_call_search: Is the search tool invoked?

Output true/false for each check and provide a short explanation for your judgment.
""".strip()


# Template for the evaluation input sent to the judge.
user_prompt_format = """
<INSTRUCTIONS>{instructions}</INSTRUCTIONS>
<QUESTION>{question}</QUESTION>
<ANSWER>{answer}</ANSWER>
<LOG>{log}</LOG>
""".strip()


def build_eval_agent(model_name: str = "openai:gpt-4o-mini") -> Agent:
    """
    Create and return the evaluation judge agent.
    """
    return Agent(
        name="eval_agent",
        model=model_name,
        instructions=evaluation_prompt,
        output_type=EvaluationChecklist,
    )


def load_log_file(log_file: str | Path) -> dict[str, Any]:
    """
    Load one saved JSON log file and attach its file path to the record.
    """
    log_path = Path(log_file)

    with log_path.open("r", encoding="utf-8") as f_in:
        log_data = json.load(f_in)

    log_data["log_file"] = log_path
    return log_data


def simplify_log_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Reduce log payload size before sending to the judge.

    Behavior mirrors the notebook logic:
    - remove timestamps from user prompts
    - remove tool_call_id from tool-call parts
    - remove tool_call_id, metadata, timestamp from tool-return parts
    - replace tool-return content with a placeholder
    - remove id from text parts
    """
    log_simplified: list[dict[str, Any]] = []

    for message in messages:
        parts: list[dict[str, Any]] = []

        for original_part in message["parts"]:
            part = original_part.copy()
            kind = part["part_kind"]

            if kind == "user-prompt":
                part.pop("timestamp", None)

            if kind == "tool-call":
                part.pop("tool_call_id", None)

            if kind == "tool-return":
                part.pop("tool_call_id", None)
                part.pop("metadata", None)
                part.pop("timestamp", None)
                part["content"] = "RETURN_RESULTS_REDACTED"

            if kind == "text":
                part.pop("id", None)

            parts.append(part)

        log_simplified.append(
            {
                "kind": message["kind"],
                "parts": parts,
            }
        )

    return log_simplified


async def evaluate_log_record(
    eval_agent: Agent,
    log_record: dict[str, Any],
) -> EvaluationChecklist:
    """
    Evaluate one saved log record with the judge agent.
    """
    messages = log_record["messages"]

    instructions = log_record["system_prompt"]
    question = messages[0]["parts"][0]["content"]
    answer = messages[-1]["parts"][0]["content"]

    log_simplified = simplify_log_messages(messages)
    log_json = json.dumps(log_simplified)

    user_prompt = user_prompt_format.format(
        instructions=instructions,
        question=question,
        answer=answer,
        log=log_json,
    )

    result = await eval_agent.run(user_prompt)
    return result.output
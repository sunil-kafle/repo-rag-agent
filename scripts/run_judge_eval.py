# Run judge-based evaluation over saved agent logs.
# This script loads saved logs, evaluates them with the judge agent,
# flattens the results, summarizes pass rates, and saves outputs to data/eval.

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

from src.config import settings
from src.evaluation.judge import build_eval_agent, evaluate_log_record
from src.evaluation.logging_utils import load_log_file
from src.evaluation.metrics import checklist_to_row, summarize_evaluation_rows


ROWS_OUTPUT_PATH = settings.eval_dir / "judge_eval_rows.json"
SUMMARY_OUTPUT_PATH = settings.eval_dir / "judge_eval_summary.json"


def load_openai_api_key() -> None:
    """
    Ensure OPENAI_API_KEY is available for Pydantic AI / OpenAI client creation.
    """
    if os.getenv("OPENAI_API_KEY"):
        return

    resolved_key = settings.resolved_openai_api_key
    if not resolved_key:
        raise ValueError(
            "OPENAI_API_KEY is not set and could not be resolved from the external key file."
        )

    os.environ["OPENAI_API_KEY"] = resolved_key


async def main() -> None:
    settings.eval_dir.mkdir(parents=True, exist_ok=True)
    settings.logs_dir.mkdir(parents=True, exist_ok=True)

    load_openai_api_key()

    log_files = sorted(settings.logs_dir.glob("*.json"))

    if not log_files:
        print(f"No log files found in: {settings.logs_dir}")
        return

    eval_agent = build_eval_agent(model_name=f"openai:{settings.openai_model}")

    rows = []

    for log_file in log_files:
        log_record = load_log_file(log_file)
        checklist = await evaluate_log_record(eval_agent, log_record)

        messages = log_record["messages"]
        question = messages[0]["parts"][0]["content"]
        answer = messages[-1]["parts"][0]["content"]

        row = checklist_to_row(
            question=question,
            answer=answer,
            checklist=checklist,
            file=Path(log_record["log_file"]).name,
            source=log_record.get("source"),
            log_path=log_record.get("log_file"),
            metadata={
                "agent_name": log_record.get("agent_name"),
                "provider": log_record.get("provider"),
                "model": log_record.get("model"),
            },
        )
        rows.append(row)

    summary = summarize_evaluation_rows(rows)

    with ROWS_OUTPUT_PATH.open("w", encoding="utf-8") as f_out:
        json.dump([row.model_dump() for row in rows], f_out, ensure_ascii=False, indent=2)

    with SUMMARY_OUTPUT_PATH.open("w", encoding="utf-8") as f_out:
        json.dump(summary.model_dump(), f_out, ensure_ascii=False, indent=2)

    print(f"Evaluated log files: {len(rows)}")
    print(f"Saved rows to: {ROWS_OUTPUT_PATH}")
    print(f"Saved summary to: {SUMMARY_OUTPUT_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
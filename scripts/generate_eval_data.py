# Script to generate evaluation questions from sampled repository documents.
# This moves the notebook question-generation flow into a runnable project script.

from __future__ import annotations

import json
import os
import random
from pathlib import Path

from src.artifacts import load_retrieval_artifacts
from src.evaluation.data_generation import (
    build_question_generator,
    generate_questions_from_documents,
)


OUTPUT_DIR = Path("data/eval")
OUTPUT_PATH = OUTPUT_DIR / "generated_questions.json"
OPENAI_KEY_FILE = Path(r"C:\projects\OPEN_AI.txt")


def load_openai_api_key() -> None:
    """
    Load OPENAI_API_KEY from the external key file used by the project.
    """
    if os.getenv("OPENAI_API_KEY"):
        return

    if not OPENAI_KEY_FILE.exists():
        raise FileNotFoundError(f"OpenAI key file not found: {OPENAI_KEY_FILE}")

    api_key = OPENAI_KEY_FILE.read_text(encoding="utf-8").strip()

    if "=" in api_key and api_key.startswith("OPENAI_API_KEY"):
        api_key = api_key.split("=", 1)[1].strip().strip('"').strip("'")

    os.environ["OPENAI_API_KEY"] = api_key


async def main(sample_size: int = 5) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    load_openai_api_key()

    artifacts = load_retrieval_artifacts()
    documents = artifacts.documents

    if not documents:
        raise ValueError("No documents were loaded from retrieval artifacts.")

    if sample_size > len(documents):
        sample_size = len(documents)

    sample_docs = random.sample(documents, sample_size)

    question_generator = build_question_generator()
    generated = await generate_questions_from_documents(
        question_generator=question_generator,
        documents=sample_docs,
    )

    payload = {
        "sample_size": sample_size,
        "questions": generated.questions,
    }

    with OUTPUT_PATH.open("w", encoding="utf-8") as f_out:
        json.dump(payload, f_out, ensure_ascii=False, indent=2)

    print(f"Saved generated questions to: {OUTPUT_PATH}")
    print(f"Question count: {len(generated.questions)}")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
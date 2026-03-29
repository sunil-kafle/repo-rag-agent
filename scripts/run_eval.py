# Run retrieval evaluation for the refactored project.
# This compares multiple retrieval strategies using simple IR metrics
# and optionally saves the results to disk.

from __future__ import annotations

import json
from pathlib import Path

from src.evaluation.ir_metrics import evaluate_search_quality, summarize_ir_metrics
from src.retrieval.hybrid import hybrid_search
from src.retrieval.lexical import bm25_search
from src.retrieval.query import formulate_search_query
from src.retrieval.vector import vector_search


OUTPUT_DIR = Path("data/eval")
OUTPUT_PATH = OUTPUT_DIR / "retrieval_eval_summary.json"


# Evaluation-only wrapper so BM25 is tested with the same
# query-formulation policy used by the runtime service.
def bm25_search_with_formulation(query: str, top_k: int = 5):
    effective_query = formulate_search_query(query)
    return bm25_search(effective_query, top_k=top_k)


SEARCH_TEST_QUERIES = [
    (
        "How do embeddings work in this repo?",
        ["articles/text_comparison_examples.md"],
    ),
    (
        "Can I fine-tune embeddings in this repo?",
        ["articles/text_comparison_examples.md", "examples/Customizing_embeddings.ipynb"],
    ),
    (
        "How is semantic search done here?",
        ["articles/text_comparison_examples.md"],
    ),
    (
        "What notebook shows how to customize embeddings?",
        ["examples/Customizing_embeddings.ipynb", "articles/text_comparison_examples.md"],
    ),
    (
        "What does this repo say about using embeddings for recommendations?",
        ["examples/Recommendation_using_embeddings.ipynb", "articles/text_comparison_examples.md"],
    ),
]


def main(save_json: bool = True) -> None:
    methods = {
        "bm25_raw": bm25_search,
        "bm25_formulated": bm25_search_with_formulation,
        "vector": vector_search,
        "hybrid": hybrid_search,
    }

    all_results = []

    print("=== Retrieval Evaluation Start ===\n")

    for method_name, search_function in methods.items():
        rows = evaluate_search_quality(
            search_function=search_function,
            test_queries=SEARCH_TEST_QUERIES,
            top_k=5,
        )
        summary = summarize_ir_metrics(rows, method=method_name)

        print(
            f"{summary.method}: "
            f"hit_rate={summary.hit_rate:.2f}, "
            f"mrr={summary.mrr:.3f}, "
            f"total_queries={summary.total_queries}"
        )

        all_results.append(
            {
                "method": summary.method,
                "hit_rate": summary.hit_rate,
                "mrr": summary.mrr,
                "total_queries": summary.total_queries,
                "rows": [row.model_dump() for row in rows],
            }
        )

    if save_json:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        with OUTPUT_PATH.open("w", encoding="utf-8") as f_out:
            json.dump(all_results, f_out, ensure_ascii=False, indent=2)
        print(f"\nSaved evaluation summary to: {OUTPUT_PATH}")

    print("\n=== Retrieval Evaluation Complete ===")


if __name__ == "__main__":
    main()
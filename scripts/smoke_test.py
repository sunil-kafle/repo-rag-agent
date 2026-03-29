# Minimal end-to-end smoke test for the refactored project.
# This checks:
# - config loading
# - artifact loading
# - BM25 retrieval
# - retrieval service
# - agent tool search
#
# It is intentionally lightweight and does not start FastAPI.

from src.artifacts import load_retrieval_artifacts
from src.config import settings
from src.retrieval.lexical import bm25_search
from app.services.retrieval_service import retrieve_context
from src.agent.tools import text_search


def main() -> None:
    print("=== Smoke Test Start ===")

    # Check config
    print("\n[1] Config")
    print("Project root:", settings.project_root)
    print("Artifacts dir:", settings.artifacts_dir)
    print("OpenAI key available:", bool(settings.resolved_openai_api_key))

    # Check artifact loading
    print("\n[2] Artifact loading")
    artifacts = load_retrieval_artifacts()
    print("Documents:", len(artifacts.documents))
    print("Embedding matrix shape:", artifacts.embedding_matrix.shape)
    print("Doc lookup size:", len(artifacts.doc_lookup))
    print("Average doc length:", round(artifacts.avg_doc_length, 2))

    # Check BM25 retrieval
    print("\n[3] BM25 retrieval")
    bm25_response = bm25_search("openai embeddings", top_k=3)
    print("Strategy:", bm25_response.strategy)
    print("Results:", len(bm25_response.results))
    if bm25_response.results:
        top = bm25_response.results[0]
        print("Top result path:", top.path)
        print("Top result chunk:", top.chunk_id)
        print("Top result score:", round(top.score, 4))

    # Check retrieval service
    print("\n[4] Retrieval service")
    retrieval_response = retrieve_context(
        question="How do embeddings work in this repo?",
        strategy="bm25",
        top_k=3,
        debug=True,
    )
    print("Original query:", retrieval_response.original_query)
    print("Effective query:", retrieval_response.effective_query)
    print("Strategy used:", retrieval_response.strategy_used)
    print("Results:", len(retrieval_response.results))

    # Check agent tool
    print("\n[5] Agent tool")
    tool_results = text_search("How do embeddings work in this repo?", top_k=2)
    print("Tool results:", len(tool_results))
    if tool_results:
        print("First tool path:", tool_results[0]["path"])
        print("First tool url:", tool_results[0]["url"])

    print("\n=== Smoke Test Passed ===")


if __name__ == "__main__":
    main()
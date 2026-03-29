# Test BM25 retrieval against the real saved artifacts.

from src.retrieval.lexical import bm25_search


def test_bm25_returns_response_object() -> None:
    response = bm25_search("openai embeddings", top_k=3)

    assert response.strategy == "bm25"
    assert response.top_k == 3
    assert len(response.results) <= 3


def test_bm25_returns_ranked_results() -> None:
    response = bm25_search("openai embeddings", top_k=3)

    assert len(response.results) > 0

    ranks = [result.rank for result in response.results]
    assert ranks == [1, 2, 3][: len(ranks)]

    scores = [result.score for result in response.results]
    assert scores == sorted(scores, reverse=True)


def test_bm25_returns_expected_fields() -> None:
    response = bm25_search("openai embeddings", top_k=1)

    assert len(response.results) == 1

    result = response.results[0]
    assert result.doc_id
    assert result.path
    assert result.chunk_id is not None
    assert result.content
    assert isinstance(result.score, float)
    assert result.source_method == "bm25"


def test_bm25_empty_query_returns_no_results() -> None:
    response = bm25_search("   ", top_k=3)

    assert response.strategy == "bm25"
    assert response.results == []
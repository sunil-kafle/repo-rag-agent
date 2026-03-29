# Test lexical query formulation behavior.

import pytest

from src.retrieval.query import formulate_search_query, tokenize
from src.exceptions import QueryFormulationError


def test_tokenize_lowercases_and_extracts_words() -> None:
    tokens = tokenize("Hello, OpenAI Repo_1!")

    assert tokens == ["hello", "openai", "repo_1"]


def test_formulate_search_query_removes_stopwords_and_preserves_order() -> None:
    query = "How do embeddings work in this repo?"
    result = formulate_search_query(query)

    assert result == "embeddings work"


def test_formulate_search_query_handles_multiple_terms() -> None:
    query = "How is semantic search done with postgres?"
    result = formulate_search_query(query)

    assert result == "semantic search done with postgres"


def test_formulate_search_query_deduplicates_terms() -> None:
    query = "embeddings embeddings embeddings semantic semantic"
    result = formulate_search_query(query)

    assert result == "embeddings semantic"


def test_formulate_search_query_raises_on_empty_string() -> None:
    with pytest.raises(QueryFormulationError):
        formulate_search_query("   ")


def test_formulate_search_query_raises_on_non_string() -> None:
    with pytest.raises(QueryFormulationError):
        formulate_search_query(None)  # type: ignore[arg-type]
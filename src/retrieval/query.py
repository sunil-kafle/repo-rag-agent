# Query formulation helpers for lexical retrieval.
# These utilities keep user questions readable for the agent while
# producing a compact keyword-style query for BM25 retrieval.

from __future__ import annotations

import re

from src.exceptions import QueryFormulationError


TOKEN_PATTERN = re.compile(r"\b[a-zA-Z0-9_]+\b")

STOPWORDS = {
    "how", "what", "why", "when", "where", "which", "who",
    "do", "does", "did", "is", "are", "was", "were", "can", "could",
    "in", "on", "at", "to", "for", "of", "and", "or", "the", "a", "an",
    "this", "that", "these", "those", "it", "they", "them", "their",
    "repo", "repository"
}


def tokenize(text: str) -> list[str]:
    """
    Normalize text into simple lowercase alphanumeric tokens.
    """
    if not isinstance(text, str):
        raise QueryFormulationError("Input text must be a string.")

    return TOKEN_PATTERN.findall(text.lower())


def formulate_search_query(user_query: str, max_terms: int = 6) -> str:
    """
    Convert a natural-language user question into a compact lexical query.

    Rules:
    - lowercase tokenization
    - remove common stopwords
    - drop very short terms
    - preserve original order
    - deduplicate repeated terms
    - fallback to original stripped query if nothing survives

    Example:
        "How do embeddings work in this repo?"
        -> "embeddings work"
    """
    if not isinstance(user_query, str):
        raise QueryFormulationError("user_query must be a string.")

    stripped_query = user_query.strip()
    if not stripped_query:
        raise QueryFormulationError("user_query must be a non-empty string.")

    tokens = tokenize(stripped_query)

    filtered_terms = []
    seen = set()

    for token in tokens:
        if token in STOPWORDS:
            continue
        if len(token) <= 2:
            continue
        if token not in seen:
            filtered_terms.append(token)
            seen.add(token)

    if not filtered_terms:
        return stripped_query

    return " ".join(filtered_terms[:max_terms])
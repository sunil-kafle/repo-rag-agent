# Centralized artifact loading for retrieval.
# This module restores the saved retrieval state from disk and validates
# that the expected artifact files and structures are present.

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from src.config import settings
from src.exceptions import ArtifactNotFoundError, ArtifactValidationError


@dataclass
class RetrievalArtifacts:
    """Container for all loaded retrieval artifacts."""

    documents: list[dict[str, Any]]
    embedding_matrix: np.ndarray
    inverted_index: dict[str, set[str]] | dict[str, list[str]] | dict[str, Any]
    doc_freq: dict[str, int]
    doc_lengths: dict[str, int]
    avg_doc_length: float
    doc_lookup: dict[str, dict[str, Any]]


def _ensure_file_exists(path: Path) -> None:
    """Raise a clear error if a required artifact file is missing."""
    if not path.exists():
        raise ArtifactNotFoundError(f"Required artifact file not found: {path}")


def _validate_documents(documents: Any) -> list[dict[str, Any]]:
    """Validate the loaded documents artifact structure."""
    if not isinstance(documents, list):
        raise ArtifactValidationError("documents.json must contain a list of documents.")

    required_keys = {"doc_id", "path", "chunk_id", "content"}

    for i, doc in enumerate(documents):
        if not isinstance(doc, dict):
            raise ArtifactValidationError(f"Document at index {i} is not a dictionary.")

        missing = required_keys - set(doc.keys())
        if missing:
            raise ArtifactValidationError(
                f"Document at index {i} is missing required keys: {sorted(missing)}"
            )

    return documents


def _validate_lexical_artifacts(lexical_artifacts: Any) -> dict[str, Any]:
    """Validate the lexical index artifact structure."""
    if not isinstance(lexical_artifacts, dict):
        raise ArtifactValidationError("lexical_index.pkl must contain a dictionary.")

    required_keys = {"inverted_index", "doc_freq", "doc_lengths", "avg_doc_length"}
    missing = required_keys - set(lexical_artifacts.keys())

    if missing:
        raise ArtifactValidationError(
            f"lexical_index.pkl is missing required keys: {sorted(missing)}"
        )

    return lexical_artifacts


def load_documents(documents_path: Path | None = None) -> list[dict[str, Any]]:
    """Load and validate saved document metadata/content."""
    path = documents_path or settings.documents_path
    _ensure_file_exists(path)

    with path.open("r", encoding="utf-8") as f:
        documents = json.load(f)

    return _validate_documents(documents)


def load_embedding_matrix(embedding_matrix_path: Path | None = None) -> np.ndarray:
    """Load the saved embedding matrix."""
    path = embedding_matrix_path or settings.embedding_matrix_path
    _ensure_file_exists(path)

    matrix = np.load(path)

    if not isinstance(matrix, np.ndarray):
        raise ArtifactValidationError("embedding_matrix.npy did not load as a NumPy array.")

    if matrix.ndim != 2:
        raise ArtifactValidationError(
            f"embedding_matrix.npy must be 2-dimensional, got shape {matrix.shape}."
        )

    return matrix


def load_lexical_artifacts(lexical_index_path: Path | None = None) -> dict[str, Any]:
    """Load and validate saved lexical retrieval artifacts."""
    path = lexical_index_path or settings.lexical_index_path
    _ensure_file_exists(path)

    with path.open("rb") as f:
        lexical_artifacts = pickle.load(f)

    return _validate_lexical_artifacts(lexical_artifacts)


def build_doc_lookup(documents: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Build a doc_id -> document mapping."""
    doc_lookup = {doc["doc_id"]: doc for doc in documents}

    if len(doc_lookup) != len(documents):
        raise ArtifactValidationError(
            "Duplicate doc_id values found while building doc_lookup."
        )

    return doc_lookup


@lru_cache(maxsize=1)
def load_retrieval_artifacts() -> RetrievalArtifacts:
    """
    Load all retrieval artifacts once and cache the result for reuse.

    Returns:
        RetrievalArtifacts: fully restored retrieval state
    """
    documents = load_documents()
    embedding_matrix = load_embedding_matrix()
    lexical_artifacts = load_lexical_artifacts()
    doc_lookup = build_doc_lookup(documents)

    if embedding_matrix.shape[0] != len(documents):
        raise ArtifactValidationError(
            "Embedding matrix row count does not match number of documents: "
            f"{embedding_matrix.shape[0]} vs {len(documents)}"
        )

    return RetrievalArtifacts(
        documents=documents,
        embedding_matrix=embedding_matrix,
        inverted_index=lexical_artifacts["inverted_index"],
        doc_freq=lexical_artifacts["doc_freq"],
        doc_lengths=lexical_artifacts["doc_lengths"],
        avg_doc_length=float(lexical_artifacts["avg_doc_length"]),
        doc_lookup=doc_lookup,
    )
# Shared formatting helpers for retrieval output, citations, and repo paths.
# This keeps source display logic separate from retrieval logic.

from __future__ import annotations

from src.exceptions import FormattingError


def normalize_repo_path(path: str) -> str:
    """
    Remove repository-root prefixes so displayed citations are cleaner.

    Example:
        openai-cookbook-main/examples/file.md -> examples/file.md
    """
    if not path or not isinstance(path, str):
        raise FormattingError("Path must be a non-empty string.")

    normalized = path.replace("\\", "/").strip()

    prefixes = [
        "openai-cookbook-main/",
        "./",
    ]

    for prefix in prefixes:
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix):]

    return normalized


def build_github_blob_url(
    repo_url: str,
    path: str,
    branch: str = "main",
) -> str:
    """
    Build a GitHub blob URL for a repository file path.

    Example:
        repo_url=https://github.com/openai/openai-cookbook
        path=examples/file.md
        -> https://github.com/openai/openai-cookbook/blob/main/examples/file.md
    """
    if not repo_url or not isinstance(repo_url, str):
        raise FormattingError("repo_url must be a non-empty string.")

    normalized_path = normalize_repo_path(path)
    clean_repo_url = repo_url.rstrip("/")

    return f"{clean_repo_url}/blob/{branch}/{normalized_path}"


def format_source_label(path: str, chunk_id: str | None = None) -> str:
    """
    Create a readable source label for citations or debug output.
    """
    normalized_path = normalize_repo_path(path)

    if chunk_id:
        return f"{normalized_path}::{chunk_id}"

    return normalized_path


def make_snippet(text: str, max_chars: int = 220) -> str:
    """
    Create a compact single-line snippet for debug or display use.
    """
    if not text or not isinstance(text, str):
        raise FormattingError("Text must be a non-empty string.")

    cleaned = " ".join(text.split())

    if len(cleaned) <= max_chars:
        return cleaned

    return cleaned[: max_chars - 3].rstrip() + "..."
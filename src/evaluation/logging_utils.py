# Utilities for building and saving evaluation/runtime logs.
# These helpers are based on the notebook logging flow but moved into a
# reusable Python module.

from __future__ import annotations

import json
import secrets
from datetime import datetime
from pathlib import Path
from typing import Any

from src.config import settings


def serializer(obj: Any) -> str:
    """
    JSON serializer helper for datetime objects.
    """
    if isinstance(obj, datetime):
        return obj.isoformat()

    raise TypeError(f"Type {type(obj)} not serializable")


def ensure_logs_dir() -> Path:
    """
    Ensure the logs directory exists and return it.
    """
    settings.logs_dir.mkdir(parents=True, exist_ok=True)
    return settings.logs_dir


def simplify_log_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Simplify verbose agent message logs to reduce token usage for evaluation.
    This mirrors the notebook logic used before judge-based evaluation.
    """
    simplified: list[dict[str, Any]] = []

    for message in messages:
        parts = []

        for original_part in message.get("parts", []):
            part = original_part.copy()
            kind = part.get("part_kind")

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

        simplified.append(
            {
                "kind": message.get("kind"),
                "parts": parts,
            }
        )

    return simplified


def build_log_entry(
    *,
    agent_name: str,
    system_prompt: str,
    provider: str,
    model: str,
    tools: list[str],
    messages: list[dict[str, Any]],
    source: str = "user",
) -> dict[str, Any]:
    """
    Build a normalized log record for one agent interaction.
    """
    return {
        "agent_name": agent_name,
        "system_prompt": system_prompt,
        "provider": provider,
        "model": model,
        "tools": tools,
        "messages": messages,
        "source": source,
    }


def _extract_timestamp_from_messages(messages: list[dict[str, Any]]) -> datetime:
    """
    Extract a timestamp from the last message. Supports either datetime objects
    or ISO-formatted strings.
    """
    if not messages:
        return datetime.now()

    last_message = messages[-1]
    parts = last_message.get("parts", [])

    for part in reversed(parts):
        ts = part.get("timestamp")
        if isinstance(ts, datetime):
            return ts
        if isinstance(ts, str):
            return datetime.fromisoformat(ts.replace("Z", "+00:00"))

    return datetime.now()


def save_log_entry(entry: dict[str, Any], filename_prefix: str | None = None) -> Path:
    """
    Save a log entry as JSON in the logs directory and return the file path.
    """
    log_dir = ensure_logs_dir()
    ts_obj = _extract_timestamp_from_messages(entry.get("messages", []))
    ts_str = ts_obj.strftime("%Y%m%d_%H%M%S")
    rand_hex = secrets.token_hex(3)

    prefix = filename_prefix or entry.get("agent_name", "agent")
    filename = f"{prefix}_{ts_str}_{rand_hex}.json"
    filepath = log_dir / filename

    with filepath.open("w", encoding="utf-8") as f_out:
        json.dump(entry, f_out, indent=2, default=serializer)

    return filepath


def load_log_file(log_file: str | Path) -> dict[str, Any]:
    """
    Load a saved JSON log file and attach its path into the returned record.
    """
    path = Path(log_file)

    with path.open("r", encoding="utf-8") as f_in:
        log_data = json.load(f_in)

    log_data["log_file"] = str(path)
    return log_data
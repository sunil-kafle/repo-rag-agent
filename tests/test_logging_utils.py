# Tests for evaluation logging helpers.

from datetime import datetime
from pathlib import Path

from src.evaluation.logging_utils import (
    build_log_entry,
    ensure_logs_dir,
    load_log_file,
    save_log_entry,
    serializer,
    simplify_log_messages,
)


def test_serializer_returns_isoformat_for_datetime():
    dt = datetime(2026, 3, 28, 12, 30, 15)
    assert serializer(dt) == "2026-03-28T12:30:15"


def test_build_log_entry_creates_expected_structure():
    messages = [
        {
            "kind": "request",
            "parts": [
                {
                    "part_kind": "user-prompt",
                    "content": "How do embeddings work?",
                }
            ],
        }
    ]

    entry = build_log_entry(
        agent_name="repo_agent_v2",
        system_prompt="Test system prompt",
        provider="openai",
        model="gpt-4o-mini",
        tools=["text_search"],
        messages=messages,
        source="user",
    )

    assert entry["agent_name"] == "repo_agent_v2"
    assert entry["system_prompt"] == "Test system prompt"
    assert entry["provider"] == "openai"
    assert entry["model"] == "gpt-4o-mini"
    assert entry["tools"] == ["text_search"]
    assert entry["messages"] == messages
    assert entry["source"] == "user"


def test_simplify_log_messages_removes_heavy_fields_and_redacts_tool_returns():
    messages = [
        {
            "kind": "request",
            "parts": [
                {
                    "part_kind": "user-prompt",
                    "content": "How do embeddings work?",
                    "timestamp": "2026-03-28T10:00:00",
                },
                {
                    "part_kind": "tool-call",
                    "tool_name": "text_search",
                    "tool_call_id": "abc123",
                    "args": {"query": "embeddings"},
                },
                {
                    "part_kind": "tool-return",
                    "tool_call_id": "abc123",
                    "content": [{"path": "articles/test.md"}],
                    "metadata": {"x": 1},
                    "timestamp": "2026-03-28T10:00:01",
                },
                {
                    "part_kind": "text",
                    "id": "msg1",
                    "content": "Embeddings are used for similarity.",
                },
            ],
        }
    ]

    simplified = simplify_log_messages(messages)

    assert len(simplified) == 1
    parts = simplified[0]["parts"]

    assert parts[0] == {
        "part_kind": "user-prompt",
        "content": "How do embeddings work?",
    }
    assert parts[1] == {
        "part_kind": "tool-call",
        "tool_name": "text_search",
        "args": {"query": "embeddings"},
    }
    assert parts[2] == {
        "part_kind": "tool-return",
        "content": "RETURN_RESULTS_REDACTED",
    }
    assert parts[3] == {
        "part_kind": "text",
        "content": "Embeddings are used for similarity.",
    }


def test_save_log_entry_and_load_log_file_round_trip(tmp_path, monkeypatch):
    from src.config import settings

    monkeypatch.setattr(settings, "logs_dir", tmp_path)

    entry = build_log_entry(
        agent_name="repo_agent_v2",
        system_prompt="Test system prompt",
        provider="openai",
        model="gpt-4o-mini",
        tools=["text_search"],
        messages=[
            {
                "kind": "response",
                "parts": [
                    {
                        "part_kind": "text",
                        "content": "Embeddings are used for similarity.",
                        "timestamp": "2026-03-28T10:00:01",
                    }
                ],
            }
        ],
        source="user",
    )

    saved_path = save_log_entry(entry)

    assert saved_path.exists()
    assert saved_path.parent == tmp_path

    loaded = load_log_file(saved_path)

    assert loaded["agent_name"] == "repo_agent_v2"
    assert loaded["system_prompt"] == "Test system prompt"
    assert loaded["provider"] == "openai"
    assert loaded["model"] == "gpt-4o-mini"
    assert loaded["tools"] == ["text_search"]
    assert loaded["source"] == "user"
    assert loaded["log_file"] == str(saved_path)


def test_ensure_logs_dir_returns_existing_directory(tmp_path, monkeypatch):
    from src.config import settings

    monkeypatch.setattr(settings, "logs_dir", tmp_path)

    result = ensure_logs_dir()

    assert isinstance(result, Path)
    assert result == tmp_path
    assert result.exists()
    assert result.is_dir()
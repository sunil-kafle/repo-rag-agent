# Tests for evaluation judge helpers that do not require live API calls.

from src.evaluation.judge import load_log_file, simplify_log_messages


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


def test_load_log_file_adds_log_file_path(tmp_path):
    log_path = tmp_path / "sample_log.json"
    log_path.write_text(
        """
        {
          "agent_name": "repo_agent_v2",
          "system_prompt": "Test prompt",
          "messages": [
            {
              "kind": "request",
              "parts": [
                {
                  "part_kind": "user-prompt",
                  "content": "How do embeddings work?"
                }
              ]
            }
          ],
          "source": "user"
        }
        """.strip(),
        encoding="utf-8",
    )

    log_record = load_log_file(log_path)

    assert log_record["agent_name"] == "repo_agent_v2"
    assert log_record["system_prompt"] == "Test prompt"
    assert log_record["source"] == "user"
    assert log_record["log_file"] == log_path
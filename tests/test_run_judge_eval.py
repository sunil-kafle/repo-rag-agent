# Tests for the judge-evaluation script.

import json

from scripts import run_judge_eval
from src.evaluation.schemas import EvaluationCheck, EvaluationChecklist


def test_main_saves_judge_eval_outputs(monkeypatch, tmp_path, capsys):
    logs_dir = tmp_path / "logs"
    eval_dir = tmp_path / "eval"
    logs_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)

    log_path = logs_dir / "sample_log.json"
    log_payload = {
        "agent_name": "repo_agent_v2",
        "system_prompt": "Test prompt",
        "provider": "openai",
        "model": "gpt-4o-mini",
        "tools": ["text_search"],
        "messages": [
            {
                "kind": "request",
                "parts": [
                    {
                        "part_kind": "user-prompt",
                        "content": "How do embeddings work?",
                    }
                ],
            },
            {
                "kind": "response",
                "parts": [
                    {
                        "part_kind": "text",
                        "content": "They are vector representations used for similarity.",
                    }
                ],
            },
        ],
        "source": "user",
    }
    log_path.write_text(json.dumps(log_payload, indent=2), encoding="utf-8")

    monkeypatch.setattr(run_judge_eval.settings, "logs_dir", logs_dir)
    monkeypatch.setattr(run_judge_eval.settings, "eval_dir", eval_dir)
    monkeypatch.setattr(run_judge_eval, "ROWS_OUTPUT_PATH", eval_dir / "judge_eval_rows.json")
    monkeypatch.setattr(run_judge_eval, "SUMMARY_OUTPUT_PATH", eval_dir / "judge_eval_summary.json")
    monkeypatch.setattr(run_judge_eval, "load_openai_api_key", lambda: None)
    monkeypatch.setattr(run_judge_eval, "build_eval_agent", lambda model_name: object())

    async def fake_evaluate_log_record(eval_agent, log_record):
        return EvaluationChecklist(
            checklist=[
                EvaluationCheck(
                    check_name="answer_relevant",
                    justification="Direct answer.",
                    check_pass=True,
                ),
                EvaluationCheck(
                    check_name="answer_clear",
                    justification="Clear enough.",
                    check_pass=True,
                ),
            ],
            summary="Good result",
        )

    monkeypatch.setattr(run_judge_eval, "evaluate_log_record", fake_evaluate_log_record)

    import asyncio
    asyncio.run(run_judge_eval.main())

    output = capsys.readouterr().out
    rows_path = eval_dir / "judge_eval_rows.json"
    summary_path = eval_dir / "judge_eval_summary.json"

    assert "Evaluated log files: 1" in output
    assert f"Saved rows to: {rows_path}" in output
    assert f"Saved summary to: {summary_path}" in output
    assert rows_path.exists()
    assert summary_path.exists()

    rows_payload = json.loads(rows_path.read_text(encoding="utf-8"))
    summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))

    assert len(rows_payload) == 1
    assert rows_payload[0]["question"] == "How do embeddings work?"
    assert rows_payload[0]["answer"] == "They are vector representations used for similarity."
    assert rows_payload[0]["summary"] == "Good result"
    assert rows_payload[0]["checks"] == {
        "answer_relevant": True,
        "answer_clear": True,
    }

    assert summary_payload["total_rows"] == 1
    assert summary_payload["metric_passes"] == {
        "answer_relevant": 1,
        "answer_clear": 1,
    }
    assert summary_payload["metric_totals"] == {
        "answer_relevant": 1,
        "answer_clear": 1,
    }
    assert summary_payload["metric_pass_rates"] == {
        "answer_relevant": 1.0,
        "answer_clear": 1.0,
    }
    assert summary_payload["overall_pass_rate"] == 1.0


def test_main_handles_empty_logs_directory(monkeypatch, tmp_path, capsys):
    logs_dir = tmp_path / "logs"
    eval_dir = tmp_path / "eval"
    logs_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(run_judge_eval.settings, "logs_dir", logs_dir)
    monkeypatch.setattr(run_judge_eval.settings, "eval_dir", eval_dir)
    monkeypatch.setattr(run_judge_eval, "ROWS_OUTPUT_PATH", eval_dir / "judge_eval_rows.json")
    monkeypatch.setattr(run_judge_eval, "SUMMARY_OUTPUT_PATH", eval_dir / "judge_eval_summary.json")
    monkeypatch.setattr(run_judge_eval, "load_openai_api_key", lambda: None)

    import asyncio
    asyncio.run(run_judge_eval.main())

    output = capsys.readouterr().out

    assert f"No log files found in: {logs_dir}" in output
    assert not (eval_dir / "judge_eval_rows.json").exists()
    assert not (eval_dir / "judge_eval_summary.json").exists()
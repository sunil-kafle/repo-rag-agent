# Tests for evaluation metrics helpers.

from src.evaluation.metrics import (
    build_rows_from_evaluations,
    checklist_to_row,
    summarize_evaluation_rows,
)
from src.evaluation.schemas import EvaluationCheck, EvaluationChecklist


def test_checklist_to_row_flattens_checks_and_justifications():
    checklist = EvaluationChecklist(
        checklist=[
            EvaluationCheck(
                check_name="answer_relevant",
                justification="Direct answer.",
                check_pass=True,
            ),
            EvaluationCheck(
                check_name="answer_clear",
                justification="Clear wording.",
                check_pass=False,
            ),
        ],
        summary="Mixed result",
    )

    row = checklist_to_row(
        question="How do embeddings work?",
        answer="They are vector representations.",
        checklist=checklist,
        source="user",
    )

    assert row.question == "How do embeddings work?"
    assert row.answer == "They are vector representations."
    assert row.summary == "Mixed result"
    assert row.checks == {
        "answer_relevant": True,
        "answer_clear": False,
    }
    assert row.justifications == {
        "answer_relevant": "Direct answer.",
        "answer_clear": "Clear wording.",
    }
    assert row.source == "user"


def test_summarize_evaluation_rows_computes_pass_counts_and_rates():
    checklist_1 = EvaluationChecklist(
        checklist=[
            EvaluationCheck(
                check_name="answer_relevant",
                justification="Direct answer.",
                check_pass=True,
            ),
            EvaluationCheck(
                check_name="answer_clear",
                justification="Clear wording.",
                check_pass=False,
            ),
        ],
        summary="Mixed result",
    )

    checklist_2 = EvaluationChecklist(
        checklist=[
            EvaluationCheck(
                check_name="answer_relevant",
                justification="Relevant enough.",
                check_pass=True,
            ),
            EvaluationCheck(
                check_name="answer_clear",
                justification="Very clear.",
                check_pass=True,
            ),
        ],
        summary="Good result",
    )

    row_1 = checklist_to_row(
        question="Q1",
        answer="A1",
        checklist=checklist_1,
    )
    row_2 = checklist_to_row(
        question="Q2",
        answer="A2",
        checklist=checklist_2,
    )

    summary = summarize_evaluation_rows([row_1, row_2])

    assert summary.total_rows == 2
    assert summary.metric_passes == {
        "answer_relevant": 2,
        "answer_clear": 1,
    }
    assert summary.metric_totals == {
        "answer_relevant": 2,
        "answer_clear": 2,
    }
    assert summary.metric_pass_rates == {
        "answer_relevant": 1.0,
        "answer_clear": 0.5,
    }
    assert summary.overall_pass_rate == 0.75


def test_build_rows_from_evaluations_converts_notebook_style_payload():
    evaluations = [
        {
            "question": "How do embeddings work?",
            "answer": "They are used for similarity.",
            "summary": "Looks good",
            "checklist": [
                {
                    "check_name": "answer_relevant",
                    "check_pass": True,
                    "justification": "Direct answer.",
                },
                {
                    "check_name": "answer_clear",
                    "check_pass": False,
                    "justification": "Too brief.",
                },
            ],
            "file": "log1.json",
            "source": "user",
            "log_path": "logs/log1.json",
            "metadata": {"run_id": "abc123"},
        }
    ]

    rows = build_rows_from_evaluations(evaluations)

    assert len(rows) == 1
    row = rows[0]

    assert row.file == "log1.json"
    assert row.question == "How do embeddings work?"
    assert row.answer == "They are used for similarity."
    assert row.summary == "Looks good"
    assert row.checks == {
        "answer_relevant": True,
        "answer_clear": False,
    }
    assert row.justifications == {
        "answer_relevant": "Direct answer.",
        "answer_clear": "Too brief.",
    }
    assert row.source == "user"
    assert row.log_path == "logs/log1.json"
    assert row.metadata == {"run_id": "abc123"}
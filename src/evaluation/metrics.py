# Utilities for flattening evaluation results and computing aggregate summaries.
# These helpers move the notebook metric rollup logic into a reusable Python module.

from __future__ import annotations

from collections import defaultdict

from src.evaluation.schemas import EvaluationChecklist, EvaluationRow, EvaluationSummary


def checklist_to_row(
    *,
    question: str,
    answer: str,
    checklist: EvaluationChecklist,
    file: str | None = None,
    source: str | None = None,
    log_path: str | None = None,
    metadata: dict | None = None,
) -> EvaluationRow:
    """
    Convert one structured checklist into one flattened evaluation row.
    """
    return EvaluationRow.from_checklist(
        question=question,
        answer=answer,
        checklist=checklist,
        file=file,
        source=source,
        log_path=log_path,
        metadata=metadata,
    )


def summarize_evaluation_rows(rows: list[EvaluationRow]) -> EvaluationSummary:
    """
    Aggregate pass counts, totals, and pass rates across flattened evaluation rows.
    """
    metric_counts = defaultdict(lambda: {"pass": 0, "total": 0})

    for row in rows:
        for check_name, check_pass in row.checks.items():
            metric_counts[check_name]["total"] += 1
            if check_pass:
                metric_counts[check_name]["pass"] += 1

    metric_passes: dict[str, int] = {}
    metric_totals: dict[str, int] = {}
    metric_pass_rates: dict[str, float] = {}

    total_passes = 0
    total_checks = 0

    for check_name, counts in metric_counts.items():
        passes = counts["pass"]
        total = counts["total"]
        pass_rate = passes / total if total > 0 else 0.0

        metric_passes[check_name] = passes
        metric_totals[check_name] = total
        metric_pass_rates[check_name] = pass_rate

        total_passes += passes
        total_checks += total

    overall_pass_rate = total_passes / total_checks if total_checks > 0 else None

    return EvaluationSummary(
        total_rows=len(rows),
        metric_passes=metric_passes,
        metric_totals=metric_totals,
        metric_pass_rates=metric_pass_rates,
        overall_pass_rate=overall_pass_rate,
    )


def build_rows_from_evaluations(
    evaluations: list[dict],
) -> list[EvaluationRow]:
    """
    Build flattened evaluation rows from a notebook-style evaluations payload.

    Expected input shape per item:
    {
        "question": "...",
        "answer": "...",
        "summary": "...",
        "checklist": [
            {
                "check_name": "...",
                "check_pass": True,
                "justification": "..."
            }
        ],
        "file": "...",        # optional
        "source": "...",      # optional
        "log_path": "...",    # optional
        "metadata": {...}     # optional
    }
    """
    rows: list[EvaluationRow] = []

    for item in evaluations:
        checks = {
            check["check_name"]: check["check_pass"]
            for check in item.get("checklist", [])
        }
        justifications = {
            check["check_name"]: check.get("justification", "")
            for check in item.get("checklist", [])
        }

        row = EvaluationRow(
            file=item.get("file"),
            question=item["question"],
            answer=item["answer"],
            summary=item.get("summary"),
            checks=checks,
            justifications=justifications,
            source=item.get("source"),
            log_path=item.get("log_path"),
            metadata=item.get("metadata", {}),
        )
        rows.append(row)

    return rows
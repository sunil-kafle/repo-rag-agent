# Structured evaluation schemas for the evaluation layer.
# These models freeze the contracts used by judge-based evaluation,
# generated evaluation questions, flattened evaluation rows,
# and aggregate evaluation summaries.

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class EvaluationCheck(BaseModel):
    """
    One checklist item returned by the evaluation judge.
    """
    model_config = ConfigDict(extra="forbid")

    check_name: str = Field(
        ...,
        description="Canonical evaluation check name."
    )
    justification: str = Field(
        ...,
        description="Short explanation for the pass/fail judgment."
    )
    check_pass: bool = Field(
        ...,
        description="Whether the check passed."
    )


class EvaluationChecklist(BaseModel):
    """
    Full structured judge output for one evaluated interaction.
    """
    model_config = ConfigDict(extra="forbid")

    checklist: list[EvaluationCheck] = Field(
        default_factory=list,
        description="List of evaluation checks."
    )
    summary: str = Field(
        ...,
        description="Overall evaluation summary."
    )


class GeneratedQuestions(BaseModel):
    """
    Structured output for repository-based question generation.
    """
    model_config = ConfigDict(extra="forbid")

    questions: list[str] = Field(
        default_factory=list,
        description="Generated evaluation questions."
    )


class EvaluationRow(BaseModel):
    """
    Flattened evaluation record for one question-answer interaction.
    Useful for JSON export, dataframe conversion, and metric aggregation.
    """
    model_config = ConfigDict(extra="forbid")

    file: str | None = Field(
        default=None,
        description="Associated log filename when available."
    )
    question: str = Field(
        ...,
        description="Original question."
    )
    answer: str = Field(
        ...,
        description="Agent answer."
    )
    summary: str | None = Field(
        default=None,
        description="Optional evaluation summary."
    )
    checks: dict[str, bool] = Field(
        default_factory=dict,
        description="Flattened check_name to pass/fail mapping."
    )
    justifications: dict[str, str] = Field(
        default_factory=dict,
        description="Flattened check_name to justification mapping."
    )
    source: str | None = Field(
        default=None,
        description="Origin of the run such as user or ai-generated."
    )
    log_path: str | None = Field(
        default=None,
        description="Path to the saved log file when available."
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Optional extra evaluation metadata."
    )

    @classmethod
    def from_checklist(
        cls,
        *,
        question: str,
        answer: str,
        checklist: EvaluationChecklist,
        file: str | None = None,
        source: str | None = None,
        log_path: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "EvaluationRow":
        """
        Build one flattened evaluation row from a structured checklist.
        """
        return cls(
            file=file,
            question=question,
            answer=answer,
            summary=checklist.summary,
            checks={item.check_name: item.check_pass for item in checklist.checklist},
            justifications={item.check_name: item.justification for item in checklist.checklist},
            source=source,
            log_path=log_path,
            metadata=metadata or {},
        )


class EvaluationSummary(BaseModel):
    """
    Aggregate evaluation summary across many evaluated rows.
    """
    model_config = ConfigDict(extra="forbid")

    total_rows: int = Field(
        ...,
        description="Number of evaluated rows included in the summary."
    )
    metric_passes: dict[str, int] = Field(
        default_factory=dict,
        description="Per-check pass counts."
    )
    metric_totals: dict[str, int] = Field(
        default_factory=dict,
        description="Per-check total counts."
    )
    metric_pass_rates: dict[str, float] = Field(
        default_factory=dict,
        description="Per-check pass rates."
    )
    overall_pass_rate: float | None = Field(
        default=None,
        description="Optional aggregate pass rate."
    )
    notes: str | None = Field(
        default=None,
        description="Optional human-readable summary notes."
    )
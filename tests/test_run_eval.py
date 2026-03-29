# Tests for the retrieval evaluation script.

import json

from scripts import run_eval
from src.evaluation.ir_metrics import IRMetricRow


def test_bm25_search_with_formulation_uses_formulated_query(monkeypatch):
    captured = {}

    def fake_formulate_search_query(query: str) -> str:
        captured["original_query"] = query
        return "embeddings work"

    def fake_bm25_search(query: str, top_k: int = 5):
        captured["effective_query"] = query
        captured["top_k"] = top_k
        return [{"path": "articles/text_comparison_examples.md"}]

    monkeypatch.setattr(run_eval, "formulate_search_query", fake_formulate_search_query)
    monkeypatch.setattr(run_eval, "bm25_search", fake_bm25_search)

    results = run_eval.bm25_search_with_formulation(
        "How do embeddings work in this repo?",
        top_k=3,
    )

    assert captured["original_query"] == "How do embeddings work in this repo?"
    assert captured["effective_query"] == "embeddings work"
    assert captured["top_k"] == 3
    assert results == [{"path": "articles/text_comparison_examples.md"}]


def test_main_saves_json_summary(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(run_eval, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(run_eval, "OUTPUT_PATH", tmp_path / "retrieval_eval_summary.json")

    monkeypatch.setattr(run_eval, "bm25_search", lambda query, top_k=5: [{"path": "articles/a.md"}])
    monkeypatch.setattr(run_eval, "bm25_search_with_formulation", lambda query, top_k=5: [{"path": "articles/a.md"}])
    monkeypatch.setattr(run_eval, "vector_search", lambda query, top_k=5: [{"path": "articles/a.md"}])
    monkeypatch.setattr(run_eval, "hybrid_search", lambda query, top_k=5: [{"path": "articles/a.md"}])

    def fake_evaluate_search_quality(search_function, test_queries, top_k=5):
        return [
            IRMetricRow(
                query="How do embeddings work in this repo?",
                expected_docs=["articles/a.md"],
                retrieved_paths=["articles/a.md"],
                hit=True,
                mrr=1.0,
            )
        ]

    def fake_summarize_ir_metrics(rows, method: str):
        class DummySummary:
            def __init__(self, method):
                self.method = method
                self.hit_rate = 1.0
                self.mrr = 1.0
                self.total_queries = len(rows)

        return DummySummary(method)

    monkeypatch.setattr(run_eval, "evaluate_search_quality", fake_evaluate_search_quality)
    monkeypatch.setattr(run_eval, "summarize_ir_metrics", fake_summarize_ir_metrics)

    run_eval.main(save_json=True)

    output = capsys.readouterr().out
    saved_path = tmp_path / "retrieval_eval_summary.json"

    assert "=== Retrieval Evaluation Start ===" in output
    assert "bm25_raw: hit_rate=1.00, mrr=1.000, total_queries=1" in output
    assert "bm25_formulated: hit_rate=1.00, mrr=1.000, total_queries=1" in output
    assert "vector: hit_rate=1.00, mrr=1.000, total_queries=1" in output
    assert "hybrid: hit_rate=1.00, mrr=1.000, total_queries=1" in output
    assert "Saved evaluation summary to:" in output
    assert "=== Retrieval Evaluation Complete ===" in output
    assert saved_path.exists()

    with saved_path.open("r", encoding="utf-8") as f_in:
        payload = json.load(f_in)

    assert len(payload) == 4
    assert payload[0]["method"] == "bm25_raw"
    assert payload[0]["hit_rate"] == 1.0
    assert payload[0]["mrr"] == 1.0
    assert payload[0]["total_queries"] == 1
    assert payload[0]["rows"][0]["query"] == "How do embeddings work in this repo?"


def test_main_can_skip_json_save(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(run_eval, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(run_eval, "OUTPUT_PATH", tmp_path / "retrieval_eval_summary.json")

    monkeypatch.setattr(run_eval, "bm25_search", lambda query, top_k=5: [{"path": "articles/a.md"}])
    monkeypatch.setattr(run_eval, "bm25_search_with_formulation", lambda query, top_k=5: [{"path": "articles/a.md"}])
    monkeypatch.setattr(run_eval, "vector_search", lambda query, top_k=5: [{"path": "articles/a.md"}])
    monkeypatch.setattr(run_eval, "hybrid_search", lambda query, top_k=5: [{"path": "articles/a.md"}])

    def fake_evaluate_search_quality(search_function, test_queries, top_k=5):
        return [
            IRMetricRow(
                query="How do embeddings work in this repo?",
                expected_docs=["articles/a.md"],
                retrieved_paths=["articles/a.md"],
                hit=True,
                mrr=1.0,
            )
        ]

    def fake_summarize_ir_metrics(rows, method: str):
        class DummySummary:
            def __init__(self, method):
                self.method = method
                self.hit_rate = 1.0
                self.mrr = 1.0
                self.total_queries = len(rows)

        return DummySummary(method)

    monkeypatch.setattr(run_eval, "evaluate_search_quality", fake_evaluate_search_quality)
    monkeypatch.setattr(run_eval, "summarize_ir_metrics", fake_summarize_ir_metrics)

    run_eval.main(save_json=False)

    output = capsys.readouterr().out
    saved_path = tmp_path / "retrieval_eval_summary.json"

    assert "=== Retrieval Evaluation Start ===" in output
    assert "=== Retrieval Evaluation Complete ===" in output
    assert not saved_path.exists()
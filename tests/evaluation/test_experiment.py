from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from support_graph.evaluation import experiment
from support_graph.evaluation.contracts import DatasetRef, ExperimentSnapshot


def _result(
    variant_id: str,
    *,
    doc: float,
    span: float,
    citation: float,
    e2e: float,
    latency_ms: float,
    failures: dict[str, int] | None = None,
) -> dict:
    variant = next(item for item in experiment.VARIANTS if item["id"] == variant_id)
    return {
        "variant": variant,
        "metrics": {
            "retrieval": {"answer": {"doc_recall_at_3": doc, "span_recall_at_5": span}},
            "generation": {
                "answer": {
                    "citation_coverage": citation,
                    "end_to_end_success_rate": e2e,
                    "rouge_l": 0.2,
                    "token_f1": 0.2,
                }
            },
            "latency_ms": {"average": latency_ms},
        },
        "failure_counts": failures or {},
    }


def test_classify_variant_accepts_primary_improvement_without_guardrail_break() -> None:
    control = _result(
        "control", doc=0.6, span=0.2, citation=0.2, e2e=0.2, latency_ms=1000
    )
    candidate = _result(
        "structured-query-rerank",
        doc=0.6,
        span=0.3,
        citation=0.25,
        e2e=0.2,
        latency_ms=1200,
    )

    status, reason = experiment.classify_variant(control, candidate)

    assert status == "worked"
    assert "reranker" in reason or "candidate pool" in reason


def test_classify_variant_rejects_doc_recall_floor_violation() -> None:
    control = _result(
        "control", doc=0.6, span=0.2, citation=0.2, e2e=0.2, latency_ms=1000
    )
    candidate = _result(
        "structured-query", doc=0.4, span=0.3, citation=0.2, e2e=0.2, latency_ms=900
    )

    status, _ = experiment.classify_variant(control, candidate)

    assert status == "didn't work"


def test_classify_variant_rejects_latency_without_material_gain() -> None:
    control = _result(
        "control", doc=0.6, span=0.2, citation=0.2, e2e=0.2, latency_ms=1000
    )
    candidate = _result(
        "structured-query-rerank-neighbors",
        doc=0.6,
        span=0.22,
        citation=0.21,
        e2e=0.2,
        latency_ms=40_000,
    )

    status, _ = experiment.classify_variant(control, candidate)

    assert status == "didn't work"


def test_variant_run_uses_tagged_hosted_experiment(monkeypatch, make_runtime_config):
    calls: list[dict] = []
    snapshot = ExperimentSnapshot(
        id="exp-variant",
        name="variant",
        dataset=DatasetRef("dataset", "dataset", "hash"),
        metadata={},
        results=(),
        url="https://smith/exp-variant",
    )

    async def fake_hosted(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(experiment=snapshot, aggregate_metrics={})

    monkeypatch.setattr(experiment, "run_hosted_evaluation", fake_hosted)
    variant = next(item for item in experiment.VARIANTS if item["id"] == "control")

    result = asyncio.run(
        experiment._run_variant(
            settings=SimpleNamespace(),
            examples=[{"example_id": "ex-1"}],
            domain="kubernetes",
            split="validation",
            subset_name="smoke",
            started_at=datetime(2026, 7, 16, tzinfo=timezone.utc),
            base_config=make_runtime_config(
                postgres_dsn="dsn",
                chat_model="chat",
                embedding_model="embed",
            ),
            variant=variant,
            run_graph_func=None,
            experiment_variant="control",
            notes="control",
            run_id_slug="control",
            subset_label="smoke",
            manifest_scope="smoke",
            gateway=object(),
            policy=object(),
            corpus_manifest_path=Path("manifest.json"),
            git_state=SimpleNamespace(),
            study_id="20260716-kubernetes-smoke-retrieval-ablation",
        )
    )

    assert calls[0]["experiment_metadata"] == {
        "study_type": "retrieval_ablation",
        "study_id": "20260716-kubernetes-smoke-retrieval-ablation",
        "variant": "control",
        "repetition": 1,
    }
    assert result["experiment_id"] == "exp-variant"
    assert result["experiment_url"] == "https://smith/exp-variant"


def test_write_experiment_summary_creates_markdown_note(
    tmp_path: Path,
    make_settings,
) -> None:
    settings = make_settings(project_root=tmp_path)
    started = datetime(2026, 3, 18, 15, 0, tzinfo=timezone.utc)
    control = _result(
        "control", doc=0.6, span=0.2, citation=0.2, e2e=0.2, latency_ms=1000
    )
    control["status"] = "control"
    control["status_reason"] = "Baseline anchor for comparison."
    improved = _result(
        "structured-query-rerank-neighbors",
        doc=0.6,
        span=0.3,
        citation=0.3,
        e2e=0.4,
        latency_ms=900,
        failures={"weak_citations": 0},
    )
    improved["status"] = "worked"
    improved["status_reason"] = "Neighbor expansion improved grounding after reranking."

    report = experiment.write_experiment_summary(
        settings=settings,
        domain="kubernetes",
        split="validation",
        limit=10,
        results=[control, improved],
        summary_timestamp=started,
    )

    path = report["artifact_paths"]["report"]
    content = path.read_text(encoding="utf-8")
    assert (
        report["report_id"] == "20260318-150000-kubernetes-smoke10-experiment-summary"
    )
    assert path.name == "report.md"
    assert "Metric Table" in content
    assert "Structured Query + Rerank + Neighbors" in content
    assert "Frozen-200" in content
    assert "Recommendation" in content

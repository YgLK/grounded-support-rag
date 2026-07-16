from __future__ import annotations

import json
from pathlib import Path

from dev_tools.embedding_map import embedding_map


def _record(
    chunk_id: str,
    doc_id: str,
    embedding: tuple[float, ...],
) -> embedding_map.EmbeddingRecord:
    return embedding_map.EmbeddingRecord(
        chunk_id=chunk_id,
        doc_id=doc_id,
        doc_title=f"Title {doc_id}",
        section_title="Section",
        token_count=3,
        path_prefix=embedding_map.path_prefix(doc_id),
        text=f"Text for {chunk_id}",
        embedding=embedding,
    )


def test_parse_pgvector_row_reads_embedding_and_metadata() -> None:
    row = {
        "embedding": "[0.1,0.2,0.3]",
        "text": "pod scheduling text",
        "metadata": {
            "chunk_id": "chunk-1",
            "doc_id": "concepts/workloads/pods",
            "doc_title": "Pods",
            "section_title": "What is a Pod?",
            "token_count": 42,
        },
    }

    record = embedding_map.parse_pgvector_row(row)

    assert record.chunk_id == "chunk-1"
    assert record.doc_id == "concepts/workloads/pods"
    assert record.path_prefix == "concepts/workloads"
    assert record.embedding == (0.1, 0.2, 0.3)
    assert record.token_count == 42


def test_projection_output_schema_and_jsonl_round_trip(tmp_path: Path) -> None:
    records = [
        _record("chunk-a", "concepts/workloads/pods", (1.0, 0.0, 0.0)),
        _record("chunk-b", "concepts/workloads/deployments", (0.9, 0.1, 0.0)),
        _record("chunk-c", "concepts/services-networking/service", (0.0, 1.0, 0.0)),
    ]
    jsonl_path = tmp_path / "embeddings.jsonl"

    embedding_map.write_embeddings_jsonl(records, jsonl_path)
    loaded = embedding_map.load_embeddings_jsonl(jsonl_path)
    points = embedding_map.project_pca(loaded)

    assert [point.chunk_id for point in points] == ["chunk-a", "chunk-b", "chunk-c"]
    assert all(
        isinstance(point.x, float) and isinstance(point.y, float) for point in points
    )


def test_summary_neighbor_purity_and_suspicious_neighbors(tmp_path: Path) -> None:
    records = [
        _record("chunk-a", "concepts/workloads/pods", (1.0, 0.0)),
        _record("chunk-b", "concepts/workloads/deployments", (0.95, 0.05)),
        _record("chunk-c", "tasks/debug/debug-cluster", (0.94, 0.06)),
    ]

    purity = embedding_map.path_prefix_purity(records, neighbor_count=1)
    suspicious = embedding_map.suspicious_neighbors(records)
    embedding_map.write_summary(
        records, tmp_path / "summary.md", projection_names=["pca"]
    )

    assert purity is not None
    assert 0.0 <= purity <= 1.0
    assert suspicious
    assert "diagnostic views, not retrieval metrics" in (
        tmp_path / "summary.md"
    ).read_text(encoding="utf-8")


def test_plotly_html_smoke(tmp_path: Path) -> None:
    records = [
        _record("chunk-a", "concepts/workloads/pods", (1.0, 0.0)),
        _record("chunk-b", "concepts/services-networking/service", (0.0, 1.0)),
    ]
    points = [
        embedding_map.ProjectionPoint(chunk_id="chunk-a", x=0.0, y=0.0),
        embedding_map.ProjectionPoint(chunk_id="chunk-b", x=1.0, y=1.0),
    ]
    output_path = tmp_path / "plot.html"

    embedding_map.write_plot_html(records, points, output_path, title="Fixture Plot")

    html = output_path.read_text(encoding="utf-8")
    assert "Fixture Plot" in html
    assert "Plotly.newPlot" in html


def test_eval_overlay_marks_expected_and_retrieved_chunks(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    prediction = {
        "expected_sources": ["concepts/workloads/pods"],
        "acceptable_sources": ["reference/glossary/pod"],
        "retrieval_ranked_chunks": [{"chunk_id": "retrieved-chunk"}],
        "citations": [{"chunk_id": "cited-chunk"}],
    }
    (run_dir / "predictions.jsonl").write_text(
        json.dumps(prediction) + "\n", encoding="utf-8"
    )
    records = [
        _record("expected-chunk", "concepts/workloads/pods", (1.0, 0.0)),
        _record("retrieved-chunk", "concepts/other", (0.0, 1.0)),
    ]

    overlay = embedding_map.load_eval_overlay(run_dir)
    updated = embedding_map.apply_overlay_status(records, overlay)

    assert updated[0].overlay_status == "expected"
    assert updated[1].overlay_status == "retrieved"

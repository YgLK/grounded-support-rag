from __future__ import annotations

import json
from datetime import datetime, timezone

from support_graph.evaluation.contracts import DatasetRef
from support_graph.evaluation.metadata import (
    GitState,
    build_experiment_metadata,
    file_sha256,
)


def test_build_experiment_metadata_captures_reproducibility_contract(
    make_runtime_config,
    tmp_path,
) -> None:
    manifest = tmp_path / "manifest.json"
    chunks = tmp_path / "chunks.jsonl"
    manifest.write_text('{"source":"kubernetes"}\n', encoding="utf-8")
    chunks.write_text('{"chunk_id":"c1"}\n', encoding="utf-8")

    metadata = build_experiment_metadata(
        config=make_runtime_config(
            chunk_artifact_path=chunks,
            prompt_version="v2",
            retrieval_top_k=5,
            retrieval_candidate_k=12,
        ),
        dataset=DatasetRef("d1", "support-graph/kubernetes/smoke/abc", "abc"),
        domain="kubernetes",
        subset="smoke",
        corpus_ref="kubernetes-current",
        corpus_manifest_path=manifest,
        evaluator_versions={"deterministic": "1", "rag_triad": "1"},
        judge_provider="openrouter",
        judge_model="openai/gpt-4.1-mini",
        git_state=GitState(sha="deadbeef", dirty=False),
        started_at=datetime(2026, 7, 16, tzinfo=timezone.utc),
        example_count=10,
    )

    assert metadata["dataset_name"] == "support-graph/kubernetes/smoke/abc"
    assert metadata["git_sha"] == "deadbeef"
    assert metadata["git_dirty"] is False
    assert metadata["retrieval_config"]["top_k"] == 5
    assert metadata["corpus_manifest_sha256"] == file_sha256(manifest)
    assert "langsmith_api_key" not in json.dumps(metadata)

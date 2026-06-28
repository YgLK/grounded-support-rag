from __future__ import annotations

from support_graph.ui.loaders import WorkbenchArtifactLoader

from tests.ui._helpers import (
    EXAMPLE_ID,
    RUN_ID,
    SECOND_RUN_ID,
    build_eval_run_artifact,
)


def test_loader_lists_eval_runs_with_domain_subset_and_provider_filters(
    make_settings,
    tmp_path,
) -> None:
    settings = make_settings(project_root=tmp_path)
    build_eval_run_artifact(settings)
    build_eval_run_artifact(
        settings,
        run_id=SECOND_RUN_ID,
        created_at="2026-03-17T09:00:00+00:00",
        domain="medicaid",
        split="validation",
        eval_subset="frozen_experiment",
        subset_label="Frozen Experiment",
        provider_type="openrouter",
    )

    loader = WorkbenchArtifactLoader(settings)
    items = loader.list_eval_runs(
        domain="dmv",
        subset="Smoke 25",
        provider="ollama",
    ).items

    assert [item.run_id for item in items] == [RUN_ID]
    assert items[0].headline_retrieval.doc_recall_at_3 == 1.0
    assert items[0].failure_counts == {"wrong_doc": 1}


def test_loader_builds_example_detail_from_prediction_retrieval_and_trace_index(
    make_settings,
    tmp_path,
) -> None:
    settings = make_settings(project_root=tmp_path)
    build_eval_run_artifact(settings)
    loader = WorkbenchArtifactLoader(settings)

    detail = loader.load_eval_example(RUN_ID, EXAMPLE_ID)

    assert detail.prediction.example_id == EXAMPLE_ID
    assert detail.retrieval_example is not None
    assert detail.retrieval_example.final_query == "title form dmv"
    assert detail.trace_index_entry.trace_file == "dmv-one-turn-2.jsonl"
    assert detail.trace_summary.final_query == "title form dmv"
    assert detail.trace_summary.decision == "answer"
    assert detail.artifact_paths["trace"].endswith("/dmv-one-turn-2.jsonl")

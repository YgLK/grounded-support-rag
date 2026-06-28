from __future__ import annotations

from fastapi.testclient import TestClient

from support_graph.ui.app import create_app
from support_graph.ui.loaders import WorkbenchArtifactLoader

from tests.ui._helpers import (
    EXAMPLE_ID,
    REPORT_ID,
    RUN_ID,
    STANDALONE_RUN_ID,
    build_eval_report_artifact,
    build_eval_run_artifact,
    build_standalone_run_artifact,
    write_json,
)


def test_api_routes_expose_artifact_explorer_and_eval_views(
    make_settings,
    tmp_path,
) -> None:
    settings = make_settings(project_root=tmp_path)
    build_eval_run_artifact(settings)
    build_eval_report_artifact(settings)
    build_standalone_run_artifact(settings)

    client = TestClient(create_app(WorkbenchArtifactLoader(settings)))

    explorer = client.get("/api/artifacts")
    assert explorer.status_code == 200
    payload = explorer.json()
    assert payload["eval_runs"][0]["run_id"] == RUN_ID
    assert payload["reports"][0]["report_id"] == REPORT_ID
    assert payload["standalone_runs"][0]["run_id"] == STANDALONE_RUN_ID

    filtered = client.get(
        "/api/evals",
        params={
            "domain": "dmv",
            "subset": "Smoke 25",
            "provider": "ollama",
            "chat_model": "ollama-chat",
        },
    )
    assert filtered.status_code == 200
    assert [item["run_id"] for item in filtered.json()["items"]] == [RUN_ID]

    eval_detail = client.get(f"/api/evals/{RUN_ID}")
    assert eval_detail.status_code == 200
    eval_payload = eval_detail.json()
    assert eval_payload["summary"]["subset_label"] == "Smoke 25"
    assert "<h1>Eval Summary</h1>" in eval_payload["summary_html"]
    assert eval_payload["failures"][0]["failure_label"] == "wrong_doc"

    failure_table = client.get(
        f"/api/evals/{RUN_ID}/failures",
        params={"label": "wrong_doc"},
    )
    assert failure_table.status_code == 200
    failure_payload = failure_table.json()
    assert failure_payload["total_failures"] == 1
    assert [item["example_id"] for item in failure_payload["items"]] == [EXAMPLE_ID]

    example_detail = client.get(f"/api/evals/{RUN_ID}/examples/{EXAMPLE_ID}")
    assert example_detail.status_code == 200
    example_payload = example_detail.json()
    assert example_payload["prediction"]["example_id"] == EXAMPLE_ID
    assert example_payload["trace_index_entry"]["trace_file"] == "dmv-one-turn-2.jsonl"
    assert example_payload["trace_summary"]["final_query"] == "title form dmv"


def test_api_routes_expose_standalone_run_and_report_details(
    make_settings,
    tmp_path,
) -> None:
    settings = make_settings(project_root=tmp_path)
    build_standalone_run_artifact(settings)
    build_eval_report_artifact(settings)

    client = TestClient(create_app(WorkbenchArtifactLoader(settings)))

    run_response = client.get(f"/api/runs/{STANDALONE_RUN_ID}")
    assert run_response.status_code == 200
    run_payload = run_response.json()
    assert run_payload["summary"]["decision"] == "answer"
    assert run_payload["trace_summary"]["final_query"] == "title form dmv"
    assert (
        run_payload["artifact_paths"]["trace"]
        == f"outputs/runs/{STANDALONE_RUN_ID}/trace.jsonl"
    )

    report_response = client.get(f"/api/reports/{REPORT_ID}")
    assert report_response.status_code == 200
    report_payload = report_response.json()
    assert report_payload["summary"]["report_type"] == "experiment_summary"
    assert "<h1>Smoke-10 Experiment Summary</h1>" in report_payload["report_html"]


def test_api_returns_404_for_missing_and_409_for_incomplete_artifacts(
    make_settings,
    tmp_path,
) -> None:
    settings = make_settings(project_root=tmp_path)
    client = TestClient(create_app(WorkbenchArtifactLoader(settings)))

    missing = client.get("/api/evals/missing-run")
    assert missing.status_code == 404
    assert "missing-run" in missing.json()["detail"]

    incomplete_dir = settings.paths.eval_runs_dir / "broken-run"
    incomplete_dir.mkdir(parents=True)
    write_json(
        incomplete_dir / "manifest.json",
        {
            "run_id": "broken-run",
            "created_at": "2026-03-18T14:30:00+00:00",
            "dataset_root": "multidoc2dial",
            "domains": ["dmv"],
            "split": "validation",
            "eval_subset": "smoke",
            "subset_label": "Smoke 25",
            "target_modes": ["answer"],
            "provider": {
                "type": "ollama",
                "chat_base_url": None,
                "embedding_type": "ollama",
                "embedding_base_url": None,
                "chat_model": "qwen3",
                "embedding_model": "qwen3-embed",
            },
            "chunking": {"strategy": "section_aware", "max_tokens_per_chunk": 512},
            "retrieval": {
                "top_k": 5,
                "candidate_k": 12,
                "max_attempts": 2,
                "use_history": True,
                "content_only_reasoning": True,
                "neighbor_expansion": True,
            },
            "graph": {"enable_retry": True, "decision_policy_version": "v1"},
            "prompt_version": "v1",
            "notes": "broken",
        },
    )

    incomplete = client.get("/api/evals/broken-run")
    assert incomplete.status_code == 409
    assert "incomplete" in incomplete.json()["detail"]

    listed = client.get("/api/evals")
    assert listed.status_code == 200
    assert listed.json()["items"] == []

    incomplete_standalone_dir = settings.paths.runs_dir / "run-broken"
    incomplete_standalone_dir.mkdir(parents=True)
    write_json(
        incomplete_standalone_dir / "manifest.json",
        {
            "run_id": "run-broken",
            "created_at": "2026-03-18T14:30:00+00:00",
            "example_id": EXAMPLE_ID,
            "domain": "dmv",
            "provider": {
                "type": "ollama",
                "chat_model": "qwen3",
                "embedding_type": "ollama",
                "embedding_model": "qwen3-embed",
            },
            "prompt_version": "v1",
        },
    )
    write_json(
        incomplete_standalone_dir / "result.json",
        {
            "example_id": EXAMPLE_ID,
            "latest_user_utterance": "What title form do I need?",
            "decision": "answer",
            "response_text": "Bring your title form.",
            "citations": [],
            "confidence_label": "high",
            "retrieval_ranked_chunks": [],
            "retrieved_chunks": [],
            "evidence_grade": {
                "verdict": "sufficient",
                "reason": "direct support",
                "missing_information": [],
            },
            "trace_summary": {
                "retrieval_attempts": 1,
                "final_query": "title form dmv",
                "graph_path": ["prepare_query", "retrieve_docs", "finalize"],
                "trace_path": "outputs/runs/run-broken/trace.jsonl",
            },
        },
    )

    standalone_list = client.get("/api/runs")
    assert standalone_list.status_code == 200
    assert standalone_list.json()["items"] == []

    incomplete_run = client.get("/api/runs/run-broken")
    assert incomplete_run.status_code == 409
    assert "run-broken" in incomplete_run.json()["detail"]


def test_html_routes_render_workbench_pages_and_distinct_evidence_panels(
    make_settings,
    tmp_path,
) -> None:
    settings = make_settings(project_root=tmp_path)
    build_eval_run_artifact(settings)
    build_eval_report_artifact(settings)
    build_standalone_run_artifact(settings)

    client = TestClient(create_app(WorkbenchArtifactLoader(settings)))

    home = client.get("/")
    assert home.status_code == 200
    assert "SupportGraph Workbench" in home.text
    assert "htmx.org@2.0.8" in home.text
    assert "alpinejs@3.15.0" in home.text
    assert "@tabler/core@1.4.0" in home.text
    assert RUN_ID in home.text
    assert "Smoke-10 Experiment Summary" in home.text
    assert STANDALONE_RUN_ID in home.text
    assert "Chat model" in home.text
    assert "ollama-chat" in home.text
    assert "qwen3:8b-q4_K_M" in home.text

    eval_detail = client.get(f"/evals/{RUN_ID}")
    assert eval_detail.status_code == 200
    assert "Failure Buckets" in eval_detail.text
    assert f"/evals/{RUN_ID}/failures?label=wrong_doc" in eval_detail.text

    example_detail = client.get(f"/evals/{RUN_ID}/examples/{EXAMPLE_ID}")
    assert example_detail.status_code == 200
    assert "retrieval_ranked_chunks" in example_detail.text
    assert "retrieved_chunks" in example_detail.text
    assert "Trace Inspector" in example_detail.text

    run_detail = client.get(f"/runs/{STANDALONE_RUN_ID}")
    assert run_detail.status_code == 200
    assert "Standalone Run" in run_detail.text
    assert "Trace Summary" in run_detail.text

    report_detail = client.get(f"/reports/{REPORT_ID}")
    assert report_detail.status_code == 200
    assert "Rendered Report" in report_detail.text
    assert "Smoke-10 Experiment Summary" in report_detail.text


def test_htmx_routes_return_partial_fragments(
    make_settings,
    tmp_path,
) -> None:
    settings = make_settings(project_root=tmp_path)
    build_eval_run_artifact(settings)
    client = TestClient(create_app(WorkbenchArtifactLoader(settings)))

    home_partial = client.get(
        "/",
        headers={"HX-Request": "true"},
        params={"domain": "dmv"},
    )
    assert home_partial.status_code == 200
    assert "<html" not in home_partial.text
    assert "Eval Runs" in home_partial.text
    assert RUN_ID in home_partial.text

    failure_partial = client.get(
        f"/evals/{RUN_ID}/failures",
        headers={"HX-Request": "true"},
        params={"label": "wrong_doc"},
    )
    assert failure_partial.status_code == 200
    assert "<html" not in failure_partial.text
    assert "Failure Review Table" in failure_partial.text
    assert EXAMPLE_ID in failure_partial.text

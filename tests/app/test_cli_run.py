from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path

import pytest

from support_graph.cli import handlers as cli


def test_run_cli_reports_missing_config_and_next_step(
    monkeypatch,
    capsys,
    make_settings,
) -> None:
    monkeypatch.setattr(
        cli.Settings,
        "load",
        classmethod(
            lambda cls, config_file=None, secrets_file=None: make_settings(
                missing_fields=[
                    ".env: SUPPORT_GRAPH_POSTGRES_DSN",
                    "support_graph.toml: runtime.chat_model",
                ]
            )
        ),
        raising=False,
    )
    monkeypatch.setattr(
        cli,
        "run_graph_async",
        lambda *args, **kwargs: pytest.fail(
            "run_graph_async should not be called when config is missing"
        ),
        raising=False,
    )

    exit_code = cli.main(
        ["run", "--example-id", "kubernetes::1409501a35697e0ce68561e29577b90a::turn_2"]
    )
    output = capsys.readouterr().out.splitlines()

    assert exit_code == 1
    assert output[0] == "SupportGraph Run"
    assert "Missing config" in output
    assert "Next" in output
    assert any("copy .env.example to .env" in line for line in output)


def test_run_cli_default_hierarchy_shows_context_then_decision_then_response(
    monkeypatch,
    capsys,
    make_settings,
    tmp_path: Path,
) -> None:
    fake_settings = make_settings(
        project_root=tmp_path,
        missing_fields=[],
        has_index=True,
    )
    monkeypatch.setattr(
        cli.Settings,
        "load",
        classmethod(lambda cls, config_file=None, secrets_file=None: fake_settings),
        raising=False,
    )
    monkeypatch.setattr(
        cli,
        "collection_row_count",
        lambda *args, **kwargs: 4316,
        raising=False,
    )

    def fake_run_graph(*args, **kwargs):
        return {
            "example_id": "kubernetes::1409501a35697e0ce68561e29577b90a::turn_2",
            "decision": "clarify",
            "response_text": "Could you clarify whether the rollout stalled or the pods are crash-looping?",
            "citations": [
                {
                    "doc_id": "concepts/workloads/controllers/deployment",
                    "chunk_id": (
                        "kubernetes::concepts/workloads/controllers/deployment"
                        "::sec::deployment-status::sub::0"
                    ),
                    "span_ids": ["24", "25", "26"],
                }
            ],
            "confidence_label": "high",
            "trace_summary": {
                "retrieval_attempts": 1,
                "final_query": "deployment rollout stalled",
                "graph_path": [
                    "prepare_query",
                    "retrieve_docs",
                    "grade_evidence",
                    "generate_response",
                    "finalize",
                ],
            },
            "latest_user_utterance": "My Deployment rollout stalled so what should I do",
        }

    monkeypatch.setattr(cli, "run_graph_async", fake_run_graph, raising=False)
    monkeypatch.setattr(
        cli,
        "load_example_record",
        lambda *args, **kwargs: {
            "example_id": "kubernetes::1409501a35697e0ce68561e29577b90a::turn_2"
        },
        raising=False,
    )

    exit_code = cli.main(
        ["run", "--example-id", "kubernetes::1409501a35697e0ce68561e29577b90a::turn_2"]
    )
    output = capsys.readouterr().out.splitlines()

    assert exit_code == 0
    assert output[0] == "SupportGraph Run"
    assert output[1].startswith("Run:")
    assert output[2].startswith("Example:")
    assert "Context" in output
    assert any(line.startswith("User:") for line in output)
    assert "Decision: clarify" in output
    assert "Response" in output
    assert any("rollout stalled" in line for line in output)
    assert "Citations" in output
    assert "Next" in output
    assert any(
        "Ask one concrete missing-condition question." in line
        or "retry with --verbose" in line
        for line in output
    )
    assert "Trace" in output
    assert any(
        "prepare_query -> retrieve_docs -> grade_evidence -> generate_response -> finalize"
        in line
        for line in output
    )
    assert "Artifacts" not in output
    assert not fake_settings.paths.runs_dir.exists()


def test_run_cli_reports_index_unavailable_when_row_count_check_fails(
    monkeypatch,
    capsys,
    make_settings,
) -> None:
    fake_settings = make_settings(missing_fields=[], has_index=True)
    monkeypatch.setattr(
        cli.Settings,
        "load",
        classmethod(lambda cls, config_file=None, secrets_file=None: fake_settings),
        raising=False,
    )
    monkeypatch.setattr(
        cli,
        "load_example_record",
        lambda *args, **kwargs: {
            "example_id": "kubernetes::1409501a35697e0ce68561e29577b90a::turn_2",
            "domain": "kubernetes",
        },
        raising=False,
    )

    def boom(*args, **kwargs):
        raise RuntimeError("database unavailable")

    monkeypatch.setattr(cli, "collection_row_count", boom, raising=False)
    monkeypatch.setattr(
        cli,
        "run_graph_async",
        lambda *args, **kwargs: pytest.fail("run_graph_async should not be called"),
        raising=False,
    )

    exit_code = cli.main(
        ["run", "--example-id", "kubernetes::1409501a35697e0ce68561e29577b90a::turn_2"]
    )
    output = capsys.readouterr().out.splitlines()

    assert exit_code == 1
    assert output[0] == "SupportGraph Run"
    assert "State: index-unavailable" in output
    assert "Index unavailable" in output


def test_run_cli_prints_langsmith_trace_url_without_persistent_artifacts(
    monkeypatch,
    capsys,
    make_settings,
    tmp_path: Path,
) -> None:
    fake_settings = make_settings(
        project_root=tmp_path,
        missing_fields=[],
        has_index=True,
        overrides={
            "langsmith_tracing_enabled": True,
            "langsmith_api_key": "ls-key",
            "langsmith_project": "support-graph",
        },
    )
    monkeypatch.setattr(
        cli.Settings,
        "load",
        classmethod(lambda cls, config_file=None, secrets_file=None: fake_settings),
        raising=False,
    )
    monkeypatch.setattr(cli, "collection_row_count", lambda *args, **kwargs: 1)
    monkeypatch.setattr(
        cli,
        "load_example_record",
        lambda *args, **kwargs: {
            "example_id": "k8s-001",
            "domain": "kubernetes",
            "latest_user_utterance": "What is a pod?",
        },
    )
    monkeypatch.setattr(cli, "build_langsmith_client", lambda config: object())
    monkeypatch.setattr(cli, "tracing_context", lambda **kwargs: nullcontext())

    class Root:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def get_url(self):
            return "https://smith.langchain.com/o/project/run-1"

    monkeypatch.setattr(cli, "trace", lambda *args, **kwargs: Root())
    monkeypatch.setattr(
        cli,
        "run_graph_async",
        lambda **kwargs: {
            "example_id": "k8s-001",
            "decision": "answer",
            "response_text": "A pod is a deployable unit.",
            "citations": [],
            "latest_user_utterance": "What is a pod?",
            "trace_summary": {"graph_path": []},
        },
    )

    exit_code = cli.main(["run", "--example-id", "k8s-001"])
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "https://smith.langchain.com/o/project/run-1" in output
    assert not fake_settings.paths.runs_dir.exists()
    assert not (tmp_path / "traces").exists()


def test_run_cli_emits_pipeline_progress_logs(
    monkeypatch,
    capsys,
    make_settings,
) -> None:
    fake_settings = make_settings(missing_fields=[], has_index=True)
    logged: list[str] = []

    monkeypatch.setattr(
        cli.Settings,
        "load",
        classmethod(lambda cls, config_file=None, secrets_file=None: fake_settings),
        raising=False,
    )
    monkeypatch.setattr(
        cli,
        "load_example_record",
        lambda *args, **kwargs: {
            "example_id": "kubernetes::1409501a35697e0ce68561e29577b90a::turn_2",
            "domain": "kubernetes",
            "latest_user_utterance": "My Deployment rollout stalled so what should I do",
        },
        raising=False,
    )
    monkeypatch.setattr(
        cli, "collection_row_count", lambda *args, **kwargs: 4316, raising=False
    )

    def fake_info(message, *args, **kwargs):
        logged.append(message % args if args else str(message))

    monkeypatch.setattr(cli.logger, "info", fake_info)

    def fake_run_graph(*args, **kwargs):
        sink = kwargs.get("_event_sink")
        assert sink is not None
        sink(
            {
                "kind": "query_ready",
                "run_id": "run-test",
                "example_id": "kubernetes::1409501a35697e0ce68561e29577b90a::turn_2",
                "query": "deployment rollout stalled",
            }
        )
        sink(
            {
                "kind": "retrieval_complete",
                "run_id": "run-test",
                "example_id": "kubernetes::1409501a35697e0ce68561e29577b90a::turn_2",
                "retrieval_attempts": 1,
                "retrieval_ranked_chunks": [{"chunk_id": "chunk-1"}],
                "retrieved_chunks": [{"chunk_id": "chunk-1"}],
            }
        )
        sink(
            {
                "kind": "evidence_graded",
                "run_id": "run-test",
                "example_id": "kubernetes::1409501a35697e0ce68561e29577b90a::turn_2",
                "evidence_grade": {"verdict": "sufficient"},
            }
        )
        sink(
            {
                "kind": "response_completed",
                "run_id": "run-test",
                "example_id": "kubernetes::1409501a35697e0ce68561e29577b90a::turn_2",
                "decision": "answer",
                "citations": [{"chunk_id": "chunk-1"}],
            }
        )
        return {
            "example_id": "kubernetes::1409501a35697e0ce68561e29577b90a::turn_2",
            "decision": "answer",
            "response_text": "Inspect rollout status and Deployment conditions.",
            "citations": [
                {
                    "doc_id": "doc",
                    "chunk_id": "chunk-1",
                    "span_ids": ["2", "3"],
                }
            ],
            "confidence_label": "high",
            "trace_summary": {
                "retrieval_attempts": 1,
                "final_query": "deployment rollout stalled",
                "graph_path": [
                    "prepare_query",
                    "retrieve_docs",
                    "grade_evidence",
                    "generate_response",
                    "finalize",
                ],
            },
            "latest_user_utterance": "My Deployment rollout stalled so what should I do",
        }

    monkeypatch.setattr(cli, "run_graph_async", fake_run_graph, raising=False)

    exit_code = cli.main(
        ["run", "--example-id", "kubernetes::1409501a35697e0ce68561e29577b90a::turn_2"]
    )
    _ = capsys.readouterr()

    assert exit_code == 0
    assert any(
        "Loading example kubernetes::1409501a35697e0ce68561e29577b90a::turn_2" in line
        for line in logged
    )
    assert any(
        "Running graph for example=kubernetes::1409501a35697e0ce68561e29577b90a::turn_2"
        in line
        for line in logged
    )
    assert any("prepared query" in line for line in logged)
    assert any("retrieval attempt 1" in line for line in logged)
    assert any("evidence verdict" in line for line in logged)
    assert any(
        "completed for kubernetes::1409501a35697e0ce68561e29577b90a::turn_2 with decision=answer"
        in line
        for line in logged
    )


def test_main_writes_command_logs_to_file(
    monkeypatch,
    tmp_path: Path,
    make_settings,
) -> None:
    settings = make_settings(
        project_root=tmp_path,
        missing_fields=[".env: SUPPORT_GRAPH_POSTGRES_DSN"],
    )

    monkeypatch.setattr(
        cli.Settings,
        "load",
        classmethod(lambda cls, config_file=None, secrets_file=None: settings),
        raising=False,
    )

    exit_code = cli.main(
        ["run", "--example-id", "kubernetes::1409501a35697e0ce68561e29577b90a::turn_2"]
    )

    assert exit_code == 1
    log_files = sorted(settings.paths.log_dir.glob("*.log"))
    assert len(log_files) == 1
    contents = log_files[0].read_text(encoding="utf-8")
    assert "Writing command logs to" in contents

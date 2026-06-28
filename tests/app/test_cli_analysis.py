from __future__ import annotations

import json
from pathlib import Path

from support_graph.cli import handlers as cli
from support_graph.runtime.traces import write_trace_event


def test_review_failures_cli_prints_counts_filtered_examples_and_artifacts(
    monkeypatch,
    capsys,
    tmp_path: Path,
    make_settings,
) -> None:
    settings = make_settings(project_root=tmp_path)
    run_dir = settings.paths.eval_runs_dir / "run-123"
    run_dir.mkdir(parents=True)
    (run_dir / "failures.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "example_id": "ex-1",
                        "failure_label": "wrong_doc",
                        "decision": "answer",
                        "latest_user_utterance": "I need title transfer requirements for my parent gift vehicle.",
                        "target_mode": "answer",
                    }
                ),
                json.dumps(
                    {
                        "example_id": "ex-2",
                        "failure_label": "weak_citations",
                        "decision": "answer",
                        "latest_user_utterance": "What is the exact form I need for renewal?",
                        "target_mode": "answer",
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (run_dir / "manual_review.csv").write_text("example_id\nex-1\n", encoding="utf-8")
    (run_dir / "retrieval_examples.jsonl").write_text("{}\n", encoding="utf-8")
    for name in (
        "manifest.json",
        "metrics.json",
        "predictions.jsonl",
        "trace_index.json",
        "summary.md",
    ):
        (run_dir / name).write_text(
            "{}\n" if name.endswith(".json") or name.endswith(".jsonl") else "\n",
            encoding="utf-8",
        )

    monkeypatch.setattr(
        cli.Settings,
        "load",
        classmethod(lambda cls, config_file=None, secrets_file=None: settings),
        raising=False,
    )

    exit_code = cli.main(
        [
            "review-failures",
            "--run-id",
            "run-123",
            "--label",
            "wrong_doc",
            "--limit",
            "5",
        ]
    )
    output = capsys.readouterr().out.splitlines()

    assert exit_code == 0
    assert output[0] == "SupportGraph Review Failures"
    assert "Failure Counts" in output
    assert any("wrong_doc: 1" in line for line in output)
    assert "Examples" in output
    assert any(line.startswith("ex-1 | wrong_doc | answer |") for line in output)
    assert all("ex-2" not in line for line in output)
    assert "Artifacts" in output
    assert any(line.endswith("failures.jsonl") for line in output)
    assert any(line.endswith("manual_review.csv") for line in output)
    assert any(line.endswith("retrieval_examples.jsonl") for line in output)


def test_trace_show_cli_prints_trace_summary_for_one_example(
    monkeypatch,
    capsys,
    tmp_path: Path,
    make_settings,
) -> None:
    settings = make_settings(project_root=tmp_path)
    run_dir = settings.paths.eval_runs_dir / "run-123"
    run_dir.mkdir(parents=True)
    trace_path = run_dir / "traces" / "ex-1.jsonl"
    write_trace_event(
        trace_path,
        {"node": "prepare_query", "query": "Domain: kubernetes", "latency_ms": 1.0},
    )
    write_trace_event(
        trace_path,
        {
            "node": "retrieve_docs",
            "retrieval_attempts": 1,
            "retrieval_ranked_count": 5,
            "retrieved_count": 4,
            "latency_ms": 2.0,
        },
    )
    write_trace_event(
        trace_path,
        {
            "node": "grade_evidence",
            "evidence_grade": {"verdict": "partial"},
            "latency_ms": 3.0,
        },
    )
    write_trace_event(
        trace_path,
        {"node": "resolve_without_answer", "decision": "clarify", "latency_ms": 4.0},
    )
    write_trace_event(
        trace_path,
        {
            "node": "finalize",
            "decision": "clarify",
            "total_latency_ms": 20.0,
            "latency_ms": 0.5,
        },
    )
    for name in (
        "manifest.json",
        "metrics.json",
        "predictions.jsonl",
        "failures.jsonl",
        "manual_review.csv",
        "retrieval_examples.jsonl",
        "summary.md",
    ):
        (run_dir / name).write_text(
            "{}\n" if name.endswith(".json") or name.endswith(".jsonl") else "\n",
            encoding="utf-8",
        )
    (run_dir / "trace_index.json").write_text(
        json.dumps(
            {
                "entries": [
                    {
                        "example_id": "ex-1",
                        "trace_file": "ex-1.jsonl",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        cli.Settings,
        "load",
        classmethod(lambda cls, config_file=None, secrets_file=None: settings),
        raising=False,
    )

    exit_code = cli.main(["trace-show", "--run-id", "run-123", "--example-id", "ex-1"])
    output = capsys.readouterr().out.splitlines()

    assert exit_code == 0
    assert output[0] == "SupportGraph Trace Show"
    assert any(line.startswith("Final Query: Domain: kubernetes") for line in output)
    assert any(line == "Attempts: 1" for line in output)
    assert any(
        "prepare_query -> retrieve_docs -> grade_evidence -> resolve_without_answer -> finalize"
        in line
        for line in output
    )
    assert "Node Latency" in output
    assert any(line.startswith("retrieve_docs: 2.00 ms") for line in output)
    assert "Evidence" in output
    assert any(line == "clarify" for line in output)


def test_review_failures_cli_fails_clearly_when_run_or_artifacts_are_missing(
    monkeypatch,
    capsys,
    tmp_path: Path,
    make_settings,
) -> None:
    settings = make_settings(project_root=tmp_path)
    monkeypatch.setattr(
        cli.Settings,
        "load",
        classmethod(lambda cls, config_file=None, secrets_file=None: settings),
        raising=False,
    )

    exit_code = cli.main(["review-failures", "--run-id", "missing-run"])
    output = capsys.readouterr().out.splitlines()

    assert exit_code == 1
    assert output[0] == "SupportGraph Review Failures"
    assert "Run missing" in output

    run_dir = settings.paths.eval_runs_dir / "run-123"
    run_dir.mkdir(parents=True)
    (run_dir / "failures.jsonl").write_text("", encoding="utf-8")

    exit_code = cli.main(["review-failures", "--run-id", "run-123"])
    output = capsys.readouterr().out.splitlines()

    assert exit_code == 1
    assert "Artifacts missing" in output
    assert any(
        "is incomplete under outputs/evals/runs/run-123" in line for line in output
    )


def test_trace_show_cli_fails_clearly_when_artifacts_or_example_are_missing(
    monkeypatch,
    capsys,
    tmp_path: Path,
    make_settings,
) -> None:
    settings = make_settings(project_root=tmp_path)
    run_dir = settings.paths.eval_runs_dir / "run-123"
    run_dir.mkdir(parents=True)
    monkeypatch.setattr(
        cli.Settings,
        "load",
        classmethod(lambda cls, config_file=None, secrets_file=None: settings),
        raising=False,
    )

    exit_code = cli.main(["trace-show", "--run-id", "run-123", "--example-id", "ex-1"])
    output = capsys.readouterr().out.splitlines()

    assert exit_code == 1
    assert output[0] == "SupportGraph Trace Show"
    assert "Artifacts missing" in output

    for name in (
        "manifest.json",
        "metrics.json",
        "predictions.jsonl",
        "failures.jsonl",
        "manual_review.csv",
        "retrieval_examples.jsonl",
        "summary.md",
    ):
        (run_dir / name).write_text(
            "{}\n" if name.endswith(".json") or name.endswith(".jsonl") else "\n",
            encoding="utf-8",
        )
    (run_dir / "trace_index.json").write_text(
        json.dumps({"entries": []}), encoding="utf-8"
    )
    exit_code = cli.main(["trace-show", "--run-id", "run-123", "--example-id", "ex-1"])
    output = capsys.readouterr().out.splitlines()

    assert exit_code == 1
    assert "Example missing" in output

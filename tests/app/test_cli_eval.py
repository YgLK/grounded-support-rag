from __future__ import annotations

from pathlib import Path

import pytest

from support_graph.cli import handlers as cli


def test_eval_cli_reports_missing_config_and_next_step(
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
        "evaluate_split_async",
        lambda *args, **kwargs: pytest.fail("evaluate_split_async should not run"),
        raising=False,
    )

    exit_code = cli.main(["eval", "--split", "validation", "--domain", "dmv"])
    output = capsys.readouterr().out.splitlines()

    assert exit_code == 1
    assert output[0] == "SupportGraph Eval"
    assert "Missing config" in output
    assert "Next" in output
    assert any("copy .env.example to .env" in line for line in output)


def test_eval_cli_reports_missing_index(monkeypatch, capsys, make_settings) -> None:
    monkeypatch.setattr(
        cli.Settings,
        "load",
        classmethod(lambda cls, config_file=None, secrets_file=None: make_settings()),
        raising=False,
    )
    monkeypatch.setattr(
        cli, "collection_row_count", lambda *args, **kwargs: 0, raising=False
    )
    monkeypatch.setattr(
        cli,
        "evaluate_split_async",
        lambda *args, **kwargs: pytest.fail("evaluate_split_async should not run"),
        raising=False,
    )

    exit_code = cli.main(["eval", "--split", "validation", "--domain", "dmv"])
    output = capsys.readouterr().out.splitlines()

    assert exit_code == 1
    assert output[0] == "SupportGraph Eval"
    assert "Index missing" in output
    assert "Next" in output
    assert any("index-docs" in line for line in output)


def test_eval_cli_reports_index_unavailable_when_row_count_check_fails(
    monkeypatch,
    capsys,
    make_settings,
) -> None:
    monkeypatch.setattr(
        cli.Settings,
        "load",
        classmethod(lambda cls, config_file=None, secrets_file=None: make_settings()),
        raising=False,
    )

    def boom(*args, **kwargs):
        raise RuntimeError("database unavailable")

    monkeypatch.setattr(cli, "collection_row_count", boom, raising=False)
    monkeypatch.setattr(
        cli,
        "evaluate_split_async",
        lambda *args, **kwargs: pytest.fail("evaluate_split_async should not run"),
        raising=False,
    )

    exit_code = cli.main(["eval", "--split", "validation", "--domain", "dmv"])
    output = capsys.readouterr().out.splitlines()

    assert exit_code == 1
    assert output[0] == "SupportGraph Eval"
    assert "State: index-unavailable" in output
    assert "Index unavailable" in output


def test_eval_cli_default_hierarchy_shows_metrics_failures_and_artifacts(
    monkeypatch,
    capsys,
    make_settings,
) -> None:
    fake_settings = make_settings()
    monkeypatch.setattr(
        cli.Settings,
        "load",
        classmethod(lambda cls, config_file=None, secrets_file=None: fake_settings),
        raising=False,
    )
    monkeypatch.setattr(
        cli, "collection_row_count", lambda *args, **kwargs: 4316, raising=False
    )
    monkeypatch.setattr(
        cli,
        "evaluate_split_async",
        lambda **kwargs: {
            "run_id": "20260318-143000-dmv-smoke",
            "subset_label": "dmv validation / smoke",
            "output_dir": Path(
                "/Users/yglk/coding/grounded-support-rag/outputs/evals/runs/20260318-143000-dmv-smoke"
            ),
            "metrics": {
                "retrieval": {
                    "answer": {
                        "doc_recall_at_1": 0.2,
                        "doc_recall_at_3": 0.4,
                        "doc_recall_at_5": 0.5,
                        "doc_recall_at_10": 0.6,
                        "span_recall_at_5": 0.3,
                    }
                },
                "generation": {
                    "answer": {
                        "rouge_l": 0.2,
                        "token_f1": 0.25,
                        "exact_match": 0.1,
                        "sacrebleu": 0.33,
                    }
                },
            },
            "failure_counts": {"wrong_doc": 3, "bad_clarification": 1},
        },
        raising=False,
    )

    exit_code = cli.main(["eval", "--split", "validation", "--domain", "dmv"])
    output = capsys.readouterr().out.splitlines()

    assert exit_code == 0
    assert output[0] == "SupportGraph Eval"
    assert output[1].startswith("Run:")
    assert output[2].startswith("Subset:")
    assert "Headline Metrics" in output
    assert "Doc Recall@3: 0.400" in output
    assert "Span Recall@5: 0.300" in output
    assert "ROUGE-L: 0.200" in output
    assert "F1: 0.250" in output
    assert "Paper Reference" in output
    assert "Recall@1: 0.200" in output
    assert "Recall@5: 0.500" in output
    assert "Recall@10: 0.600" in output
    assert "Exact Match: 0.100" in output
    assert "SacreBLEU: 0.330" in output
    assert "Failure Snapshot" in output
    assert any("wrong_doc: 3" in line for line in output)
    assert "Artifacts" in output
    assert any(line.endswith("summary.md") for line in output)
    assert any(line.endswith("failures.jsonl") for line in output)


def test_eval_cli_marks_unavailable_rank_metrics_as_na(
    monkeypatch,
    capsys,
    make_settings,
) -> None:
    fake_settings = make_settings()
    monkeypatch.setattr(
        cli.Settings,
        "load",
        classmethod(lambda cls, config_file=None, secrets_file=None: fake_settings),
        raising=False,
    )
    monkeypatch.setattr(
        cli, "collection_row_count", lambda *args, **kwargs: 4316, raising=False
    )
    monkeypatch.setattr(
        cli,
        "evaluate_split_async",
        lambda **kwargs: {
            "run_id": "20260318-143000-dmv-smoke",
            "subset_label": "dmv validation / smoke",
            "output_dir": Path(
                "/Users/yglk/coding/grounded-support-rag/outputs/evals/runs/20260318-143000-dmv-smoke"
            ),
            "retrieval_top_k": 5,
            "metrics": {
                "retrieval": {
                    "answer": {
                        "doc_recall_at_1": 0.2,
                        "doc_recall_at_3": 0.4,
                        "doc_recall_at_5": 0.5,
                        "doc_recall_at_10": None,
                        "span_recall_at_5": 0.3,
                    }
                },
                "generation": {
                    "answer": {
                        "rouge_l": 0.2,
                        "token_f1": 0.25,
                        "exact_match": 0.1,
                        "sacrebleu": 0.33,
                    }
                },
            },
            "failure_counts": {},
        },
        raising=False,
    )

    exit_code = cli.main(["eval", "--split", "validation", "--domain", "dmv"])
    output = capsys.readouterr().out.splitlines()

    assert exit_code == 0
    assert "Recall@10: n/a (retrieval_top_k=5)" in output

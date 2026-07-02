from __future__ import annotations

import argparse
from pathlib import Path
from typing import cast

import pytest

from support_graph.cli import handlers as cli
from support_graph.cli.handlers import build_parser


def _stub_settings(monkeypatch, make_settings, **kwargs) -> None:
    fake_settings = make_settings(**kwargs)
    monkeypatch.setattr(
        cli.Settings,
        "load",
        classmethod(lambda cls, config_file=None, secrets_file=None: fake_settings),
        raising=False,
    )


def test_doctor_cli_reports_missing_config_and_next_step(
    monkeypatch,
    capsys,
    make_settings,
) -> None:
    _stub_settings(
        monkeypatch,
        make_settings,
        missing_fields=[
            ".env: SUPPORT_GRAPH_POSTGRES_DSN",
            "support_graph.toml: runtime.chat_model",
        ],
    )
    monkeypatch.setattr(
        cli,
        "collection_row_count",
        lambda *args, **kwargs: pytest.fail(
            "collection_row_count should not be called when config is missing"
        ),
        raising=False,
    )

    exit_code = cli.main(["doctor", "--domain", "kubernetes"])
    output = capsys.readouterr().out.splitlines()

    assert exit_code == 1
    assert output[0] == "SupportGraph Doctor"
    assert "Domain: kubernetes" in output
    assert "Config: missing" in output
    assert "Index: skipped" in output
    assert "State: needs-action" in output
    assert "Next" in output
    assert any("copy .env.example to .env" in line for line in output)


def test_doctor_cli_reports_index_unavailable_on_row_count_exception(
    monkeypatch,
    capsys,
    make_settings,
) -> None:
    _stub_settings(monkeypatch, make_settings, missing_fields=[])

    def boom(*args, **kwargs):
        raise RuntimeError("database unavailable")

    monkeypatch.setattr(cli, "collection_row_count", boom, raising=False)

    exit_code = cli.main(["doctor", "--domain", "kubernetes"])
    output = capsys.readouterr().out.splitlines()

    assert exit_code == 1
    assert output[0] == "SupportGraph Doctor"
    assert "Config: ok" in output
    assert "Index: unavailable" in output
    assert "State: needs-action" in output
    assert any("Verify Postgres is reachable" in line for line in output)


def test_doctor_cli_reports_index_missing_when_collection_empty(
    monkeypatch,
    capsys,
    make_settings,
) -> None:
    _stub_settings(monkeypatch, make_settings, missing_fields=[])
    monkeypatch.setattr(cli, "collection_row_count", lambda *a, **k: 0, raising=False)

    exit_code = cli.main(["doctor", "--domain", "kubernetes"])
    output = capsys.readouterr().out.splitlines()

    assert exit_code == 1
    assert "Index: missing" in output
    assert "State: needs-action" in output
    assert any(
        "uv run grounded-support-rag index-docs --domain kubernetes" in line
        for line in output
    )


def test_doctor_cli_reports_ready_when_collection_populated(
    monkeypatch,
    capsys,
    make_settings,
) -> None:
    _stub_settings(monkeypatch, make_settings, missing_fields=[])
    monkeypatch.setattr(
        cli, "collection_row_count", lambda *a, **k: 4316, raising=False
    )

    exit_code = cli.main(["doctor", "--domain", "kubernetes"])
    output = capsys.readouterr().out.splitlines()

    assert exit_code == 0
    assert "Config: ok" in output
    assert "Index: ok" in output
    assert "Eval Artifacts: not-checked" in output
    assert "State: ready" in output
    assert any(
        "uv run grounded-support-rag eval --domain kubernetes --subset smoke" in line
        for line in output
    )


def test_doctor_cli_reports_missing_eval_artifacts_for_run_id(
    monkeypatch,
    capsys,
    make_settings,
) -> None:
    _stub_settings(monkeypatch, make_settings, missing_fields=[])
    monkeypatch.setattr(
        cli, "collection_row_count", lambda *a, **k: 4316, raising=False
    )

    exit_code = cli.main(
        ["doctor", "--domain", "kubernetes", "--run-id", "missing-run-123"]
    )
    output = capsys.readouterr().out.splitlines()

    assert exit_code == 1
    assert "Eval Artifacts: missing" in output
    assert "State: needs-action" in output
    assert any("eval --domain kubernetes --subset smoke" in line for line in output)


def test_doctor_cli_reports_ok_eval_artifacts_for_complete_run(
    monkeypatch,
    capsys,
    make_settings,
    tmp_path: Path,
) -> None:
    settings = make_settings(
        project_root=tmp_path,
        missing_fields=[],
    )
    monkeypatch.setattr(
        cli.Settings,
        "load",
        classmethod(lambda cls, config_file=None, secrets_file=None: settings),
        raising=False,
    )
    monkeypatch.setattr(
        cli, "collection_row_count", lambda *a, **k: 4316, raising=False
    )

    from support_graph.artifacts import EVAL_RUN_REQUIRED_FILES

    run_id = "complete-run-456"
    run_dir = settings.paths.eval_runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    for name in EVAL_RUN_REQUIRED_FILES:
        (run_dir / name).write_text("{}", encoding="utf-8")

    exit_code = cli.main(["doctor", "--domain", "kubernetes", "--run-id", run_id])
    output = capsys.readouterr().out.splitlines()

    assert exit_code == 0
    assert "Eval Artifacts: ok" in output
    assert "State: ready" in output


def test_doctor_cli_preserves_config_file_and_secrets_in_next_commands(
    monkeypatch,
    capsys,
    make_settings,
    tmp_path: Path,
) -> None:
    """Suggested remediation commands must reuse the --config-file/--secrets-file
    the user passed to doctor, so they target the same setup doctor inspected
    instead of falling back to the default support_graph.toml.
    """
    settings = make_settings(
        project_root=tmp_path,
        missing_fields=[],
    )
    monkeypatch.setattr(
        cli.Settings,
        "load",
        classmethod(lambda cls, config_file=None, secrets_file=None: settings),
        raising=False,
    )
    # Empty collection -> index-missing remediation command is suggested.
    monkeypatch.setattr(cli, "collection_row_count", lambda *a, **k: 0, raising=False)

    exit_code = cli.main(
        [
            "--config-file",
            "support_graph.kubernetes.toml",
            "--secrets-file",
            ".env.kubernetes",
            "doctor",
            "--domain",
            "kubernetes",
        ]
    )
    output = capsys.readouterr().out.splitlines()

    assert exit_code == 1
    assert "Index: missing" in output
    assert any(
        "uv run grounded-support-rag --config-file support_graph.kubernetes.toml "
        "--secrets-file .env.kubernetes index-docs --domain kubernetes" in line
        for line in output
    )


def test_doctor_parser_exposes_help() -> None:
    parser = build_parser()
    assert parser._subparsers is not None
    subparsers_action = cast(
        argparse._SubParsersAction, parser._subparsers._group_actions[0]
    )
    subcommands = subparsers_action.choices
    assert "doctor" in subcommands

    doctor_parser = subcommands["doctor"]
    help_text = doctor_parser.format_help()
    assert "--domain" in help_text
    assert "--run-id" in help_text

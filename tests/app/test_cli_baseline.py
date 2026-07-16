from __future__ import annotations

from pathlib import Path

from support_graph.cli.parser import CliHandlers, build_parser
from support_graph.cli import handlers as cli


def test_baseline_promote_parser_requires_id_and_reason() -> None:
    args = build_parser(handlers()).parse_args(
        ["baseline-promote", "--experiment-id", "exp-1", "--reason", "reviewed"]
    )

    assert args.experiment_id == "exp-1"
    assert args.reason == "reviewed"


def test_baseline_export_parser_requires_experiment_id() -> None:
    args = build_parser(handlers()).parse_args(
        ["baseline-export", "--experiment-id", "exp-1"]
    )

    assert args.experiment_id == "exp-1"


def handlers() -> CliHandlers:
    def command(_: object) -> int:
        return 0

    return CliHandlers(
        fetch_kubernetes_docs=command,
        build_chunks=command,
        build_subsets=command,
        benchmark_embeddings=command,
        index_docs=command,
        run_example=command,
        eval_split=command,
        experiment_smoke10=command,
        review_failures=command,
        trace_show=command,
        serve_ui=command,
        validate_eval_examples=command,
        promote_eval_examples=command,
        eval_variance=command,
        model_ab_compatibility=command,
        baseline_promote=command,
        baseline_export=command,
        doctor=command,
    )


def test_baseline_export_prints_verified_path_and_checksum(
    monkeypatch,
    capsys,
    make_settings,
    tmp_path: Path,
) -> None:
    settings = make_settings(project_root=tmp_path)
    settings.runtime.validate_for_hosted_eval = lambda: None
    export_path = tmp_path / "outputs/baselines/kubernetes-smoke-exp-1.zip"
    export_path.parent.mkdir(parents=True)
    export_path.write_bytes(b"zip evidence")
    monkeypatch.setattr(
        cli.Settings,
        "load",
        classmethod(lambda cls, config_file=None, secrets_file=None: settings),
        raising=False,
    )
    monkeypatch.setattr(cli, "build_langsmith_gateway", lambda config: object())
    monkeypatch.setattr(cli, "export_baseline", lambda *args, **kwargs: export_path)

    exit_code = cli.main(["baseline-export", "--experiment-id", "exp-1"])
    output = capsys.readouterr().out

    assert exit_code == 0
    assert str(export_path) in output
    assert "sha256:" in output
    assert "verified: yes" in output


def test_baseline_promote_refuses_dirty_experiment_without_update(
    monkeypatch,
    capsys,
    make_settings,
    tmp_path: Path,
) -> None:
    settings = make_settings(project_root=tmp_path)
    settings.runtime.validate_for_hosted_eval = lambda: None
    updated = False

    class Gateway:
        async def update_experiment_metadata(self, *args, **kwargs):
            nonlocal updated
            updated = True

    monkeypatch.setattr(
        cli.Settings,
        "load",
        classmethod(lambda cls, config_file=None, secrets_file=None: settings),
        raising=False,
    )
    monkeypatch.setattr(cli, "build_langsmith_gateway", lambda config: Gateway())
    monkeypatch.setattr(cli, "_baseline_policy", lambda *args: object())
    monkeypatch.setattr(
        cli,
        "promote_baseline",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            ValueError("Baseline promotion requires git_dirty=false")
        ),
    )

    exit_code = cli.main(
        [
            "baseline-promote",
            "--experiment-id",
            "exp-1",
            "--reason",
            "reviewed",
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 2
    assert "git_dirty=false" in captured.err
    assert updated is False

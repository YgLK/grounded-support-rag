from __future__ import annotations

from dataclasses import dataclass

from support_graph.cli import handlers as cli
from support_graph.evaluation.contracts import (
    DatasetRef,
    ExperimentSnapshot,
    GateResult,
    MetricDelta,
)


@dataclass(frozen=True)
class HostedResult:
    experiment: ExperimentSnapshot
    gate: GateResult


def _snapshot() -> ExperimentSnapshot:
    return ExperimentSnapshot(
        id="candidate-1",
        name="support-graph-kubernetes-smoke",
        dataset=DatasetRef("dataset-1", "support-graph-kubernetes-smoke-hash", "hash"),
        metadata={},
        results=(),
        url="https://smith.langchain.com/o/project/candidate-1",
    )


def _configure_hosted(monkeypatch, make_settings):
    settings = make_settings()
    settings.runtime.validate_for_hosted_eval = lambda: None
    monkeypatch.setattr(
        cli.Settings,
        "load",
        classmethod(lambda cls, config_file=None, secrets_file=None: settings),
        raising=False,
    )
    monkeypatch.setattr(
        cli,
        "load_eval_examples",
        lambda *args, **kwargs: ([{"example_id": "k8s-001"}], "smoke"),
        raising=False,
    )
    monkeypatch.setattr(cli, "_baseline_policy", lambda *args: object())
    monkeypatch.setattr(cli, "build_langsmith_gateway", lambda config: object())
    monkeypatch.setattr(cli, "read_git_state", lambda root: object())
    return settings


def test_eval_cli_reports_missing_hosted_config(monkeypatch, capsys, make_settings):
    settings = make_settings()
    monkeypatch.setattr(
        cli.Settings,
        "load",
        classmethod(lambda cls, config_file=None, secrets_file=None: settings),
        raising=False,
    )

    exit_code = cli.main(["eval", "--split", "validation", "--domain", "kubernetes"])
    captured = capsys.readouterr()

    assert exit_code == 2
    assert "SupportGraph Eval" not in captured.out
    assert "LANGSMITH_API_KEY" in captured.err


def test_eval_cli_reports_hosted_success(monkeypatch, capsys, make_settings):
    _configure_hosted(monkeypatch, make_settings)
    snapshot = _snapshot()
    delta = MetricDelta("doc_recall_at_3", 0.8, 0.75, 0.05, 0.1)
    hosted = HostedResult(
        experiment=snapshot,
        gate=GateResult(
            status="passed",
            baseline_experiment_id="baseline-1",
            candidate_experiment_id=snapshot.id,
            deltas=(delta,),
            reasons=(),
        ),
    )
    monkeypatch.setattr(cli, "run_hosted_evaluation", lambda **kwargs: hosted)

    exit_code = cli.main(["eval", "--split", "validation", "--domain", "kubernetes"])
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "experiment: candidate-1" in output
    assert "status: passed" in output
    assert "baseline: baseline-1" in output
    assert "doc_recall_at_3: 0.75" in output
    assert "https://smith.langchain.com/o/project/candidate-1" in output


def test_eval_cli_no_baseline_is_invalid(monkeypatch, capsys, make_settings):
    _configure_hosted(monkeypatch, make_settings)
    snapshot = _snapshot()
    hosted = HostedResult(
        experiment=snapshot,
        gate=GateResult(
            status="invalid",
            baseline_experiment_id=None,
            candidate_experiment_id=snapshot.id,
            reasons=("No promoted baseline for dataset",),
        ),
    )
    monkeypatch.setattr(cli, "run_hosted_evaluation", lambda **kwargs: hosted)

    exit_code = cli.main(["eval", "--split", "validation", "--domain", "kubernetes"])
    output = capsys.readouterr().out

    assert exit_code == 2
    assert "status: invalid" in output
    assert "baseline: none" in output
    assert "reason: No promoted baseline for dataset" in output


def test_eval_cli_connection_error_returns_two_without_eval_artifacts(
    monkeypatch,
    capsys,
    make_settings,
    tmp_path,
):
    settings = make_settings(project_root=tmp_path)
    settings.runtime.validate_for_hosted_eval = lambda: None
    monkeypatch.setattr(
        cli.Settings,
        "load",
        classmethod(lambda cls, config_file=None, secrets_file=None: settings),
        raising=False,
    )
    monkeypatch.setattr(
        cli,
        "load_eval_examples",
        lambda *args, **kwargs: ([{"example_id": "k8s-001"}], "smoke"),
        raising=False,
    )
    monkeypatch.setattr(cli, "_baseline_policy", lambda *args: object())
    monkeypatch.setattr(cli, "build_langsmith_gateway", lambda config: object())
    monkeypatch.setattr(cli, "read_git_state", lambda root: object())

    def fail(**kwargs):
        raise OSError("LangSmith unavailable")

    monkeypatch.setattr(cli, "run_hosted_evaluation", fail)

    exit_code = cli.main(["eval", "--split", "validation", "--domain", "kubernetes"])
    captured = capsys.readouterr()

    assert exit_code == 2
    assert "LangSmith unavailable" in captured.err
    assert not settings.paths.eval_runs_dir.exists()

from __future__ import annotations

from support_graph.cli import handlers as cli
from support_graph.evaluation.contracts import DatasetRef


def test_doctor_reports_missing_hosted_configuration(
    monkeypatch, capsys, make_settings
):
    settings = make_settings()
    monkeypatch.setattr(
        cli.Settings,
        "load",
        classmethod(lambda cls, config_file=None, secrets_file=None: settings),
        raising=False,
    )

    exit_code = cli.main(["doctor", "--domain", "kubernetes"])
    output = capsys.readouterr().out

    assert exit_code == 1
    assert "LangSmith connectivity: blocked" in output
    assert "LANGSMITH_API_KEY" in output


def test_doctor_checks_dataset_and_promoted_baseline_without_mutation(
    monkeypatch,
    capsys,
    make_settings,
):
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

    class Gateway:
        async def ping(self):
            return None

        async def get_dataset(self, name):
            return DatasetRef("dataset", name, "hash")

    monkeypatch.setattr(cli, "build_langsmith_gateway", lambda config: Gateway())
    monkeypatch.setattr(
        cli,
        "find_promoted_baseline",
        lambda *args, **kwargs: None,
        raising=False,
    )
    monkeypatch.setattr(cli, "dataset_sha256", lambda examples: "hash")

    exit_code = cli.main(["doctor", "--domain", "kubernetes"])
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "LangSmith connectivity: ok" in output
    assert "Dataset: ok" in output
    assert "Promoted baseline: missing" in output


def test_doctor_does_not_create_output_directories(
    monkeypatch,
    capsys,
    make_settings,
    tmp_path,
):
    settings = make_settings(project_root=tmp_path)
    monkeypatch.setattr(
        cli.Settings,
        "load",
        classmethod(lambda cls, config_file=None, secrets_file=None: settings),
        raising=False,
    )

    cli.main(["doctor", "--domain", "kubernetes"])
    capsys.readouterr()

    assert not (tmp_path / "outputs").exists()
    assert not settings.paths.log_dir.exists()

from __future__ import annotations

from pathlib import Path

from support_graph.cli import handlers as cli


def test_fetch_kubernetes_docs_cli_prints_manifest(
    monkeypatch,
    capsys,
    tmp_path: Path,
    make_settings,
) -> None:
    settings = make_settings(project_root=tmp_path)
    captured: dict = {}

    def fake_fetch(output_dir, *, ref, replace):
        captured["output_dir"] = output_dir
        captured["ref"] = ref
        captured["replace"] = replace
        return {
            "requested_ref": ref,
            "resolved_sha": "abc123",
            "docs_path": "content/en/docs",
        }

    monkeypatch.setattr(
        cli.Settings,
        "load",
        classmethod(lambda cls, config_file=None, secrets_file=None: settings),
        raising=False,
    )
    monkeypatch.setattr(cli, "fetch_kubernetes_docs", fake_fetch, raising=False)

    exit_code = cli.main(
        [
            "fetch-kubernetes-docs",
            "--ref",
            "v1.30.0",
            "--output",
            "raw/kubernetes/current",
            "--replace",
        ]
    )
    output = capsys.readouterr().out.splitlines()

    assert exit_code == 0
    assert captured == {
        "output_dir": tmp_path / "raw/kubernetes/current",
        "ref": "v1.30.0",
        "replace": True,
    }
    assert output[0] == "SupportGraph Fetch Kubernetes Docs"
    assert "Resolved SHA: abc123" in output

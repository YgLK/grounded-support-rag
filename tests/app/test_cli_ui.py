from __future__ import annotations

from support_graph.cli import handlers as cli


def test_ui_cli_serves_workbench_with_local_defaults(
    monkeypatch, make_settings
) -> None:
    fake_settings = make_settings()
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        cli.Settings,
        "load",
        classmethod(lambda cls, config_file=None, secrets_file=None: fake_settings),
        raising=False,
    )
    monkeypatch.setattr(cli, "configure_logging", lambda **kwargs: None, raising=False)

    import support_graph.ui as ui
    import uvicorn

    monkeypatch.setattr(ui, "create_app", lambda: "workbench-app", raising=False)

    def fake_run(app, *, host, port):
        captured["app"] = app
        captured["host"] = host
        captured["port"] = port

    monkeypatch.setattr(uvicorn, "run", fake_run, raising=False)

    exit_code = cli.main(["ui"])

    assert exit_code == 0
    assert captured == {
        "app": "workbench-app",
        "host": "127.0.0.1",
        "port": 8008,
    }

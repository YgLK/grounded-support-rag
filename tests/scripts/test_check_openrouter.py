from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

from support_graph.providers import Provider


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPT_PATH = REPO_ROOT / "scripts" / "check_openrouter.py"


def _load_script_module():
    spec = importlib.util.spec_from_file_location(
        "check_openrouter_script", SCRIPT_PATH
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_run_checks_requires_openrouter_configuration() -> None:
    module = _load_script_module()
    settings = SimpleNamespace(
        runtime=SimpleNamespace(
            chat_provider_type=Provider.OLLAMA,
            embedding_provider_type=Provider.OLLAMA,
            openrouter_api_key=None,
            chat_model=None,
            embedding_model=None,
        )
    )

    try:
        module.run_checks(
            settings,
            check_chat=True,
            check_embeddings=True,
        )
    except ValueError as exc:
        message = str(exc)
    else:
        raise AssertionError("Expected ValueError")

    assert 'support_graph.toml: runtime.chat_provider_type = "openrouter"' in message
    assert (
        'support_graph.toml: runtime.embedding_provider_type = "openrouter"' in message
    )
    assert ".env: SUPPORT_GRAPH_OPENROUTER_API_KEY" in message
    assert "support_graph.toml: runtime.chat_model" in message
    assert "support_graph.toml: runtime.embedding_model" in message


def test_run_checks_calls_chat_and_embeddings(monkeypatch) -> None:
    module = _load_script_module()

    calls: list[str] = []

    class FakeChatModel:
        def invoke(self, prompt: str):
            calls.append(f"chat:{prompt}")
            return SimpleNamespace(content="OK")

    class FakeEmbeddings:
        def embed_query(self, text: str):
            calls.append(f"embed:{text}")
            return [0.1, 0.2, 0.3]

    monkeypatch.setattr(module, "build_chat_model", lambda settings: FakeChatModel())
    monkeypatch.setattr(module, "build_embeddings", lambda settings: FakeEmbeddings())

    settings = SimpleNamespace(
        runtime=SimpleNamespace(
            chat_provider_type=Provider.OPENROUTER,
            embedding_provider_type=Provider.OPENROUTER,
            ollama_base_url="http://localhost:11434",
            openrouter_base_url="https://openrouter.ai/api/v1",
            openrouter_api_key="sk-or-v1-test",
            chat_model="openai/gpt-4.1-mini",
            embedding_model="openai/text-embedding-3-small",
        )
    )

    results = module.run_checks(
        settings,
        check_chat=True,
        check_embeddings=True,
    )

    assert calls == [
        f"chat:{module.DEFAULT_CHAT_PROMPT}",
        f"embed:{module.DEFAULT_EMBED_TEXT}",
    ]
    assert [(item.name, item.ok) for item in results] == [
        ("chat", True),
        ("embeddings", True),
    ]


def test_extract_text_handles_list_content() -> None:
    module = _load_script_module()
    response = SimpleNamespace(content=[{"text": "O"}, {"text": "K"}])

    assert module._extract_text(response) == "OK"

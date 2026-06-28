from __future__ import annotations

from types import SimpleNamespace

import pytest

from support_graph import providers
from support_graph.types import DatasetSplit, Domain, EvalSubset


def test_shared_enums_expose_supported_values_without_duplicate_constants() -> None:
    assert Domain.values() == ("kubernetes",)
    assert DatasetSplit.value_set() == frozenset({"train", "validation", "test"})
    assert EvalSubset.value_set() == frozenset(
        {"smoke", "frozen_experiment", "full_validation"}
    )
    assert providers.Provider.values() == ("ollama", "openrouter")


def test_build_chat_model_uses_langchain_provider_abstraction_for_ollama() -> None:
    captured: dict = {}

    def fake_init_chat_model(model, *, model_provider=None, **kwargs):
        captured["model"] = model
        captured["model_provider"] = model_provider
        captured["kwargs"] = kwargs
        return "chat-client"

    config = SimpleNamespace(
        chat_provider_type="ollama",
        embedding_provider_type="ollama",
        chat_model="qwen3:8b-q4_K_M",
        openrouter_api_key=None,
        openrouter_base_url=None,
        ollama_base_url="http://localhost:11434",
    )

    result = providers.build_chat_model(
        config,
        init_chat_model_fn=fake_init_chat_model,
    )

    assert result == "chat-client"
    assert captured == {
        "model": "qwen3:8b-q4_K_M",
        "model_provider": "ollama",
        "kwargs": {
            "temperature": 0,
            "base_url": "http://localhost:11434",
        },
    }


def test_build_chat_model_routes_openrouter_through_openai_provider() -> None:
    captured: dict = {}

    def fake_init_chat_model(model, *, model_provider=None, **kwargs):
        captured["model"] = model
        captured["model_provider"] = model_provider
        captured["kwargs"] = kwargs
        return "chat-client"

    config = SimpleNamespace(
        chat_provider_type="openrouter",
        embedding_provider_type="openrouter",
        chat_model="google/gemini-2.5-flash-preview",
        openrouter_api_key="or-test",
        openrouter_base_url="https://openrouter.ai/api/v1",
        ollama_base_url=None,
    )

    result = providers.build_chat_model(
        config,
        init_chat_model_fn=fake_init_chat_model,
    )

    assert result == "chat-client"
    assert captured == {
        "model": "google/gemini-2.5-flash-preview",
        "model_provider": "openai",
        "kwargs": {
            "temperature": 0,
            "api_key": "or-test",
            "base_url": "https://openrouter.ai/api/v1",
        },
    }


def test_build_embeddings_routes_openrouter_through_openai_provider() -> None:
    captured: dict = {}

    def fake_init_embeddings(model, *, provider=None, **kwargs):
        captured["model"] = model
        captured["provider"] = provider
        captured["kwargs"] = kwargs
        return "embedding-client"

    config = SimpleNamespace(
        chat_provider_type="ollama",
        embedding_provider_type="openrouter",
        embedding_model="openai/text-embedding-3-small",
        embedding_client=None,
        openrouter_api_key="or-test",
        openrouter_base_url="https://openrouter.ai/api/v1",
        ollama_base_url="http://localhost:11434",
    )

    result = providers.build_embeddings(
        config,
        init_embeddings_fn=fake_init_embeddings,
    )

    assert result == "embedding-client"
    assert captured == {
        "model": "openai/text-embedding-3-small",
        "provider": "openai",
        "kwargs": {
            "api_key": "or-test",
            "base_url": "https://openrouter.ai/api/v1",
        },
    }


def test_validate_provider_type_rejects_removed_openai_provider() -> None:
    with pytest.raises(ValueError, match="Unsupported provider_type"):
        providers.validate_chat_provider_type("openai")


@pytest.mark.parametrize("value", [None, "", "   "])
def test_validate_provider_type_rejects_none_or_blank(value: str | None) -> None:
    with pytest.raises(ValueError, match="Unsupported provider_type"):
        providers.validate_chat_provider_type(value)

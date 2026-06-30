from __future__ import annotations

from types import SimpleNamespace

from support_graph import providers
from support_graph.types import EvalSubset


def test_eval_subset_expanded_is_a_known_choice() -> None:
    assert EvalSubset.EXPANDED == "expanded"
    assert "expanded" in EvalSubset.value_set()
    assert EvalSubset.EXPANDED in {
        EvalSubset.SMOKE,
        EvalSubset.EXPANDED,
        EvalSubset.FROZEN_EXPERIMENT,
    }


def _config(**overrides) -> SimpleNamespace:
    base = dict(
        chat_provider_type="openrouter",
        embedding_provider_type="openrouter",
        chat_model="openai/gpt-oss-120b:nitro",
        openrouter_api_key="or-test",
        openrouter_base_url="https://openrouter.ai/api/v1",
        ollama_base_url=None,
        chat_temperature=0.0,
        chat_seed=None,
        openrouter_provider_order=None,
        openrouter_allow_fallbacks=None,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def test_chat_provider_kwargs_defaults_preserve_temperature_zero() -> None:
    captured: dict = {}

    def fake_init(model, *, model_provider=None, **kwargs):
        captured.update(kwargs)
        return "chat"

    providers.build_chat_model(_config(), init_chat_model_fn=fake_init)
    assert captured["temperature"] == 0.0
    assert "seed" not in captured
    assert "extra_body" not in captured


def test_chat_provider_kwargs_passes_seed_when_set() -> None:
    captured: dict = {}

    def fake_init(model, *, model_provider=None, **kwargs):
        captured.update(kwargs)
        return "chat"

    providers.build_chat_model(_config(chat_seed=12345), init_chat_model_fn=fake_init)
    assert captured["seed"] == 12345


def test_chat_provider_kwargs_passes_provider_routing_extra_body() -> None:
    captured: dict = {}

    def fake_init(model, *, model_provider=None, **kwargs):
        captured.update(kwargs)
        return "chat"

    providers.build_chat_model(
        _config(
            openrouter_provider_order=("OpenAI",),
            openrouter_allow_fallbacks=False,
        ),
        init_chat_model_fn=fake_init,
    )
    assert captured["extra_body"] == {
        "provider": {"order": ["OpenAI"], "allow_fallback": False}
    }


def test_chat_provider_kwargs_omits_extra_body_when_routing_unset() -> None:
    captured: dict = {}

    def fake_init(model, *, model_provider=None, **kwargs):
        captured.update(kwargs)
        return "chat"

    providers.build_chat_model(_config(), init_chat_model_fn=fake_init)
    assert "extra_body" not in captured


def test_chat_provider_kwargs_uses_configurable_temperature() -> None:
    captured: dict = {}

    def fake_init(model, *, model_provider=None, **kwargs):
        captured.update(kwargs)
        return "chat"

    providers.build_chat_model(
        _config(chat_temperature=0.2), init_chat_model_fn=fake_init
    )
    assert captured["temperature"] == 0.2

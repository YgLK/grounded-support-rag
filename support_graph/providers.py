"""Provider abstraction helpers for chat models and embeddings."""

from __future__ import annotations

from typing import Any

from langchain.chat_models import init_chat_model
from langchain.embeddings import init_embeddings


DEFAULT_PROVIDER_TYPE = "ollama"
SUPPORTED_CHAT_PROVIDERS = frozenset({"anthropic", "ollama", "openai"})
SUPPORTED_EMBEDDING_PROVIDERS = frozenset({"ollama", "openai"})


def normalize_provider_type(
    value: str | None,
    *,
    default: str = DEFAULT_PROVIDER_TYPE,
) -> str:
    resolved = str(value or default).strip().lower()
    return resolved or default


def validate_chat_provider_type(value: str | None) -> str:
    provider = normalize_provider_type(value)
    if provider not in SUPPORTED_CHAT_PROVIDERS:
        supported = ", ".join(sorted(SUPPORTED_CHAT_PROVIDERS))
        raise ValueError(
            f"Unsupported provider_type '{provider}'. Supported providers: {supported}."
        )
    return provider


def resolve_embedding_provider_type(config: Any) -> str:
    explicit_provider = getattr(config, "embedding_provider_type", None)
    provider = normalize_provider_type(
        explicit_provider or getattr(config, "provider_type", None)
    )
    if provider == "anthropic":
        raise ValueError(
            "Anthropic chat models do not provide embeddings through LangChain. "
            "Set SUPPORT_GRAPH_EMBEDDING_PROVIDER_TYPE to 'openai' or 'ollama'."
        )
    if provider not in SUPPORTED_EMBEDDING_PROVIDERS:
        supported = ", ".join(sorted(SUPPORTED_EMBEDDING_PROVIDERS))
        raise ValueError(
            "Unsupported embedding provider_type "
            f"'{provider}'. Supported providers: {supported}."
        )
    return provider


def chat_provider_base_url(config: Any) -> str | None:
    provider = validate_chat_provider_type(getattr(config, "provider_type", None))
    if provider == "ollama":
        return getattr(config, "ollama_base_url", None)
    if provider == "openai":
        return getattr(config, "openai_base_url", None)
    if provider == "anthropic":
        return getattr(config, "anthropic_base_url", None)
    return None


def embedding_provider_base_url(config: Any) -> str | None:
    provider = resolve_embedding_provider_type(config)
    if provider == "ollama":
        return getattr(config, "ollama_base_url", None)
    if provider == "openai":
        return getattr(config, "openai_base_url", None)
    return None


def _chat_provider_kwargs(config: Any, provider: str) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"temperature": 0}
    if provider == "ollama":
        kwargs["base_url"] = getattr(config, "ollama_base_url", None)
    elif provider == "openai":
        kwargs["api_key"] = getattr(config, "openai_api_key", None)
        kwargs["base_url"] = getattr(config, "openai_base_url", None)
    elif provider == "anthropic":
        kwargs["api_key"] = getattr(config, "anthropic_api_key", None)
        kwargs["base_url"] = getattr(config, "anthropic_base_url", None)
    return {key: value for key, value in kwargs.items() if value is not None}


def _embedding_provider_kwargs(config: Any, provider: str) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    if provider == "ollama":
        kwargs["base_url"] = getattr(config, "ollama_base_url", None)
    elif provider == "openai":
        kwargs["api_key"] = getattr(config, "openai_api_key", None)
        kwargs["base_url"] = getattr(config, "openai_base_url", None)
    return {key: value for key, value in kwargs.items() if value is not None}


def _allow_legacy_embedding_injection(config: Any) -> bool:
    return (
        getattr(config, "provider_type", None) is None
        and getattr(config, "embedding_provider_type", None) is None
        and getattr(config, "ollama_base_url", None) is None
        and getattr(config, "openai_api_key", None) is None
        and getattr(config, "openai_base_url", None) is None
    )


def build_chat_model(
    config: Any,
    *,
    chat_model_cls: type[Any] | None = None,
    init_chat_model_fn: Any = init_chat_model,
) -> Any:
    chat_model = getattr(config, "chat_model", None)
    if not chat_model:
        raise ValueError("Missing chat_model for provider-backed runtime.")

    provider = validate_chat_provider_type(getattr(config, "provider_type", None))
    if chat_model_cls is not None:
        if provider != "ollama":
            raise ValueError(
                "Legacy chat_model_cls injection is only supported for provider_type='ollama'."
            )
        return chat_model_cls(
            model=chat_model,
            base_url=getattr(config, "ollama_base_url", None),
            temperature=0,
        )

    return init_chat_model_fn(
        chat_model,
        model_provider=provider,
        **_chat_provider_kwargs(config, provider),
    )


def build_embeddings(
    config: Any,
    *,
    embeddings_cls: type[Any] | None = None,
    init_embeddings_fn: Any = init_embeddings,
) -> Any:
    if getattr(config, "embedding_client", None) is not None:
        return config.embedding_client

    embedding_model = getattr(config, "embedding_model", None)
    if not embedding_model:
        raise ValueError("Missing embedding_model for provider-backed retrieval.")

    if _allow_legacy_embedding_injection(config):
        return embedding_model

    provider = resolve_embedding_provider_type(config)
    if embeddings_cls is not None:
        if provider != "ollama":
            raise ValueError(
                "Legacy embeddings_cls injection is only supported for provider_type='ollama'."
            )
        return embeddings_cls(
            model=embedding_model,
            base_url=getattr(config, "ollama_base_url", None),
        )

    return init_embeddings_fn(
        embedding_model,
        provider=provider,
        **_embedding_provider_kwargs(config, provider),
    )


__all__ = [
    "DEFAULT_PROVIDER_TYPE",
    "SUPPORTED_CHAT_PROVIDERS",
    "SUPPORTED_EMBEDDING_PROVIDERS",
    "build_chat_model",
    "build_embeddings",
    "chat_provider_base_url",
    "embedding_provider_base_url",
    "normalize_provider_type",
    "resolve_embedding_provider_type",
    "validate_chat_provider_type",
]

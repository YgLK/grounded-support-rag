"""Provider abstraction helpers for chat models and embeddings."""

from __future__ import annotations

from typing import Any, Protocol

from langchain.chat_models import init_chat_model
from langchain.embeddings import init_embeddings

from support_graph.types import ChoiceStrEnum


class Provider(ChoiceStrEnum):
    OLLAMA = "ollama"
    OPENROUTER = "openrouter"


DEFAULT_PROVIDER_TYPE = Provider.OPENROUTER
DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class ProviderConfigLike(Protocol):
    provider_type: Provider | None
    embedding_provider_type: Provider | None
    ollama_base_url: str | None
    openrouter_base_url: str | None
    openrouter_api_key: str | None
    chat_model: str | None
    embedding_model: str | None
    embedding_client: Any | None


def normalize_provider_type(value: str | Provider | None) -> Provider:
    if value is None:
        return DEFAULT_PROVIDER_TYPE
    if isinstance(value, Provider):
        return value
    normalized = str(value).strip()
    if not normalized:
        return DEFAULT_PROVIDER_TYPE
    return Provider.parse(normalized)


def validate_chat_provider_type(value: str | Provider | None) -> Provider:
    try:
        return normalize_provider_type(value)
    except ValueError as exc:
        supported = ", ".join(sorted(Provider.value_set()))
        raise ValueError(
            f"Unsupported provider_type '{value}'. Supported providers: {supported}."
        ) from exc


def validate_embedding_provider_type(value: str | Provider | None) -> Provider:
    return validate_chat_provider_type(value)


def resolve_embedding_provider_type(config: ProviderConfigLike) -> Provider:
    return validate_embedding_provider_type(
        config.embedding_provider_type or config.provider_type
    )


def chat_provider_base_url(config: ProviderConfigLike) -> str | None:
    provider = validate_chat_provider_type(config.provider_type)
    match provider:
        case Provider.OLLAMA:
            return config.ollama_base_url
        case Provider.OPENROUTER:
            return config.openrouter_base_url


def embedding_provider_base_url(config: ProviderConfigLike) -> str | None:
    provider = resolve_embedding_provider_type(config)
    match provider:
        case Provider.OLLAMA:
            return config.ollama_base_url
        case Provider.OPENROUTER:
            return config.openrouter_base_url


def _langchain_provider_name(provider: Provider) -> str:
    if provider is Provider.OPENROUTER:
        return "openai"
    return str(provider)


def _without_none(kwargs: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in kwargs.items() if value is not None}


def _chat_provider_kwargs(
    config: ProviderConfigLike,
    provider: Provider,
) -> dict[str, Any]:
    match provider:
        case Provider.OLLAMA:
            return _without_none(
                {
                    "temperature": 0,
                    "base_url": config.ollama_base_url,
                }
            )
        case Provider.OPENROUTER:
            return _without_none(
                {
                    "temperature": 0,
                    "api_key": config.openrouter_api_key,
                    "base_url": config.openrouter_base_url,
                }
            )


def _embedding_provider_kwargs(
    config: ProviderConfigLike,
    provider: Provider,
) -> dict[str, Any]:
    match provider:
        case Provider.OLLAMA:
            return _without_none({"base_url": config.ollama_base_url})
        case Provider.OPENROUTER:
            return _without_none(
                {
                    "api_key": config.openrouter_api_key,
                    "base_url": config.openrouter_base_url,
                }
            )


def _allow_legacy_embedding_injection(config: ProviderConfigLike) -> bool:
    return (
        config.provider_type is None
        and config.embedding_provider_type is None
        and config.ollama_base_url is None
        and config.openrouter_api_key is None
        and config.openrouter_base_url is None
    )


def build_chat_model(
    config: ProviderConfigLike,
    *,
    chat_model_cls: type[Any] | None = None,
    init_chat_model_fn: Any = init_chat_model,
) -> Any:
    chat_model = config.chat_model
    if not chat_model:
        raise ValueError("Missing chat_model for provider-backed runtime.")

    provider = validate_chat_provider_type(config.provider_type)
    if chat_model_cls is not None:
        if provider is not Provider.OLLAMA:
            raise ValueError(
                "Legacy chat_model_cls injection is only supported for provider_type='ollama'."
            )
        return chat_model_cls(
            model=chat_model,
            base_url=config.ollama_base_url,
            temperature=0,
        )

    return init_chat_model_fn(
        chat_model,
        model_provider=_langchain_provider_name(provider),
        **_chat_provider_kwargs(config, provider),
    )


def build_embeddings(
    config: ProviderConfigLike,
    *,
    embeddings_cls: type[Any] | None = None,
    init_embeddings_fn: Any = init_embeddings,
) -> Any:
    if config.embedding_client is not None:
        return config.embedding_client

    embedding_model = config.embedding_model
    if not embedding_model:
        raise ValueError("Missing embedding_model for provider-backed retrieval.")

    if _allow_legacy_embedding_injection(config):
        return embedding_model

    provider = resolve_embedding_provider_type(config)
    if embeddings_cls is not None:
        if provider is not Provider.OLLAMA:
            raise ValueError(
                "Legacy embeddings_cls injection is only supported for provider_type='ollama'."
            )
        return embeddings_cls(
            model=embedding_model,
            base_url=config.ollama_base_url,
        )

    return init_embeddings_fn(
        embedding_model,
        provider=_langchain_provider_name(provider),
        **_embedding_provider_kwargs(config, provider),
    )


__all__ = [
    "DEFAULT_PROVIDER_TYPE",
    "DEFAULT_OPENROUTER_BASE_URL",
    "Provider",
    "ProviderConfigLike",
    "build_chat_model",
    "build_embeddings",
    "chat_provider_base_url",
    "embedding_provider_base_url",
    "normalize_provider_type",
    "resolve_embedding_provider_type",
    "validate_embedding_provider_type",
    "validate_chat_provider_type",
]

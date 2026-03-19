"""Provider abstraction helpers for chat models and embeddings."""

from __future__ import annotations

from typing import Any, Literal, Protocol, cast

from langchain.chat_models import init_chat_model
from langchain.embeddings import init_embeddings


ChatProviderType = Literal["anthropic", "ollama", "openai"]
EmbeddingProviderType = Literal["ollama", "openai"]

DEFAULT_PROVIDER_TYPE: ChatProviderType = "ollama"
SUPPORTED_CHAT_PROVIDERS = frozenset({"anthropic", "ollama", "openai"})
SUPPORTED_EMBEDDING_PROVIDERS = frozenset({"ollama", "openai"})


class ProviderConfigLike(Protocol):
    provider_type: ChatProviderType | None
    embedding_provider_type: EmbeddingProviderType | None
    ollama_base_url: str | None
    openai_base_url: str | None
    openai_api_key: str | None
    anthropic_base_url: str | None
    anthropic_api_key: str | None
    chat_model: str | None
    embedding_model: str | None
    embedding_client: Any | None


def normalize_provider_type(value: str | None) -> str:
    if value is None:
        return DEFAULT_PROVIDER_TYPE
    resolved = value.strip().lower()
    return resolved or DEFAULT_PROVIDER_TYPE


def validate_chat_provider_type(value: str | None) -> ChatProviderType:
    provider = normalize_provider_type(value)
    if provider not in SUPPORTED_CHAT_PROVIDERS:
        supported = ", ".join(sorted(SUPPORTED_CHAT_PROVIDERS))
        raise ValueError(
            f"Unsupported provider_type '{provider}'. Supported providers: {supported}."
        )
    return cast(ChatProviderType, provider)


def validate_embedding_provider_type(value: str | None) -> EmbeddingProviderType:
    provider = normalize_provider_type(value)
    if provider not in SUPPORTED_EMBEDDING_PROVIDERS:
        supported = ", ".join(sorted(SUPPORTED_EMBEDDING_PROVIDERS))
        raise ValueError(
            "Unsupported embedding provider_type "
            f"'{provider}'. Supported providers: {supported}."
        )
    return cast(EmbeddingProviderType, provider)


def resolve_embedding_provider_type(
    config: ProviderConfigLike,
) -> EmbeddingProviderType:
    provider = normalize_provider_type(
        config.embedding_provider_type or config.provider_type
    )
    if provider == "anthropic":
        raise ValueError(
            "Anthropic chat models do not provide embeddings through LangChain. "
            "Set SUPPORT_GRAPH_EMBEDDING_PROVIDER_TYPE to 'openai' or 'ollama'."
        )
    return validate_embedding_provider_type(provider)


def chat_provider_base_url(config: ProviderConfigLike) -> str | None:
    provider = validate_chat_provider_type(config.provider_type)
    if provider == "ollama":
        return config.ollama_base_url
    if provider == "openai":
        return config.openai_base_url
    if provider == "anthropic":
        return config.anthropic_base_url
    raise AssertionError(provider)


def embedding_provider_base_url(config: ProviderConfigLike) -> str | None:
    provider = resolve_embedding_provider_type(config)
    if provider == "ollama":
        return config.ollama_base_url
    if provider == "openai":
        return config.openai_base_url
    raise AssertionError(provider)


def _without_none(kwargs: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in kwargs.items() if value is not None}


def _chat_provider_kwargs(
    config: ProviderConfigLike,
    provider: ChatProviderType,
) -> dict[str, Any]:
    if provider == "ollama":
        return _without_none(
            {
                "temperature": 0,
                "base_url": config.ollama_base_url,
            }
        )
    if provider == "openai":
        return _without_none(
            {
                "temperature": 0,
                "api_key": config.openai_api_key,
                "base_url": config.openai_base_url,
            }
        )
    if provider == "anthropic":
        return _without_none(
            {
                "temperature": 0,
                "api_key": config.anthropic_api_key,
                "base_url": config.anthropic_base_url,
            }
        )
    raise AssertionError(provider)


def _embedding_provider_kwargs(
    config: ProviderConfigLike,
    provider: EmbeddingProviderType,
) -> dict[str, Any]:
    if provider == "ollama":
        return _without_none({"base_url": config.ollama_base_url})
    if provider == "openai":
        return _without_none(
            {
                "api_key": config.openai_api_key,
                "base_url": config.openai_base_url,
            }
        )
    raise AssertionError(provider)


def _allow_legacy_embedding_injection(config: ProviderConfigLike) -> bool:
    return (
        config.provider_type is None
        and config.embedding_provider_type is None
        and config.ollama_base_url is None
        and config.openai_api_key is None
        and config.openai_base_url is None
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
        if provider != "ollama":
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
        model_provider=provider,
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
        if provider != "ollama":
            raise ValueError(
                "Legacy embeddings_cls injection is only supported for provider_type='ollama'."
            )
        return embeddings_cls(
            model=embedding_model,
            base_url=config.ollama_base_url,
        )

    return init_embeddings_fn(
        embedding_model,
        provider=provider,
        **_embedding_provider_kwargs(config, provider),
    )


__all__ = [
    "ChatProviderType",
    "DEFAULT_PROVIDER_TYPE",
    "EmbeddingProviderType",
    "ProviderConfigLike",
    "SUPPORTED_CHAT_PROVIDERS",
    "SUPPORTED_EMBEDDING_PROVIDERS",
    "build_chat_model",
    "build_embeddings",
    "chat_provider_base_url",
    "embedding_provider_base_url",
    "normalize_provider_type",
    "resolve_embedding_provider_type",
    "validate_embedding_provider_type",
    "validate_chat_provider_type",
]

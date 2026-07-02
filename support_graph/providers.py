"""Provider abstraction helpers for chat models and embeddings."""

from __future__ import annotations

from typing import Any, Protocol

from langchain.chat_models import init_chat_model
from langchain.embeddings import init_embeddings

from support_graph.types import ChoiceStrEnum

__all__ = [
    "DEFAULT_PROVIDER_TYPE",
    "DEFAULT_OPENROUTER_BASE_URL",
    "Provider",
    "ProviderConfigLike",
    "build_chat_model",
    "build_embeddings",
    "chat_provider",
    "chat_provider_base_url",
    "embedding_provider",
    "embedding_provider_base_url",
    "normalize_provider_type",
    "validate_chat_provider_type",
]


class Provider(ChoiceStrEnum):
    OLLAMA = "ollama"
    OPENROUTER = "openrouter"


DEFAULT_PROVIDER_TYPE = Provider.OPENROUTER
DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class ProviderConfigLike(Protocol):
    """Read-only protocol for provider-backed config objects.

    Attributes are declared as ``@property`` so frozen dataclasses (such as
    ``RuntimeConfig``) satisfy the protocol without exposing mutable setters.
    """

    @property
    def chat_provider_type(self) -> Provider | None: ...

    @property
    def embedding_provider_type(self) -> Provider | None: ...

    @property
    def ollama_base_url(self) -> str | None: ...

    @property
    def openrouter_base_url(self) -> str | None: ...

    @property
    def openrouter_api_key(self) -> str | None: ...

    @property
    def chat_model(self) -> str | None: ...

    @property
    def embedding_model(self) -> str | None: ...

    @property
    def embedding_client(self) -> Any | None: ...

    @property
    def chat_temperature(self) -> float: ...

    @property
    def chat_seed(self) -> int | None: ...

    @property
    def openrouter_provider_order(self) -> tuple[str, ...] | None: ...

    @property
    def openrouter_allow_fallbacks(self) -> bool | None: ...


def normalize_provider_type(value: str | Provider | None) -> Provider:
    if isinstance(value, Provider):
        return value
    normalized = str(value).strip()
    return Provider.parse(normalized)


def validate_chat_provider_type(value: str | Provider | None) -> Provider:
    try:
        return normalize_provider_type(value)
    except ValueError as exc:
        supported = ", ".join(sorted(Provider.value_set()))
        raise ValueError(
            f"Unsupported provider_type '{value}'. Supported providers: {supported}."
        ) from exc


def chat_provider(config: ProviderConfigLike) -> Provider:
    return validate_chat_provider_type(config.chat_provider_type)


def embedding_provider(config: ProviderConfigLike) -> Provider:
    return validate_chat_provider_type(config.embedding_provider_type)


def chat_provider_base_url(config: ProviderConfigLike) -> str | None:
    provider = chat_provider(config)
    match provider:
        case Provider.OLLAMA:
            return config.ollama_base_url
        case Provider.OPENROUTER:
            return config.openrouter_base_url


def embedding_provider_base_url(config: ProviderConfigLike) -> str | None:
    provider = embedding_provider(config)
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


def _openrouter_routing_extra_body(
    config: ProviderConfigLike,
) -> dict[str, Any] | None:
    """Build the OpenRouter provider-routing extra body, or None if unset.

    OpenRouter accepts ``provider.order`` (a list of provider names to try in
    order) and ``provider.allow_fallback`` (bool) in the request body. We pass
    them through langchain-openai's ``extra_body`` so a run can pin a single
    provider instead of using the ``:nitro`` load-balancer.
    """
    provider_body: dict[str, Any] = {}
    order = getattr(config, "openrouter_provider_order", None)
    if order:
        provider_body["order"] = list(order)
    allow_fallbacks = getattr(config, "openrouter_allow_fallbacks", None)
    if allow_fallbacks is not None:
        provider_body["allow_fallback"] = bool(allow_fallbacks)
    if not provider_body:
        return None
    return {"provider": provider_body}


def _chat_provider_kwargs(
    config: ProviderConfigLike,
    provider: Provider,
) -> dict[str, Any]:
    temperature = float(getattr(config, "chat_temperature", 0.0))
    seed = getattr(config, "chat_seed", None)
    match provider:
        case Provider.OLLAMA:
            return _without_none(
                {
                    "temperature": temperature,
                    "base_url": config.ollama_base_url,
                    "seed": seed,
                }
            )
        case Provider.OPENROUTER:
            kwargs: dict[str, Any] = _without_none(
                {
                    "temperature": temperature,
                    "api_key": config.openrouter_api_key,
                    "base_url": config.openrouter_base_url,
                    "seed": seed,
                }
            )
            extra_body = _openrouter_routing_extra_body(config)
            if extra_body is not None:
                kwargs["extra_body"] = extra_body
            return kwargs


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


def build_chat_model(
    config: ProviderConfigLike,
    *,
    chat_model_cls: type[Any] | None = None,
    init_chat_model_fn: Any = init_chat_model,
) -> Any:
    chat_model = config.chat_model
    if not chat_model:
        raise ValueError("Missing chat_model for provider-backed runtime.")

    provider = chat_provider(config)
    if chat_model_cls is not None:
        if provider is not Provider.OLLAMA:
            raise ValueError(
                "Legacy chat_model_cls injection is only supported for chat_provider_type='ollama'."
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

    provider = embedding_provider(config)
    if embeddings_cls is not None:
        if provider is not Provider.OLLAMA:
            raise ValueError(
                "Legacy embeddings_cls injection is only supported for embedding_provider_type='ollama'."
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

"""Optional hosted observability helpers."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Literal, Protocol

from langsmith import Client
from langsmith.run_helpers import tracing_context
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.trace import ProxyTracerProvider


_OTEL_PROVIDER: TracerProvider | None = None
OtelExporter = Literal["console", "otlp"]


class ObservabilityConfigLike(Protocol):
    otel_enabled: bool
    otel_service_name: str
    otel_exporter: str | None
    otel_endpoint: str | None
    otel_headers: str | None
    langsmith_tracing_enabled: bool
    langsmith_project: str | None
    langsmith_api_key: str | None
    langsmith_endpoint: str | None


@dataclass(slots=True)
class Observability:
    tracer: Any | None = None
    otel_service_name: str | None = None
    otel_exporter: OtelExporter | None = None
    langsmith_project: str | None = None
    langsmith_client: Client | None = None

    @property
    def otel_enabled(self) -> bool:
        return self.tracer is not None

    @property
    def langsmith_enabled(self) -> bool:
        return self.langsmith_client is not None

    def summary(self) -> dict[str, Any]:
        summary: dict[str, Any] = {}
        if self.tracer is not None:
            summary["opentelemetry"] = {
                "enabled": True,
                "service_name": self.otel_service_name or "support-graph",
                "exporter": self.otel_exporter or "console",
            }
        if self.langsmith_client is not None:
            summary["langsmith"] = {
                "enabled": True,
                "project": self.langsmith_project or "support-graph",
            }
        return summary


def _parse_header_mapping(raw_headers: str | None) -> dict[str, str] | None:
    if not raw_headers:
        return None
    headers: dict[str, str] = {}
    for item in raw_headers.split(","):
        key, _, value = item.partition("=")
        normalized_key = key.strip()
        normalized_value = value.strip()
        if normalized_key and normalized_value:
            headers[normalized_key] = normalized_value
    return headers or None


def _resolve_otel_exporter(config: ObservabilityConfigLike) -> OtelExporter:
    exporter = str(config.otel_exporter or "").strip().lower()
    endpoint = config.otel_endpoint
    match exporter:
        case "console":
            return "console"
        case "" | "otlp":
            return "console" if endpoint is None else "otlp"
    raise ValueError(
        "Unsupported SUPPORT_GRAPH_OTEL_EXPORTER "
        f"'{exporter}'. Supported values: console, otlp."
    )


def _build_otel_exporter(
    config: ObservabilityConfigLike,
    exporter: OtelExporter,
) -> Any:
    match exporter:
        case "console":
            return ConsoleSpanExporter()
        case "otlp":
            endpoint = config.otel_endpoint
    if endpoint is None:
        raise ValueError("OTLP exporter requires otel_endpoint.")
    headers = _parse_header_mapping(config.otel_headers)
    return OTLPSpanExporter(endpoint=endpoint, headers=headers)


def _ensure_otel_provider(config: ObservabilityConfigLike) -> Any:
    global _OTEL_PROVIDER

    current_provider = trace.get_tracer_provider()
    if not isinstance(current_provider, ProxyTracerProvider):
        return current_provider

    if _OTEL_PROVIDER is None:
        exporter = _resolve_otel_exporter(config)
        provider = TracerProvider(
            resource=Resource.create(
                {
                    "service.name": config.otel_service_name or "support-graph",
                }
            )
        )
        provider.add_span_processor(
            BatchSpanProcessor(_build_otel_exporter(config, exporter))
        )
        trace.set_tracer_provider(provider)
        _OTEL_PROVIDER = provider
    return trace.get_tracer_provider()


def build_observability(config: ObservabilityConfigLike) -> Observability | None:
    if not config.otel_enabled and not config.langsmith_tracing_enabled:
        return None

    observability = Observability()

    if config.otel_enabled:
        exporter = _resolve_otel_exporter(config)
        provider = _ensure_otel_provider(config)
        observability.tracer = trace.get_tracer(
            "support_graph.runtime",
            tracer_provider=provider,
        )
        observability.otel_service_name = config.otel_service_name
        observability.otel_exporter = exporter

    if config.langsmith_tracing_enabled:
        client_kwargs: dict[str, Any] = {}
        if config.langsmith_api_key:
            client_kwargs["api_key"] = config.langsmith_api_key
        if config.langsmith_endpoint:
            client_kwargs["api_url"] = config.langsmith_endpoint
        observability.langsmith_project = config.langsmith_project or "support-graph"
        observability.langsmith_client = Client(**client_kwargs)

    return observability


def graph_run_context(
    observability: Observability | None,
    *,
    project_name: str | None = None,
    tags: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> Any:
    if observability is None or observability.langsmith_client is None:
        return nullcontext()
    project = project_name or observability.langsmith_project
    if project is None:
        raise ValueError("LangSmith tracing requires a project name.")
    return tracing_context(
        project_name=project,
        tags=tags,
        metadata=metadata,
        enabled=True,
        client=observability.langsmith_client,
    )


def span_context(
    observability: Observability | None,
    name: str,
    *,
    attributes: dict[str, Any] | None = None,
) -> Any:
    if observability is None or observability.tracer is None:
        return nullcontext()
    return observability.tracer.start_as_current_span(
        name,
        attributes=attributes,
        record_exception=True,
        set_status_on_exception=True,
    )


__all__ = [
    "Observability",
    "ObservabilityConfigLike",
    "OtelExporter",
    "build_observability",
    "graph_run_context",
    "span_context",
]

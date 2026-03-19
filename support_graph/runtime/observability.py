"""Optional hosted observability helpers."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

from langsmith import Client
from langsmith.run_helpers import tracing_context
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.trace import ProxyTracerProvider


_OTEL_PROVIDER: TracerProvider | None = None


@dataclass(slots=True)
class Observability:
    tracer: Any | None = None
    otel_enabled: bool = False
    otel_service_name: str | None = None
    otel_exporter: str | None = None
    langsmith_enabled: bool = False
    langsmith_project: str | None = None
    langsmith_client: Any | None = None

    def summary(self) -> dict[str, Any]:
        summary: dict[str, Any] = {}
        if self.otel_enabled:
            summary["opentelemetry"] = {
                "enabled": True,
                "service_name": self.otel_service_name or "support-graph",
                "exporter": self.otel_exporter or "console",
            }
        if self.langsmith_enabled:
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


def _build_otel_exporter(config: Any) -> Any:
    exporter = str(getattr(config, "otel_exporter", "") or "").strip().lower()
    endpoint = getattr(config, "otel_endpoint", None)
    headers = _parse_header_mapping(getattr(config, "otel_headers", None))

    if exporter == "console":
        return ConsoleSpanExporter()
    if exporter in {"", "otlp"}:
        if endpoint is None:
            return ConsoleSpanExporter()
        return OTLPSpanExporter(endpoint=endpoint, headers=headers)
    raise ValueError(
        "Unsupported SUPPORT_GRAPH_OTEL_EXPORTER "
        f"'{exporter}'. Supported values: console, otlp."
    )


def _ensure_otel_provider(config: Any) -> Any:
    global _OTEL_PROVIDER

    current_provider = trace.get_tracer_provider()
    if not isinstance(current_provider, ProxyTracerProvider):
        return current_provider

    if _OTEL_PROVIDER is None:
        provider = TracerProvider(
            resource=Resource.create(
                {
                    "service.name": getattr(
                        config, "otel_service_name", "support-graph"
                    )
                    or "support-graph",
                }
            )
        )
        provider.add_span_processor(BatchSpanProcessor(_build_otel_exporter(config)))
        trace.set_tracer_provider(provider)
        _OTEL_PROVIDER = provider
    return trace.get_tracer_provider()


def build_observability(config: Any) -> Observability:
    observability = Observability()

    if bool(getattr(config, "otel_enabled", False)):
        provider = _ensure_otel_provider(config)
        observability.tracer = trace.get_tracer(
            "support_graph.runtime",
            tracer_provider=provider,
        )
        observability.otel_enabled = True
        observability.otel_service_name = getattr(
            config, "otel_service_name", "support-graph"
        )
        observability.otel_exporter = (
            str(getattr(config, "otel_exporter", "") or "").strip().lower() or "console"
        )

    if bool(getattr(config, "langsmith_tracing_enabled", False)):
        client_kwargs: dict[str, Any] = {}
        if getattr(config, "langsmith_api_key", None):
            client_kwargs["api_key"] = getattr(config, "langsmith_api_key")
        if getattr(config, "langsmith_endpoint", None):
            client_kwargs["api_url"] = getattr(config, "langsmith_endpoint")
        observability.langsmith_enabled = True
        observability.langsmith_project = (
            getattr(config, "langsmith_project", None) or "support-graph"
        )
        observability.langsmith_client = Client(**client_kwargs)

    return observability


def graph_run_context(
    observability: Observability | None,
    *,
    project_name: str | None = None,
    tags: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> Any:
    if observability is None or not observability.langsmith_enabled:
        return nullcontext()
    return tracing_context(
        project_name=project_name or observability.langsmith_project,
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
    "build_observability",
    "graph_run_context",
    "span_context",
]

"""Optional hosted observability helpers."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any, Protocol

from langsmith import Client
from langsmith.run_helpers import tracing_context


class ObservabilityConfigLike(Protocol):
    langsmith_tracing_enabled: bool
    langsmith_project: str | None
    langsmith_api_key: str | None
    langsmith_endpoint: str | None


class Observability:
    langsmith_project: str | None = None
    langsmith_client: Client | None = None

    def summary(self) -> dict[str, Any]:
        summary: dict[str, Any] = {}
        if self.langsmith_client is not None:
            summary["langsmith"] = {
                "enabled": True,
                "project": self.langsmith_project or "support-graph",
            }
        return summary


def build_observability(config: ObservabilityConfigLike) -> Observability | None:
    if not config.langsmith_tracing_enabled:
        return None

    observability = Observability()

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


__all__ = [
    "Observability",
    "ObservabilityConfigLike",
    "build_observability",
    "graph_run_context",
]

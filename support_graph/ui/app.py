"""FastAPI app for the SupportGraph Workbench."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import cast

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from support_graph.config.settings import Settings
from support_graph.ui.dependencies import init_dependencies
from support_graph.ui.loaders import (
    ArtifactNotFoundError,
    InvalidArtifactError,
    WorkbenchArtifactLoader,
    build_loader,
)
from support_graph.ui.routers import api, pages


TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"


def create_app(loader: WorkbenchArtifactLoader | None = None) -> FastAPI:
    artifact_loader = loader or build_loader()
    ui_settings = cast(Settings, artifact_loader.settings)
    templates = _build_templates()
    app = FastAPI(title="SupportGraph Workbench")

    init_dependencies(artifact_loader, templates, ui_settings)

    @app.exception_handler(ArtifactNotFoundError)
    async def _handle_not_found(
        request: Request, exc: ArtifactNotFoundError
    ) -> HTMLResponse | JSONResponse:
        return _error_response(
            request,
            templates,
            title="Artifact Not Found",
            detail=str(exc),
            status_code=404,
        )

    @app.exception_handler(InvalidArtifactError)
    async def _handle_invalid_artifact(
        request: Request, exc: InvalidArtifactError
    ) -> HTMLResponse | JSONResponse:
        return _error_response(
            request,
            templates,
            title="Artifact Invalid",
            detail=str(exc),
            status_code=409,
        )

    app.include_router(api.router)
    app.include_router(pages.router)

    return app


def _build_templates() -> Jinja2Templates:
    templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
    templates.env.filters.update(
        {
            "datetime_label": _datetime_label,
            "metric": _metric_label,
            "percent": _percent_label,
            "slug_label": _slug_label,
            "pretty_json": _pretty_json,
            "decision_tone": _decision_tone,
            "failure_tone": _failure_tone,
            "evidence_tone": _evidence_tone,
            "confidence_tone": _confidence_tone,
        }
    )
    return templates


def _error_response(
    request: Request,
    templates: Jinja2Templates,
    *,
    title: str,
    detail: str,
    status_code: int,
) -> HTMLResponse | JSONResponse:
    if _is_api_request(request):
        return JSONResponse(status_code=status_code, content={"detail": detail})
    template_name = (
        "partials/error_alert.html" if _is_htmx(request) else "pages/error.html"
    )
    return _template_response(
        templates,
        request,
        template_name,
        _page_context(
            title=title,
            active_nav="artifacts",
            error_title=title,
            detail=detail,
        ),
        status_code=status_code,
    )


def _is_api_request(request: Request) -> bool:
    return request.url.path.startswith("/api/")


def _is_htmx(request: Request) -> bool:
    return request.headers.get("HX-Request") == "true"


def _template_response(
    templates: Jinja2Templates,
    request: Request,
    name: str,
    context: dict[str, object],
    *,
    status_code: int = 200,
) -> HTMLResponse:
    return templates.TemplateResponse(
        request=request,
        name=name,
        context={"request": request, **context},
        status_code=status_code,
    )


def _page_context(
    *,
    title: str,
    active_nav: str,
    **extra: object,
) -> dict[str, object]:
    return {
        "page_title": title,
        "active_nav": active_nav,
        **extra,
    }


def _datetime_label(value: datetime | str | None) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value)
        except ValueError:
            return value
    return value.astimezone().strftime("%Y-%m-%d %H:%M")


def _metric_label(value: float | None, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def _percent_label(value: float | None, digits: int = 0) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100:.{digits}f}%"


def _slug_label(value: object) -> str:
    if value is None:
        return "n/a"
    return str(value).replace("_", " ").strip().title()


def _pretty_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True)


def _decision_tone(value: object) -> str:
    return {
        "answer": "status-answer",
        "clarify": "status-clarify",
        "abstain": "status-abstain",
    }.get(str(value), "status-neutral")


def _failure_tone(value: object) -> str:
    return {
        "runtime_error": "status-runtime-error",
        "wrong_doc": "status-failure",
        "unsupported_answer": "status-failure",
        "weak_citations": "status-warning",
    }.get(str(value), "status-warning")


def _evidence_tone(value: object) -> str:
    return {
        "sufficient": "status-answer",
        "partial": "status-warning",
        "insufficient": "status-abstain",
    }.get(str(value), "status-neutral")


def _confidence_tone(value: object) -> str:
    return {
        "high": "status-answer",
        "medium": "status-warning",
        "low": "status-abstain",
    }.get(str(value), "status-neutral")


__all__ = ["create_app"]

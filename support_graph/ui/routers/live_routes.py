"""Live run routes for the SupportGraph Workbench."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates

from support_graph.config.settings import Settings
from support_graph.ui.dependencies import get_settings, get_templates
from support_graph.ui.live import (
    LiveRunConfigurationError,
    LiveRunExampleNotFoundError,
    iter_live_run_stream,
    prepare_live_run_session,
)


router = APIRouter(prefix="/live")


@router.get("", response_class=HTMLResponse)
def live_page(
    request: Request,
    templates: Jinja2Templates = Depends(get_templates),
) -> HTMLResponse:
    return _template_response(
        templates,
        request,
        "pages/live.html",
        _page_context(
            title="Live Run",
            active_nav="live",
        ),
    )


@router.get("/session", response_class=HTMLResponse)
def live_session(
    request: Request,
    example_id: str,
    templates: Jinja2Templates = Depends(get_templates),
    settings: Settings = Depends(get_settings),
) -> HTMLResponse:
    cleaned_example_id = example_id.strip()
    if not cleaned_example_id:
        return _template_response(
            templates,
            request,
            "partials/live_error.html",
            {"detail": "Enter an example id to start a live run."},
            status_code=400,
        )
    try:
        session = prepare_live_run_session(
            settings,
            example_id=cleaned_example_id,
        )
    except LiveRunConfigurationError as exc:
        return _template_response(
            templates,
            request,
            "partials/live_error.html",
            {"detail": str(exc)},
            status_code=400,
        )
    except LiveRunExampleNotFoundError as exc:
        return _template_response(
            templates,
            request,
            "partials/live_error.html",
            {"detail": str(exc)},
            status_code=404,
        )
    return _template_response(
        templates,
        request,
        "partials/live_session.html",
        {"session": session},
    )


@router.get("/stream")
async def live_stream(
    example_id: str,
    run_id: str,
    settings: Settings = Depends(get_settings),
) -> StreamingResponse:
    headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    }
    return StreamingResponse(
        iter_live_run_stream(
            settings,
            example_id=example_id,
            run_id=run_id,
        ),
        headers=headers,
        media_type="text/event-stream",
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

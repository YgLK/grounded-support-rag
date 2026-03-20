"""FastAPI app for the SupportGraph Workbench."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import cast

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.templating import Jinja2Templates

from support_graph.ui.live import (
    LiveRunConfigurationError,
    LiveRunExampleNotFoundError,
    LiveRunSettings,
    iter_live_run_stream,
    prepare_live_run_session,
)
from support_graph.ui.loaders import (
    ArtifactNotFoundError,
    InvalidArtifactError,
    WorkbenchArtifactLoader,
    build_loader,
)
from support_graph.ui.models import (
    ArtifactExplorerView,
    EvalFailureTableView,
    EvalReportDetailView,
    EvalReportListView,
    EvalRunDetailView,
    EvalRunListView,
    EvalRunSummary,
    ExampleDetailView,
    FailureLabel,
    SortOrder,
    StandaloneRunDetailView,
    StandaloneRunListView,
    TargetMode,
)


TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"
TARGET_MODE_OPTIONS: tuple[TargetMode, ...] = ("answer", "follow_up")


def create_app(loader: WorkbenchArtifactLoader | None = None) -> FastAPI:
    artifact_loader = loader or build_loader()
    ui_settings = cast(LiveRunSettings, artifact_loader.settings)
    templates = _build_templates()
    app = FastAPI(title="SupportGraph Workbench")

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

    @app.get("/api/artifacts", response_model=ArtifactExplorerView)
    def artifact_explorer(
        domain: str | None = None,
        subset: str | None = None,
        provider: str | None = None,
        order: SortOrder = "desc",
    ) -> ArtifactExplorerView:
        return artifact_loader.artifact_explorer(
            domain=domain,
            subset=subset,
            provider=provider,
            order=order,
        )

    @app.get("/api/evals", response_model=EvalRunListView)
    def list_eval_runs(
        domain: str | None = None,
        subset: str | None = None,
        provider: str | None = None,
        order: SortOrder = "desc",
    ) -> EvalRunListView:
        return artifact_loader.list_eval_runs(
            domain=domain,
            subset=subset,
            provider=provider,
            order=order,
        )

    @app.get("/api/evals/{run_id}", response_model=EvalRunDetailView)
    def eval_run_detail(run_id: str) -> EvalRunDetailView:
        return artifact_loader.load_eval_run(run_id)

    @app.get("/api/evals/{run_id}/failures", response_model=EvalFailureTableView)
    def eval_failures(
        run_id: str,
        label: FailureLabel | None = None,
        target_mode: TargetMode | None = None,
    ) -> EvalFailureTableView:
        return artifact_loader.load_eval_failures(
            run_id,
            failure_label=label,
            target_mode=target_mode,
        )

    @app.get(
        "/api/evals/{run_id}/examples/{example_id}",
        response_model=ExampleDetailView,
    )
    def eval_example_detail(run_id: str, example_id: str) -> ExampleDetailView:
        return artifact_loader.load_eval_example(run_id, example_id)

    @app.get("/api/runs", response_model=StandaloneRunListView)
    def list_standalone_runs(order: SortOrder = "desc") -> StandaloneRunListView:
        return artifact_loader.list_standalone_runs(order=order)

    @app.get("/api/runs/{run_id}", response_model=StandaloneRunDetailView)
    def standalone_run_detail(run_id: str) -> StandaloneRunDetailView:
        return artifact_loader.load_standalone_run(run_id)

    @app.get("/api/reports", response_model=EvalReportListView)
    def list_reports(order: SortOrder = "desc") -> EvalReportListView:
        return artifact_loader.list_eval_reports(order=order)

    @app.get("/api/reports/{report_id}", response_model=EvalReportDetailView)
    def report_detail(report_id: str) -> EvalReportDetailView:
        return artifact_loader.load_eval_report(report_id)

    @app.get("/", response_class=HTMLResponse)
    def workbench_home(
        request: Request,
        domain: str | None = None,
        subset: str | None = None,
        provider: str | None = None,
        order: SortOrder = "desc",
    ) -> HTMLResponse:
        explorer = artifact_loader.artifact_explorer(
            domain=domain,
            subset=subset,
            provider=provider,
            order=order,
        )
        if _is_htmx(request):
            return _template_response(
                templates,
                request,
                "partials/artifact_sections.html",
                {"explorer": explorer},
            )
        filter_options = _artifact_filter_options(
            artifact_loader.list_eval_runs(order="desc").items
        )
        return _template_response(
            templates,
            request,
            "pages/home.html",
            _page_context(
                title="Artifact Explorer",
                active_nav="artifacts",
                explorer=explorer,
                filter_options=filter_options,
                current_filters={
                    "domain": domain or "",
                    "subset": subset or "",
                    "provider": provider or "",
                    "order": order,
                },
            ),
        )

    @app.get("/evals/{run_id}", response_class=HTMLResponse)
    def eval_run_page(request: Request, run_id: str) -> HTMLResponse:
        detail = artifact_loader.load_eval_run(run_id)
        example_index = {
            entry.example_id: entry for entry in detail.trace_index_entries
        }
        failure_rows = [
            {
                "record": record,
                "trace_entry": example_index.get(record.example_id),
            }
            for record in detail.failures[:8]
        ]
        return _template_response(
            templates,
            request,
            "pages/eval_detail.html",
            _page_context(
                title=f"Eval {run_id}",
                active_nav="artifacts",
                detail=detail,
                failure_rows=failure_rows,
                failure_buckets=_sorted_failure_counts(detail.metrics.failure_counts),
            ),
        )

    @app.get("/evals/{run_id}/failures", response_class=HTMLResponse)
    def eval_failures_page(
        request: Request,
        run_id: str,
        label: FailureLabel | None = None,
        target_mode: TargetMode | None = None,
    ) -> HTMLResponse:
        table = artifact_loader.load_eval_failures(
            run_id,
            failure_label=label,
            target_mode=target_mode,
        )
        if _is_htmx(request):
            return _template_response(
                templates,
                request,
                "partials/failure_table.html",
                {
                    "table": table,
                    "run_id": run_id,
                },
            )
        labels = _failure_label_options(table)
        return _template_response(
            templates,
            request,
            "pages/eval_failures.html",
            _page_context(
                title=f"Failures {run_id}",
                active_nav="artifacts",
                table=table,
                run_id=run_id,
                available_failure_labels=labels,
                target_mode_options=TARGET_MODE_OPTIONS,
            ),
        )

    @app.get(
        "/evals/{run_id}/examples/{example_id}",
        response_class=HTMLResponse,
    )
    def eval_example_page(
        request: Request,
        run_id: str,
        example_id: str,
    ) -> HTMLResponse:
        detail = artifact_loader.load_eval_example(run_id, example_id)
        return _template_response(
            templates,
            request,
            "pages/example_detail.html",
            _page_context(
                title=f"Example {example_id}",
                active_nav="artifacts",
                detail=detail,
            ),
        )

    @app.get("/runs/{run_id}", response_class=HTMLResponse)
    def standalone_run_page(request: Request, run_id: str) -> HTMLResponse:
        detail = artifact_loader.load_standalone_run(run_id)
        return _template_response(
            templates,
            request,
            "pages/run_detail.html",
            _page_context(
                title=f"Run {run_id}",
                active_nav="artifacts",
                detail=detail,
            ),
        )

    @app.get("/reports/{report_id}", response_class=HTMLResponse)
    def report_page(request: Request, report_id: str) -> HTMLResponse:
        detail = artifact_loader.load_eval_report(report_id)
        return _template_response(
            templates,
            request,
            "pages/report_detail.html",
            _page_context(
                title=detail.summary.title,
                active_nav="artifacts",
                detail=detail,
            ),
        )

    @app.get("/live", response_class=HTMLResponse)
    def live_page(request: Request) -> HTMLResponse:
        return _template_response(
            templates,
            request,
            "pages/live.html",
            _page_context(
                title="Live Run",
                active_nav="live",
            ),
        )

    @app.get("/live/session", response_class=HTMLResponse)
    def live_session(request: Request, example_id: str) -> HTMLResponse:
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
                ui_settings,
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

    @app.get("/live/stream")
    async def live_stream(example_id: str, run_id: str) -> StreamingResponse:
        headers = {
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
        return StreamingResponse(
            iter_live_run_stream(
                ui_settings,
                example_id=example_id,
                run_id=run_id,
            ),
            headers=headers,
            media_type="text/event-stream",
        )

    return app


def _artifact_filter_options(items: list[EvalRunSummary]) -> dict[str, list[str]]:
    return {
        "domains": sorted({domain for item in items for domain in item.domains}),
        "subsets": sorted({item.subset_label for item in items}),
        "providers": sorted({item.provider.type for item in items}),
    }


def _failure_label_options(table: EvalFailureTableView) -> list[str]:
    labels = {str(label) for label in table.failure_counts}
    labels.update(str(item.failure_label) for item in table.items)
    return sorted(labels)


def _sorted_failure_counts(failure_counts: dict[str, int]) -> list[tuple[str, int]]:
    return sorted(
        failure_counts.items(),
        key=lambda item: (-item[1], item[0]),
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

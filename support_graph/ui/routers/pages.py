"""Page routes for the SupportGraph Workbench."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from support_graph.ui.dependencies import get_loader, get_templates
from support_graph.ui.loaders import WorkbenchArtifactLoader
from support_graph.ui.models import (
    EvalFailureTableView,
    EvalRunSummary,
    FailureLabel,
    SortOrder,
    TargetMode,
)


TARGET_MODE_OPTIONS: tuple[TargetMode, ...] = ("answer", "follow_up")

router = APIRouter()


@router.get("/", response_class=HTMLResponse)
def workbench_home(
    request: Request,
    domain: str | None = None,
    subset: str | None = None,
    provider: str | None = None,
    chat_model: str | None = None,
    order: SortOrder = "desc",
    loader: WorkbenchArtifactLoader = Depends(get_loader),
    templates: Jinja2Templates = Depends(get_templates),
) -> HTMLResponse:
    explorer = loader.artifact_explorer(
        domain=domain,
        subset=subset,
        provider=provider,
        chat_model=chat_model,
        order=order,
    )
    if _is_htmx(request):
        return _template_response(
            templates,
            request,
            "partials/artifact_sections.html",
            {"explorer": explorer},
        )
    filter_options = _artifact_filter_options(loader.list_eval_runs(order="desc").items)
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
                "chat_model": chat_model or "",
                "order": order,
            },
        ),
    )


@router.get("/evals/{run_id}", response_class=HTMLResponse)
def eval_run_page(
    request: Request,
    run_id: str,
    loader: WorkbenchArtifactLoader = Depends(get_loader),
    templates: Jinja2Templates = Depends(get_templates),
) -> HTMLResponse:
    detail = loader.load_eval_run(run_id)
    example_index = {entry.example_id: entry for entry in detail.trace_index_entries}
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


@router.get("/evals/{run_id}/failures", response_class=HTMLResponse)
def eval_failures_page(
    request: Request,
    run_id: str,
    label: FailureLabel | None = None,
    target_mode: TargetMode | None = None,
    loader: WorkbenchArtifactLoader = Depends(get_loader),
    templates: Jinja2Templates = Depends(get_templates),
) -> HTMLResponse:
    table = loader.load_eval_failures(
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


@router.get(
    "/evals/{run_id}/examples/{example_id}",
    response_class=HTMLResponse,
)
def eval_example_page(
    request: Request,
    run_id: str,
    example_id: str,
    loader: WorkbenchArtifactLoader = Depends(get_loader),
    templates: Jinja2Templates = Depends(get_templates),
) -> HTMLResponse:
    detail = loader.load_eval_example(run_id, example_id)
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


@router.get("/runs/{run_id}", response_class=HTMLResponse)
def standalone_run_page(
    request: Request,
    run_id: str,
    loader: WorkbenchArtifactLoader = Depends(get_loader),
    templates: Jinja2Templates = Depends(get_templates),
) -> HTMLResponse:
    detail = loader.load_standalone_run(run_id)
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


@router.get("/reports/{report_id}", response_class=HTMLResponse)
def report_page(
    request: Request,
    report_id: str,
    loader: WorkbenchArtifactLoader = Depends(get_loader),
    templates: Jinja2Templates = Depends(get_templates),
) -> HTMLResponse:
    detail = loader.load_eval_report(report_id)
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


def _artifact_filter_options(items: list[EvalRunSummary]) -> dict[str, list[str]]:
    return {
        "domains": sorted({domain for item in items for domain in item.domains}),
        "subsets": sorted({item.subset_label for item in items}),
        "providers": sorted({item.provider.type for item in items}),
        "chat_models": sorted(
            {
                item.provider.chat_model
                for item in items
                if item.provider.chat_model is not None
            }
        ),
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


def _is_htmx(request: Request) -> bool:
    return request.headers.get("HX-Request") == "true"

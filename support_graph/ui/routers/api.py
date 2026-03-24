"""API routes for the SupportGraph Workbench."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from support_graph.ui.dependencies import get_loader
from support_graph.ui.loaders import WorkbenchArtifactLoader
from support_graph.ui.models import (
    ArtifactExplorerView,
    EvalFailureTableView,
    EvalReportDetailView,
    EvalReportListView,
    EvalRunDetailView,
    EvalRunListView,
    ExampleDetailView,
    FailureLabel,
    SortOrder,
    StandaloneRunDetailView,
    StandaloneRunListView,
    TargetMode,
)


router = APIRouter(prefix="/api")


@router.get("/artifacts", response_model=ArtifactExplorerView)
def artifact_explorer(
    domain: str | None = None,
    subset: str | None = None,
    provider: str | None = None,
    chat_model: str | None = None,
    order: SortOrder = "desc",
    loader: WorkbenchArtifactLoader = Depends(get_loader),
) -> ArtifactExplorerView:
    return loader.artifact_explorer(
        domain=domain,
        subset=subset,
        provider=provider,
        chat_model=chat_model,
        order=order,
    )


@router.get("/evals", response_model=EvalRunListView)
def list_eval_runs(
    domain: str | None = None,
    subset: str | None = None,
    provider: str | None = None,
    chat_model: str | None = None,
    order: SortOrder = "desc",
    loader: WorkbenchArtifactLoader = Depends(get_loader),
) -> EvalRunListView:
    return loader.list_eval_runs(
        domain=domain,
        subset=subset,
        provider=provider,
        chat_model=chat_model,
        order=order,
    )


@router.get("/evals/{run_id}", response_model=EvalRunDetailView)
def eval_run_detail(
    run_id: str,
    loader: WorkbenchArtifactLoader = Depends(get_loader),
) -> EvalRunDetailView:
    return loader.load_eval_run(run_id)


@router.get("/evals/{run_id}/failures", response_model=EvalFailureTableView)
def eval_failures(
    run_id: str,
    label: FailureLabel | None = None,
    target_mode: TargetMode | None = None,
    loader: WorkbenchArtifactLoader = Depends(get_loader),
) -> EvalFailureTableView:
    return loader.load_eval_failures(
        run_id,
        failure_label=label,
        target_mode=target_mode,
    )


@router.get(
    "/evals/{run_id}/examples/{example_id}",
    response_model=ExampleDetailView,
)
def eval_example_detail(
    run_id: str,
    example_id: str,
    loader: WorkbenchArtifactLoader = Depends(get_loader),
) -> ExampleDetailView:
    return loader.load_eval_example(run_id, example_id)


@router.get("/runs", response_model=StandaloneRunListView)
def list_standalone_runs(
    order: SortOrder = "desc",
    loader: WorkbenchArtifactLoader = Depends(get_loader),
) -> StandaloneRunListView:
    return loader.list_standalone_runs(order=order)


@router.get("/runs/{run_id}", response_model=StandaloneRunDetailView)
def standalone_run_detail(
    run_id: str,
    loader: WorkbenchArtifactLoader = Depends(get_loader),
) -> StandaloneRunDetailView:
    return loader.load_standalone_run(run_id)


@router.get("/reports", response_model=EvalReportListView)
def list_reports(
    order: SortOrder = "desc",
    loader: WorkbenchArtifactLoader = Depends(get_loader),
) -> EvalReportListView:
    return loader.list_eval_reports(order=order)


@router.get("/reports/{report_id}", response_model=EvalReportDetailView)
def report_detail(
    report_id: str,
    loader: WorkbenchArtifactLoader = Depends(get_loader),
) -> EvalReportDetailView:
    return loader.load_eval_report(report_id)

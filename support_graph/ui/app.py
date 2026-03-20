"""FastAPI app for the SupportGraph Workbench backend."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.responses import JSONResponse

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
    ExampleDetailView,
    FailureLabel,
    SortOrder,
    StandaloneRunDetailView,
    StandaloneRunListView,
    TargetMode,
)


def create_app(loader: WorkbenchArtifactLoader | None = None) -> FastAPI:
    artifact_loader = loader or build_loader()
    app = FastAPI(title="SupportGraph Workbench API")

    @app.exception_handler(ArtifactNotFoundError)
    async def _handle_not_found(_, exc: ArtifactNotFoundError) -> JSONResponse:
        return JSONResponse(status_code=404, content={"detail": str(exc)})

    @app.exception_handler(InvalidArtifactError)
    async def _handle_invalid_artifact(_, exc: InvalidArtifactError) -> JSONResponse:
        return JSONResponse(status_code=409, content={"detail": str(exc)})

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

    return app

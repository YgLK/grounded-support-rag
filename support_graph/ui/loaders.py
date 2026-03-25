"""Filesystem-backed Workbench loaders for the FastAPI backend."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

import markdown as markdown_lib
from pydantic import BaseModel, ValidationError

from support_graph.artifacts import (
    EVAL_REPORT_REQUIRED_FILES,
    EVAL_RUN_REQUIRED_FILES,
    STANDALONE_RUN_REQUIRED_FILES,
    eval_report_artifacts,
    eval_run_artifacts,
    project_relative_path,
    standalone_run_artifacts,
)
from support_graph.config.settings import Settings
from support_graph.runtime.traces import load_trace_events, summarize_trace_events
from support_graph.ui.models import (
    ArtifactExplorerView,
    EvalFailureTableView,
    EvalReportDetailView,
    EvalReportListView,
    EvalReportManifest,
    EvalReportSummary,
    EvalRunDetailView,
    EvalRunListView,
    EvalRunManifest,
    EvalRunMetrics,
    EvalRunSummary,
    ExampleDetailView,
    FailureLabel,
    FailureRecordView,
    PredictionRecord,
    RetrievalExampleRecord,
    SortOrder,
    StandaloneRunDetailView,
    StandaloneRunListView,
    StandaloneRunManifest,
    StandaloneRunResult,
    StandaloneRunSummary,
    TargetMode,
    TraceEventSummary,
    TraceEventView,
    TraceIndexEntry,
)


ModelT = TypeVar("ModelT", bound=BaseModel)


@dataclass(frozen=True, slots=True)
class ArtifactNotFoundError(Exception):
    detail: str

    def __str__(self) -> str:
        return self.detail


@dataclass(frozen=True, slots=True)
class InvalidArtifactError(Exception):
    detail: str

    def __str__(self) -> str:
        return self.detail


class WorkbenchArtifactLoader:
    """Loads and normalizes artifacts for the Workbench UI.

    Responsibilities:
    - Acts as a data access layer between the FastAPI backend and on-disk artifacts
    - Reads manifests, metrics, predictions, and other files from various run types
    - Transforms raw data into structured Pydantic models for API responses
    - Locates and validates artifact directories and required files
    - Encapsulates the on-disk artifact contract logic
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self.paths = settings.paths
        self.project_root = self.paths.project_root

    def artifact_explorer(
        self,
        *,
        domain: str | None = None,
        subset: str | None = None,
        provider: str | None = None,
        chat_model: str | None = None,
        order: SortOrder = "desc",
    ) -> ArtifactExplorerView:
        return ArtifactExplorerView(
            eval_runs=self.list_eval_runs(
                domain=domain,
                subset=subset,
                provider=provider,
                chat_model=chat_model,
                order=order,
            ).items,
            standalone_runs=self.list_standalone_runs(order=order).items,
            reports=self.list_eval_reports(order=order).items,
        )

    def list_eval_runs(
        self,
        *,
        domain: str | None = None,
        subset: str | None = None,
        provider: str | None = None,
        chat_model: str | None = None,
        order: SortOrder = "desc",
    ) -> EvalRunListView:
        items: list[EvalRunSummary] = []
        for run_dir in self._iter_dirs(self.paths.eval_runs_dir):
            try:
                manifest = self._load_eval_manifest(run_dir)
                metrics = self._load_model(run_dir / "metrics.json", EvalRunMetrics)
            except InvalidArtifactError:
                continue
            summary = self._build_eval_run_summary(manifest, metrics)
            if not self._matches_eval_filters(
                summary,
                domain=domain,
                subset=subset,
                provider=provider,
                chat_model=chat_model,
            ):
                continue
            items.append(summary)
        return EvalRunListView(items=self._sort_by_created_at(items, order=order))

    def load_eval_run(self, run_id: str) -> EvalRunDetailView:
        artifacts = eval_run_artifacts(self.project_root, run_id)
        self._require_eval_run_dir(artifacts.output_dir)
        manifest = self._load_model(artifacts.manifest, EvalRunManifest)
        metrics = self._load_model(artifacts.metrics, EvalRunMetrics)
        failure_records = self._load_failure_views(artifacts.failures)
        trace_index_entries = self._load_trace_index_entries(artifacts.trace_index)
        summary_markdown = self._read_text(artifacts.summary)
        return EvalRunDetailView(
            summary=self._build_eval_run_summary(manifest, metrics),
            manifest=manifest,
            metrics=metrics,
            summary_markdown=summary_markdown,
            summary_html=self._render_markdown(summary_markdown),
            failures=failure_records,
            trace_index_entries=trace_index_entries,
            artifact_paths=self._eval_run_paths(run_id),
        )

    def load_eval_failures(
        self,
        run_id: str,
        *,
        failure_label: FailureLabel | None = None,
        target_mode: TargetMode | None = None,
    ) -> EvalFailureTableView:
        detail = self.load_eval_run(run_id)
        items = detail.failures
        if failure_label is not None:
            items = [
                record for record in items if record.failure_label == failure_label
            ]
        if target_mode is not None:
            items = [record for record in items if record.target_mode == target_mode]
        return EvalFailureTableView(
            run=detail.summary,
            items=items,
            total_failures=len(detail.failures),
            failure_counts=detail.metrics.failure_counts,
            applied_failure_label=failure_label,
            applied_target_mode=target_mode,
        )

    def load_eval_example(self, run_id: str, example_id: str) -> ExampleDetailView:
        artifacts = eval_run_artifacts(self.project_root, run_id)
        self._require_eval_run_dir(artifacts.output_dir)
        manifest = self._load_model(artifacts.manifest, EvalRunManifest)
        metrics = self._load_model(artifacts.metrics, EvalRunMetrics)
        prediction = self._load_prediction(artifacts.predictions, example_id)
        retrieval_example = self._load_retrieval_example(
            artifacts.retrieval_examples,
            example_id,
        )
        trace_index_entry = self._find_trace_index_entry(
            artifacts.trace_index, example_id
        )
        trace_path = artifacts.trace_path(trace_index_entry.trace_file)
        if not trace_path.exists():
            raise InvalidArtifactError(
                f"Trace file not found for {example_id}: {self._relative_path(trace_path)}."
            )
        trace_events = load_trace_events(trace_path)
        trace_summary = self._build_trace_event_summary(
            trace_events,
            trace_path=self._relative_path(trace_path),
        )
        return ExampleDetailView(
            run=self._build_eval_run_summary(manifest, metrics),
            prediction=prediction,
            retrieval_example=retrieval_example,
            trace_index_entry=trace_index_entry,
            trace_summary=trace_summary,
            trace_events=self._build_trace_event_views(trace_events),
            artifact_paths={
                **self._eval_run_paths(run_id),
                "trace": self._relative_path(trace_path),
            },
        )

    def list_standalone_runs(
        self,
        *,
        order: SortOrder = "desc",
    ) -> StandaloneRunListView:
        items: list[StandaloneRunSummary] = []
        for run_dir in self._iter_dirs(self.paths.runs_dir):
            try:
                manifest = self._load_standalone_manifest(run_dir)
                result = self._load_model(run_dir / "result.json", StandaloneRunResult)
            except InvalidArtifactError:
                continue
            items.append(self._build_standalone_run_summary(manifest, result))
        return StandaloneRunListView(items=self._sort_by_created_at(items, order=order))

    def load_standalone_run(self, run_id: str) -> StandaloneRunDetailView:
        artifacts = standalone_run_artifacts(self.project_root, run_id)
        self._require_standalone_run_dir(artifacts.output_dir)
        manifest = self._load_model(artifacts.manifest, StandaloneRunManifest)
        result = self._load_model(artifacts.result, StandaloneRunResult)
        if not artifacts.trace.exists():
            raise InvalidArtifactError(
                f"Trace file not found for standalone run {run_id}: {self._relative_path(artifacts.trace)}."
            )
        trace_events = load_trace_events(artifacts.trace)
        return StandaloneRunDetailView(
            summary=self._build_standalone_run_summary(manifest, result),
            manifest=manifest,
            result=result,
            trace_summary=self._build_trace_event_summary(
                trace_events,
                trace_path=self._relative_path(artifacts.trace),
            ),
            trace_events=self._build_trace_event_views(trace_events),
            artifact_paths=self._standalone_run_paths(run_id),
        )

    def list_eval_reports(
        self,
        *,
        order: SortOrder = "desc",
    ) -> EvalReportListView:
        items: list[EvalReportSummary] = []
        for report_dir in self._iter_dirs(self.paths.eval_reports_dir):
            manifest = self._load_eval_report_manifest(report_dir)
            items.append(self._build_eval_report_summary(manifest))
        return EvalReportListView(items=self._sort_by_created_at(items, order=order))

    def load_eval_report(self, report_id: str) -> EvalReportDetailView:
        artifacts = eval_report_artifacts(self.project_root, report_id)
        self._require_eval_report_dir(artifacts.output_dir)
        manifest = self._load_model(artifacts.manifest, EvalReportManifest)
        report_markdown = self._read_text(artifacts.report)
        return EvalReportDetailView(
            summary=self._build_eval_report_summary(manifest),
            manifest=manifest,
            report_markdown=report_markdown,
            report_html=self._render_markdown(report_markdown),
            artifact_paths=self._eval_report_paths(report_id),
        )

    def _load_eval_manifest(self, run_dir: Path) -> EvalRunManifest:
        self._require_eval_run_dir(run_dir)
        return self._load_model(run_dir / "manifest.json", EvalRunManifest)

    def _load_standalone_manifest(self, run_dir: Path) -> StandaloneRunManifest:
        self._require_standalone_run_dir(run_dir)
        return self._load_model(run_dir / "manifest.json", StandaloneRunManifest)

    def _load_eval_report_manifest(self, report_dir: Path) -> EvalReportManifest:
        self._require_eval_report_dir(report_dir)
        return self._load_model(report_dir / "manifest.json", EvalReportManifest)

    def _load_trace_index_entries(self, path: Path) -> list[TraceIndexEntry]:
        payload = self._load_json(path)
        raw_entries = payload.get("entries")
        if not isinstance(raw_entries, list):
            raise InvalidArtifactError(
                f"Trace index is malformed at {self._relative_path(path)}."
            )
        entries: list[TraceIndexEntry] = []
        for raw_entry in raw_entries:
            entries.append(self._validate_model(raw_entry, TraceIndexEntry, path))
        return entries

    def _find_trace_index_entry(
        self,
        path: Path,
        example_id: str,
    ) -> TraceIndexEntry:
        entries = self._load_trace_index_entries(path)
        for entry in entries:
            if entry.example_id == example_id:
                return entry
        raise ArtifactNotFoundError(
            f"Example {example_id} not found in {self._relative_path(path)}."
        )

    def _load_failure_views(self, path: Path) -> list[FailureRecordView]:
        items = self._load_jsonl_models(path, PredictionRecord)
        return [
            self._to_failure_view(item)
            for item in items
            if item.failure_label is not None
        ]

    def _to_failure_view(self, record: PredictionRecord) -> FailureRecordView:
        failure_label = record.failure_label
        if failure_label is None:
            raise InvalidArtifactError(
                f"Failure record for {record.example_id} is missing failure_label."
            )
        return FailureRecordView(
            example_id=record.example_id,
            target_mode=record.target_mode,
            failure_label=failure_label,
            latest_user_utterance=record.latest_user_utterance,
            target_text=record.target_text,
            final_decision=record.decision,
            response_text=record.response_text,
            gold_doc_ids=record.gold_doc_ids,
            gold_span_ids=record.gold_span_ids,
            doc_recall_at_3=record.metrics.doc_recall_at_3,
            span_recall_at_5=record.metrics.span_recall_at_5,
            mrr_at_5=record.metrics.mrr_at_5,
            citation_coverage=record.metrics.citation_coverage,
            citations_valid=record.metrics.citations_valid,
            end_to_end_success=record.metrics.end_to_end_success,
            trace_path=record.trace_summary.trace_path,
        )

    def _build_eval_run_summary(
        self,
        manifest: EvalRunManifest,
        metrics: EvalRunMetrics,
    ) -> EvalRunSummary:
        return EvalRunSummary(
            run_id=manifest.run_id,
            created_at=manifest.created_at,
            domains=manifest.domains,
            split=manifest.split,
            eval_subset=manifest.eval_subset,
            subset_label=manifest.subset_label,
            provider=manifest.provider,
            headline_retrieval=metrics.retrieval.answer,
            headline_generation=metrics.generation.answer,
            failure_counts=metrics.failure_counts,
            artifact_paths=self._eval_run_paths(manifest.run_id),
        )

    def _build_standalone_run_summary(
        self,
        manifest: StandaloneRunManifest,
        result: StandaloneRunResult,
    ) -> StandaloneRunSummary:
        return StandaloneRunSummary(
            run_id=manifest.run_id,
            created_at=manifest.created_at,
            example_id=manifest.example_id,
            domain=manifest.domain,
            provider=manifest.provider,
            decision=result.decision,
            final_query=result.trace_summary.final_query,
            trace_path=result.trace_summary.trace_path,
            artifact_paths=self._standalone_run_paths(manifest.run_id),
        )

    def _build_eval_report_summary(
        self,
        manifest: EvalReportManifest,
    ) -> EvalReportSummary:
        return EvalReportSummary(
            report_id=manifest.report_id,
            created_at=manifest.created_at,
            report_type=manifest.report_type,
            title=manifest.title,
            related_run_ids=manifest.related_run_ids,
            domain=manifest.domain,
            split=manifest.split,
            subset_label=manifest.subset_label,
            artifact_paths=self._eval_report_paths(manifest.report_id),
        )

    def _build_trace_event_summary(
        self,
        events: list[dict],
        *,
        trace_path: str,
    ) -> TraceEventSummary:
        summary = summarize_trace_events(events, trace_path=trace_path)
        return TraceEventSummary.model_validate(
            {
                **summary,
                "evidence_grade": summary.get("evidence_grade") or None,
                "observability": summary.get("observability") or None,
            }
        )

    def _build_trace_event_views(self, events: list[dict]) -> list[TraceEventView]:
        return [
            TraceEventView(
                timestamp=self._optional_str(event.get("timestamp")),
                node=self._optional_str(event.get("node")),
                payload=dict(event),
            )
            for event in events
        ]

    def _render_markdown(self, text: str) -> str:
        return markdown_lib.markdown(
            text,
            extensions=["extra", "fenced_code", "tables"],
        )

    def _eval_run_paths(self, run_id: str) -> dict[str, str]:
        artifacts = eval_run_artifacts(self.project_root, run_id)
        return {
            "directory": self._relative_path(artifacts.output_dir),
            "manifest": self._relative_path(artifacts.manifest),
            "metrics": self._relative_path(artifacts.metrics),
            "predictions": self._relative_path(artifacts.predictions),
            "failures": self._relative_path(artifacts.failures),
            "manual_review": self._relative_path(artifacts.manual_review),
            "retrieval_examples": self._relative_path(artifacts.retrieval_examples),
            "trace_index": self._relative_path(artifacts.trace_index),
            "summary": self._relative_path(artifacts.summary),
            "traces_dir": self._relative_path(artifacts.traces_dir),
        }

    def _standalone_run_paths(self, run_id: str) -> dict[str, str]:
        artifacts = standalone_run_artifacts(self.project_root, run_id)
        return {
            "directory": self._relative_path(artifacts.output_dir),
            "manifest": self._relative_path(artifacts.manifest),
            "result": self._relative_path(artifacts.result),
            "trace": self._relative_path(artifacts.trace),
        }

    def _eval_report_paths(self, report_id: str) -> dict[str, str]:
        artifacts = eval_report_artifacts(self.project_root, report_id)
        return {
            "directory": self._relative_path(artifacts.output_dir),
            "manifest": self._relative_path(artifacts.manifest),
            "report": self._relative_path(artifacts.report),
        }

    def _matches_eval_filters(
        self,
        item: EvalRunSummary,
        *,
        domain: str | None,
        subset: str | None,
        provider: str | None,
        chat_model: str | None,
    ) -> bool:
        if domain is not None and domain not in item.domains:
            return False
        if provider is not None and item.provider.type != provider:
            return False
        if chat_model is not None and item.provider.chat_model != chat_model:
            return False
        if subset is None:
            return True
        normalized_subset = subset.strip().lower()
        return normalized_subset in {
            item.eval_subset.lower(),
            item.subset_label.lower(),
        }

    def _require_eval_run_dir(self, path: Path) -> None:
        self._require_artifact_dir(
            path,
            EVAL_RUN_REQUIRED_FILES,
            artifact_label=f"Eval run {path.name}",
        )

    def _require_standalone_run_dir(self, path: Path) -> None:
        self._require_artifact_dir(
            path,
            STANDALONE_RUN_REQUIRED_FILES,
            artifact_label=f"Standalone run {path.name}",
        )

    def _require_eval_report_dir(self, path: Path) -> None:
        self._require_artifact_dir(
            path,
            EVAL_REPORT_REQUIRED_FILES,
            artifact_label=f"Eval report {path.name}",
        )

    def _require_artifact_dir(
        self,
        path: Path,
        required_files: tuple[str, ...],
        *,
        artifact_label: str,
    ) -> None:
        if not path.exists():
            raise ArtifactNotFoundError(
                f"{artifact_label} not found at {self._relative_path(path)}."
            )
        if not path.is_dir():
            raise InvalidArtifactError(
                f"{artifact_label} path is not a directory: {self._relative_path(path)}."
            )
        missing = [name for name in required_files if not (path / name).exists()]
        if missing:
            missing_text = ", ".join(sorted(missing))
            raise InvalidArtifactError(
                f"{artifact_label} is incomplete under "
                f"{self._relative_path(path)}; "
                f"missing {missing_text}."
            )

    def _load_model(self, path: Path, model_type: type[ModelT]) -> ModelT:
        payload = self._load_json(path)
        return self._validate_model(payload, model_type, path)

    def _load_jsonl_models(self, path: Path, model_type: type[ModelT]) -> list[ModelT]:
        if not path.exists():
            raise InvalidArtifactError(
                f"Artifact file not found: {self._relative_path(path)}."
            )
        items: list[ModelT] = []
        lines = path.read_text(encoding="utf-8").splitlines()
        for raw_line in lines:
            if not raw_line.strip():
                continue
            try:
                items.append(model_type.model_validate_json(raw_line))
            except ValidationError as exc:
                raise InvalidArtifactError(
                    f"Artifact file is malformed: {self._relative_path(path)}."
                ) from exc
        return items

    def _load_prediction(self, path: Path, example_id: str) -> PredictionRecord:
        for item in self._load_jsonl_models(path, PredictionRecord):
            if item.example_id == example_id:
                return item
        raise ArtifactNotFoundError(
            f"Example {example_id} not found in {self._relative_path(path)}."
        )

    def _load_retrieval_example(
        self,
        path: Path,
        example_id: str,
    ) -> RetrievalExampleRecord | None:
        for item in self._load_jsonl_models(path, RetrievalExampleRecord):
            if item.example_id == example_id:
                return item
        return None

    def _load_json(self, path: Path) -> dict[str, object]:
        if not path.exists():
            raise InvalidArtifactError(
                f"Artifact file not found: {self._relative_path(path)}."
            )
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise InvalidArtifactError(
                f"Artifact file is malformed: {self._relative_path(path)}."
            ) from exc
        if not isinstance(payload, dict):
            raise InvalidArtifactError(
                f"Artifact file must contain a JSON object: {self._relative_path(path)}."
            )
        return payload

    def _validate_model(
        self,
        payload: object,
        model_type: type[ModelT],
        path: Path,
    ) -> ModelT:
        try:
            return model_type.model_validate(payload)
        except ValidationError as exc:
            raise InvalidArtifactError(
                f"Artifact file does not match the expected schema: {self._relative_path(path)}."
            ) from exc

    def _read_text(self, path: Path) -> str:
        if not path.exists():
            raise InvalidArtifactError(
                f"Artifact file not found: {self._relative_path(path)}."
            )
        return path.read_text(encoding="utf-8")

    def _iter_dirs(self, root: Path) -> list[Path]:
        if not root.exists() or not root.is_dir():
            return []
        return [path for path in root.iterdir() if path.is_dir()]

    def _sort_by_created_at(
        self,
        items: list[EvalRunSummary | StandaloneRunSummary | EvalReportSummary],
        *,
        order: SortOrder,
    ) -> list[EvalRunSummary | StandaloneRunSummary | EvalReportSummary]:
        reverse = order == "desc"
        return sorted(
            items,
            key=lambda item: (item.created_at, self._item_identifier(item)),
            reverse=reverse,
        )

    def _item_identifier(
        self,
        item: EvalRunSummary | StandaloneRunSummary | EvalReportSummary,
    ) -> str:
        if isinstance(item, EvalRunSummary):
            return item.run_id
        if isinstance(item, StandaloneRunSummary):
            return item.run_id
        return item.report_id

    def _optional_str(self, value: object) -> str | None:
        if value is None:
            return None
        return str(value)

    def _relative_path(self, path: Path) -> str:
        return project_relative_path(path, self.project_root)


def build_loader(settings: Settings | None = None) -> WorkbenchArtifactLoader:
    resolved_settings = settings or Settings.load()
    return WorkbenchArtifactLoader(resolved_settings)

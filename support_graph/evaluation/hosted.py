"""LangSmith-hosted evaluation target and orchestration."""

from __future__ import annotations

import tempfile
import uuid
from collections import defaultdict
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

from support_graph.config.runtime import RuntimeConfig
from support_graph.evaluation.contracts import (
    ExperimentSnapshot,
    GateResult,
    HostedEvaluation,
    LangSmithGateway,
)
from support_graph.evaluation.datasets import publish_dataset
from support_graph.evaluation.evaluate import (
    RAGTriadJudge,
    failure_label,
    has_rag_eval_fields,
    prediction_metrics,
    prediction_record,
)
from support_graph.evaluation.metadata import GitState, build_experiment_metadata
from support_graph.evaluation.policy import RegressionPolicy, compare_experiments
from support_graph.runtime.graph import resolve_runtime_resources_async, run_graph_async


def build_hosted_target(
    *,
    config: RuntimeConfig,
    run_graph_func: Any,
    runtime_resources: Any,
) -> Any:
    """Build a graph target that retains traces only for the active invocation."""

    async def target(inputs: dict[str, Any]) -> dict[str, Any]:
        example = inputs["example"]
        example_id = str(example["example_id"])
        with tempfile.TemporaryDirectory(prefix="support-graph-trace-") as temp_dir:
            trace_path = Path(temp_dir) / f"{example_id}.jsonl"
            kwargs = {
                "example": example,
                "config": config,
                "run_id": f"langsmith-{example_id}-{uuid.uuid4().hex[:8]}",
                "trace_path": trace_path,
                "max_attempts": config.max_retrieval_attempts,
            }
            if runtime_resources is not None:
                kwargs["_runtime_resources"] = runtime_resources
            prediction = await run_graph_func(**kwargs)
            trace_summary = dict(prediction.get("trace_summary", {}))
            trace_summary.pop("trace_path", None)
            return {**prediction, "trace_summary": trace_summary}

    return target


def feedback_from_record(record: dict[str, Any]) -> list[dict[str, Any]]:
    """Convert a prediction record to LangSmith feedback entries."""
    feedback: list[dict[str, Any]] = []
    for key, value in sorted(record["metrics"].items()):
        if isinstance(value, bool):
            feedback.append({"key": key, "score": float(value)})
        elif isinstance(value, int | float):
            feedback.append({"key": key, "score": float(value)})
    feedback.extend(
        [
            {"key": "decision", "value": str(record.get("decision", ""))},
            {
                "key": "failure_label",
                "value": str(record.get("failure_label") or "none"),
            },
        ]
    )
    return feedback


def build_feedback_evaluator(
    *,
    config: RuntimeConfig,
    policy: RegressionPolicy,
    judge: RAGTriadJudge | Any | None,
) -> Any:
    """Build an evaluator that records scorer failures as feedback errors."""

    async def evaluator(run: Any, example: Any) -> list[dict[str, Any]]:
        try:
            source_example = _source_example(example)
            prediction = _mapping_value(run, "outputs")
            if not isinstance(prediction, dict):
                raise ValueError("LangSmith run has no prediction outputs")
            metrics = prediction_metrics(
                source_example,
                prediction,
                retrieval_top_k=config.retrieval_top_k,
            )
            if (
                judge is not None
                and judge.enabled
                and has_rag_eval_fields(source_example)
            ):
                verdict = await judge.evaluate(
                    query=str(
                        source_example.get("latest_user_utterance")
                        or source_example.get("target_turn", {}).get("utterance", "")
                    ),
                    answer=str(prediction.get("response_text", "")),
                    context=_retrieved_context(prediction),
                    required_points=source_example.get("required_points", []),
                    forbidden_claims=source_example.get("forbidden_claims", []),
                )
                if verdict is not None:
                    metrics = prediction_metrics(
                        source_example,
                        prediction,
                        retrieval_top_k=config.retrieval_top_k,
                        judge_verdict=verdict,
                    )
            record = prediction_record(
                source_example,
                prediction,
                metrics,
                failure_label=failure_label(source_example, prediction, metrics),
            )
            return feedback_from_record(record)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            return [{"key": key, "error": error} for key in policy.required_feedback]

    return evaluator


async def run_hosted_evaluation(
    *,
    gateway: LangSmithGateway,
    examples: list[dict[str, Any]],
    config: RuntimeConfig,
    domain: str,
    subset: str,
    policy: RegressionPolicy,
    corpus_ref: str,
    corpus_manifest_path: Path,
    git_state: GitState,
    max_concurrency: int = 1,
    run_graph_func: Any = run_graph_async,
    judge: RAGTriadJudge | None = None,
    experiment_metadata: dict[str, Any] | None = None,
    baseline: ExperimentSnapshot | None = None,
) -> HostedEvaluation:
    """Execute one LangSmith candidate experiment and classify its result."""
    config.validate_for_hosted_eval()
    if max_concurrency <= 0:
        raise ValueError("max_concurrency must be positive")
    if not policy.metrics:
        raise ValueError("Regression policy must define at least one metric")

    dataset = await publish_dataset(
        gateway,
        domain=domain,
        subset=subset,
        examples=examples,
    )
    runtime_resources = (
        await resolve_runtime_resources_async(config)
        if run_graph_func is run_graph_async
        else None
    )
    target = build_hosted_target(
        config=config,
        run_graph_func=run_graph_func,
        runtime_resources=runtime_resources,
    )
    evaluator = build_feedback_evaluator(config=config, policy=policy, judge=judge)
    metadata = build_experiment_metadata(
        config=config,
        dataset=dataset,
        domain=domain,
        subset=subset,
        corpus_ref=corpus_ref,
        corpus_manifest_path=corpus_manifest_path,
        evaluator_versions={"deterministic": "1", "rag_triad": "1"},
        judge_provider=None,
        judge_model=None,
        git_state=git_state,
        started_at=datetime.now(timezone.utc),
        example_count=len(examples),
    )
    if experiment_metadata:
        metadata.update(experiment_metadata)
    metadata["status"] = "running"

    experiment = await gateway.evaluate(
        target=target,
        dataset_name=dataset.name,
        evaluators=[evaluator],
        metadata=metadata,
        experiment_prefix=f"support-graph-{domain}-{subset}",
        max_concurrency=max_concurrency,
    )
    aggregate = _aggregate_feedback(experiment)
    candidate_reasons = _candidate_reasons(experiment, policy)
    if candidate_reasons:
        gate = GateResult(
            status="invalid",
            baseline_experiment_id=baseline.id if baseline else None,
            candidate_experiment_id=experiment.id,
            reasons=candidate_reasons,
        )
    elif baseline is None:
        gate = GateResult(
            status="invalid",
            baseline_experiment_id=None,
            candidate_experiment_id=experiment.id,
            reasons=("No promoted baseline for dataset",),
        )
    else:
        gate = compare_experiments(experiment, baseline, policy)

    completed_metadata = {
        **metadata,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "example_count": len(experiment.results),
        "status": "invalid" if candidate_reasons else "completed",
    }
    completed_experiment = await gateway.update_experiment_metadata(
        experiment.id,
        completed_metadata,
    )
    return HostedEvaluation(
        experiment=completed_experiment,
        aggregate_metrics=aggregate,
        gate=gate,
    )


def _source_example(example: Any) -> dict[str, Any]:
    inputs = _mapping_value(example, "inputs")
    if not isinstance(inputs, Mapping) or not isinstance(inputs.get("example"), dict):
        raise ValueError("LangSmith example is missing inputs.example")
    return inputs["example"]


def _mapping_value(value: Any, key: str) -> Any:
    if isinstance(value, Mapping):
        return value.get(key)
    return getattr(value, key, None)


def _retrieved_context(prediction: dict[str, Any]) -> str:
    return "\n\n".join(
        str(chunk.get("text", ""))
        for chunk in prediction.get("retrieved_chunks", [])
        if str(chunk.get("text", "")).strip()
    )


def _aggregate_feedback(experiment: ExperimentSnapshot) -> dict[str, float]:
    values: defaultdict[str, list[float]] = defaultdict(list)
    for result in experiment.results:
        for feedback in result.feedback:
            if feedback.score is not None and feedback.error is None:
                values[feedback.key].append(feedback.score)
    return {key: mean(scores) for key, scores in sorted(values.items())}


def _candidate_reasons(
    experiment: ExperimentSnapshot,
    policy: RegressionPolicy,
) -> tuple[str, ...]:
    reasons: list[str] = []
    for result in experiment.results:
        if result.error:
            reasons.append(f"candidate execution error: {result.example_id}")
        feedback = {item.key: item for item in result.feedback}
        for key in policy.required_feedback:
            item = feedback.get(key)
            if item is None:
                reasons.append(
                    f"missing required feedback: {key} for {result.example_id}"
                )
            elif item.error:
                reasons.append(
                    f"feedback error for {result.example_id}: {key}: {item.error}"
                )
    return tuple(reasons)


__all__ = [
    "build_feedback_evaluator",
    "build_hosted_target",
    "feedback_from_record",
    "run_hosted_evaluation",
]

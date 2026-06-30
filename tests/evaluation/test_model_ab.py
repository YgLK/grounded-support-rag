from __future__ import annotations

from support_graph.evaluation.model_ab import (
    fallback_count_for_result,
    runtime_error_count_for_result,
)


def test_fallback_count_for_result_sums_across_predictions() -> None:
    result = {
        "predictions": [
            {"trace_summary": {"fallback_count": 1, "fallback_nodes": ["route_query"]}},
            {
                "trace_summary": {
                    "fallback_count": 2,
                    "fallback_nodes": ["answer", "route_query"],
                }
            },
            {"trace_summary": {}},
        ]
    }
    total, nodes = fallback_count_for_result(result)
    assert total == 3
    assert nodes == ["answer", "route_query"]


def test_fallback_count_for_result_handles_empty_predictions() -> None:
    total, nodes = fallback_count_for_result({})
    assert total == 0
    assert nodes == []


def test_runtime_error_count_detects_runtime_error_dict() -> None:
    result = {
        "predictions": [
            {"runtime_error": {"exception_type": "ValueError", "error": "bad slug"}},
            {"trace_summary": {"fallback_count": 0}},
        ]
    }
    assert runtime_error_count_for_result(result) == 1


def test_runtime_error_count_detects_runtime_error_failure_label() -> None:
    result = {
        "predictions": [
            {"failure_label": "runtime_error", "trace_summary": {"fallback_count": 0}},
            {
                "failure_label": "incomplete_answer",
                "trace_summary": {"fallback_count": 0},
            },
        ]
    }
    assert runtime_error_count_for_result(result) == 1


def test_runtime_error_count_handles_empty_predictions() -> None:
    assert runtime_error_count_for_result({}) == 0


def test_runtime_error_count_counts_both_forms() -> None:
    result = {
        "predictions": [
            {
                "runtime_error": {
                    "exception_type": "ConnectionError",
                    "error": "timeout",
                }
            },
            {"failure_label": "runtime_error"},
            {"failure_label": "weak_citations"},
        ]
    }
    assert runtime_error_count_for_result(result) == 2

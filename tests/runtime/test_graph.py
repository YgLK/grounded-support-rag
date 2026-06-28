from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

from support_graph.runtime import graph
from support_graph.runtime import nodes as runtime_nodes
from support_graph.runtime.traces import load_trace_events


def test_run_graph_emits_ranked_and_expanded_chunk_lists(
    monkeypatch,
    tmp_path: Path,
    runtime_example: dict,
    make_runtime_config,
) -> None:
    call_log: list[str] = []
    title_chunk_id = "kubernetes::doc::sec::t_1::sub::0"
    content_chunk_id = "kubernetes::doc::sec::2::sub::0"

    def fake_route_query(*args, **kwargs):
        call_log.append("route_query")
        return {"intent": "document_query", "reason": "test"}

    def fake_prepare_query(*args, **kwargs):
        call_log.append("prepare_query")
        return "deployment stalled kubernetes"

    def fake_retrieve_docs(*args, **kwargs):
        call_log.append("retrieve_docs")
        return [
            {
                "rank": 1,
                "chunk_id": title_chunk_id,
                "doc_id": "doc",
                "section_id": "t_1",
                "section_title": "What happens if my rollout ends?",
                "span_ids": ["1"],
                "text": "What happens if my rollout ends?",
                "token_count": 7,
                "score": 0.20,
            },
            {
                "rank": 2,
                "chunk_id": content_chunk_id,
                "doc_id": "doc",
                "section_id": "2",
                "section_title": "Deployment rollout guidance",
                "span_ids": ["2", "3"],
                "text": "Restore coverage immediately to avoid a deployment unavailable replicas.",
                "token_count": 12,
                "score": 0.21,
            },
        ]

    def fake_grade_evidence(*args, **kwargs):
        call_log.append("grade_evidence")
        return {
            "verdict": "sufficient",
            "reason": "direct support",
            "missing_information": [],
        }

    def fake_generate_response(*args, **kwargs):
        call_log.append("generate_response")
        return {
            "decision": "answer",
            "response_text": "Restore coverage immediately to avoid a deployment unavailable replicas.",
            "citation_chunk_ids": [content_chunk_id],
            "confidence_label": "high",
        }

    original_finalize = graph.finalize

    def fake_finalize(*args, **kwargs):
        call_log.append("finalize")
        return original_finalize(*args, **kwargs)

    monkeypatch.setattr(graph, "route_query", fake_route_query, raising=False)
    monkeypatch.setattr(graph, "prepare_query", fake_prepare_query, raising=False)
    monkeypatch.setattr(graph, "retrieve_docs", fake_retrieve_docs, raising=False)
    monkeypatch.setattr(graph, "grade_evidence", fake_grade_evidence, raising=False)
    monkeypatch.setattr(
        graph, "generate_response", fake_generate_response, raising=False
    )
    monkeypatch.setattr(graph, "finalize", fake_finalize, raising=False)
    monkeypatch.setattr(
        graph, "_trace", lambda *args, **kwargs: asyncio.sleep(0), raising=False
    )

    result = asyncio.run(
        graph.run_graph_async(
            example=runtime_example,
            config=make_runtime_config(neighbor_expansion=False),
            run_id="run-test",
            trace_path=tmp_path / "run-test.jsonl",
            max_attempts=2,
        )
    )

    assert result["decision"] == "answer"
    assert [chunk["chunk_id"] for chunk in result["retrieval_ranked_chunks"]] == [
        title_chunk_id,
        content_chunk_id,
    ]
    assert [chunk["chunk_id"] for chunk in result["retrieved_chunks"]] == [
        content_chunk_id
    ]
    assert result["citations"][0]["chunk_id"] == content_chunk_id
    assert result["trace_summary"]["graph_path"] == [
        "route_query",
        "prepare_query",
        "retrieve_docs",
        "grade_evidence",
        "generate_response",
        "finalize",
    ]
    assert Path(result["trace_summary"]["trace_path"]).parent == tmp_path
    assert Path(result["trace_summary"]["trace_path"]).suffix == ".jsonl"
    assert call_log == [
        "route_query",
        "prepare_query",
        "retrieve_docs",
        "grade_evidence",
        "generate_response",
        "finalize",
    ]


def test_expand_neighbor_sections_includes_previous_and_next_numeric_sections() -> None:
    anchor = {
        "chunk_id": "kubernetes::doc::sec::5::sub::0",
        "doc_id": "doc",
        "section_id": "5",
        "section_title": "Unavailable replicas",
        "span_ids": ["5"],
        "text": "Section 5",
        "token_count": 20,
    }
    chunk_records_by_doc = {
        "doc": [
            {
                "chunk_id": "kubernetes::doc::sec::4::sub::0",
                "doc_id": "doc",
                "section_id": "4",
                "section_title": "Notice",
                "span_ids": ["4"],
                "text": "Section 4",
                "token_count": 20,
                "subchunk_index": 0,
                "start_sec": 4,
            },
            {
                "chunk_id": "kubernetes::doc::sec::5::sub::0",
                "doc_id": "doc",
                "section_id": "5",
                "section_title": "Unavailable replicas",
                "span_ids": ["5"],
                "text": "Section 5",
                "token_count": 20,
                "subchunk_index": 0,
                "start_sec": 5,
            },
            {
                "chunk_id": "kubernetes::doc::sec::6::sub::0",
                "doc_id": "doc",
                "section_id": "6",
                "section_title": "Reinstatement",
                "span_ids": ["6"],
                "text": "Section 6",
                "token_count": 20,
                "subchunk_index": 0,
                "start_sec": 6,
            },
        ]
    }

    expanded = graph.expand_neighbor_sections([anchor], chunk_records_by_doc)

    assert [chunk["chunk_id"] for chunk in expanded] == [
        "kubernetes::doc::sec::4::sub::0",
        "kubernetes::doc::sec::5::sub::0",
        "kubernetes::doc::sec::6::sub::0",
    ]


def test_expand_neighbor_sections_ignores_title_anchors_dedupes_and_caps_at_eight() -> (
    None
):
    title_anchor = {
        "chunk_id": "kubernetes::doc::sec::t_4::sub::0",
        "doc_id": "doc",
        "section_id": "t_4",
        "section_title": "Question heading",
        "span_ids": ["t4"],
        "text": "Question heading",
        "token_count": 5,
    }
    anchors = [
        {
            "chunk_id": f"kubernetes::doc::sec::{section}::sub::0",
            "doc_id": "doc",
            "section_id": str(section),
            "section_title": f"Section {section}",
            "span_ids": [str(section)],
            "text": f"Section {section}",
            "token_count": 20,
        }
        for section in (3, 5, 7, 9, 11)
    ]
    chunk_records_by_doc = {
        "doc": [
            {
                "chunk_id": f"kubernetes::doc::sec::{section}::sub::0",
                "doc_id": "doc",
                "section_id": str(section),
                "section_title": f"Section {section}",
                "span_ids": [str(section)],
                "text": f"Section {section}",
                "token_count": 20,
                "subchunk_index": 0,
                "start_sec": section,
            }
            for section in range(2, 13)
        ]
    }

    title_only = graph.expand_neighbor_sections([title_anchor], chunk_records_by_doc)
    expanded = graph.expand_neighbor_sections(
        [title_anchor, *anchors], chunk_records_by_doc
    )

    assert [chunk["chunk_id"] for chunk in title_only] == [title_anchor["chunk_id"]]
    assert len(expanded) == 8
    assert len({chunk["chunk_id"] for chunk in expanded}) == 8
    assert expanded[0]["chunk_id"] == "kubernetes::doc::sec::2::sub::0"
    assert expanded[-1]["chunk_id"] == "kubernetes::doc::sec::9::sub::0"


def test_run_graph_retries_once_before_resolving(
    monkeypatch,
    tmp_path: Path,
    runtime_example: dict,
    make_runtime_config,
) -> None:
    call_log: list[str] = []
    grade_outcomes = iter(
        [
            {
                "verdict": "partial",
                "reason": "need one detail",
                "missing_information": ["rollout status"],
            },
            {
                "verdict": "insufficient",
                "reason": "still missing detail",
                "missing_information": ["rollout status"],
            },
        ]
    )

    def fake_prepare_query(*args, **kwargs):
        call_log.append("prepare_query")
        return "deployment stalled kubernetes"

    def fake_retrieve_docs(*args, **kwargs):
        call_log.append("retrieve_docs")
        return [
            {
                "chunk_id": "kubernetes::chunk::1",
                "doc_id": "doc",
                "section_id": "1",
                "span_ids": ["1"],
                "text": "text",
            }
        ]

    def fake_grade_evidence(*args, **kwargs):
        call_log.append("grade_evidence")
        return next(grade_outcomes)

    def fake_refine_query(*args, **kwargs):
        call_log.append("refine_query")
        return "deployment stalled kubernetes\nMissing condition: rollout status"

    def fake_finalize(*args, **kwargs):
        call_log.append("finalize")
        payload = kwargs.get("payload")
        if payload is not None:
            return payload
        if args:
            return args[0]
        return {}

    monkeypatch.setattr(graph, "prepare_query", fake_prepare_query, raising=False)
    monkeypatch.setattr(graph, "retrieve_docs", fake_retrieve_docs, raising=False)
    monkeypatch.setattr(graph, "grade_evidence", fake_grade_evidence, raising=False)
    monkeypatch.setattr(graph, "refine_query", fake_refine_query, raising=False)
    monkeypatch.setattr(graph, "finalize", fake_finalize, raising=False)
    monkeypatch.setattr(
        graph,
        "generate_response",
        lambda *args, **kwargs: {
            "decision": "answer",
            "response_text": "This should not be allowed after retries are exhausted.",
            "citations": [],
            "confidence_label": "high",
        },
        raising=False,
    )
    monkeypatch.setattr(
        graph, "_trace", lambda *args, **kwargs: asyncio.sleep(0), raising=False
    )

    result = asyncio.run(
        graph.run_graph_async(
            example=runtime_example,
            config=make_runtime_config(neighbor_expansion=False),
            run_id="run-retry",
            trace_path=tmp_path / "run-retry.jsonl",
            max_attempts=2,
        )
    )

    assert result["decision"] in {"clarify", "abstain"}
    assert result["trace_summary"]["retrieval_attempts"] == 2
    assert call_log.count("retrieve_docs") == 2
    assert call_log.count("refine_query") == 1
    assert call_log.count("generate_response") == 0
    assert call_log[0] == "prepare_query"
    assert call_log[-1] == "finalize"


def test_run_graph_logs_and_traces_llm_fallbacks(
    monkeypatch,
    tmp_path: Path,
    runtime_example: dict,
    make_runtime_config,
) -> None:
    class BrokenChatModel:
        def with_structured_output(self, *args, **kwargs):
            raise ValueError("structured output unavailable")

    warnings: list[str] = []

    def fake_warning(message, *args, **kwargs):
        formatted = message % args if args else str(message)
        warnings.append(formatted)

    monkeypatch.setattr(runtime_nodes.logger, "warning", fake_warning)
    monkeypatch.setattr(
        graph,
        "retrieve_docs",
        lambda *args, **kwargs: [
            {
                "rank": 1,
                "chunk_id": "kubernetes::doc::sec::2::sub::0",
                "doc_id": "doc",
                "section_id": "2",
                "section_title": "Deployment rollout guidance",
                "span_ids": ["2", "3"],
                "text": "Restore coverage immediately to avoid unavailable replicas.",
                "token_count": 12,
                "score": 0.2,
            }
        ],
        raising=False,
    )

    result = asyncio.run(
        graph.run_graph_async(
            example=runtime_example,
            config=make_runtime_config(chat_model="chat-model"),
            run_id="run-fallback",
            trace_path=tmp_path / "run-fallback.jsonl",
            max_attempts=1,
            chat_model=BrokenChatModel(),
        )
    )

    assert result["decision"] == "answer"
    assert (
        result["response_text"]
        == "Restore coverage immediately to avoid unavailable replicas."
    )
    assert result["citations"][0]["chunk_id"] == "kubernetes::doc::sec::2::sub::0"
    assert result["trace_summary"]["fallback_count"] == 2
    assert result["trace_summary"]["fallback_nodes"] == [
        "grade_evidence",
        "generate_response",
    ]
    assert any(
        "grade_evidence falling back to heuristics" in message for message in warnings
    )
    assert any(
        "generate_response falling back to heuristics" in message
        for message in warnings
    )

    events = load_trace_events(result["trace_summary"]["trace_path"])
    fallback_nodes = [
        event["fallback"]["node"] for event in events if event.get("fallback")
    ]
    assert fallback_nodes == ["grade_evidence", "generate_response"]


def test_astream_graph_events_emits_milestones_and_response_deltas(
    monkeypatch,
    tmp_path: Path,
    runtime_example: dict,
    make_runtime_config,
) -> None:
    content_chunk_id = "kubernetes::doc::sec::2::sub::0"

    monkeypatch.setattr(
        graph,
        "prepare_query",
        lambda *args, **kwargs: "deployment stalled kubernetes",
        raising=False,
    )
    monkeypatch.setattr(
        graph,
        "retrieve_docs",
        lambda *args, **kwargs: [
            {
                "rank": 1,
                "chunk_id": content_chunk_id,
                "doc_id": "doc",
                "section_id": "2",
                "section_title": "Deployment rollout guidance",
                "span_ids": ["2", "3"],
                "text": "Restore coverage immediately to avoid unavailable replicas.",
                "token_count": 12,
                "score": 0.2,
            }
        ],
        raising=False,
    )
    monkeypatch.setattr(
        graph,
        "grade_evidence",
        lambda *args, **kwargs: {
            "verdict": "sufficient",
            "reason": "direct support",
            "missing_information": [],
        },
        raising=False,
    )

    class FakeStreamChatModel:
        async def astream(self, messages):
            flattened = "\n".join(str(message.content) for message in messages)
            assert "Deployment rollout guidance" in flattened
            for delta in ("Restore ", "coverage ", "immediately."):
                yield SimpleNamespace(content=delta)

    async def fake_structured_prompt(*args, **kwargs):
        return {
            "decision": "answer",
            "response_text": "Restore coverage immediately.",
            "citation_chunk_ids": [content_chunk_id],
            "confidence_label": "high",
        }

    monkeypatch.setattr(
        runtime_nodes,
        "_ainvoke_structured_prompt",
        fake_structured_prompt,
        raising=False,
    )
    monkeypatch.setattr(
        graph, "_trace", lambda *args, **kwargs: asyncio.sleep(0), raising=False
    )

    async def collect_events() -> list[dict]:
        return [
            event
            async for event in graph.astream_graph_events(
                example=runtime_example,
                config=make_runtime_config(
                    neighbor_expansion=False,
                    chat_model="chat-model",
                ),
                run_id="run-stream",
                trace_path=tmp_path / "run-stream.jsonl",
                max_attempts=1,
                chat_model=FakeStreamChatModel(),
            )
        ]

    events = asyncio.run(collect_events())
    kinds = [event["kind"] for event in events]

    assert kinds[:4] == [
        "query_ready",
        "retrieval_complete",
        "evidence_graded",
        "response_started",
    ]
    assert "response_completed" in kinds
    deltas = "".join(
        event["delta"] for event in events if event["kind"] == "response_delta"
    )
    assert deltas == "Restore coverage immediately."
    completed = next(event for event in events if event["kind"] == "response_completed")
    assert completed["decision"] == "answer"
    assert completed["citations"][0]["chunk_id"] == content_chunk_id
    assert completed["trace_summary"]["graph_path"] == [
        "route_query",
        "prepare_query",
        "retrieve_docs",
        "grade_evidence",
        "generate_response",
        "finalize",
    ]

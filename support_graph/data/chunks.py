"""Section-aware retrieval chunk builder for MultiDoc2Dial."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from support_graph.data._utils import normalize_domains


def _approximate_token_count(text: str) -> int:
    """Approximate model tokens with a whitespace word count.

    The chunker uses this only as a deterministic size heuristic. It is not a
    tokenizer-aware model context measurement.
    """

    return len(text.split())


def _normalize_parent_titles(parent_titles: Iterable[str]) -> list[str]:
    return [title for title in parent_titles if title]


def _join_span_text(spans: list[dict]) -> str:
    return "".join(span.get("text_sp", "") for span in spans).strip()


def _section_sort_key(section: dict) -> tuple:
    return (
        section.get("start_sec") if section.get("start_sec") is not None else 10**18,
        section.get("start_sp") if section.get("start_sp") is not None else 10**18,
        section.get("section_id", ""),
    )


def _group_sections(document: dict) -> list[dict]:
    section_map: dict[str, list[dict]] = defaultdict(list)
    heading_context_by_title: dict[str, list[str]] = {}
    for span in document.get("spans", []) or []:
        section_id = str(span.get("id_sec") or "doc")
        section_map[section_id].append(span)
        tag = str(span.get("tag", ""))
        title = str(span.get("title", "")).strip()
        if tag.startswith("h") and title:
            heading_context_by_title[title] = _normalize_parent_titles(
                span.get("parent_titles", [])
            )

    sections: list[dict] = []
    for section_id, spans in section_map.items():
        ordered_spans = sorted(
            spans,
            key=lambda span: (
                span.get("start_sp") if span.get("start_sp") is not None else 10**18,
                span.get("end_sp") if span.get("end_sp") is not None else 10**18,
                span.get("id_sp", ""),
            ),
        )
        first_span = ordered_spans[0] if ordered_spans else {}
        section_text = (
            _join_span_text(ordered_spans)
            if ordered_spans
            else document.get("doc_text", "")
        )
        section_title = first_span.get("title", "") or document.get("title", "")
        parent_titles = []
        for span in ordered_spans:
            candidate = _normalize_parent_titles(span.get("parent_titles", []))
            if candidate:
                parent_titles = candidate
                break
        if not parent_titles and section_title:
            parent_titles = heading_context_by_title.get(section_title, [])
        sections.append(
            {
                "section_id": section_id,
                "section_title": section_title,
                "parent_titles": parent_titles,
                "spans": ordered_spans,
                "text": section_text,
                "start_sec": first_span.get("start_sec"),
                "end_sec": ordered_spans[-1].get("end_sec") if ordered_spans else None,
            }
        )

    return sorted(sections, key=_section_sort_key)


def _emit_section_chunks(
    document: dict, section: dict, max_tokens_per_chunk: int
) -> list[dict]:
    spans = section["spans"]
    if not spans:
        text = section.get("text") or document.get("doc_text", "")
        return [
            {
                "chunk_id": f"{document['domain']}::{document['doc_id']}::sec::{section['section_id']}::sub::0",
                "domain": document["domain"],
                "doc_id": document["doc_id"],
                "doc_title": document.get("title", ""),
                "section_id": section["section_id"],
                "section_title": section.get("section_title", ""),
                "parent_titles": section.get("parent_titles", []),
                "subchunk_index": 0,
                "text": text.strip(),
                "span_ids": [],
                "token_count": _approximate_token_count(text),
                "start_sec": section.get("start_sec"),
                "end_sec": section.get("end_sec"),
            }
        ]

    section_text = section.get("text", "")
    section_token_count = _approximate_token_count(section_text)
    if section_token_count <= max_tokens_per_chunk:
        return [
            {
                "chunk_id": f"{document['domain']}::{document['doc_id']}::sec::{section['section_id']}::sub::0",
                "domain": document["domain"],
                "doc_id": document["doc_id"],
                "doc_title": document.get("title", ""),
                "section_id": section["section_id"],
                "section_title": section.get("section_title", ""),
                "parent_titles": section.get("parent_titles", []),
                "subchunk_index": 0,
                "text": section_text.strip(),
                "span_ids": [
                    span.get("id_sp", "") for span in spans if span.get("id_sp")
                ],
                "token_count": section_token_count,
                "start_sec": section.get("start_sec"),
                "end_sec": section.get("end_sec"),
            }
        ]

    chunks: list[dict] = []
    current_spans: list[dict] = []
    current_token_count = 0
    subchunk_index = 0

    def flush() -> None:
        nonlocal current_spans, current_token_count, subchunk_index
        if not current_spans:
            return
        text = _join_span_text(current_spans)
        chunks.append(
            {
                "chunk_id": f"{document['domain']}::{document['doc_id']}::sec::{section['section_id']}::sub::{subchunk_index}",
                "domain": document["domain"],
                "doc_id": document["doc_id"],
                "doc_title": document.get("title", ""),
                "section_id": section["section_id"],
                "section_title": section.get("section_title", ""),
                "parent_titles": section.get("parent_titles", []),
                "subchunk_index": subchunk_index,
                "text": text.strip(),
                "span_ids": [
                    span.get("id_sp", "") for span in current_spans if span.get("id_sp")
                ],
                "token_count": _approximate_token_count(text),
                "start_sec": section.get("start_sec"),
                "end_sec": section.get("end_sec"),
            }
        )
        subchunk_index += 1
        current_spans = []
        current_token_count = 0

    for span in spans:
        span_tokens = _approximate_token_count(span.get("text_sp", ""))
        if current_spans and current_token_count + span_tokens > max_tokens_per_chunk:
            flush()
        current_spans.append(span)
        current_token_count += span_tokens

    flush()
    return chunks


def build_chunks(
    documents: list[dict],
    max_tokens_per_chunk: int = 512,
    domains: Iterable[str] | str | None = None,
) -> list[dict]:
    """Build deterministic section-aware chunks from document records.

    `max_tokens_per_chunk` is enforced with the local approximation in
    `_approximate_token_count`, not a model tokenizer.
    """

    domain_filter = normalize_domains(domains)
    chunks: list[dict] = []

    for document in documents:
        if domain_filter is not None and document.get("domain") not in domain_filter:
            continue
        sections = _group_sections(document)
        for section in sections:
            chunks.extend(_emit_section_chunks(document, section, max_tokens_per_chunk))

    return chunks


def write_chunks_jsonl(chunks: list[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(json.dumps(chunk, ensure_ascii=True))
            handle.write("\n")

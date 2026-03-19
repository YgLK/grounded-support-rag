"""MultiDoc2Dial raw dataset loaders.

This module stays close to the source JSON so downstream chunking and example
builders can work from predictable, serializable records.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Iterator

from support_graph.data._utils import normalize_domains, normalize_turn

DOC_FILENAME = "multidoc2dial_doc.json"
DIAL_FILENAME_TEMPLATE = "multidoc2dial_dial_{split}.json"
SUPPORTED_SPLITS = {"train", "validation", "test"}
_MISSING_POSITION = 10**18


def _sorted_items(mapping: dict) -> Iterator[tuple[str, object]]:
    for key in sorted(mapping):
        yield key, mapping[key]


def _load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _normalize_span(span: dict) -> dict:
    parent_titles = span.get("parent_titles")
    assert isinstance(parent_titles, list)
    tag = span.get("tag")
    text_sp = span.get("text_sp")
    title = span.get("title")
    text_sec = span.get("text_sec")
    assert tag is not None
    assert text_sp is not None
    assert title is not None
    assert text_sec is not None
    return {
        "id_sp": str(span.get("id_sp", "")),
        "tag": str(tag),
        "start_sp": span.get("start_sp"),
        "end_sp": span.get("end_sp"),
        "text_sp": str(text_sp),
        "title": str(title),
        "parent_titles": [
            item.get("text", "")
            for item in parent_titles
            if isinstance(item, dict) and item.get("text")
        ],
        "id_sec": str(span.get("id_sec", "")),
        "start_sec": span.get("start_sec"),
        "end_sec": span.get("end_sec"),
        "text_sec": str(text_sec),
    }


def load_documents(
    dataset_root: str | Path, domains: Iterable[str] | str | None = None
) -> list[dict]:
    """Load raw MultiDoc2Dial documents.

    The returned records preserve source metadata and add a normalized, sorted
    span list for downstream section grouping.
    """

    dataset_root = Path(dataset_root)
    domain_filter = normalize_domains(domains)
    payload = _load_json(dataset_root / DOC_FILENAME)
    doc_data = payload.get("doc_data")
    assert isinstance(doc_data, dict)
    documents: list[dict] = []

    for domain, docs in _sorted_items(doc_data):
        if domain_filter is not None and domain not in domain_filter:
            continue
        assert isinstance(docs, dict)
        for doc_id, doc in _sorted_items(docs):
            assert isinstance(doc, dict)
            spans = doc.get("spans")
            assert isinstance(spans, dict)
            normalized_spans = sorted(
                (_normalize_span(span) for span in spans.values()),
                key=lambda span: (
                    span["start_sec"]
                    if span["start_sec"] is not None
                    else _MISSING_POSITION,
                    span["start_sp"]
                    if span["start_sp"] is not None
                    else _MISSING_POSITION,
                    span.get("id_sp", ""),
                ),
            )
            documents.append(
                {
                    "domain": domain,
                    "doc_id": str(doc_id),
                    "title": str(doc["title"]),
                    "doc_text": str(doc["doc_text"]),
                    "doc_html_ts": str(doc["doc_html_ts"]),
                    "doc_html_raw": str(doc["doc_html_raw"]),
                    "spans": normalized_spans,
                    "raw_spans": spans,
                }
            )

    return documents


def load_dialogues(
    dataset_root: str | Path,
    split: str,
    domains: Iterable[str] | str | None = None,
) -> list[dict]:
    """Load raw MultiDoc2Dial dialogue records for a split."""

    dataset_root = Path(dataset_root)
    domain_filter = normalize_domains(domains)
    if split not in SUPPORTED_SPLITS:
        supported = ", ".join(sorted(SUPPORTED_SPLITS))
        raise ValueError(f"Unsupported split '{split}'. Expected one of: {supported}")
    payload = _load_json(dataset_root / DIAL_FILENAME_TEMPLATE.format(split=split))
    dial_data = payload.get("dial_data")
    assert isinstance(dial_data, dict)
    dialogues: list[dict] = []

    for domain, dials in _sorted_items(dial_data):
        if domain_filter is not None and domain not in domain_filter:
            continue
        assert isinstance(dials, list)
        for dial in dials:
            assert isinstance(dial, dict)
            turns = dial.get("turns")
            assert isinstance(turns, list)
            turns = sorted(
                turns,
                key=lambda turn: (
                    turn.get("turn_id") if turn.get("turn_id") is not None else 10**18,
                    turn.get("role", ""),
                ),
            )
            dialogues.append(
                {
                    "domain": domain,
                    "dial_id": str(dial["dial_id"]),
                    "turns": [normalize_turn(turn) for turn in turns],
                }
            )

    return dialogues

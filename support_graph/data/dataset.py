"""MultiDoc2Dial raw dataset loaders.

This module stays close to the source JSON so downstream chunking and example
builders can work from predictable, serializable records.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Iterator

DOC_FILENAME = "multidoc2dial_doc.json"
DIAL_FILENAME_TEMPLATE = "multidoc2dial_dial_{split}.json"
SUPPORTED_SPLITS = {"train", "validation", "test"}


def _normalize_domains(domains: Iterable[str] | str | None) -> set[str] | None:
    if domains is None:
        return None
    if isinstance(domains, str):
        values = [part.strip() for part in domains.split(",")]
    else:
        values = [str(domain).strip() for domain in domains]
    normalized = {value for value in values if value}
    return normalized or None


def _sorted_items(mapping: dict) -> Iterator[tuple[str, object]]:
    for key in sorted(mapping):
        yield key, mapping[key]


def _load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _normalize_reference(reference: dict) -> dict:
    return {
        "label": reference.get("label", ""),
        "id_sp": str(reference.get("id_sp", "")),
        "doc_id": str(reference.get("doc_id", "")),
    }


def _normalize_turn(turn: dict) -> dict:
    return {
        "turn_id": turn.get("turn_id"),
        "role": turn.get("role", ""),
        "da": turn.get("da", ""),
        "utterance": turn.get("utterance", ""),
        "references": [
            _normalize_reference(reference)
            for reference in turn.get("references", [])
            if reference is not None
        ],
    }


def _normalize_span(span: dict) -> dict:
    parent_titles = span.get("parent_titles", []) or []
    return {
        "id_sp": str(span.get("id_sp", "")),
        "tag": span.get("tag", ""),
        "start_sp": span.get("start_sp"),
        "end_sp": span.get("end_sp"),
        "text_sp": span.get("text_sp", ""),
        "title": span.get("title", ""),
        "parent_titles": [
            item.get("text", "")
            for item in parent_titles
            if isinstance(item, dict) and item.get("text")
        ],
        "id_sec": str(span.get("id_sec", "")),
        "start_sec": span.get("start_sec"),
        "end_sec": span.get("end_sec"),
        "text_sec": span.get("text_sec", ""),
    }


def load_documents(
    dataset_root: str | Path, domains: Iterable[str] | str | None = None
) -> list[dict]:
    """Load raw MultiDoc2Dial documents.

    The returned records preserve source metadata and add a normalized, sorted
    span list for downstream section grouping.
    """

    dataset_root = Path(dataset_root)
    domain_filter = _normalize_domains(domains)
    payload = _load_json(dataset_root / DOC_FILENAME)
    documents: list[dict] = []

    for domain, docs in _sorted_items(payload.get("doc_data", {})):
        if domain_filter is not None and domain not in domain_filter:
            continue
        for doc_id, doc in _sorted_items(docs):
            spans = doc.get("spans", {}) or {}
            normalized_spans = sorted(
                (_normalize_span(span) for span in spans.values()),
                key=lambda span: (
                    span.get("start_sec")
                    if span.get("start_sec") is not None
                    else 10**18,
                    span.get("start_sp")
                    if span.get("start_sp") is not None
                    else 10**18,
                    span.get("id_sp", ""),
                ),
            )
            documents.append(
                {
                    "domain": domain,
                    "doc_id": str(doc_id),
                    "title": doc.get("title", ""),
                    "doc_text": doc.get("doc_text", ""),
                    "doc_html_ts": doc.get("doc_html_ts", ""),
                    "doc_html_raw": doc.get("doc_html_raw", ""),
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
    domain_filter = _normalize_domains(domains)
    if split not in SUPPORTED_SPLITS:
        supported = ", ".join(sorted(SUPPORTED_SPLITS))
        raise ValueError(f"Unsupported split '{split}'. Expected one of: {supported}")
    payload = _load_json(dataset_root / DIAL_FILENAME_TEMPLATE.format(split=split))
    dialogues: list[dict] = []

    for domain, dials in _sorted_items(payload.get("dial_data", {})):
        if domain_filter is not None and domain not in domain_filter:
            continue
        for dial in dials:
            turns = sorted(
                dial.get("turns", []) or [],
                key=lambda turn: (
                    turn.get("turn_id") if turn.get("turn_id") is not None else 10**18,
                    turn.get("role", ""),
                ),
            )
            dialogues.append(
                {
                    "domain": domain,
                    "dial_id": str(dial.get("dial_id", "")),
                    "turns": [_normalize_turn(turn) for turn in turns],
                }
            )

    return dialogues

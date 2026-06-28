"""MultiDoc2Dial raw dataset loaders.

This module stays close to the source JSON so downstream chunking and example
builders can work from predictable, serializable records.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Iterator, cast

from support_graph.data._utils import normalize_domains, normalize_turn
from support_graph.data.kubernetes import load_kubernetes_documents
from support_graph.types import (
    DatasetSplitLike,
    Dialogue,
    Document,
    DocumentSpan,
    Domain,
    DomainLike,
    parse_dataset_split,
    parse_domain,
)

__all__ = [
    "load_documents",
    "load_dialogues",
]

DOC_FILENAME = "multidoc2dial_doc.json"
DIAL_FILENAME_TEMPLATE = "multidoc2dial_dial_{split}.json"
_MISSING_POSITION = 10**18


def _sorted_items(mapping: dict) -> Iterator[tuple[str, object]]:
    for key in sorted(mapping):
        yield key, mapping[key]


def _load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _require_dict(value: object, name: str) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"Expected {name} to be an object.")
    return cast(dict, value)


def _require_list(value: object, name: str) -> list:
    if not isinstance(value, list):
        raise ValueError(f"Expected {name} to be a list.")
    return cast(list, value)


def _require_value(record: dict, field: str) -> object:
    value = record.get(field)
    if value is None:
        raise ValueError(f"Missing required field: {field}")
    return value


def _normalize_span(span: dict[str, Any]) -> DocumentSpan:
    """Normalize a raw document span from the dataset into a typed dictionary.

    - Validates that required fields are present.
    - Flattens the `parent_titles` structure into a simple list of strings.
    - Ensures all identifiers and text fields are strings.

    Args:
        span: A raw span dictionary from the source JSON.

    Returns:
        A validated and normalized `DocumentSpan` object.
    """
    parent_titles = _require_list(span.get("parent_titles"), "parent_titles")
    tag = _require_value(span, "tag")
    text_sp = _require_value(span, "text_sp")
    title = _require_value(span, "title")
    text_sec = _require_value(span, "text_sec")
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
    dataset_root: str | Path,
    domains: Iterable[DomainLike] | DomainLike | None = None,
) -> list[Document]:
    """Load raw MultiDoc2Dial document records from the dataset.

    Parses the source JSON and returns document records with normalized metadata
    and a sorted span list suitable for downstream section grouping and chunking.

    Args:
        dataset_root: Path to the MultiDoc2Dial dataset directory.
        domains: Optional domain filter. Can be a single domain, an iterable of
            domains, or None to load all domains.

    Returns:
        List of Document records with normalized spans sorted by section and
        span positions.

    Raises:
        FileNotFoundError: If the document file does not exist.
        ValueError: If required fields are missing from the source data.
    """
    dataset_root = Path(dataset_root)
    domain_filter = normalize_domains(domains)
    if domain_filter == {Domain.KUBERNETES}:
        return load_kubernetes_documents(dataset_root)

    payload = _load_json(dataset_root / DOC_FILENAME)
    doc_data = _require_dict(payload.get("doc_data"), "doc_data")
    documents: list[Document] = []

    for domain, docs in _sorted_items(doc_data):
        resolved_domain = parse_domain(domain)
        if domain_filter is not None and resolved_domain not in domain_filter:
            continue
        docs = _require_dict(docs, f"doc_data[{domain}]")
        for doc_id, doc in _sorted_items(docs):
            doc = _require_dict(doc, f"doc_data[{domain}][{doc_id}]")
            spans = _require_dict(
                doc.get("spans"),
                f"doc_data[{domain}][{doc_id}].spans",
            )
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
                    "domain": str(resolved_domain),
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
    split: DatasetSplitLike,
    domains: Iterable[DomainLike] | DomainLike | None = None,
) -> list[Dialogue]:
    """Load raw MultiDoc2Dial dialogue records for a given split.

    Parses the source JSON for a dataset split (e.g., 'train', 'validation')
    and returns dialogue records with normalized, sorted turns.

    A dialogue logically represents a single conversation, containing:
    - A unique dialogue ID.
    - A list of conversational turns, sorted chronologically.
    - Each turn includes the speaker's role (user/agent), their utterance,
      and any document references made.

    Args:
        dataset_root: Path to the MultiDoc2Dial dataset directory.
        split: The dataset split to load ('train', 'validation', or 'test').
        domains: Optional domain filter. Can be a single domain, an iterable of
            domains, or None to load all domains.

    Returns:
        A list of `Dialogue` records with normalized turns sorted by turn ID.

    Raises:
        FileNotFoundError: If the dialogue file for the split does not exist.
        ValueError: If required fields are missing from the source data.
    """

    dataset_root = Path(dataset_root)
    domain_filter = normalize_domains(domains)
    if domain_filter and Domain.KUBERNETES in domain_filter:
        raise ValueError("Kubernetes uses curated eval subsets, not dialogue files.")
    resolved_split = parse_dataset_split(split)
    payload = _load_json(
        dataset_root / DIAL_FILENAME_TEMPLATE.format(split=resolved_split)
    )
    dial_data = _require_dict(payload.get("dial_data"), "dial_data")
    dialogues: list[Dialogue] = []

    for domain, dials in _sorted_items(dial_data):
        resolved_domain = parse_domain(domain)
        if domain_filter is not None and resolved_domain not in domain_filter:
            continue
        dials = _require_list(dials, f"dial_data[{domain}]")
        for dial in dials:
            dial = _require_dict(dial, f"dial_data[{domain}][]")
            turns = _require_list(dial.get("turns"), f"dial_data[{domain}][].turns")
            turns = sorted(
                turns,
                key=lambda turn: (
                    turn.get("turn_id") if turn.get("turn_id") is not None else 10**18,
                    turn.get("role", ""),
                ),
            )
            dialogues.append(
                {
                    "domain": str(resolved_domain),
                    "dial_id": str(dial["dial_id"]),
                    "turns": [normalize_turn(turn) for turn in turns],
                }
            )

    return dialogues

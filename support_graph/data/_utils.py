"""Shared normalization helpers for dataset-derived records."""

from __future__ import annotations

from typing import Iterable


def normalize_domains(domains: Iterable[str] | str | None) -> set[str] | None:
    if domains is None:
        return None
    if isinstance(domains, str):
        values = [part.strip() for part in domains.split(",")]
    else:
        values = [str(domain).strip() for domain in domains]
    normalized = {value for value in values if value}
    return normalized or None


def normalize_reference(reference: dict) -> dict:
    label = reference.get("label")
    id_sp = reference.get("id_sp")
    doc_id = reference.get("doc_id")
    assert label is not None
    assert id_sp is not None
    assert doc_id is not None
    return {
        "label": str(label),
        "id_sp": str(id_sp),
        "doc_id": str(doc_id),
    }


def normalize_turn(turn: dict) -> dict:
    references = turn.get("references")
    assert isinstance(references, list)
    role = turn.get("role")
    da = turn.get("da")
    utterance = turn.get("utterance")
    assert role is not None
    assert da is not None
    assert utterance is not None
    return {
        "turn_id": turn.get("turn_id"),
        "role": str(role),
        "da": str(da),
        "utterance": str(utterance),
        "references": [
            normalize_reference(reference)
            for reference in references
            if reference is not None
        ],
    }


__all__ = [
    "normalize_domains",
    "normalize_reference",
    "normalize_turn",
]

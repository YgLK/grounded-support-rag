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
    return {
        "label": reference.get("label", ""),
        "id_sp": str(reference.get("id_sp", "")),
        "doc_id": str(reference.get("doc_id", "")),
    }


def normalize_turn(turn: dict) -> dict:
    return {
        "turn_id": turn.get("turn_id"),
        "role": turn.get("role", ""),
        "da": turn.get("da", ""),
        "utterance": turn.get("utterance", ""),
        "references": [
            normalize_reference(reference)
            for reference in turn.get("references", [])
            if reference is not None
        ],
    }


__all__ = [
    "normalize_domains",
    "normalize_reference",
    "normalize_turn",
]

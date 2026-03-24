"""Shared normalization helpers for dataset-derived records."""

from __future__ import annotations

from typing import Any, Iterable, cast

from support_graph.types import (
    DialogueTurn,
    Domain,
    DomainLike,
    Reference,
    TurnRole,
    parse_domain,
)

__all__ = [
    "normalize_domains",
    "normalize_reference",
    "normalize_turn",
]


SUPPORTED_TURN_ROLES = frozenset({"agent", "user"})


def _required_value(record: dict, field: str) -> object:
    value = record.get(field)
    if value is None:
        raise ValueError(f"Missing required field: {field}")
    return value


def _required_list(record: dict, field: str) -> list:
    value = record.get(field)
    if not isinstance(value, list):
        raise ValueError(f"Expected {field} to be a list.")
    return value


def normalize_domains(
    domains: Iterable[DomainLike] | DomainLike | None,
) -> set[Domain] | None:
    if domains is None:
        return None
    if isinstance(domains, str):
        values = [part.strip() for part in domains.split(",")]
    else:
        values = [str(domain).strip() for domain in domains]
    normalized = {parse_domain(value) for value in values if value}
    return normalized or None


def _normalize_turn_role(value: object) -> TurnRole:
    normalized = str(value).strip().lower()
    if normalized not in SUPPORTED_TURN_ROLES:
        supported = ", ".join(sorted(SUPPORTED_TURN_ROLES))
        raise ValueError(
            f"Unsupported turn role '{value}'. Expected one of: {supported}"
        )
    return cast(TurnRole, normalized)


def normalize_reference(reference: dict[str, Any]) -> Reference:
    label = _required_value(reference, "label")
    id_sp = _required_value(reference, "id_sp")
    doc_id = _required_value(reference, "doc_id")
    return {
        "label": str(label),
        "id_sp": str(id_sp),
        "doc_id": str(doc_id),
    }


def normalize_turn(turn: dict[str, Any]) -> DialogueTurn:
    references = _required_list(turn, "references")
    role = _required_value(turn, "role")
    da = _required_value(turn, "da")
    utterance = _required_value(turn, "utterance")
    return {
        "turn_id": turn.get("turn_id"),
        "role": _normalize_turn_role(role),
        "da": str(da),
        "utterance": str(utterance),
        "references": [
            normalize_reference(reference)
            for reference in references
            if reference is not None
        ],
    }

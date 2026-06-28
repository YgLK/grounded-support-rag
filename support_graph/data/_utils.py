"""Shared normalization helpers for data records."""

from __future__ import annotations

from typing import Iterable

from support_graph.types import Domain, DomainLike, parse_domain

__all__ = ["normalize_domains"]


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

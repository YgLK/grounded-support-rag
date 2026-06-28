"""Document loaders for supported corpora."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from support_graph.data._utils import normalize_domains
from support_graph.data.kubernetes import load_kubernetes_documents
from support_graph.types import Document, Domain, DomainLike

__all__ = ["load_documents"]


def load_documents(
    dataset_root: str | Path,
    domains: Iterable[DomainLike] | DomainLike | None = None,
) -> list[Document]:
    domain_filter = normalize_domains(domains)
    if domain_filter in (None, {Domain.KUBERNETES}):
        return load_kubernetes_documents(dataset_root)

    requested = ", ".join(sorted(str(domain) for domain in domain_filter))
    raise ValueError(f"Unsupported document domain(s): {requested}")

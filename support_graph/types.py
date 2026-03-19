"""Shared enums and literal aliases for public SupportGraph contracts."""

from __future__ import annotations

from enum import StrEnum
from typing import Iterable, Literal, TypeAlias


class Domain(StrEnum):
    DMV = "dmv"
    SSA = "ssa"
    STUDENTAID = "studentaid"
    VA = "va"


class DatasetSplit(StrEnum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


class EvalSubset(StrEnum):
    SMOKE = "smoke"
    FROZEN_ABLATION = "frozen_ablation"
    FULL_VALIDATION = "full_validation"


DomainLike: TypeAlias = Domain | str
DatasetSplitLike: TypeAlias = DatasetSplit | str
EvalSubsetLike: TypeAlias = EvalSubset | str
TargetMode = Literal["answer", "follow_up"]
TurnRole = Literal["agent", "user"]
QueryContextRole = Literal["agent", "user", "unknown"]

SUPPORTED_DOMAINS = frozenset(domain.value for domain in Domain)
SUPPORTED_SPLITS = frozenset(split.value for split in DatasetSplit)
SUPPORTED_EVAL_SUBSETS = frozenset(subset.value for subset in EvalSubset)


def _normalize_choice(value: str) -> str:
    return value.strip().lower()


def parse_domain(value: DomainLike) -> Domain:
    if isinstance(value, Domain):
        return value
    normalized = _normalize_choice(str(value))
    if normalized not in SUPPORTED_DOMAINS:
        supported = ", ".join(sorted(SUPPORTED_DOMAINS))
        raise ValueError(f"Unsupported domain '{value}'. Expected one of: {supported}")
    return Domain(normalized)


def parse_domains(values: Iterable[DomainLike]) -> tuple[Domain, ...]:
    ordered: list[Domain] = []
    seen: set[Domain] = set()
    for value in values:
        domain = parse_domain(value)
        if domain in seen:
            continue
        seen.add(domain)
        ordered.append(domain)
    return tuple(ordered)


def parse_dataset_split(value: DatasetSplitLike) -> DatasetSplit:
    if isinstance(value, DatasetSplit):
        return value
    normalized = _normalize_choice(str(value))
    if normalized not in SUPPORTED_SPLITS:
        supported = ", ".join(sorted(SUPPORTED_SPLITS))
        raise ValueError(f"Unsupported split '{value}'. Expected one of: {supported}")
    return DatasetSplit(normalized)


def parse_eval_subset(value: EvalSubsetLike) -> EvalSubset:
    if isinstance(value, EvalSubset):
        return value
    normalized = _normalize_choice(str(value))
    if normalized not in SUPPORTED_EVAL_SUBSETS:
        supported = ", ".join(sorted(SUPPORTED_EVAL_SUBSETS))
        raise ValueError(
            f"Unsupported eval subset '{value}'. Expected one of: {supported}"
        )
    return EvalSubset(normalized)


__all__ = [
    "DatasetSplit",
    "DatasetSplitLike",
    "Domain",
    "DomainLike",
    "EvalSubset",
    "EvalSubsetLike",
    "QueryContextRole",
    "SUPPORTED_DOMAINS",
    "SUPPORTED_EVAL_SUBSETS",
    "SUPPORTED_SPLITS",
    "TargetMode",
    "TurnRole",
    "parse_dataset_split",
    "parse_domain",
    "parse_domains",
    "parse_eval_subset",
]

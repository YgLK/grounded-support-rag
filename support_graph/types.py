"""Shared enums and literal aliases for public SupportGraph contracts."""

from __future__ import annotations

from enum import StrEnum
from typing import Iterable, Literal, Self, TypeAlias


class ChoiceStrEnum(StrEnum):
    @classmethod
    def values(cls) -> tuple[str, ...]:
        return tuple(item.value for item in cls)

    @classmethod
    def value_set(cls) -> frozenset[str]:
        return frozenset(cls.values())

    @classmethod
    def parse(cls, value: str | Self) -> Self:
        if isinstance(value, cls):
            return value
        normalized = str(value).strip().lower()
        try:
            return cls(normalized)
        except ValueError as exc:
            supported = ", ".join(sorted(cls.value_set()))
            raise ValueError(
                f"Unsupported {cls.__name__.lower()} '{value}'. Expected one of: {supported}"
            ) from exc


class Domain(ChoiceStrEnum):
    DMV = "dmv"
    SSA = "ssa"
    STUDENTAID = "studentaid"
    VA = "va"


class DatasetSplit(ChoiceStrEnum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


class EvalSubset(ChoiceStrEnum):
    SMOKE = "smoke"
    FROZEN_ABLATION = "frozen_ablation"
    FULL_VALIDATION = "full_validation"


DomainLike: TypeAlias = Domain | str
DatasetSplitLike: TypeAlias = DatasetSplit | str
EvalSubsetLike: TypeAlias = EvalSubset | str
TargetMode = Literal["answer", "follow_up"]
TurnRole = Literal["agent", "user"]
QueryContextRole = Literal["agent", "user", "unknown"]


def parse_domain(value: DomainLike) -> Domain:
    return Domain.parse(value)


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
    return DatasetSplit.parse(value)


def parse_eval_subset(value: EvalSubsetLike) -> EvalSubset:
    return EvalSubset.parse(value)


__all__ = [
    "DatasetSplit",
    "DatasetSplitLike",
    "Domain",
    "DomainLike",
    "EvalSubset",
    "EvalSubsetLike",
    "QueryContextRole",
    "ChoiceStrEnum",
    "TargetMode",
    "TurnRole",
    "parse_dataset_split",
    "parse_domain",
    "parse_domains",
    "parse_eval_subset",
]

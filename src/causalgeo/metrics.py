"""Declare how each KPI should be tested."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

_KINDS = frozenset({"continuous", "count", "rate", "categorical"})
_ROLES = frozenset({"outcome", "delivery", "covariate"})


@dataclass(frozen=True)
class MetricSpec:
    """Type-correct metric declaration for ``diagnose()``."""

    name: str
    kind: str = "continuous"
    column: Optional[str] = None
    successes: Optional[str] = None
    trials: Optional[str] = None
    categories: Optional[str] = None
    counts: Optional[str] = None
    role: str = "outcome"
    primary: bool = False

    def __post_init__(self) -> None:
        kind = self.kind.strip().lower()
        role = self.role.strip().lower()
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "role", role)
        if kind not in _KINDS:
            raise ValueError(f"kind must be one of {sorted(_KINDS)}")
        if role not in _ROLES:
            raise ValueError(f"role must be one of {sorted(_ROLES)}")
        if kind == "rate" and not (self.successes and self.trials):
            raise ValueError("rate metrics need successes= and trials= columns")
        if kind == "categorical" and not (self.categories and self.counts):
            raise ValueError("categorical metrics need categories= and counts=")

    @property
    def value_column(self) -> str:
        if self.kind == "rate":
            return str(self.successes)
        if self.kind == "categorical":
            return str(self.counts)
        return self.column or self.name

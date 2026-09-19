"""Treatment and control geo assignment (GeoLift locations / GeoX geos)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import pandas as pd

from causalgeo.exceptions import EmptyAssignmentError, MixedRoleError, OverlapError

_TREATMENT_ROLES = frozenset(
    {
        "treatment",
        "treat",
        "test",
        "treated",
        "holdout",
        "go_dark",
        "go-dark",
        "holdback",
        "heavy_up",
        "heavy-up",
    }
)
_CONTROL_ROLES = frozenset({"control", "bau", "business_as_usual", "donor"})


def _norm_role(value: object) -> str:
    return str(value).strip().lower().replace(" ", "_")


@dataclass(frozen=True)
class AssignmentMap:
    """Maps geos (locations) to treatment vs control.

    Parameters
    ----------
    treatment_geos :
        Test markets (GeoLift) / treatment group (GeoX).
    control_geos :
        Control markets / control group. Required for incrementality.
    universe :
        Optional full set of geos in the study region. Geos in ``universe``
        that are neither treatment nor control are ``unassigned_geos``.
    """

    treatment_geos: frozenset[str]
    control_geos: frozenset[str]
    universe: Optional[frozenset[str]] = None

    def __post_init__(self) -> None:
        mixed = self.treatment_geos & self.control_geos
        if mixed:
            raise MixedRoleError(
                f"Geos listed as both treatment and control: {sorted(mixed)[:10]}"
            )
        if not self.treatment_geos and not self.control_geos:
            raise EmptyAssignmentError("Assignment has no treatment or control geos")

    @property
    def unassigned_geos(self) -> frozenset[str]:
        """Geos in the universe that are not in this test (GeoX unused geos)."""
        if self.universe is None:
            return frozenset()
        return self.universe - self.treatment_geos - self.control_geos

    @property
    def experimental_geos(self) -> frozenset[str]:
        return self.treatment_geos | self.control_geos

    # GeoLift aliases
    @property
    def test_locations(self) -> frozenset[str]:
        return self.treatment_geos

    @property
    def control_locations(self) -> frozenset[str]:
        return self.control_geos

    def require_control(self) -> None:
        if not self.control_geos:
            from causalgeo.exceptions import MissingControlError

            raise MissingControlError(
                "Incrementality needs a control group (BAU geos). "
                "Comparing treatment arms to each other is not a lift study."
            )

    def assert_overlay_allowed(self, overlay: "AssignmentMap") -> None:
        """Heavy-up / holdback overlay must not use this test's treatment or control geos.

        Changing spend in an existing control group contaminates that test's
        counterfactual. Changing spend in an existing treatment group breaks
        the intervention.
        """
        blocked = self.experimental_geos
        hit = overlay.experimental_geos & blocked
        if hit:
            raise OverlapError(
                "Overlay geos overlap another test's treatment or control group: "
                f"{sorted(hit)[:15]}"
            )

    @classmethod
    def from_sets(
        cls,
        treatment_geos: Iterable[object],
        control_geos: Iterable[object],
        universe: Optional[Iterable[object]] = None,
    ) -> "AssignmentMap":
        uni = frozenset(str(x) for x in universe) if universe is not None else None
        return cls(
            treatment_geos=frozenset(str(x) for x in treatment_geos),
            control_geos=frozenset(str(x) for x in control_geos),
            universe=uni,
        )

    @classmethod
    def from_frame(
        cls,
        frame: pd.DataFrame,
        location: str = "location",
        role: str = "role",
        universe: Optional[Iterable[object]] = None,
    ) -> "AssignmentMap":
        """Build from a long assignment table (one row per geo × cell × role).

        Duplicate geos across cells are collapsed to unique locations.
        ``role`` accepts GeoX/GeoLift-style labels: treatment, test, holdout,
        go_dark, holdback, heavy_up, control, bau, donor.
        """
        if location not in frame.columns or role not in frame.columns:
            raise ValueError(f"Assignment needs columns {location!r} and {role!r}")
        work = frame[[location, role]].copy()
        work[location] = work[location].astype(str)
        work["_role"] = work[role].map(_norm_role)
        treat = set()
        ctrl = set()
        unknown = set()
        for loc, r in zip(work[location], work["_role"]):
            if r in _TREATMENT_ROLES:
                treat.add(loc)
            elif r in _CONTROL_ROLES:
                ctrl.add(loc)
            else:
                unknown.add(r)
        if unknown:
            raise ValueError(f"Unknown role values: {sorted(unknown)}")
        return cls.from_sets(treat, ctrl, universe=universe)

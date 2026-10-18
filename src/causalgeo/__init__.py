"""causalgeo: geo incrementality experiments (holdback, go-dark, heavy-up)."""

from causalgeo.assignment import AssignmentMap
from causalgeo.calibrated import disentangle
from causalgeo.design import design_concurrent_test
from causalgeo.diagnostics import DiagnosticsReport, diagnose
from causalgeo.estimate import LiftReport
from causalgeo.exceptions import (
    CausalGeoError,
    MissingControlError,
    MissingSpendError,
    MixedRoleError,
    OverlapError,
)
from causalgeo.experiment import GeoExperiment
from causalgeo.interventions import HEAVY_UP, HOLDBACK, GO_DARK, Intervention
from causalgeo.metrics import MetricSpec

__version__ = "0.0.1"

__all__ = [
    "AssignmentMap",
    "CausalGeoError",
    "DiagnosticsReport",
    "GeoExperiment",
    "GO_DARK",
    "HEAVY_UP",
    "HOLDBACK",
    "Intervention",
    "LiftReport",
    "MetricSpec",
    "MissingControlError",
    "MissingSpendError",
    "MixedRoleError",
    "OverlapError",
    "design_concurrent_test",
    "diagnose",
    "disentangle",
    "__version__",
]

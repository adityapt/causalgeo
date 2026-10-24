"""causalgeo: geo incrementality experiments (holdback, go-dark, heavy-up)."""

from causalgeo.assignment import AssignmentMap
from causalgeo.bootstrap import bootstrap_channel_intervals
from causalgeo.calibrated import compare_calibration, disentangle, experiment_only_split
from causalgeo.design import design_concurrent_test, design_from_assignment
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
from causalgeo.grid import GridIdentification, identify_grid
from causalgeo.interventions import HEAVY_UP, HOLDBACK, GO_DARK, Intervention
from causalgeo.metrics import MetricSpec

__version__ = "0.2.0"

__all__ = [
    "AssignmentMap",
    "CausalGeoError",
    "DiagnosticsReport",
    "GeoExperiment",
    "GridIdentification",
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
    "bootstrap_channel_intervals",
    "compare_calibration",
    "design_concurrent_test",
    "design_from_assignment",
    "diagnose",
    "disentangle",
    "experiment_only_split",
    "identify_grid",
    "__version__",
]

"""causalgeo: geo incrementality experiments (holdback, go-dark, heavy-up)."""

from causalgeo.allocation import (
    AnchoredAllocation,
    anchored_channel_allocation,
    potency_misspecification_ratio,
    potency_tie_ratio,
    prevented_effective_dose,
)
from causalgeo.assignment import AssignmentMap
from causalgeo.bayesian import BayesianAllocation, bayesian_channel_allocation
from causalgeo.bootstrap import bootstrap_channel_intervals
from causalgeo.calibrated import (
    bundle_impression_contrasts,
    compare_calibration,
    disentangle,
    experiment_only_split,
)
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
    "AnchoredAllocation",
    "AssignmentMap",
    "BayesianAllocation",
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
    "anchored_channel_allocation",
    "bayesian_channel_allocation",
    "bundle_impression_contrasts",
    "compare_calibration",
    "design_concurrent_test",
    "design_from_assignment",
    "diagnose",
    "disentangle",
    "experiment_only_split",
    "potency_misspecification_ratio",
    "potency_tie_ratio",
    "prevented_effective_dose",
    "identify_grid",
    "__version__",
]

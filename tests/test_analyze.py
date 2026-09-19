"""Synthetic geo tests: recover ATT; reject analyses with no control group."""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from causalgeo import (
    AssignmentMap,
    GeoExperiment,
    Intervention,
    MissingControlError,
    MissingSpendError,
    OverlapError,
)


def _kpi_panel(true_att: float, treatment_geos, control_geos, seed: int = 1):
    rng = np.random.default_rng(seed)
    start = datetime(2026, 1, 1)
    dates = [start + timedelta(days=i) for i in range(40)]
    treat_start = datetime(2026, 1, 21)
    rows = []
    for geo in list(treatment_geos) + list(control_geos):
        fe = rng.normal(20, 2)
        is_t = geo in set(treatment_geos)
        for t, day in enumerate(dates):
            post = int(day >= treat_start)
            y = fe + 0.1 * t + true_att * is_t * post + rng.normal(0, 0.35)
            spend = 10.0 * (0 if (is_t and post) else 1)
            rows.append(
                {
                    "location": geo,
                    "date": day,
                    "conversions": y,
                    "spend": spend,
                }
            )
    return pd.DataFrame(rows), treat_start


def test_go_dark_recovers_att():
    treat = [f"G{i}" for i in range(8)]
    ctrl = [f"C{i}" for i in range(8)]
    true = -4.0
    data, t0 = _kpi_panel(true, treat, ctrl)
    assignment = AssignmentMap.from_sets(treat, ctrl)
    report = GeoExperiment(data, assignment, t0, spend="spend").analyze(Intervention.GO_DARK)
    assert report.lift == pytest.approx(true, abs=0.3)
    assert report.n_treatment_geos == 8
    assert report.n_control_geos == 8
    assert report.cov_type == "cluster"
    # Go-dark: treatment spend falls, conversions fall, so iCPD is positive.
    assert report.spend_delta == pytest.approx(-10.0, abs=0.01)
    assert report.icpd == pytest.approx(true / -10.0, abs=0.05)


def test_icpd_requires_spend_series():
    treat = [f"G{i}" for i in range(4)]
    ctrl = [f"C{i}" for i in range(4)]
    data, t0 = _kpi_panel(-2.0, treat, ctrl)
    assignment = AssignmentMap.from_sets(treat, ctrl)
    exp = GeoExperiment(data, assignment, t0)  # no spend column
    assert exp.analyze(Intervention.GO_DARK).icpd is None
    with pytest.raises(MissingSpendError):
        exp.analyze(Intervention.GO_DARK, require_icpd=True)


def test_assignment_from_long_frame_unique_geos():
    frame = pd.DataFrame(
        {
            "location": ["A", "A", "B", "C"],
            "role": ["control", "control", "holdout", "test"],
            "cell": [1, 2, 1, 1],
        }
    )
    amap = AssignmentMap.from_frame(frame, universe=["A", "B", "C", "D"])
    assert amap.control_geos == frozenset({"A"})
    assert amap.treatment_geos == frozenset({"B", "C"})
    assert amap.unassigned_geos == frozenset({"D"})
    assert amap.test_locations == amap.treatment_geos


def test_missing_control_is_not_incrementality():
    treat = ["T1", "T2"]
    data, t0 = _kpi_panel(-1.0, treat, ["C1"])
    assignment = AssignmentMap.from_sets(treat, [])
    with pytest.raises(MissingControlError):
        GeoExperiment(data, assignment, t0)


def test_overlay_cannot_use_existing_control_geos():
    base = AssignmentMap.from_sets(["T1"], ["C1", "C2"], universe=["T1", "C1", "C2", "U1"])
    overlay = AssignmentMap.from_sets(["U1"], ["C1"])
    with pytest.raises(OverlapError):
        base.assert_overlay_allowed(overlay)
    ok = AssignmentMap.from_sets(["U1"], base.unassigned_geos - {"U1"})
    base.assert_overlay_allowed(ok)

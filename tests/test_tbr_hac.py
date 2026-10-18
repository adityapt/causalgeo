"""TBR inference uses HAC SEs when pretest residuals are serially correlated."""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from causalgeo import AssignmentMap, GeoExperiment, Intervention
from causalgeo.counterfactual import estimate_tbr
from causalgeo.panel import build_panel


def _ar1_panel(phi: float, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    start = datetime(2026, 1, 1)
    dates = [start + timedelta(days=i) for i in range(50)]
    treat_start = datetime(2026, 2, 10)
    treat = [f"G{i}" for i in range(6)]
    ctrl = [f"C{i}" for i in range(12)]
    shock = 0.0
    rows = []
    for day in dates:
        shock = phi * shock + rng.normal(0, 1.0)
        post = int(day >= treat_start)
        for geo in treat:
            rows.append(
                {
                    "location": geo,
                    "date": day,
                    "conversions": 20.0 + shock + rng.normal(0, 0.05),
                    "is_t": 1,
                    "is_post": post,
                }
            )
        for geo in ctrl:
            rows.append(
                {
                    "location": geo,
                    "date": day,
                    "conversions": 20.0 + rng.normal(0, 0.05),
                    "is_t": 0,
                    "is_post": post,
                }
            )
    frame = pd.DataFrame(rows)
    assignment = AssignmentMap.from_sets(treat, ctrl)
    return build_panel(
        frame,
        assignment,
        treatment_start=treat_start,
    )


def test_tbr_default_reports_hac_covariance():
    treat = [f"G{i}" for i in range(8)]
    ctrl = [f"C{i}" for i in range(8)]
    rng = np.random.default_rng(1)
    start = datetime(2026, 1, 1)
    dates = [start + timedelta(days=i) for i in range(40)]
    t0 = datetime(2026, 1, 21)
    rows = []
    for geo in treat + ctrl:
        fe = rng.normal(20, 2)
        is_t = geo in set(treat)
        for t, day in enumerate(dates):
            post = int(day >= t0)
            rows.append(
                {
                    "location": geo,
                    "date": day,
                    "conversions": fe + 0.1 * t - 4.0 * is_t * post + rng.normal(0, 0.35),
                }
            )
    data = pd.DataFrame(rows)
    report = GeoExperiment(data, AssignmentMap.from_sets(treat, ctrl), t0).analyze(
        Intervention.GO_DARK
    )
    assert report.cov_type == "hac"


def test_tbr_hac_se_exceeds_iid_when_residuals_are_ar1():
    panel = _ar1_panel(phi=0.8)
    hac = estimate_tbr(panel)
    iid = estimate_tbr(panel, se="iid")
    assert hac.cov_type == "hac"
    assert iid.cov_type == "iid"
    assert hac.lift == pytest.approx(iid.lift, rel=1e-12)
    assert hac.std_error > iid.std_error * 1.15

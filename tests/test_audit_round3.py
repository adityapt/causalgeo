"""Edge cases found in the second bug hunt."""

import numpy as np
import pandas as pd
import pytest

from causalgeo import AssignmentMap, GeoExperiment, design_concurrent_test
from causalgeo.calibrated import _adstock as calibrated_adstock
from causalgeo.design import design_from_assignment
from causalgeo.identified import estimate_identified
from causalgeo.simulate import simulate_panel


def _panel(seed=3):
    data, start = simulate_panel(np.random.default_rng(seed), n_treatment=4, n_control=14, att=-4.0)
    assignment = AssignmentMap.from_sets([f"T{i}" for i in range(4)], [f"C{i}" for i in range(4, 18)])
    return data, start, assignment


def test_tbr_drops_a_treated_geo_whose_pretest_outcomes_are_blank():
    data, start, assignment = _panel()
    intact = GeoExperiment(data, assignment, start).analyze().lift
    blanked = data.copy()
    pre = pd.to_datetime(blanked["date"]) < pd.to_datetime(start)
    blanked.loc[(blanked["location"] == "T0") & pre, "conversions"] = np.nan
    with pytest.warns(UserWarning, match="dropped geos"):
        lift = GeoExperiment(blanked, assignment, start).analyze().lift
    assert abs(lift - intact) < 0.5


def _design_and_metrics():
    geos = pd.DataFrame({"geo": [f"g{i}" for i in range(12)], "stratum": ["s"] * 6 + ["t"] * 6})
    design = design_concurrent_test(geos, ["A", "B"], [["A"], ["B"]], control_share=0.34, seed=1)
    dates = pd.date_range("2024-01-01", periods=8)
    metrics = pd.DataFrame(
        [
            {"geo": g, "date": d, "sales": 10.0 + i + 0.1 * j}
            for i, g in enumerate(geos["geo"])
            for j, d in enumerate(dates)
        ]
    )
    return design, metrics


def test_estimate_identified_accepts_a_non_unique_index():
    design, metrics = _design_and_metrics()
    clean = estimate_identified(design, metrics, "sales", "2024-01-05").cells["coef"].to_numpy()
    shuffled = metrics.copy()
    shuffled.index = np.r_[np.arange(len(metrics) // 2), np.arange(len(metrics) - len(metrics) // 2)]
    again = estimate_identified(design, shuffled, "sales", "2024-01-05").cells["coef"].to_numpy()
    assert np.allclose(clean, again)


def test_adstock_accepts_a_non_unique_index():
    dates = pd.date_range("2024-01-01", periods=5)
    frame = pd.concat(
        [
            pd.DataFrame({"geo": g, "date": dates, "channel": "c", "impressions": 10.0})
            for g in ("g1", "g2")
        ],
        ignore_index=True,
    )
    frame.index = np.r_[np.arange(5), np.arange(5)]
    out = calibrated_adstock(frame, "geo", "date", "channel", "impressions", 0.5)
    expected = [10.0, 15.0, 17.5, 18.75, 19.375]
    for geo in ("g1", "g2"):
        assert out.loc[out["geo"] == geo, "exposure"].round(4).tolist() == expected


def test_assignment_refuses_a_market_listed_in_two_cells():
    assignment = pd.DataFrame({"geo": ["a", "a", "b", "c"], "cell": ["1", "control", "control", "2"]})
    with pytest.raises(ValueError, match="more than one cell"):
        design_from_assignment(assignment, ["A", "B"], [["A"], ["B"]])

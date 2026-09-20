"""Diagnostics: Hedges g, type-correct tests, gates, TBR sensitivity."""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from causalgeo import (
    AssignmentMap,
    GeoExperiment,
    Intervention,
    MetricSpec,
    diagnose,
)
from causalgeo.metric_tests import contingency_table, hedges_g


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
                    "closes": 10 + int(is_t) + rng.integers(0, 3),
                    "leads": 50,
                }
            )
    return pd.DataFrame(rows), treat_start


def test_rate_spec_requires_successes_and_trials():
    with pytest.raises(ValueError, match="successes"):
        MetricSpec("close_rate", kind="rate")


def test_categorical_spec_requires_counts():
    with pytest.raises(ValueError, match="categories"):
        MetricSpec("stage", kind="categorical")


def test_balanced_panel_does_not_fail_gates():
    treat = [f"G{i}" for i in range(8)]
    ctrl = [f"C{i}" for i in range(8)]
    data, t0 = _kpi_panel(-4.0, treat, ctrl)
    exp = GeoExperiment(data, AssignmentMap.from_sets(treat, ctrl), t0, spend="spend")
    report = exp.diagnose(
        [MetricSpec("conversions", kind="continuous", primary=True)],
        Intervention.GO_DARK,
    )
    assert "fail" not in set(report.gates["status"])
    assert "hedges_g" in report.balance.columns
    assert report.power.iloc[0]["mde_80"] > 0
    assert set(report.sensitivity["method"]) == {"tbr", "sc", "twfe"}
    assert "p_bh" in report.metric_tests.columns


def test_imbalanced_pretest_warns_on_hedges_g():
    treat = [f"G{i}" for i in range(8)]
    ctrl = [f"C{i}" for i in range(8)]
    data, t0 = _kpi_panel(0.0, treat, ctrl)
    data = data.copy()
    mask = (data["location"].isin(treat)) & (data["date"] < t0)
    data.loc[mask, "conversions"] = data.loc[mask, "conversions"] + 15.0
    report = diagnose(
        GeoExperiment(data, AssignmentMap.from_sets(treat, ctrl), t0, spend="spend"),
        [MetricSpec("conversions", kind="continuous", primary=True)],
    )
    assert "pretest_imbalance" in report.warnings
    assert report.gates.loc[report.gates["check"] == "smd", "status"].iloc[0] == "warn"


def test_chi2_matches_scipy_on_known_table():
    frame = pd.DataFrame(
        {
            "arm": ["treatment"] * 3 + ["control"] * 3,
            "stage": ["a", "b", "c", "a", "b", "c"],
            "n": [10, 20, 30, 12, 18, 28],
        }
    )
    table, result = contingency_table(frame, "arm", "stage", "n")
    chi2, p, _, _ = stats.chi2_contingency(table.to_numpy())
    assert result["p_raw"] == pytest.approx(p)
    assert result["chi2"] == pytest.approx(chi2)
    assert result["method"] == "chi2"


def test_fisher_when_expected_cell_small():
    frame = pd.DataFrame(
        {
            "arm": ["treatment", "treatment", "control", "control"],
            "stage": ["yes", "no", "yes", "no"],
            "n": [1, 10, 8, 1],
        }
    )
    _, result = contingency_table(frame, "arm", "stage", "n")
    assert result["method"] == "fisher"
    assert result["min_expected"] < 5


def test_rate_and_categorical_on_experiment():
    treat = [f"G{i}" for i in range(6)]
    ctrl = [f"C{i}" for i in range(6)]
    data, t0 = _kpi_panel(0.0, treat, ctrl)
    long_rows = []
    for _, row in data.iterrows():
        arm_boost = 3 if row["location"] in treat else 0
        long_rows.append({**row.to_dict(), "stage": "app", "n": 5 + arm_boost})
        long_rows.append({**{k: row[k] for k in data.columns}, "stage": "web", "n": 7})
    long = pd.DataFrame(long_rows)
    exp = GeoExperiment(long, AssignmentMap.from_sets(treat, ctrl), t0, spend="spend")
    report = exp.diagnose(
        [
            MetricSpec("conversions", kind="continuous", primary=True),
            MetricSpec("close_rate", kind="rate", successes="closes", trials="leads"),
            MetricSpec("funnel", kind="categorical", categories="stage", counts="n"),
        ]
    )
    assert "funnel" in report.contingency
    assert "close_rate" in report.contingency
    assert (report.metric_tests["estimand"] == "association_not_att").any()


def test_hedges_g_zero_when_identical():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    assert hedges_g(x, x) == pytest.approx(0.0)


def test_sensitivity_recovers_planted_att():
    treat = [f"G{i}" for i in range(8)]
    ctrl = [f"C{i}" for i in range(8)]
    data, t0 = _kpi_panel(-4.0, treat, ctrl)
    report = GeoExperiment(
        data, AssignmentMap.from_sets(treat, ctrl), t0, spend="spend"
    ).diagnose([MetricSpec("conversions", kind="continuous", primary=True)])
    for _, row in report.sensitivity.iterrows():
        assert row["lift"] == pytest.approx(-4.0, abs=0.6)

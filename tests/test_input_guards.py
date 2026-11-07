"""Wrong inputs raise, and a rate is tested as a rate."""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from causalgeo import AssignmentMap, GeoExperiment, Intervention, MetricSpec, diagnose
from causalgeo.allocation import prevented_effective_dose
from causalgeo.design import design_concurrent_test
from causalgeo.metric_tests import contingency_table


def test_duplicate_impression_rows_are_rejected():
    geos = pd.DataFrame({"geo": ["g1", "g2"], "stratum": ["s", "s"]})
    design = design_concurrent_test(geos, ["A"], [["A"]], control_share=0.5, seed=1)
    design.assignment["geo"] = ["g1", "g2"]
    design.assignment["cell"] = ["1", "control"]
    design.assignment["role"] = ["test", "control"]
    impressions = pd.DataFrame(
        [
            {"geo": "g1", "date": "2024-01-01", "channel": "A", "impressions": 10.0},
            {"geo": "g1", "date": "2024-01-01", "channel": "A", "impressions": 10.0},
            {"geo": "g2", "date": "2024-01-01", "channel": "A", "impressions": 10.0},
        ]
    )
    with pytest.raises(ValueError, match="more than one row"):
        prevented_effective_dose(impressions, design, treatment_start="2024-01-08")


def test_one_arm_contingency_has_no_p_value():
    frame = pd.DataFrame(
        {"arm": ["treatment"] * 4, "cat": ["a", "a", "b", "b"], "n": [10, 10, 3, 3]}
    )
    with pytest.warns(UserWarning, match="two groups"):
        _, stats = contingency_table(frame, "arm", "cat", "n")
    assert np.isnan(stats["p_raw"])


def test_missing_stratum_and_colliding_ids_are_rejected():
    geos = pd.DataFrame({"geo": ["a", "b", "c", "d"], "stratum": ["x", "x", np.nan, np.nan]})
    with pytest.raises(ValueError, match="missing values"):
        design_concurrent_test(geos, ["A"], [["A"]], control_share=0.5, seed=0)
    with pytest.raises(ValueError, match="missing id"):
        AssignmentMap.from_frame(pd.DataFrame({"location": [np.nan, "b"], "role": ["test", "control"]}))
    with pytest.raises(ValueError, match="number and a string"):
        AssignmentMap.from_frame(pd.DataFrame({"location": [1, "1"], "role": ["test", "test"]}))


def _panel():
    rng = np.random.default_rng(2)
    start = datetime(2026, 1, 1)
    dates = [start + timedelta(days=i) for i in range(30)]
    treat = [f"G{i}" for i in range(6)]
    ctrl = [f"C{i}" for i in range(6)]
    rows = []
    for geo in treat + ctrl:
        is_t = geo in treat
        for t, day in enumerate(dates):
            trials = (100 + 10 * t) if is_t else 80.0
            rate = 0.2 + float(rng.normal(0, 0.002))
            rows.append(
                {
                    "location": geo,
                    "date": day,
                    "conversions": 50 + float(rng.normal(0, 1)),
                    "successes": rate * trials,
                    "trials": trials,
                    "noise": (3 * t if is_t else 0) + float(rng.normal(0, 0.2)),
                }
            )
    data = pd.DataFrame(rows)
    experiment = GeoExperiment(data, AssignmentMap.from_sets(treat, ctrl), dates[20])
    return experiment


def test_rate_pretrend_uses_the_rate_not_the_counts():
    report = diagnose(
        _panel(),
        [MetricSpec("cvr", kind="rate", successes="successes", trials="trials", primary=True)],
    )
    trend = report.pretest.iloc[0]
    assert abs(trend["pretrend_coef"]) < 0.05
    gate = report.gates.set_index("check").loc["pretrend"]
    assert gate["status"] == "pass"


def test_a_covariate_does_not_change_the_outcome_gate():
    experiment = _panel()
    primary = MetricSpec("conversions", primary=True)
    covariate = MetricSpec("noise", column="noise", role="covariate")
    alone = diagnose(experiment, [primary])
    both = diagnose(experiment, [primary, covariate])
    assert alone.gates.set_index("check").loc["pretrend", "status"] == both.gates.set_index("check").loc[
        "pretrend", "status"
    ]
    assert "noise" not in set(both.metric_tests["metric"])


def test_diagnose_rejects_an_unknown_counterfactual_and_uses_the_one_given():
    experiment = _panel()
    with pytest.raises(ValueError, match="counterfactual"):
        diagnose(experiment, counterfactual="not-a-method")
    report = diagnose(experiment, counterfactual="twfe")
    detail = report.gates.set_index("check").loc["placebo", "detail"]
    assert "twfe" in detail

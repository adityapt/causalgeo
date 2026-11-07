"""Edge cases and behavior from the third bug pass."""

import numpy as np
import pandas as pd
import pytest

from causalgeo import AssignmentMap, GeoExperiment, MetricSpec, anchored_channel_allocation
from causalgeo.calibrated import _adstock as calibrated_adstock
from causalgeo.metric_tests import contingency_table, rate_two_by_two, welch_mannwhitney
from causalgeo.simulate import simulate_panel


def _experiment(scale=1.0, n_treatment=6):
    data, start = simulate_panel(np.random.default_rng(5), n_treatment=n_treatment, n_control=12, att=-4.0)
    data = data.copy()
    treated = [f"T{i}" for i in range(n_treatment)]
    data.loc[data["location"].isin(treated), "conversions"] *= scale
    control = [f"C{i}" for i in range(n_treatment, n_treatment + 12)]
    return GeoExperiment(data, AssignmentMap.from_sets(treated, control), start), data, start


def test_percent_lift_is_relative_to_the_treated_counterfactual():
    exp, data, start = _experiment(scale=3.0)
    report = exp.analyze()
    post = pd.to_datetime(data["date"]) >= pd.to_datetime(start)
    treated_post = data.loc[post & data["location"].str.startswith("T"), "conversions"].mean()
    assert report.percent_lift == pytest.approx(report.lift / (treated_post - report.lift))


def test_twfe_uses_a_t_distribution_and_warns_with_few_treated_geos():
    exp, _, _ = _experiment(n_treatment=3)
    with pytest.warns(UserWarning, match="unreliable"):
        report = exp.analyze(counterfactual="twfe")
    assert report.ci_upper - report.lift == pytest.approx(report.lift - report.ci_lower)
    z_half_width = 1.96 * report.std_error
    assert (report.ci_upper - report.ci_lower) / 2 > z_half_width


def test_tbr_warns_when_the_pretest_is_short():
    data, start = simulate_panel(np.random.default_rng(1), n_treatment=4, n_control=10, att=-4.0)
    dates = pd.to_datetime(data["date"])
    keep = dates >= pd.to_datetime(start) - pd.Timedelta(days=9)
    short = data.loc[keep]
    assignment = AssignmentMap.from_sets([f"T{i}" for i in range(4)], [f"C{i}" for i in range(4, 14)])
    with pytest.warns(UserWarning, match="pretest has"):
        GeoExperiment(short, assignment, start).analyze()


def test_constant_equal_arrays_give_no_ks_p_value():
    out = welch_mannwhitney(np.array([2.0, 2.0, 2.0]), np.array([2.0, 2.0, 2.0]))
    assert np.isnan(out["ks_p"])


def test_contingency_refuses_missing_or_negative_counts():
    frame = pd.DataFrame({"arm": ["t", "t", "c", "c"], "cat": ["a", "b", "a", "b"], "n": [1.0, np.nan, 2.0, 3.0]})
    with pytest.raises(ValueError, match="missing"):
        contingency_table(frame, "arm", "cat", "n")
    with pytest.raises(ValueError, match="nonnegative"):
        contingency_table(frame.assign(n=[1.0, -2.0, 2.0, 3.0]), "arm", "cat", "n")


def test_rate_test_refuses_more_successes_than_trials():
    with pytest.raises(ValueError, match="between 0 and trials"):
        rate_two_by_two(15, 10, 3, 10)


def test_metric_spec_refuses_a_non_string_kind():
    with pytest.raises(ValueError, match="strings"):
        MetricSpec("x", kind=None)


def test_allocation_refuses_colliding_potency_keys_and_orphan_dose_cells():
    effects = pd.DataFrame([{"cell": "1", "coef": 5.0, "std_err": 1.0}])
    dose = pd.DataFrame([{"cell": "1", "channel": "1", "dose": 1.0}, {"cell": "1", "channel": "B", "dose": 2.0}])
    with pytest.raises(ValueError, match="more than once"):
        anchored_channel_allocation(effects, dose, potency={1: 1.0, "1": 3.0})
    orphan = pd.concat([dose, pd.DataFrame([{"cell": "2", "channel": "B", "dose": 1.0}])])
    with pytest.raises(ValueError, match="no bundle effect"):
        anchored_channel_allocation(effects, orphan)


def test_carryover_weakens_across_a_missing_day():
    frame = pd.DataFrame(
        {
            "geo": "g",
            "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-10", "2024-01-11"]),
            "channel": "c",
            "impressions": 100.0,
        }
    )
    out = calibrated_adstock(frame, "geo", "date", "channel", "impressions", 0.5)["exposure"].tolist()
    assert out[0] == pytest.approx(100.0)
    assert out[1] == pytest.approx(150.0)
    assert out[2] == pytest.approx(100.0 + 150.0 * 0.5**8)
    assert out[3] == pytest.approx(100.0 + out[2] * 0.5)

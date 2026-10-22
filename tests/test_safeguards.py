"""Defaults and diagnostics for the concurrent channel split."""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from causalgeo import bootstrap_channel_intervals, compare_calibration
from causalgeo.calibrated import disentangle
from causalgeo.design import ConcurrentDesign, design_concurrent_test
from causalgeo.identified import estimate_identified


def _design():
    geos = [f"g{i}" for i in range(12)]
    frame = pd.DataFrame({"geo": geos, "stratum": ["a"] * len(geos)})
    design = design_concurrent_test(
        frame,
        channels=["Facebook", "Search"],
        bundles=[["Facebook"], ["Search"]],
        control_share=0.34,
        seed=0,
    )
    return design


def _panel(design, search_from_facebook: float = 0.0):
    rng = np.random.default_rng(1)
    dates = pd.date_range("2025-01-01", periods=30, freq="D")
    treatment = pd.Timestamp("2025-01-21")
    cell_of = design.assignment.set_index("geo")["cell"].astype(str)
    outcomes = []
    impressions = []
    for geo, cell in cell_of.items():
        for day in dates:
            post = day >= treatment
            facebook = 0.0 if post and cell == "1" else 8.0
            search = search_from_facebook * facebook + (1 - search_from_facebook) * (
                0.0 if post and cell == "2" else 8.0
            )
            facebook += float(rng.normal(0, 0.05))
            search += float(rng.normal(0, 0.05))
            impressions.append({"geo": geo, "date": day, "channel": "Facebook", "impressions": facebook})
            impressions.append({"geo": geo, "date": day, "channel": "Search", "impressions": max(search, 0)})
            outcomes.append(
                {
                    "geo": geo,
                    "date": day,
                    "sales": 0.4 * facebook + 0.2 * max(search, 0) + float(rng.normal(0, 0.05)),
                }
            )
    return pd.DataFrame(outcomes), pd.DataFrame(impressions)


def test_a_missing_impression_record_is_not_stored_as_zero_exposure():
    design = _design()
    outcomes, impressions = _panel(design)
    missing = impressions.drop(index=impressions.index[0])
    recorded_zero = impressions.copy()
    recorded_zero.loc[recorded_zero.index[0], "impressions"] = 0.0
    start = "2025-01-21"
    identified = estimate_identified(design, outcomes, outcome="sales", treatment_start=start)
    from causalgeo.calibrated import fit_channel_model

    withheld_missing = fit_channel_model(
        outcomes, missing, identified, design, start, "sales", calibrate=False
    )
    withheld_zero = fit_channel_model(
        outcomes, recorded_zero, identified, design, start, "sales", calibrate=False
    )
    assert not np.allclose(
        withheld_missing.channels["coef"].fillna(0),
        withheld_zero.channels["coef"].fillna(0),
    )


def test_a_nan_impression_is_not_filled_with_zero():
    design = _design()
    outcomes, impressions = _panel(design)
    blank = impressions.copy()
    blank.loc[blank.index[0], "impressions"] = np.nan
    recorded_zero = impressions.copy()
    recorded_zero.loc[recorded_zero.index[0], "impressions"] = 0.0
    start = "2025-01-21"
    identified = estimate_identified(design, outcomes, outcome="sales", treatment_start=start)
    from causalgeo.calibrated import fit_channel_model

    with pytest.warns(UserWarning, match="missing impressions"):
        withheld_blank = fit_channel_model(
            outcomes, blank, identified, design, start, "sales", calibrate=False
        )
    withheld_zero = fit_channel_model(
        outcomes, recorded_zero, identified, design, start, "sales", calibrate=False
    )
    assert not np.allclose(
        withheld_blank.channels["coef"].fillna(0),
        withheld_zero.channels["coef"].fillna(0),
    )


def test_a_channel_never_turned_off_is_not_labeled_separate():
    geos = [f"g{i}" for i in range(12)]
    design = design_concurrent_test(
        pd.DataFrame({"geo": geos, "stratum": ["a"] * len(geos)}),
        channels=["Facebook", "Search", "Display"],
        bundles=[["Facebook"], ["Search"]],
        control_share=0.34,
        seed=0,
    )
    rng = np.random.default_rng(2)
    dates = pd.date_range("2025-01-01", periods=30, freq="D")
    treatment = pd.Timestamp("2025-01-21")
    cell_of = design.assignment.set_index("geo")["cell"].astype(str)
    outcomes = []
    impressions = []
    for geo, cell in cell_of.items():
        for day in dates:
            post = day >= treatment
            facebook = 0.0 if post and cell == "1" else 8.0
            search = 0.0 if post and cell == "2" else 8.0
            display = 4.0 + float(rng.normal(0, 0.1))
            facebook += float(rng.normal(0, 0.05))
            search += float(rng.normal(0, 0.05))
            impressions.append({"geo": geo, "date": day, "channel": "Facebook", "impressions": facebook})
            impressions.append({"geo": geo, "date": day, "channel": "Search", "impressions": max(search, 0)})
            impressions.append({"geo": geo, "date": day, "channel": "Display", "impressions": display})
            outcomes.append(
                {
                    "geo": geo,
                    "date": day,
                    "sales": 0.4 * facebook + 0.2 * max(search, 0) + 0.5 * display,
                }
            )
    result = disentangle(
        design,
        pd.DataFrame(outcomes),
        pd.DataFrame(impressions),
        outcome="sales",
        treatment_start="2025-01-21",
        check_stability=False,
    )
    channels = result.channels.set_index("channel")
    assert channels.loc["Display", "status"] == "not tested"
    assert np.isnan(channels.loc["Display", "coef"])
    assert "never turned off" in channels.loc["Display", "reading"]
    assert abs(channels.loc["Facebook", "coef"] - 0.4) < 0.05


def test_bootstrap_respects_a_custom_market_column():
    design = _design()
    outcomes, impressions = _panel(design)
    renamed = ConcurrentDesign(
        assignment=design.assignment.rename(columns={"geo": "market"}),
        schedule=design.schedule,
        separation=design.separation,
    )
    intervals = bootstrap_channel_intervals(
        renamed,
        outcomes.rename(columns={"geo": "market"}),
        impressions.rename(columns={"geo": "market"}),
        outcome="sales",
        treatment_start="2025-01-21",
        n_boot=4,
        seed=0,
        geo_col="market",
        check_stability=False,
    )
    assert set(intervals["channel"]) >= {"Facebook", "Search"}


def test_a_slope_that_moves_with_the_calibration_weight_is_marked_unstable():
    design = _design()
    rng = np.random.default_rng(0)
    dates = pd.date_range("2025-01-01", periods=80, freq="D")
    treatment = pd.Timestamp("2025-02-20")
    cell_of = design.assignment.set_index("geo")["cell"].astype(str)
    off = {"1": {"Facebook"}, "2": {"Search"}}
    outcomes = []
    impressions = []
    for geo, cell in cell_of.items():
        for day in dates:
            facebook = 10.0 + float(rng.normal())
            search = 10.0 + float(rng.normal())
            post = day >= treatment
            if post and "Facebook" in off.get(cell, set()):
                facebook = 0.0
            if post and "Search" in off.get(cell, set()):
                search = 0.0
            # Before the test, Facebook and sales share a shock the experiment does not repeat.
            sales = (3.0 * facebook if not post or cell == "control" else 0.2 * facebook) + 0.2 * search
            impressions.append({"geo": geo, "date": day, "channel": "Facebook", "impressions": facebook})
            impressions.append({"geo": geo, "date": day, "channel": "Search", "impressions": search})
            outcomes.append({"geo": geo, "date": day, "sales": sales + float(rng.normal(0, 0.2))})
    result = disentangle(
        design,
        pd.DataFrame(outcomes),
        pd.DataFrame(impressions),
        outcome="sales",
        treatment_start="2025-02-20",
    )
    readings = " ".join(result.channels["reading"].astype(str))
    assert "unstable" in readings


def test_calibration_pulls_the_bundle_gap_below_the_observational_fit():
    design = _design()
    outcomes, impressions = _panel(design)
    identified = estimate_identified(
        design, outcomes, outcome="sales", treatment_start="2025-01-21"
    )
    compared = compare_calibration(
        outcomes, impressions, identified, design, "2025-01-21", "sales"
    )
    assert compared["calibrated_bundle_gap"].iloc[0] <= compared["observational_bundle_gap"].iloc[0]


def test_bootstrap_intervals_refit_whole_market_histories():
    design = _design()
    outcomes, impressions = _panel(design)
    intervals = bootstrap_channel_intervals(
        design,
        outcomes,
        impressions,
        outcome="sales",
        treatment_start="2025-01-21",
        n_boot=8,
        seed=0,
        adjust_exposure=False,
    )
    reported = intervals.dropna(subset=["ci_lower"])
    assert not reported.empty
    assert (reported["ci_lower"] < reported["ci_upper"]).all()
    assert (reported["draws"] >= 2).all()


def test_a_small_shift_does_not_trigger_when_the_bundle_effect_is_near_zero():
    design = _design()
    dates = [datetime(2025, 1, 1) + timedelta(days=i) for i in range(20)]
    outcomes = []
    impressions = []
    for geo in design.assignment["geo"]:
        for day in dates:
            outcomes.append({"geo": geo, "date": day, "sales": 100 + np.sin(day.day)})
            for channel in ("Facebook", "Search"):
                impressions.append(
                    {"geo": geo, "date": day, "channel": channel, "impressions": 10.0}
                )
    outcomes = pd.DataFrame(outcomes)
    impressions = pd.DataFrame(impressions)
    result = disentangle(
        design,
        outcomes,
        impressions,
        outcome="sales",
        treatment_start="2025-01-15",
        check_stability=False,
    )
    assert not bool(result.exposure["adjust"].any())

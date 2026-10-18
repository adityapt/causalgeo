"""Bundle effects split into channels only when impressions are not copies."""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from causalgeo.design import design_concurrent_test
from causalgeo.calibrated import disentangle


def _panel(seed: int = 0):
    rng = np.random.default_rng(seed)
    geos = [f"g{i}" for i in range(40)]
    # Fixed cells so the bundles overlap the way the regression needs.
    cell = {geo: "control" for geo in geos}
    for geo in geos[0:10]:
        cell[geo] = "1"
    for geo in geos[10:20]:
        cell[geo] = "2"
    for geo in geos[20:30]:
        cell[geo] = "3"
    start = datetime(2025, 1, 1)
    dates = [start + timedelta(days=i) for i in range(60)]
    treatment = datetime(2025, 1, 31)
    beta = {"Facebook": 0.5, "TikTok": 0.2, "Search": 0.1}
    off = {
        "1": {"Facebook", "TikTok"},
        "2": {"Facebook", "Search"},
        "3": {"TikTok", "Search"},
        "control": set(),
    }
    outcome_rows = []
    impression_rows = []
    for geo in geos:
        level = rng.normal(0, 0.2)
        for day in dates:
            post = day >= treatment
            shock = rng.normal(0, 0.05)
            for channel, slope in beta.items():
                base = 10 + shock
                dark = post and channel in off[cell[geo]]
                impressions = 0.0 if dark else base
                impression_rows.append(
                    {"geo": geo, "date": day, "channel": channel, "impressions": impressions}
                )
            sales = level + sum(
                slope * (0.0 if post and channel in off[cell[geo]] else 10 + shock)
                for channel, slope in beta.items()
            )
            sales += rng.normal(0, 0.05)
            outcome_rows.append({"geo": geo, "date": day, "sales": sales})
    # The design object only needs the schedule and the assignment the estimator uses.
    frame = pd.DataFrame({"geo": geos, "stratum": ["a"] * 40})
    design = design_concurrent_test(
        frame,
        channels=["Facebook", "TikTok", "Search"],
        bundles=[["Facebook", "TikTok"], ["Facebook", "Search"], ["TikTok", "Search"]],
        control_share=0.25,
        seed=0,
    )
    design.assignment["geo"] = geos
    design.assignment["cell"] = [cell[geo] for geo in geos]
    design.assignment["role"] = ["control" if cell[geo] == "control" else "test" for geo in geos]
    return design, pd.DataFrame(outcome_rows), pd.DataFrame(impression_rows), beta


def test_overlapping_bundles_recover_channel_effects():
    design, outcomes, impressions, beta = _panel()
    result = disentangle(
        design, outcomes, impressions, outcome="sales", treatment_start="2025-01-31"
    )
    fitted = result.channels.set_index("channel")
    for channel, slope in beta.items():
        assert fitted.loc[channel, "status"] == "separate"
        assert fitted.loc[channel, "coef"] == pytest.approx(slope, abs=0.05)
    daily = result.daily
    facebook = daily.loc[daily["channel"] == "Facebook"]
    assert np.allclose(facebook["impact"], facebook["exposure"] * fitted.loc["Facebook", "coef"])


def test_channels_that_always_move_together_stay_combined():
    design = design_concurrent_test(
        pd.DataFrame({"geo": [f"g{i}" for i in range(20)], "stratum": ["a"] * 20}),
        channels=["Facebook", "TikTok"],
        bundles=[["Facebook", "TikTok"]],
        control_share=0.5,
        seed=0,
    )
    dates = pd.date_range("2025-01-01", periods=40, freq="D")
    rows = []
    impressions = []
    cell_of = design.assignment.set_index("geo")["cell"]
    for geo, cell in cell_of.items():
        for day in dates:
            post = day >= pd.Timestamp("2025-01-21")
            dark = post and cell == "1"
            impressions.append({"geo": geo, "date": day, "channel": "Facebook", "impressions": 0.0 if dark else 10.0})
            impressions.append({"geo": geo, "date": day, "channel": "TikTok", "impressions": 0.0 if dark else 10.0})
            rows.append({"geo": geo, "date": day, "sales": 0.0 if dark else 7.0})
    result = disentangle(
        design,
        pd.DataFrame(rows),
        pd.DataFrame(impressions),
        outcome="sales",
        treatment_start="2025-01-21",
    )
    status = result.channels.set_index("channel")["status"]
    assert status["Facebook"] == "tied"
    assert status["TikTok"] == "tied"
    assert "Facebook" not in set(result.daily["channel"])
    assert "TikTok" not in set(result.daily["channel"])

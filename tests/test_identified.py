"""The estimator must refuse a channel the design cannot name."""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from causalgeo.design import design_concurrent_test
from causalgeo.identified import NotIdentified, channel_effect, estimate_identified


def _geos(n: int = 30) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "geo": [f"g{i}" for i in range(n)],
            "stratum": (["small"] * 10 + ["medium"] * 10 + ["large"] * 10)[:n],
        }
    )


def _panel(design, effects: dict[str, float], seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    start = datetime(2025, 1, 1)
    dates = [start + timedelta(days=i) for i in range(40)]
    treatment_start = datetime(2025, 1, 21)
    cell_of = design.assignment.set_index("geo")["cell"]
    rows = []
    for geo, cell in cell_of.items():
        level = rng.normal(0, 1)
        for t, day in enumerate(dates):
            post = int(day >= treatment_start)
            lift = effects.get(cell, 0.0) * post
            rows.append(
                {
                    "geo": geo,
                    "date": day,
                    "sales": level + 0.05 * t + lift + rng.normal(0, 0.2),
                    "spend": rng.normal(10, 1),
                }
            )
    return pd.DataFrame(rows)


def test_tied_channel_has_no_number_and_the_bundle_does():
    design = design_concurrent_test(
        _geos(),
        channels=["Facebook", "TikTok", "Search"],
        bundles=[["Facebook", "TikTok"], ["Facebook", "TikTok"], ["Search"]],
        control_share=0.4,
        seed=0,
    )
    metrics = _panel(design, {"1": -4.0, "2": -4.0, "3": -2.0})
    result = estimate_identified(design, metrics, outcome="sales", treatment_start="2025-01-21")

    search = result.channels.set_index("channel").loc["Search"]
    assert search["status"] == "separate"
    assert search["coef"] == pytest.approx(-2.0, abs=0.25)
    assert channel_effect(result, "Search") == pytest.approx(-2.0, abs=0.25)

    facebook = result.channels.set_index("channel").loc["Facebook"]
    assert facebook["status"] == "tied"
    assert pd.isna(facebook["coef"])
    with pytest.raises(NotIdentified, match="tied with TikTok"):
        channel_effect(result, "Facebook")

    bundles = set(result.cells["bundle"])
    assert "Facebook+TikTok" in bundles
    assert "Search" in bundles
    assert "spend" not in " ".join(result.estimate.formula.split())


def test_a_channel_turned_off_alone_is_reported():
    design = design_concurrent_test(
        _geos(),
        channels=["Facebook", "TikTok"],
        bundles=[["Facebook"], ["TikTok"], ["Facebook", "TikTok"]],
        control_share=0.4,
        seed=1,
    )
    metrics = _panel(design, {"1": -3.0, "2": -6.0, "3": -8.0})
    result = estimate_identified(design, metrics, outcome="sales", treatment_start="2025-01-21")
    assert channel_effect(result, "Facebook") == pytest.approx(-3.0, abs=0.3)
    assert channel_effect(result, "TikTok") == pytest.approx(-6.0, abs=0.3)
    bundle = result.cells.set_index("bundle").loc["Facebook+TikTok", "coef"]
    assert bundle == pytest.approx(-8.0, abs=0.35)


def test_spend_is_not_accepted_as_the_outcome_stand_in():
    design = design_concurrent_test(
        _geos(),
        channels=["Search"],
        bundles=[["Search"]],
        control_share=0.4,
        seed=0,
    )
    metrics = _panel(design, {"1": -1.0})
    with pytest.raises(ValueError, match="spend is not an outcome"):
        estimate_identified(design, metrics, outcome="spend", treatment_start="2025-01-21")

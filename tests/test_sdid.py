"""Synthetic difference-in-differences estimation and inference."""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from causalgeo.sdid import estimate_sdid


def _known_effect_panel(effect: float, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    geos = [f"C{i}" for i in range(10)] + ["T0"]
    dates = [datetime(2025, 1, 1) + timedelta(days=i) for i in range(30)]
    rows = []
    common = np.sin(np.arange(30) / 4) * 2 + np.arange(30) * 0.08
    control_loadings = rng.normal(1.0, 0.12, size=10)
    control_levels = rng.normal(20, 2, size=10)
    controls = np.array(
        [
            control_levels[i]
            + control_loadings[i] * common
            + rng.normal(0, 0.12, size=30)
            for i in range(10)
        ]
    )
    treated_untreated = (
        0.5 * controls[1] + 0.3 * controls[4] + 0.2 * controls[8] + 1.5
    )
    outcomes = np.vstack(
        [controls, treated_untreated + effect * (np.arange(30) >= 20)]
    )
    for i, geo in enumerate(geos):
        for t, date in enumerate(dates):
            rows.append(
                {
                    "location": geo,
                    "date": date,
                    "conversions": outcomes[i, t],
                    "is_treatment": int(geo == "T0"),
                    "is_post": int(t >= 20),
                }
            )
    return pd.DataFrame(rows)


def test_sdid_recovers_known_effect():
    report = estimate_sdid(
        _known_effect_panel(-5.0),
        conversions="conversions",
        location="location",
        date="date",
        placebo_replications=20,
        random_state=4,
    )

    assert report.method == "sdid"
    assert abs(report.lift - (-5.0)) < 0.5
    assert report.pre_r2 is not None
    assert report.pre_r2 > 0.95


def test_sdid_is_stable_when_outcome_units_are_larger():
    panel = _known_effect_panel(-5.0)
    panel["conversions"] = panel["conversions"] * 10.0

    report = estimate_sdid(
        panel,
        placebo_replications=10,
        random_state=4,
    )

    assert report.lift == pytest.approx(-50.0, abs=5.0)


def test_sdid_placebo_inference_covers_zero_when_effect_is_zero():
    report = estimate_sdid(
        _known_effect_panel(0.0, seed=7),
        conversions="conversions",
        location="location",
        date="date",
        placebo_replications=40,
        random_state=11,
    )

    assert report.cov_type == "placebo"
    assert np.isfinite(report.std_error)
    assert report.std_error > 0
    assert report.ci_lower <= 0 <= report.ci_upper

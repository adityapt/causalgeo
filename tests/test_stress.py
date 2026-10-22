"""Tests for the manuscript's misspecification stress-test helpers."""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

PAPER = Path(__file__).resolve().parents[1] / "paper"
sys.path.insert(0, str(PAPER))

import stress  # noqa: E402


def test_geometric_adstock_carries_exposure_forward():
    transformed = stress.geometric_adstock(np.array([10.0, 0.0, 0.0]), decay=0.5)
    assert transformed.tolist() == pytest.approx([10.0, 5.0, 2.5])


def test_hill_response_has_linear_slope_at_zero_and_half_saturates():
    response = stress.hill_response(
        np.array([0.0, 10.0]),
        slope=2.0,
        half_saturation=10.0,
    )
    assert response.tolist() == pytest.approx([0.0, 10.0])


def test_hill_response_rejects_a_nonpositive_half_saturation():
    with pytest.raises(ValueError, match="positive"):
        stress.hill_response(np.array([1.0]), slope=2.0, half_saturation=0.0)


def test_true_channel_contrasts_use_the_experimental_difference_in_differences():
    dates = pd.to_datetime(["2026-01-01", "2026-01-08"])
    signal = pd.DataFrame(
        {
            "geo": ["control", "control", "dark", "dark"],
            "date": [*dates, *dates],
            "Channel0": [10.0, 10.0, 10.0, 6.0],
        }
    )
    assignment = pd.Series({"control": "control", "dark": "1"}, name="cell")

    result = stress.true_channel_contrasts(
        signal,
        assignment,
        {"1": ["Channel0"]},
        treatment_start="2026-01-08",
    )

    assert result.to_dict(orient="records") == [
        {
            "cell": "1",
            "channel": "Channel0",
            "true_effect": -4.0,
        }
    ]

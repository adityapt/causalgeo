"""Proposition 99 is the public single-cell check, and the synthetic control must fit it."""

from pathlib import Path

import pandas as pd
import pytest

from causalgeo import AssignmentMap, GeoExperiment

PANEL = Path(__file__).parent / "data" / "california_prop99.csv"
START = "1989-01-01"


def _experiment() -> GeoExperiment:
    raw = pd.read_csv(PANEL, sep=";")
    data = raw.rename(columns={"State": "location", "PacksPerCapita": "conversions"})
    data["date"] = pd.to_datetime(data["Year"].astype(str) + "-01-01")
    donors = set(data["location"].unique()) - {"California"}
    return GeoExperiment(data, AssignmentMap.from_sets(["California"], donors), START)


def test_prop99_synthetic_control_beats_equal_weights():
    with pytest.warns(UserWarning, match="synthetic-control interval"):
        report = _experiment().analyze(counterfactual="sc")
    # Equal weights stop at a pre-period R^2 of about -0.99 and a lift of about -42.
    # A fitted simplex reaches R^2 0.98 and a lift near -20.
    assert report.pre_r2 == pytest.approx(0.979, abs=0.01)
    assert report.lift == pytest.approx(-19.5, abs=0.5)

"""Edge cases found in the package audit."""

import numpy as np
import pandas as pd
import pytest

from causalgeo import design_concurrent_test, experiment_only_split
from causalgeo.calibrated import _adstock
from causalgeo.design import design_from_assignment
from causalgeo.identified import IdentifiedEstimate, estimate_identified


def test_design_refuses_a_draw_with_no_control_markets():
    geos = pd.DataFrame({"geo": list("abcdef"), "stratum": list("uvwxyz")})
    with pytest.raises(ValueError, match="no control markets"):
        design_concurrent_test(geos, ["A", "B"], [["A"], ["B"]], control_share=0.4, seed=0)


def test_design_refuses_an_empty_cell():
    geos = pd.DataFrame({"geo": list("abcd"), "stratum": ["s"] * 4})
    with pytest.raises(ValueError, match="have no markets"):
        design_concurrent_test(
            geos, ["A", "B", "C"], [["A"], ["B"], ["C"]], control_share=0.5, seed=0
        )


def test_assignment_refuses_an_unknown_bundle_channel():
    assignment = pd.DataFrame({"geo": list("abcd"), "cell": ["control", "control", "1", "2"]})
    with pytest.raises(ValueError, match="not in the channel list"):
        design_from_assignment(assignment, ["A", "B"], [["A"], ["Bee"]])


def test_assignment_refuses_an_empty_cell():
    assignment = pd.DataFrame({"geo": list("abc"), "cell": ["control", "control", "1"]})
    with pytest.raises(ValueError, match="have no markets"):
        design_from_assignment(assignment, ["A", "B"], [["A"], ["B"]])


def test_estimate_names_a_cell_with_no_markets_in_the_metrics():
    assignment = pd.DataFrame({"geo": list("abcd"), "cell": ["control", "control", "1", "2"]})
    design = design_from_assignment(assignment, ["A", "B"], [["A"], ["B"]])
    dates = pd.date_range("2024-01-01", periods=6)
    metrics = pd.DataFrame(
        [{"geo": g, "date": d, "sales": float(i)} for g in "abc" for i, d in enumerate(dates)]
    )
    with pytest.raises(ValueError, match="cells \\['2'\\]"):
        estimate_identified(design, metrics, "sales", "2024-01-04")


def test_adstock_keeps_the_days_after_a_blank_impression():
    frame = pd.DataFrame(
        {
            "geo": "g",
            "date": pd.date_range("2024-01-01", periods=6),
            "channel": "c",
            "impressions": [10, 10, np.nan, 10, 10, 10.0],
        }
    )
    out = _adstock(frame, "geo", "date", "channel", "impressions", 0.5)["exposure"].tolist()
    assert np.isnan(out[2])
    assert out[:2] == [10.0, 15.0]
    # The blank adds nothing. Carryover from the days before it still decays forward.
    assert out[3] == pytest.approx(13.75)


def _one_cell_two_channels():
    geos = pd.DataFrame({"geo": [f"m{i}" for i in range(8)], "stratum": ["s"] * 8})
    design = design_concurrent_test(geos, ["A", "B"], [["A", "B"]], control_share=0.5, seed=0)
    cell_of = design.assignment.set_index("geo")["cell"]
    rows = []
    for geo in cell_of.index:
        for date in pd.date_range("2024-01-01", periods=6):
            dark = cell_of[geo] == "1" and date >= pd.Timestamp("2024-01-04")
            for channel, base in (("A", 100.0), ("B", 50.0)):
                rows.append(
                    {"geo": geo, "date": date, "channel": channel, "impressions": 0.0 if dark else base}
                )
    cells = pd.DataFrame(
        {
            "cell": ["1"],
            "bundle": ["A+B"],
            "coef": [-500.0],
            "std_err": [50.0],
            "p_value": [0.0],
            "ci_lower": [-600.0],
            "ci_upper": [-400.0],
        }
    )
    return design, pd.DataFrame(rows), IdentifiedEstimate(cells=cells, channels=pd.DataFrame(), estimate=None)


def test_holdout_only_split_leaves_unseparated_channels_blank():
    design, impressions, identified = _one_cell_two_channels()
    split = experiment_only_split(impressions, identified, design, "2024-01-04")
    assert split["coef"].isna().all()
    assert split["std_err"].isna().all()

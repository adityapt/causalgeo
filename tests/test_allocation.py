import numpy as np
import pandas as pd
import pytest

from causalgeo import (
    anchored_channel_allocation,
    design_from_assignment,
    potency_misspecification_ratio,
    potency_tie_ratio,
    prevented_effective_dose,
)


def test_allocation_preserves_the_bundle_total_and_score_ratios():
    dose = pd.DataFrame(
        [
            {"cell": "1", "channel": "Search", "dose": 1.0},
            {"cell": "1", "channel": "Social", "dose": 3.0},
        ]
    )
    result = anchored_channel_allocation(
        pd.DataFrame([{"cell": "1", "coef": 10.0, "std_err": 2.0}]),
        dose,
        potency={"Search": 2.0, "Social": 1.0},
    )

    channels = result.channels.set_index("channel")
    assert channels.loc["Search", "share"] == pytest.approx(0.4)
    assert channels.loc["Social", "share"] == pytest.approx(0.6)
    assert channels["allocation"].sum() == pytest.approx(10.0)
    assert channels.loc["Search", "allocation_std_err"] == pytest.approx(0.8)
    assert channels.loc["Search", "ci_lower"] == pytest.approx(4.0 - 1.96 * 0.8)
    assert "assumption-driven" in channels.loc["Search", "reading"]


def test_common_dose_rescaling_does_not_change_shares():
    base = pd.DataFrame(
        [
            {"cell": "1", "channel": "Search", "dose": 2.0},
            {"cell": "1", "channel": "Video", "dose": 5.0},
        ]
    )
    effects = pd.DataFrame([{"cell": "1", "coef": -8.0, "std_err": 1.0}])
    original = anchored_channel_allocation(effects, base)
    scaled = anchored_channel_allocation(effects, base.assign(dose=base["dose"] * 10))

    assert scaled.channels["share"].tolist() == pytest.approx(
        original.channels["share"].tolist()
    )
    assert scaled.channels["allocation"].sum() == pytest.approx(-8.0)


def test_potency_bounds_are_sharp_and_flip_when_the_total_is_negative():
    dose = pd.DataFrame(
        [
            {"cell": "1", "channel": "Search", "dose": 1.0},
            {"cell": "1", "channel": "Social", "dose": 1.0},
        ]
    )
    bounds = {"Search": (1.0, 2.0), "Social": (1.0, 2.0)}
    positive = anchored_channel_allocation(
        pd.DataFrame([{"cell": "1", "coef": 12.0, "std_err": 1.0}]),
        dose,
        potency={"Search": 1.0, "Social": 1.0},
        potency_bounds=bounds,
    )
    search = positive.channels.set_index("channel").loc["Search"]
    assert search["share_lower"] == pytest.approx(1 / 3)
    assert search["share_upper"] == pytest.approx(2 / 3)
    assert search["assumption_lower"] == pytest.approx(4.0)
    assert search["assumption_upper"] == pytest.approx(8.0)

    negative = anchored_channel_allocation(
        pd.DataFrame([{"cell": "1", "coef": -12.0, "std_err": 1.0}]),
        dose,
        potency_bounds=bounds,
    )
    dark = negative.channels.set_index("channel").loc["Search"]
    assert dark["assumption_lower"] == pytest.approx(-8.0)
    assert dark["assumption_upper"] == pytest.approx(-4.0)


def test_sampling_and_assumption_ranges_form_the_outer_envelope():
    result = anchored_channel_allocation(
        pd.DataFrame([{"cell": "1", "coef": 10.0, "std_err": 1.0}]),
        pd.DataFrame(
            [
                {"cell": "1", "channel": "A", "dose": 1.0},
                {"cell": "1", "channel": "B", "dose": 1.0},
            ]
        ),
        potency_bounds={"A": (1.0, 3.0), "B": (1.0, 3.0)},
    )
    channel = result.channels.set_index("channel").loc["A"]
    tau = [10 - 1.96, 10 + 1.96]
    shares = [0.25, 0.75]
    corners = [left * right for left in tau for right in shares]
    assert channel["envelope_lower"] == pytest.approx(min(corners))
    assert channel["envelope_upper"] == pytest.approx(max(corners))


def test_each_cell_reconciles_to_its_own_bundle():
    result = anchored_channel_allocation(
        pd.DataFrame(
            [
                {"cell": "1", "coef": 9.0, "std_err": 1.0},
                {"cell": "2", "coef": 4.0, "std_err": 0.5},
            ]
        ),
        pd.DataFrame(
            [
                {"cell": "1", "channel": "Search", "dose": 1.0},
                {"cell": "1", "channel": "Social", "dose": 2.0},
                {"cell": "2", "channel": "Video", "dose": 4.0},
            ]
        ),
    )
    totals = result.channels.groupby("cell")["allocation"].sum()
    assert totals.loc["1"] == pytest.approx(9.0)
    assert totals.loc["2"] == pytest.approx(4.0)


def test_misspecified_potency_has_the_exact_bias_ratio():
    dose = {"Search": 2.0, "Social": 4.0}
    true = {"Search": 1.0, "Social": 3.0}
    assumed = {"Search": 2.0, "Social": 1.0}
    ratio = potency_misspecification_ratio(dose, assumed, true)

    denominator = 1 * 2 + 3 * 4
    assumed_total = 2 * 2 + 1 * 4
    assert ratio.loc["Search"] == pytest.approx((2 / 1) * (denominator / assumed_total))
    assert ratio.loc["Social"] == pytest.approx((1 / 3) * (denominator / assumed_total))
    assert potency_tie_ratio(2.0, 4.0) == pytest.approx(2.0)


def test_allocation_rejects_unusable_scores():
    effects = pd.DataFrame([{"cell": "1", "coef": 5.0, "std_err": 1.0}])
    dose = pd.DataFrame(
        [
            {"cell": "1", "channel": "A", "dose": 1.0},
            {"cell": "1", "channel": "B", "dose": 0.0},
        ]
    )
    with pytest.raises(ValueError, match="positive"):
        anchored_channel_allocation(effects, dose, potency={"A": -1.0, "B": 1.0})
    with pytest.raises(ValueError, match="positive"):
        anchored_channel_allocation(
            effects.assign(std_err=0),
            dose,
        )
    with pytest.raises(ValueError, match="dose"):
        anchored_channel_allocation(
            effects,
            dose.assign(dose=[-1.0, 1.0]),
        )


def test_prevented_dose_uses_quality_reach_and_saturation():
    assignment = pd.DataFrame(
        {"geo": ["dark", "control"], "cell": ["1", "control"]}
    )
    design = design_from_assignment(
        assignment,
        channels=["Search", "Video"],
        bundles=[["Search"]],
    )
    rows = []
    for geo in ("dark", "control"):
        for date, post in (("2026-08-01", False), ("2026-08-08", True)):
            search_impressions = 0.0 if geo == "dark" and post else 10.0
            rows.append(
                {
                    "geo": geo,
                    "date": date,
                    "channel": "Search",
                    "impressions": search_impressions,
                    "quality": 0.5,
                    "reach": 100.0 if search_impressions else 0.0,
                    "frequency": 1.0,
                }
            )
            rows.append(
                {
                    "geo": geo,
                    "date": date,
                    "channel": "Video",
                    "impressions": 99.0,
                    "quality": 1.0,
                    "reach": 99.0,
                    "frequency": 9.0,
                }
            )
    panel = pd.DataFrame(rows)

    quality = prevented_effective_dose(
        panel,
        design,
        "2026-08-08",
        quality_col="quality",
    )
    assert quality.loc[quality["channel"].eq("Search"), "dose"].item() == pytest.approx(5.0)
    assert quality.loc[quality["channel"].eq("Video"), "dose"].item() == pytest.approx(0.0)

    reach = prevented_effective_dose(
        panel,
        design,
        "2026-08-08",
        reach_col="reach",
        frequency_col="frequency",
        frequency_decay=np.log(2),
    )
    assert reach.loc[reach["channel"].eq("Search"), "dose"].item() == pytest.approx(50.0)

    saturated = prevented_effective_dose(
        panel.drop(columns=["quality", "reach", "frequency"]),
        design,
        "2026-08-08",
        hill_slope={"Search": 1.0},
        hill_half={"Search": 10.0},
    )
    assert saturated.loc[saturated["channel"].eq("Search"), "dose"].item() == pytest.approx(0.5)

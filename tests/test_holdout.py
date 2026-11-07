"""Tests for concurrent geo test design."""

import numpy as np
import pandas as pd
import pytest

from causalgeo.design import design_concurrent_test


def _geos() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "geo": [f"g{i}" for i in range(30)],
            "stratum": ["small"] * 10 + ["medium"] * 10 + ["large"] * 10,
        }
    )


def _design(seed: int = 0):
    return design_concurrent_test(
        _geos(),
        channels=["Facebook", "TikTok", "Search"],
        bundles=[["Facebook", "TikTok"], ["Facebook", "TikTok"], ["Search"]],
        control_share=0.4,
        seed=seed,
    )


def test_spare_markets_rotate_across_strata_instead_of_piling_into_cell_one():
    geos = pd.DataFrame(
        {"geo": [f"g{i}" for i in range(60)], "stratum": [f"s{i // 6}" for i in range(60)]}
    )
    design = design_concurrent_test(
        geos,
        channels=["A", "B", "C"],
        bundles=[["A"], ["B"], ["C"]],
        control_share=1 / 3,
        seed=0,
    )
    counts = design.assignment.loc[design.assignment["role"] == "test", "cell"].value_counts()
    # 10 strata, 4 test markets, 3 cells. Starting each stratum at cell 1
    # deals 20/10/10. Rotating the spare market deals 14/13/13.
    assert counts.max() - counts.min() <= 1
    assert counts.sum() == 40


def test_each_stratum_is_split_evenly_across_cells():
    design = _design()
    test = design.assignment[design.assignment["role"] == "test"]
    counts = test.groupby(["stratum", "cell"]).size().unstack("cell")
    # 10 markets, 40% control -> 4 control, 6 test, 3 cells, so 2 per cell.
    assert (counts == 2).all().all()


def test_control_share_is_applied_inside_every_stratum():
    design = _design()
    for _, group in design.assignment.groupby("stratum"):
        assert (group["role"] == "control").mean() == pytest.approx(0.4)


def test_same_seed_gives_the_same_markets():
    first = _design(seed=3).assignment
    second = _design(seed=3).assignment
    pd.testing.assert_frame_equal(first, second)


def test_a_channel_off_alone_is_separate_and_a_pair_stays_tied():
    design = _design()
    status = design.separation.set_index("channel")["status"]
    assert status["Search"] == "separate"
    assert status["Facebook"] == "tied"
    assert status["TikTok"] == "tied"
    tied = design.separation.set_index("channel")["tied_with"]
    assert tied["Facebook"] == "TikTok"
    assert tied["TikTok"] == "Facebook"


def test_cells_that_split_a_pair_mark_both_channels_separate():
    design = design_concurrent_test(
        _geos(),
        channels=["Facebook", "TikTok"],
        bundles=[["Facebook"], ["TikTok"], ["Facebook", "TikTok"]],
        control_share=0.4,
        seed=1,
    )
    assert set(design.separation["status"]) == {"separate"}


def test_unknown_channel_in_a_bundle_is_rejected():
    with pytest.raises(ValueError, match="not in the channel list"):
        design_concurrent_test(
            _geos(),
            channels=["Search"],
            bundles=[["Facebook"]],
        )

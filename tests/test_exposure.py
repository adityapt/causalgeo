"""A still-on channel is controlled for only when its exposure shift is large."""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from causalgeo.design import design_concurrent_test
from causalgeo.calibrated import disentangle
from causalgeo.identified import estimate_identified


def _world():
    rng = np.random.default_rng(0)
    geos = [f"g{i}" for i in range(40)]
    cell = {geo: "control" for geo in geos}
    for geo in geos[0:10]:
        cell[geo] = "1"
    for geo in geos[10:20]:
        cell[geo] = "2"
    for geo in geos[20:30]:
        cell[geo] = "3"
    frame = pd.DataFrame({"geo": geos, "stratum": ["a"] * 40})
    design = design_concurrent_test(
        frame,
        channels=["Channel0", "Channel1", "Channel2"],
        bundles=[["Channel0", "Channel1"], ["Channel0", "Channel2"], ["Channel1", "Channel2"]],
        control_share=0.25,
        seed=0,
    )
    design.assignment["geo"] = geos
    design.assignment["cell"] = [cell[geo] for geo in geos]
    design.assignment["role"] = ["control" if cell[geo] == "control" else "test" for geo in geos]
    start = datetime(2025, 1, 1)
    dates = [start + timedelta(days=i) for i in range(40)]
    treatment = datetime(2025, 1, 21)
    beta = {"Channel0": 0.5, "Channel1": 0.4, "Channel2": 0.1}
    off = {
        "1": {"Channel0", "Channel1"},
        "2": {"Channel0", "Channel2"},
        "3": {"Channel1", "Channel2"},
        "control": set(),
    }
    outcomes = []
    impressions = []
    for geo in geos:
        level = rng.normal(0, 0.1)
        for day in dates:
            post = day >= treatment
            dark = off[cell[geo]] if post else set()
            sales = level
            for channel, slope in beta.items():
                impressions_value = 10.0 + float(rng.normal(0, 1))
                # Channel1 falls only in cell 2 during the test, even though it stays on there.
                if channel == "Channel1" and cell[geo] == "2" and post:
                    impressions_value = 4.0
                if channel in dark:
                    impressions_value = 0.0
                sales += slope * impressions_value
                impressions.append(
                    {"geo": geo, "date": day, "channel": channel, "impressions": impressions_value}
                )
            outcomes.append({"geo": geo, "date": day, "sales": sales})
    return design, pd.DataFrame(outcomes), pd.DataFrame(impressions)


def test_only_the_cell_with_a_large_exposure_shift_is_adjusted():
    design, outcomes, impressions = _world()
    plain = estimate_identified(design, outcomes, outcome="sales", treatment_start="2025-01-21")
    result = disentangle(design, outcomes, impressions, outcome="sales", treatment_start="2025-01-21")
    check = result.exposure.set_index(["cell", "channel"])
    assert bool(check.loc[("2", "Channel1"), "adjust"])
    assert not bool(check.loc[("1", "Channel2"), "adjust"])
    assert not bool(check.loc[("3", "Channel0"), "adjust"])

    plain_cells = plain.cells.set_index("cell")["coef"]
    adjusted = result.bundles.set_index("cell")["measured_effect"]
    # Cell 2's unadjusted number contains the Channel1 drop. The adjusted number does not.
    assert abs(adjusted["2"] - plain_cells["2"]) > 1
    assert abs(adjusted["1"] - plain_cells["1"]) < 1e-8
    assert abs(adjusted["3"] - plain_cells["3"]) < 1e-8

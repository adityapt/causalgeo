"""Walk through one concurrent holdout on a small synthetic panel.

Run:
    python examples/concurrent_holdout.py

The panel has three channels and three overlapping cells. Search is an exact
copy of Facebook, so the fit refuses to give that pair two slopes. Display
is a distinct path and is reported, with a reading that says the slope is
model-assisted.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from causalgeo import design_concurrent_test, disentangle


def main() -> None:
    markets = pd.DataFrame(
        {"geo": [f"m{i}" for i in range(16)], "stratum": ["north"] * 8 + ["south"] * 8}
    )
    design = design_concurrent_test(
        markets,
        channels=["Facebook", "Search", "Display"],
        bundles=[["Facebook", "Search"], ["Facebook", "Display"], ["Search", "Display"]],
        control_share=0.25,
        seed=1,
    )
    outcomes, impressions = _panel(design)
    result = disentangle(
        design,
        outcomes,
        impressions,
        outcome="sales",
        treatment_start="2026-03-01",
        check_stability=False,
    )
    print("Assignment")
    print(design.assignment.to_string(index=False))
    print("\nWhat the design can name")
    print(design.separation.to_string(index=False))
    print("\nChannel table")
    print(result.channels[["channel", "status", "coef", "std_err", "reading"]].to_string(index=False))
    print("\nBundle totals and the gap left by the channel slopes")
    print(result.bundles[["cell", "measured_effect", "predicted_effect", "gap"]].to_string(index=False))
    flagged = result.exposure.loc[result.exposure["adjust"], ["cell", "channel", "sales_shift", "applied"]]
    print("\nStill-on shifts large enough to flag")
    print(flagged.to_string(index=False) if len(flagged) else "none")
    print("\nThe reported cell total stays unadjusted. A flagged still-on shift is removed before the channel split.")
    print("Pass adjust_exposure=True to replace the reported cell total with that refit.")
    refused = result.channels.loc[result.channels["status"] != "separate", "channel"].tolist()
    print("Refused split:", ", ".join(refused) if refused else "none")
    print("The design can name Facebook and Search. Their paths are the same series, so neither receives a slope.")


def _panel(design):
    rng = np.random.default_rng(0)
    dates = pd.date_range("2026-01-01", periods=70, freq="D")
    treatment = pd.Timestamp("2026-03-01")
    cell_of = design.assignment.set_index("geo")["cell"].astype(str)
    off = {
        (row.cell, row.channel)
        for row in design.schedule.itertuples(index=False)
        if row.off == 1
    }
    outcomes = []
    impressions = []
    for geo, cell in cell_of.items():
        for day in dates:
            facebook = 12.0 + float(rng.normal(0, 1))
            search = facebook
            display = 8.0 + float(rng.normal(0, 1))
            if day >= treatment and (cell, "Facebook") in off:
                facebook = 0.0
                search = 0.0
            if day >= treatment and (cell, "Search") in off:
                search = 0.0
                facebook = 0.0
            if day >= treatment and (cell, "Display") in off:
                display = 0.0
            impressions.append({"geo": geo, "date": day, "channel": "Facebook", "impressions": facebook})
            impressions.append({"geo": geo, "date": day, "channel": "Search", "impressions": search})
            impressions.append({"geo": geo, "date": day, "channel": "Display", "impressions": display})
            outcomes.append(
                {
                    "geo": geo,
                    "date": day,
                    "sales": 0.15 * display + float(rng.normal(0, 0.4)),
                }
            )
    return pd.DataFrame(outcomes), pd.DataFrame(impressions)


if __name__ == "__main__":
    main()

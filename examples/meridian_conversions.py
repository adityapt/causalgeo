"""Run the concurrent split on Meridian's simulated conversions.

The conversions column is Meridian's own simulated outcome. It is not a
holdout with known channel effects. This script only checks that the
package runs from an impression file through a cell table and a channel
table. It is not a recovery check.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "paper"))

from example import CHANNELS, CONTROL_SHARE, DATA, load_media  # noqa: E402
from causalgeo import design_concurrent_test, disentangle  # noqa: E402


def main() -> None:
    raw = load_media()
    weeks = raw["time"].sort_values().unique()
    start = pd.Timestamp(weeks[-16]).strftime("%Y-%m-%d")
    means = raw.groupby("geo")["conversions"].mean()
    geos = pd.qcut(means, 4, labels=["q1", "q2", "q3", "q4"]).astype(str)
    geos = geos.rename("stratum").rename_axis("geo").reset_index()
    design = design_concurrent_test(
        geos,
        channels=CHANNELS,
        bundles=[["Channel0", "Channel1"], ["Channel0", "Channel2"], ["Channel1", "Channel2"]],
        control_share=CONTROL_SHARE,
        seed=1,
    )
    outcomes = raw[["geo", "time", "conversions"]].rename(columns={"time": "date", "conversions": "sales"})
    pieces = []
    for channel in CHANNELS:
        piece = raw[["geo", "time", f"{channel}_impression"]].rename(
            columns={"time": "date", f"{channel}_impression": "impressions"}
        )
        piece["channel"] = channel
        pieces.append(piece)
    impressions = pd.concat(pieces, ignore_index=True)
    result = disentangle(
        design,
        outcomes,
        impressions,
        outcome="sales",
        treatment_start=start,
        check_stability=False,
    )
    print(f"source {DATA}")
    print("No planted slopes. The cell table is the experiment. The channel table is model-assisted.")
    print(result.cells[["cell", "bundle", "coef", "std_err"]].to_string(index=False))
    print(result.channels[["channel", "status", "coef", "reading"]].to_string(index=False))
    return result


if __name__ == "__main__":
    main()

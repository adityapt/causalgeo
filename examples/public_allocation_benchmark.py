"""Score bundle allocation on Google's public Meridian geo panel.

The impression paths are meridian/data/simulated_data/csv/geo_all_channels.csv.
No proprietary panel is read. Sales are constructed from known channel slopes
because the public file has no channel-level holdout result to score against.
"""

from __future__ import annotations

import json
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import pandas as pd

from causalgeo import (
    anchored_channel_allocation,
    design_concurrent_test,
    potency_misspecification_ratio,
    prevented_effective_dose,
)
from causalgeo.identified import estimate_identified

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "paper" / "data" / "geo_all_channels.csv"
DATA_URL = (
    "https://raw.githubusercontent.com/google/meridian/main/"
    "meridian/data/simulated_data/csv/geo_all_channels.csv"
)
OUT = ROOT / "paper" / "results" / "public_allocation_benchmark.json"
CHANNELS = [f"Channel{i}" for i in range(5)]
SLOPES = {
    "Channel0": 0.0002,
    "Channel1": 0.0005,
    "Channel2": 0.0001,
    "Channel3": 0.0004,
    "Channel4": 0.0003,
}
TEST_WEEKS = 16
CONTROL_SHARE = 0.6
DESIGN_SEED = 1
SALES_SEED = 7
NOISE_SD = 30.0


def load_media() -> pd.DataFrame:
    if not DATA.exists():
        DATA.parent.mkdir(parents=True, exist_ok=True)
        urlretrieve(DATA_URL, DATA)
    raw = pd.read_csv(DATA)
    raw["time"] = pd.to_datetime(raw["time"])
    return raw


def build_panel(raw: pd.DataFrame):
    weeks = np.sort(raw["time"].unique())
    treatment_start = pd.Timestamp(weeks[-TEST_WEEKS]).strftime("%Y-%m-%d")
    means = raw.groupby("geo")["conversions"].mean()
    strata = pd.qcut(means, 4, labels=["q1", "q2", "q3", "q4"]).astype(str)
    geos = strata.rename("stratum").rename_axis("geo").reset_index()
    design = design_concurrent_test(
        geos,
        channels=CHANNELS,
        bundles=[CHANNELS],
        control_share=CONTROL_SHARE,
        seed=DESIGN_SEED,
    )
    cell_of = design.assignment.set_index("geo")["cell"]
    pieces = []
    for channel in CHANNELS:
        piece = raw[["geo", "time", f"{channel}_impression"]].rename(
            columns={"time": "date", f"{channel}_impression": "impressions"}
        )
        piece["channel"] = channel
        pieces.append(piece)
    impressions = pd.concat(pieces, ignore_index=True)
    impressions["cell"] = impressions["geo"].map(cell_of)
    post = impressions["date"] >= pd.Timestamp(treatment_start)
    dark = post & impressions["cell"].ne("control")
    impressions.loc[dark, "impressions"] = 0.0

    wide = impressions.pivot_table(
        index=["geo", "date"],
        columns="channel",
        values="impressions",
        aggfunc="sum",
    ).reset_index()
    rng = np.random.default_rng(SALES_SEED)
    geos_present = wide["geo"].unique()
    dates = wide["date"].drop_duplicates().sort_values()
    geo_fe = dict(zip(geos_present, rng.normal(0, 50, len(geos_present)), strict=True))
    date_fe = dict(zip(dates, rng.normal(0, 20, len(dates)), strict=True))
    signal = sum(SLOPES[channel] * wide[channel] for channel in CHANNELS)
    wide["sales"] = (
        wide["geo"].map(geo_fe)
        + wide["date"].map(date_fe)
        + signal
        + rng.normal(0, NOISE_SD, len(wide))
    )
    outcomes = wide[["geo", "date", "sales"]]
    return design, outcomes, impressions.drop(columns="cell"), treatment_start


def run() -> dict:
    raw = load_media()
    design, outcomes, impressions, treatment_start = build_panel(raw)
    identified = estimate_identified(
        design,
        outcomes,
        outcome="sales",
        treatment_start=treatment_start,
    )
    dose = prevented_effective_dose(impressions, design, treatment_start)
    effects = identified.cells[["cell", "coef", "std_err"]]
    equal = anchored_channel_allocation(effects, dose)
    informed = anchored_channel_allocation(effects, dose, potency=SLOPES)
    planted = dose.copy()
    planted["planted"] = [
        SLOPES[channel] * value
        for channel, value in zip(planted["channel"], planted["dose"], strict=True)
    ]
    removed = float(planted["planted"].sum())
    planted["planted_share"] = planted["planted"] / removed
    planted_total = -removed
    comparison = informed.channels.merge(
        equal.channels[["channel", "share", "allocation"]],
        on="channel",
        suffixes=("_informed", "_equal"),
    ).merge(planted[["channel", "planted", "planted_share"]], on="channel")
    comparison["informed_share_error"] = (
        comparison["share_informed"] - comparison["planted_share"]
    )
    comparison["equal_share_error"] = comparison["share_equal"] - comparison["planted_share"]
    ratio = potency_misspecification_ratio(
        planted.set_index("channel")["dose"].to_dict(),
        {channel: 1.0 for channel in CHANNELS},
        SLOPES,
    )
    bundle = float(effects.iloc[0]["coef"])
    result = {
        "source": DATA_URL,
        "treatment_start": treatment_start,
        "n_geos": int(design.assignment["geo"].nunique()),
        "n_weeks": int(outcomes["date"].nunique()),
        "bundle_effect": bundle,
        "bundle_std_err": float(effects.iloc[0]["std_err"]),
        "planted_bundle_effect": planted_total,
        "reconciliation_error": float(informed.channels["allocation"].sum() - bundle),
        "max_abs_informed_share_error": float(comparison["informed_share_error"].abs().max()),
        "max_abs_equal_share_error": float(comparison["equal_share_error"].abs().max()),
        "channels": comparison[
            [
                "channel",
                "dose",
                "planted_share",
                "share_equal",
                "share_informed",
                "allocation_equal",
                "allocation_informed",
            ]
        ].to_dict(orient="records"),
        "equal_potency_bias_ratio": {channel: float(ratio[channel]) for channel in CHANNELS},
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2))
    return result


def main() -> None:
    result = run()
    print(json.dumps({key: value for key, value in result.items() if key != "channels"}, indent=2))
    print(pd.DataFrame(result["channels"]).to_string(index=False))


if __name__ == "__main__":
    main()

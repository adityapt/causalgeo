"""Reproducible example for the SoftwareX manuscript.

Impression paths are the published Meridian geo series
(geo_all_channels.csv). Sales are constructed from known channel slopes so
the recovered numbers can be checked. Meridian's conversions column is used
only to form strata. It is not the outcome.
"""

from __future__ import annotations

import json
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import pandas as pd

from causalgeo import design_concurrent_test, disentangle, experiment_only_split
from causalgeo.identified import estimate_identified

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "geo_all_channels.csv"
DATA_URL = (
    "https://raw.githubusercontent.com/google/meridian/main/"
    "meridian/data/simulated_data/csv/geo_all_channels.csv"
)
OUT = ROOT / "results"
CHANNELS = ["Channel0", "Channel1", "Channel2"]
SLOPES = {"Channel0": 0.0002, "Channel1": 0.0005, "Channel2": 0.0001}
BUNDLES = [["Channel0", "Channel1"], ["Channel0", "Channel2"], ["Channel1", "Channel2"]]
TEST_WEEKS = 16
CONTROL_SHARE = 0.4
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


def build_panel(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    weeks = np.sort(raw["time"].unique())
    treatment_start = pd.Timestamp(weeks[-TEST_WEEKS]).strftime("%Y-%m-%d")
    means = raw.groupby("geo")["conversions"].mean()
    strata = pd.qcut(means, 4, labels=["q1", "q2", "q3", "q4"]).astype(str)
    geos = strata.rename("stratum").rename_axis("geo").reset_index()

    design = design_concurrent_test(
        geos,
        channels=CHANNELS,
        bundles=BUNDLES,
        control_share=CONTROL_SHARE,
        seed=DESIGN_SEED,
    )
    cell_of = design.assignment.set_index("geo")["cell"]
    off = {
        (row.cell, row.channel)
        for row in design.schedule.itertuples(index=False)
        if row.off == 1
    }

    long = []
    for channel in CHANNELS:
        piece = raw[["geo", "time", f"{channel}_impression"]].rename(
            columns={"time": "date", f"{channel}_impression": "impressions"}
        )
        piece["channel"] = channel
        long.append(piece)
    impressions = pd.concat(long, ignore_index=True)
    impressions["cell"] = impressions["geo"].map(cell_of)
    post = impressions["date"] >= pd.Timestamp(treatment_start)
    dark = [
        (cell, channel) in off
        for cell, channel in zip(impressions["cell"], impressions["channel"])
    ]
    impressions.loc[post & pd.Series(dark, index=impressions.index), "impressions"] = 0.0

    wide = impressions.pivot_table(
        index=["geo", "date"], columns="channel", values="impressions", aggfunc="sum"
    ).reset_index()
    rng = np.random.default_rng(SALES_SEED)
    geo_fe = {geo: float(value) for geo, value in zip(wide["geo"].unique(), rng.normal(0, 50, wide["geo"].nunique()))}
    dates = wide["date"].drop_duplicates().sort_values()
    date_fe = {date: float(value) for date, value in zip(dates, rng.normal(0, 20, len(dates)))}
    signal = sum(SLOPES[channel] * wide[channel] for channel in CHANNELS)
    wide["sales"] = (
        wide["geo"].map(geo_fe)
        + wide["date"].map(date_fe)
        + signal
        + rng.normal(0, NOISE_SD, len(wide))
    )
    outcomes = wide[["geo", "date", "sales"]]
    return design, outcomes, impressions.drop(columns="cell"), treatment_start


def slope_table(result) -> pd.DataFrame:
    rows = []
    for record in result.channels.itertuples(index=False):
        planted = SLOPES.get(record.channel, float("nan"))
        estimated = float(record.coef) if pd.notna(record.coef) else float("nan")
        error = estimated - planted if pd.notna(estimated) else float("nan")
        rows.append(
            {
                "channel": record.channel,
                "status": record.status,
                "planted": planted,
                "estimated": estimated,
                "std_err": float(record.std_err) if pd.notna(record.std_err) else float("nan"),
                "error": error,
                "abs_pct_error": abs(error) / planted * 100 if pd.notna(error) else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def tied_case(outcomes: pd.DataFrame, impressions: pd.DataFrame, design, treatment_start: str):
    """Channel1 is replaced by a copy of Channel0. Both should stay blank."""
    copied = impressions.copy()
    base = copied.loc[copied["channel"] == "Channel0", ["geo", "date", "impressions"]]
    other = copied.loc[copied["channel"] == "Channel1", ["geo", "date"]].merge(base, on=["geo", "date"])
    other["channel"] = "Channel1"
    copied = pd.concat(
        [copied.loc[copied["channel"] != "Channel1"], other],
        ignore_index=True,
    )
    # Rebuild sales from the copied paths so the outcome matches the impressions.
    wide = copied.pivot_table(index=["geo", "date"], columns="channel", values="impressions", aggfunc="sum").reset_index()
    rng = np.random.default_rng(SALES_SEED)
    geo_fe = {geo: float(value) for geo, value in zip(wide["geo"].unique(), rng.normal(0, 50, wide["geo"].nunique()))}
    dates = wide["date"].drop_duplicates().sort_values()
    date_fe = {date: float(value) for date, value in zip(dates, rng.normal(0, 20, len(dates)))}
    signal = sum(SLOPES[channel] * wide[channel] for channel in CHANNELS)
    wide["sales"] = (
        wide["geo"].map(geo_fe)
        + wide["date"].map(date_fe)
        + signal
        + rng.normal(0, NOISE_SD, len(wide))
    )
    result = disentangle(
        design,
        wide[["geo", "date", "sales"]],
        copied,
        outcome="sales",
        treatment_start=treatment_start,
        adjust_exposure=True,
        check_stability=False,
    )
    return result.channels[["channel", "status", "coef", "reason"]]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    raw = load_media()
    design, outcomes, impressions, treatment_start = build_panel(raw)
    result = disentangle(
        design,
        outcomes,
        impressions[["geo", "date", "channel", "impressions"]],
        outcome="sales",
        treatment_start=treatment_start,
        adjust_exposure=True,
        check_stability=False,
    )
    identified = estimate_identified(design, outcomes, outcome="sales", treatment_start=treatment_start)
    alone = experiment_only_split(
        impressions[["geo", "date", "channel", "impressions"]],
        identified,
        design,
        treatment_start,
    )
    slopes = slope_table(result)
    alone = alone.merge(slopes[["channel", "planted"]], on="channel", how="left")
    alone["per_10000"] = alone["coef"] * 1e4
    alone["se_per_10000"] = alone["std_err"] * 1e4
    exposure = result.exposure.loc[result.exposure["adjust"], ["cell", "channel", "impression_shift", "sales_shift", "share"]]
    design.separation.to_csv(OUT / "separation.csv", index=False)
    result.bundles.to_csv(OUT / "bundles.csv", index=False)
    result.cells.to_csv(OUT / "cells.csv", index=False)
    slopes.to_csv(OUT / "slopes.csv", index=False)
    alone.to_csv(OUT / "experiment_only.csv", index=False)
    result.exposure.to_csv(OUT / "exposure.csv", index=False)
    tied = tied_case(outcomes, impressions, design, treatment_start)
    tied.to_csv(OUT / "tied.csv", index=False)

    counts = design.assignment.groupby("cell").size().to_dict()
    summary = {
        "treatment_start": treatment_start,
        "n_geos": int(design.assignment["geo"].nunique()),
        "n_weeks": int(outcomes["date"].nunique()),
        "cell_counts": {str(key): int(value) for key, value in counts.items()},
        "max_abs_pct_error": float(slopes["abs_pct_error"].max()),
        "adjustments": exposure.to_dict(orient="records"),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(design.separation.to_string(index=False))
    print(result.bundles.to_string(index=False))
    print(slopes.to_string(index=False))
    print(alone.to_string(index=False))
    print(exposure.to_string(index=False))
    print(tied.to_string(index=False))


if __name__ == "__main__":
    main()

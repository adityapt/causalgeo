"""Repeated checks for the SoftwareX example.

Varies the sales-noise seed, the size of the planted slopes, and how closely
Channel1 follows Channel0. Prints slope error, interval coverage, and whether
an exact copy is withheld.
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from example import (
    BUNDLES,
    CHANNELS,
    CONTROL_SHARE,
    NOISE_SD,
    SLOPES,
    TEST_WEEKS,
    load_media,
)
from causalgeo import design_concurrent_test, disentangle

OUT = Path(__file__).resolve().parent / "results"
SCALES = (1.0, 0.25)
RHOS = (0.0, 0.9, 1.0)
SEEDS = range(200)
_RAW: pd.DataFrame | None = None


def _panel(raw: pd.DataFrame, seed: int, scale: float, rho: float):
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
        seed=1,
    )
    cell_of = design.assignment.set_index("geo")["cell"]
    off = {(row.cell, row.channel) for row in design.schedule.itertuples(index=False) if row.off == 1}
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
    dark = [(cell, channel) in off for cell, channel in zip(impressions["cell"], impressions["channel"])]
    impressions["impressions"] = impressions["impressions"].astype(float)
    if rho > 0:
        wide = impressions.pivot(index=["geo", "date"], columns="channel", values="impressions")
        mixed = rho * wide["Channel0"] + np.sqrt(max(1 - rho**2, 0)) * wide["Channel1"]
        impressions = impressions.merge(
            mixed.rename("mixed").reset_index(),
            on=["geo", "date"],
            how="left",
        )
        copied = impressions["channel"] == "Channel1"
        impressions.loc[copied, "impressions"] = impressions.loc[copied, "mixed"]
        impressions = impressions.drop(columns="mixed")
    # Correlate the untreated paths first. Applying the assignments afterward
    # avoids making one channel's holdout change a channel that stayed on.
    impressions.loc[post & pd.Series(dark, index=impressions.index), "impressions"] = 0.0
    wide = impressions.pivot_table(
        index=["geo", "date"], columns="channel", values="impressions", aggfunc="sum"
    ).reset_index()
    rng = np.random.default_rng(seed)
    geo_fe = dict(zip(wide["geo"].unique(), rng.normal(0, 50, wide["geo"].nunique())))
    dates = wide["date"].drop_duplicates().sort_values()
    date_fe = dict(zip(dates, rng.normal(0, 20, len(dates))))
    slopes = {channel: SLOPES[channel] * scale for channel in CHANNELS}
    signal = sum(slopes[channel] * wide[channel] for channel in CHANNELS)
    wide["sales"] = wide["geo"].map(geo_fe) + wide["date"].map(date_fe) + signal + rng.normal(0, NOISE_SD, len(wide))
    return design, wide[["geo", "date", "sales"]], impressions[["geo", "date", "channel", "impressions"]], treatment_start, slopes


def _init_worker() -> None:
    global _RAW
    _RAW = load_media()


def _run_one(task: tuple[float, float, int]) -> list[dict]:
    scale, rho, seed = task
    if _RAW is None:
        raise RuntimeError("simulation worker has no media data")
    design, outcomes, impressions, start, slopes = _panel(_RAW, seed, scale, rho)
    result = disentangle(
        design,
        outcomes,
        impressions,
        outcome="sales",
        treatment_start=start,
        adjust_exposure=True,
        check_stability=False,
    )
    fitted = result.channels.set_index("channel")
    rows = []
    for channel, planted in slopes.items():
        status = str(fitted.loc[channel, "status"])
        estimate = float(fitted.loc[channel, "coef"]) if pd.notna(fitted.loc[channel, "coef"]) else np.nan
        se = float(fitted.loc[channel, "std_err"]) if pd.notna(fitted.loc[channel, "std_err"]) else np.nan
        covered = bool(status == "separate" and abs(estimate - planted) <= 1.96 * se)
        rows.append(
            {
                "scale": scale,
                "rho": rho,
                "seed": seed,
                "channel": channel,
                "status": status,
                "planted": planted,
                "estimate": estimate,
                "se": se,
                "abs_pct_error": abs(estimate - planted) / planted * 100 if status == "separate" else np.nan,
                "covered": covered,
            }
        )
    return rows


def main() -> None:
    tasks = [
        (scale, rho, seed)
        for scale in SCALES
        for rho in RHOS
        for seed in SEEDS
    ]
    workers = min(8, os.cpu_count() or 1)
    rows = []
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker) as pool:
        for completed, result_rows in enumerate(pool.map(_run_one, tasks), start=1):
            rows.extend(result_rows)
            if completed % 50 == 0 or completed == len(tasks):
                print(f"completed {completed}/{len(tasks)} simulations", flush=True)
    detail = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    detail.to_csv(OUT / "reliability_detail.csv", index=False)
    summary = []
    for (scale, rho), group in detail.groupby(["scale", "rho"]):
        separate = group.loc[group["status"] == "separate"]
        copies = group.loc[group["channel"].isin(["Channel0", "Channel1"])]
        run_coverage = separate.groupby("seed")["covered"].mean()
        coverage = float(run_coverage.mean())
        coverage_mcse = float(run_coverage.std(ddof=1) / np.sqrt(len(run_coverage)))
        summary.append(
            {
                "scale": scale,
                "rho": rho,
                "runs": int(group["seed"].nunique()),
                "median_abs_pct_error": float(separate["abs_pct_error"].median()),
                "coverage": coverage,
                "coverage_mcse": coverage_mcse,
                "coverage_mc_lower": max(0.0, coverage - 1.96 * coverage_mcse),
                "coverage_mc_upper": min(1.0, coverage + 1.96 * coverage_mcse),
                "copy_pair_withheld": float((copies["status"] == "tied").mean()),
            }
        )
    table = pd.DataFrame(summary)
    table.to_csv(OUT / "reliability.csv", index=False)
    print(table.to_string(index=False))
    (OUT / "reliability.json").write_text(table.to_json(orient="records", indent=2))


if __name__ == "__main__":
    main()

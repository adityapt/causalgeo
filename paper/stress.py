"""Misspecification stress tests for the SoftwareX example.

The fitted channel coefficient has no single planted value under carryover or
saturation. Evaluation therefore uses the channel's contribution to the same
pre/post, test/control contrast as the experiment.
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from causalgeo import design_concurrent_test, disentangle
from causalgeo.calibrated import _adstock, _wide, _withheld
from example import (
    BUNDLES,
    CHANNELS,
    CONTROL_SHARE,
    DESIGN_SEED,
    NOISE_SD,
    SLOPES,
    TEST_WEEKS,
    load_media,
)

OUT = Path(__file__).resolve().parent / "results"
SEEDS = range(200)
CARRYOVER_DECAY = 0.5
SATURATION_MULTIPLIER = 2.0
SCENARIOS = (
    ("carryover_omitted", "Carryover omitted", CARRYOVER_DECAY, 0.0, False),
    ("carryover_modeled", "Carryover modeled", CARRYOVER_DECAY, CARRYOVER_DECAY, False),
    ("saturation_omitted", "Saturation omitted", 0.0, 0.0, True),
)
_RAW: pd.DataFrame | None = None


def geometric_adstock(values: np.ndarray, decay: float) -> np.ndarray:
    """Return unnormalized geometric adstock for one ordered exposure path."""
    out = np.empty(len(values), dtype=float)
    level = 0.0
    for index, value in enumerate(np.asarray(values, dtype=float)):
        level = float(value) + decay * level
        out[index] = level
    return out


def hill_response(
    values: np.ndarray,
    slope: float,
    half_saturation: float,
) -> np.ndarray:
    """A concave response whose derivative at zero is ``slope``."""
    if half_saturation <= 0:
        raise ValueError("half_saturation must be positive")
    values = np.asarray(values, dtype=float)
    return slope * values / (1.0 + values / half_saturation)


def true_channel_contrasts(
    factual_signal: pd.DataFrame,
    assignment: pd.Series,
    bundles: dict[str, list[str]],
    treatment_start: str,
) -> pd.DataFrame:
    """Known channel contribution to the experimental difference-in-differences."""
    frame = factual_signal.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame["cell"] = frame["geo"].map(assignment.astype(str))
    post = frame["date"] >= pd.Timestamp(treatment_start)
    control_post = frame.loc[post & (frame["cell"] == "control")]
    control_pre = frame.loc[~post & (frame["cell"] == "control")]
    rows = []
    for cell, off_channels in bundles.items():
        dark_post = frame.loc[post & (frame["cell"] == str(cell))]
        dark_pre = frame.loc[~post & (frame["cell"] == str(cell))]
        for channel in off_channels:
            effect = (
                dark_post[channel].mean()
                - dark_pre[channel].mean()
                - (control_post[channel].mean() - control_pre[channel].mean())
            )
            rows.append(
                {
                    "cell": str(cell),
                    "channel": channel,
                    "true_effect": float(effect),
                }
            )
    return pd.DataFrame(rows)


def _design_and_exposure(raw: pd.DataFrame):
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
    pieces = []
    for channel in CHANNELS:
        piece = raw[["geo", "time", f"{channel}_impression"]].rename(
            columns={"time": "date", f"{channel}_impression": "impressions"}
        )
        piece["channel"] = channel
        pieces.append(piece)
    bau = pd.concat(pieces, ignore_index=True)
    factual = bau.copy()
    cell_of = design.assignment.set_index("geo")["cell"].astype(str)
    factual["cell"] = factual["geo"].map(cell_of)
    schedule = design.schedule.loc[design.schedule["off"] == 1]
    off = {(str(row.cell), row.channel) for row in schedule.itertuples(index=False)}
    post = factual["date"] >= pd.Timestamp(treatment_start)
    dark = pd.Series(
        [
            (str(cell), channel) in off
            for cell, channel in zip(factual["cell"], factual["channel"])
        ],
        index=factual.index,
    )
    factual.loc[post & dark, "impressions"] = 0.0
    bundles = {
        str(cell): group["channel"].tolist()
        for cell, group in schedule.groupby("cell", sort=True)
    }
    return (
        design,
        bau[["geo", "date", "channel", "impressions"]],
        factual[["geo", "date", "channel", "impressions"]],
        treatment_start,
        cell_of,
        bundles,
    )


def _signal(
    impressions: pd.DataFrame,
    *,
    carryover_decay: float,
    saturation: bool,
    half_saturation: dict[str, float],
) -> pd.DataFrame:
    frame = impressions.sort_values(["geo", "channel", "date"]).copy()
    if carryover_decay:
        frame["effective"] = frame.groupby(["geo", "channel"], sort=False)[
            "impressions"
        ].transform(lambda values: geometric_adstock(values.to_numpy(), carryover_decay))
    else:
        frame["effective"] = frame["impressions"].astype(float)
    if saturation:
        frame["contribution"] = [
            float(hill_response(np.array([value]), SLOPES[channel], half_saturation[channel])[0])
            for channel, value in zip(frame["channel"], frame["effective"])
        ]
    else:
        frame["contribution"] = [
            SLOPES[channel] * float(value)
            for channel, value in zip(frame["channel"], frame["effective"])
        ]
    return frame.pivot_table(
        index=["geo", "date"],
        columns="channel",
        values="contribution",
        aggfunc="sum",
    ).reset_index()


def _run_one(task: tuple[str, str, float, float, bool, int]) -> tuple[list[dict], list[dict]]:
    key, label, true_decay, fit_decay, saturation, seed = task
    if _RAW is None:
        raise RuntimeError("simulation worker has no media data")
    design, bau, factual, start, cell_of, bundles = _design_and_exposure(_RAW)
    half_saturation = {
        channel: SATURATION_MULTIPLIER
        * float(
            bau.loc[
                (bau["channel"] == channel) & (bau["impressions"] > 0),
                "impressions",
            ].median()
        )
        for channel in CHANNELS
    }
    factual_signal = _signal(
        factual,
        carryover_decay=true_decay,
        saturation=saturation,
        half_saturation=half_saturation,
    )
    rng = np.random.default_rng(seed)
    geos = factual_signal["geo"].unique()
    dates = factual_signal["date"].drop_duplicates().sort_values()
    geo_fe = dict(zip(geos, rng.normal(0, 50, len(geos))))
    date_fe = dict(zip(dates, rng.normal(0, 20, len(dates))))
    outcomes = factual_signal[["geo", "date"]].copy()
    outcomes["sales"] = (
        outcomes["geo"].map(geo_fe)
        + outcomes["date"].map(date_fe)
        + factual_signal[CHANNELS].sum(axis=1)
        + rng.normal(0, NOISE_SD, len(outcomes))
    )
    result = disentangle(
        design,
        outcomes,
        factual,
        outcome="sales",
        treatment_start=start,
        decay=fit_decay,
        adjust_exposure=True,
        check_stability=False,
    )
    true = true_channel_contrasts(
        factual_signal,
        cell_of,
        bundles,
        start,
    )
    transformed = _adstock(factual, "geo", "date", "channel", "impressions", fit_decay)
    wide = _wide(transformed, "geo", "date", "channel", "impressions", CHANNELS)
    dose = _withheld(wide, design, CHANNELS, "geo", "date", start)
    fitted = result.channels.set_index("channel")
    channel_rows = []
    for record in true.merge(dose[["cell", "channel", "withheld"]], on=["cell", "channel"]).itertuples(index=False):
        coef = float(fitted.loc[record.channel, "coef"])
        se = float(fitted.loc[record.channel, "std_err"])
        estimate = -coef * float(record.withheld)
        contrast_se = se * float(record.withheld)
        channel_rows.append(
            {
                "scenario": key,
                "label": label,
                "seed": seed,
                "cell": record.cell,
                "channel": record.channel,
                "true_effect": float(record.true_effect),
                "estimated_effect": estimate,
                "se": contrast_se,
                "abs_pct_error": abs(estimate - record.true_effect)
                / max(abs(record.true_effect), 1e-12)
                * 100,
                "covered": abs(estimate - record.true_effect) <= 1.96 * contrast_se,
            }
        )
    bundle_rows = []
    for record in result.bundles.itertuples(index=False):
        measured = float(record.measured_effect)
        bundle_rows.append(
            {
                "scenario": key,
                "label": label,
                "seed": seed,
                "cell": str(record.cell),
                "abs_gap_pct": abs(float(record.gap)) / max(abs(measured), 1e-12) * 100,
            }
        )
    return channel_rows, bundle_rows


def _init_worker() -> None:
    global _RAW
    _RAW = load_media()


def summarize(channel_detail: pd.DataFrame, bundle_detail: pd.DataFrame) -> pd.DataFrame:
    """Summarize contribution error, empirical coverage, and calibration gap."""
    rows = []
    for (scenario, label), group in channel_detail.groupby(["scenario", "label"], sort=False):
        run_coverage = group.groupby("seed")["covered"].mean()
        coverage = float(run_coverage.mean())
        mcse = float(run_coverage.std(ddof=1) / np.sqrt(len(run_coverage)))
        gaps = bundle_detail.loc[bundle_detail["scenario"] == scenario, "abs_gap_pct"]
        rows.append(
            {
                "scenario": scenario,
                "label": label,
                "runs": int(group["seed"].nunique()),
                "median_abs_contribution_error_pct": float(group["abs_pct_error"].median()),
                "coverage": coverage,
                "coverage_mc_lower": max(0.0, coverage - 1.96 * mcse),
                "coverage_mc_upper": min(1.0, coverage + 1.96 * mcse),
                "median_abs_bundle_gap_pct": float(gaps.median()),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    tasks = [(*scenario, seed) for scenario in SCENARIOS for seed in SEEDS]
    workers = min(8, os.cpu_count() or 1)
    channel_rows: list[dict] = []
    bundle_rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker) as pool:
        for completed, (channels, bundles) in enumerate(pool.map(_run_one, tasks), start=1):
            channel_rows.extend(channels)
            bundle_rows.extend(bundles)
            if completed % 50 == 0 or completed == len(tasks):
                print(f"completed {completed}/{len(tasks)} stress simulations", flush=True)
    channel_detail = pd.DataFrame(channel_rows)
    bundle_detail = pd.DataFrame(bundle_rows)
    table = summarize(channel_detail, bundle_detail)
    OUT.mkdir(parents=True, exist_ok=True)
    channel_detail.to_csv(OUT / "stress_detail.csv", index=False)
    bundle_detail.to_csv(OUT / "stress_bundles.csv", index=False)
    table.to_csv(OUT / "stress.csv", index=False)
    (OUT / "stress.json").write_text(table.to_json(orient="records", indent=2))
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()

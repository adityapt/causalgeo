"""Extra checks beyond the linear recovery and the carryover stress test.

Each setting uses 30 sales-noise seeds on the same Meridian impression paths
and the same assignment as the manuscript example. The channel error is the
median absolute percent error among slopes the fit is willing to report.
Coverage is the share of those slopes whose interval contains the planted value.
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from reliability import _panel, load_media

OUT = Path(__file__).resolve().parent / "results"
SEEDS = range(30)
_RAW: pd.DataFrame | None = None


def _init_worker() -> None:
    global _RAW
    _RAW = load_media()


def _score(result, slopes) -> dict:
    fitted = result.channels.set_index("channel")
    errors = []
    covered = []
    for channel, planted in slopes.items():
        status = str(fitted.loc[channel, "status"])
        if status not in {"separate", "unstable"} or not np.isfinite(fitted.loc[channel, "coef"]):
            continue
        estimate = float(fitted.loc[channel, "coef"])
        se = float(fitted.loc[channel, "std_err"])
        errors.append(abs(estimate - planted) / abs(planted) * 100)
        covered.append(abs(estimate - planted) <= 1.96 * se)
    gap = result.bundles["gap"].abs()
    measured = result.bundles["measured_effect"].abs().replace(0, np.nan)
    reallocation = np.nan
    if "experimental_effect" in result.bundles.columns:
        raw = result.bundles["experimental_effect"].abs().replace(0, np.nan)
        reallocation = float(
            (
                (result.bundles["experimental_effect"] - result.bundles["measured_effect"]).abs() / raw * 100
            ).median()
        )
    return {
        "median_abs_error_pct": float(np.median(errors)) if errors else np.nan,
        "max_abs_error_pct": float(np.max(errors)) if errors else np.nan,
        "median_se": float(np.median([
            float(fitted.loc[channel, "std_err"])
            for channel, planted in slopes.items()
            if str(fitted.loc[channel, "status"]) in {"separate", "unstable"}
            and np.isfinite(fitted.loc[channel, "coef"])
        ])) if errors else np.nan,
        "coverage": float(np.mean(covered)) if covered else np.nan,
        "reported": len(errors),
        "median_abs_bundle_gap_pct": float((gap / measured * 100).median()),
        "median_abs_reallocation_pct": reallocation,
    }


def _mutate(kind: str, design, outcomes, impressions, start, slopes, seed: int):
    rng = np.random.default_rng(10_000 + seed)
    outcomes = outcomes.copy()
    impressions = impressions.copy()
    if kind == "serial":
        outcomes = outcomes.sort_values(["geo", "date"])
        noise = []
        for _, group in outcomes.groupby("geo", sort=False):
            shock = np.zeros(len(group))
            previous = 0.0
            for index in range(len(group)):
                previous = 0.8 * previous + float(rng.normal(0, 30))
                shock[index] = previous
            noise.append(pd.Series(shock, index=group.index))
        outcomes["sales"] = outcomes["sales"] + pd.concat(noise)
    elif kind == "heterogeneous":
        factors = {
            geo: float(rng.uniform(0.5, 1.5))
            for geo in outcomes["geo"].unique()
        }
        wide = impressions.pivot(index=["geo", "date"], columns="channel", values="impressions").reset_index()
        bump = sum((wide["geo"].map(factors) - 1.0) * slopes[channel] * wide[channel] for channel in slopes)
        wide["bump"] = bump
        outcomes = outcomes.merge(wide[["geo", "date", "bump"]], on=["geo", "date"], how="left")
        outcomes["sales"] = outcomes["sales"] + outcomes["bump"].fillna(0)
        outcomes = outcomes.drop(columns="bump")
    elif kind in {"redistribution", "redistribution_adjusted"}:
        impressions, outcomes = _move_budget(impressions, outcomes, design, start, slopes)
    elif kind == "incomplete":
        impressions, outcomes = _partial_holdout(impressions, outcomes, design, start, slopes)
    elif kind == "confounded":
        raise ValueError("the confounder is applied before this mutation")
    else:
        raise ValueError(kind)
    return outcomes, impressions


def _confound(outcomes: pd.DataFrame, bau: pd.DataFrame, slopes: dict[str, float]) -> pd.DataFrame:
    """Add a shock that tracks Channel0's business-as-usual path.

    The holdout zeros the measured impressions. The shock does not turn off,
    so the observational slope absorbs it and the cell contrast does not.
    """
    path = bau.loc[bau["channel"] == "Channel0", ["geo", "date", "impressions"]]
    updated = outcomes.merge(path, on=["geo", "date"], how="left")
    updated["sales"] = updated["sales"] + 0.5 * slopes["Channel0"] * updated["impressions"].fillna(0)
    return updated.drop(columns="impressions")


def _apply_sales(before, after, outcomes, slopes):
    wide_before = before.pivot(index=["geo", "date"], columns="channel", values="impressions")
    wide_after = after.pivot(index=["geo", "date"], columns="channel", values="impressions")
    delta = sum(slopes[channel] * (wide_after[channel] - wide_before[channel]) for channel in slopes)
    bump = delta.rename("bump").reset_index()
    updated = outcomes.merge(bump, on=["geo", "date"], how="left")
    updated["sales"] = updated["sales"] + updated["bump"].fillna(0)
    return updated.drop(columns="bump")


def _move_budget(impressions, outcomes, design, start, slopes):
    schedule = design.schedule
    cell_of = design.assignment.set_index("geo")["cell"].astype(str)
    frame = impressions.copy()
    frame["cell"] = frame["geo"].map(cell_of)
    post = frame["date"] >= pd.Timestamp(start)
    for cell, group in schedule.groupby("cell"):
        off = set(group.loc[group["off"] == 1, "channel"])
        on = [channel for channel in group["channel"] if channel not in off]
        if not on:
            continue
        dark = frame["cell"].eq(str(cell)) & post & frame["channel"].isin(off)
        lost = frame.loc[dark].groupby(["geo", "date"])["impressions"].sum()
        # The holdout already set these to zero. Put half of the pre-test level
        # onto the first channel that this cell left on. That level is recovered
        # from the control markets' post-period mean for the same channel.
        donor = frame.loc[frame["cell"].eq("control") & post & frame["channel"].isin(off)]
        typical = donor.groupby("channel")["impressions"].mean()
        receiver = on[0]
        for (geo, date), _ in lost.items():
            moved = 0.5 * float(typical.sum())
            mask = (
                frame["geo"].eq(geo)
                & frame["date"].eq(date)
                & frame["channel"].eq(receiver)
            )
            frame.loc[mask, "impressions"] = frame.loc[mask, "impressions"] + moved
    after = frame.drop(columns="cell")
    return after, _apply_sales(impressions, after, outcomes, slopes)


def _partial_holdout(impressions, outcomes, design, start, slopes):
    cell_of = design.assignment.set_index("geo")["cell"].astype(str)
    off = {(row.cell, row.channel) for row in design.schedule.itertuples(index=False) if row.off == 1}
    frame = impressions.copy()
    frame["cell"] = frame["geo"].map(cell_of)
    control = (
        frame.loc[frame["cell"].eq("control")]
        .groupby(["date", "channel"])["impressions"]
        .mean()
    )
    post = frame["date"] >= pd.Timestamp(start)
    for index, row in frame.loc[post].iterrows():
        if (str(row["cell"]), row["channel"]) in off:
            frame.at[index, "impressions"] = 0.3 * float(control.get((row["date"], row["channel"]), 0.0))
    after = frame.drop(columns="cell")
    return after, _apply_sales(impressions, after, outcomes, slopes)


def _run_one(task: tuple[str, int]) -> dict:
    kind, seed = task
    if _RAW is None:
        raise RuntimeError("simulation worker has no media data")
    from causalgeo import disentangle
    from causalgeo.calibrated import fit_channel_model
    from causalgeo.identified import estimate_identified

    if kind == "collinear_weak":
        design, outcomes, impressions, start, slopes = _panel(_RAW, seed, 0.05, 0.999)
        identified = estimate_identified(design, outcomes, outcome="sales", treatment_start=start)
        observational = fit_channel_model(
            outcomes, impressions, identified, design, start, "sales", calibrate=False
        )
        calibrated = fit_channel_model(
            outcomes, impressions, identified, design, start, "sales", calibrate=True
        )
        left = _score(observational, slopes)
        right = _score(calibrated, slopes)
        return {
            "scenario": "collinear_weak",
            "seed": seed,
            **{f"observational_{key}": value for key, value in left.items()},
            **{f"calibrated_{key}": value for key, value in right.items()},
        }
    if kind == "confounded":
        design, outcomes, impressions, start, slopes, bau, _correlation = _panel(
            _RAW, seed, 1.0, 0.0, return_bau=True
        )
        outcomes = _confound(outcomes, bau, slopes)
        result = disentangle(
            design,
            outcomes,
            impressions,
            outcome="sales",
            treatment_start=start,
            adjust_exposure=True,
            check_stability=False,
        )
        identified = estimate_identified(design, outcomes, outcome="sales", treatment_start=start)
        observational = fit_channel_model(
            outcomes, impressions, identified, design, start, "sales", calibrate=False
        )
        scored = _score(result, slopes)
        observed = _score(observational, slopes)
        return {
            "scenario": "confounded",
            "seed": seed,
            **scored,
            "observational_median_abs_error_pct": observed["median_abs_error_pct"],
            "observational_max_abs_error_pct": observed["max_abs_error_pct"],
            "observational_coverage": observed["coverage"],
        }
    if kind == "experiment_only":
        from causalgeo.calibrated import experiment_only_split

        design, outcomes, impressions, start, slopes = _panel(_RAW, seed, 1.0, 0.0)
        identified = estimate_identified(design, outcomes, outcome="sales", treatment_start=start)
        alone = experiment_only_split(impressions, identified, design, start)
        alone["status"] = np.where(np.isfinite(alone["coef"]), "separate", "not tested")
        fake = type("Fit", (), {})()
        fake.channels = alone
        fake.bundles = pd.DataFrame({"gap": [np.nan], "measured_effect": [np.nan]})
        scored = _score(fake, slopes)
        return {"scenario": "experiment_only", "seed": seed, **scored}
    design, outcomes, impressions, start, slopes = _panel(_RAW, seed, 1.0, 0.0)
    if kind == "calibration":
        identified = estimate_identified(design, outcomes, outcome="sales", treatment_start=start)
        observational = fit_channel_model(
            outcomes, impressions, identified, design, start, "sales", calibrate=False
        )
        calibrated = fit_channel_model(
            outcomes, impressions, identified, design, start, "sales", calibrate=True
        )
        left = _score(observational, slopes)
        right = _score(calibrated, slopes)
        return {
            "scenario": "calibration",
            "seed": seed,
            **{f"observational_{key}": value for key, value in left.items()},
            **{f"calibrated_{key}": value for key, value in right.items()},
        }
    outcomes, impressions = _mutate(kind, design, outcomes, impressions, start, slopes, seed)
    result = disentangle(
        design,
        outcomes,
        impressions,
        outcome="sales",
        treatment_start=start,
        adjust_exposure=kind == "redistribution_adjusted",
        check_stability=False,
    )
    return {"scenario": kind, "seed": seed, **_score(result, slopes)}


def main() -> pd.DataFrame:
    tasks = [
        (kind, seed)
        for kind in (
            "serial",
            "heterogeneous",
            "redistribution",
            "redistribution_adjusted",
            "incomplete",
            "calibration",
            "confounded",
            "collinear_weak",
            "experiment_only",
        )
        for seed in SEEDS
    ]
    workers = min(8, os.cpu_count() or 1)
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker) as pool:
        rows = list(pool.map(_run_one, tasks))
    detail = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    detail.to_csv(OUT / "robustness_detail.csv", index=False)
    summary = summarize(detail)
    summary.to_csv(OUT / "robustness.csv", index=False)
    print(summary.to_string(index=False))
    return summary


def _interval(values: pd.Series) -> tuple[float, float]:
    clean = values.dropna()
    if len(clean) < 2:
        return float("nan"), float("nan")
    center = float(clean.mean())
    half = 1.96 * float(clean.std(ddof=1) / np.sqrt(len(clean)))
    return max(0.0, center - half), min(1.0, center + half)


def summarize(detail: pd.DataFrame) -> pd.DataFrame:
    rows = []
    ordinary = detail.dropna(subset=["median_abs_error_pct"]) if "median_abs_error_pct" in detail else detail.iloc[0:0]
    for scenario, group in ordinary.groupby("scenario"):
        cover_low, cover_high = _interval(group["coverage"])
        rows.append(
            {
                "scenario": scenario,
                "runs": int(len(group)),
                "median_abs_error_pct": round(float(group["median_abs_error_pct"].median()), 2),
                "max_abs_error_pct": round(float(group["max_abs_error_pct"].median()), 2),
                "median_se": round(float(group["median_se"].median()), 6),
                "coverage": round(float(group["coverage"].mean()), 3),
                "coverage_mc_lower": round(cover_low, 3),
                "coverage_mc_upper": round(cover_high, 3),
                "median_abs_bundle_gap_pct": round(float(group["median_abs_bundle_gap_pct"].median()), 2),
                "median_abs_reallocation_pct": round(float(group["median_abs_reallocation_pct"].median()), 2)
                if group["median_abs_reallocation_pct"].notna().any()
                else np.nan,
                "observational_max_abs_error_pct": round(float(group["observational_max_abs_error_pct"].median()), 2)
                if "observational_max_abs_error_pct" in group and group["observational_max_abs_error_pct"].notna().any()
                else np.nan,
            }
        )
    paired = detail.dropna(subset=["calibrated_median_abs_error_pct"]) if "calibrated_median_abs_error_pct" in detail else detail.iloc[0:0]
    for scenario, group in paired.groupby("scenario"):
        if scenario in {"calibration", "observational_only"}:
            labels = (
                ("observational_only", "observational"),
                ("calibrated", "calibrated"),
            )
        else:
            labels = (
                (f"{scenario}_observational", "observational"),
                (f"{scenario}_calibrated", "calibrated"),
            )
        for name, prefix in labels:
            cover_low, cover_high = _interval(group[f"{prefix}_coverage"])
            rows.append(
                {
                    "scenario": name,
                    "runs": int(len(group)),
                    "median_abs_error_pct": round(float(group[f"{prefix}_median_abs_error_pct"].median()), 2),
                    "max_abs_error_pct": round(float(group[f"{prefix}_max_abs_error_pct"].median()), 2),
                    "median_se": round(float(group[f"{prefix}_median_se"].median()), 6),
                    "coverage": round(float(group[f"{prefix}_coverage"].mean()), 3),
                    "coverage_mc_lower": round(cover_low, 3),
                    "coverage_mc_upper": round(cover_high, 3),
                    "median_abs_bundle_gap_pct": round(float(group[f"{prefix}_median_abs_bundle_gap_pct"].median()), 2),
                }
            )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    main()

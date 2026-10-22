"""Market bootstrap for the concurrent channel split.

Each draw resamples markets inside their assigned cells and keeps every
selected market's full time series. Copied draws receive new market ids, so
the cluster bootstrap does not treat two copies as one market.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from causalgeo.calibrated import disentangle
from causalgeo.design import ConcurrentDesign


def bootstrap_channel_intervals(
    design: ConcurrentDesign,
    outcomes: pd.DataFrame,
    impressions: pd.DataFrame,
    outcome: str,
    treatment_start: str,
    n_boot: int = 50,
    seed: int = 0,
    geo_col: str = "geo",
    **kwargs,
) -> pd.DataFrame:
    """Percentile intervals from refitting the full channel pipeline."""
    if n_boot < 2:
        raise ValueError("n_boot must be at least 2")
    kwargs["check_stability"] = False
    point = disentangle(
        design,
        outcomes,
        impressions,
        outcome=outcome,
        treatment_start=treatment_start,
        geo_col=geo_col,
        **kwargs,
    )
    rng = np.random.default_rng(seed)
    draws = []
    for draw in range(n_boot):
        sample = _resample(design, rng, draw, geo_col)
        fitted = disentangle(
            _replace_assignment(design, sample, geo_col),
            _expand(outcomes, sample, geo_col),
            _expand(impressions, sample, geo_col),
            outcome=outcome,
            treatment_start=treatment_start,
            geo_col=geo_col,
            **kwargs,
        )
        draws.append(fitted.channels.assign(draw=draw))
    stacked = pd.concat(draws, ignore_index=True)
    rows = []
    reported = point.channels.set_index("channel")
    for channel, group in stacked.groupby("channel", sort=False):
        values = group.loc[group["status"].isin(["separate", "unstable"]), "coef"].dropna()
        coef = reported.loc[channel, "coef"]
        rows.append(
            {
                "channel": channel,
                "coef": coef,
                "draws": int(values.shape[0]),
                "boot_se": float(values.std(ddof=1)) if len(values) > 1 else float("nan"),
                "ci_lower": float(np.quantile(values, 0.025)) if len(values) else float("nan"),
                "ci_upper": float(np.quantile(values, 0.975)) if len(values) else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def _resample(design: ConcurrentDesign, rng: np.random.Generator, draw: int, geo_col: str) -> pd.DataFrame:
    rows = []
    for cell, group in design.assignment.groupby("cell", sort=True):
        chosen = rng.choice(group[geo_col].to_numpy(), size=len(group), replace=True)
        for index, geo in enumerate(chosen):
            rows.append({geo_col: geo, "resampled_geo": f"{cell}:{draw}:{index}", "cell": cell})
    return pd.DataFrame(rows)


def _replace_assignment(design: ConcurrentDesign, sample: pd.DataFrame, geo_col: str) -> ConcurrentDesign:
    replacement = design.assignment.iloc[:0].copy()
    replacement[geo_col] = sample["resampled_geo"].to_numpy()
    replacement["cell"] = sample["cell"].astype(str).to_numpy()
    if "role" in design.assignment.columns:
        replacement["role"] = np.where(replacement["cell"].eq("control"), "control", "test")
    if "stratum" in design.assignment.columns:
        replacement["stratum"] = "resampled"
    return ConcurrentDesign(
        assignment=replacement,
        schedule=design.schedule.copy(),
        separation=design.separation.copy(),
    )


def _expand(frame: pd.DataFrame, sample: pd.DataFrame, geo_col: str) -> pd.DataFrame:
    expanded = sample.merge(frame, on=geo_col, how="left")
    expanded[geo_col] = expanded["resampled_geo"]
    return expanded.drop(columns=["resampled_geo", "cell"])

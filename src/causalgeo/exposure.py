"""Decide when a channel left on has to be controlled for.

A channel that stays on can still move sales if its impressions change by a
different amount in the dark markets than in the control markets. The check
is that extra shift, valued at the channel's pre-test sales slope. A small
shift leaves the bundle estimate alone. A large one is added to the bundle
regression.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from causalgeo.design import ConcurrentDesign
from causalgeo.identified import IdentifiedEstimate
from causalgeo.panel_ols import run_panel_ols_did


def adjust_for_exposure(
    design: ConcurrentDesign,
    outcomes: pd.DataFrame,
    impressions: pd.DataFrame,
    identified: IdentifiedEstimate,
    treatment_start: str,
    outcome: str,
    geo_col: str = "geo",
    date_col: str = "date",
    channel_col: str = "channel",
    impression_col: str = "impressions",
    material: float = 0.10,
    apply: bool = False,
) -> tuple[IdentifiedEstimate, pd.DataFrame]:
    """Flag a still-on exposure shift, and adjust the bundle only when asked.

    An impression shift does not say whether the holdout moved the budget.
    The default leaves the experimental estimate unchanged and records the
    shift. Pass ``apply=True`` to refit the flagged cells.
    """
    if not 0 < material < 1:
        raise ValueError("material must be between 0 and 1")
    check = _exposure_check(
        design,
        outcomes,
        impressions,
        identified,
        treatment_start,
        outcome,
        geo_col,
        date_col,
        channel_col,
        impression_col,
        material,
    )
    check["applied"] = False
    if check.empty or not apply or not bool(check["adjust"].any()):
        cells = identified.cells.copy()
        cells["unadjusted_effect"] = cells["coef"]
        cells["adjusted"] = False
        return _replace_cells(identified, cells), check

    panel = _panel(outcomes, impressions, design, outcome, geo_col, date_col, channel_col, impression_col)
    cells = identified.cells.copy()
    cells["unadjusted_effect"] = cells["coef"]
    cells["adjusted"] = False
    start = pd.Timestamp(treatment_start)
    for cell in check.loc[check["adjust"], "cell"].unique():
        covariates = check.loc[(check["cell"] == cell) & check["adjust"], "channel"].tolist()
        use = panel.loc[panel["cell"].isin([cell, "control"])].copy()
        use["dmacode"] = use[geo_col]
        use["did"] = ((use["cell"] == cell) & (use["date"] >= start)).astype(float)
        fitted = run_panel_ols_did(use, outcome, ["did", *covariates])
        row = fitted.summary_frame().loc["did"]
        cells.loc[cells["cell"] == cell, ["coef", "std_err", "p_value", "ci_lower", "ci_upper"]] = [
            float(row["coef"]),
            float(row["std_err"]),
            float(row["p_value"]),
            float(row["ci_lower"]),
            float(row["ci_upper"]),
        ]
        cells.loc[cells["cell"] == cell, "adjusted"] = True
        check.loc[(check["cell"] == cell) & check["adjust"], "applied"] = True
    return _replace_cells(identified, cells), check


def _replace_cells(identified: IdentifiedEstimate, cells: pd.DataFrame) -> IdentifiedEstimate:
    return IdentifiedEstimate(cells=cells, channels=identified.channels, estimate=identified.estimate)


def _exposure_check(
    design,
    outcomes,
    impressions,
    identified,
    treatment_start,
    outcome,
    geo_col,
    date_col,
    channel_col,
    impression_col,
    material,
) -> pd.DataFrame:
    bundles = _bundles(design)
    panel = _panel(outcomes, impressions, design, outcome, geo_col, date_col, channel_col, impression_col)
    start = pd.Timestamp(treatment_start)
    pre = panel["date"] < start
    effects = identified.cells.set_index("cell")["coef"]
    media_columns = [
        channel
        for channel in panel.columns
        if channel not in {geo_col, date_col, outcome, "cell"}
    ]
    slopes = _pre_slopes(panel.loc[pre], outcome, media_columns, geo_col, date_col, media_columns)
    rows = []
    for cell, label in bundles.items():
        dark_channels = set(label.split("+")) if label else set()
        still_on = [channel for channel in media_columns if channel not in dark_channels]
        for channel in still_on:
            shift = _impression_shift(panel, channel, cell, start)
            sales_shift = slopes[channel] * shift
            bundle_effect = float(effects[cell]) if cell in effects.index else float("nan")
            outcome_scale = float(panel.loc[pre, outcome].std(ddof=1))
            scale = max(abs(bundle_effect), outcome_scale, 1e-8)
            rows.append(
                {
                    "cell": cell,
                    "channel": channel,
                    "impression_shift": shift,
                    "sales_shift": sales_shift,
                    "bundle_effect": bundle_effect,
                    "share": abs(sales_shift) / scale,
                    "adjust": bool(abs(sales_shift) > material * scale),
                }
            )
    return pd.DataFrame(rows)


def _bundles(design: ConcurrentDesign) -> dict[str, str]:
    labels = {}
    for cell, group in design.schedule.groupby("cell", sort=True):
        off = group.loc[group["off"] == 1, "channel"].tolist()
        labels[str(cell)] = "+".join(off)
    return labels


def _panel(outcomes, impressions, design, outcome, geo_col, date_col, channel_col, impression_col) -> pd.DataFrame:
    wide = impressions.pivot_table(
        index=[geo_col, date_col],
        columns=channel_col,
        values=impression_col,
        aggfunc=_sum_or_missing,
    ).reset_index()
    wide[date_col] = pd.to_datetime(wide[date_col])
    frame = outcomes[[geo_col, date_col, outcome]].copy()
    frame[date_col] = pd.to_datetime(frame[date_col])
    frame = frame.merge(wide, on=[geo_col, date_col], how="inner")
    assignment = design.assignment.drop_duplicates(geo_col).set_index(geo_col)["cell"]
    frame["cell"] = frame[geo_col].map(assignment)
    return frame


def _sum_or_missing(values: pd.Series) -> float:
    """Sum recorded impressions. A missing value stays missing."""
    if values.isna().any():
        return float("nan")
    return float(values.sum())


def warn_if_impressions_missing(impressions: pd.DataFrame, impression_col: str) -> None:
    """Tell the caller that a blank impression was not filled in as zero."""
    if impression_col in impressions.columns and impressions[impression_col].isna().any():
        warnings.warn(
            "missing impressions were left missing; those market-days are dropped from the channel fit",
            UserWarning,
            stacklevel=2,
        )


def _impression_shift(panel: pd.DataFrame, channel: str, cell: str, start: pd.Timestamp) -> float:
    def mean(group: str, post: bool) -> float:
        mask = panel["cell"] == group
        mask = mask & (panel["date"] >= start if post else panel["date"] < start)
        return float(panel.loc[mask, channel].mean())

    return (mean(cell, True) - mean(cell, False)) - (mean("control", True) - mean("control", False))


def _pre_slopes(
    pre: pd.DataFrame,
    outcome: str,
    channels: list[str],
    geo_col: str,
    date_col: str,
    controls: list[str],
) -> dict[str, float]:
    """Value still-on shifts at the joint pre-test slope, not a one-channel slope."""
    from causalgeo.panel_ols import _twoway_demean

    slopes = {channel: 0.0 for channel in channels}
    columns = [channel for channel in controls if channel in pre.columns]
    if pre.empty or not columns:
        return slopes
    work = pre[[geo_col, date_col, outcome, *columns]].dropna()
    if len(work) <= len(columns) + 1:
        return slopes
    demeaned = _twoway_demean(work, geo_col, date_col, [outcome, *columns])
    y = demeaned[outcome].to_numpy(dtype=float)
    x = demeaned[columns].to_numpy(dtype=float)
    alive = [index for index in range(len(columns)) if float(np.std(x[:, index])) > 1e-8]
    if not alive:
        return slopes
    fitted = np.linalg.lstsq(x[:, alive], y, rcond=1e-8)[0]
    found = {columns[index]: float(coef) for index, coef in zip(alive, fitted)}
    for channel in channels:
        slopes[channel] = found.get(channel, 0.0)
    return slopes

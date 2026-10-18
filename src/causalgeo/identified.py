"""Estimate only the contrasts a concurrent geo design can name.

Each cell coefficient is the effect of that cell's bundle going off.
A channel gets its own number only when one cell turns that channel off
and nothing else. Tied channels stay with the bundle.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from causalgeo.design import ConcurrentDesign
from causalgeo.panel_ols import PanelEstimate, run_panel_ols_did


class NotIdentified(ValueError):
    """The design does not name this channel on its own."""


@dataclass(frozen=True)
class IdentifiedEstimate:
    """Cell effects, and a channel table that withholds numbers the design cannot support."""

    cells: pd.DataFrame
    channels: pd.DataFrame
    estimate: PanelEstimate


def estimate_identified(
    design: ConcurrentDesign,
    metrics: pd.DataFrame,
    outcome: str,
    treatment_start: str,
    geo_col: str = "geo",
    date_col: str = "date",
    experiment_start: str | None = None,
    experiment_end: str | None = None,
) -> IdentifiedEstimate:
    """Fit one holdout flag per cell. Do not fit spend, and do not split a tied channel."""
    _check_metrics(metrics, outcome, geo_col, date_col)
    panel, did_cols = _cell_panel(
        design,
        metrics,
        outcome,
        geo_col,
        date_col,
        treatment_start,
        experiment_start,
        experiment_end,
    )
    estimate = run_panel_ols_did(panel, outcome_col=outcome, treatment_col=did_cols)
    summary = estimate.summary_frame()
    bundles = _bundles(design.schedule)
    cells = _cell_table(summary, did_cols, bundles)
    channels = _channel_table(design.separation, cells, bundles)
    return IdentifiedEstimate(cells=cells, channels=channels, estimate=estimate)


def channel_effect(result: IdentifiedEstimate, channel: str) -> float:
    """Return the channel's own effect, or refuse when the design cannot name it."""
    row = result.channels.loc[result.channels["channel"] == channel]
    if row.empty:
        raise NotIdentified(f"{channel} is not in this design")
    reason = str(row.iloc[0]["reason"])
    if reason:
        raise NotIdentified(reason)
    return float(row.iloc[0]["coef"])


def _check_metrics(metrics: pd.DataFrame, outcome: str, geo_col: str, date_col: str) -> None:
    if outcome in {"spend", "cost", "impressions"}:
        raise ValueError("spend is not an outcome of the holdout; pass the sales or visits column")
    missing = [col for col in (geo_col, date_col, outcome) if col not in metrics.columns]
    if missing:
        raise ValueError(f"metrics is missing {missing}")


def _cell_panel(
    design: ConcurrentDesign,
    metrics: pd.DataFrame,
    outcome: str,
    geo_col: str,
    date_col: str,
    treatment_start: str,
    experiment_start: str | None,
    experiment_end: str | None,
) -> tuple[pd.DataFrame, list[str]]:
    assignment = design.assignment
    if geo_col not in assignment.columns:
        raise ValueError(f"assignment has no '{geo_col}' column")
    cell_of = assignment.drop_duplicates(geo_col).set_index(geo_col)["cell"]
    panel = metrics.loc[metrics[geo_col].isin(cell_of.index), [geo_col, date_col, outcome]].copy()
    if panel.empty:
        raise ValueError("no markets in the metrics are in the design")
    panel["dmacode"] = panel[geo_col]
    panel["date"] = pd.to_datetime(panel[date_col])
    if experiment_start is not None:
        panel = panel.loc[panel["date"] >= pd.Timestamp(experiment_start)]
    if experiment_end is not None:
        panel = panel.loc[panel["date"] <= pd.Timestamp(experiment_end)]
    if panel.empty:
        raise ValueError("no rows remain in the experiment window")
    panel["cell"] = panel["dmacode"].map(cell_of)
    post = panel["date"] >= pd.Timestamp(treatment_start)
    if not post.any() or post.all():
        raise ValueError("the window needs days before the test and days during the test")
    test_cells = [cell for cell in sorted(panel["cell"].unique()) if cell != "control"]
    if not test_cells:
        raise ValueError("the design has no test markets")
    did_cols = []
    for cell in test_cells:
        column = f"did_cell_{cell}"
        panel[column] = ((panel["cell"] == cell) & post).astype(float)
        did_cols.append(column)
    return panel, did_cols


def _bundles(schedule: pd.DataFrame) -> dict[str, str]:
    labels = {}
    for cell, group in schedule.groupby("cell", sort=True):
        off = group.loc[group["off"] == 1, "channel"].tolist()
        labels[str(cell)] = "+".join(off) if off else ""
    return labels


def _cell_table(summary: pd.DataFrame, did_cols: list[str], bundles: dict[str, str]) -> pd.DataFrame:
    rows = []
    for column in did_cols:
        cell = column.removeprefix("did_cell_")
        coef = summary.loc[column]
        rows.append(
            {
                "cell": cell,
                "bundle": bundles.get(cell, ""),
                "coef": float(coef["coef"]),
                "std_err": float(coef["std_err"]),
                "p_value": float(coef["p_value"]),
                "ci_lower": float(coef["ci_lower"]),
                "ci_upper": float(coef["ci_upper"]),
            }
        )
    return pd.DataFrame(rows)


def _singleton_cells(bundles: dict[str, str]) -> dict[str, list[str]]:
    found: dict[str, list[str]] = {}
    for cell, label in bundles.items():
        parts = [part for part in label.split("+") if part]
        if len(parts) == 1:
            found.setdefault(parts[0], []).append(cell)
    return found


def _channel_table(
    separation: pd.DataFrame,
    cells: pd.DataFrame,
    bundles: dict[str, str],
) -> pd.DataFrame:
    singletons = _singleton_cells(bundles)
    coef_of = cells.set_index("cell")["coef"]
    rows = []
    for record in separation.itertuples(index=False):
        channel = record.channel
        status = record.status
        owners = singletons.get(channel, [])
        reason = _refusal_reason(channel, status, record.tied_with, owners)
        coef = float(coef_of[owners[0]]) if not reason and len(owners) == 1 else float("nan")
        cell = owners[0] if not reason and len(owners) == 1 else ""
        rows.append(
            {
                "channel": channel,
                "status": status,
                "cell": cell,
                "bundle": bundles.get(cell, ""),
                "coef": coef,
                "reason": reason,
            }
        )
    return pd.DataFrame(rows)


def _refusal_reason(channel: str, status: str, tied_with: str, owners: list[str]) -> str:
    if status == "not tested":
        return f"{channel} was never turned off"
    if status == "tied":
        partners = tied_with or "another channel"
        return f"{channel} is tied with {partners}; the design names the bundle, not {channel} alone"
    if status != "separate":
        return f"{channel} is not separable from the other channels in this design"
    if len(owners) == 0:
        return f"{channel} is only identified as a mix of cells, not by a cell that turns it off alone"
    if len(owners) > 1:
        listed = ", ".join(owners)
        return f"{channel} is turned off alone in more than one cell ({listed})"
    return ""

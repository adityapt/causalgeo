"""Pretest / test panel at the geo × date grain."""

from __future__ import annotations

from typing import Optional

import pandas as pd

from causalgeo.assignment import AssignmentMap


def build_panel(
    data: pd.DataFrame,
    assignment: AssignmentMap,
    *,
    location: str = "location",
    date: str = "date",
    conversions: str = "conversions",
    treatment_start,
    treatment_end=None,
    spend: Optional[str] = None,
    impressions: Optional[str] = None,
) -> pd.DataFrame:
    """Restrict to experimental geos and add treatment / post indicators.

    Column names follow GeoX: ``location``, ``date``, ``conversions``,
    optional ``spend``. GeoLift users can pass ``location="geo"``.
    """
    need = {location, date, conversions}
    missing = need - set(data.columns)
    if missing:
        raise ValueError(f"Metric data missing columns: {sorted(missing)}")
    assignment.require_control()
    geos = assignment.experimental_geos
    panel = data.loc[data[location].astype(str).isin(geos)].copy()
    panel[location] = panel[location].astype(str)
    panel[date] = pd.to_datetime(panel[date])
    start = pd.to_datetime(treatment_start)
    panel["is_treatment"] = panel[location].isin(assignment.treatment_geos).astype(int)
    panel["is_post"] = (panel[date] >= start).astype(int)
    if treatment_end is not None:
        end = pd.to_datetime(treatment_end)
        panel = panel.loc[panel[date] <= end]
    panel["treated"] = panel["is_treatment"] * panel["is_post"]
    if spend is not None:
        if spend not in panel.columns:
            raise ValueError(f"Spend column {spend!r} not in data")
        panel["_spend"] = pd.to_numeric(panel[spend], errors="coerce")
    if impressions is not None:
        if impressions not in panel.columns:
            raise ValueError(f"Impressions column {impressions!r} not in data")
        panel["_impressions"] = pd.to_numeric(panel[impressions], errors="coerce")
        pre = panel.loc[panel["is_post"] == 0]
        baseline = pre.groupby(location)["_impressions"].mean()
        panel["_dose_ratio"] = panel[location].map(baseline)
        panel["_dose_ratio"] = panel["_impressions"] / panel["_dose_ratio"]
        panel["did_dose"] = panel["is_post"] * (panel["_dose_ratio"] - 1.0)
    return panel.sort_values([location, date]).reset_index(drop=True)

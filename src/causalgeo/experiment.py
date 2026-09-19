"""Geo experiment analysis — GeoX-style design object + analyze()."""

from __future__ import annotations

from typing import Optional, Union

import pandas as pd

from causalgeo.assignment import AssignmentMap
from causalgeo.estimate import LiftReport, estimate_att
from causalgeo.exceptions import MissingSpendError
from causalgeo.interventions import Intervention
from causalgeo.panel import build_panel


class GeoExperiment:
    """Analyze a geo experiment given assignment and geo-date KPI series.

    Parameters
    ----------
    data :
        Daily (or finer) panel with ``location``, ``date``, ``conversions``.
        Optional ``spend`` and ``impressions`` (delivered, not planned).
    assignment :
        Treatment and control geos. Control is required.
    treatment_start :
        First date of the test period (post / intervention).
    treatment_end :
        Optional last date of the test period.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        assignment: AssignmentMap,
        treatment_start,
        treatment_end=None,
        *,
        location: str = "location",
        date: str = "date",
        conversions: str = "conversions",
        spend: Optional[str] = None,
        impressions: Optional[str] = None,
    ) -> None:
        assignment.require_control()
        self.data = data
        self.assignment = assignment
        self.treatment_start = treatment_start
        self.treatment_end = treatment_end
        self.location = location
        self.date = date
        self.conversions = conversions
        self.spend = spend
        self.impressions = impressions

    def analyze(
        self,
        intervention: Union[Intervention, str] = Intervention.GO_DARK,
        *,
        use_impressions_as_dose: bool = False,
        require_icpd: bool = False,
    ) -> LiftReport:
        """Estimate lift for the stated intervention.

        ``intervention`` uses GeoX names: ``go_dark``, ``holdback``, ``heavy_up``.
        The ATT is the effect of being treated in the test period.

        If ``impressions`` were provided and ``use_impressions_as_dose`` is True,
        the regressor is delivered-impression dose relative to pretest (heavy-up
        / dose-response), not a 0/1 treated flag.
        """
        if isinstance(intervention, str):
            intervention = Intervention(intervention.lower().replace("-", "_"))
        if require_icpd and not self.spend:
            raise MissingSpendError(
                "iCPD / iROAS needs a spend series (GeoX: spend by geo and date)."
            )
        panel = build_panel(
            self.data,
            self.assignment,
            location=self.location,
            date=self.date,
            conversions=self.conversions,
            treatment_start=self.treatment_start,
            treatment_end=self.treatment_end,
            spend=self.spend,
            impressions=self.impressions,
        )
        term = "treated"
        if use_impressions_as_dose:
            if "did_dose" not in panel.columns:
                raise ValueError("Pass impressions= to use delivered dose")
            if intervention != Intervention.HEAVY_UP:
                raise ValueError("Impression dose is for heavy_up (incremental activity)")
            term = "did_dose"
        spend_col = "_spend" if self.spend else None
        return estimate_att(
            panel,
            conversions=self.conversions,
            location=self.location,
            date=self.date,
            treatment_term=term,
            spend_col=spend_col,
        )
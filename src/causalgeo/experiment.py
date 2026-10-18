"""Geo experiment analysis — GeoX-style design object + analyze()."""

from __future__ import annotations

from typing import Optional, Sequence, Union

import pandas as pd

from causalgeo.assignment import AssignmentMap
from causalgeo.counterfactual import estimate_sc, estimate_tbr
from causalgeo.estimate import LiftReport, estimate_att
from causalgeo.exceptions import MissingSpendError
from causalgeo.interventions import Intervention
from causalgeo.panel import build_panel
from causalgeo.sdid import estimate_sdid

_COUNTERFACTUALS = frozenset({"sdid", "tbr", "twfe", "sc"})


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
        counterfactual: str = "tbr",
        sdid_placebo_replications: int = 200,
        random_state: Optional[int] = 0,
    ) -> LiftReport:
        """Estimate lift for the stated intervention.

        ``intervention`` uses GeoX names: ``go_dark``, ``holdback``, ``heavy_up``.
        ``counterfactual`` is the measurement: ``tbr`` (default; pretest path),
        ``sc`` (synthetic control on the treated mean), ``sdid`` (synthetic
        difference-in-differences), or ``twfe`` (geo + date FE).

        If ``impressions`` were provided and ``use_impressions_as_dose`` is True,
        the regressor is delivered-impression dose relative to pretest (heavy-up
        / dose-response). Dose is TWFE-only.
        """
        if isinstance(intervention, str):
            intervention = Intervention(intervention.lower().replace("-", "_"))
        method = counterfactual.strip().lower()
        if method not in _COUNTERFACTUALS:
            raise ValueError(f"counterfactual must be one of {sorted(_COUNTERFACTUALS)}")
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
        spend_col = "_spend" if self.spend else None
        if use_impressions_as_dose:
            if "did_dose" not in panel.columns:
                raise ValueError("Pass impressions= to use delivered dose")
            if intervention != Intervention.HEAVY_UP:
                raise ValueError("Impression dose is for heavy_up (incremental activity)")
            if method != "twfe":
                raise ValueError("Impression dose is TWFE-only; pass counterfactual='twfe'")
            return estimate_att(
                panel,
                conversions=self.conversions,
                location=self.location,
                date=self.date,
                treatment_term="did_dose",
                spend_col=spend_col,
            )
        if method == "tbr":
            return estimate_tbr(
                panel,
                conversions=self.conversions,
                location=self.location,
                date=self.date,
                spend_col=spend_col,
            )
        if method == "sc":
            return estimate_sc(
                panel,
                conversions=self.conversions,
                location=self.location,
                date=self.date,
                spend_col=spend_col,
            )
        if method == "sdid":
            return estimate_sdid(
                panel,
                conversions=self.conversions,
                location=self.location,
                date=self.date,
                spend_col=spend_col,
                placebo_replications=sdid_placebo_replications,
                random_state=random_state,
            )
        return estimate_att(
            panel,
            conversions=self.conversions,
            location=self.location,
            date=self.date,
            treatment_term="treated",
            spend_col=spend_col,
        )

    def placebo(
        self,
        *,
        location: object,
        counterfactual: str = "sdid",
        sdid_placebo_replications: int = 200,
        random_state: Optional[int] = 0,
    ) -> LiftReport:
        """Fake-treat one donor geo to test the measurement procedure.

        Original treated geos are excluded: a placebo counterfactual must be
        built only from geos that were untreated in the real experiment.
        """
        placebo_geo = str(location)
        if placebo_geo not in self.assignment.control_geos:
            raise ValueError("placebo location must be one of the control geos")
        donors = self.assignment.control_geos - {placebo_geo}
        placebo_assignment = AssignmentMap.from_sets([placebo_geo], donors)
        placebo_experiment = GeoExperiment(
            self.data,
            placebo_assignment,
            self.treatment_start,
            self.treatment_end,
            location=self.location,
            date=self.date,
            conversions=self.conversions,
            spend=self.spend,
            impressions=self.impressions,
        )
        return placebo_experiment.analyze(
            counterfactual=counterfactual,
            sdid_placebo_replications=sdid_placebo_replications,
            random_state=random_state,
        )


    def diagnose(
        self,
        metrics: Optional[Sequence] = None,
        intervention: Union[Intervention, str] = Intervention.GO_DARK,
        *,
        counterfactual: str = "tbr",
        adjust: str = "bh",
        bh_q: float = 0.10,
    ):
        from causalgeo.diagnostics import diagnose as _diagnose

        return _diagnose(
            self,
            metrics=metrics,
            intervention=intervention,
            counterfactual=counterfactual,
            adjust=adjust,
            bh_q=bh_q,
        )
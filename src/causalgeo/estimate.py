"""Two-way fixed effects lift (geo FE + time FE), SEs clustered by geo."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import pandas as pd
import statsmodels.api as sm


@dataclass
class LiftReport:
    """Analysis output in GeoX / GeoLift vocabulary.

    ``lift`` is the ATT of the intervention on the conversion KPI (per geo-day).
    ``percent_lift`` is lift relative to mean control conversions in the test
    period. ``icpd`` is incremental conversions per dollar when spend is
    present (iROAS if ``conversions`` are revenue); its denominator is
    ``spend_delta``, the difference-in-differences of spend.
    """

    lift: float
    std_error: float
    p_value: float
    ci_lower: float
    ci_upper: float
    percent_lift: float
    nobs: int
    n_treatment_geos: int
    n_control_geos: int
    cov_type: str
    icpd: Optional[float] = None
    spend_delta: Optional[float] = None
    treatment_term: str = "treated"
    method: str = "twfe"
    pre_r2: Optional[float] = None

    def summary_frame(self) -> pd.DataFrame:
        row = {
            "lift": self.lift,
            "std_error": self.std_error,
            "p_value": self.p_value,
            "ci_lower": self.ci_lower,
            "ci_upper": self.ci_upper,
            "percent_lift": self.percent_lift,
            "icpd": self.icpd,
            "method": self.method,
            "pre_r2": self.pre_r2,
        }
        return pd.DataFrame([row])


def _twoway_demean(
    frame: pd.DataFrame,
    entity: str,
    time: str,
    columns: list[str],
) -> pd.DataFrame:
    entity_means = frame.groupby(entity)[columns].transform("mean")
    time_means = frame.groupby(time)[columns].transform("mean")
    grand = frame[columns].mean()
    return frame[columns] - entity_means - time_means + grand


def _incremental_spend(panel: pd.DataFrame, spend_col: Optional[str]) -> Optional[float]:
    """Difference-in-differences of spend, per geo-day.

    The iCPD denominator has to be the spend *caused by the intervention*:
    ``(treatment post - treatment pre) - (control post - control pre)``.
    A plain post-period gap between groups would also pick up any baseline
    spend imbalance between treatment and control geos.
    """
    if not spend_col or spend_col not in panel.columns:
        return None
    cells = {}
    for treat in (0, 1):
        for post in (0, 1):
            series = panel.loc[
                (panel["is_treatment"] == treat) & (panel["is_post"] == post), spend_col
            ]
            if series.empty:
                return None
            cells[(treat, post)] = float(series.mean())
    delta = (cells[(1, 1)] - cells[(1, 0)]) - (cells[(0, 1)] - cells[(0, 0)])
    if math.isnan(delta):
        return None
    return delta


def estimate_att(
    panel: pd.DataFrame,
    *,
    conversions: str = "conversions",
    location: str = "location",
    date: str = "date",
    treatment_term: str = "treated",
    spend_col: Optional[str] = None,
) -> LiftReport:
    """Y ~ treated + geo FE + date FE, cluster-robust SEs by geo."""
    work = panel[[conversions, location, date, treatment_term]].dropna()
    demeaned = _twoway_demean(work, location, date, [conversions, treatment_term])
    y = demeaned[conversions]
    x = demeaned[[treatment_term]]
    x = x.loc[:, x.std() > 1e-12]
    if x.empty:
        raise ValueError("Treatment is collinear with geo and time fixed effects")
    model = sm.OLS(y, x, hasconst=False)
    fitted = model.fit(
        cov_type="cluster",
        cov_kwds={"groups": work.loc[x.index, location]},
    )
    term = treatment_term
    att = float(fitted.params[term])
    se = float(fitted.bse[term])
    pval = float(fitted.pvalues[term])
    ci = fitted.conf_int().loc[term]
    ctrl_post = panel.loc[(panel["is_treatment"] == 0) & (panel["is_post"] == 1), conversions]
    base = float(ctrl_post.mean()) if len(ctrl_post) else math.nan
    pct = att / base if base and not math.isnan(base) else math.nan
    spend_delta = _incremental_spend(panel, spend_col)
    icpd = None
    if spend_delta is not None and abs(spend_delta) > 1e-12:
        icpd = att / spend_delta
    return LiftReport(
        lift=att,
        std_error=se,
        p_value=pval,
        ci_lower=float(ci.iloc[0]),
        ci_upper=float(ci.iloc[1]),
        percent_lift=pct,
        nobs=int(fitted.nobs),
        n_treatment_geos=int(panel.loc[panel["is_treatment"] == 1, location].nunique()),
        n_control_geos=int(panel.loc[panel["is_treatment"] == 0, location].nunique()),
        cov_type="cluster",
        icpd=icpd,
        spend_delta=spend_delta,
        treatment_term=term,
        method="twfe",
    )

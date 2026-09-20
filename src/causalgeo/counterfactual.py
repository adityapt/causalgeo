"""Pretest-fit counterfactuals: time-based regression and synthetic control.

TWFE only removes a geo level and a common calendar. These estimators use the
pretest *path*: how treated conversions moved with control conversions (TBR)
or with a weighted donor pool (SC) before the intervention.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from scipy.optimize import minimize

from causalgeo.estimate import LiftReport, _incremental_spend


def _daily_means(
    panel: pd.DataFrame,
    conversions: str,
    date: str,
) -> pd.DataFrame:
    daily = panel.groupby([date, "is_treatment"], as_index=False)[conversions].mean()
    treat = daily.loc[daily["is_treatment"] == 1, [date, conversions]].rename(
        columns={conversions: "y_treat"}
    )
    ctrl = daily.loc[daily["is_treatment"] == 0, [date, conversions]].rename(
        columns={conversions: "y_ctrl"}
    )
    out = treat.merge(ctrl, on=date)
    post = panel.groupby(date, as_index=False)["is_post"].max()
    return out.merge(post, on=date).sort_values(date)


def _finish(
    att: float,
    se: float,
    df: float,
    panel: pd.DataFrame,
    conversions: str,
    location: str,
    spend_col: Optional[str],
    *,
    nobs: int,
    cov_type: str,
    method: str,
    treatment_term: str,
    pre_r2: Optional[float],
) -> LiftReport:
    if se <= 0 or math.isnan(se):
        pval = math.nan
        lo, hi = math.nan, math.nan
    else:
        tcrit = float(stats.t.ppf(0.975, df)) if df > 1 else 1.96
        pval = float(2 * stats.t.sf(abs(att / se), df))
        lo, hi = att - tcrit * se, att + tcrit * se
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
        ci_lower=lo,
        ci_upper=hi,
        percent_lift=pct,
        nobs=nobs,
        n_treatment_geos=int(panel.loc[panel["is_treatment"] == 1, location].nunique()),
        n_control_geos=int(panel.loc[panel["is_treatment"] == 0, location].nunique()),
        cov_type=cov_type,
        icpd=icpd,
        spend_delta=spend_delta,
        treatment_term=treatment_term,
        method=method,
        pre_r2=pre_r2,
    )


def estimate_tbr(
    panel: pd.DataFrame,
    *,
    conversions: str = "conversions",
    location: str = "location",
    date: str = "date",
    spend_col: Optional[str] = None,
) -> LiftReport:
    """GeoX-style time-based regression on daily treatment vs control means.

    Pretest: ``mean(Y | treated geos) = a + b * mean(Y | control geos)``.
    Test period lift is the average gap vs that fitted path.
    """
    series = _daily_means(panel, conversions, date)
    pre = series.loc[series["is_post"] == 0]
    post = series.loc[series["is_post"] == 1]
    if len(pre) < 5:
        raise ValueError("TBR needs at least 5 pretest dates")
    if len(post) < 1:
        raise ValueError("TBR needs test-period dates")
    if float(pre["y_ctrl"].std()) < 1e-12:
        raise ValueError("Control pretest series has no variation for TBR")
    x = sm.add_constant(pre["y_ctrl"], has_constant="add")
    fitted = sm.OLS(pre["y_treat"], x).fit()
    a = float(fitted.params["const"])
    b = float(fitted.params["y_ctrl"])
    xbar_post = float(post["y_ctrl"].mean())
    att = float(post["y_treat"].mean() - a - b * xbar_post)
    cov = fitted.cov_params().loc[["const", "y_ctrl"], ["const", "y_ctrl"]].to_numpy(dtype=float)
    grad = np.array([1.0, xbar_post])
    var_cf = float(grad @ cov @ grad)
    var_avg = float(fitted.scale) / len(post)
    se = math.sqrt(max(var_cf + var_avg, 0.0))
    return _finish(
        att,
        se,
        df=float(fitted.df_resid),
        panel=panel,
        conversions=conversions,
        location=location,
        spend_col=spend_col,
        nobs=int(len(series)),
        cov_type="tbr",
        method="tbr",
        treatment_term="tbr_gap",
        pre_r2=float(fitted.rsquared),
    )


def _simplex_weights(donors: np.ndarray, treated: np.ndarray) -> np.ndarray:
    n_donors = donors.shape[1]
    if n_donors == 1:
        return np.array([1.0])

    def loss(w: np.ndarray) -> float:
        err = treated - donors @ w
        return float(err @ err)

    w0 = np.ones(n_donors) / n_donors
    result = minimize(
        loss,
        w0,
        bounds=[(0.0, 1.0)] * n_donors,
        constraints={"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)},
        method="SLSQP",
        options={"maxiter": 500, "ftol": 1e-12},
    )
    w = np.clip(result.x if result.success else w0, 0.0, 1.0)
    s = float(w.sum())
    return w / s if s > 0 else w0


def estimate_sc(
    panel: pd.DataFrame,
    *,
    conversions: str = "conversions",
    location: str = "location",
    date: str = "date",
    spend_col: Optional[str] = None,
) -> LiftReport:
    """Synthetic control on the treated-geo mean, donors = control geos.

    Weights are nonnegative and sum to one, fit on pretest dates only.
    """
    pre = panel.loc[panel["is_post"] == 0]
    post = panel.loc[panel["is_post"] == 1]
    if pre.empty or post.empty:
        raise ValueError("Synthetic control needs pretest and test-period rows")
    y_pre = pre.pivot_table(index=date, columns=location, values=conversions)
    roles = pre.groupby(location)["is_treatment"].first()
    treat_cols = [c for c in y_pre.columns if int(roles.get(c, 0)) == 1]
    ctrl_cols = [c for c in y_pre.columns if int(roles.get(c, 0)) == 0]
    if not treat_cols or not ctrl_cols:
        raise ValueError("Synthetic control needs treatment and control geos")
    treated_pre = y_pre[treat_cols].mean(axis=1).to_numpy(dtype=float)
    donors_pre = y_pre[ctrl_cols].to_numpy(dtype=float)
    mask = np.isfinite(treated_pre) & np.isfinite(donors_pre).all(axis=1)
    weights = _simplex_weights(donors_pre[mask], treated_pre[mask])
    y_post = post.pivot_table(index=date, columns=location, values=conversions)
    treated_post = y_post[treat_cols].mean(axis=1).to_numpy(dtype=float)
    donors_post = y_post.reindex(columns=ctrl_cols).to_numpy(dtype=float)
    gap_post = treated_post - donors_post @ weights
    gap_pre = treated_pre[mask] - donors_pre[mask] @ weights
    att = float(np.nanmean(gap_post))
    sigma = float(np.std(gap_pre, ddof=1)) if len(gap_pre) > 1 else float(np.std(gap_pre))
    n_pre, n_post = max(len(gap_pre), 1), max(len(gap_post), 1)
    se = sigma * math.sqrt(1.0 / n_post + 1.0 / n_pre)
    ss_tot = float(np.sum((treated_pre[mask] - treated_pre[mask].mean()) ** 2))
    pre_r2 = 1.0 - float(np.sum(gap_pre**2)) / ss_tot if ss_tot > 0 else math.nan
    return _finish(
        att,
        se,
        df=float(max(n_pre - 1, 1)),
        panel=panel,
        conversions=conversions,
        location=location,
        spend_col=spend_col,
        nobs=int(panel.dropna(subset=[conversions]).shape[0]),
        cov_type="sc",
        method="sc",
        treatment_term="sc_gap",
        pre_r2=pre_r2,
    )

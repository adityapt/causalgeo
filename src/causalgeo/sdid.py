"""Synthetic difference-in-differences for balanced geo panels."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from causalgeo.counterfactual import _finish
from causalgeo.estimate import LiftReport


@dataclass(frozen=True)
class _SdidFit:
    lift: float
    pre_r2: float
    unit_weights: np.ndarray
    time_weights: np.ndarray


def _simplex_weights(
    design: np.ndarray,
    target: np.ndarray,
    *,
    penalty: float,
) -> tuple[np.ndarray, float]:
    """Least-squares simplex weights with an unpenalized intercept."""
    n_weights = design.shape[1]
    objective_scale = max(
        float(np.mean(np.square(design))),
        float(np.mean(np.square(target))),
        float(penalty),
        1.0,
    )

    def objective(weights: np.ndarray) -> float:
        gap = design @ weights - target
        centered = gap - gap.mean()
        return float(
            (centered @ centered + penalty * (weights @ weights))
            / objective_scale
        )

    def gradient(weights: np.ndarray) -> np.ndarray:
        gap = design @ weights - target
        centered = gap - gap.mean()
        return (2.0 * design.T @ centered + 2.0 * penalty * weights) / objective_scale

    result = minimize(
        objective,
        np.repeat(1.0 / n_weights, n_weights),
        jac=gradient,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * n_weights,
        constraints={
            "type": "eq",
            "fun": lambda weights: weights.sum() - 1.0,
            "jac": lambda weights: np.ones_like(weights),
        },
        options={"ftol": 1e-12, "maxiter": 10_000},
    )
    if not result.success:
        raise RuntimeError(f"sDiD weight optimization failed: {result.message}")
    weights = np.clip(np.asarray(result.x, dtype=float), 0.0, None)
    weights /= weights.sum()
    gap = design @ weights - target
    intercept = -float(gap.mean())
    return weights, intercept


def _sdid_point(
    outcomes: np.ndarray,
    *,
    n_controls: int,
    n_pre: int,
) -> _SdidFit:
    n_units, n_periods = outcomes.shape
    n_treated = n_units - n_controls
    n_post = n_periods - n_pre
    if min(n_controls, n_treated, n_pre, n_post) < 1:
        raise ValueError("sDiD needs controls, treated geos, pretest, and test periods")

    control_pre = outcomes[:n_controls, :n_pre]
    control_post = outcomes[:n_controls, n_pre:]
    treated_pre = outcomes[n_controls:, :n_pre]
    treated_post = outcomes[n_controls:, n_pre:]

    differences = np.diff(control_pre, axis=1)
    noise = float(np.std(differences, ddof=1)) if differences.size > 1 else 0.0
    if not np.isfinite(noise) or noise <= 1e-12:
        noise = 1e-12

    zeta_unit = (n_treated * n_post) ** 0.25 * noise
    unit_weights, unit_intercept = _simplex_weights(
        control_pre.T,
        treated_pre.mean(axis=0),
        penalty=(zeta_unit**2) * n_pre,
    )

    zeta_time = 1e-6 * noise
    time_weights, _ = _simplex_weights(
        control_pre,
        control_post.mean(axis=1),
        penalty=(zeta_time**2) * n_controls,
    )

    treated_change = float(treated_post.mean() - treated_pre.mean(axis=0) @ time_weights)
    control_change = float(
        unit_weights @ control_post.mean(axis=1)
        - unit_weights @ control_pre @ time_weights
    )
    lift = treated_change - control_change

    treated_path = treated_pre.mean(axis=0)
    fitted_path = unit_intercept + unit_weights @ control_pre
    residual = treated_path - fitted_path
    total = treated_path - treated_path.mean()
    denominator = float(total @ total)
    pre_r2 = (
        float(1.0 - (residual @ residual) / denominator)
        if denominator > 1e-12
        else math.nan
    )
    return _SdidFit(
        lift=lift,
        pre_r2=pre_r2,
        unit_weights=unit_weights,
        time_weights=time_weights,
    )


def _balanced_matrix(
    panel: pd.DataFrame,
    *,
    conversions: str,
    location: str,
    date: str,
) -> tuple[np.ndarray, int, int]:
    required = {conversions, location, date, "is_treatment", "is_post"}
    missing = required - set(panel.columns)
    if missing:
        raise ValueError(f"sDiD panel is missing columns: {sorted(missing)}")

    work = panel[list(required)].copy()
    work[location] = work[location].astype(str)
    work[date] = pd.to_datetime(work[date])
    work[conversions] = pd.to_numeric(work[conversions], errors="raise")
    if work.duplicated([location, date]).any():
        raise ValueError("sDiD needs one outcome per geo and date")

    treatment_by_geo = work.groupby(location)["is_treatment"].nunique()
    if (treatment_by_geo != 1).any():
        raise ValueError("Treatment role must be constant within each geo")
    roles = work.groupby(location)["is_treatment"].first().astype(int)
    controls = sorted(roles[roles == 0].index)
    treated = sorted(roles[roles == 1].index)
    if not controls or not treated:
        raise ValueError("sDiD needs at least one control and one treated geo")

    wide = work.pivot(index=location, columns=date, values=conversions)
    dates = sorted(wide.columns)
    wide = wide.reindex(index=controls + treated, columns=dates)
    if wide.isna().any().any():
        raise ValueError("sDiD needs a balanced geo-by-date outcome panel")

    post_by_date = work.groupby(date)["is_post"].agg(["min", "max"]).reindex(dates)
    if (post_by_date["min"] != post_by_date["max"]).any():
        raise ValueError("sDiD v1 needs one common treatment start")
    post = post_by_date["max"].to_numpy(dtype=int)
    if not np.array_equal(post, np.sort(post)):
        raise ValueError("sDiD needs all pretest dates before all test dates")
    n_pre = int(np.sum(post == 0))
    return wide.to_numpy(dtype=float), len(controls), n_pre


def _placebo_standard_error(
    outcomes: np.ndarray,
    *,
    n_controls: int,
    n_pre: int,
    replications: int,
    random_state: Optional[int],
) -> float:
    """Algorithm 4 placebo uncertainty from untreated units."""
    if replications < 2:
        raise ValueError("sDiD placebo inference needs at least 2 replications")
    n_treated = outcomes.shape[0] - n_controls
    if n_controls <= n_treated:
        raise ValueError(
            "sDiD placebo inference needs more control geos than treated geos"
        )

    controls = outcomes[:n_controls]
    rng = np.random.default_rng(random_state)
    estimates = np.empty(replications, dtype=float)
    placebo_controls = n_controls - n_treated
    for index in range(replications):
        order = rng.permutation(n_controls)
        placebo = controls[order]
        estimates[index] = _sdid_point(
            placebo,
            n_controls=placebo_controls,
            n_pre=n_pre,
        ).lift
    return float(
        math.sqrt((replications - 1) / replications)
        * np.std(estimates, ddof=1)
    )


def estimate_sdid(
    panel: pd.DataFrame,
    *,
    conversions: str = "conversions",
    location: str = "location",
    date: str = "date",
    spend_col: Optional[str] = None,
    placebo_replications: int = 200,
    random_state: Optional[int] = 0,
) -> LiftReport:
    """Estimate ATT using Arkhangelsky et al. synthetic DiD.

    Inference uses control-only placebo assignments (Algorithm 4), which also
    works when only one geo is treated.
    """
    outcomes, n_controls, n_pre = _balanced_matrix(
        panel,
        conversions=conversions,
        location=location,
        date=date,
    )
    fit = _sdid_point(outcomes, n_controls=n_controls, n_pre=n_pre)
    standard_error = _placebo_standard_error(
        outcomes,
        n_controls=n_controls,
        n_pre=n_pre,
        replications=placebo_replications,
        random_state=random_state,
    )
    return _finish(
        fit.lift,
        standard_error,
        math.inf,
        panel,
        conversions,
        location,
        spend_col,
        nobs=len(panel),
        cov_type="placebo",
        method="sdid",
        treatment_term="treated",
        pre_r2=fit.pre_r2,
    )

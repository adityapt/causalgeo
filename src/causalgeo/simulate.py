"""Monte Carlo checks that TWFE lift is not a lucky one-draw recovery.

The DGP is the one TWFE is supposed to get right: geo FE + common time
shocks + a constant ATT in the test period. Errors are AR(1) within geo,
which is why SEs have to be clustered — iid OLS intervals will undercover.

Garbage would look like: large bias, 95% CIs that miss the truth most of
the time, or an A/A (true ATT = 0) that rejects far more than 5%.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import statsmodels.api as sm

from causalgeo.assignment import AssignmentMap
from causalgeo.estimate import LiftReport, _twoway_demean, estimate_att
from causalgeo.experiment import GeoExperiment
from causalgeo.interventions import Intervention
from causalgeo.panel import build_panel


@dataclass
class SimulationSummary:
    true_att: float
    n_sims: int
    mean_lift: float
    bias: float
    rmse: float
    coverage_clustered: float
    coverage_iid: float
    reject_clustered: float
    mean_icpd: Optional[float]
    true_icpd: Optional[float]


def simulate_panel(
    rng: np.random.Generator,
    *,
    n_treatment: int = 12,
    n_control: int = 12,
    n_pre: int = 21,
    n_post: int = 21,
    att: float = -4.0,
    rho: float = 0.6,
    sigma: float = 1.0,
    geo_fe_sd: float = 5.0,
    time_sd: float = 1.5,
    treatment_base_spend: float = 10.0,
    control_base_spend: float = 10.0,
) -> tuple[pd.DataFrame, datetime]:
    """One geo × date panel with a planted ATT and AR(1) shocks."""
    n_geo = n_treatment + n_control
    n_days = n_pre + n_post
    treat = np.array([1] * n_treatment + [0] * n_control)
    post = np.array([0] * n_pre + [1] * n_post)
    geo_fe = rng.normal(20.0, geo_fe_sd, size=n_geo)
    time_fe = rng.normal(0.0, time_sd, size=n_days)
    innov = rng.normal(0.0, sigma, size=(n_geo, n_days))
    err = np.zeros((n_geo, n_days))
    scale0 = 1.0 / np.sqrt(max(1e-12, 1.0 - rho**2))
    err[:, 0] = innov[:, 0] * scale0
    for t in range(1, n_days):
        err[:, t] = rho * err[:, t - 1] + innov[:, t]
    y = (
        geo_fe[:, None]
        + time_fe[None, :]
        + att * treat[:, None] * post[None, :]
        + err
    )
    start = datetime(2026, 1, 1)
    dates = [start + timedelta(days=i) for i in range(n_days)]
    treat_start = dates[n_pre]
    rows = []
    for i in range(n_geo):
        loc = f"T{i}" if treat[i] else f"C{i}"
        base = treatment_base_spend if treat[i] else control_base_spend
        for t, day in enumerate(dates):
            spend = 0.0 if (treat[i] and post[t]) else base
            rows.append(
                {
                    "location": loc,
                    "date": day,
                    "conversions": float(y[i, t]),
                    "spend": spend,
                }
            )
    return pd.DataFrame(rows), treat_start


def _iid_interval(panel: pd.DataFrame, conversions: str = "conversions") -> tuple[float, float]:
    """Same TWFE point estimate, iid SEs — the interval you must not trust."""
    work = panel[[conversions, "location", "date", "treated"]].dropna()
    demeaned = _twoway_demean(work, "location", "date", [conversions, "treated"])
    fitted = sm.OLS(demeaned[conversions], demeaned[["treated"]], hasconst=False).fit()
    ci = fitted.conf_int().loc["treated"]
    return float(ci.iloc[0]), float(ci.iloc[1])


def analyze_draw(
    data: pd.DataFrame,
    treatment_start,
    spend: str = "spend",
) -> tuple[LiftReport, tuple[float, float]]:
    treat = sorted(data.loc[data["location"].str.startswith("T"), "location"].unique())
    ctrl = sorted(data.loc[data["location"].str.startswith("C"), "location"].unique())
    assignment = AssignmentMap.from_sets(treat, ctrl)
    report = GeoExperiment(
        data, assignment, treatment_start, spend=spend
    ).analyze(Intervention.GO_DARK, counterfactual="twfe")
    panel = build_panel(
        data,
        assignment,
        treatment_start=treatment_start,
        spend=spend,
    )
    return report, _iid_interval(panel)


def run_monte_carlo(
    n_sims: int = 200,
    *,
    att: float = -4.0,
    seed: int = 7,
    **panel_kwargs,
) -> SimulationSummary:
    rng = np.random.default_rng(seed)
    lifts = []
    cover_cl = []
    cover_iid = []
    reject = []
    icpds = []
    spend_deltas = []
    for _ in range(n_sims):
        data, t0 = simulate_panel(rng, att=att, **panel_kwargs)
        report, iid_ci = analyze_draw(data, t0)
        lifts.append(report.lift)
        cover_cl.append(report.ci_lower <= att <= report.ci_upper)
        cover_iid.append(iid_ci[0] <= att <= iid_ci[1])
        reject.append(report.p_value < 0.05)
        if report.icpd is not None:
            icpds.append(report.icpd)
        if report.spend_delta is not None:
            spend_deltas.append(report.spend_delta)
    lifts_a = np.asarray(lifts, dtype=float)
    true_icpd = None
    mean_icpd = None
    if spend_deltas and abs(float(np.mean(spend_deltas))) > 1e-12:
        true_icpd = att / float(np.mean(spend_deltas))
        mean_icpd = float(np.mean(icpds)) if icpds else None
    return SimulationSummary(
        true_att=att,
        n_sims=n_sims,
        mean_lift=float(lifts_a.mean()),
        bias=float(lifts_a.mean() - att),
        rmse=float(np.sqrt(np.mean((lifts_a - att) ** 2))),
        coverage_clustered=float(np.mean(cover_cl)),
        coverage_iid=float(np.mean(cover_iid)),
        reject_clustered=float(np.mean(reject)),
        mean_icpd=mean_icpd,
        true_icpd=true_icpd,
    )


def main() -> None:
    print("A/A (true ATT = 0), 200 draws")
    aa = run_monte_carlo(n_sims=200, att=0.0)
    print(
        f"  bias={aa.bias:.3f}  clustered coverage={aa.coverage_clustered:.3f}  "
        f"iid coverage={aa.coverage_iid:.3f}  type I={aa.reject_clustered:.3f}"
    )
    print("Go-dark ATT = -4, 200 draws")
    gd = run_monte_carlo(n_sims=200, att=-4.0)
    print(
        f"  bias={gd.bias:.3f}  clustered coverage={gd.coverage_clustered:.3f}  "
        f"iid coverage={gd.coverage_iid:.3f}  power={gd.reject_clustered:.3f}  "
        f"icpd={gd.mean_icpd}"
    )


if __name__ == "__main__":
    main()

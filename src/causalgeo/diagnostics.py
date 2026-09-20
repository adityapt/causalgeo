"""Pretest diagnostics: balance, delivery, type-correct tests, MDE, sensitivity."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Sequence, Union

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm

from causalgeo.counterfactual import estimate_sc, estimate_tbr
from causalgeo.estimate import _incremental_spend, estimate_att
from causalgeo.interventions import Intervention
from causalgeo.metric_tests import (
    benjamini_hochberg,
    contingency_table,
    hedges_g,
    rate_two_by_two,
    welch_mannwhitney,
)
from causalgeo.metrics import MetricSpec
from causalgeo.panel import build_panel

_Z_MDE = float(norm.ppf(0.975) + norm.ppf(0.80))  # α=0.05 two-sided, 80% power


@dataclass
class DiagnosticsReport:
    metrics: list[MetricSpec]
    balance: pd.DataFrame
    pretest: pd.DataFrame
    metric_tests: pd.DataFrame
    contingency: dict[str, pd.DataFrame]
    power: pd.DataFrame
    sensitivity: pd.DataFrame
    gates: pd.DataFrame
    warnings: list[str] = field(default_factory=list)


def _primary_spec(metrics: Sequence[MetricSpec], default_name: str) -> MetricSpec:
    marked = [m for m in metrics if m.primary]
    if marked:
        return marked[0]
    for m in metrics:
        if m.name == default_name or m.value_column == default_name:
            return m
    return metrics[0]


def _geo_means(pre: pd.DataFrame, col: str, location: str) -> tuple[np.ndarray, np.ndarray]:
    g = pre.groupby([location, "is_treatment"], as_index=False)[col].mean()
    t = g.loc[g["is_treatment"] == 1, col].to_numpy(dtype=float)
    c = g.loc[g["is_treatment"] == 0, col].to_numpy(dtype=float)
    return t, c


def _pretrend(pre: pd.DataFrame, col: str, location: str, date: str) -> dict[str, float]:
    work = pre[[location, date, col, "is_treatment"]].dropna().copy()
    work["_t"] = (work[date] - work[date].min()).dt.days.astype(float)
    work["_tx"] = work["is_treatment"] * work["_t"]
    y = work[col]
    x = sm.add_constant(work[["is_treatment", "_t", "_tx"]])
    try:
        fitted = sm.OLS(y, x).fit(
            cov_type="cluster",
            cov_kwds={"groups": work[location]},
        )
        return {
            "pretrend_coef": float(fitted.params["_tx"]),
            "pretrend_se": float(fitted.bse["_tx"]),
            "pretrend_p": float(fitted.pvalues["_tx"]),
        }
    except Exception:
        return {"pretrend_coef": float("nan"), "pretrend_se": float("nan"), "pretrend_p": float("nan")}


def _spearman_series(pre: pd.DataFrame, col: str, date: str) -> float:
    daily = pre.groupby([date, "is_treatment"], as_index=False)[col].mean()
    t = daily.loc[daily["is_treatment"] == 1, [date, col]].rename(columns={col: "t"})
    c = daily.loc[daily["is_treatment"] == 0, [date, col]].rename(columns={col: "c"})
    m = t.merge(c, on=date)
    if len(m) < 3:
        return float("nan")
    return float(m["t"].corr(m["c"], method="spearman"))


def diagnose(
    experiment: Any,
    metrics: Optional[Sequence[MetricSpec]] = None,
    intervention: Union[Intervention, str] = Intervention.GO_DARK,
    *,
    counterfactual: str = "tbr",
    adjust: str = "bh",
    bh_q: float = 0.10,
) -> DiagnosticsReport:
    if isinstance(intervention, str):
        intervention = Intervention(intervention.lower().replace("-", "_"))
    if metrics is None:
        metrics = [MetricSpec(experiment.conversions, kind="continuous", primary=True)]
    metrics = list(metrics)
    primary = _primary_spec(metrics, experiment.conversions)

    panel = build_panel(
        experiment.data,
        experiment.assignment,
        location=experiment.location,
        date=experiment.date,
        conversions=experiment.conversions,
        treatment_start=experiment.treatment_start,
        treatment_end=experiment.treatment_end,
        spend=experiment.spend,
        impressions=experiment.impressions,
    )
    location, date = experiment.location, experiment.date
    pre = panel.loc[panel["is_post"] == 0]
    post = panel.loc[panel["is_post"] == 1]
    warnings: list[str] = []
    gates: list[dict] = []
    p_rows: list[dict] = []

    n_pre_dates = int(pre[date].nunique())
    n_post_dates = int(post[date].nunique())
    if n_pre_dates < 5:
        gates.append(
            {
                "check": "pretest_length",
                "status": "fail",
                "detail": f"{n_pre_dates} pretest dates; TBR needs ≥5",
            }
        )
    else:
        gates.append(
            {
                "check": "pretest_length",
                "status": "pass",
                "detail": f"{n_pre_dates} pretest dates, {n_post_dates} test dates",
            }
        )

    n_t = int(panel.loc[panel["is_treatment"] == 1, location].nunique())
    n_c = int(panel.loc[panel["is_treatment"] == 0, location].nunique())
    n_u = len(experiment.assignment.unassigned_geos)
    missing_from_panel = experiment.assignment.experimental_geos - set(
        panel[location].astype(str)
    )
    if missing_from_panel:
        warnings.append(f"Assignment geos missing from panel: {sorted(missing_from_panel)[:10]}")

    spend_col = "_spend" if experiment.spend else None
    spend_delta = _incremental_spend(panel, spend_col)
    if spend_col and spend_delta is not None:
        if intervention == Intervention.GO_DARK and spend_delta >= 0:
            gates.append(
                {
                    "check": "delivery",
                    "status": "warn",
                    "detail": f"go-dark spend DiD is {spend_delta:.4g} (expected negative)",
                }
            )
            warnings.append("delivery_not_detected")
        elif intervention in (Intervention.HOLDBACK, Intervention.HEAVY_UP) and spend_delta <= 0:
            gates.append(
                {
                    "check": "delivery",
                    "status": "warn",
                    "detail": f"{intervention.value} spend DiD is {spend_delta:.4g} (expected positive)",
                }
            )
            warnings.append("delivery_not_detected")
        else:
            gates.append(
                {
                    "check": "delivery",
                    "status": "pass",
                    "detail": f"spend DiD {spend_delta:.4g}",
                }
            )
    else:
        gates.append({"check": "delivery", "status": "pass", "detail": "no spend series"})

    balance_rows = []
    pretest_rows = []
    contingency: dict[str, pd.DataFrame] = {}

    for spec in metrics:
        if spec.kind == "categorical":
            cat_src = experiment.data.copy()
            cat_src[location] = cat_src[location].astype(str)
            cat_src[date] = pd.to_datetime(cat_src[date])
            start = pd.to_datetime(experiment.treatment_start)
            cat_pre = cat_src.loc[cat_src[date] < start]
            cat_pre = cat_pre.loc[
                cat_pre[location].isin(experiment.assignment.experimental_geos)
            ]
            cat_pre = cat_pre.copy()
            cat_pre["arm"] = np.where(
                cat_pre[location].isin(experiment.assignment.treatment_geos),
                "treatment",
                "control",
            )
            table, cstats = contingency_table(
                cat_pre, "arm", spec.categories, spec.counts
            )
            contingency[spec.name] = table
            p_rows.append(
                {
                    "metric": spec.name,
                    "test": cstats["method"],
                    "window": "pretest",
                    "estimand": "association_not_att",
                    "p_raw": cstats["p_raw"],
                    "stat": cstats["chi2"],
                }
            )
            continue

        col = spec.value_column
        if spec.kind == "rate":
            if spec.successes not in pre.columns or spec.trials not in pre.columns:
                raise ValueError(f"Rate metric {spec.name} missing columns")
            g = pre.groupby([location, "is_treatment"], as_index=False)[
                [spec.successes, spec.trials]
            ].sum()
            g["_rate"] = g[spec.successes] / g[spec.trials].replace(0, np.nan)
            t = g.loc[g["is_treatment"] == 1, "_rate"].to_numpy(dtype=float)
            c = g.loc[g["is_treatment"] == 0, "_rate"].to_numpy(dtype=float)
            s_t = float(g.loc[g["is_treatment"] == 1, spec.successes].sum())
            n_tt = float(g.loc[g["is_treatment"] == 1, spec.trials].sum())
            s_c = float(g.loc[g["is_treatment"] == 0, spec.successes].sum())
            n_cc = float(g.loc[g["is_treatment"] == 0, spec.trials].sum())
            table, cstats = rate_two_by_two(s_t, n_tt, s_c, n_cc)
            contingency[spec.name] = table
            p_rows.append(
                {
                    "metric": spec.name,
                    "test": cstats["method"],
                    "window": "pretest",
                    "estimand": "association_not_att",
                    "p_raw": cstats["p_raw"],
                    "stat": cstats["chi2"],
                }
            )
        else:
            if col not in pre.columns:
                raise ValueError(f"Metric column {col!r} not in panel")
            t, c = _geo_means(pre, col, location)

        g_hat = hedges_g(t, c)
        tests = welch_mannwhitney(t, c)
        rho = _spearman_series(pre, col if spec.kind != "rate" else spec.successes, date)
        trend = _pretrend(pre, col if spec.kind != "rate" else spec.successes, location, date)
        balance_rows.append(
            {
                "metric": spec.name,
                "kind": spec.kind,
                "n_treatment_geos": int(np.isfinite(t).sum()),
                "n_control_geos": int(np.isfinite(c).sum()),
                "mean_treatment": float(np.nanmean(t)),
                "mean_control": float(np.nanmean(c)),
                "sd_treatment": float(np.nanstd(t, ddof=1)) if len(t) > 1 else float("nan"),
                "sd_control": float(np.nanstd(c, ddof=1)) if len(c) > 1 else float("nan"),
                "hedges_g": g_hat,
                "variance_ratio": float(np.nanvar(t, ddof=1) / np.nanvar(c, ddof=1))
                if len(c) > 1 and np.nanvar(c, ddof=1) > 0
                else float("nan"),
                "spearman_tbr_series": rho,
            }
        )
        pretest_rows.append({"metric": spec.name, **trend, "hedges_g": g_hat})
        for test_name, pval in tests.items():
            p_rows.append(
                {
                    "metric": spec.name,
                    "test": test_name.replace("_p", ""),
                    "window": "pretest",
                    "estimand": "balance_not_att",
                    "p_raw": pval,
                    "stat": float("nan"),
                }
            )
        p_rows.append(
            {
                "metric": spec.name,
                "test": "pretrend_interaction",
                "window": "pretest",
                "estimand": "identification",
                "p_raw": trend["pretrend_p"],
                "stat": trend["pretrend_coef"],
            }
        )
        if spec is primary and abs(g_hat) > 0.25 and np.isfinite(g_hat):
            gates.append(
                {
                    "check": "smd",
                    "status": "warn",
                    "detail": f"|Hedges g|={g_hat:.3f} > 0.25 on {spec.name}",
                }
            )
            warnings.append("pretest_imbalance")

    if not any(g["check"] == "smd" for g in gates):
        gates.append({"check": "smd", "status": "pass", "detail": "primary |Hedges g| ≤ 0.25"})

    # TBR fit + placebo + sensitivity on primary outcome column
    outcome_col = primary.value_column if primary.kind != "rate" else experiment.conversions
    if outcome_col not in panel.columns:
        outcome_col = experiment.conversions
    tbr = estimate_tbr(
        panel, conversions=outcome_col, location=location, date=date, spend_col=spend_col
    )
    pretest_rows.append(
        {
            "metric": primary.name,
            "tbr_pre_r2": tbr.pre_r2,
            "tbr_lift": tbr.lift,
            "tbr_se": tbr.std_error,
        }
    )
    if tbr.pre_r2 is not None and tbr.pre_r2 < 0.5:
        gates.append(
            {
                "check": "tbr_fit",
                "status": "warn",
                "detail": f"TBR pretest R²={tbr.pre_r2:.3f} < 0.5",
            }
        )
        warnings.append("weak_tbr_pretest_fit")
    else:
        gates.append(
            {
                "check": "tbr_fit",
                "status": "pass",
                "detail": f"TBR pretest R²={tbr.pre_r2}",
            }
        )

    placebo_ok = True
    pre_dates = sorted(pre[date].unique())
    if len(pre_dates) >= 10:
        mid = pre_dates[len(pre_dates) // 2]
        fake_end = pd.Timestamp(pre_dates[-1])
        from causalgeo.experiment import GeoExperiment

        placebo_exp = GeoExperiment(
            experiment.data.loc[
                pd.to_datetime(experiment.data[experiment.date]) < pd.to_datetime(experiment.treatment_start)
            ],
            experiment.assignment,
            treatment_start=mid,
            treatment_end=fake_end,
            location=location,
            date=date,
            conversions=experiment.conversions,
            spend=experiment.spend,
        )
        try:
            placebo = placebo_exp.analyze(intervention, counterfactual="tbr")
            covers = placebo.ci_lower <= 0 <= placebo.ci_upper
            p_rows.append(
                {
                    "metric": primary.name,
                    "test": "placebo_tbr",
                    "window": "pretest",
                    "estimand": "placebo_att",
                    "p_raw": placebo.p_value,
                    "stat": placebo.lift,
                }
            )
            if not covers:
                placebo_ok = False
                gates.append(
                    {
                        "check": "placebo",
                        "status": "warn",
                        "detail": f"placebo TBR CI ({placebo.ci_lower:.3g}, {placebo.ci_upper:.3g}) excludes 0",
                    }
                )
                warnings.append("placebo_excludes_zero")
        except Exception as exc:
            warnings.append(f"placebo_failed: {exc}")
    if placebo_ok and not any(g["check"] == "placebo" for g in gates):
        gates.append({"check": "placebo", "status": "pass", "detail": "placebo TBR CI covers 0"})

    twfe = estimate_att(
        panel, conversions=outcome_col, location=location, date=date, spend_col=spend_col
    )
    sc = estimate_sc(
        panel, conversions=outcome_col, location=location, date=date, spend_col=spend_col
    )
    sensitivity = pd.DataFrame(
        [
            {
                "method": r.method,
                "lift": r.lift,
                "std_error": r.std_error,
                "ci_lower": r.ci_lower,
                "ci_upper": r.ci_upper,
                "pre_r2": r.pre_r2,
            }
            for r in (tbr, sc, twfe)
        ]
    )
    if np.isfinite(tbr.lift) and np.isfinite(tbr.std_error) and tbr.std_error > 0:
        for other in (sc, twfe):
            gap = abs(other.lift - tbr.lift)
            if gap > 0.25 * abs(tbr.lift) and gap > 0.5 * tbr.std_error:
                warnings.append(f"estimator_disagreement_{other.method}")
                gates.append(
                    {
                        "check": "sensitivity",
                        "status": "warn",
                        "detail": f"{other.method} lift {other.lift:.3g} vs TBR {tbr.lift:.3g}",
                    }
                )
    if not any(g["check"] == "sensitivity" for g in gates):
        gates.append({"check": "sensitivity", "status": "pass", "detail": "TBR/SC/TWFE agree"})

    mde = _Z_MDE * float(tbr.std_error) if tbr.std_error and np.isfinite(tbr.std_error) else float("nan")
    ctrl_pre = float(pre.loc[pre["is_treatment"] == 0, outcome_col].mean()) if outcome_col in pre.columns else float("nan")
    power = pd.DataFrame(
        [
            {
                "metric": primary.name,
                "method": "tbr",
                "mde_80": mde,
                "mde_pct_control_pre": mde / ctrl_pre if ctrl_pre else float("nan"),
                "observed_se": tbr.std_error,
                "n_treatment_geos": n_t,
                "n_control_geos": n_c,
                "n_unassigned": n_u,
                "n_pre_dates": n_pre_dates,
                "n_post_dates": n_post_dates,
                "spend_delta": spend_delta,
            }
        ]
    )

    tests_df = pd.DataFrame(p_rows)
    if not tests_df.empty and adjust == "bh":
        tests_df["p_bh"] = benjamini_hochberg(tests_df["p_raw"].tolist(), q=bh_q)
        tests_df["family"] = "diagnose"
    elif not tests_df.empty:
        tests_df["p_bh"] = tests_df["p_raw"]
        tests_df["family"] = "diagnose"

    pretrend_bh = tests_df.loc[tests_df["test"] == "pretrend_interaction", "p_bh"] if not tests_df.empty else pd.Series(dtype=float)
    p_use = float(pretrend_bh.iloc[0]) if len(pretrend_bh) else float("nan")
    if np.isfinite(p_use) and p_use < 0.05:
        gates.append(
            {
                "check": "pretrend",
                "status": "warn",
                "detail": f"pretrend BH p={p_use:.3g} < 0.05",
            }
        )
        warnings.append("pretrend_detected")
    else:
        gates.append({"check": "pretrend", "status": "pass", "detail": f"pretrend BH p={p_use}"})

    return DiagnosticsReport(
        metrics=list(metrics),
        balance=pd.DataFrame(balance_rows),
        pretest=pd.DataFrame(pretest_rows),
        metric_tests=tests_df,
        contingency=contingency,
        power=power,
        sensitivity=sensitivity,
        gates=pd.DataFrame(gates),
        warnings=warnings,
    )

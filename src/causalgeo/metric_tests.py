"""Type-correct pretest tests. Contingency tables are not an ATT."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

from causalgeo.metrics import MetricSpec


def benjamini_hochberg(p_values: list[float], q: float = 0.10) -> list[float]:
    arr = np.asarray(p_values, dtype=float)
    out = np.full(arr.shape, np.nan)
    ok = np.isfinite(arr)
    if not ok.any():
        return out.tolist()
    adj = stats.false_discovery_control(arr[ok], method="bh")
    out[ok] = np.minimum(adj, 1.0)
    return out.tolist()


def hedges_g(treat: np.ndarray, ctrl: np.ndarray) -> float:
    """Hedges' g on two samples (geo-level means)."""
    t = np.asarray(treat, dtype=float)
    c = np.asarray(ctrl, dtype=float)
    t, c = t[np.isfinite(t)], c[np.isfinite(c)]
    n1, n2 = len(t), len(c)
    if n1 < 2 or n2 < 2:
        return float("nan")
    s1, s2 = float(np.std(t, ddof=1)), float(np.std(c, ddof=1))
    pooled = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
    if pooled <= 0:
        return 0.0 if float(np.mean(t) - np.mean(c)) == 0 else float("nan")
    d = (float(np.mean(t)) - float(np.mean(c))) / pooled
    j = 1.0 - 3.0 / (4.0 * (n1 + n2) - 9.0)
    return float(j * d)


def welch_mannwhitney(treat: np.ndarray, ctrl: np.ndarray) -> dict[str, float]:
    t = np.asarray(treat, dtype=float)
    c = np.asarray(ctrl, dtype=float)
    t, c = t[np.isfinite(t)], c[np.isfinite(c)]
    if len(t) < 2 or len(c) < 2:
        return {"welch_p": float("nan"), "mannwhitney_p": float("nan"), "ks_p": float("nan")}
    welch = stats.ttest_ind(t, c, equal_var=False)
    mw = stats.mannwhitneyu(t, c, alternative="two-sided")
    ks = stats.ks_2samp(t, c)
    return {
        "welch_p": float(welch.pvalue),
        "mannwhitney_p": float(mw.pvalue),
        "ks_p": float(ks.pvalue),
    }


def contingency_table(
    frame: pd.DataFrame,
    arm: str,
    category: str,
    count: str,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Arm × category counts; χ² or Fisher. Not an incrementality estimate."""
    table = (
        frame.groupby([arm, category], as_index=False)[count]
        .sum()
        .pivot(index=arm, columns=category, values=count)
        .fillna(0.0)
    )
    observed = table.to_numpy(dtype=float)
    chi2, p_chi, dof, expected = stats.chi2_contingency(observed)
    use_fisher = bool(expected.min() < 5) and observed.shape == (2, 2)
    p_fisher = float("nan")
    method = "chi2"
    p_raw = float(p_chi)
    if use_fisher:
        _, p_fisher = stats.fisher_exact(observed)
        method = "fisher"
        p_raw = float(p_fisher)
    detail = table.copy()
    exp_df = pd.DataFrame(expected, index=table.index, columns=table.columns)
    resid = (table - exp_df) / np.sqrt(np.clip(exp_df, 1e-12, None))
    return table, {
        "chi2": float(chi2),
        "p_chi2": float(p_chi),
        "p_fisher": float(p_fisher) if use_fisher else float("nan"),
        "p_raw": p_raw,
        "dof": float(dof),
        "min_expected": float(expected.min()),
        "method": method,
        "residual_max": float(np.nanmax(np.abs(resid.to_numpy()))),
    }


def rate_two_by_two(
    successes_t: float,
    trials_t: float,
    successes_c: float,
    trials_c: float,
) -> tuple[pd.DataFrame, dict[str, float]]:
    table = pd.DataFrame(
        {
            "success": [successes_t, successes_c],
            "failure": [trials_t - successes_t, trials_c - successes_c],
        },
        index=["treatment", "control"],
    )
    _, stats_d = contingency_table(
        pd.DataFrame(
            {
                "arm": ["treatment", "treatment", "control", "control"],
                "category": ["success", "failure", "success", "failure"],
                "n": [
                    successes_t,
                    trials_t - successes_t,
                    successes_c,
                    trials_c - successes_c,
                ],
            }
        ),
        arm="arm",
        category="category",
        count="n",
    )
    return table, stats_d

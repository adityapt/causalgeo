"""Two-way demeaning on balanced and unbalanced panels."""

import numpy as np
import pandas as pd
import statsmodels.api as sm

from causalgeo.panel_ols import run_panel_ols_did


def _panel(drop_early: bool) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for geo in range(6):
        for day in range(12):
            if drop_early and geo == 5 and day < 4:
                continue
            treated = float(geo < 3 and day >= 6)
            sales = 5.0 * treated + geo + 0.3 * day + float(rng.normal(0, 0.2))
            rows.append({"geo": geo, "date": day, "sales": sales, "did": treated})
    return pd.DataFrame(rows)


def _dummy_coef(panel: pd.DataFrame) -> float:
    geo = pd.get_dummies(panel["geo"], prefix="g", drop_first=True, dtype=float)
    day = pd.get_dummies(panel["date"], prefix="d", drop_first=True, dtype=float)
    design = pd.concat([panel[["did"]].astype(float), geo, day], axis=1)
    fitted = sm.OLS(panel["sales"].astype(float), sm.add_constant(design)).fit()
    return float(fitted.params["did"])


def test_balanced_panel_matches_market_and_date_dummies():
    panel = _panel(drop_early=False)
    estimate = run_panel_ols_did(panel, "sales", "did", dma_col="geo", date_col="date")
    assert abs(float(estimate.params["did"]) - _dummy_coef(panel)) < 1e-8


def test_unbalanced_panel_matches_market_and_date_dummies():
    panel = _panel(drop_early=True)
    estimate = run_panel_ols_did(panel, "sales", "did", dma_col="geo", date_col="date")
    assert abs(float(estimate.params["did"]) - _dummy_coef(panel)) < 1e-6

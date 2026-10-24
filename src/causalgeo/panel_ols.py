"""
Two-way fixed-effects DiD with DMA-clustered standard errors.

Two-way demeaned panel with standard errors clustered by market. Used by the
bundle estimator. It is not the default `analyze()` path; that remains TBR.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Union

import numpy as np
import pandas as pd
import statsmodels.api as sm


@dataclass
class PanelEstimate:
    """TWFE / clustered-OLS estimates for one or more treatment columns."""

    params: pd.Series
    bse: pd.Series
    pvalues: pd.Series
    conf_int: pd.DataFrame
    nobs: int
    n_entities: int
    cov_type: str
    formula: str
    entity_col: str
    time_col: str

    def summary_frame(self) -> pd.DataFrame:
        """Coefficient table: estimate, clustered SE, p-value, 95% CI."""
        ci = self.conf_int
        return pd.DataFrame(
            {
                "coef": self.params,
                "std_err": self.bse,
                "p_value": self.pvalues,
                "ci_lower": ci.iloc[:, 0],
                "ci_upper": ci.iloc[:, 1],
            }
        )


def _as_list(columns: Union[str, Sequence[str]]) -> List[str]:
    if isinstance(columns, str):
        return [columns]
    return list(columns)


def _twoway_demean(
    frame: pd.DataFrame,
    entity_col: str,
    time_col: str,
    columns: Sequence[str],
) -> pd.DataFrame:
    cols = list(columns)
    out = frame[cols].astype(float).copy()
    # One subtraction is exact only when every market is observed every period.
    # On an unbalanced panel, repeat until the market and period means are gone.
    for _ in range(200):
        previous = out.to_numpy()
        out = out - out.groupby(frame[entity_col]).transform("mean")
        out = out - out.groupby(frame[time_col]).transform("mean")
        if np.max(np.abs(out.to_numpy() - previous)) < 1e-12:
            break
    return out


def _fit_twoway_clustered(
    panel: pd.DataFrame,
    outcome_col: str,
    treatment_cols: Sequence[str],
    entity_col: str,
    time_col: str,
    formula: str,
) -> PanelEstimate:
    missing = [c for c in [outcome_col, entity_col, time_col, *treatment_cols] if c not in panel.columns]
    if missing:
        raise ValueError(f"Panel is missing required columns: {missing}")
    if panel.empty:
        raise ValueError("Panel is empty")

    work = panel[[outcome_col, entity_col, time_col, *treatment_cols]].dropna()
    if work[entity_col].nunique() < 2:
        raise ValueError("Need at least two entities (DMAs/cells) for two-way FE")
    if work[time_col].nunique() < 2:
        raise ValueError("Need at least two time periods for two-way FE")

    demeaned = _twoway_demean(work, entity_col, time_col, [outcome_col, *treatment_cols])
    y = demeaned[outcome_col]
    x = demeaned[list(treatment_cols)]
    # Drop collinear (fully absorbed) regressors.
    x = x.loc[:, x.std() > 1e-12]
    if x.empty:
        raise ValueError("Treatment columns are collinear with entity/time fixed effects")

    model = sm.OLS(y, x, hasconst=False)
    fitted = model.fit(
        cov_type="cluster",
        cov_kwds={"groups": work.loc[x.index, entity_col]},
    )
    ci = fitted.conf_int()
    ci.columns = ["ci_lower", "ci_upper"]
    return PanelEstimate(
        params=fitted.params,
        bse=fitted.bse,
        pvalues=fitted.pvalues,
        conf_int=ci,
        nobs=int(fitted.nobs),
        n_entities=int(work[entity_col].nunique()),
        cov_type="cluster",
        formula=formula,
        entity_col=entity_col,
        time_col=time_col,
    )


def run_panel_ols_did(
    panel: pd.DataFrame,
    outcome_col: str,
    treatment_col: Union[str, List[str]],
    dma_col: str = "dmacode",
    cluster_entity: bool = True,
    date_col: str = "date",
) -> PanelEstimate:
    """DiD via two-way FE: Y ~ did_* + DMA FE + time FE, SEs clustered by DMA.

    ``cluster_entity=False`` still absorbs DMA and time FE but uses heteroskedasticity-
    robust (HC1) SEs. That is not recommended for geo panels.
    """
    del cluster_entity  # clustering by DMA is the supported inference path
    treatment_cols = _as_list(treatment_col)
    formula = f"{outcome_col} ~ {' + '.join(treatment_cols)} + C({dma_col}) + C({date_col})"
    return _fit_twoway_clustered(
        panel=panel,
        outcome_col=outcome_col,
        treatment_cols=treatment_cols,
        entity_col=dma_col,
        time_col=date_col,
        formula=formula,
    )


def run_panel_ols_cell_fe(
    panel: pd.DataFrame,
    outcome_col: str,
    treatment_col: Union[str, List[str]],
    cell_effects: bool = True,
    dma_col: str = "dmacode",
    cell_col: str = "cell",
    date_col: str = "date",
) -> PanelEstimate:
    """Panel OLS with cell FE and time FE; SEs still clustered by DMA.

    Use when the Latin-square cell is the relevant grouping and treatment is
    not collinear with cell. DMA clustering keeps serial correlation inside
    each market.
    """
    if not cell_effects:
        return run_panel_ols_did(
            panel,
            outcome_col=outcome_col,
            treatment_col=treatment_col,
            dma_col=dma_col,
            date_col=date_col,
        )
    if cell_col not in panel.columns:
        raise ValueError(f"Panel is missing cell column {cell_col!r}")
    treatment_cols = _as_list(treatment_col)
    formula = f"{outcome_col} ~ {' + '.join(treatment_cols)} + C({cell_col}) + C({date_col})"
    work = panel[[outcome_col, dma_col, cell_col, date_col, *treatment_cols]].dropna()
    if work.empty:
        raise ValueError("Panel is empty")
    demeaned = _twoway_demean(work, cell_col, date_col, [outcome_col, *treatment_cols])
    y = demeaned[outcome_col]
    x = demeaned[list(treatment_cols)]
    x = x.loc[:, x.std() > 1e-12]
    if x.empty:
        raise ValueError("Treatment columns are collinear with cell/time fixed effects")
    model = sm.OLS(y, x, hasconst=False)
    fitted = model.fit(
        cov_type="cluster",
        cov_kwds={"groups": work.loc[x.index, dma_col]},
    )
    ci = fitted.conf_int()
    ci.columns = ["ci_lower", "ci_upper"]
    return PanelEstimate(
        params=fitted.params,
        bse=fitted.bse,
        pvalues=fitted.pvalues,
        conf_int=ci,
        nobs=int(fitted.nobs),
        n_entities=int(work[dma_col].nunique()),
        cov_type="cluster",
        formula=formula,
        entity_col=cell_col,
        time_col=date_col,
    )

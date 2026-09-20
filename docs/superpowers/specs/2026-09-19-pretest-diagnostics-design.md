# Pretest diagnostics and metric tests

Date: 2026-09-19  
Status: draft (awaiting implementation plan)  
Package: `causalgeo`

## Problem

`analyze()` currently returns a lift point estimate, clustered or TBR/SC uncertainty, and optional iCPD. It does not tell the user whether the assignment is balanced, whether pretest paths track, whether delivery happened, or whether a KPI is even a valid incrementality metric. A go-to measurement package has to produce those checks **before** (and beside) the ATT, with tests that match the metric type.

This is not a kitchen-sink p-value dump. Invalid tests (chi-square on continuous sales, unclustered t-tests as the primary geo inference, uncorrected fishing across 40 KPIs) would make the package look complete and be statistically worse.

## Goals

1. Attach a structured `DiagnosticsReport` to an experiment: data quality, pretest balance, trend/fit, delivery, power/MDE, estimator sensitivity, and type-correct metric tests.
2. Support **continuous**, **count**, **rate**, and **categorical** metrics.
3. Accept categorical data as **long category counts** (`geo, date, metric, category, count`) **and** **wide success/trial** columns.
4. Record assumptions, raw p-values, and **Benjamini–Hochberg** adjusted p-values. No single “experiment passed” score.
5. Contingency tables assess **categorical association/balance**. They do **not** estimate geo incrementality.

## Non-goals (v1)

- HTML/PDF report generation (Python objects + DataFrames only).
- Design search, market selection, or GeoLift-style power *before* assignment exists (MDE from the locked assignment is in scope).
- Spillover / adjacency models.
- Sequential testing or always-valid p-values.
- Automatic primary-metric selection.

## API

```python
from causalgeo import GeoExperiment, MetricSpec, diagnose

specs = [
    MetricSpec("conversions", kind="count"),
    MetricSpec("revenue", kind="continuous"),
    MetricSpec("close_rate", kind="rate", successes="closes", trials="leads"),
    MetricSpec(
        "funnel_stage",
        kind="categorical",
        categories="stage",
        counts="n",
    ),
]

report = exp.diagnose(metrics=specs, intervention=Intervention.GO_DARK)
report.balance          # DataFrame
report.pretest          # DataFrame
report.metric_tests     # DataFrame
report.contingency      # dict[str, DataFrame]
report.power            # DataFrame
report.sensitivity      # DataFrame (TBR / SC / TWFE)
report.warnings         # list[str]
report.gates            # DataFrame: check, status, detail
```

`GeoExperiment.diagnose()` is the user-facing entry. `diagnose()` as a module function is the same implementation.

Lift analysis stays `analyze()`. Diagnostics **must not** replace the ATT. If the user calls `analyze()` with no prior `diagnose()`, that remains allowed; `analyze()` may attach a short warning that diagnostics were not run.

## Metric registry

`MetricSpec` fields:

| Field | Meaning |
|---|---|
| `name` | KPI id used in tables |
| `kind` | `continuous` \| `count` \| `rate` \| `categorical` |
| `column` | Panel column for continuous/count (default: `name`) |
| `successes`, `trials` | Wide rate or binomial columns |
| `categories`, `counts` | Long categorical: category label column and count column |
| `role` | `outcome` (default), `delivery`, or `covariate` |
| `primary` | If true, used as the ATT series unless `analyze(conversions=)` overrides |

Invalid combinations raise at construction (rate without successes/trials **or** a single proportion column with explicit `trials`; categorical without counts).

A metric with `role="delivery"` (spend, impressions) is tested for **manipulation** (did treatment geos actually go dark / heavy-up?), not for lift as the primary estimand.

## Diagnostic blocks

### 1. Panel and assignment

- Geos in assignment missing from KPI panel, and the reverse (unassigned KPI geos are listed, not errors).
- Dates: pretest length, test length, missing geo-days, duplicate keys.
- Cell overlap already handled by `AssignmentMap`; diagnostics surface `n_treatment`, `n_control`, `n_unassigned`.
- **Gate:** fewer than 5 pretest dates (TBR cannot run); zero control geos (already a hard error).

### 2. Delivery

Using spend and/or impressions when present:

- Pre vs post mean spend/impressions by arm.
- Spend DiD (same formula as iCPD denominator).
- **Gate (go-dark):** treatment post spend not materially below treatment pretest *and* not below control post → `delivery_not_detected` warning, not a silent ATT.
- **Gate (holdback / heavy-up):** opposite sign expected.

### 3. Pretest balance (per metric, pretest window only)

Continuous / count / rate (geo-level pretest means):

- Mean, SD, median by arm.
- Standardized mean difference as **Hedges’ *g*** on geo-level pretest means (Cohen’s *d* with small-sample correction). Pooled SD of geo means; not geo-day rows.
- Variance ratio.
- Welch t-test **and** Mann–Whitney as **balance** tests (labeled as such; not the causal estimand).
- Spearman correlation of treated-mean series vs control-mean series (same objects TBR uses).

Rates use the trial-weighted mean when `trials` is present.

**Gate (warning, not hard fail):** |Hedges’ *g*| > 0.25 on the primary outcome.

### 4. Pretest dynamics / identification threats

- Linear pretrend: geo-mean KPI ~ arm × calendar time in the pretest only. Report the interaction coefficient, clustered SE by geo, p-value.
- TBR pretest `R²` and residual SD (reuse `estimate_tbr` fit, do not refit a different model).
- SC pretest RMSPE / `pre_r2` when `counterfactual` includes `sc` or when sensitivity is requested.
- Placebo: fake `treatment_start` at the midpoint of the pretest; TBR ATT should cover 0. One placebo, pre-registered in code, not a search over dates.

**Gate (warning):** pretrend interaction p < 0.05 (BH-adjusted if multiple metrics); TBR `pre_r2` < 0.5 on the primary outcome; placebo CI excludes 0.

### 5. Metric tests (type-correct)

These are **descriptive / balance / association** tests on the pretest, plus optional post-period **descriptive** contrasts clearly labeled `not_att`.

| Kind | Tests | Tables |
|---|---|---|
| continuous | Welch t, Mann–Whitney, KS (sensitivity) | none |
| count | Poisson rate test or Welch on geo-level means if overdispersed (report both; prefer negative binomial / quasi-Poisson note if variance ≫ mean) | none |
| rate | Two-proportion or binomial GLM with trials as exposure | 2×2 of successes/failures by arm if aggregated |
| categorical | χ² test of independence; Fisher exact if any expected cell < 5 | full contingency (arm × category), expected counts, residuals |

v1 aggregation for categorical: **sum counts over the pretest** (and separately over the test window as a descriptive table). Do not treat daily category shares as independent Bernoulli trials without saying so; primary categorical test is on **geo-aggregated** pretest totals to avoid inflating N.

### 6. Power / MDE

From the **locked assignment** and pretest residual variance of the chosen counterfactual (default TBR residual SD of the daily treated mean):

- Two-sided 5% MDE at 80% power for the primary metric, on the same scale as `lift` (per geo-day mean).
- Implied percent of control pretest mean.
- Observed `analyze()` SE vs that MDE after the test exists (if test-period data are in the panel).

Formula for TBR daily series (v1, documented as an approximation that treats post days as independent given the control path):  
`MDE ≈ (z_{1-α/2} + z_{1-β}) * se_att_hat`  
where `se_att_hat` uses pretest TBR residual variance and the same variance components as `estimate_tbr` (parameter uncertainty + `sigma2 / T_post`).

This is **not** a clustered TWFE MDE unless `counterfactual="twfe"`.

### 7. Estimator sensitivity

Always compute TBR, SC, and TWFE ATT on the primary continuous/count outcome when the panel supports it. Table: method, lift, SE, CI, `pre_r2` if applicable.

**Warning** if pairwise ATT differs by more than 25% of the TBR |lift| **and** more than 0.5 TBR SE (both conditions, to avoid noise at ~0).

## Multiplicity

- Family: all **hypothesis tests** in one `diagnose()` call (balance t-tests, pretrend, categorical χ², placebo, etc.).
- Adjustment: Benjamini–Hochberg at `q=0.10` default (configurable).
- Columns: `p_raw`, `p_bh`, `family`.
- Gates that say “significant pretrend” use `p_bh` unless the user sets `adjust="none"`.

Delivery gates and SMD thresholds are not p-value tests and are not in the BH family.

## Output objects

```python
@dataclass
class DiagnosticsReport:
    metrics: list[MetricSpec]
    balance: pd.DataFrame
    pretest: pd.DataFrame
    metric_tests: pd.DataFrame
    contingency: dict[str, pd.DataFrame]
    power: pd.DataFrame
    sensitivity: pd.DataFrame
    gates: pd.DataFrame  # check, status in {pass, warn, fail}, detail
    warnings: list[str]
```

`status="fail"` is reserved for **cannot analyze** (no pretest dates, empty cell, collinear TBR). Identification threats are `warn`.

## Warnings vs refusals

Reuse existing exceptions (`MissingControlError`, `MixedRoleError`, …). Diagnostics never swallow those.

New: `DiagnosticsWarning` is not an exception; strings on the report.

## Testing

- Synthetic balanced panel: SMD near 0, pretrend not flagged, placebo covers 0, BH does not create false `fail` gates.
- Imbalanced pretest means: SMD gate warns.
- Divergent pretest trends: pretrend gate warns; TBR still returns a number.
- Categorical: known 2×3 table recovers χ² p-value vs scipy; Fisher path when a cell is 0.
- Rate: wide successes/trials vs long counts for a binary category agree on the 2×2.
- Placebo with true ATT = 0 in a fake post carved from pretest: CI includes 0 at least as often as the Monte Carlo coverage suite allows (document as a unit test with one DGP, not 200 draws in CI).
- Sensitivity: planted ATT recovered by all three methods on the existing `_kpi_panel` DGP.

## Files

| File | Role |
|---|---|
| `src/causalgeo/metrics.py` | `MetricSpec` validation |
| `src/causalgeo/diagnostics.py` | `diagnose`, report dataclass, gates |
| `src/causalgeo/tests_metrics.py` | type-correct tests + BH (name may be `metric_tests.py` to avoid pytest collection of the module) |
| `tests/test_diagnostics.py` | unit tests above |
| `README.md` | diagnose() example; state that contingency ≠ ATT |

Prefer `metric_tests.py` so pytest does not collect the library module.

## Implementation order

1. `MetricSpec` + empty `DiagnosticsReport`.
2. Panel/assignment + delivery gates.
3. Balance + SMD.
4. Pretrend, TBR R², one placebo.
5. Type-correct metric tests + contingency + BH.
6. Power/MDE + TBR/SC/TWFE sensitivity.
7. README + public exports.

## Open decisions (locked for v1)

- Default counterfactual for placebo and MDE: **TBR** (same as `analyze()`).
- Categorical primary test: **geo-aggregated pretest totals**, not geo-day iid.
- HTML reports: **out of v1**.
- Primary metric: first `MetricSpec` with `primary=True`, else the experiment `conversions` column.

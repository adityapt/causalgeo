# Pretest Diagnostics Implementation Plan

> **For agentic workers:** Execute inline in this session. Do not commit unless the user asks. Steps use checkbox syntax for tracking.

**Goal:** Add `exp.diagnose()` so causalgeo reports pretest balance, type-correct metric tests, contingency tables, delivery, placebo, MDE, and TBR/SC/TWFE sensitivity beside the ATT.

**Architecture:** `MetricSpec` validates metric types. `diagnose()` builds the same panel as `analyze()`, fills a `DiagnosticsReport` of DataFrames and gates. Hypothesis tests go through Benjamini–Hochberg. Contingency tables are association/balance, not lift.

**Tech Stack:** pandas, numpy, scipy, statsmodels (existing).

**Spec:** `docs/superpowers/specs/2026-09-19-pretest-diagnostics-design.md`

## Global Constraints

- SMD is Hedges’ *g* on geo-level pretest means, gate |g| > 0.25 on primary outcome.
- Default counterfactual for placebo and MDE is TBR.
- Categorical tests use geo-aggregated window totals, not geo-day iid.
- `fail` gates only when analysis cannot run; identification issues are `warn`.
- Do not commit unless the user asks.

## Files

- Create: `src/causalgeo/metrics.py`
- Create: `src/causalgeo/metric_tests.py`
- Create: `src/causalgeo/diagnostics.py`
- Create: `tests/test_diagnostics.py`
- Modify: `src/causalgeo/experiment.py`, `src/causalgeo/__init__.py`, `README.md`

### Task 1: MetricSpec + diagnose skeleton

- [x] Tests for invalid rate/categorical specs and a report object from `diagnose()`
- [x] Implement `MetricSpec`, `DiagnosticsReport`, `GeoExperiment.diagnose`

### Task 2: Balance, delivery, pretrend, placebo, power, sensitivity

- [x] Hedges’ *g*, Welch + Mann–Whitney labeled as balance, spend DiD delivery gate, TBR R², midpoint placebo, MDE, TBR/SC/TWFE table

### Task 3: Type-correct tests + BH + contingency

- [x] χ² vs scipy on a known table, Fisher when expected cell < 5, wide rate 2×2, BH columns

### Task 4: README + exports

- [x] Public API and note that contingency ≠ ATT

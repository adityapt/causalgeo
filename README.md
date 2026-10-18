# causalgeo

Python library for **geo experiments** — incrementality tests on geographic units (geos / locations), in the same setting as [GeoLift](https://facebookincubator.github.io/GeoLift/) and [Meridian GeoX](https://developers.google.com/meridian/geox).

You supply **treatment geos**, **control geos**, and a **pretest / test** conversion series. `analyze()` estimates **lift** (ATT), **percent lift**, and **iCPD**. Measurement default is **time-based regression** (the pretest path), with synthetic control and clustered TWFE as options.

## Intervention types (GeoX)

| Type | Treatment group in the test period | Control group |
|---|---|---|
| **Go-dark** | Existing media off | Business-as-usual (BAU) |
| **Holdback** | New activity on | Zero / held back |
| **Heavy-up** | Incremental activity (e.g. +X% delivered impressions) | BAU |

Incrementality always needs a **control group**. Comparing two spend arms with no BAU geos is not a lift study; the library raises instead of reporting iROAS.

## How this differs from GeoLift and GeoX

GeoLift and GeoX both **design the experiment and then analyze the design they produced**. They assume one channel, one clean treatment/control split, and spend that behaves as planned. `causalgeo` starts one step later: **the assignment already exists, it is messy, and delivery missed plan.**

| | GeoLift | Meridian GeoX | causalgeo |
|---|---|---|---|
| Starting point | Pick test markets | Search designs (strata, MDE, budget) | **Assignment you were handed** |
| Counterfactual | Synthetic control (ASCM / GSC) | Time-based regression | **TBR (default)**, synthetic control, or clustered TWFE |
| Treatment variable | Treated / not | `budget_pct` on **planned spend** | Treated / not **or delivered-impression dose** |
| Overlapping cell lists | Not modeled | Not modeled | Collapsed to unique geos; role conflicts caught; unassigned pool exposed |
| Concurrent tests | Assumes isolation | Requires isolation | `assert_overlay_allowed` blocks collisions |
| Language | R | Python | Python |

Three things it does that the others do not:

**1. Overlapping assignment is a first-class input.** Real programs run several cells and slots at once, and the same geo appears in many lists. `AssignmentMap.from_frame` collapses those to unique treatment and control geos and tells you which geos are in **neither** — the only ones a second test may touch. When a geo is treated in one cell and a control donor in another, that is a valid matrix but not a single contrast, so it raises `MixedRoleError` and you analyze one cell at a time with `cell_col=` / `cell=`.

**2. A second test cannot silently break the first.** Raising spend in another study's control geos turns "media off vs BAU" into "media off vs BAU plus extra spend." `assert_overlay_allowed` raises instead of letting that reach a readout.

**3. Dose is delivered exposure, not planned spend.** Geo bid adjustments do not deliver the budget you asked for. When impressions are supplied, the regressor is impressions relative to each geo's own pretest baseline, so a planned +50% that delivered +18% is estimated as +18%.

It also refuses to produce numbers that look like incrementality but are not: no control group raises `MissingControlError`, and iCPD without a spend series raises `MissingSpendError` rather than dividing by the wrong denominator. When spend is present, the denominator is the **difference-in-differences of spend**, so a treatment group that was already spending more than control at baseline does not inflate the per-dollar number.

**Use GeoLift or GeoX instead when** you are **designing** a single-channel test from scratch and want market selection, power curves, and budget search. `causalgeo` does not do design search. For **analysis of an assignment you already have**, TBR / SC / TWFE plus overlap rules are in this package.

## Design vs analysis

* **Assignment** — which locations are treatment vs control (GeoLift: test vs control markets; GeoX: geo assignment). Overlapping cell-level lists are collapsed to unique geos, and a geo holding both roles across cells is analyzed one cell at a time. Geos in the **universe** that are in neither group are **unassigned** (available for a separate heavy-up, not for the first test’s counterfactual).
* **Analyze** — default **TBR**: pretest fit of treated-geo mean on control-geo mean, then the test-period gap. `counterfactual="sc"` fits nonnegative donor weights on the pretest path. `counterfactual="twfe"` is geo FE + date FE with SEs clustered by geo. Optional regressor for heavy-up: **delivered impressions** relative to pretest (TWFE-only).

Market selection / power search (GeoLift `GeoLiftMarketSelection`, GeoX design engine) is out of scope for v0.

## Concurrent bundles

When several cells run in the same weeks and each cell turns off a bundle, `disentangle` estimates the bundle from the holdout flag, adjusts that estimate when a channel that was supposed to stay on shifted in the dark markets, and splits the bundle totals into channel slopes. A channel whose impressions are a copy of another channel stays combined. Daily impact is exposure times the channel slope.

```python
from causalgeo import design_concurrent_test, disentangle

design = design_concurrent_test(
    markets,  # geo, stratum
    channels=["Facebook", "TikTok", "Search"],
    bundles=[["Facebook", "TikTok"], ["Facebook", "Search"], ["TikTok", "Search"]],
    control_share=0.5,
)
result = disentangle(
    design,
    outcomes,      # geo, date, sales
    impressions,   # geo, date, channel, impressions
    outcome="sales",
    treatment_start="2026-08-03",
)
```

## Vocabulary

| Term | Meaning |
|---|---|
| Geo / location | Experimental unit (DMA, city, region, …) |
| Pretest period | Before `treatment_start` |
| Test period | On and after `treatment_start` |
| Lift / ATT | Effect of the intervention on conversions, per geo-day |
| TBR | Pretest regression of treated mean on control mean; test gap is lift |
| SC | Weighted control geos matched to the treated pretest path |
| Percent lift | Lift ÷ mean control conversions in the test period |
| iCPD / iROAS | Lift ÷ **incremental** spend, where incremental spend is the DiD of spend |
| Hedges’ *g* | Standardized mean difference of **geo-level** pretest means (Cohen’s *d* with small-sample correction) |
| BAU | Business-as-usual spend in control |

## Example

```python
from causalgeo import AssignmentMap, GeoExperiment, Intervention

assignment = AssignmentMap.from_frame(
    assign_df,  # columns: location, role  (treatment|control|test|holdout|...)
    universe=kpi["location"].unique(),
)
exp = GeoExperiment(
    kpi,  # location, date, conversions [, spend] [, impressions]
    assignment,
    treatment_start="2026-10-01",
    treatment_end="2026-10-31",
    spend="spend",
)
report = exp.analyze(Intervention.GO_DARK)  # TBR by default
print(report.lift, report.percent_lift, report.icpd, report.pre_r2)
twfe = exp.analyze(Intervention.GO_DARK, counterfactual="twfe")
sc = exp.analyze(Intervention.GO_DARK, counterfactual="sc")

from causalgeo import MetricSpec

diag = exp.diagnose(
    [
        MetricSpec("conversions", kind="continuous", primary=True),
        MetricSpec("close_rate", kind="rate", successes="closes", trials="leads"),
    ]
)
print(diag.balance, diag.gates, diag.power)
# Contingency / χ² tables are association checks, not lift.
```

Heavy-up on **unassigned** geos only:

```python
overlay = AssignmentMap.from_sets(heavy_up_geos, overlay_control_geos)
assignment.assert_overlay_allowed(overlay)  # no overlap with existing test/control
```

## Does the estimator return garbage?

A single synthetic recovery can look right by luck. `python -m causalgeo.simulate` runs a Monte Carlo on the DGP TWFE is identified for (geo FE + common time shocks + constant ATT, AR(1) errors within geo):

| Check | What “not garbage” looks like |
|---|---|
| A/A (`ATT = 0`) | Bias near 0; clustered 95% CIs cover ~95%; reject rate ~5% |
| Planted go-dark ATT | Bias near 0; clustered CIs cover the planted value |
| iid SEs (same point estimate) | Coverage well below 95% when days are serially correlated |

CI tests in `tests/test_simulate.py` fail if those gates break. This does **not** validate misspecified designs (no parallel trends, contamination, concurrent cells).

## Install

```bash
pip install -e ".[dev]"
```

Private repository while the API is unstable.

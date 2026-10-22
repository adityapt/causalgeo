# causalgeo

[![tests](https://github.com/adityapt/causalgeo/actions/workflows/check.yml/badge.svg)](https://github.com/adityapt/causalgeo/actions/workflows/check.yml)
[![license](https://img.shields.io/badge/license-MIT-blue)](LICENSE.txt)
[![python](https://img.shields.io/badge/python-%E2%89%A53.10-blue)](pyproject.toml)

Python library for **geo experiments**. Two jobs:

1. **One cell you were handed.** `analyze()` estimates lift from treatment geos, control geos, and a pretest/test series. The default is time-based regression. Synthetic difference-in-differences, synthetic control, and clustered two-way fixed effects are options.
2. **Several cells in the same weeks, each turning off a bundle of channels.** Start here when the question is a channel split. `design_concurrent_test` assigns markets. `disentangle` estimates each bundle from the holdout flag and splits the bundle totals into channel slopes with a soft calibration. The experimental cell estimate is left as measured. A still-on channel that gained or lost impressions is flagged, and that shift is removed from the total before the channel split. Pass `adjust_exposure=True` to replace the reported cell estimate with that same refit. The bundle table reports the gap that remains. A channel whose impressions are a copy of another channel is not given a slope; the shared path stays in the fit as one control. A slope that moves when the calibration weight is weakened, or when one market is left out, is marked `unstable` and should be read as model-assisted. Daily impact is exposure times the channel slope. Missing channel-day rows stay missing; a recorded zero stays zero.

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

`result.channels` is one row per channel, with a `reading` column. `result.bundles` is the measured bundle effect next to the effect implied by the channel slopes. `result.exposure` flags a still-on shift and says whether that flag was applied. `result.daily` is the daily channel impact. `bootstrap_channel_intervals` resamples whole markets and refits. `compare_calibration` puts the observational fit next to the calibrated fit.

`examples/concurrent_holdout.py` builds a small panel, prints the channel table, and shows a split the package refuses because two channels are the same series.

The single-cell path is unchanged. You supply treatment geos, control geos, and a pretest/test conversion series. `analyze()` returns lift, percent lift, and iCPD. The setting is the same as [GeoLift](https://facebookincubator.github.io/GeoLift/) and [Meridian GeoX](https://developers.google.com/meridian/geox).

## Intervention types (GeoX)

| Type | Treatment group in the test period | Control group |
|---|---|---|
| **Go-dark** | Existing media off | Business-as-usual (BAU) |
| **Holdback** | New activity on | Zero / held back |
| **Heavy-up** | Incremental activity (e.g. +X% delivered impressions) | BAU |

Incrementality always needs a **control group**. Comparing two spend arms with no BAU geos is not a lift study; the library raises instead of reporting iROAS.

## How this differs from GeoLift and GeoX

GeoLift and GeoX design a market test and then analyze that design. Their search tools are built around one treatment contrast. `causalgeo` also starts from an assignment that already exists, including several cells in the same weeks and delivery that missed the plan. It does not replace their market-selection or power search.

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
* **Analyze** — default **TBR**: pretest fit of treated-geo mean on control-geo mean, then the test-period gap, with **Newey–West (HAC) standard errors** so serial correlation in the residual does not shrink p-values. `counterfactual="sdid"` combines nonnegative donor and pre-period weights and uses control-geo placebo inference. `counterfactual="sc"` fits nonnegative donor weights on the pretest path. `counterfactual="twfe"` is geo FE + date FE with SEs clustered by geo. Optional regressor for heavy-up: **delivered impressions** relative to pretest (TWFE-only).

Market selection / power search (GeoLift `GeoLiftMarketSelection`, GeoX design engine) is out of scope. The concurrent design above assigns markets within strata. It does not search for the lowest minimum detectable effect.

### Synthetic DiD and placebo validation

```python
report = exp.analyze(
    counterfactual="sdid",
    sdid_placebo_replications=200,
    random_state=0,
)

# Pretend an untreated donor was treated on the same date.
placebo = exp.placebo(
    location="control_geo_1",
    counterfactual="sdid",
    sdid_placebo_replications=200,
    random_state=0,
)
```

sDiD v1 supports a single common treatment start. Placebo uncertainty follows
Arkhangelsky et al.'s control-only reassignment method and works with one
treated geo. `examples/prop99_measure.py` reproduces the published Proposition
99 result (about −15.6 packs per capita) and checks a fake-treated Nebraska.

## Concurrent-intervention identification

Before fitting outcomes, `identify_grid()` checks whether a cell-by-channel
dose matrix can separate the requested effects:

```python
from causalgeo import identify_grid

report = identify_grid(dose)  # columns: cell, channel, dose
report.rank
report.alias_groups
report.channel_status
```

Full column rank means each channel is separately identified by the design.
When channels always move together, the report names the bundle that can be
estimated instead of inventing channel-specific effects.

`examples/prop99_grid_identification.py` downloads the published Proposition
99 cigarette-sales panel used by the Synthetic Interventions benchmark. It
shows that tax increases and anti-tobacco programs are separable when policy
type is recorded (rank 2 of 2), but become one aliased bundle when both are
collapsed to a generic policy indicator (rank 1 of 2).

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

The tagged release:

```bash
pip install "causalgeo @ git+https://github.com/adityapt/causalgeo.git@v0.1.0"
```

To run the tests and rebuild the manuscript:

```bash
pip install -e ".[dev,paper]"
make check
```

See [CONTRIBUTING.md](CONTRIBUTING.md) and the [code of conduct](CODE_OF_CONDUCT.md).

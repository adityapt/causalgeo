# causalgeo

Python library for **geo experiments** — incrementality tests on geographic units (geos / locations), in the same setting as [GeoLift](https://facebookincubator.github.io/GeoLift/) and [Meridian GeoX](https://developers.google.com/meridian/geox).

You supply **treatment geos**, **control geos**, and a **pretest / test** conversion series. `analyze()` estimates **lift** (ATT), **percent lift**, and **iCPD** (incremental conversions per dollar; **iROAS** if the KPI is revenue).

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
| Counterfactual | Synthetic control (ASCM / GSC) | Time-based regression | Two-way fixed effects, clustered by geo |
| Treatment variable | Treated / not | `budget_pct` on **planned spend** | Treated / not **or delivered-impression dose** |
| Overlapping cell lists | Not modeled | Not modeled | Collapsed to unique geos; unassigned pool exposed |
| Concurrent tests | Assumes isolation | Requires isolation | `assert_overlay_allowed` blocks collisions |
| Language | R | Python | Python |

Three things it does that the others do not:

**1. Overlapping assignment is a first-class input.** Real programs run several cells and slots at once, and the same geo appears in many lists. `AssignmentMap.from_frame` collapses those to unique treatment and control geos and tells you which geos are in **neither** — the only ones a second test may touch.

**2. A second test cannot silently break the first.** Raising spend in another study's control geos turns "media off vs BAU" into "media off vs BAU plus extra spend." `assert_overlay_allowed` raises instead of letting that reach a readout.

**3. Dose is delivered exposure, not planned spend.** Geo bid adjustments do not deliver the budget you asked for. When impressions are supplied, the regressor is impressions relative to each geo's own pretest baseline, so a planned +50% that delivered +18% is estimated as +18%.

It also refuses to produce numbers that look like incrementality but are not: no control group raises `MissingControlError`, and iCPD without a spend series raises `MissingSpendError` rather than dividing by the wrong denominator.

**Use GeoLift or GeoX instead when** you are designing a single-channel test from scratch and want market selection, power curves, and synthetic-control or TBR counterfactuals. `causalgeo` does not do design search, and does not try to.

## Design vs analysis

* **Assignment** — which locations are treatment vs control (GeoLift: test vs control markets; GeoX: geo assignment). Overlapping cell-level lists are collapsed to unique geos. Geos in the **universe** that are in neither group are **unassigned** (available for a separate heavy-up, not for the first test’s counterfactual).
* **Analyze** — two-way fixed effects on conversions: geo FE + date FE, standard errors clustered by geo. Optional regressor: **delivered impressions** relative to pretest (dose), not planned spend.

Market selection / power search (GeoLift `GeoLiftMarketSelection`, GeoX design engine) is out of scope for v0.

## Vocabulary

| Term | Meaning |
|---|---|
| Geo / location | Experimental unit (DMA, city, region, …) |
| Pretest period | Before `treatment_start` |
| Test period | On and after `treatment_start` |
| Lift / ATT | Effect of the intervention on conversions, per geo-day |
| Percent lift | Lift ÷ mean control conversions in the test period |
| iCPD / iROAS | Lift per dollar of **incremental** spend |
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
report = exp.analyze(Intervention.GO_DARK)
print(report.lift, report.percent_lift, report.icpd)
```

Heavy-up on **unassigned** geos only:

```python
overlay = AssignmentMap.from_sets(heavy_up_geos, overlay_control_geos)
assignment.assert_overlay_allowed(overlay)  # no overlap with existing test/control
```

## Install

```bash
pip install -e ".[dev]"
```

Private repository while the API is unstable.

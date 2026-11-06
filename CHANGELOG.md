# Changelog

## Unreleased

- A channel update whose posterior covariance is singular, or has a negative variance, stays on nonnegative slopes instead of failing. The update uses only the directions that covariance supports.
- The stability check refits against the same cell lift as the channel split, including a lift with a still-on impression shift removed. Leaving one market out estimates that cell lift again without the market.
- `run_panel_ols_did(cluster_entity=False)` uses HC1 standard errors. Before, the argument was ignored and the standard errors stayed clustered by market.

## 0.2.3 — 2026-11-06

- `diagnose` reads the pretrend p-value of the primary metric. Before, it read the first metric's row.
- The `smd` gate reports `skip` when Hedges g cannot be computed or the primary metric has no g. Before, it reported `pass`. The `pretrend` gate also reports `skip` when there is no p-value.
- `diagnose` raises on an `adjust` value other than `"bh"` or `"none"`. `bh_q` does not change Benjamini-Hochberg adjusted p-values, and the docstring now says so.
- `contingency_table` and `rate_two_by_two` return a missing p-value with a warning when a row or column is empty. Before, they raised a raw scipy error.
- `estimate_sc` warns that its interval uses the pre-test residual spread only and is too narrow for inference.
- `design_concurrent_test` and `design_from_assignment` refuse an assignment with no control markets or an empty test cell. `design_from_assignment` also refuses a bundle channel that is not in the channel list.
- `estimate_identified` names a designed cell that has no markets in the metrics instead of failing with a `KeyError`.
- Geometric adstock keeps the days after a blank impression. Before, one blank turned every later day for that market and channel into a blank.
- `experiment_only_split` returns a blank for a channel the cell equations cannot separate. Before, it returned an arbitrary minimum-norm split with a standard error.
- The balance table skips the variance ratio when either arm has fewer than two markets.
- Channel slopes are sales per impression and stay at or above zero. A slope that would be negative is held at zero, and its usual standard error is left blank. The bundle lift is unchanged and can still be negative.

## 0.2.2 — 2026-10-29

- `examples/public_allocation_benchmark.py` scores the allocation on Google's public Meridian geo panel.
- `anchored_channel_allocation` divides a measured bundle effect by prevented effective dose and reports potency sensitivity separately from the experimental interval.
- `prevented_effective_dose` builds that dose from impressions, quality, reach and frequency, carryover, or saturation.
- `bayesian_channel_allocation` updates a joint MMM prior with measured bundle effects and reports how many channel directions remain prior-driven.
- `bundle_impression_contrasts` exposes the off-channel exposure loss used by the bundle equations.

## 0.2.0 — 2026-10-23

- The concurrent cell estimate stays unadjusted unless `adjust_exposure=True`. A still-on impression shift is removed from the total before the channel split.
- Channel slopes that move under a weaker calibration weight, or without one market, are marked `unstable`.
- `bootstrap_channel_intervals` resamples markets and refits the channel split.
- `compare_calibration` reports the observational fit beside the calibrated fit.
- A missing channel-day record is left missing. A recorded zero stays zero. A blank impression stays blank, the fit drops that market-day, and it warns.
- A channel the design never turns off is labeled `not tested` and gets no slope.
- The market bootstrap uses the caller's market column.
- `design_from_assignment` accepts an assignment that already exists.
- The still-on check uses the joint pre-test slope. `material` and `copy_tolerance` are arguments.
- Two-way demeaning is iterated so a market that misses some periods matches market and period dummies.
- `experiment_only_split` solves the channel slopes from the cell totals alone.
- `examples/concurrent_holdout.py` walks through one design, including a refused split.
- `examples/meridian_conversions.py` runs the split on Meridian's conversions column. That series has no known channel effects.
- `paper/reproduce.py --fast` runs two draws per setting for continuous integration. The full command still checks the published numbers.

## 0.1.0

First citable release of the concurrent bundle estimator.

- Channel slopes use a soft calibration on the same pre/post impression contrast as the experiment.
- Copied channels stay in the fit as one shared control and do not receive separate slopes.
- Time-based regression intervals use a HAC variance that does not fall below the ordinary prediction variance.
- `paper/reproduce.py` rebuilds the manuscript tables and figures.

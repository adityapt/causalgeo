# Changelog

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

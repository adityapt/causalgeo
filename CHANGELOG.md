# Changelog

## 0.1.0

First citable release of the concurrent bundle estimator.

- Channel slopes use a soft calibration on the same pre/post impression contrast as the experiment.
- Copied channels stay in the fit as one shared control and do not receive separate slopes.
- Time-based regression intervals use a HAC variance that does not fall below the ordinary prediction variance.
- `paper/reproduce.py` rebuilds the manuscript tables and figures.

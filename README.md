# causalgeo

Estimand-first **geo incrementality** for marketing experiments: holdout, heavy-up, leftover overlay, and dose on **delivered** exposure (not planned spend).

Private while the API is unstable. Not a GeoLift or GeoX clone; not a Zillow-internal scheduler.

## Scope (v0)

- Parse overlapping assignment lists → unique holdouts, controls, leftover
- Lock a named contrast: average (dark vs BAU), marginal (BAU+X vs BAU), or curve
- DMA-day (or geo-day) panel; TWFE with unit-clustered SEs
- Refuse B-vs-A with no BAU control; refuse iROAS with the wrong denominator

Thin `assign()` may come later. Design-search (GeoX/GeoLift) is out of scope for v0.

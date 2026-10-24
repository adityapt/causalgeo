"""Coverage of the cell intervals when the assignment is redrawn.

Forty markets, eight treated markets in each cell, and thirty periods. The
outcome is a constant cell effect plus an AR(1) residual. This is not the
Meridian example.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from causalgeo.design import design_concurrent_test
from causalgeo.identified import estimate_identified

OUT = Path(__file__).resolve().parent / "results"
N_RUNS = 200
TRUTH = {"1": -8.0, "2": -12.0, "3": -5.0}


def main() -> pd.DataFrame:
    geos = pd.DataFrame({"geo": [f"g{i}" for i in range(40)], "stratum": ["a"] * 40})
    dates = pd.date_range("2024-01-01", periods=30, freq="D")
    start = "2024-01-21"
    start_ts = pd.Timestamp(start)
    rows = []
    for seed in range(N_RUNS):
        design = design_concurrent_test(
            geos,
            ["A", "B", "C"],
            [["A", "B"], ["A", "C"], ["B", "C"]],
            control_share=0.4,
            seed=seed,
        )
        cell = design.assignment.set_index("geo")["cell"].astype(str)
        rng = np.random.default_rng(1000 + seed)
        shock = {geo: 0.0 for geo in cell.index}
        records = []
        for day in dates:
            date_fe = float(rng.normal(0, 2))
            post = day >= start_ts
            for geo, name in cell.items():
                shock[geo] = 0.8 * shock[geo] + float(rng.normal(0, 3))
                effect = TRUTH.get(name, 0.0) if post else 0.0
                records.append({"geo": geo, "date": day, "sales": 20 + effect + date_fe + shock[geo]})
        result = estimate_identified(design, pd.DataFrame(records), "sales", start)
        treated = cell.value_counts().to_dict()
        for rec in result.cells.itertuples(index=False):
            rows.append(
                {
                    "seed": seed,
                    "cell": str(rec.cell),
                    "coef": rec.coef,
                    "se": rec.std_err,
                    "cover": bool(rec.ci_lower <= TRUTH[str(rec.cell)] <= rec.ci_upper),
                    "n_treated": int(treated.get(str(rec.cell), treated.get(rec.cell, 0))),
                }
            )
    detail = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    detail.to_csv(OUT / "cell_coverage_detail.csv", index=False)
    summary = pd.DataFrame(
        [
            {
                "runs": int(detail["seed"].nunique()),
                "coverage": float(detail["cover"].mean()),
                **{
                    f"cell_{name}": float(group["cover"].mean())
                    for name, group in detail.groupby("cell")
                },
            }
        ]
    )
    summary.to_csv(OUT / "cell_coverage.csv", index=False)
    return summary


if __name__ == "__main__":
    print(main().to_string(index=False))

"""Rebuild every computed manuscript table and both figures.

Run from this directory:

    PYTHONPATH=../src python3.11 reproduce.py

Table 1 is manuscript metadata, so this script checks the reported version
against pyproject.toml. It rebuilds Tables 2--4 and both figures, then checks
the rounded values that appear in the manuscript.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

import example
import figures
import reliability

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "results"


def _rounded(frame: pd.DataFrame, columns: dict[str, int]) -> pd.DataFrame:
    out = frame.copy()
    for column, digits in columns.items():
        out[column] = out[column].map(lambda value, digits=digits: round(float(value), digits))
    return out


def _check_tables() -> None:
    cells = pd.read_csv(OUT / "cells.csv")
    bundles = pd.read_csv(OUT / "bundles.csv").set_index("cell")
    bundles.index = bundles.index.astype(str)
    reported_cells = _rounded(
        pd.DataFrame(
            {
                "cell": cells["cell"].astype(str),
                "unadjusted": cells["unadjusted_effect"],
                "adjusted": cells["coef"],
                "se": cells["std_err"],
                "model": cells["cell"].astype(str).map(bundles["predicted_effect"]),
            }
        ),
        {"unadjusted": 1, "adjusted": 1, "se": 1, "model": 1},
    )
    expected_cells = pd.DataFrame(
        [
            ["1", -555.7, -587.2, 71.4, -570.2],
            ["2", -318.0, -251.5, 36.2, -253.6],
            ["3", -399.9, -399.9, 54.4, -396.2],
        ],
        columns=["cell", "unadjusted", "adjusted", "se", "model"],
    )
    pd.testing.assert_frame_equal(reported_cells.reset_index(drop=True), expected_cells)

    slopes = pd.read_csv(OUT / "slopes.csv")
    reported_slopes = pd.DataFrame(
        {
            "channel": slopes["channel"],
            "planted": (slopes["planted"] * 1e4).round(3),
            "recovered": (slopes["estimated"] * 1e4).round(3),
            "se": (slopes["std_err"] * 1e4).round(3),
            "abs_pct_error": slopes["abs_pct_error"].round(2),
        }
    )
    expected_slopes = pd.DataFrame(
        [
            ["Channel0", 2.000, 1.997, 0.006, 0.17],
            ["Channel1", 5.000, 5.009, 0.008, 0.17],
            ["Channel2", 1.000, 1.002, 0.008, 0.21],
        ],
        columns=["channel", "planted", "recovered", "se", "abs_pct_error"],
    )
    pd.testing.assert_frame_equal(reported_slopes.reset_index(drop=True), expected_slopes)

    runs = pd.read_csv(OUT / "reliability.csv")
    reported_runs = _rounded(
        runs,
        {
            "median_abs_pct_error": 2,
            "coverage": 3,
            "coverage_mc_lower": 3,
            "coverage_mc_upper": 3,
            "copy_pair_withheld": 2,
        },
    )[
        [
            "scale",
            "rho",
            "runs",
            "median_abs_pct_error",
            "coverage",
            "coverage_mc_lower",
            "coverage_mc_upper",
            "copy_pair_withheld",
        ]
    ]
    expected_runs = pd.DataFrame(
        [
            [0.25, 0.0, 200, 0.77, 0.958, 0.942, 0.974, 0.00],
            [0.25, 0.9, 200, 1.63, 0.957, 0.938, 0.975, 0.00],
            [0.25, 1.0, 200, 2.24, 0.940, 0.907, 0.973, 1.00],
            [1.0, 0.0, 200, 0.20, 0.958, 0.942, 0.974, 0.00],
            [1.0, 0.9, 200, 0.40, 0.953, 0.934, 0.973, 0.00],
            [1.0, 1.0, 200, 0.58, 0.940, 0.907, 0.973, 1.00],
        ],
        columns=reported_runs.columns,
    )
    pd.testing.assert_frame_equal(
        reported_runs.sort_values(["scale", "rho"]).reset_index(drop=True),
        expected_runs,
    )
    version = re.search(
        r'(?m)^version = "([^"]+)"',
        (ROOT / "pyproject.toml").read_text(),
    ).group(1)
    if version != "0.1.0":
        raise AssertionError(f"manuscript metadata C1 says 0.1.0, pyproject says {version}")
    for name in ("bundle_contrast.pdf", "separation.pdf"):
        path = OUT / name
        if path.stat().st_size < 1000:
            raise AssertionError(f"{name} was not written")


def main() -> None:
    example.main()
    reliability.main()
    figures.main()
    _check_tables()
    print("Reproduced Tables 2-4 and both figures.")


if __name__ == "__main__":
    main()

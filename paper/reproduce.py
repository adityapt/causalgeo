"""Rebuild every computed manuscript table and the four figures.

Run from this directory:

    PYTHONPATH=../src python3.11 reproduce.py

That rebuilds Tables 2--6 and the four figures, then checks the rounded values
in the manuscript. The full run refits every Monte Carlo setting.

    PYTHONPATH=../src python3.11 reproduce.py --fast

That uses two draws per setting, writes to results_fast/, and only checks that
the scripts finish. It does not compare the published numbers. Continuous
integration uses this shorter run.
"""

from __future__ import annotations

import argparse
import importlib.util
import re
from pathlib import Path

import numpy as np
import pandas as pd

import cell_coverage
import example
import figures
import reliability
import robustness
import stress

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
            ["1", -555.7, -555.7, 62.9, -570.2],
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
            "correlation": 2,
            "median_abs_pct_error": 2,
            "coverage": 3,
            "coverage_mc_lower": 3,
            "coverage_mc_upper": 3,
            "copy_pair_withheld": 2,
        },
    )[
        [
            "noise",
            "rho",
            "correlation",
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
            [1.0, 0.0, 0.34, 200, 0.19, 0.958, 0.942, 0.974, 0.00],
            [1.0, 0.9, 0.96, 200, 0.40, 0.953, 0.934, 0.973, 0.00],
            [1.0, 1.0, 1.00, 200, 0.58, 0.940, 0.907, 0.973, 1.00],
            [4.0, 0.0, 0.34, 200, 0.76, 0.958, 0.942, 0.974, 0.00],
            [4.0, 0.9, 0.96, 200, 1.62, 0.957, 0.938, 0.975, 0.00],
            [4.0, 1.0, 1.00, 200, 2.25, 0.940, 0.907, 0.973, 1.00],
        ],
        columns=reported_runs.columns,
    )
    pd.testing.assert_frame_equal(
        reported_runs.sort_values(["noise", "rho"]).reset_index(drop=True),
        expected_runs,
    )
    stress_runs = pd.read_csv(OUT / "stress.csv")
    reported_stress = _rounded(
        stress_runs,
        {
            "median_abs_contribution_error_pct": 2,
            "coverage": 3,
            "coverage_mc_lower": 3,
            "coverage_mc_upper": 3,
            "median_abs_bundle_gap_pct": 2,
        },
    )[
        [
            "scenario",
            "runs",
            "median_abs_contribution_error_pct",
            "coverage",
            "coverage_mc_lower",
            "coverage_mc_upper",
            "median_abs_bundle_gap_pct",
        ]
    ]
    expected_stress = pd.DataFrame(
        [
            ["carryover_omitted", 200, 47.27, 0.000, 0.000, 0.000, 48.09],
            ["carryover_modeled", 200, 0.19, 0.942, 0.922, 0.961, 2.22],
            ["saturation_omitted", 200, 26.99, 0.000, 0.000, 0.000, 26.97],
        ],
        columns=reported_stress.columns,
    )
    pd.testing.assert_frame_equal(reported_stress.reset_index(drop=True), expected_stress)
    robust = pd.read_csv(OUT / "robustness.csv")
    displayed = robust.copy()
    confounded = displayed["scenario"].eq("confounded")
    displayed.loc[confounded, "median_abs_error_pct"] = displayed.loc[confounded, "max_abs_error_pct"]
    reported_robust = _rounded(
        displayed,
        {"median_abs_error_pct": 2, "coverage": 3, "median_abs_bundle_gap_pct": 2},
    )[["scenario", "runs", "median_abs_error_pct", "coverage", "median_abs_bundle_gap_pct"]]
    expected_robust = pd.DataFrame(
        [
            ["heterogeneous", 30, 4.81, 0.344, 4.01],
            ["incomplete", 30, 0.22, 0.967, 2.83],
            ["redistribution", 30, 0.22, 0.967, 12.60],
            ["redistribution_adjusted", 30, 0.22, 0.967, 12.60],
            ["serial", 30, 0.43, 0.956, 3.56],
            ["observational_only", 30, 0.22, 0.967, 2.12],
            ["calibrated", 30, 0.22, 0.967, 2.13],
            ["confounded", 30, 49.94, 0.644, 21.12],
            ["collinear_weak_observational", 30, 44.64, 0.956, 43.33],
            ["collinear_weak_calibrated", 30, 10.52, 0.989, 8.79],
            ["experiment_only", 30, 11.16, 1.000, float("nan")],
        ],
        columns=reported_robust.columns,
    )
    pd.testing.assert_frame_equal(
        reported_robust.sort_values("scenario").reset_index(drop=True),
        expected_robust.sort_values("scenario").reset_index(drop=True),
    )
    version = re.search(
        r'(?m)^version = "([^"]+)"',
        (ROOT / "pyproject.toml").read_text(),
    ).group(1)
    if version != "0.2.0":
        raise AssertionError(f"manuscript metadata C1 says 0.2.0, pyproject says {version}")
    alone = pd.read_csv(OUT / "experiment_only.csv")
    reported_alone = pd.DataFrame(
        {
            "channel": alone["channel"],
            "per_10000": alone["per_10000"].round(2),
            "se_per_10000": alone["se_per_10000"].round(2),
        }
    )
    expected_alone = pd.DataFrame(
        [
            ["Channel0", 2.22, 0.43],
            ["Channel1", 4.48, 0.64],
            ["Channel2", 2.01, 1.14],
        ],
        columns=["channel", "per_10000", "se_per_10000"],
    )
    pd.testing.assert_frame_equal(reported_alone, expected_alone)
    cells = pd.read_csv(OUT / "cell_coverage.csv")
    if round(float(cells["coverage"].iloc[0]), 3) != 0.918:
        raise AssertionError(f"cell coverage is {cells['coverage'].iloc[0]}, manuscript says 0.918")
    refused = pd.read_csv(OUT / "refused.csv")
    reported_refused = _rounded(refused, {"per_10000": 2})
    expected_refused = pd.DataFrame(
        [["Pair kept as one control", 0.97], ["Pair dropped from the fit", 3.02]],
        columns=["analysis", "per_10000"],
    )
    pd.testing.assert_frame_equal(reported_refused[["analysis", "per_10000"]], expected_refused)
    for name in ("bundle_contrast.pdf", "separation.pdf", "refused.pdf", "failures.pdf"):
        path = OUT / name
        if path.stat().st_size < 1000:
            raise AssertionError(f"{name} was not written")


def _redirect(out: Path, n_seeds: int | None) -> None:
    for module in (example, reliability, stress, robustness, cell_coverage, figures):
        module.OUT = out
    if n_seeds is None:
        return
    reliability.SEEDS = range(n_seeds)
    stress.SEEDS = range(n_seeds)
    robustness.SEEDS = range(n_seeds)
    cell_coverage.N_RUNS = n_seeds


def _check_meridian() -> None:
    path = ROOT / "examples" / "meridian_conversions.py"
    spec = importlib.util.spec_from_file_location("meridian_conversions", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result = module.main()
    coefs = result.cells["coef"].to_numpy(dtype=float)
    if len(coefs) != 3 or not np.isfinite(coefs).all():
        raise AssertionError("Meridian conversions run did not return three cell effects")


def _check_fast(out: Path) -> None:
    reliability_table = pd.read_csv(out / "reliability.csv")
    if set(zip(reliability_table["noise"], reliability_table["rho"])) != {
        (1.0, 0.0),
        (1.0, 0.9),
        (1.0, 1.0),
        (4.0, 0.0),
        (4.0, 0.9),
        (4.0, 1.0),
    }:
        raise AssertionError("fast reliability run missed a noise or mixing setting")
    if not reliability_table["coverage"].between(0, 1).all():
        raise AssertionError("fast reliability coverage left [0, 1]")
    stress_table = pd.read_csv(out / "stress.csv")
    if set(stress_table["scenario"]) != {"carryover_omitted", "carryover_modeled", "saturation_omitted"}:
        raise AssertionError("fast stress run missed a scenario")
    robust = pd.read_csv(out / "robustness.csv")
    expected = {
        "serial",
        "heterogeneous",
        "redistribution",
        "redistribution_adjusted",
        "incomplete",
        "observational_only",
        "calibrated",
        "confounded",
        "collinear_weak_observational",
        "collinear_weak_calibrated",
        "experiment_only",
    }
    if set(robust["scenario"]) != expected:
        raise AssertionError(f"fast robustness scenarios were {sorted(robust['scenario'])}")
    cells = pd.read_csv(out / "cell_coverage.csv")
    coverage = float(cells["coverage"].iloc[0])
    if not 0 <= coverage <= 1:
        raise AssertionError(f"fast cell coverage was {coverage}")
    for name in ("bundle_contrast.pdf", "separation.pdf", "refused.pdf", "failures.pdf"):
        path = out / name
        if path.stat().st_size < 1000:
            raise AssertionError(f"{name} was not written")


def main(fast: bool = False) -> None:
    out = (OUT.parent / "results_fast") if fast else OUT
    _redirect(out, 2 if fast else None)
    example.main()
    reliability.main()
    stress.main()
    robustness.main()
    cell_coverage.main()
    figures.main()
    _check_meridian()
    if fast:
        _check_fast(out)
        print("Fast check finished. Published table values were not compared.")
        return
    _check_tables()
    print("Reproduced Tables 2-6 and the four figures.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true")
    main(fast=parser.parse_args().fast)

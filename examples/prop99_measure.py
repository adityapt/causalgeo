"""Measure Proposition 99 and reproduce the published sDiD benchmark.

The canonical public panel has California plus 38 never-treated donor states
from 1970 through 2000. Proposition 99 starts in 1989.

Run:
    python examples/prop99_measure.py
"""

from __future__ import annotations

import pandas as pd

from causalgeo import AssignmentMap, GeoExperiment

DATA_URL = (
    "https://raw.githubusercontent.com/synth-inference/synthdid/"
    "master/data/california_prop99.csv"
)
START = "1989-01-01"
PLACEBO = "Nebraska"
PUBLISHED_SDID_ATT = -15.604
PUBLISHED_SDID_SE = 9.338


def _panel() -> pd.DataFrame:
    raw = pd.read_csv(DATA_URL, sep=";")
    out = raw.rename(
        columns={"State": "location", "PacksPerCapita": "conversions"}
    )
    out["date"] = pd.to_datetime(out["Year"].astype(str) + "-01-01")
    return out


def _print(label, report) -> None:
    print(
        f"{label:24s} lift={report.lift:7.2f}  "
        f"se={report.std_error:6.2f}  95% [{report.ci_lower:7.2f}, {report.ci_upper:7.2f}]  "
        f"p={report.p_value:6.3f}  pre_R2={report.pre_r2:.3f}"
        if report.pre_r2 is not None
        else f"{label:24s} lift={report.lift:7.2f}  "
        f"se={report.std_error:6.2f}  p={report.p_value:6.3f}"
    )


def main() -> None:
    data = _panel()
    donors = set(data["location"].unique()) - {"California"}
    exp = GeoExperiment(
        data,
        AssignmentMap.from_sets(["California"], donors),
        START,
    )

    print("Proposition 99: California vs 38 donor states, 1970-2000")
    print("(negative = fewer packs than the donor prediction)")
    reports = {}
    for method in ("tbr", "sc", "sdid"):
        reports[method] = exp.analyze(
            counterfactual=method,
            sdid_placebo_replications=200,
            random_state=1213,
        )
        _print(method.upper(), reports[method])

    placebo = exp.placebo(
        location=PLACEBO,
        counterfactual="sdid",
        sdid_placebo_replications=200,
        random_state=1213,
    )
    _print("SDID placebo Nebraska", placebo)

    sdid = reports["sdid"]
    print("Sense checks")
    print(
        "  Published ATT reproduced within 0.1:",
        abs(sdid.lift - PUBLISHED_SDID_ATT) < 0.1,
        f"(ours {sdid.lift:.3f}, published {PUBLISHED_SDID_ATT:.3f})",
    )
    print(
        "  Published placebo SE reproduced within 1.0:",
        abs(sdid.std_error - PUBLISHED_SDID_SE) < 1.0,
        f"(ours {sdid.std_error:.3f}, published about {PUBLISHED_SDID_SE:.3f})",
    )
    print(
        "  Nebraska placebo interval covers 0:",
        placebo.ci_lower <= 0 <= placebo.ci_upper,
        f"[{placebo.ci_lower:.1f}, {placebo.ci_upper:.1f}]",
    )


if __name__ == "__main__":
    main()

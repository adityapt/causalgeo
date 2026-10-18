"""Reproduce a public multi-intervention identification check.

Data:
    Abadie, Diamond, and Hainmueller's Proposition 99 cigarette-sales panel,
    as used by the Synthetic Interventions benchmark.

Run:
    python examples/prop99_grid_identification.py
"""

from __future__ import annotations

import pandas as pd

from causalgeo import identify_grid

DATA_URL = (
    "https://raw.githubusercontent.com/jgreathouse9/mlsynth/"
    "main/basedata/prop99_packsales.csv"
)

TAX_STATES = {
    "Alaska",
    "Hawaii",
    "Maryland",
    "Michigan",
    "New Jersey",
    "New York",
    "Washington",
}
PROGRAM_STATES = {"Arizona", "California", "Florida", "Massachusetts", "Oregon"}


def main() -> None:
    panel = pd.read_csv(DATA_URL)
    states = sorted(panel["state"].unique())

    rows = []
    for state in states:
        rows.extend(
            [
                {
                    "cell": state,
                    "channel": "tax increase",
                    "dose": float(state in TAX_STATES),
                },
                {
                    "cell": state,
                    "channel": "anti-tobacco program",
                    "dose": float(state in PROGRAM_STATES),
                },
            ]
        )

    report = identify_grid(pd.DataFrame(rows))
    print("Prop 99 / Synthetic Interventions design")
    print(f"Panel: {len(states)} states x {panel['year'].nunique()} years")
    print(f"Design rank: {report.rank} of {report.n_channels}")
    print(f"Point identified: {report.point_identified}")
    print(report.channel_status.to_string(index=False))

    collapsed = pd.DataFrame(rows)
    collapsed["dose"] = collapsed.groupby("cell")["dose"].transform("max")
    collapsed_report = identify_grid(collapsed)
    print("\nIf the policy type were not recorded:")
    print(f"Design rank: {collapsed_report.rank} of {collapsed_report.n_channels}")
    print(f"Point identified: {collapsed_report.point_identified}")
    print(f"Aliased bundle: {collapsed_report.alias_groups[0]}")


if __name__ == "__main__":
    main()

"""Allocate one measured bundle when no channel prior is available.

The channel numbers add up to the experimental total. Their split follows
the prevented effective dose and the stated potency, so changing the potency
changes the split without changing the total.
"""

import pandas as pd

from causalgeo import (
    anchored_channel_allocation,
    potency_misspecification_ratio,
    potency_tie_ratio,
)


def main() -> None:
    effects = pd.DataFrame(
        [{"cell": "bundle", "coef": 1000.0, "std_err": 120.0}]
    )
    dose = pd.DataFrame(
        [
            {"cell": "bundle", "channel": "Search", "dose": 40_000.0},
            {"cell": "bundle", "channel": "Social", "dose": 80_000.0},
            {"cell": "bundle", "channel": "Video", "dose": 80_000.0},
        ]
    )
    result = anchored_channel_allocation(
        effects,
        dose,
        potency={"Search": 2.0, "Social": 1.0, "Video": 0.5},
        potency_bounds={
            "Search": (1.0, 3.0),
            "Social": (0.5, 1.5),
            "Video": (0.25, 1.0),
        },
    )
    print(
        result.channels[
            [
                "channel",
                "share",
                "allocation",
                "ci_lower",
                "ci_upper",
                "assumption_lower",
                "assumption_upper",
            ]
        ].to_string(index=False)
    )
    print()
    print(potency_misspecification_ratio(
        {"Search": 40_000.0, "Social": 80_000.0, "Video": 80_000.0},
        {"Search": 2.0, "Social": 1.0, "Video": 0.5},
        {"Search": 1.0, "Social": 1.0, "Video": 1.0},
    ))
    print(
        "Search and Social tie when Search potency / Social potency = "
        f"{potency_tie_ratio(40_000.0, 80_000.0):.2f}"
    )


if __name__ == "__main__":
    main()

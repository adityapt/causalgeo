"""Allocate one measured eight-channel bundle with an external MMM prior.

The experiment identifies one direction. The other seven posterior directions
remain prior-driven, which is reported explicitly.
"""

import numpy as np
import pandas as pd

from causalgeo import bayesian_channel_allocation


CHANNELS = [f"Channel{i}" for i in range(8)]


def main() -> None:
    prior_mean = {channel: value for channel, value in zip(CHANNELS, np.linspace(0.1, 0.8, 8))}
    prior_sd = np.linspace(0.04, 0.10, 8)
    correlation = np.full((8, 8), 0.8)
    np.fill_diagonal(correlation, 1.0)
    prior_covariance = pd.DataFrame(
        correlation * np.outer(prior_sd, prior_sd),
        index=CHANNELS,
        columns=CHANNELS,
    )
    withheld = pd.DataFrame(
        {
            "cell": ["all-eight"] * 8,
            "channel": CHANNELS,
            "withheld": np.linspace(100, 800, 8),
        }
    )
    bundle_effects = pd.DataFrame(
        [{"cell": "all-eight", "coef": -1750.0, "std_err": 180.0}]
    )

    result = bayesian_channel_allocation(
        prior_mean,
        prior_covariance,
        bundle_effects,
        withheld,
    )

    print(
        result.channels[
            [
                "channel",
                "prior_coef",
                "posterior_coef",
                "posterior_std_err",
                "variance_reduction",
            ]
        ].to_string(index=False)
    )
    print(
        f"\nThe experiment tested rank {result.tested_rank}; "
        f"{result.remaining_dimensions} channel directions remain prior-driven."
    )


if __name__ == "__main__":
    main()

import numpy as np
import pandas as pd
import pytest

from causalgeo import (
    bayesian_channel_allocation,
    bundle_impression_contrasts,
    design_concurrent_test,
)


def test_one_bundle_updates_only_one_direction():
    effects = pd.DataFrame(
        [{"cell": "1", "coef": -4.0, "std_err": 1.0}]
    )
    withheld = pd.DataFrame(
        [
            {"cell": "1", "channel": "Video", "withheld": 1.0},
            {"cell": "1", "channel": "Search", "withheld": 1.0},
        ]
    )

    result = bayesian_channel_allocation(
        prior_mean={"Video": 1.0, "Search": 1.0},
        prior_covariance=pd.DataFrame(
            [[1.0, 0.0], [0.0, 1.0]],
            index=["Video", "Search"],
            columns=["Video", "Search"],
        ),
        bundle_effects=effects,
        withheld=withheld,
    )

    channels = result.channels.set_index("channel")
    assert channels.loc["Video", "posterior_coef"] == pytest.approx(5 / 3)
    assert channels.loc["Search", "posterior_coef"] == pytest.approx(5 / 3)
    assert channels.loc["Video", "variance_reduction"] == pytest.approx(1 / 3)
    assert channels.loc["Video", "ci_lower"] == pytest.approx(
        5 / 3 - 1.96 * np.sqrt(2 / 3)
    )
    assert channels.loc["Video", "ci_upper"] == pytest.approx(
        5 / 3 + 1.96 * np.sqrt(2 / 3)
    )
    assert result.posterior_covariance.loc["Video", "Search"] == pytest.approx(-1 / 3)
    assert result.tested_rank == 1
    assert result.remaining_dimensions == 1
    assert result.bundles.loc[0, "posterior_predicted_effect"] == pytest.approx(-10 / 3)


def test_channel_labels_align_prior_covariance_and_withheld_rows():
    effects = pd.DataFrame(
        [{"cell": "bundle", "coef": -5.0, "std_err": 0.5}]
    )
    withheld = pd.DataFrame(
        [
            {"cell": "bundle", "channel": "B", "withheld": 2.0},
            {"cell": "bundle", "channel": "A", "withheld": 1.0},
        ]
    )
    covariance = pd.DataFrame(
        [[4.0, 0.0], [0.0, 1.0]],
        index=["B", "A"],
        columns=["B", "A"],
    )

    result = bayesian_channel_allocation(
        prior_mean={"A": 1.0, "B": 1.0},
        prior_covariance=covariance,
        bundle_effects=effects,
        withheld=withheld,
    )

    channels = result.channels.set_index("channel")
    assert channels.loc["B", "posterior_update"] > channels.loc["A", "posterior_update"]
    assert result.bundles.loc[0, "prior_predicted_effect"] == pytest.approx(-3.0)


def test_allocation_rejects_non_positive_definite_prior_covariance():
    effects = pd.DataFrame(
        [{"cell": "1", "coef": -2.0, "std_err": 1.0}]
    )
    withheld = pd.DataFrame(
        [
            {"cell": "1", "channel": "A", "withheld": 1.0},
            {"cell": "1", "channel": "B", "withheld": 1.0},
        ]
    )

    with pytest.raises(ValueError, match="positive definite"):
        bayesian_channel_allocation(
            prior_mean={"A": 1.0, "B": 1.0},
            prior_covariance=np.array([[1.0, 1.0], [1.0, 1.0]]),
            bundle_effects=effects,
            withheld=withheld,
        )


def test_bundle_impression_contrasts_builds_the_bayesian_design_row():
    design = design_concurrent_test(
        pd.DataFrame({"geo": ["dark", "control"], "stratum": ["a", "a"]}),
        channels=["A", "B"],
        bundles=[["A", "B"]],
        control_share=0.5,
        seed=0,
    )
    design.assignment["geo"] = ["dark", "control"]
    design.assignment["cell"] = ["1", "control"]
    design.assignment["role"] = ["test", "control"]
    impressions = pd.DataFrame(
        [
            {"geo": geo, "date": date, "channel": channel, "impressions": value}
            for geo, date, channel, value in [
                ("dark", "2025-01-01", "A", 10.0),
                ("dark", "2025-01-02", "A", 0.0),
                ("control", "2025-01-01", "A", 8.0),
                ("control", "2025-01-02", "A", 8.0),
                ("dark", "2025-01-01", "B", 20.0),
                ("dark", "2025-01-02", "B", 0.0),
                ("control", "2025-01-01", "B", 16.0),
                ("control", "2025-01-02", "B", 16.0),
            ]
        ]
    )

    contrast = bundle_impression_contrasts(
        impressions,
        design,
        treatment_start="2025-01-02",
    ).set_index("channel")

    assert contrast.loc["A", "withheld"] == pytest.approx(10.0)
    assert contrast.loc["B", "withheld"] == pytest.approx(20.0)

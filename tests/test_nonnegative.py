import numpy as np
import pandas as pd
import pytest

from causalgeo import bayesian_channel_allocation
from causalgeo.nonnegative import nonnegative_gaussian_mode, nonnegative_least_squares


def test_least_squares_holds_a_downward_slope_at_zero():
    design = np.array([[1.0], [1.0], [1.0]])
    sales = np.array([-2.0, -2.0, -2.0])

    coef, se = nonnegative_least_squares(design, sales)

    assert coef == pytest.approx([0.0])
    assert np.isnan(se[0])


def test_least_squares_matches_an_unconstrained_positive_slope():
    design = np.array([[1.0, 0.0], [0.0, 2.0], [1.0, 0.0], [0.0, 2.0]])
    sales = design @ np.array([0.4, 0.25])

    coef, se = nonnegative_least_squares(design, sales)

    assert coef == pytest.approx([0.4, 0.25])
    assert np.isfinite(se).all()


def test_gaussian_mode_leaves_a_positive_posterior_unchanged():
    mean = np.array([0.2, 0.5])
    covariance = np.array([[0.04, 0.0], [0.0, 0.09]])

    mode, se, reported = nonnegative_gaussian_mode(mean, covariance)

    assert mode == pytest.approx(mean)
    assert se == pytest.approx([0.2, 0.3])
    assert reported == pytest.approx(covariance)


def test_gaussian_mode_holds_a_negative_channel_at_zero():
    mode, se, reported = nonnegative_gaussian_mode(
        np.array([-0.4, 0.3]),
        np.eye(2),
    )

    assert mode == pytest.approx([0.0, 0.3])
    assert np.isnan(se[0])
    assert se[1] == pytest.approx(1.0)
    assert reported[0, 0] == pytest.approx(0.0)
    assert reported[1, 1] == pytest.approx(1.0)


def test_bayesian_allocation_does_not_report_a_negative_slope():
    result = bayesian_channel_allocation(
        prior_mean={"Search": 0.0, "Video": 0.0},
        prior_covariance=np.eye(2),
        bundle_effects=pd.DataFrame([{"cell": "1", "coef": 10.0, "std_err": 0.1}]),
        withheld=pd.DataFrame(
            [
                {"cell": "1", "channel": "Search", "withheld": 1.0},
                {"cell": "1", "channel": "Video", "withheld": 1.0},
            ]
        ),
    )

    assert (result.channels["posterior_coef"] >= 0).all()
    assert result.channels["posterior_coef"].sum() == pytest.approx(0.0)

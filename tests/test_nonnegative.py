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


def test_gaussian_mode_keeps_the_free_slope_when_channels_are_correlated():
    mode, se, reported = nonnegative_gaussian_mode(
        np.array([-0.5, 0.2]),
        np.array([[1.0, 0.8], [0.8, 1.0]]),
    )

    assert mode == pytest.approx([0.0, 0.6])
    assert np.isnan(se[0])
    assert se[1] == pytest.approx(0.6)
    assert reported[1, 1] == pytest.approx(0.36)
    assert reported[0, 0] == pytest.approx(0.0)


def test_gaussian_mode_moves_a_tied_pair_along_its_shared_direction():
    with pytest.warns(UserWarning, match="not positive definite"):
        mode, se, reported = nonnegative_gaussian_mode(
            np.array([-1.0, 0.5]),
            np.array([[1.0, 1.0], [1.0, 1.0]]),
        )

    assert mode == pytest.approx([0.0, 1.5])
    assert np.isnan(se[0])
    assert se[1] == pytest.approx(0.0, abs=1e-8)
    assert np.linalg.eigvalsh(reported).min() >= -1e-8


def test_gaussian_mode_drops_a_negative_variance():
    with pytest.warns(UserWarning, match="not positive definite"):
        mode, se, reported = nonnegative_gaussian_mode(
            np.array([-1.0, 0.4]),
            np.diag([1.0, -1e-12]),
        )

    assert mode == pytest.approx([0.0, 0.4])
    assert np.isnan(se[0])
    assert reported[1, 1] >= -1e-8
    assert np.linalg.eigvalsh(reported).min() >= -1e-8


def test_gaussian_mode_of_a_point_mass_below_zero_is_zero():
    with pytest.warns(UserWarning) as caught:
        mode, se, reported = nonnegative_gaussian_mode(
            np.array([-2.0, -3.0]),
            np.zeros((2, 2)),
        )
    messages = " ".join(str(item.message) for item in caught)
    assert "not positive definite" in messages
    assert "no nonnegative slope lies on the posterior support" in messages

    assert mode == pytest.approx([0.0, 0.0])
    assert np.isnan(se).all()
    assert reported == pytest.approx(np.zeros((2, 2)))


def test_precise_bundle_with_negative_slopes_stays_nonnegative():
    with pytest.warns(UserWarning) as caught:
        result = bayesian_channel_allocation(
            prior_mean={"Search": -2.0, "Video": 0.4},
            prior_covariance=np.eye(2),
            bundle_effects=pd.DataFrame([{"cell": "1", "coef": 8.0, "std_err": 1e-8}]),
            withheld=pd.DataFrame(
                [
                    {"cell": "1", "channel": "Search", "withheld": 1.0},
                    {"cell": "1", "channel": "Video", "withheld": 1.0},
                ]
            ),
        )
    messages = " ".join(str(item.message) for item in caught)
    assert "not positive definite" in messages
    assert "no nonnegative slope lies on the posterior support" in messages

    assert (result.channels["posterior_coef"] >= -1e-8).all()
    assert result.channels["posterior_coef"].sum() == pytest.approx(0.0)
    assert np.isnan(result.channels["posterior_std_err"]).all()
    covariance = result.posterior_covariance.to_numpy()
    assert np.linalg.eigvalsh(covariance).min() >= -1e-8


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

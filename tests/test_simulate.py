"""Monte Carlo gates: bias, coverage, A/A type I, iid SEs undercover."""

import pytest

from causalgeo.simulate import run_monte_carlo


def test_aa_clustered_intervals_are_not_garbage():
    """True ATT = 0: clustered 95% CIs should cover and not reject ~half the time."""
    summary = run_monte_carlo(n_sims=80, att=0.0, seed=11)
    assert abs(summary.bias) < 0.25
    assert summary.coverage_clustered >= 0.85
    assert summary.reject_clustered <= 0.15


def test_planted_att_coverage_and_bias():
    summary = run_monte_carlo(n_sims=80, att=-4.0, seed=13)
    assert abs(summary.bias) < 0.25
    assert summary.rmse < 1.0
    assert summary.coverage_clustered >= 0.85
    assert summary.reject_clustered >= 0.80
    assert summary.mean_icpd == pytest.approx(summary.true_icpd, rel=0.15)


def test_iid_ses_undercount_serial_correlation():
    """AR(1) within geo: iid CIs should cover less often than clustered CIs."""
    summary = run_monte_carlo(n_sims=80, att=-4.0, seed=17, rho=0.8)
    assert summary.coverage_iid < summary.coverage_clustered
    assert summary.coverage_iid < 0.85

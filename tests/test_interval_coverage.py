"""Coverage of HAC TBR and sDiD placebo intervals, including small designs."""

from causalgeo.simulate import run_sdid_coverage, run_tbr_coverage


def test_hac_tbr_intervals_cover_on_a_moderate_panel():
    summary = run_tbr_coverage(
        n_sims=80,
        n_pre=21,
        n_post=14,
        n_treatment=8,
        n_control=8,
        seed=3,
    )
    assert abs(summary.bias) < 0.35
    assert summary.coverage >= 0.80


def test_hac_tbr_intervals_cover_with_a_short_preperiod_and_few_controls():
    summary = run_tbr_coverage(
        n_sims=80,
        n_pre=6,
        n_post=8,
        n_treatment=3,
        n_control=3,
        seed=5,
    )
    assert abs(summary.bias) < 0.50
    assert summary.coverage >= 0.80


def test_sdid_placebo_intervals_cover_with_a_short_preperiod_and_few_controls():
    """Four controls and six pretest dates do not justify a nominal-95% claim.

    The gate fails if the placebo interval misses substantially more often than
    the checked small design, or if its standard error is far below the
    empirical spread.
    """
    summary = run_sdid_coverage(
        n_sims=80,
        n_control=4,
        n_pre=6,
        n_post=6,
        seed=9,
    )
    assert abs(summary.bias) < 0.40
    assert summary.mean_se >= 0.6 * summary.empirical_sd
    assert summary.coverage >= 0.70

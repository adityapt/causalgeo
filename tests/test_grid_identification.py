"""Identification checks for concurrent interventions sharing a donor pool."""

import numpy as np
import pandas as pd

from causalgeo import identify_grid

# Synthetic Interventions / mlsynth Prop 99 groups (binary assignment, not outcomes).
PROP99_STATES = [
    "Alabama",
    "Alaska",
    "Arizona",
    "Arkansas",
    "California",
    "Colorado",
    "Connecticut",
    "Delaware",
    "Florida",
    "Georgia",
    "Hawaii",
    "Idaho",
    "Illinois",
    "Indiana",
    "Iowa",
    "Kansas",
    "Kentucky",
    "Louisiana",
    "Maine",
    "Maryland",
    "Massachusetts",
    "Michigan",
    "Minnesota",
    "Mississippi",
    "Missouri",
    "Montana",
    "Nebraska",
    "Nevada",
    "New Hampshire",
    "New Jersey",
    "New Mexico",
    "New York",
    "North Carolina",
    "North Dakota",
    "Ohio",
    "Oklahoma",
    "Oregon",
    "Pennsylvania",
    "Rhode Island",
    "South Carolina",
    "South Dakota",
    "Tennessee",
    "Texas",
    "Utah",
    "Vermont",
    "Virginia",
    "Washington",
    "West Virginia",
    "Wisconsin",
    "Wyoming",
]
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


def _prop99_intervention_design() -> pd.DataFrame:
    rows = []
    for state in PROP99_STATES:
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
    return pd.DataFrame(rows)


def test_prop99_tax_and_program_interventions_are_separable():
    report = identify_grid(_prop99_intervention_design())
    dose = report.dose_matrix

    assert report.n_cells == 50
    assert report.n_channels == 2
    assert report.rank == 2
    assert report.point_identified
    assert report.alias_groups == ()
    assert report.null_space.empty
    assert np.isfinite(report.condition_number)
    assert report.condition_number < 2

    tax = dose["tax increase"]
    program = dose["anti-tobacco program"]
    assert set(tax[tax == 1].index) == TAX_STATES
    assert set(program[program == 1].index) == PROGRAM_STATES
    assert int((tax * program).sum()) == 0
    assert int(((dose == 0).all(axis=1)).sum()) == 38
    assert set(report.channel_status["status"]) == {"identified"}
    norms = dict(zip(report.channel_status["channel"], report.channel_status["dose_norm"]))
    assert norms["tax increase"] == np.sqrt(7)
    assert norms["anti-tobacco program"] == np.sqrt(5)


def test_collapsing_prop99_interventions_creates_an_alias():
    design = _prop99_intervention_design()
    design["dose"] = design.groupby("cell")["dose"].transform("max")

    report = identify_grid(design)

    assert report.rank == 1
    assert not report.point_identified
    assert report.alias_groups == (("anti-tobacco program", "tax increase"),)
    assert np.isinf(report.condition_number)
    assert set(report.channel_status["status"]) == {"aliased"}
    null = report.null_space["basis_1"].to_numpy()
    assert np.isclose(np.abs(null[0]), np.sqrt(0.5))
    assert np.isclose(null.sum(), 0.0)
    residual = report.dose_matrix.to_numpy() @ null
    assert np.linalg.norm(residual) < 1e-12


def test_proportional_doses_are_aliased_even_if_scales_differ():
    design = pd.DataFrame(
        {
            "cell": ["a", "a", "b", "b", "c", "c"],
            "channel": ["facebook", "tiktok", "facebook", "tiktok", "facebook", "tiktok"],
            "dose": [1.0, 2.0, 1.0, 2.0, 0.0, 0.0],
        }
    )
    report = identify_grid(design)
    assert report.rank == 1
    assert report.alias_groups == (("facebook", "tiktok"),)
    assert set(report.channel_status["status"]) == {"aliased"}


def test_unused_channel_is_marked_unused():
    design = pd.DataFrame(
        {
            "cell": ["a", "a", "b", "b"],
            "channel": ["search", "display", "search", "display"],
            "dose": [1.0, 0.0, 0.0, 0.0],
        }
    )
    report = identify_grid(design)
    status = dict(zip(report.channel_status["channel"], report.channel_status["status"]))
    assert report.rank == 1
    assert status["search"] == "identified"
    assert status["display"] == "unused"


def test_general_linear_dependency_is_reported_even_without_pairwise_alias():
    wide = pd.DataFrame(
        {
            "cell": ["a", "b", "c"],
            "search": [1.0, 0.0, 1.0],
            "social": [0.0, 1.0, 1.0],
            "all_paid": [1.0, 1.0, 2.0],
        }
    )
    design = wide.melt(id_vars="cell", var_name="channel", value_name="dose")

    report = identify_grid(design)

    assert report.rank == 2
    assert not report.point_identified
    assert report.alias_groups == ()
    assert report.null_space.shape == (3, 1)
    assert set(report.channel_status["status"]) == {"linearly dependent"}

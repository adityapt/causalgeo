"""Bayesian allocation of measured bundle effects to channel slopes.

The experiment supplies one noisy equation per bundle. The prior mean and
covariance come from an observational mix model or other external evidence.
The update is linear Gaussian: it changes only the directions measured by the
bundle equations and leaves the remaining directions prior-driven.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BayesianAllocation:
    """Posterior channel allocation and the bundle equations that informed it."""

    channels: pd.DataFrame
    posterior_covariance: pd.DataFrame
    bundles: pd.DataFrame
    tested_rank: int
    remaining_dimensions: int


def bayesian_channel_allocation(
    prior_mean: Mapping[str, float] | pd.Series,
    prior_covariance: pd.DataFrame | np.ndarray,
    bundle_effects: pd.DataFrame,
    withheld: pd.DataFrame,
) -> BayesianAllocation:
    """Update channel slopes with one or more measured bundle effects.

    ``bundle_effects`` must contain ``cell``, ``coef``, and ``std_err``.
    Effects use the geo-experiment sign convention: a go-dark loss is
    negative. ``withheld`` must contain ``cell``, ``channel``, and
    ``withheld``, where withheld exposure is positive.

    The prior is not created by the bundle experiment. It should be a joint
    mean and covariance from an observational model or other external
    evidence. When there are fewer independent bundle equations than
    channels, ``remaining_dimensions`` reports how many posterior directions
    remain prior-driven.
    """
    mean = pd.Series(prior_mean, dtype=float)
    if mean.empty or mean.index.has_duplicates:
        raise ValueError("prior_mean must name each channel exactly once")
    channels = [str(channel) for channel in mean.index]
    mean.index = channels
    covariance = _covariance_frame(prior_covariance, channels)
    _validate_covariance(covariance)
    effects = _effects(bundle_effects)
    contrast = _contrast(withheld, effects["cell"].tolist(), channels)

    h = contrast.to_numpy(dtype=float)
    y = -effects["coef"].to_numpy(dtype=float)
    standard_errors = effects["std_err"].to_numpy(dtype=float)
    prior = mean.to_numpy(dtype=float)
    covariance_array = covariance.to_numpy(dtype=float)

    innovation = y - h @ prior
    innovation_covariance = h @ covariance_array @ h.T + np.diag(standard_errors**2)
    gain = np.linalg.solve(
        innovation_covariance,
        h @ covariance_array,
    ).T
    posterior = prior + gain @ innovation
    posterior_covariance_array = covariance_array - gain @ h @ covariance_array
    posterior_covariance_array = (
        posterior_covariance_array + posterior_covariance_array.T
    ) / 2

    prior_variance = np.diag(covariance_array)
    posterior_variance = np.clip(np.diag(posterior_covariance_array), 0, None)
    posterior_standard_error = np.sqrt(posterior_variance)
    channel_table = pd.DataFrame(
        {
            "channel": channels,
            "prior_coef": prior,
            "prior_std_err": np.sqrt(prior_variance),
            "posterior_coef": posterior,
            "posterior_std_err": posterior_standard_error,
            "ci_lower": posterior - 1.96 * posterior_standard_error,
            "ci_upper": posterior + 1.96 * posterior_standard_error,
            "posterior_update": posterior - prior,
            "variance_reduction": 1 - posterior_variance / prior_variance,
            "reading": "Bayesian model-assisted allocation; not a channel-level experimental estimate",
        }
    )
    bundle_table = effects.copy()
    bundle_table["prior_predicted_effect"] = -(h @ prior)
    bundle_table["posterior_predicted_effect"] = -(h @ posterior)
    bundle_table["innovation"] = innovation
    bundle_table["posterior_std_err"] = np.sqrt(
        np.clip(np.diag(h @ posterior_covariance_array @ h.T), 0, None)
    )
    rank = int(np.linalg.matrix_rank(h))
    return BayesianAllocation(
        channels=channel_table,
        posterior_covariance=pd.DataFrame(
            posterior_covariance_array,
            index=channels,
            columns=channels,
        ),
        bundles=bundle_table,
        tested_rank=rank,
        remaining_dimensions=len(channels) - rank,
    )


def _covariance_frame(
    covariance: pd.DataFrame | np.ndarray,
    channels: list[str],
) -> pd.DataFrame:
    if isinstance(covariance, pd.DataFrame):
        if set(covariance.index.astype(str)) != set(channels) or set(
            covariance.columns.astype(str)
        ) != set(channels):
            raise ValueError("prior_covariance labels must match prior_mean channels")
        frame = covariance.copy()
        frame.index = frame.index.astype(str)
        frame.columns = frame.columns.astype(str)
        return frame.loc[channels, channels].astype(float)
    array = np.asarray(covariance, dtype=float)
    if array.shape != (len(channels), len(channels)):
        raise ValueError("prior_covariance must have one row and column per channel")
    return pd.DataFrame(array, index=channels, columns=channels)


def _validate_covariance(covariance: pd.DataFrame) -> None:
    values = covariance.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("prior_covariance must contain finite values")
    if not np.allclose(values, values.T, atol=1e-10, rtol=1e-10):
        raise ValueError("prior_covariance must be symmetric")
    try:
        np.linalg.cholesky(values)
    except np.linalg.LinAlgError as error:
        raise ValueError("prior_covariance must be positive definite") from error


def _effects(bundle_effects: pd.DataFrame) -> pd.DataFrame:
    required = {"cell", "coef", "std_err"}
    missing = required - set(bundle_effects.columns)
    if missing:
        raise ValueError(f"bundle_effects is missing {sorted(missing)}")
    effects = bundle_effects[["cell", "coef", "std_err"]].copy()
    effects["cell"] = effects["cell"].astype(str)
    if effects["cell"].duplicated().any():
        raise ValueError("bundle_effects must have one row per cell")
    effects[["coef", "std_err"]] = effects[["coef", "std_err"]].astype(float)
    if not np.isfinite(effects[["coef", "std_err"]].to_numpy()).all():
        raise ValueError("bundle effects and standard errors must be finite")
    if (effects["std_err"] <= 0).any():
        raise ValueError("bundle standard errors must be positive")
    return effects.reset_index(drop=True)


def _contrast(
    withheld: pd.DataFrame,
    cells: list[str],
    channels: list[str],
) -> pd.DataFrame:
    required = {"cell", "channel", "withheld"}
    missing = required - set(withheld.columns)
    if missing:
        raise ValueError(f"withheld is missing {sorted(missing)}")
    frame = withheld[["cell", "channel", "withheld"]].copy()
    frame["cell"] = frame["cell"].astype(str)
    frame["channel"] = frame["channel"].astype(str)
    unknown = set(frame["channel"]) - set(channels)
    if unknown:
        raise ValueError(f"withheld has channels absent from prior_mean: {sorted(unknown)}")
    unknown_cells = set(frame["cell"]) - set(cells)
    if unknown_cells:
        raise ValueError(f"withheld has cells absent from bundle_effects: {sorted(unknown_cells)}")
    frame["withheld"] = frame["withheld"].astype(float)
    if not np.isfinite(frame["withheld"]).all() or (frame["withheld"] < 0).any():
        raise ValueError("withheld exposure must be finite and nonnegative")
    return (
        frame.pivot_table(
            index="cell",
            columns="channel",
            values="withheld",
            aggfunc="sum",
            fill_value=0.0,
        )
        .reindex(index=cells, columns=channels, fill_value=0.0)
        .astype(float)
    )

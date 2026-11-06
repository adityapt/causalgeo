"""Channel slopes are sales per impression, so they stay at or above zero.

The bundle lift is a different number. A go-dark loss is negative, and that
cell total is not passed through these functions.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize, nnls


def nonnegative_least_squares(
    x: np.ndarray,
    y: np.ndarray,
    weight: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit ``y`` on ``x`` with every coefficient held at or above zero.

    A coefficient that sits at zero was not free to move below zero, so its
    standard error is missing. The standard errors on the positive coefficients
    come from the ordinary fit on those columns alone.
    """
    design = np.asarray(x, dtype=float)
    target = np.asarray(y, dtype=float)
    if design.ndim != 2:
        raise ValueError("x must have one column per channel")
    if target.shape != (design.shape[0],):
        raise ValueError("y must have one row per row of x")
    coefficients = np.zeros(design.shape[1], dtype=float)
    standard_errors = np.full(design.shape[1], np.nan)
    if design.size == 0 or target.size == 0:
        return coefficients, standard_errors
    scale = np.ones(design.shape[0], dtype=float) if weight is None else np.sqrt(np.asarray(weight, dtype=float))
    if scale.shape != (design.shape[0],):
        raise ValueError("weight must have one value per row of x")
    fitted, _ = nnls(design * scale[:, None], target * scale)
    floor = 1e-12 * max(float(np.max(fitted)), 1.0)
    fitted[fitted <= floor] = 0.0
    active = fitted > 0
    if active.any():
        columns = design[:, active] * scale[:, None]
        gram = columns.T @ columns
        covariance = np.linalg.pinv(gram, rcond=1e-8)
        standard_errors[active] = np.sqrt(np.clip(np.diag(covariance), 0, None))
    coefficients[active] = fitted[active]
    return coefficients, standard_errors


def nonnegative_gaussian_mode(
    mean: np.ndarray,
    covariance: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Move a Gaussian channel posterior onto slopes that are at least zero.

    When the unconstrained mean is already nonnegative, the mean and covariance
    are returned unchanged. A slope held at zero has a missing standard error.
    """
    center = np.asarray(mean, dtype=float).copy()
    dispersion = np.asarray(covariance, dtype=float).copy()
    if center.ndim != 1 or dispersion.shape != (center.size, center.size):
        raise ValueError("covariance must be square and match the mean")
    if np.all(center >= -1e-10):
        center[center < 0] = 0.0
        return center, np.sqrt(np.clip(np.diag(dispersion), 0, None)), dispersion

    precision = np.linalg.inv(dispersion)

    def objective(point: np.ndarray) -> float:
        gap = point - center
        return float(0.5 * gap @ precision @ gap)

    def gradient(point: np.ndarray) -> np.ndarray:
        return precision @ (point - center)

    start = np.maximum(center, 0.0)
    fitted = minimize(
        objective,
        start,
        jac=gradient,
        bounds=[(0.0, None)] * center.size,
        method="L-BFGS-B",
    )
    if not fitted.success:
        raise ValueError("the nonnegative channel update did not converge")
    mode = np.maximum(np.asarray(fitted.x, dtype=float), 0.0)
    floor = 1e-8 * max(float(np.max(mode)), 1.0)
    mode[mode <= floor] = 0.0
    active = mode > 0
    standard_errors = np.full(center.size, np.nan)
    reported = np.zeros_like(dispersion)
    if active.any():
        free = np.linalg.inv(precision[np.ix_(active, active)])
        reported[np.ix_(active, active)] = (free + free.T) / 2
        standard_errors[active] = np.sqrt(np.clip(np.diag(free), 0, None))
    return mode, standard_errors, reported

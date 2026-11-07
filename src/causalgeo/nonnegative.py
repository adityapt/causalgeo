"""Channel slopes are sales per impression, so they stay at or above zero.

The bundle lift is a different number. A go-dark loss is negative, and that
cell total is not passed through these functions.
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy.optimize import minimize, nnls

_NOT_POSITIVE_DEFINITE = (
    "the channel posterior covariance is not positive definite, so slopes "
    "at or above zero use only the directions it supports"
)
_OFF_SUPPORT = (
    "no nonnegative slope lies on the posterior support; slopes below zero "
    "were set to zero"
)


def nonnegative_least_squares(
    x: np.ndarray,
    y: np.ndarray,
    weight: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit ``y`` on ``x`` with every coefficient held at or above zero.

    A coefficient that sits at zero was not free to move below zero, so its
    standard error is missing. The standard errors on the positive coefficients
    are sqrt(diag((X'WX)^-1)) on those columns. That is the standard error of a
    weighted fit only when the weights are the inverse variances of the rows, so
    pass weights on that scale. No residual variance is estimated here.
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

    When the unconstrained mean is already nonnegative and the covariance is
    positive definite, both are returned unchanged. A singular or indefinite
    covariance is reduced to the directions it supports. A slope held at zero
    has a missing standard error.
    """
    center = np.asarray(mean, dtype=float).copy()
    dispersion = np.asarray(covariance, dtype=float).copy()
    if center.ndim != 1 or dispersion.shape != (center.size, center.size):
        raise ValueError("covariance must be square and match the mean")
    dispersion = (dispersion + dispersion.T) / 2
    eigenvalues, eigenvectors = np.linalg.eigh(dispersion)
    scale = max(float(np.max(np.abs(eigenvalues))), 1.0)
    tolerance = 1e-10 * scale
    indefinite = bool(np.any(eigenvalues < -tolerance))
    singular = bool(np.any(eigenvalues <= tolerance))
    if np.all(center >= -1e-10) and not indefinite:
        center[center < 0] = 0.0
        return center, np.sqrt(np.clip(np.diag(dispersion), 0, None)), dispersion
    if indefinite or singular:
        warnings.warn(_NOT_POSITIVE_DEFINITE, UserWarning, stacklevel=2)

    eigenvalues = np.maximum(eigenvalues, 0.0)
    supported = (eigenvectors * eigenvalues) @ eigenvectors.T
    supported = (supported + supported.T) / 2
    mode = _nearest_nonnegative(center, eigenvalues, eigenvectors, tolerance)
    floor = 1e-8 * max(float(np.max(mode)), 1.0)
    mode[mode <= floor] = 0.0
    standard_errors, reported = _covariance_on_free_slopes(supported, mode > 0)
    return mode, standard_errors, reported


def _nearest_nonnegative(
    center: np.ndarray,
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    tolerance: float,
) -> np.ndarray:
    """Closest nonnegative point on the covariance's supporting subspace."""
    free = eigenvalues > tolerance
    if not free.any():
        warnings.warn(_OFF_SUPPORT, UserWarning, stacklevel=3)
        return np.maximum(center, 0.0)
    basis = eigenvectors[:, free] * np.sqrt(eigenvalues[free])

    def slope(weight: np.ndarray) -> np.ndarray:
        return center + basis @ weight

    start = (basis.T @ (np.maximum(center, 0.0) - center)) / eigenvalues[free]
    fitted = minimize(
        lambda weight: 0.5 * float(weight @ weight),
        start,
        jac=lambda weight: weight,
        constraints={"type": "ineq", "fun": slope, "jac": lambda weight: basis},
        method="SLSQP",
        options={"ftol": 1e-14, "maxiter": 500},
    )
    candidate = slope(np.asarray(fitted.x, dtype=float))
    if fitted.success and np.all(candidate >= -1e-8):
        return np.maximum(candidate, 0.0)
    warnings.warn(_OFF_SUPPORT, UserWarning, stacklevel=3)
    return np.maximum(center, 0.0)


def _covariance_on_free_slopes(
    covariance: np.ndarray,
    active: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Conditional covariance of slopes that stayed above zero."""
    standard_errors = np.full(covariance.shape[0], np.nan)
    reported = np.zeros_like(covariance)
    if not active.any():
        return standard_errors, reported
    held = ~active
    block = covariance[np.ix_(active, active)]
    if held.any():
        cross = covariance[np.ix_(active, held)]
        held_block = covariance[np.ix_(held, held)]
        block = block - cross @ np.linalg.pinv(held_block, rcond=1e-10) @ cross.T
        block = (block + block.T) / 2
    reported[np.ix_(active, active)] = block
    standard_errors[active] = np.sqrt(np.clip(np.diag(block), 0, None))
    return standard_errors, reported

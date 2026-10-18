"""Identification diagnostics for concurrent treatment grids."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class GridIdentification:
    """What a cell-by-channel dose matrix can identify before outcomes are used."""

    rank: int
    n_cells: int
    n_channels: int
    point_identified: bool
    condition_number: float
    singular_values: tuple[float, ...]
    alias_groups: tuple[tuple[str, ...], ...]
    channel_status: pd.DataFrame
    null_space: pd.DataFrame
    dose_matrix: pd.DataFrame


def _proportional_groups(matrix: np.ndarray, names: list[str]) -> tuple[tuple[str, ...], ...]:
    """Return groups of nonzero columns that carry the same dose pattern."""
    norms = np.linalg.norm(matrix, axis=0)
    grouped: list[list[int]] = []
    used: set[int] = set()
    for left in range(matrix.shape[1]):
        if left in used or norms[left] == 0:
            continue
        group = [left]
        normalized_left = matrix[:, left] / norms[left]
        for right in range(left + 1, matrix.shape[1]):
            if right in used or norms[right] == 0:
                continue
            normalized_right = matrix[:, right] / norms[right]
            if np.allclose(
                normalized_left,
                normalized_right,
                rtol=1e-10,
                atol=1e-12,
            ):
                group.append(right)
                used.add(right)
        if len(group) > 1:
            grouped.append(group)
            used.add(left)
    return tuple(tuple(names[index] for index in group) for group in grouped)


def identify_grid(
    dose: pd.DataFrame,
    *,
    cell: str = "cell",
    channel: str = "channel",
    value: str = "dose",
) -> GridIdentification:
    """Check whether concurrent interventions are separable from their doses.

    The check uses no outcome data. Full column rank means every channel
    coefficient is point-identified by the treatment design. Rank deficiency
    means at least one channel effect is identified only as part of a linear
    combination.
    """
    required = {cell, channel, value}
    missing = required - set(dose.columns)
    if missing:
        raise ValueError(f"Dose data is missing columns: {sorted(missing)}")
    if dose.empty:
        raise ValueError("Dose data is empty")

    work = dose[[cell, channel, value]].copy()
    work[value] = pd.to_numeric(work[value], errors="raise")
    if not np.isfinite(work[value]).all():
        raise ValueError("Dose values must be finite")

    wide = work.pivot_table(
        index=cell,
        columns=channel,
        values=value,
        aggfunc="sum",
        fill_value=0.0,
        observed=False,
    ).sort_index(axis=0).sort_index(axis=1)
    matrix = wide.to_numpy(dtype=float)
    if matrix.shape[1] == 0:
        raise ValueError("Dose data has no channels")

    _, singular_values, right_vectors = np.linalg.svd(matrix, full_matrices=True)
    largest = singular_values[0] if singular_values.size else 0.0
    tolerance = np.finfo(float).eps * max(matrix.shape) * largest
    rank = int(np.sum(singular_values > tolerance))
    n_channels = matrix.shape[1]
    point_identified = rank == n_channels
    condition_number = (
        float(singular_values[0] / singular_values[-1])
        if point_identified and singular_values[-1] > 0
        else float("inf")
    )

    channel_names = [str(name) for name in wide.columns]
    null_values = right_vectors[rank:, :].T
    null_space = pd.DataFrame(
        null_values,
        index=channel_names,
        columns=[f"basis_{index + 1}" for index in range(null_values.shape[1])],
    )
    dependent = (
        np.linalg.norm(null_values, axis=1) > 1e-8
        if null_values.shape[1]
        else np.zeros(n_channels, dtype=bool)
    )
    aliases = _proportional_groups(matrix, channel_names)
    aliased_names = {name for group in aliases for name in group}
    dose_norms = np.linalg.norm(matrix, axis=0)
    statuses = []
    for name, is_dependent, dose_norm in zip(channel_names, dependent, dose_norms):
        if dose_norm == 0:
            statuses.append("unused")
        elif name in aliased_names:
            statuses.append("aliased")
        elif is_dependent:
            statuses.append("linearly dependent")
        else:
            statuses.append("identified")

    channel_status = pd.DataFrame(
        {
            "channel": channel_names,
            "status": statuses,
            "dose_norm": dose_norms,
        }
    )
    return GridIdentification(
        rank=rank,
        n_cells=matrix.shape[0],
        n_channels=n_channels,
        point_identified=point_identified,
        condition_number=condition_number,
        singular_values=tuple(float(value) for value in singular_values),
        alias_groups=aliases,
        channel_status=channel_status,
        null_space=null_space,
        dose_matrix=wide,
    )

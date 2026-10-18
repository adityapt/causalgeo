"""
Concurrent geo test design.

Several cells run in the same weeks. Each cell turns off one bundle of
channels. Markets are drawn at random inside each stratum, so every size
group shows up in the test cells and in the control.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ConcurrentDesign:
    """Markets, the channel bundle each cell turns off, and what can be named."""

    assignment: pd.DataFrame
    schedule: pd.DataFrame
    separation: pd.DataFrame


def design_concurrent_test(
    geos: pd.DataFrame,
    channels: list[str],
    bundles: list[list[str]],
    geo_col: str = "geo",
    stratum_col: str = "stratum",
    control_share: float = 0.5,
    seed: int = 0,
) -> ConcurrentDesign:
    """
    Build one concurrent test.

    ``bundles`` has one list per cell: the channels turned off together in
    that cell. Markets are shuffled inside each stratum. About
    ``control_share`` of each stratum stays on. The rest are dealt evenly
    across cells.
    """
    _check_inputs(geos, channels, bundles, geo_col, stratum_col, control_share)
    assignment = _assign_stratified(
        geos, geo_col, stratum_col, n_cells=len(bundles), control_share=control_share, seed=seed
    )
    schedule = _schedule(channels, bundles)
    separation = _separation(schedule)
    return ConcurrentDesign(assignment=assignment, schedule=schedule, separation=separation)


def _check_inputs(geos, channels, bundles, geo_col, stratum_col, control_share) -> None:
    if geo_col not in geos.columns or stratum_col not in geos.columns:
        raise ValueError(f"geos needs '{geo_col}' and '{stratum_col}' columns")
    if geos.empty:
        raise ValueError("geos is empty")
    if geos[geo_col].duplicated().any():
        raise ValueError(f"'{geo_col}' has duplicate markets")
    if not channels:
        raise ValueError("channels is empty")
    if len(set(channels)) != len(channels):
        raise ValueError("channels has duplicates")
    if not bundles:
        raise ValueError("bundles is empty; pass one channel list per cell")
    if not 0 <= control_share < 1:
        raise ValueError("control_share must be between 0 and 1, not including 1")
    known = set(channels)
    for bundle in bundles:
        unknown = set(bundle) - known
        if unknown:
            raise ValueError(f"bundle has channels that are not in the channel list: {sorted(unknown)}")


def _assign_stratified(geos, geo_col, stratum_col, n_cells, control_share, seed) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    for stratum, group in geos.groupby(stratum_col, sort=True):
        markets = group[geo_col].to_numpy().copy()
        rng.shuffle(markets)
        n_control = int(round(len(markets) * control_share))
        n_control = min(n_control, len(markets))
        for market in markets[:n_control]:
            rows.append(
                {geo_col: market, stratum_col: stratum, "cell": "control", "role": "control"}
            )
        for i, market in enumerate(markets[n_control:]):
            rows.append(
                {
                    geo_col: market,
                    stratum_col: stratum,
                    "cell": str((i % n_cells) + 1),
                    "role": "test",
                }
            )
    out = pd.DataFrame(rows)
    return out.sort_values([stratum_col, "cell", geo_col]).reset_index(drop=True)


def _schedule(channels: list[str], bundles: list[list[str]]) -> pd.DataFrame:
    rows = []
    for cell, bundle in enumerate(bundles, start=1):
        off = set(bundle)
        for channel in channels:
            rows.append({"cell": str(cell), "channel": channel, "off": int(channel in off)})
    return pd.DataFrame(rows)


def _separation(schedule: pd.DataFrame) -> pd.DataFrame:
    """Say which channels this set of cells can name on their own."""
    wide = schedule.pivot(index="cell", columns="channel", values="off").sort_index()
    names = list(wide.columns)
    matrix = wide.to_numpy(dtype=float)
    full_rank = int(np.linalg.matrix_rank(matrix))
    rows = []
    for j, name in enumerate(names):
        column = matrix[:, j]
        if np.allclose(column, 0):
            rows.append({"channel": name, "status": "not tested", "tied_with": ""})
            continue
        partners = [
            other
            for k, other in enumerate(names)
            if k != j and np.array_equal(column, matrix[:, k])
        ]
        without = np.delete(matrix, j, axis=1)
        drops = without.size == 0 or int(np.linalg.matrix_rank(without)) < full_rank
        if partners:
            status = "tied"
        elif drops:
            status = "separate"
        else:
            status = "not separable"
        rows.append({"channel": name, "status": status, "tied_with": ", ".join(partners)})
    return pd.DataFrame(rows)

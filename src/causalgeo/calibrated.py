"""Split bundle effects into channel effects with a calibrated mix model.

The geo test supplies the combined effect of each bundle. A linear mix model
explains sales with channel impressions, market effects, and day effects.
Each bundle result is a calibration equation: the channel effects inside that
bundle have to add up to the measured combined effect. A channel gets its own
number only when its impressions are not a copy of another channel's.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from causalgeo.design import ConcurrentDesign
from causalgeo.exposure import adjust_for_exposure
from causalgeo.identified import IdentifiedEstimate, estimate_identified


@dataclass(frozen=True)
class ChannelModel:
    """Channel effects that add back to the measured bundle effects."""

    channels: pd.DataFrame
    bundles: pd.DataFrame
    daily: pd.DataFrame
    exposure: pd.DataFrame


def disentangle(
    design: ConcurrentDesign,
    outcomes: pd.DataFrame,
    impressions: pd.DataFrame,
    outcome: str,
    treatment_start: str,
    geo_col: str = "geo",
    date_col: str = "date",
    channel_col: str = "channel",
    impression_col: str = "impressions",
    decay: float = 0.0,
) -> ChannelModel:
    """Estimate bundle effects from the assignment, then split them by channel."""
    identified = estimate_identified(
        design,
        outcomes,
        outcome=outcome,
        treatment_start=treatment_start,
        geo_col=geo_col,
        date_col=date_col,
    )
    identified, exposure = adjust_for_exposure(
        design,
        outcomes,
        impressions,
        identified,
        treatment_start=treatment_start,
        outcome=outcome,
        geo_col=geo_col,
        date_col=date_col,
        channel_col=channel_col,
        impression_col=impression_col,
    )
    model = fit_channel_model(
        outcomes,
        impressions,
        identified,
        design,
        treatment_start=treatment_start,
        outcome=outcome,
        geo_col=geo_col,
        date_col=date_col,
        channel_col=channel_col,
        impression_col=impression_col,
        decay=decay,
    )
    return ChannelModel(
        channels=model.channels,
        bundles=model.bundles,
        daily=model.daily,
        exposure=exposure,
    )


def fit_channel_model(
    outcomes: pd.DataFrame,
    impressions: pd.DataFrame,
    identified: IdentifiedEstimate,
    design: ConcurrentDesign,
    treatment_start: str,
    outcome: str,
    geo_col: str = "geo",
    date_col: str = "date",
    channel_col: str = "channel",
    impression_col: str = "impressions",
    decay: float = 0.0,
) -> ChannelModel:
    """Fit channel effects so each bundle's channels add up to its measured effect."""
    if not 0 <= decay < 1:
        raise ValueError("decay must be between 0 and 1, not including 1")
    transformed = _adstock(impressions, geo_col, date_col, channel_col, impression_col, decay)
    channels = _channel_list(transformed, channel_col)
    wide = _wide(transformed, geo_col, date_col, channel_col, impression_col, channels)
    withheld = _withheld(
        wide, design, channels, geo_col, date_col, treatment_start
    )
    obs_x, obs_y = _observational_block(
        outcomes, wide, design, outcome, geo_col, date_col, channels, treatment_start
    )
    beta, se, identified_mask = _fit_weighted(
        obs_x, obs_y, identified, withheld, channels
    )
    channel_table = _channel_table(channels, beta, se, identified_mask)
    bundle_table = _bundle_table(identified, withheld, channels, beta, identified_mask)
    daily = _daily(wide, channels, beta, identified_mask, geo_col, date_col)
    return ChannelModel(
        channels=channel_table,
        bundles=bundle_table,
        daily=daily,
        exposure=pd.DataFrame(),
    )


def _adstock(impressions, geo_col, date_col, channel_col, impression_col, decay: float) -> pd.DataFrame:
    frame = impressions[[geo_col, date_col, channel_col, impression_col]].copy()
    frame[date_col] = pd.to_datetime(frame[date_col])
    frame = frame.sort_values([geo_col, channel_col, date_col])
    if decay == 0:
        frame["exposure"] = frame[impression_col].astype(float)
        return frame
    carried = []
    for _, group in frame.groupby([geo_col, channel_col], sort=False):
        level = 0.0
        values = []
        for raw in group[impression_col].astype(float):
            level = raw + decay * level
            values.append(level)
        carried.append(pd.Series(values, index=group.index))
    frame["exposure"] = pd.concat(carried).sort_index()
    return frame


def _channel_list(impressions: pd.DataFrame, channel_col: str) -> list[str]:
    return sorted(impressions[channel_col].astype(str).unique())


def _wide(impressions, geo_col, date_col, channel_col, impression_col, channels) -> pd.DataFrame:
    del impression_col
    grid = impressions.pivot_table(
        index=[geo_col, date_col],
        columns=channel_col,
        values="exposure",
        aggfunc="sum",
        fill_value=0.0,
    )
    for channel in channels:
        if channel not in grid.columns:
            grid[channel] = 0.0
    return grid[channels].reset_index()


def _withheld(wide, design, channels, geo_col, date_col, treatment_start) -> pd.DataFrame:
    assignment = design.assignment.drop_duplicates(geo_col).set_index(geo_col)["cell"]
    frame = wide.copy()
    frame["cell"] = frame[geo_col].map(assignment)
    post = frame[date_col] >= pd.Timestamp(treatment_start)
    control = frame.loc[post & (frame["cell"] == "control"), channels]
    if control.empty:
        raise ValueError("the test window has no control markets")
    control_mean = control.mean()
    rows = []
    bundles = _bundles(design)
    for cell, label in bundles.items():
        dark = frame.loc[post & (frame["cell"] == cell), channels]
        if dark.empty:
            continue
        gap = control_mean - dark.mean()
        for channel in channels:
            rows.append(
                {
                    "cell": cell,
                    "bundle": label,
                    "channel": channel,
                    "withheld": float(max(gap[channel], 0.0)),
                }
            )
    if not rows:
        raise ValueError("no test cell has markets in the impression data")
    return pd.DataFrame(rows)


def _bundles(design: ConcurrentDesign) -> dict[str, str]:
    labels = {}
    for cell, group in design.schedule.groupby("cell", sort=True):
        off = group.loc[group["off"] == 1, "channel"].tolist()
        labels[str(cell)] = "+".join(off)
    return labels


def _observational_block(outcomes, wide, design, outcome, geo_col, date_col, channels, treatment_start):
    assignment = design.assignment.drop_duplicates(geo_col).set_index(geo_col)["cell"]
    frame = outcomes[[geo_col, date_col, outcome]].copy()
    frame[date_col] = pd.to_datetime(frame[date_col])
    frame = frame.merge(wide, on=[geo_col, date_col], how="inner")
    frame["cell"] = frame[geo_col].map(assignment)
    post = frame[date_col] >= pd.Timestamp(treatment_start)
    # Dark market-days belong to the calibration equations, not to this regression.
    observational = frame.loc[~post | (frame["cell"] == "control")].copy()
    if observational.empty:
        return np.zeros((0, len(channels))), np.zeros(0)
    demeaned = _demean(observational, geo_col, date_col, [outcome, *channels])
    return demeaned[channels].to_numpy(dtype=float), demeaned[outcome].to_numpy(dtype=float)


def _demean(frame, geo_col, date_col, columns):
    out = frame[columns].astype(float).copy()
    for _ in range(100):
        previous = out.to_numpy().copy()
        out = out - out.groupby(frame[geo_col]).transform("mean")
        out = out - out.groupby(frame[date_col]).transform("mean")
        if np.nanmax(np.abs(out.to_numpy() - previous)) < 1e-10:
            break
    return out


def _fit_weighted(obs_x, obs_y, identified, withheld, channels):
    sigma2 = _observational_variance(obs_x, obs_y)
    x_rows = []
    y_rows = []
    w_rows = []
    if len(obs_y):
        x_rows.append(obs_x)
        y_rows.append(obs_y)
        w_rows.append(np.full(len(obs_y), 1.0 / sigma2))
    effects = identified.cells.set_index("cell")
    for cell, group in withheld.groupby("cell", sort=True):
        if cell not in effects.index:
            continue
        effect = float(effects.loc[cell, "coef"])
        se = float(effects.loc[cell, "std_err"])
        se = max(se, 1e-8)
        row = np.array([float(group.set_index("channel").loc[channel, "withheld"]) for channel in channels])
        # Being dark removes these impressions, so the measured change is minus their contribution.
        x_rows.append(row.reshape(1, -1))
        y_rows.append(np.array([-effect]))
        w_rows.append(np.array([1.0 / se**2]))
    if not x_rows:
        raise ValueError("nothing to fit: no observational rows and no bundle effects")
    x = np.vstack(x_rows)
    y = np.concatenate(y_rows)
    w = np.concatenate(w_rows)
    keep = _unique_channels(x * np.sqrt(w)[:, None])
    beta = np.full(len(channels), np.nan)
    se_hat = np.full(len(channels), np.nan)
    if not keep.any():
        return beta, se_hat, keep
    x_keep = x[:, keep]
    sw = np.sqrt(w)
    xtwx = (x_keep * sw[:, None]).T @ (x_keep * sw[:, None])
    xtwy = (x_keep * sw[:, None]).T @ (y * sw)
    fitted = np.linalg.lstsq(xtwx, xtwy, rcond=1e-8)[0]
    cov = np.linalg.pinv(xtwx, rcond=1e-8)
    beta[keep] = fitted
    se_hat[keep] = np.sqrt(np.clip(np.diag(cov), 0, None))
    return beta, se_hat, keep


def _observational_variance(obs_x, obs_y) -> float:
    if len(obs_y) <= obs_x.shape[1] + 1:
        return 1.0
    beta, _, _, _ = np.linalg.lstsq(obs_x, obs_y, rcond=None)
    resid = obs_y - obs_x @ beta
    dof = max(len(obs_y) - obs_x.shape[1], 1)
    return float(max(resid @ resid / dof, 1e-8))


def _unique_channels(weighted_x: np.ndarray) -> np.ndarray:
    """Keep a channel only when its column is not a copy of another channel."""
    count = weighted_x.shape[1]
    keep = np.zeros(count, dtype=bool)
    if weighted_x.size == 0 or np.allclose(weighted_x, 0):
        return keep
    norms = np.linalg.norm(weighted_x, axis=0)
    scale = max(float(norms.max()), 1e-12)
    alive = norms > 1e-8 * scale
    copies = np.zeros(count, dtype=bool)
    indexes = np.flatnonzero(alive)
    for left, first in enumerate(indexes):
        for second in indexes[left + 1 :]:
            cosine = float(np.dot(weighted_x[:, first], weighted_x[:, second]) / (norms[first] * norms[second]))
            if abs(cosine) > 1 - 1e-6:
                copies[first] = True
                copies[second] = True
    chosen: list[int] = []
    for index in indexes:
        if copies[index]:
            continue
        trial = weighted_x[:, [*chosen, index]]
        if np.linalg.matrix_rank(trial, tol=1e-6) > len(chosen):
            chosen.append(int(index))
    keep[chosen] = True
    return keep


def _channel_table(channels, beta, se, keep) -> pd.DataFrame:
    partners = _tied_partners(channels, keep)
    rows = []
    for index, channel in enumerate(channels):
        if keep[index]:
            reason = ""
            status = "separate"
            coef = float(beta[index])
            std_err = float(se[index])
        else:
            status = "tied" if partners[channel] else "not separable"
            others = ", ".join(partners[channel]) or "the other channels"
            reason = f"{channel} moves with {others}; only the bundle total is used"
            coef = float("nan")
            std_err = float("nan")
        rows.append(
            {
                "channel": channel,
                "status": status,
                "coef": coef,
                "std_err": std_err,
                "reason": reason,
            }
        )
    return pd.DataFrame(rows)


def _tied_partners(channels, keep) -> dict[str, list[str]]:
    unidentified = [channel for channel, flagged in zip(channels, keep) if not flagged]
    return {
        channel: [other for other in unidentified if other != channel]
        for channel in channels
    }


def _bundle_table(identified, withheld, channels, beta, keep) -> pd.DataFrame:
    rows = []
    for record in identified.cells.itertuples(index=False):
        group = withheld.loc[withheld["cell"] == record.cell].set_index("channel")
        parts = record.bundle.split("+") if record.bundle else []
        usable = [channel for channel in parts if channel in group.index and keep[channels.index(channel)]]
        if usable and len(usable) == len(parts):
            predicted = -sum(float(beta[channels.index(channel)]) * float(group.loc[channel, "withheld"]) for channel in parts)
        else:
            predicted = float("nan")
        rows.append(
            {
                "cell": record.cell,
                "bundle": record.bundle,
                "measured_effect": float(record.coef),
                "predicted_effect": predicted,
            }
        )
    return pd.DataFrame(rows)


def _daily(wide, channels, beta, keep, geo_col, date_col) -> pd.DataFrame:
    rows = []
    for index, channel in enumerate(channels):
        if not keep[index]:
            continue
        piece = wide[[geo_col, date_col, channel]].copy()
        piece["channel"] = channel
        piece["impact"] = piece[channel] * float(beta[index])
        piece = piece.rename(columns={channel: "exposure"})
        rows.append(piece[[geo_col, date_col, "channel", "exposure", "impact"]])
    if not rows:
        return pd.DataFrame(columns=[geo_col, date_col, "channel", "exposure", "impact"])
    return pd.concat(rows, ignore_index=True)

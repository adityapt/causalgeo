"""Split bundle effects into channel effects with a calibrated mix model.

The geo test supplies the combined effect of each bundle. A linear mix model
explains sales with channel impressions, market effects, and day effects.
Each bundle result is one weighted calibration row. The weight pulls the
channel slopes toward the measured total. It does not force them to match it.
A channel gets its own number only when its impressions are not a copy of
another channel's. Copied channels stay in the fit as one shared control, and
neither of them is given a slope.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from causalgeo.design import ConcurrentDesign
from causalgeo.exposure import adjust_for_exposure, warn_if_impressions_missing, _sum_or_missing
from causalgeo.identified import IdentifiedEstimate, estimate_identified


@dataclass(frozen=True)
class ChannelModel:
    """Channel slopes from a soft calibration to the measured bundle effects."""

    channels: pd.DataFrame
    bundles: pd.DataFrame
    daily: pd.DataFrame
    exposure: pd.DataFrame
    cells: pd.DataFrame


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
    adjust_exposure: bool = False,
    check_stability: bool = True,
    material: float = 0.10,
    copy_tolerance: float = 1e-6,
) -> ChannelModel:
    """Estimate bundle effects from the assignment, then split them by channel.

    The reported cell estimate is the unadjusted experiment unless
    ``adjust_exposure`` is true. When a still-on channel's impressions moved
    enough to flag, the channel split uses the cell total after that shift is
    removed. ``check_stability`` refits with a weaker calibration weight and
    without one influential market.
    """
    identified = estimate_identified(
        design,
        outcomes,
        outcome=outcome,
        treatment_start=treatment_start,
        geo_col=geo_col,
        date_col=date_col,
    )
    raw, exposure = adjust_for_exposure(
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
        material=material,
        apply=False,
    )
    cleaned, _ = adjust_for_exposure(
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
        material=material,
        apply=True,
    )
    warn_if_impressions_missing(impressions, impression_col)
    flagged = bool(len(exposure) and bool(exposure["adjust"].any()))
    target = cleaned if flagged else raw
    exposure["removed_from_split"] = exposure["adjust"] if flagged else False
    if adjust_exposure:
        exposure.loc[exposure["adjust"], "applied"] = True
        identified = cleaned
    else:
        identified = raw
    model = fit_channel_model(
        outcomes,
        impressions,
        target,
        design,
        treatment_start=treatment_start,
        outcome=outcome,
        geo_col=geo_col,
        date_col=date_col,
        channel_col=channel_col,
        impression_col=impression_col,
        decay=decay,
        warn_missing=False,
        copy_tolerance=copy_tolerance,
    )
    experimental = raw.cells.set_index("cell")["coef"]
    model = ChannelModel(
        channels=model.channels,
        bundles=model.bundles.assign(
            experimental_effect=model.bundles["cell"].astype(str).map(experimental)
        ),
        daily=model.daily,
        exposure=model.exposure,
        cells=model.cells,
    )
    if check_stability:
        model = ChannelModel(
            channels=_mark_unstable(
                model.channels,
                outcomes,
                impressions,
                identified,
                design,
                treatment_start,
                outcome,
                geo_col,
                date_col,
                channel_col,
                impression_col,
                decay,
                copy_tolerance,
            ),
            bundles=model.bundles,
            daily=model.daily,
            exposure=pd.DataFrame(),
            cells=model.cells,
        )
    return ChannelModel(
        channels=model.channels,
        bundles=model.bundles,
        daily=model.daily,
        exposure=exposure,
        cells=identified.cells,
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
    calibrate: bool = True,
    calibration_scale: float = 1.0,
    warn_missing: bool = True,
    copy_tolerance: float = 1e-6,
) -> ChannelModel:
    """Fit channel effects softly calibrated to each measured bundle effect."""
    if not 0 <= decay < 1:
        raise ValueError("decay must be between 0 and 1, not including 1")
    if not 0 < copy_tolerance < 1:
        raise ValueError("copy_tolerance must be between 0 and 1")
    if warn_missing:
        warn_if_impressions_missing(impressions, impression_col)
    transformed = _adstock(impressions, geo_col, date_col, channel_col, impression_col, decay)
    channels = _channel_list(transformed, channel_col)
    wide = _wide(transformed, geo_col, date_col, channel_col, impression_col, channels)
    withheld = _withheld(
        wide, design, channels, geo_col, date_col, treatment_start
    )
    obs_x, obs_y = _observational_block(
        outcomes, wide, design, outcome, geo_col, date_col, channels, treatment_start
    )
    beta, se, identified_mask, tied = _fit_weighted(
        obs_x,
        obs_y,
        identified,
        withheld,
        channels,
        calibrate=calibrate,
        calibration_scale=calibration_scale,
        copy_tolerance=copy_tolerance,
    )
    untested = _never_turned_off(design)
    for index, channel in enumerate(channels):
        if channel in untested:
            identified_mask[index] = False
    channel_table = _channel_table(channels, beta, se, identified_mask, tied, untested)
    bundle_table = _bundle_table(identified, withheld, channels, beta, identified_mask)
    daily = _daily(wide, channels, beta, identified_mask, geo_col, date_col)
    return ChannelModel(
        channels=channel_table,
        bundles=bundle_table,
        daily=daily,
        exposure=pd.DataFrame(),
        cells=identified.cells,
    )


def experiment_only_split(
    impressions: pd.DataFrame,
    identified: IdentifiedEstimate,
    design: ConcurrentDesign,
    treatment_start: str,
    geo_col: str = "geo",
    date_col: str = "date",
    channel_col: str = "channel",
    impression_col: str = "impressions",
    decay: float = 0.0,
) -> pd.DataFrame:
    """Solve channel slopes from the cell totals and the withheld impressions.

    Observational market-weeks do not enter. A channel the design never
    turns off has no column in the system and stays blank.
    """
    transformed = _adstock(impressions, geo_col, date_col, channel_col, impression_col, decay)
    channels = _channel_list(transformed, channel_col)
    wide = _wide(transformed, geo_col, date_col, channel_col, impression_col, channels)
    withheld = _withheld(wide, design, channels, geo_col, date_col, treatment_start)
    effects = identified.cells.set_index("cell")
    present = set(withheld["cell"].astype(str))
    width = []
    target = []
    weight = []
    for cell in effects.index:
        if str(cell) not in present:
            continue
        group = withheld.loc[withheld["cell"].astype(str) == str(cell)].set_index("channel")
        width.append(
            [float(group.loc[channel, "withheld"]) if channel in group.index else 0.0 for channel in channels]
        )
        target.append(-float(effects.loc[cell, "coef"]))
        weight.append(1.0 / max(float(effects.loc[cell, "std_err"]), 1e-8) ** 2)
    beta = np.full(len(channels), np.nan)
    se_hat = np.full(len(channels), np.nan)
    if width:
        matrix = np.asarray(width, dtype=float)
        alive = [index for index in range(len(channels)) if np.max(np.abs(matrix[:, index])) > 0]
        if alive:
            used = matrix[:, alive]
            scale = np.sqrt(np.asarray(weight, dtype=float))
            xtwx = (used * scale[:, None]).T @ (used * scale[:, None])
            xtwy = (used * scale[:, None]).T @ (np.asarray(target, dtype=float) * scale)
            fitted = np.linalg.lstsq(xtwx, xtwy, rcond=1e-8)[0]
            covariance = np.linalg.pinv(xtwx, rcond=1e-8)
            for cursor, index in enumerate(alive):
                beta[index] = fitted[cursor]
                se_hat[index] = float(np.sqrt(max(covariance[cursor, cursor], 0.0)))
    return pd.DataFrame({"channel": channels, "coef": beta, "std_err": se_hat})


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
        aggfunc=_sum_or_missing,
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
    control_post = frame.loc[post & (frame["cell"] == "control"), channels]
    control_pre = frame.loc[~post & (frame["cell"] == "control"), channels]
    if control_post.empty:
        raise ValueError("the test window has no control markets")
    if control_pre.empty:
        raise ValueError("the pre-test window has no control markets")
    control_change = control_post.mean() - control_pre.mean()
    rows = []
    bundles = _bundles(design)
    for cell, label in bundles.items():
        dark_post = frame.loc[post & (frame["cell"] == cell), channels]
        dark_pre = frame.loc[~post & (frame["cell"] == cell), channels]
        if dark_post.empty:
            continue
        if dark_pre.empty:
            raise ValueError(f"the pre-test window has no markets for cell {cell}")
        lost = control_change - (dark_post.mean() - dark_pre.mean())
        off_channels = set(label.split("+")) if label else set()
        for channel in channels:
            rows.append(
                {
                    "cell": cell,
                    "bundle": label,
                    "channel": channel,
                    "withheld": (
                        float(max(lost[channel], 0.0))
                        if channel in off_channels
                        else 0.0
                    ),
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
    observational = observational.dropna(subset=[outcome, *channels])
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


def _fit_weighted(
    obs_x,
    obs_y,
    identified,
    withheld,
    channels,
    calibrate: bool = True,
    calibration_scale: float = 1.0,
    copy_tolerance: float = 1e-6,
):
    sigma2 = _observational_variance(obs_x, obs_y)
    x_rows = []
    y_rows = []
    w_rows = []
    if len(obs_y):
        x_rows.append(obs_x)
        y_rows.append(obs_y)
        w_rows.append(np.full(len(obs_y), 1.0 / sigma2))
    effects = identified.cells.set_index("cell")
    if calibrate:
        for cell, group in withheld.groupby("cell", sort=True):
            if cell not in effects.index:
                continue
            effect = float(effects.loc[cell, "coef"])
            se = max(float(effects.loc[cell, "std_err"]), 1e-8)
            row = np.array(
                [float(group.set_index("channel").loc[channel, "withheld"]) for channel in channels]
            )
            # Being dark removes these impressions, so the measured change is minus their contribution.
            x_rows.append(row.reshape(1, -1))
            y_rows.append(np.array([-effect]))
            w_rows.append(np.array([calibration_scale / se**2]))
    if not x_rows:
        raise ValueError("nothing to fit: no observational rows and no bundle effects")
    x = np.vstack(x_rows)
    y = np.concatenate(y_rows)
    w = np.concatenate(w_rows)
    report, used, tied = _select_columns(
        x * np.sqrt(w)[:, None],
        channels,
        copy_x=obs_x,
        copy_tolerance=copy_tolerance,
    )
    beta = np.full(len(channels), np.nan)
    se_hat = np.full(len(channels), np.nan)
    if not used.any():
        return beta, se_hat, report, tied
    x_used = x[:, used]
    sw = np.sqrt(w)
    xtwx = (x_used * sw[:, None]).T @ (x_used * sw[:, None])
    xtwy = (x_used * sw[:, None]).T @ (y * sw)
    fitted = np.linalg.lstsq(xtwx, xtwy, rcond=1e-8)[0]
    cov = np.linalg.pinv(xtwx, rcond=1e-8)
    se_used = np.sqrt(np.clip(np.diag(cov), 0, None))
    cursor = 0
    for index in np.flatnonzero(used):
        if report[index]:
            beta[index] = fitted[cursor]
            se_hat[index] = se_used[cursor]
        cursor += 1
    return beta, se_hat, report, tied


def _observational_variance(obs_x, obs_y) -> float:
    if len(obs_y) <= obs_x.shape[1] + 1:
        return 1.0
    beta, _, _, _ = np.linalg.lstsq(obs_x, obs_y, rcond=None)
    resid = obs_y - obs_x @ beta
    dof = max(len(obs_y) - obs_x.shape[1], 1)
    return float(max(resid @ resid / dof, 1e-8))


def _select_columns(
    weighted_x: np.ndarray,
    channels: list[str],
    copy_x: np.ndarray | None = None,
    copy_tolerance: float = 1e-6,
):
    """Choose reported slopes, and one control column for each copied pair.

    A copied pair cannot be split. Dropping both columns lets their sales leak
    into every channel that remains. One column of the pair stays in the
    regression as a shared control. Neither member is given its own slope.
    """
    count = weighted_x.shape[1]
    report = np.zeros(count, dtype=bool)
    used = np.zeros(count, dtype=bool)
    tied: dict[str, list[str]] = {}
    if weighted_x.size == 0 or np.allclose(weighted_x, 0):
        return report, used, tied
    norms = np.linalg.norm(weighted_x, axis=0)
    scale = max(float(norms.max()), 1e-12)
    alive = list(np.flatnonzero(norms > 1e-8 * scale))
    parent = {index: index for index in alive}
    copy_x = weighted_x if copy_x is None else copy_x
    copy_norms = np.linalg.norm(copy_x, axis=0)
    copy_scale = max(float(copy_norms.max()), 1e-12)
    copy_alive = set(np.flatnonzero(copy_norms > 1e-8 * copy_scale))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for left, first in enumerate(alive):
        for second in alive[left + 1 :]:
            if first in copy_alive and second in copy_alive:
                cosine = float(
                    np.dot(copy_x[:, first], copy_x[:, second])
                    / (copy_norms[first] * copy_norms[second])
                )
            elif first not in copy_alive and second not in copy_alive:
                cosine = float(
                    np.dot(weighted_x[:, first], weighted_x[:, second])
                    / (norms[first] * norms[second])
                )
            else:
                continue
            if abs(cosine) > 1 - copy_tolerance:
                parent[find(second)] = find(first)
    groups: dict[int, list[int]] = {}
    for index in alive:
        groups.setdefault(find(index), []).append(index)
    nuisance: list[int] = []
    for members in groups.values():
        if len(members) < 2:
            continue
        nuisance.append(members[0])
        names = [channels[index] for index in members]
        for index in members:
            tied[channels[index]] = [name for name in names if name != channels[index]]
    chosen = list(nuisance)
    for index in alive:
        if channels[index] in tied:
            continue
        trial = weighted_x[:, [*chosen, index]]
        if np.linalg.matrix_rank(trial, tol=1e-6) > len(chosen):
            chosen.append(index)
            report[index] = True
    used[chosen] = True
    return report, used, tied


def _never_turned_off(design: ConcurrentDesign) -> set[str]:
    separation = design.separation
    return set(separation.loc[separation["status"] == "not tested", "channel"].astype(str))


def _channel_table(channels, beta, se, report, tied, untested) -> pd.DataFrame:
    rows = []
    for index, channel in enumerate(channels):
        if channel in untested:
            status = "not tested"
            reason = f"{channel} was never turned off"
            coef = float("nan")
            std_err = float("nan")
        elif report[index]:
            status = "separate"
            reason = ""
            coef = float(beta[index])
            std_err = float(se[index])
        elif channel in tied:
            status = "tied"
            others = ", ".join(tied[channel])
            reason = (
                f"{channel} moves with {others}; their shared path is controlled for "
                "and is not split into a channel slope"
            )
            coef = float("nan")
            std_err = float("nan")
        else:
            status = "not separable"
            reason = f"{channel} is not separable from the other channels in this fit"
            coef = float("nan")
            std_err = float("nan")
        rows.append(
            {
                "channel": channel,
                "status": status,
                "coef": coef,
                "std_err": std_err,
                "reason": reason,
                "reading": _reading(status),
            }
        )
    return pd.DataFrame(rows)


def _reading(status: str) -> str:
    if status == "separate":
        return "model-assisted estimate"
    if status == "unstable":
        return "model-assisted estimate; unstable across sensitivity checks"
    if status == "not tested":
        return "withheld; this channel was never turned off"
    if status == "tied":
        return "withheld; the shared path is controlled for"
    return "withheld; the design cannot name this channel"


def _bundle_table(identified, withheld, channels, beta, keep) -> pd.DataFrame:
    rows = []
    for record in identified.cells.itertuples(index=False):
        group = withheld.loc[withheld["cell"] == record.cell].set_index("channel")
        parts = record.bundle.split("+") if record.bundle else []
        usable = [channel for channel in parts if channel in group.index and keep[channels.index(channel)]]
        measured = float(record.coef)
        if usable and len(usable) == len(parts):
            predicted = -sum(float(beta[channels.index(channel)]) * float(group.loc[channel, "withheld"]) for channel in parts)
            gap = predicted - measured
        else:
            predicted = float("nan")
            gap = float("nan")
        rows.append(
            {
                "cell": record.cell,
                "bundle": record.bundle,
                "measured_effect": measured,
                "predicted_effect": predicted,
                "gap": gap,
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


def compare_calibration(
    outcomes: pd.DataFrame,
    impressions: pd.DataFrame,
    identified: IdentifiedEstimate,
    design: ConcurrentDesign,
    treatment_start: str,
    outcome: str,
    **kwargs,
) -> pd.DataFrame:
    """Compare the observational channel fit with the calibrated fit."""
    observational = fit_channel_model(
        outcomes, impressions, identified, design, treatment_start, outcome, calibrate=False, **kwargs
    )
    calibrated = fit_channel_model(
        outcomes, impressions, identified, design, treatment_start, outcome, calibrate=True, **kwargs
    )
    left = observational.channels[["channel", "status", "coef", "std_err"]].rename(
        columns={"status": "observational_status", "coef": "observational_coef", "std_err": "observational_se"}
    )
    right = calibrated.channels[["channel", "status", "coef", "std_err", "reading"]].rename(
        columns={"status": "calibrated_status", "coef": "calibrated_coef", "std_err": "calibrated_se"}
    )
    compared = left.merge(right, on="channel")
    observational_gap = observational.bundles["gap"].abs().median()
    calibrated_gap = calibrated.bundles["gap"].abs().median()
    compared["observational_bundle_gap"] = float(observational_gap)
    compared["calibrated_bundle_gap"] = float(calibrated_gap)
    return compared


def _mark_unstable(
    channels,
    outcomes,
    impressions,
    identified,
    design,
    treatment_start,
    outcome,
    geo_col,
    date_col,
    channel_col,
    impression_col,
    decay,
    copy_tolerance: float = 1e-6,
) -> pd.DataFrame:
    """Refit under a weaker calibration weight and without one market."""
    variants = [
        fit_channel_model(
            outcomes,
            impressions,
            identified,
            design,
            treatment_start,
            outcome,
            geo_col=geo_col,
            date_col=date_col,
            channel_col=channel_col,
            impression_col=impression_col,
            decay=decay,
            calibration_scale=0.25,
            warn_missing=False,
            copy_tolerance=copy_tolerance,
        )
    ]
    geo = _influential_geo(outcomes, design, treatment_start, outcome, geo_col, date_col)
    if geo is not None:
        variants.append(
            fit_channel_model(
                outcomes.loc[outcomes[geo_col] != geo],
                impressions.loc[impressions[geo_col] != geo],
                identified,
                design,
                treatment_start,
                outcome,
                geo_col=geo_col,
                date_col=date_col,
                channel_col=channel_col,
                impression_col=impression_col,
                decay=decay,
                warn_missing=False,
                copy_tolerance=copy_tolerance,
            )
        )
    marked = channels.copy()
    base = marked.set_index("channel")
    for channel in base.index:
        if base.loc[channel, "status"] != "separate":
            continue
        coef = float(base.loc[channel, "coef"])
        se = float(base.loc[channel, "std_err"])
        for variant in variants:
            other = variant.channels.set_index("channel")
            moved = (
                other.loc[channel, "status"] != "separate"
                or not np.isfinite(other.loc[channel, "coef"])
                or abs(float(other.loc[channel, "coef"]) - coef) > max(0.2 * abs(coef), 2.0 * se)
            )
            if moved:
                marked.loc[marked["channel"] == channel, "status"] = "unstable"
                marked.loc[marked["channel"] == channel, "reading"] = _reading("unstable")
                marked.loc[marked["channel"] == channel, "reason"] = (
                    "the slope moves when the calibration weight changes or one market is left out"
                )
                break
    return marked


def _influential_geo(outcomes, design, treatment_start, outcome, geo_col, date_col):
    assignment = design.assignment.drop_duplicates(geo_col).set_index(geo_col)["cell"].astype(str)
    frame = outcomes[[geo_col, date_col, outcome]].copy()
    frame[date_col] = pd.to_datetime(frame[date_col])
    frame["cell"] = frame[geo_col].map(assignment)
    post = frame[date_col] >= pd.Timestamp(treatment_start)
    test = frame.loc[post & (frame["cell"] != "control")]
    if test.empty:
        return None
    score = test.groupby(geo_col)[outcome].mean().abs()
    return score.idxmax()

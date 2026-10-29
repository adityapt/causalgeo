"""Allocate a measured bundle effect without calling the split causal.

The geo experiment identifies one bundle total. A prevented-dose score can
divide that total among channels, but only after the user states how much
outcome each unit of effective exposure is worth. The pieces add back to the
experimental total under every potency and quality choice.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

from causalgeo.design import ConcurrentDesign


READING = (
    "Causally anchored allocation; the bundle total is experimental "
    "and the channel split is assumption-driven."
)


@dataclass(frozen=True)
class AnchoredAllocation:
    """Channel allocations that reconcile to each measured bundle effect."""

    channels: pd.DataFrame


def anchored_channel_allocation(
    bundle_effects: pd.DataFrame,
    dose: pd.DataFrame,
    potency: Mapping[str, float] | None = None,
    potency_bounds: Mapping[str, tuple[float, float]] | None = None,
) -> AnchoredAllocation:
    """Divide each bundle effect in proportion to potency times prevented dose.

    ``bundle_effects`` has ``cell``, ``coef``, and ``std_err``. ``dose`` has
    ``cell``, ``channel``, and nonnegative ``dose``. A missing potency is 1,
    which states that one effective-dose unit has the same response in every
    channel. ``potency_bounds`` maps a channel to an inclusive interval and
    is reported separately from the experimental interval.
    """
    effects = _effects(bundle_effects)
    doses = _doses(dose)
    names = sorted(set(doses["channel"]))
    point = _point_potency(names, potency)
    bounds = _bounds(names, point, potency_bounds)

    rows: list[dict[str, float | str]] = []
    for cell, effect in effects.iterrows():
        cell_dose = doses.loc[doses["cell"].eq(cell)].set_index("channel")["dose"]
        if cell_dose.empty:
            raise ValueError(f"cell {cell} has no prevented-dose rows")
        scores = pd.Series(
            {channel: point[channel] * float(cell_dose.get(channel, 0.0)) for channel in names},
            dtype=float,
        )
        scores = scores.loc[scores.index.isin(cell_dose.index)]
        total = float(scores.sum())
        if total <= 0:
            raise ValueError(f"cell {cell} needs a positive prevented-dose score")
        tau = float(effect["coef"])
        standard_error = float(effect["std_err"])
        for channel, score in scores.items():
            share = float(score / total)
            allocation = tau * share
            sampling = abs(share) * standard_error
            share_lower, share_upper = _share_bounds(
                str(channel),
                scores.index.astype(str).tolist(),
                cell_dose,
                bounds,
            )
            assumption_lower, assumption_upper = _signed_range(
                tau, share_lower, share_upper
            )
            tau_lower = tau - 1.96 * standard_error
            tau_upper = tau + 1.96 * standard_error
            corners = [
                edge * weight
                for edge in (tau_lower, tau_upper)
                for weight in (share_lower, share_upper)
            ]
            rows.append(
                {
                    "cell": cell,
                    "channel": str(channel),
                    "dose": float(cell_dose.get(channel, 0.0)),
                    "potency": point[str(channel)],
                    "score": float(score),
                    "share": share,
                    "allocation": allocation,
                    "allocation_std_err": sampling,
                    "ci_lower": allocation - 1.96 * sampling,
                    "ci_upper": allocation + 1.96 * sampling,
                    "share_lower": share_lower,
                    "share_upper": share_upper,
                    "assumption_lower": assumption_lower,
                    "assumption_upper": assumption_upper,
                    "envelope_lower": min(corners),
                    "envelope_upper": max(corners),
                    "reading": READING,
                }
            )
    return AnchoredAllocation(channels=pd.DataFrame(rows))


def potency_misspecification_ratio(
    dose: Mapping[str, float],
    assumed_potency: Mapping[str, float],
    true_potency: Mapping[str, float],
) -> pd.Series:
    """Return allocated value divided by the allocation at the true potency.

    For positive doses and potencies,
    ``a_k / theta_k = (assumed_k / true_k) * sum(true * dose) / sum(assumed * dose)``.
    """
    channels = sorted(set(dose) | set(assumed_potency) | set(true_potency))
    doses = _positive_mapping("dose", dose, channels)
    assumed = _positive_mapping("assumed potency", assumed_potency, channels)
    truth = _positive_mapping("true potency", true_potency, channels)
    assumed_total = sum(assumed[channel] * doses[channel] for channel in channels)
    true_total = sum(truth[channel] * doses[channel] for channel in channels)
    if assumed_total <= 0 or true_total <= 0:
        raise ValueError("dose and potency must produce a positive score")
    return pd.Series(
        {
            channel: (assumed[channel] / truth[channel]) * (true_total / assumed_total)
            for channel in channels
        },
        dtype=float,
    )


def potency_tie_ratio(dose_a: float, dose_b: float) -> float:
    """Return the potency ratio that gives these two doses the same allocation.

    The result is ``potency_a / potency_b``.
    """
    if dose_a <= 0 or dose_b <= 0:
        raise ValueError("both doses must be positive")
    return float(dose_b) / float(dose_a)


def prevented_effective_dose(
    impressions: pd.DataFrame,
    design: ConcurrentDesign,
    treatment_start: str,
    geo_col: str = "geo",
    date_col: str = "date",
    channel_col: str = "channel",
    impression_col: str = "impressions",
    quality_col: str | None = None,
    reach_col: str | None = None,
    frequency_col: str | None = None,
    frequency_decay: Mapping[str, float] | float | None = None,
    decay: float = 0.0,
    hill_slope: Mapping[str, float] | None = None,
    hill_half: Mapping[str, float] | None = None,
) -> pd.DataFrame:
    """Return the difference-in-differences of effective exposure by cell.

    Reach and frequency, when both are supplied, replace impressions. Quality
    multiplies that base, geometric adstock carries it forward, and a Hill
    curve is applied to the carried level. The prevented dose is the control
    market's pre/post change minus the holdout market's change.
    """
    if not 0 <= decay < 1:
        raise ValueError("decay must be between 0 and 1, not including 1")
    frame = _exposure_frame(
        impressions,
        geo_col,
        date_col,
        channel_col,
        impression_col,
        quality_col,
        reach_col,
        frequency_col,
    )
    channels = sorted(set(design.schedule["channel"].astype(str)))
    decay_by_channel = _frequency_decay(channels, frequency_decay, reach_col is not None)
    slopes, halves = _hill_parameters(channels, hill_slope, hill_half)
    frame["base"] = _base_exposure(
        frame,
        channel_col,
        impression_col,
        quality_col,
        reach_col,
        frequency_col,
        decay_by_channel,
    )
    frame = _adstock(frame, geo_col, date_col, channel_col, decay)
    frame["effective"] = [
        _hill(value, slopes.get(channel), halves.get(channel))
        for channel, value in zip(frame[channel_col], frame["carried"], strict=True)
    ]
    wide = frame.pivot_table(
        index=[geo_col, date_col],
        columns=channel_col,
        values="effective",
        aggfunc="sum",
    )
    for channel in channels:
        if channel not in wide.columns:
            wide[channel] = 0.0
    wide = wide[channels].reset_index()
    basis = _basis(reach_col is not None, quality_col is not None, bool(slopes))
    return _prevented(wide, design, channels, geo_col, date_col, treatment_start, basis)


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
    return effects.set_index("cell")


def _doses(dose: pd.DataFrame) -> pd.DataFrame:
    required = {"cell", "channel", "dose"}
    missing = required - set(dose.columns)
    if missing:
        raise ValueError(f"dose is missing {sorted(missing)}")
    frame = dose[["cell", "channel", "dose"]].copy()
    frame["cell"] = frame["cell"].astype(str)
    frame["channel"] = frame["channel"].astype(str)
    frame["dose"] = frame["dose"].astype(float)
    if not np.isfinite(frame["dose"]).all():
        raise ValueError("dose must be finite")
    if (frame["dose"] < 0).any():
        raise ValueError("dose must be nonnegative")
    if frame.duplicated(["cell", "channel"]).any():
        raise ValueError("dose must have one row per cell and channel")
    return frame


def _point_potency(
    channels: list[str],
    potency: Mapping[str, float] | None,
) -> dict[str, float]:
    values = {channel: 1.0 for channel in channels}
    if potency is None:
        return values
    unknown = set(map(str, potency)) - set(channels)
    if unknown:
        raise ValueError(f"potency has channels without a dose: {sorted(unknown)}")
    for channel, value in potency.items():
        number = float(value)
        if not np.isfinite(number) or number < 0:
            raise ValueError("potency must be positive")
        values[str(channel)] = number
    return values


def _bounds(
    channels: list[str],
    point: Mapping[str, float],
    potency_bounds: Mapping[str, tuple[float, float]] | None,
) -> dict[str, tuple[float, float]]:
    bounds = {channel: (point[channel], point[channel]) for channel in channels}
    if potency_bounds is None:
        return bounds
    unknown = set(map(str, potency_bounds)) - set(channels)
    if unknown:
        raise ValueError(f"potency bounds have channels without a dose: {sorted(unknown)}")
    for channel, interval in potency_bounds.items():
        lower = float(interval[0])
        upper = float(interval[1])
        if not np.isfinite([lower, upper]).all() or lower < 0 or upper < lower:
            raise ValueError("potency bounds must be nonnegative and ordered")
        name = str(channel)
        if not lower <= point[name] <= upper:
            raise ValueError(f"potency for {name} is outside its bounds")
        bounds[name] = (lower, upper)
    return bounds


def _share_bounds(
    channel: str,
    companions: list[str],
    dose: pd.Series,
    bounds: Mapping[str, tuple[float, float]],
) -> tuple[float, float]:
    own_dose = float(dose.get(channel, 0.0))
    own_lower = bounds[channel][0] * own_dose
    own_upper = bounds[channel][1] * own_dose
    others_lower = 0.0
    others_upper = 0.0
    for other in companions:
        if other == channel:
            continue
        other_dose = float(dose.get(other, 0.0))
        others_lower += bounds[other][0] * other_dose
        others_upper += bounds[other][1] * other_dose
    lower_total = own_lower + others_upper
    upper_total = own_upper + others_lower
    lower = 0.0 if lower_total <= 0 else own_lower / lower_total
    upper = 0.0 if upper_total <= 0 else own_upper / upper_total
    return lower, upper


def _signed_range(total: float, lower: float, upper: float) -> tuple[float, float]:
    ends = (total * lower, total * upper)
    return min(ends), max(ends)


def _positive_mapping(
    label: str,
    values: Mapping[str, float],
    channels: list[str],
) -> dict[str, float]:
    missing = [channel for channel in channels if channel not in values]
    if missing:
        raise ValueError(f"{label} is missing {missing}")
    parsed = {}
    for channel in channels:
        number = float(values[channel])
        if not np.isfinite(number) or number <= 0:
            raise ValueError(f"{label} must be positive")
        parsed[channel] = number
    return parsed


def _exposure_frame(
    impressions: pd.DataFrame,
    geo_col: str,
    date_col: str,
    channel_col: str,
    impression_col: str,
    quality_col: str | None,
    reach_col: str | None,
    frequency_col: str | None,
) -> pd.DataFrame:
    if (reach_col is None) != (frequency_col is None):
        raise ValueError("reach and frequency must be supplied together")
    columns = [geo_col, date_col, channel_col]
    if reach_col is None:
        columns.append(impression_col)
    else:
        columns.extend([reach_col, frequency_col])
    if quality_col is not None:
        columns.append(quality_col)
    missing = [column for column in columns if column not in impressions.columns]
    if missing:
        raise ValueError(f"impressions are missing {missing}")
    frame = impressions[columns].copy()
    frame[date_col] = pd.to_datetime(frame[date_col])
    frame[channel_col] = frame[channel_col].astype(str)
    for column in columns[3:]:
        frame[column] = frame[column].astype(float)
        if not np.isfinite(frame[column]).all() or (frame[column] < 0).any():
            raise ValueError(f"{column} must be finite and nonnegative")
    return frame


def _frequency_decay(
    channels: list[str],
    frequency_decay: Mapping[str, float] | float | None,
    using_reach: bool,
) -> dict[str, float]:
    if not using_reach:
        return {}
    if frequency_decay is None:
        raise ValueError("frequency_decay is required with reach and frequency")
    if isinstance(frequency_decay, Mapping):
        values = {str(channel): float(value) for channel, value in frequency_decay.items()}
    else:
        values = {channel: float(frequency_decay) for channel in channels}
    missing = [channel for channel in channels if channel not in values]
    if missing:
        raise ValueError(f"frequency_decay is missing {missing}")
    if any(value <= 0 or not np.isfinite(value) for value in values.values()):
        raise ValueError("frequency_decay must be positive")
    return values


def _hill_parameters(
    channels: list[str],
    hill_slope: Mapping[str, float] | None,
    hill_half: Mapping[str, float] | None,
) -> tuple[dict[str, float], dict[str, float]]:
    if (hill_slope is None) != (hill_half is None):
        raise ValueError("hill_slope and hill_half must be supplied together")
    if hill_slope is None or hill_half is None:
        return {}, {}
    slopes = {str(channel): float(value) for channel, value in hill_slope.items()}
    halves = {str(channel): float(value) for channel, value in hill_half.items()}
    if set(slopes) != set(halves):
        raise ValueError("hill_slope and hill_half must name the same channels")
    unknown = set(slopes) - set(channels)
    if unknown:
        raise ValueError(f"Hill parameters have unknown channels: {sorted(unknown)}")
    if any(value <= 0 or not np.isfinite(value) for value in (*slopes.values(), *halves.values())):
        raise ValueError("Hill slope and half-saturation must be positive")
    return slopes, halves


def _base_exposure(
    frame: pd.DataFrame,
    channel_col: str,
    impression_col: str,
    quality_col: str | None,
    reach_col: str | None,
    frequency_col: str | None,
    frequency_decay: Mapping[str, float],
) -> pd.Series:
    if reach_col is not None and frequency_col is not None:
        decay = frame[channel_col].map(frequency_decay).astype(float)
        base = frame[reach_col] * (1 - np.exp(-decay * frame[frequency_col]))
    else:
        base = frame[impression_col].astype(float)
    if quality_col is not None:
        base = base * frame[quality_col]
    return base


def _adstock(
    frame: pd.DataFrame,
    geo_col: str,
    date_col: str,
    channel_col: str,
    decay: float,
) -> pd.DataFrame:
    ordered = frame.sort_values([geo_col, channel_col, date_col]).copy()
    if decay == 0:
        ordered["carried"] = ordered["base"].astype(float)
        return ordered
    carried: list[pd.Series] = []
    for _, group in ordered.groupby([geo_col, channel_col], sort=False):
        level = 0.0
        values = []
        for raw in group["base"].astype(float):
            level = raw + decay * level
            values.append(level)
        carried.append(pd.Series(values, index=group.index))
    ordered["carried"] = pd.concat(carried).sort_index()
    return ordered


def _hill(value: float, slope: float | None, half: float | None) -> float:
    if slope is None or half is None:
        return float(value)
    if value <= 0:
        return 0.0
    powered = float(value) ** slope
    return powered / (powered + half**slope)


def _basis(reach: bool, quality: bool, hill: bool) -> str:
    if reach:
        label = "effective reach"
    else:
        label = "impressions"
    if quality:
        label = f"quality-adjusted {label}"
    if hill:
        label = f"saturated {label}"
    return label


def _prevented(
    wide: pd.DataFrame,
    design: ConcurrentDesign,
    channels: list[str],
    geo_col: str,
    date_col: str,
    treatment_start: str,
    basis: str,
) -> pd.DataFrame:
    assignment = design.assignment.drop_duplicates(geo_col).set_index(geo_col)["cell"].astype(str)
    frame = wide.copy()
    frame["cell"] = frame[geo_col].map(assignment)
    if frame["cell"].isna().any():
        raise ValueError("impression markets are missing from the assignment")
    post = frame[date_col] >= pd.Timestamp(treatment_start)
    control_post = frame.loc[post & frame["cell"].eq("control"), channels]
    control_pre = frame.loc[~post & frame["cell"].eq("control"), channels]
    if control_post.empty:
        raise ValueError("the test window has no control markets")
    if control_pre.empty:
        raise ValueError("the pre-test window has no control markets")
    control_change = control_post.mean() - control_pre.mean()
    off = {
        (str(row.cell), str(row.channel))
        for row in design.schedule.itertuples(index=False)
        if int(row.off) == 1
    }
    rows = []
    for cell in sorted(set(design.schedule["cell"].astype(str))):
        dark_post = frame.loc[post & frame["cell"].eq(cell), channels]
        dark_pre = frame.loc[~post & frame["cell"].eq(cell), channels]
        if dark_post.empty or dark_pre.empty:
            raise ValueError(f"cell {cell} is missing a pre-test or test window")
        lost = control_change - (dark_post.mean() - dark_pre.mean())
        for channel in channels:
            active = (cell, channel) in off
            rows.append(
                {
                    "cell": cell,
                    "channel": channel,
                    "dose": float(max(lost[channel], 0.0)) if active else 0.0,
                    "basis": basis,
                }
            )
    return pd.DataFrame(rows)

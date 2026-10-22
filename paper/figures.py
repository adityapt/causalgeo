"""Figures for the SoftwareX manuscript."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from example import build_panel, load_media
from causalgeo import disentangle
from causalgeo.exposure import adjust_for_exposure
from causalgeo.identified import estimate_identified

OUT = Path(__file__).resolve().parent / "results"
DARK = "#4C566A"
BLUE = "#5E81AC"
ACCENT = "#BF616A"


def bundle_contrast(raw: pd.DataFrame) -> pd.DataFrame:
    design, outcomes, impressions, start = build_panel(raw)
    result = disentangle(
        design,
        outcomes,
        impressions,
        outcome="sales",
        treatment_start=start,
        adjust_exposure=True,
        check_stability=False,
    )
    identified = estimate_identified(design, outcomes, outcome="sales", treatment_start=start)
    identified, _ = adjust_for_exposure(
        design,
        outcomes,
        impressions,
        identified,
        treatment_start=start,
        outcome="sales",
        apply=True,
    )
    model = result.bundles.set_index("cell")
    rows = []
    cells = identified.cells.set_index("cell")
    for record in identified.cells.itertuples(index=False):
        rows.append(
            {
                "cell": record.cell,
                "bundle": record.bundle,
                "experimental": float(cells.loc[record.cell, "coef"]),
                "se": float(cells.loc[record.cell, "std_err"]),
                "ci_lower": float(cells.loc[record.cell, "ci_lower"]),
                "ci_upper": float(cells.loc[record.cell, "ci_upper"]),
                "model_implied": float(model.loc[record.cell, "predicted_effect"]),
            }
        )
    return pd.DataFrame(rows)


def plot_separation(detail: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 3.05), sharex=True)
    for scale, color, label in ((1.0, DARK, "Original slopes"), (0.25, BLUE, "Quarter slopes")):
        group = detail.loc[detail["scale"] == scale]
        rhos = sorted(group["rho"].unique())
        error, cover, cover_error, withheld = [], [], [], []
        for rho in rhos:
            block = group.loc[group["rho"] == rho]
            reported = block.loc[block["status"] == "separate"]
            error.append(float(reported["abs_pct_error"].median()))
            run_coverage = reported.groupby("seed")["covered"].mean()
            cover.append(float(run_coverage.mean()))
            cover_error.append(
                1.96 * float(run_coverage.std(ddof=1) / np.sqrt(len(run_coverage)))
            )
            pair = block.loc[block["channel"].isin(["Channel0", "Channel1"])]
            withheld.append(float((pair["status"] == "tied").mean()))
        x = list(range(len(rhos)))
        axes[0].plot(x, error, marker="o", color=color, label=label)
        axes[1].errorbar(
            x,
            cover,
            yerr=cover_error,
            marker="o",
            color=color,
            capsize=2,
            label=label,
        )
        axes[2].plot(x, withheld, marker="o", color=color, label=label)
    titles = ["Error of reported slopes", "95% interval coverage", "Copied pair withheld"]
    ylabels = ["Median |error| (%)", "Share covered", "Share withheld"]
    for ax, title, ylabel in zip(axes, titles, ylabels):
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("Mix weight")
        ax.set_ylabel(ylabel)
        ax.set_xticks([0, 1, 2], ["0", "0.9", "1"])
        ax.set_xlim(-0.25, 2.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[1].set_ylim(-0.05, 1.05)
    axes[2].set_ylim(-0.05, 1.05)
    axes[1].axhline(0.95, color="#D8DEE9", linewidth=0.8, zorder=0)
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"), dpi=200)
    plt.close(fig)


def plot_bundles(contrast: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.4, 2.8))
    y = np.arange(len(contrast))
    ax.errorbar(
        contrast["experimental"],
        y + 0.12,
        xerr=1.96 * contrast["se"],
        fmt="o",
        color=DARK,
        ecolor=DARK,
        capsize=3,
        label="Experiment",
        linestyle="none",
    )
    ax.scatter(
        contrast["model_implied"],
        y - 0.12,
        marker="D",
        color=ACCENT,
        s=28,
        label="Model, same impression change",
        zorder=3,
    )
    ax.set_yticks(y)
    ax.set_yticklabels(contrast["bundle"])
    ax.set_xlabel("Sales per market-week")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(
        frameon=False,
        fontsize=8,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.22),
        ncol=2,
    )
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.28)
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def refused_split(raw: pd.DataFrame) -> pd.DataFrame:
    """Channel2 when a copied pair is kept as a control, and when it is dropped."""
    from example import CHANNELS, NOISE_SD, SALES_SEED, SLOPES

    from causalgeo.calibrated import _observational_block, _observational_variance, _withheld
    from causalgeo.exposure import adjust_for_exposure
    from causalgeo.identified import estimate_identified

    design, _outcomes, impressions, start = build_panel(raw)
    copied = impressions.copy()
    base = copied.loc[copied["channel"] == "Channel0", ["geo", "date", "impressions"]]
    other = copied.loc[copied["channel"] == "Channel1", ["geo", "date"]].merge(base, on=["geo", "date"])
    other["channel"] = "Channel1"
    copied = pd.concat([copied.loc[copied["channel"] != "Channel1"], other], ignore_index=True)
    wide = copied.pivot_table(
        index=["geo", "date"], columns="channel", values="impressions", aggfunc="sum"
    ).reset_index()
    rng = np.random.default_rng(SALES_SEED)
    geo_fe = dict(zip(wide["geo"].unique(), rng.normal(0, 50, wide["geo"].nunique())))
    dates = wide["date"].drop_duplicates().sort_values()
    date_fe = dict(zip(dates, rng.normal(0, 20, len(dates))))
    signal = sum(SLOPES[channel] * wide[channel] for channel in CHANNELS)
    wide["sales"] = (
        wide["geo"].map(geo_fe) + wide["date"].map(date_fe) + signal + rng.normal(0, NOISE_SD, len(wide))
    )
    sales = wide[["geo", "date", "sales"]]
    paths = wide[["geo", "date", *CHANNELS]]
    identified = estimate_identified(design, sales, outcome="sales", treatment_start=start)
    identified, _ = adjust_for_exposure(
        design, sales, copied, identified, treatment_start=start, outcome="sales", apply=True
    )
    obs_x, obs_y = _observational_block(sales, paths, design, "sales", "geo", "date", CHANNELS, start)
    withheld = _withheld(paths, design, CHANNELS, "geo", "date", start)
    kept = disentangle(
        design,
        sales,
        copied,
        outcome="sales",
        treatment_start=start,
        adjust_exposure=True,
        check_stability=False,
    )
    kept_row = kept.channels.set_index("channel").loc["Channel2"]
    sigma2 = _observational_variance(obs_x, obs_y)
    x_rows = [obs_x[:, [2]]]
    y_rows = [obs_y]
    w_rows = [np.full(len(obs_y), 1.0 / sigma2)]
    effects = identified.cells.set_index("cell")
    for cell, group in withheld.groupby("cell", sort=True):
        effect = float(effects.loc[cell, "coef"])
        se = max(float(effects.loc[cell, "std_err"]), 1e-8)
        dose = float(group.set_index("channel").loc["Channel2", "withheld"])
        x_rows.append(np.array([[dose]]))
        y_rows.append(np.array([-effect]))
        w_rows.append(np.array([1.0 / se**2]))
    x = np.vstack(x_rows)
    y = np.concatenate(y_rows)
    w = np.concatenate(w_rows)
    sw = np.sqrt(w)
    xtwx = (x * sw[:, None]).T @ (x * sw[:, None])
    xtwy = (x * sw[:, None]).T @ (y * sw)
    fitted = float(np.linalg.lstsq(xtwx, xtwy, rcond=1e-8)[0][0])
    dropped_se = float(np.sqrt(np.linalg.pinv(xtwx, rcond=1e-8)[0, 0]))
    return pd.DataFrame(
        [
            {
                "analysis": "Pair kept as one control",
                "per_10000": float(kept_row["coef"]) * 1e4,
                "se_per_10000": float(kept_row["std_err"]) * 1e4,
            },
            {
                "analysis": "Pair dropped from the fit",
                "per_10000": fitted * 1e4,
                "se_per_10000": dropped_se * 1e4,
            },
        ]
    )


def plot_refused(comparison: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.4, 2.5))
    y = np.arange(len(comparison))[::-1]
    ax.errorbar(
        comparison["per_10000"],
        y,
        xerr=1.96 * comparison["se_per_10000"],
        fmt="o",
        color=DARK,
        ecolor=DARK,
        capsize=3,
        linestyle="none",
    )
    ax.axvline(1.0, color=ACCENT, linewidth=1.0, label="Planted Channel2")
    ax.set_yticks(y)
    ax.set_yticklabels(comparison["analysis"])
    ax.set_xlabel("Channel2, sales per 10,000 impressions")
    ax.set_xlim(0, 3.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def failure_chart() -> pd.DataFrame:
    robust = pd.read_csv(OUT / "robustness.csv").set_index("scenario")
    stress = pd.read_csv(OUT / "stress.csv").set_index("scenario")
    rows = [
        ("Linear", robust.loc["calibrated"]),
        ("Serial correlation", robust.loc["serial"]),
        ("Incomplete delivery", robust.loc["incomplete"]),
        ("Heterogeneous slopes", robust.loc["heterogeneous"]),
        ("Budget moved", robust.loc["redistribution"]),
        ("Budget moved, refit", robust.loc["redistribution_adjusted"]),
        ("Carryover omitted", stress.loc["carryover_omitted"]),
        ("Saturation omitted", stress.loc["saturation_omitted"]),
    ]
    return pd.DataFrame(
        [
            {
                "setting": label,
                "coverage": float(row["coverage"]),
                "gap": float(row["median_abs_bundle_gap_pct"]),
            }
            for label, row in rows
        ]
    )


def plot_failures(frame: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.35), sharey=True)
    y = np.arange(len(frame))[::-1]
    coverage_colors = [
        ACCENT if coverage < 0.9 else BLUE if setting.endswith("refit") else DARK
        for setting, coverage in zip(frame["setting"], frame["coverage"])
    ]
    gap_colors = [
        ACCENT if gap > 10 else BLUE if setting.endswith("refit") else DARK
        for setting, gap in zip(frame["setting"], frame["gap"])
    ]
    axes[0].scatter(frame["coverage"], y, c=coverage_colors, s=28, zorder=3)
    axes[0].axvline(0.95, color="#D8DEE9", linewidth=0.8, zorder=0)
    axes[0].set_xlim(-0.05, 1.05)
    axes[0].set_xlabel("Interval coverage")
    axes[1].scatter(frame["gap"], y, c=gap_colors, s=28, zorder=3)
    axes[1].set_xlabel("Bundle gap (%)")
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(frame["setting"])
    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    detail = pd.read_csv(OUT / "reliability_detail.csv")
    plot_separation(detail, OUT / "separation.pdf")
    raw = load_media()
    contrast = bundle_contrast(raw)
    contrast.to_csv(OUT / "bundle_contrast.csv", index=False)
    plot_bundles(contrast, OUT / "bundle_contrast.pdf")
    refused = refused_split(raw)
    refused.to_csv(OUT / "refused.csv", index=False)
    plot_refused(refused, OUT / "refused.pdf")
    failures = failure_chart()
    failures.to_csv(OUT / "failures.csv", index=False)
    plot_failures(failures, OUT / "failures.pdf")
    print(contrast.to_string(index=False))
    print(refused.to_string(index=False))
    print(failures.to_string(index=False))


if __name__ == "__main__":
    main()

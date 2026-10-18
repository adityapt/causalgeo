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
    result = disentangle(design, outcomes, impressions, outcome="sales", treatment_start=start)
    identified = estimate_identified(design, outcomes, outcome="sales", treatment_start=start)
    identified, _ = adjust_for_exposure(
        design, outcomes, impressions, identified, treatment_start=start, outcome="sales"
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


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    detail = pd.read_csv(OUT / "reliability_detail.csv")
    plot_separation(detail, OUT / "separation.pdf")
    contrast = bundle_contrast(load_media())
    contrast.to_csv(OUT / "bundle_contrast.csv", index=False)
    plot_bundles(contrast, OUT / "bundle_contrast.pdf")
    print(contrast.to_string(index=False))


if __name__ == "__main__":
    main()

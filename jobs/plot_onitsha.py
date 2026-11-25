"""Show satellite-derived surface-water change and uncertainty for each target date."""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path("reports/satellite_replay")


def main():
    with (ROOT / "onitsha_south.csv").open() as f:
        rows = {(int(r["year"]), int(r["month"])): r for r in csv.DictReader(f)}
    fig, axes = plt.subplots(
        2, 1, figsize=(8, 5), sharex=True, sharey=True, layout="constrained"
    )
    for ax, month, title in zip(axes, (7, 10), ("July", "October")):
        ax.axvspan(2022.5, 2025.5, color="#edf4f5")
        ax.axhline(0.5, color="#934923", linestyle="--", linewidth=1)
        for year in range(2018, 2026):
            row = rows.get((year, month))
            if not row:
                ax.text(
                    year,
                    0.06,
                    "missing",
                    ha="center",
                    fontsize=7,
                    transform=ax.get_xaxis_transform(),
                )
                continue
            low = int(row["known_gained_pixels"]) / 10000
            high = low + int(row["unknown_pixels"]) / 10000
            ax.vlines(year, low, high, color="#447385", linewidth=3)
            ax.scatter(
                year, high, s=25, edgecolor="#447385", facecolor="white", zorder=3
            )
            ax.scatter(year, low, s=20, color="#184c5a", zorder=4)
        ax.set_title(title, loc="left", fontsize=11)
        ax.set_ylabel("Area (km²)")
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlim(2017.5, 2025.5)
        ax.set_ylim(
            -0.025,
            max(
                0.55,
                1.08
                * max(
                    (int(r["known_gained_pixels"]) + int(r["unknown_pixels"])) / 10000
                    for r in rows.values()
                ),
            ),
        )
    axes[1].set_xticks(range(2018, 2026))
    fig.suptitle("Onitsha South · change in mapped water since February", fontsize=15)
    fig.legend(
        handles=[
            Line2D(
                [], [], color="#184c5a", marker="o", linestyle="", label="Known change"
            ),
            Line2D(
                [],
                [],
                color="#447385",
                marker="o",
                markerfacecolor="white",
                linestyle="",
                label="Including uncertain pixels",
            ),
            Line2D([], [], color="#934923", linestyle="--", label="0.5 km² threshold"),
        ],
        loc="outside lower center",
        ncol=3,
        frameon=False,
    )
    fig.savefig(ROOT / "onitsha_change.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()

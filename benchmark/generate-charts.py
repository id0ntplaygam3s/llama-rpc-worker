#!/usr/bin/env python3
"""Regenerate benchmark/prompt-chart.png and benchmark/gen-chart.png.

Reads results-single.csv and results-dual.csv from the same folder as this
script (columns: label,prompt_tokens,run,pp_tps,tg_tps,pp_s,tg_s,gen_tokens)
and writes the two PNGs next to them.

Requires: matplotlib  (pip install matplotlib)
Usage:    python benchmark/generate-charts.py
"""
import csv
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
SINGLE_CSV = HERE / "results-single.csv"
DUAL_CSV = HERE / "results-dual.csv"

SINGLE_COLOR = "#1f77b4"
DUAL_COLOR = "#e8710a"
DODGE = 0.09  # horizontal offset so the two series' error bars don't overlap


def label(tokens: int) -> str:
    if tokens < 1000:
        return str(tokens)
    return f"{tokens // 1000}K"


def load(path: Path):
    """Return {prompt_tokens: {"pp": [...], "tg": [...]}}.

    Skips lines starting with '#' (benchmark context comments).
    """
    data = defaultdict(lambda: {"pp": [], "tg": []})
    with open(path, newline="", encoding="utf-8") as f:
        lines = (line for line in f if not line.startswith("#"))
        for row in csv.DictReader(lines):
            size = int(row["prompt_tokens"])
            data[size]["pp"].append(float(row["pp_tps"]))
            data[size]["tg"].append(float(row["tg_tps"]))
    return data


def stats(values):
    return statistics.mean(values), min(values), max(values)


def draw(metric, title, ylabel, out_path, single, dual, note):
    sizes = sorted(set(single) & set(dual))
    xs = list(range(len(sizes)))  # evenly spaced: no overlapping labels, no implied in-between values

    fig, ax = plt.subplots(figsize=(11, 6), dpi=150)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    for series, color, marker, name, offset, style in (
        (single, SINGLE_COLOR, "o", "Single GPU", -DODGE, "-"),
        (dual, DUAL_COLOR, "s", "Dual GPU", DODGE, "--"),
    ):
        means, lows, highs = [], [], []
        for s in sizes:
            m, lo, hi = stats(series[s][metric])
            means.append(m)
            lows.append(m - lo)
            highs.append(hi - m)
        pos = [x + offset for x in xs]
        ax.errorbar(
            pos,
            means,
            yerr=[lows, highs],
            color=color,
            marker=marker,
            markersize=7,
            linewidth=2,
            linestyle=style,
            capsize=4,
            elinewidth=1.4,
            label=name,
        )

    ax.set_xticks(xs)
    ax.set_xticklabels([label(s) for s in sizes])
    ax.set_xlabel("Context size (tokens)", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=14, pad=26)
    ax.text(
        0.5,
        1.015,
        note,
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=9.5,
        color="#555555",
    )
    ax.grid(True, axis="y", alpha=0.3)
    ax.grid(False, axis="x")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.legend(frameon=False, fontsize=10, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, facecolor="white")
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    single = load(SINGLE_CSV)
    dual = load(DUAL_CSV)
    note = "Markers = mean of 5 runs; bars = min–max across runs"

    draw(
        "pp",
        "Prompt processing (tok/s) — higher is better",
        "Prompt tokens/s",
        HERE / "prompt-chart.png",
        single,
        dual,
        note,
    )
    draw(
        "tg",
        "Generation speed (tok/s) — higher is better",
        "Generated tokens/s",
        HERE / "gen-chart.png",
        single,
        dual,
        note,
    )


if __name__ == "__main__":
    main()

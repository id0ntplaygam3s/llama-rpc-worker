#!/usr/bin/env python3
"""Regenerate benchmark charts from results-single.csv and results-dual.csv.

Usage:
    python generate-charts.py          # reads ./results-single.csv and ./results-dual.csv
    python generate-charts.py --single path --dual path --out-dir path

Outputs:
    prompt-chart.png  — prompt processing speed (tok/s), higher is better
    gen-chart.png     — generation speed (tok/s), higher is better
"""

import argparse
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── helpers ──────────────────────────────────────────────────────────────────

def read_csv(path):
    """Return dict: {context_size: (mean_pp, mean_tg)}."""
    data = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        buckets = {}
        for row in reader:
            ctx = int(row["prompt_tokens"])
            pp = float(row["pp_tps"])
            tg = float(row["tg_tps"])
            buckets.setdefault(ctx, []).append((pp, tg))
        for ctx, runs in buckets.items():
            n = len(runs)
            data[ctx] = (
                sum(r[0] for r in runs) / n,
                sum(r[1] for r in runs) / n,
            )
    return data


def read_minmax(path):
    """Return dict: {context_size: (min_tg, max_tg)} for dual generation."""
    mm = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        buckets = {}
        for row in reader:
            ctx = int(row["prompt_tokens"])
            tg = float(row["tg_tps"])
            buckets.setdefault(ctx, []).append(tg)
        for k, vals in buckets.items():
            mm[k] = (min(vals), max(vals))
    return mm


def fmt_ctx(ctx):
    """Format context size for axis labels."""
    if ctx >= 1000:
        return f"{ctx // 1000}K"
    return str(ctx)


# ── plotting ─────────────────────────────────────────────────────────────────

def plot_chart(title, y_label, data_single, data_dual, error_key,
               output_path, dual_minmax=None):
    """Draw a line chart with single/dual lines, save to *output_path*."""
    all_ctxs = sorted(set(data_single) | set(data_dual))

    fig, ax = plt.subplots(figsize=(10, 5.5))

    color_single = "#1f77b4"
    color_dual = "#ff7f0e"
    marker_single = "o"
    marker_dual = "s"

    # Single line
    ys_single = [data_single[c][1] if error_key == "tg" else data_single[c][0]
                 for c in all_ctxs]
    ax.plot(all_ctxs, ys_single, marker=marker_single, color=color_single,
            linewidth=2, markersize=7, label="Single GPU", zorder=3)

    # Dual line
    ys_dual = [data_dual[c][1] if error_key == "tg" else data_dual[c][0]
               for c in all_ctxs]
    ax.plot(all_ctxs, ys_dual, marker=marker_dual, color=color_dual,
            linewidth=2, linestyle="--", markersize=7,
            label="Dual GPU", zorder=3)

    # Dual error bars (fill + errorbar)
    if dual_minmax and error_key == "tg":
        valid_ctxs = [c for c in all_ctxs if c in dual_minmax]
        lo = [dual_minmax[c][0] for c in valid_ctxs]
        hi = [dual_minmax[c][1] for c in valid_ctxs]
        means = [data_dual[c][1] for c in valid_ctxs]

        ax.fill_between(valid_ctxs, lo, hi, color=color_dual, alpha=0.15,
                        zorder=1, label="Dual GPU (min–max range)")
        ax.errorbar(valid_ctxs, means,
                    yerr=[[means[i] - lo[i] for i in range(len(means))],
                          [hi[i] - means[i] for i in range(len(means))]],
                    fmt="s", color=color_dual, capsize=4, linewidth=2,
                    markersize=7, zorder=4,
                    label="Dual GPU (± range)")

    # ── styling ───────────────────────────────────────────────────────────
    ax.set_title(title, fontsize=14, pad=12)
    ax.set_ylabel(y_label, fontsize=12)
    ax.set_xlabel("Context size (tokens)", fontsize=12)
    ax.set_xscale("log")
    ax.set_xticks(all_ctxs)
    ax.set_xticklabels([fmt_ctx(c) for c in all_ctxs])
    ax.grid(True, alpha=0.3, which="both")
    ax.tick_params(axis="both", labelsize=11)

    ax.legend(fontsize=10, loc="best")

    plt.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate benchmark charts")
    parser.add_argument("--single", default="results-single.csv")
    parser.add_argument("--dual", default="results-dual.csv")
    parser.add_argument("--out-dir", default=".")
    args = parser.parse_args()

    data_single = read_csv(args.single)
    data_dual = read_csv(args.dual)

    print(f"Single: {len(data_single)} context sizes")
    print(f"Dual:   {len(data_dual)} context sizes")

    # Read dual min-max for error bars
    dual_csv = os.path.join(os.path.dirname(os.path.abspath(args.dual)),
                            "results-dual.csv")
    dual_minmax = None
    if os.path.exists(dual_csv):
        dual_minmax = read_minmax(dual_csv)
        print(f"Dual min-max: {len(dual_minmax)} context sizes")

    plot_chart(
        "Prompt processing (tok/s) — higher is better",
        "Prompt tokens/s",
        data_single, data_dual, "pp",
        os.path.join(args.out_dir, "prompt-chart.png"),
    )

    plot_chart(
        "Generation speed (tok/s) — higher is better",
        "Generated tokens/s",
        data_single, data_dual, "tg",
        os.path.join(args.out_dir, "gen-chart.png"),
        dual_minmax=dual_minmax,
    )

    print("Done.")


if __name__ == "__main__":
    main()

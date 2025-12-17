# baseline_comparison.py
# Usage:
#   python3 baseline_comparison.py \
#       --csv data.csv \
#       --out chart.png \
#       --baseline-throughput 40.0 \
#       --baseline-pass1 90.0
#
# CSV columns (two supported formats):
#  A) Absolute values:
#     technique,throughput_tps,pass1,train_tokens_B
#     (If --baseline-* not provided, baseline = first row.)
#  B) Precomputed deltas (skip --baseline-*):
#     technique,delta_throughput_tps,delta_pass1,train_tokens_B
#
# Notes:
# - Bubble size is proportional to sqrt(training tokens, in billions).
# - Labels have a small offset; tune --label-offset-x/--label-offset-y.

import argparse
import math
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PURPLE = "#7e57c2"  # single color for all data points

def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def _parse_tokens_to_float(x) -> float:
    """Accept numbers or strings like '~1', '322+', '50B' and return a float."""
    if pd.isna(x):
        return 0.0
    if isinstance(x, (int, float)):
        return float(x)
    m = re.search(r'(\d+(\.\d+)?)', str(x))
    return float(m.group(1)) if m else 0.0

def compute_deltas(df: pd.DataFrame,
                   baseline_tps: float | None,
                   baseline_pass1: float | None) -> pd.DataFrame:
    df = df.copy()

    has_deltas = {"delta_throughput_tps", "delta_pass1"}.issubset(df.columns)
    if has_deltas:
        # Normalize delta column names
        df.rename(columns={
            "delta_throughput_tps": "delta_throughput",
            "delta_pass1": "delta_pass1"
        }, inplace=True)
        # Recover absolute metrics if baseline provided or can be inferred
        if baseline_tps is not None and baseline_pass1 is not None:
            df["abs_throughput"] = df["delta_throughput"] + float(baseline_tps)
            df["abs_pass1"]      = df["delta_pass1"] + float(baseline_pass1)
        else:
            # Cannot compute absolute values without a baseline -> mark NaN
            df["abs_throughput"] = np.nan
            df["abs_pass1"]      = np.nan
        return df

    # Absolute metrics path -> need a baseline (explicit or first row)
    needed = {"throughput_tps", "pass1"}
    if not needed.issubset(df.columns):
        raise ValueError("CSV must include either {delta_throughput_tps, delta_pass1} "
                         "or {throughput_tps, pass1}.")
    if baseline_tps is None or baseline_pass1 is None:
        baseline_tps = float(df.iloc[0]["throughput_tps"])
        baseline_pass1 = float(df.iloc[0]["pass1"])

    df["delta_throughput"] = df["throughput_tps"] - baseline_tps
    df["delta_pass1"] = df["pass1"] - baseline_pass1
    df["abs_throughput"] = df["throughput_tps"]
    df["abs_pass1"]      = df["pass1"]
    return df

def bubble_sizes(tokens_B_series: pd.Series, scale: float = 80.0):
    # Matplotlib scatter 's=' is in points^2; area grows with sqrt(tokens).
    vals = tokens_B_series.apply(_parse_tokens_to_float).clip(lower=0.0)
    return [scale * math.sqrt(v) for v in vals]

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--out", default="paper/baseline_comparisons.png")
    # AR-adapted Parallel Decoders: Speed-Quality Trade-offs (with Costs)
    p.add_argument("--title", default="Speed-Quality Trade-offs (on A100 GPU)")
    p.add_argument("--baseline-throughput", type=float, default=None)
    p.add_argument("--baseline-pass1", type=float, default=None)
    p.add_argument("--size-scale", type=float, default=80.0)
    p.add_argument("--label-offset-x", type=float, default=+8.0)
    p.add_argument("--label-offset-y", type=float, default=0.0)
    args = p.parse_args()

    df = load_data(args.csv)
    df_delta = compute_deltas(df, args.baseline_throughput, args.baseline_pass1)

    if "train_tokens_B" not in df.columns:
        raise ValueError("CSV must include 'train_tokens_B' (billions of tokens) for bubble sizes.")

    # Bubble sizes (area)
    sizes = bubble_sizes(df["train_tokens_B"], scale=args.size_scale)

    fig, ax = plt.subplots(figsize=(8, 6))
    
    sc = ax.scatter(
        df_delta["delta_throughput"],
        df_delta["delta_pass1"],
        s=sizes,
        color=PURPLE,
    )

    # Labels near each point
    color="black"
    for i, r in df_delta.iterrows():
        if "Jacobi Forcing" in df_delta.loc[i, "technique"]:
            label_offset_x = -38.0
            label_offset_y = 1.0
            color = "red"
        elif "qwen-2.5-coder-7b-instruct" in df_delta.loc[i, "technique"]:
            label_offset_x = -8.0
            label_offset_y = 1.0
        else:
            label_offset_x = args.label_offset_x
            label_offset_y = args.label_offset_y
        ax.annotate(
            df_delta.loc[i, "technique"],
            xy=(r["delta_throughput"], r["delta_pass1"]),
            xytext=(
                r["delta_throughput"] + label_offset_x,
                r["delta_pass1"] + label_offset_y
            ),
            fontsize=16,
            color=color
        )


    # Axes lines (dashed) through zero
    ax.axhline(0, linewidth=1, linestyle="--")
    ax.axvline(0, linewidth=1, linestyle="--")

    # Mark the baseline (origin) and annotate absolute baseline values
    ax.scatter(0, 0, 
               s=260, 
               marker="X", 
               color=PURPLE, 
               zorder=5,
               alpha=0.5
    )
    
    ax.annotate(
        "Baseline (41.3 t/s, 87.8%)",
        xy=(0, 0),
        xytext=(-40, 4),         # adjust offsets if needed
        fontsize=16,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="gray", alpha=0.8),
    )

    # mark the data point with highest (accuracy × throughput) ---
    # Use absolute metrics when available (df_delta['abs_*'] may be NaN if deltas w/o baseline).
    if "abs_throughput" in df_delta.columns and "abs_pass1" in df_delta.columns:
        score = df_delta["abs_throughput"] * df_delta["abs_pass1"]
        if score.notna().any():
            idx = score.idxmax()
            x_star = df_delta.loc[idx, "delta_throughput"]
            y_star = df_delta.loc[idx, "delta_pass1"]
            abs_tps = df_delta.loc[idx, "abs_throughput"]
            abs_acc = df_delta.loc[idx, "abs_pass1"]

            #ax.scatter(x_star, y_star, 
            #           s=320, 
            #           marker="*", 
            #           color="red",
            #           edgecolor="white", 
            #           linewidth=0.8, 
            #           zorder=6,
            #           alpha=0.5,
            #)

            ax.annotate(
                f"{abs_tps:.1f} t/s, {abs_acc:.1f}%",
                xy=(x_star, y_star),
                xytext=(x_star - 26.75, y_star - 8.0),
                fontsize=16,
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="gray", alpha=0.8),
                arrowprops=dict(arrowstyle="->", lw=1)
            )

    ax.set_xlabel("Δ Throughput (tokens/sec) vs. Baseline", fontsize=20)
    ax.set_ylabel("Δ pass@1 on HumanEval vs. Baseline", fontsize=20)
    ax.set_title(args.title, fontsize=20)

    # --- Size legend: fixed 0B, 1B, 10B, 100B ---
    legend_levels = [0.001, 1.0, 10.0, 100.0]
    legend_sizes = [args.size_scale * math.sqrt(lv) for lv in legend_levels]
    handles = [ax.scatter([], [], s=s, color=PURPLE, alpha=1.0) for s in legend_sizes]
    labels = [f"{int(lv)}B" for lv in legend_levels]
    size_legend = ax.legend(handles, labels, title="adaptation cost (train tokens)", loc="lower right",
                            frameon=True, labelspacing=1.0, fontsize=12, title_fontsize=12)
    ax.add_artist(size_legend)

    # Make axis numbers larger (optional)
    ax.tick_params(axis="both", which="major", labelsize=16)

    # Ensure output directory exists
    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    
    ax.set_ylim(top=7)
    
    fig.tight_layout()
    fig.savefig(args.out, dpi=200)
    print(f"Saved {args.out}")

if __name__ == "__main__":
    main()

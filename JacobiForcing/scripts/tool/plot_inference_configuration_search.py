#!/usr/bin/env python3
import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def load_data(jsonl_path: Path) -> pd.DataFrame:
    df = pd.read_json(jsonl_path, lines=True)

    # basic sanity checks
    required_cols = ["K", "r", "block_size", "ngram_size", "avg_toks_per_sec"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in JSONL: {missing}")

    return df


def filter_for_hparams(df: pd.DataFrame, K: int, r: float, r_tol: float = 1e-3):
    """Filter dataframe for given K and r (with tolerance on r)."""
    mask = (df["K"] == K) & (df["r"].sub(r).abs() <= r_tol)
    filtered = df[mask].copy()
    if filtered.empty:
        raise ValueError(f"No rows found with K={K} and r≈{r}")
    return filtered


def aggregate_tokens_per_sec(df: pd.DataFrame) -> pd.DataFrame:
    """Average tokens/s in case of duplicates for same config."""
    return (
        df.groupby(["block_size", "ngram_size"], as_index=False)["avg_toks_per_sec"]
        .mean()
        .rename(columns={"avg_toks_per_sec": "toks_per_sec"})
    )


def plot_tokens_vs_block_size(df_agg: pd.DataFrame, out_path: Path):
    """
    First set:
    x-axis: block_size
    y-axis: tokens/s
    one curve per ngram_size

    Color scheme: darker for smaller ngram_size, lighter for larger ngram_size.
    """
    plt.figure()

    ngram_values = sorted(df_agg["ngram_size"].unique())
    cmap = plt.get_cmap("Blues_r")  # reversed: small -> dark, large -> light
    n = len(ngram_values)
    denom = max(1, n - 1)

    for i, ngram_size in enumerate(ngram_values):
        sub = df_agg[df_agg["ngram_size"] == ngram_size]
        sub_sorted = sub.sort_values("block_size")

        # normalized position in [0,1]
        t = i / denom
        color = cmap(t)

        plt.plot(
            sub_sorted["block_size"],
            sub_sorted["toks_per_sec"],
            marker="o",
            label=f"ngram={ngram_size}",
            color=color,
        )

    plt.xlabel("Block size (n_token_seq_len)")
    plt.ylabel("Tokens / second")
    plt.title("Tokens/s vs Block Size (fixed K=2, r=0.85)")
    plt.grid(True, alpha=0.3)
    plt.legend(title="N-gram size")
    plt.tight_layout()
    plt.savefig(out_path)
    print(f"[PLOT] Saved tokens_vs_block_size to {out_path}")


def plot_tokens_vs_ngram_size(df_agg: pd.DataFrame, out_path: Path):
    """
    Second set:
    x-axis: ngram_size
    y-axis: tokens/s
    one curve per block_size

    Color scheme: darker for smaller block_size, lighter for larger block_size.
    """
    plt.figure()

    block_values = sorted(df_agg["block_size"].unique())
    cmap = plt.get_cmap("Greens_r")  # reversed: small -> dark, large -> light
    n = len(block_values)
    denom = max(1, n - 1)

    for i, block_size in enumerate(block_values):
        sub = df_agg[df_agg["block_size"] == block_size]
        sub_sorted = sub.sort_values("ngram_size")

        # normalized position in [0,1]
        t = i / denom
        color = cmap(t)

        plt.plot(
            sub_sorted["ngram_size"],
            sub_sorted["toks_per_sec"],
            marker="o",
            label=f"block={block_size}",
            color=color,
        )

    plt.xlabel("N-gram size")
    plt.ylabel("Tokens / second")
    plt.title("Tokens/s vs N-gram Size (fixed K=2, r=0.85)")
    plt.grid(True, alpha=0.3)
    plt.legend(title="Block size")
    plt.tight_layout()
    plt.savefig(out_path)
    print(f"[PLOT] Saved tokens_vs_ngram_size to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_jsonl",
        default="/home/lah003/workspace/CLLM2/profiling_results/summary/shiftedattn-10-16-7b-qwen2p5-coder-n32w16-n16distill-data-v2-ar-1-cyclic-noise-all-1e-6-summary.jsonl",
        help="Path to summary JSONL generated from logs",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Directory to save plots (default: same dir as JSONL)",
    )
    parser.add_argument("--K", type=int, default=2, help="Fixed K value")
    parser.add_argument("--r", type=float, default=0.85, help="Fixed r value")
    parser.add_argument(
        "--r_tol",
        type=float,
        default=1e-3,
        help="Tolerance when matching r (default: 1e-3)",
    )
    args = parser.parse_args()

    jsonl_path = Path(args.input_jsonl)
    if not jsonl_path.is_file():
        raise SystemExit(f"JSONL file not found: {jsonl_path}")

    out_dir = Path(args.out_dir) if args.out_dir else jsonl_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_data(jsonl_path)
    df_filtered = filter_for_hparams(df, K=args.K, r=args.r, r_tol=args.r_tol)

    df_agg = aggregate_tokens_per_sec(df_filtered)

    # First plot: tokens/s vs block size (curves by ngram_size)
    plot_tokens_vs_block_size(
        df_agg, out_dir / "tokens_vs_block_size_K2_r0.85.png"
    )

    # Second plot: tokens/s vs ngram size (curves by block_size)
    plot_tokens_vs_ngram_size(
        df_agg, out_dir / "tokens_vs_ngram_size_K2_r0.85.png"
    )

    print("[DONE]")


if __name__ == "__main__":
    main()

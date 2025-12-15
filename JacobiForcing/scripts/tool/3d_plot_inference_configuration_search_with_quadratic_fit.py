#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


def load_data(jsonl_path: Path) -> pd.DataFrame:
    df = pd.read_json(jsonl_path, lines=True)

    required_cols = ["K", "r", "block_size", "ngram_size", "avg_toks_per_sec"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in JSONL: {missing}")

    return df


def filter_for_r(df: pd.DataFrame, r: float, r_tol: float = 1e-3) -> pd.DataFrame:
    """Filter dataframe for given r (with tolerance)."""
    mask = df["r"].sub(r).abs() <= r_tol
    filtered = df[mask].copy()
    if filtered.empty:
        raise ValueError(f"No rows found with r≈{r}")
    return filtered


def aggregate_tokens_per_sec(df: pd.DataFrame) -> pd.DataFrame:
    """
    Average tokens/s in case of duplicates for same (block_size, ngram_size, K).
    """
    return (
        df.groupby(["block_size", "ngram_size", "K"], as_index=False)["avg_toks_per_sec"]
        .mean()
        .rename(columns={"avg_toks_per_sec": "toks_per_sec"})
    )


def fit_quadratic_surface(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    """
    Fit z = a0 + a1*x + a2*y + a3*x*y + a4*x^2 + a5*y^2 using least squares.
    Returns coeffs [a0..a5].
    """
    A = np.column_stack(
        [
            np.ones_like(x),
            x,
            y,
            x * y,
            x ** 2,
            y ** 2,
        ]
    )
    coeffs, *_ = np.linalg.lstsq(A, z, rcond=None)
    return coeffs


def eval_quadratic_surface(coeffs: np.ndarray, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    a0, a1, a2, a3, a4, a5 = coeffs
    return a0 + a1 * X + a2 * Y + a3 * X * Y + a4 * X ** 2 + a5 * Y ** 2


def plot_best_fit_for_all_K(df_agg: pd.DataFrame, out_dir: Path, r: float):
    """
    For each K:
      - fit quadratic surface
      - save 3D surface plot:   best_fit_surface_K{K}_r{r}.png
      - save 2D contour plot:   best_fit_contour_K{K}_r{r}.png

    Also:
      - overlay 3D surface for all K  -> best_fit_surface_overlay_r{r}.png
      - overlay 2D contour for all K -> best_fit_contour_overlay_r{r}.png
    """
    Ks = sorted(df_agg["K"].unique())
    if not Ks:
        print("[INFO] No K values found after aggregation.")
        return

    # Global grid so all K plots are comparable in x/y range
    x_min, x_max = df_agg["block_size"].min(), df_agg["block_size"].max()
    y_min, y_max = df_agg["ngram_size"].min(), df_agg["ngram_size"].max()

    x_grid = np.linspace(x_min, x_max, 60)
    y_grid = np.linspace(y_min, y_max, 60)
    Xg, Yg = np.meshgrid(x_grid, y_grid)

    # Overlay figures (one 3D, one 2D) with multi-K
    fig_overlay_3d = plt.figure()
    ax_overlay_3d = fig_overlay_3d.add_subplot(111, projection="3d")

    fig_overlay_2d = plt.figure()
    ax_overlay_2d = fig_overlay_2d.add_subplot(111)

    cmap = plt.get_cmap("Blues_r")  # smaller K index -> darker, larger -> lighter
    from matplotlib.lines import Line2D
    legend_handles = []

    nK = len(Ks)
    denom = max(1, nK - 1)

    any_overlay_plotted = False

    for i, K in enumerate(Ks):
        df_k = df_agg[df_agg["K"] == K]
        if len(df_k) < 6:
            print(f"[WARN] Not enough points to fit a reliable surface for K={K} (have {len(df_k)}), skipping.")
            continue

        x = df_k["block_size"].to_numpy(dtype=float)
        y = df_k["ngram_size"].to_numpy(dtype=float)
        z = df_k["toks_per_sec"].to_numpy(dtype=float)

        coeffs = fit_quadratic_surface(x, y, z)
        Zg = eval_quadratic_surface(coeffs, Xg, Yg)

        # ---------- individual 3D surface ----------
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        surf = ax.plot_surface(Xg, Yg, Zg, alpha=0.8, cmap="viridis")
        ax.scatter(x, y, z, s=35, color="k")

        ax.set_xlabel("Block size (n_token_seq_len)")
        ax.set_ylabel("N-gram size")
        ax.set_zlabel("Tokens / second")
        ax.set_title(f"Best-fit surface (K={K}, r={r})")

        cbar = fig.colorbar(surf, ax=ax, shrink=0.7)
        cbar.set_label("Tokens / second")

        ax.view_init(elev=25, azim=-135)
        plt.tight_layout()

        out_surface = out_dir / f"best_fit_surface_K{K}_r{r}.png"
        plt.savefig(out_surface)
        plt.close(fig)
        print(f"[PLOT] Saved 3D best-fit surface to {out_surface}")

        # ---------- individual 2D contour ----------
        fig2 = plt.figure()
        cs = plt.contourf(Xg, Yg, Zg, levels=20)
        plt.scatter(x, y, c="k", s=20)

        plt.xlabel("Block size (n_token_seq_len)")
        plt.ylabel("N-gram size")
        plt.title(f"Best-fit surface (contour, K={K}, r={r})")

        cbar2 = plt.colorbar(cs)
        cbar2.set_label("Tokens / second")

        plt.tight_layout()
        out_contour = out_dir / f"best_fit_contour_K{K}_r{r}.png"
        plt.savefig(out_contour)
        plt.close(fig2)
        print(f"[PLOT] Saved 2D best-fit contour to {out_contour}")

        # ---------- overlay contributions ----------
        # Color for this K (smaller K -> darker)
        t = i / denom
        color = cmap(t)

        # 3D overlay: uniform-colored surface + scatter
        ax_overlay_3d.plot_surface(
            Xg,
            Yg,
            Zg,
            alpha=0.4,
            color=color,
            linewidth=0,
            antialiased=True,
        )
        ax_overlay_3d.scatter(x, y, z, color=color, s=20)

        # 2D overlay: contour lines + scatter
        zmin, zmax = Zg.min(), Zg.max()
        if zmin != zmax:
            levels = np.linspace(zmin, zmax, 7)
            ax_overlay_2d.contour(
                Xg,
                Yg,
                Zg,
                levels=levels,
                colors=[color],
                alpha=0.7,
            )
        ax_overlay_2d.scatter(x, y, color=color, s=20, edgecolor="none")

        legend_handles.append(
            Line2D([0], [0], color=color, lw=2, label=f"K={K}")
        )

        any_overlay_plotted = True
        print(f"[INFO] Finished K={K} with {len(df_k)} points.")

    # Finalize & save overlay figures
    if any_overlay_plotted:
        # 3D overlay
        ax_overlay_3d.set_xlabel("Block size (n_token_seq_len)")
        ax_overlay_3d.set_ylabel("N-gram size")
        ax_overlay_3d.set_zlabel("Tokens / second")
        ax_overlay_3d.set_title(f"Overlay best-fit surfaces across K (r={r})")
        ax_overlay_3d.view_init(elev=25, azim=-135)
        plt.tight_layout()
        out_overlay_surface = out_dir / f"best_fit_surface_overlay_r{r}.png"
        fig_overlay_3d.savefig(out_overlay_surface)
        plt.close(fig_overlay_3d)
        print(f"[PLOT] Saved overlay 3D best-fit surfaces to {out_overlay_surface}")

        # 2D overlay
        ax_overlay_2d.set_xlabel("Block size (n_token_seq_len)")
        ax_overlay_2d.set_ylabel("N-gram size")
        ax_overlay_2d.set_title(f"Overlay best-fit contours across K (r={r})")
        if legend_handles:
            ax_overlay_2d.legend(handles=legend_handles, title="K")
        ax_overlay_2d.grid(True, alpha=0.2)
        plt.tight_layout()
        out_overlay_contour = out_dir / f"best_fit_contour_overlay_r{r}.png"
        fig_overlay_2d.savefig(out_overlay_contour)
        plt.close(fig_overlay_2d)
        print(f"[PLOT] Saved overlay 2D best-fit contours to {out_overlay_contour}")
    else:
        plt.close(fig_overlay_3d)
        plt.close(fig_overlay_2d)
        print("[WARN] No valid K slices to plot overlays.")


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
    parser.add_argument(
        "--r",
        type=float,
        default=0.85,
        help="Fixed r value to filter on (default: 0.85)",
    )
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
    df_r = filter_for_r(df, r=args.r, r_tol=args.r_tol)
    df_agg = aggregate_tokens_per_sec(df_r)

    plot_best_fit_for_all_K(df_agg, out_dir=out_dir, r=args.r)

    print("[DONE]")


if __name__ == "__main__":
    main()

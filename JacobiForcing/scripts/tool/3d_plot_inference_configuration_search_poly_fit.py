#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from scipy.interpolate import griddata


def load_data(jsonl_path: Path) -> pd.DataFrame:
    df = pd.read_json(jsonl_path, lines=True)

    required_cols = ["K", "r", "block_size", "ngram_size", "avg_toks_per_sec"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in JSONL: {missing}")

    return df


def filter_for_r(df: pd.DataFrame, r: float, r_tol: float = 1e-3) -> pd.DataFrame:
    """Filter dataframe for given r (with tolerance)."""
    mask = (df["r"].sub(r).abs() <= r_tol)
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


# ---------------------------
# Polynomial surface fitting (only for choosing best deg)
# ---------------------------

def build_design_matrix(x: np.ndarray, y: np.ndarray, degree: int):
    """
    Build design matrix for 2D polynomial of total degree <= degree.

    Monomials: x^i * y^j for i, j >= 0, i + j <= degree.
    Returns:
      A: (N, n_terms)
      terms: list of (i, j) exponents in same order as columns.
    """
    terms = []
    cols = []
    for i in range(degree + 1):
        for j in range(degree + 1 - i):
            terms.append((i, j))
            cols.append((x ** i) * (y ** j))
    A = np.column_stack(cols)
    return A, terms


def fit_polynomial_surface(x: np.ndarray, y: np.ndarray, z: np.ndarray, degree: int):
    """
    Fit polynomial surface of given degree with least squares.
    Returns:
      coeffs: array of shape (n_terms,)
      terms: list of (i, j) for each coefficient
      mse: mean squared error on training points
    """
    A, terms = build_design_matrix(x, y, degree)
    n_samples, n_terms = A.shape
    if n_samples < n_terms:
        raise ValueError(
            f"Not enough points ({n_samples}) for degree {degree} with {n_terms} terms"
        )

    coeffs, *_ = np.linalg.lstsq(A, z, rcond=None)
    z_hat = A @ coeffs
    mse = float(np.mean((z - z_hat) ** 2))
    return coeffs, terms, mse


def choose_best_degree_for_K(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    degree_min: int = 1,
    degree_max: int = 5,
):
    """
    For a single K, scan polynomial degrees in [degree_min, degree_max],
    return the best degree (lowest MSE) and its fit.
    """
    best = None
    for deg in range(degree_min, degree_max + 1):
        try:
            coeffs, terms, mse = fit_polynomial_surface(x, y, z, deg)
        except ValueError as e:
            print(f"[WARN] Skipping degree={deg}: {e}")
            continue

        print(f"[INFO]   degree={deg}, MSE={mse:.6g}")
        if best is None or mse < best["mse"]:
            best = {
                "degree": deg,
                "coeffs": coeffs,
                "terms": terms,
                "mse": mse,
            }

    if best is None:
        raise ValueError(
            f"Could not fit any polynomial in degree range [{degree_min}, {degree_max}]"
        )
    print(f"[BEST K-slice] degree={best['degree']}, MSE={best['mse']:.6g}")
    return best


# ---------------------------
# Interpolation + smoothing
# ---------------------------

def interpolate_surface_on_grid(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    Xg: np.ndarray,
    Yg: np.ndarray,
) -> np.ndarray:
    """
    Interpolate z(x,y) onto grid (Xg, Yg) using linear interpolation,
    fill gaps with nearest neighbor.
    """
    points = np.column_stack((x, y))

    Z_lin = griddata(points, z, (Xg, Yg), method="linear")
    Z_near = griddata(points, z, (Xg, Yg), method="nearest")

    # Where linear is NaN (outside convex hull), fall back to nearest
    Z = np.where(np.isnan(Z_lin), Z_near, Z_lin)
    return Z


def smooth_grid(Z: np.ndarray, passes: int = 1) -> np.ndarray:
    """
    Smooth a 2D grid with a simple Gaussian-like kernel
    to avoid jagged edges while not creating crazy dips.
    """
    kernel = np.array([[1, 2, 1],
                       [2, 4, 2],
                       [1, 2, 1]], dtype=float)
    kernel /= kernel.sum()

    Zs = Z.astype(float)
    for _ in range(passes):
        Z_pad = np.pad(Zs, 1, mode="edge")
        Zs = (
            kernel[0, 0] * Z_pad[:-2, :-2]
            + kernel[0, 1] * Z_pad[:-2, 1:-1]
            + kernel[0, 2] * Z_pad[:-2, 2:]
            + kernel[1, 0] * Z_pad[1:-1, :-2]
            + kernel[1, 1] * Z_pad[1:-1, 1:-1]
            + kernel[1, 2] * Z_pad[1:-1, 2:]
            + kernel[2, 0] * Z_pad[2:, :-2]
            + kernel[2, 1] * Z_pad[2:, 1:-1]
            + kernel[2, 2] * Z_pad[2:, 2:]
        )
    return Zs


# ---------------------------
# Plotting (best degree per K, interpolated surface)
# ---------------------------

def plot_best_deg_for_all_K(df_agg: pd.DataFrame, out_dir: Path, r: float):
    """
    For each K:
      - scan polynomial degrees 1–5, pick best degree by MSE
      - interpolate z(x,y) on a fine grid using linear interpolation
      - lightly smooth the grid
      - save 3D surface plot:   best_fit_surface_K{K}_r{r}_deg{deg}.png
      - save 2D contour plot:   best_fit_contour_K{K}_r{r}_deg{deg}.png

    Also:
      - overlay 3D surfaces for all K  -> best_fit_surface_overlay_r{r}_deg1to5.png
      - overlay 2D contours for all K -> best_fit_contour_overlay_r{r}_deg1to5.png
    """
    Ks = sorted(df_agg["K"].unique())
    if not Ks:
        print("[INFO] No K values found after aggregation.")
        return

    # Global grid so all K plots are comparable in x/y range
    x_min, x_max = df_agg["block_size"].min(), df_agg["block_size"].max()
    y_min, y_max = df_agg["ngram_size"].min(), df_agg["ngram_size"].max()

    x_grid = np.linspace(x_min, x_max, 120)  # denser grid for smoothness
    y_grid = np.linspace(y_min, y_max, 120)
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
            print(f"[WARN] Not enough points to fit surfaces for K={K} (have {len(df_k)}), skipping.")
            continue

        print(f"\n[INFO] Processing K={K} (points={len(df_k)})")
        x = df_k["block_size"].to_numpy(dtype=float)
        y = df_k["ngram_size"].to_numpy(dtype=float)
        z = df_k["toks_per_sec"].to_numpy(dtype=float)

        # 1) find best polynomial degree just for reporting / model complexity
        best = choose_best_degree_for_K(x, y, z, degree_min=1, degree_max=5)
        deg = best["degree"]
        mse = best["mse"]

        # 2) interpolate + smooth for actual surface (no crazy dips)
        Zg = interpolate_surface_on_grid(x, y, z, Xg, Yg)
        Zg_smooth = smooth_grid(Zg, passes=1)

        # ---------- individual 3D surface ----------
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        surf = ax.plot_surface(Xg, Yg, Zg_smooth, alpha=0.85, cmap="viridis")
        ax.scatter(x, y, z, s=35, color="k")

        ax.set_xlabel("Block size (n_token_seq_len)")
        ax.set_ylabel("N-gram size")
        ax.set_zlabel("Tokens / second")
        ax.set_title(f"Best poly deg={deg} (K={K}, r={r}, MSE={mse:.3g})")

        cbar = fig.colorbar(surf, ax=ax, shrink=0.7)
        cbar.set_label("Tokens / second")

        ax.view_init(elev=25, azim=-135)
        plt.tight_layout()

        out_surface = out_dir / f"best_fit_surface_K{K}_r{r}_deg{deg}.png"
        plt.savefig(out_surface)
        plt.close(fig)
        print(f"[PLOT] Saved 3D interpolated surface to {out_surface}")

        # ---------- individual 2D contour ----------
        fig2 = plt.figure()
        cs = plt.contourf(Xg, Yg, Zg_smooth, levels=20)
        plt.scatter(x, y, c="k", s=20)

        plt.xlabel("Block size (n_token_seq_len)")
        plt.ylabel("N-gram size")
        plt.title(f"Best poly deg={deg} (K={K}, r={r}, MSE={mse:.3g})")

        cbar2 = plt.colorbar(cs)
        cbar2.set_label("Tokens / second")

        plt.tight_layout()
        out_contour = out_dir / f"best_fit_contour_K{K}_r{r}_deg{deg}.png"
        plt.savefig(out_contour)
        plt.close(fig2)
        print(f"[PLOT] Saved 2D interpolated contour to {out_contour}")

        # ---------- overlay contributions ----------
        # Color for this K (smaller K -> darker)
        t = i / denom
        color = cmap(t)

        # 3D overlay: uniform-colored surface + scatter
        ax_overlay_3d.plot_surface(
            Xg,
            Yg,
            Zg_smooth,
            alpha=0.35,
            color=color,
            linewidth=0,
            antialiased=True,
        )
        ax_overlay_3d.scatter(x, y, z, color=color, s=20)

        # 2D overlay: contour lines + scatter
        zmin, zmax = Zg_smooth.min(), Zg_smooth.max()
        if zmin != zmax:
            levels = np.linspace(zmin, zmax, 7)
            ax_overlay_2d.contour(
                Xg,
                Yg,
                Zg_smooth,
                levels=levels,
                colors=[color],
                alpha=0.7,
            )
        ax_overlay_2d.scatter(x, y, color=color, s=20, edgecolor="none")

        legend_handles.append(
            Line2D([0], [0], color=color, lw=2, label=f"K={K} (deg={deg})")
        )

        any_overlay_plotted = True
        print(f"[INFO] Finished K={K} with best degree {deg} and MSE={mse:.3g}.")

    # Finalize & save overlay figures
    if any_overlay_plotted:
        # 3D overlay
        ax_overlay_3d.set_xlabel("Block size (n_token_seq_len)")
        ax_overlay_3d.set_ylabel("N-gram size")
        ax_overlay_3d.set_zlabel("Tokens / second")
        ax_overlay_3d.set_title(f"Overlay interpolated surfaces across K (r={r}, deg∈[1,5])")
        ax_overlay_3d.view_init(elev=25, azim=-135)
        plt.tight_layout()
        out_overlay_surface = out_dir / f"best_fit_surface_overlay_r{r}_deg1to5.png"
        fig_overlay_3d.savefig(out_overlay_surface)
        plt.close(fig_overlay_3d)
        print(f"[PLOT] Saved overlay 3D surfaces to {out_overlay_surface}")

        # 2D overlay
        ax_overlay_2d.set_xlabel("Block size (n_token_seq_len)")
        ax_overlay_2d.set_ylabel("N-gram size")
        ax_overlay_2d.set_title(f"Overlay interpolated contours across K (r={r}, deg∈[1,5])")
        if legend_handles:
            ax_overlay_2d.legend(handles=legend_handles, title="K / degree")
        ax_overlay_2d.grid(True, alpha=0.2)
        plt.tight_layout()
        out_overlay_contour = out_dir / f"best_fit_contour_overlay_r{r}_deg1to5.png"
        fig_overlay_2d.savefig(out_overlay_contour)
        plt.close(fig_overlay_2d)
        print(f"[PLOT] Saved overlay 2D contours to {out_overlay_contour}")
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

    plot_best_deg_for_all_K(df_agg, out_dir=out_dir, r=args.r)

    print("[DONE]")


if __name__ == "__main__":
    main()

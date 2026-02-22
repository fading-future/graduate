import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _setup_style():
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#333333",
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 9,
            "grid.alpha": 0.25,
            "font.family": "DejaVu Sans",
        }
    )


def _load_inputs(eval_dir: Path):
    summary_path = eval_dir / "summary.json"
    per_sample_path = eval_dir / "per_sample_metrics.csv"
    phi_cell_path = eval_dir / "phi_cell_metrics.csv"
    phi_cell_summary_path = eval_dir / "phi_cell_summary.csv"

    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary: {summary_path}")
    if not per_sample_path.exists():
        raise FileNotFoundError(f"Missing per-sample csv: {per_sample_path}")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    per_sample = pd.read_csv(per_sample_path)
    phi_cell = pd.read_csv(phi_cell_path) if phi_cell_path.exists() else None
    phi_cell_summary = pd.read_csv(phi_cell_summary_path) if phi_cell_summary_path.exists() else None
    return summary, per_sample, phi_cell, phi_cell_summary


def _short_name(name: str, max_len: int = 28) -> str:
    stem = os.path.splitext(os.path.basename(name))[0]
    if len(stem) <= max_len:
        return stem
    return stem[: max_len - 3] + "..."


def plot_overview(summary: Dict, df: pd.DataFrame, out_path: Path):
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.28, wspace=0.22)

    # A: key metrics as mean +/- std
    ax = fig.add_subplot(gs[0, 0])
    metric_cfg = [
        ("voxel_dice", "Dice"),
        ("voxel_iou", "IoU"),
        ("pore_porosity_abs_err", "Pore|Err|"),
        ("bin_phi_mae", "Phi MAE"),
        ("tp2_corr", "TP2 Corr"),
    ]
    labels, means, stds = [], [], []
    for key, label in metric_cfg:
        if key in df.columns:
            vals = pd.to_numeric(df[key], errors="coerce").dropna().values
            if len(vals) > 0:
                labels.append(label)
                means.append(float(np.mean(vals)))
                stds.append(float(np.std(vals)))
    if labels:
        x = np.arange(len(labels))
        ax.errorbar(
            x, means, yerr=stds, fmt="o", color="#1f77b4", ecolor="#1f77b4",
            capsize=5, elinewidth=1.2, markersize=7
        )
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_title("Key Metrics (mean +/- std)")
    ax.set_ylabel("Value")

    # B: pore porosity parity
    ax = fig.add_subplot(gs[0, 1])
    xk = "pore_porosity_gt_aligned" if "pore_porosity_gt_aligned" in df.columns else "porosity_gt"
    yk = "pore_porosity_pred_aligned" if "pore_porosity_pred_aligned" in df.columns else "porosity_pred"
    if xk in df.columns and yk in df.columns:
        x = pd.to_numeric(df[xk], errors="coerce").values
        y = pd.to_numeric(df[yk], errors="coerce").values
        m = np.isfinite(x) & np.isfinite(y)
        x = x[m]
        y = y[m]
        names = df.loc[m, "basename"].astype(str).tolist() if "basename" in df.columns else [f"s{i}" for i in range(len(x))]
        ax.scatter(x, y, s=55, alpha=0.9, color="#2a9d8f", edgecolors="black", linewidths=0.4)
        if len(x) > 0:
            lo = float(min(np.min(x), np.min(y)))
            hi = float(max(np.max(x), np.max(y)))
            pad = 0.02 * (hi - lo + 1e-9)
            ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "--", color="#666666", linewidth=1.0)
            ax.set_xlim(lo - pad, hi + pad)
            ax.set_ylim(lo - pad, hi + pad)
            for xi, yi, nm in zip(x, y, names):
                ax.text(xi, yi, _short_name(nm, 18), fontsize=8, alpha=0.8)
    ax.set_title("Parity: Pore Fraction (Pred vs GT)")
    ax.set_xlabel("GT")
    ax.set_ylabel("Pred")

    # C: per-sample metric bars
    ax = fig.add_subplot(gs[1, 0])
    show = []
    for c in ["voxel_dice", "voxel_iou", "bin_phi_corr", "pore_porosity_abs_err", "tp2_corr"]:
        if c in df.columns:
            show.append(c)
    if show:
        tmp = df[show].apply(pd.to_numeric, errors="coerce")
        x = np.arange(len(df))
        width = max(0.12, 0.8 / max(1, len(show)))
        for i, c in enumerate(show):
            ax.bar(x + (i - (len(show) - 1) / 2) * width, tmp[c].values, width=width, label=c, alpha=0.9)
        ax.set_xticks(x)
        labels = [_short_name(n, 12) for n in df["basename"]] if "basename" in df.columns else [str(i) for i in range(len(df))]
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.legend(loc="best", ncols=2)
    ax.set_title("Per-sample Metric Profile")
    ax.set_ylabel("Value")

    # D: runtime
    ax = fig.add_subplot(gs[1, 1])
    if "time_sec" in df.columns:
        t = pd.to_numeric(df["time_sec"], errors="coerce").values
        x = np.arange(len(t))
        ax.bar(x, t, color="#e76f51", alpha=0.9)
        ax.plot(x, t, color="#b23b2a", linewidth=1.2)
        ax.set_xticks(x)
        labels = [_short_name(n, 12) for n in df["basename"]] if "basename" in df.columns else [str(i) for i in range(len(df))]
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.set_ylabel("Seconds")
    ax.set_title("Runtime per Sample")

    n = int(summary.get("num_samples", len(df)))
    title = f"Batch Evaluation Overview (N={n})"
    fig.suptitle(title, fontsize=14, y=0.98)
    fig.subplots_adjust(left=0.055, right=0.985, bottom=0.085, top=0.93, wspace=0.23, hspace=0.30)
    fig.savefig(out_path, dpi=260)
    plt.close(fig)


def _cube_from_cell_summary(phi_cell_summary: pd.DataFrame, key: str) -> Tuple[np.ndarray, int, int, int]:
    g_d = int(phi_cell_summary["cell_i"].max()) + 1
    g_h = int(phi_cell_summary["cell_j"].max()) + 1
    g_w = int(phi_cell_summary["cell_k"].max()) + 1
    arr = np.full((g_d, g_h, g_w), np.nan, dtype=np.float64)
    for _, r in phi_cell_summary.iterrows():
        i, j, k = int(r["cell_i"]), int(r["cell_j"]), int(r["cell_k"])
        arr[i, j, k] = float(r[key])
    return arr, g_d, g_h, g_w


def _plot_layer_row(fig, gs_row, cube: np.ndarray, title: str, cmap: str, vmin: Optional[float], vmax: Optional[float]):
    g_d, _, _ = cube.shape
    ims = []
    for i in range(g_d):
        ax = fig.add_subplot(gs_row[i])
        im = ax.imshow(cube[i], cmap=cmap, vmin=vmin, vmax=vmax, origin="lower")
        ims.append(im)
        for yy in range(cube.shape[1]):
            for xx in range(cube.shape[2]):
                val = cube[i, yy, xx]
                if np.isfinite(val):
                    ax.text(xx, yy, f"{val:.2f}", ha="center", va="center", fontsize=7, color="black")
        ax.set_title(f"{title} | layer i={i}")
        ax.set_xlabel("k")
        ax.set_ylabel("j")
    return ims


def plot_phi_cell_layers(phi_cell_summary: pd.DataFrame, out_path: Path):
    gt, g_d, _, _ = _cube_from_cell_summary(phi_cell_summary, "gt_phi_mean")
    pred, _, _, _ = _cube_from_cell_summary(phi_cell_summary, "pred_phi_bin_mean")
    bias, _, _, _ = _cube_from_cell_summary(phi_cell_summary, "bin_bias_mean")
    mae, _, _, _ = _cube_from_cell_summary(phi_cell_summary, "bin_mae_mean")

    fig = plt.figure(figsize=(4.8 * g_d, 13))
    gs = fig.add_gridspec(4, g_d, hspace=0.36, wspace=0.28)

    ims0 = _plot_layer_row(fig, [gs[0, i] for i in range(g_d)], gt, "GT phi", "viridis", 0.0, 1.0)
    ims1 = _plot_layer_row(fig, [gs[1, i] for i in range(g_d)], pred, "Pred phi (bin)", "viridis", 0.0, 1.0)
    vmax_bias = float(np.nanmax(np.abs(bias))) if np.isfinite(bias).any() else 0.2
    ims2 = _plot_layer_row(fig, [gs[2, i] for i in range(g_d)], bias, "Bias (Pred-GT)", "coolwarm", -vmax_bias, vmax_bias)
    ims3 = _plot_layer_row(fig, [gs[3, i] for i in range(g_d)], mae, "Abs Error", "magma", 0.0, float(np.nanmax(mae)))

    for ims in [ims0, ims1, ims2, ims3]:
        if ims:
            cbar = fig.colorbar(ims[-1], ax=[im.axes for im in ims], fraction=0.015, pad=0.01)
            cbar.ax.tick_params(labelsize=8)

    fig.suptitle("Cell-level Phi Comparison by Layer", fontsize=14, y=0.995)
    fig.subplots_adjust(left=0.04, right=0.985, bottom=0.04, top=0.95, wspace=0.30, hspace=0.38)
    fig.savefig(out_path, dpi=260)
    plt.close(fig)


def plot_phi_cell_parity(phi_cell: pd.DataFrame, out_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))

    x = pd.to_numeric(phi_cell["gt_phi"], errors="coerce").values
    y = pd.to_numeric(phi_cell["pred_phi_bin"], errors="coerce").values
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]
    y = y[m]
    axes[0].scatter(x, y, s=20, alpha=0.6, color="#264653", edgecolors="none")
    lo = float(min(np.min(x), np.min(y))) if len(x) else 0.0
    hi = float(max(np.max(x), np.max(y))) if len(x) else 1.0
    axes[0].plot([lo, hi], [lo, hi], "--", color="#777777", linewidth=1.1)
    axes[0].set_title("Cell Parity: Pred phi(bin) vs GT")
    axes[0].set_xlabel("GT phi")
    axes[0].set_ylabel("Pred phi(bin)")
    axes[0].set_xlim(lo, hi)
    axes[0].set_ylim(lo, hi)

    err = pd.to_numeric(phi_cell["bin_abs_err"], errors="coerce").dropna().values
    axes[1].hist(err, bins=18, color="#f4a261", edgecolor="black", alpha=0.9)
    axes[1].axvline(float(np.mean(err)), color="#d62728", linestyle="--", linewidth=1.6, label=f"mean={np.mean(err):.3f}")
    axes[1].set_title("Distribution: Cell Absolute Error")
    axes[1].set_xlabel("|Pred-GT|")
    axes[1].set_ylabel("Count")
    axes[1].legend(loc="best")

    fig.tight_layout()
    fig.savefig(out_path, dpi=260)
    plt.close(fig)


def plot_physics_diag(df: pd.DataFrame, out_path: Path):
    axes_list = []
    for a in "xyz":
        if f"kabs_pred_{a}_ok" in df.columns or f"kabs_gt_{a}_ok" in df.columns:
            axes_list.append(a)
    if not axes_list:
        return False

    fig = plt.figure(figsize=(13, 5.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 1.0], wspace=0.28)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])

    # Success rates
    x = np.arange(len(axes_list))
    pred_ok = []
    gt_ok = []
    for a in axes_list:
        pk = pd.to_numeric(df.get(f"kabs_pred_{a}_ok", np.nan), errors="coerce").dropna().values
        gk = pd.to_numeric(df.get(f"kabs_gt_{a}_ok", np.nan), errors="coerce").dropna().values
        pred_ok.append(float(pk.mean()) if len(pk) else np.nan)
        gt_ok.append(float(gk.mean()) if len(gk) else np.nan)
    w = 0.36
    ax0.bar(x - w / 2, pred_ok, width=w, label="Pred network solve success", color="#2a9d8f")
    ax0.bar(x + w / 2, gt_ok, width=w, label="GT network solve success", color="#457b9d")
    ax0.set_xticks(x)
    ax0.set_xticklabels(axes_list)
    ax0.set_ylim(0.0, 1.05)
    ax0.set_ylabel("Success rate")
    ax0.set_title("OpenPNM Solve Success")
    ax0.legend(loc="best")

    # Pred vs GT permeability parity for successful pairs
    xs = []
    ys = []
    for _, r in df.iterrows():
        for a in axes_list:
            kp = r.get(f"kabs_pred_{a}", np.nan)
            kg = r.get(f"kabs_gt_{a}", np.nan)
            if np.isfinite(kp) and np.isfinite(kg):
                xs.append(float(kg))
                ys.append(float(kp))
    if len(xs) > 0:
        xs = np.array(xs, dtype=np.float64)
        ys = np.array(ys, dtype=np.float64)
        ax1.scatter(xs, ys, s=34, alpha=0.85, color="#1d3557")
        lo = float(min(xs.min(), ys.min()))
        hi = float(max(xs.max(), ys.max()))
        ax1.plot([lo, hi], [lo, hi], "--", color="#666666", linewidth=1.1)
        ax1.set_xlim(lo, hi)
        ax1.set_ylim(lo, hi)
    ax1.set_title("Absolute Permeability Parity")
    ax1.set_xlabel("GT k_abs (voxel^2)")
    ax1.set_ylabel("Pred k_abs (voxel^2)")

    fig.subplots_adjust(left=0.07, right=0.985, bottom=0.12, top=0.92, wspace=0.30)
    fig.savefig(out_path, dpi=260)
    plt.close(fig)
    return True


def save_markdown_report(summary: Dict, out_path: Path):
    lines: List[str] = []
    lines.append("# Batch Evaluation Report")
    lines.append("")
    lines.append("## Core Metrics")
    keys = [
        "num_samples",
        "voxel_dice_mean",
        "voxel_iou_mean",
        "pore_porosity_abs_err_mean",
        "pore_parity_corr",
        "pore_parity_slope",
        "pore_parity_bias",
        "pore_phi_mae_mean",
        "pore_phi_corr_mean",
        "tp2_corr_mean",
        "tp2_mae_mean",
        "time_sec_mean",
    ]
    for k in keys:
        if k in summary:
            v = summary[k]
            if isinstance(v, (int, float)):
                lines.append(f"- `{k}`: {v:.6f}")
            else:
                lines.append(f"- `{k}`: {v}")
    lines.append("")
    lines.append("## Physics Notes")
    for k in ["kabs_pred_x_ok_mean", "kabs_pred_y_ok_mean", "kabs_pred_z_ok_mean", "kabs_gt_x_ok_mean", "kabs_gt_y_ok_mean", "kabs_gt_z_ok_mean"]:
        if k in summary:
            lines.append(f"- `{k}`: {summary[k]:.6f}")
    lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Create publication-style visualizations for eval_batch outputs.")
    parser.add_argument("--eval-dir", required=True, help="Path to eval batch directory")
    parser.add_argument("--out-dir", default="", help="Output directory (default: <eval-dir>/figures)")
    args = parser.parse_args()

    eval_dir = Path(args.eval_dir).resolve()
    out_dir = Path(args.out_dir).resolve() if args.out_dir else (eval_dir / "figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    _setup_style()
    summary, per_sample, phi_cell, phi_cell_summary = _load_inputs(eval_dir)

    plot_overview(summary, per_sample, out_dir / "overview_panel.png")
    if phi_cell_summary is not None and len(phi_cell_summary) > 0:
        plot_phi_cell_layers(phi_cell_summary, out_dir / "phi_cell_layers.png")
    if phi_cell is not None and len(phi_cell) > 0:
        plot_phi_cell_parity(phi_cell, out_dir / "phi_cell_parity.png")
    _ = plot_physics_diag(per_sample, out_dir / "physics_diag.png")
    save_markdown_report(summary, out_dir / "report.md")

    print("[done] visualization generated")
    print(f"  out_dir: {out_dir}")
    for fn in ["overview_panel.png", "phi_cell_layers.png", "phi_cell_parity.png", "physics_diag.png", "report.md"]:
        p = out_dir / fn
        if p.exists():
            print(f"  - {p}")


if __name__ == "__main__":
    main()

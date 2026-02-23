#!/usr/bin/env python3
"""
Plot thesis-style figures from KLVAE quantitative evaluation outputs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def apply_thesis_style():
    # ================= 论文统一绘图配置 =================
    # 1. 字体与大小设置 (适配 A4 纸张)
    # 论文插图一般分为：半栏图(width=3.5英寸) 和 通栏图(width=7英寸)
    # 这里默认设置为半栏图大小，适合并排显示或小图
    plt.rcParams["figure.figsize"] = (3.5, 2.625)  # 4:3 比例，宽 3.5 英寸
    plt.rcParams["figure.dpi"] = 300  # 高分辨率，满足打印要求

    # 2. 字体设置 (Times New Roman 是 SCI 和 毕业论文首选)
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman"]
    plt.rcParams["mathtext.fontset"] = "stix"  # 数学公式字体与 Times 接近
    plt.rcParams["font.size"] = 10.5  # 五号字 (约 10.5pt)
    plt.rcParams["axes.labelsize"] = 10.5  # 轴标签大小
    plt.rcParams["xtick.labelsize"] = 9  # 刻度标签略小
    plt.rcParams["ytick.labelsize"] = 9
    plt.rcParams["legend.fontsize"] = 9
    plt.rcParams["axes.titlesize"] = 11  # 标题略大

    # 3. 线条与布局
    plt.rcParams["axes.linewidth"] = 1.0  # 边框线宽
    plt.rcParams["lines.linewidth"] = 1.5  # 折线线宽
    plt.rcParams["grid.alpha"] = 0.3  # 网格透明度
    plt.rcParams["axes.grid"] = True  # 默认开启网格，方便读数
    plt.rcParams["xtick.direction"] = "in"  # 刻度向内 (很多期刊偏好)
    plt.rcParams["ytick.direction"] = "in"

    # 4. 推荐配色方案 (Hex Codes)
    # 这一组颜色专为学术图表设计：
    # - 对比度高 (适合黑白打印也能分清)
    # - 沉稳不刺眼 (避免大红大绿)
    # - 包含 8 种颜色，足够多曲线使用
    thesis_colors = [
        "#0072B2",  # 深蓝 (主数据/模型A)
        "#D55E00",  # 赭红 (对比数据/模型B - 强对比)
        "#E69F00",  # 橙黄 (辅助数据)
        "#009E73",  # 蓝绿 (地质属性常用)
        "#56B4E9",  # 天蓝 (浅色背景或次要线)
        "#F0E442",  # 亮黄 (尽量少用，或用于高亮)
        "#CC79A7",  # 紫粉 (适合第四/五个变量)
        "#333333",  # 深灰 (基准线/Reference)
    ]

    # 设置为默认颜色循环
    plt.rcParams["axes.prop_cycle"] = plt.cycler(color=thesis_colors)
    # ================= 配置结束 =================


def _save(fig, out_path: Path):
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_porosity_parity(df: pd.DataFrame, out_path: Path):
    if not {"pore_porosity_gt", "pore_porosity_pred"}.issubset(df.columns):
        return
    x = df["pore_porosity_gt"].astype(float).to_numpy()
    y = df["pore_porosity_pred"].astype(float).to_numpy()
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if x.size == 0:
        return

    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    ax.scatter(x, y, s=16, alpha=0.75, edgecolors="none", label="Samples")
    lo = min(float(x.min()), float(y.min()))
    hi = max(float(x.max()), float(y.max()))
    pad = max(1e-6, (hi - lo) * 0.05)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "--", color="#333333", label="y=x")
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlabel("GT pore porosity")
    ax.set_ylabel("Pred pore porosity")
    ax.set_title("Porosity Parity")

    if x.size > 2 and x.std() > 1e-12:
        p = np.polyfit(x, y, deg=1)
        ax.plot([lo - pad, hi + pad], np.polyval(p, [lo - pad, hi + pad]), color="#D55E00", label=f"Fit: y={p[0]:.2f}x+{p[1]:.2f}")
        corr = float(np.corrcoef(x, y)[0, 1])
        ax.text(0.03, 0.97, f"r={corr:.3f}", ha="left", va="top", transform=ax.transAxes)
    ax.legend(frameon=True, loc="lower right")
    _save(fig, out_path)


def plot_metric_box(df: pd.DataFrame, out_path: Path):
    metric_cols = ["voxel_dice", "pore_porosity_abs_err", "phi16_pore_mae", "tp2_mae"]
    cols = [c for c in metric_cols if c in df.columns]
    if not cols:
        return
    long_df = df[cols].melt(var_name="Metric", value_name="Value")
    long_df = long_df.replace([np.inf, -np.inf], np.nan).dropna()
    if long_df.empty:
        return

    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    sns.boxplot(data=long_df, x="Metric", y="Value", ax=ax, width=0.62, fliersize=2.0)
    ax.set_xlabel("")
    ax.set_ylabel("Value")
    ax.set_title("Core Metrics Distribution")
    ax.tick_params(axis="x", rotation=20)
    _save(fig, out_path)


def plot_phi_multiscale(summary: Dict, out_path: Path):
    scales = [8, 16, 32, 64]
    means: List[float] = []
    stds: List[float] = []
    xs: List[int] = []
    for s in scales:
        mk = f"phi{s}_pore_mae_mean"
        sk = f"phi{s}_pore_mae_std"
        if mk in summary:
            xs.append(s)
            means.append(float(summary[mk]))
            stds.append(float(summary.get(sk, 0.0)))
    if not xs:
        return

    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    ax.errorbar(xs, means, yerr=stds, fmt="o-", capsize=3, label="MAE ± std")
    ax.set_xlabel("Block size (voxel)")
    ax.set_ylabel("Pore-phi MAE")
    ax.set_title("Multi-scale Porosity Error")
    ax.set_xticks(xs)
    ax.legend(frameon=True)
    _save(fig, out_path)


def plot_tp2_curve(tp2_npz: Path, out_path: Path):
    if not tp2_npz.exists():
        return
    data = np.load(tp2_npz)
    lag = data["lag"].astype(np.int32)
    pm = data["pred_mean"].astype(np.float64)
    ps = data["pred_std"].astype(np.float64)
    gm = data["gt_mean"].astype(np.float64)
    gs = data["gt_std"].astype(np.float64)

    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    ax.plot(lag, gm, label="GT S2 mean", color="#0072B2")
    ax.fill_between(lag, gm - gs, gm + gs, color="#0072B2", alpha=0.15)
    ax.plot(lag, pm, label="Recon S2 mean", color="#D55E00")
    ax.fill_between(lag, pm - ps, pm + ps, color="#D55E00", alpha=0.15)
    ax.set_xlabel("Lag r (voxel)")
    ax.set_ylabel("S2(r)")
    ax.set_title("Two-Point Probability")
    ax.legend(frameon=True)
    _save(fig, out_path)


def _collect_kabs_pairs(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Prefer per-axis columns because they are available even when mean columns are absent.
    x_parts: List[np.ndarray] = []
    y_parts: List[np.ndarray] = []
    tag_parts: List[np.ndarray] = []

    for ax_tag in ("x", "y", "z"):
        gcol = f"kabs_gt_{ax_tag}"
        pcol = f"kabs_pred_{ax_tag}"
        if not {gcol, pcol}.issubset(df.columns):
            continue
        gx = pd.to_numeric(df[gcol], errors="coerce").to_numpy(dtype=np.float64)
        py = pd.to_numeric(df[pcol], errors="coerce").to_numpy(dtype=np.float64)
        m = np.isfinite(gx) & np.isfinite(py) & (gx > 0.0) & (py > 0.0)
        if np.any(m):
            x_parts.append(gx[m])
            y_parts.append(py[m])
            tag_parts.append(np.full(int(np.sum(m)), ax_tag, dtype=object))

    if x_parts:
        return np.concatenate(x_parts), np.concatenate(y_parts), np.concatenate(tag_parts)

    # Fallback for legacy outputs that only have mean columns.
    if {"kabs_gt_mean", "kabs_pred_mean"}.issubset(df.columns):
        x = pd.to_numeric(df["kabs_gt_mean"], errors="coerce").to_numpy(dtype=np.float64)
        y = pd.to_numeric(df["kabs_pred_mean"], errors="coerce").to_numpy(dtype=np.float64)
        m = np.isfinite(x) & np.isfinite(y) & (x > 0.0) & (y > 0.0)
        if np.any(m):
            return x[m], y[m], np.full(int(np.sum(m)), "mean", dtype=object)

    return np.array([], dtype=np.float64), np.array([], dtype=np.float64), np.array([], dtype=object)


def plot_kabs_parity(df: pd.DataFrame, out_path: Path) -> bool:
    x, y, tags = _collect_kabs_pairs(df)
    if x.size < 3:
        return False

    lx, ly = np.log10(x), np.log10(y)
    fig, ax = plt.subplots(figsize=(3.5, 2.8))

    color_map = {"x": "#0072B2", "y": "#D55E00", "z": "#009E73", "mean": "#0072B2"}
    for ax_tag in ("x", "y", "z", "mean"):
        m = tags == ax_tag
        if not np.any(m):
            continue
        ax.scatter(
            lx[m],
            ly[m],
            s=16,
            alpha=0.75,
            edgecolors="none",
            color=color_map[ax_tag],
            label=f"{ax_tag.upper()} (n={int(np.sum(m))})",
        )

    lo = min(float(lx.min()), float(ly.min()))
    hi = max(float(lx.max()), float(ly.max()))
    pad = max(1e-6, (hi - lo) * 0.05)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "--", color="#333333", label="y=x")
    ax.set_xlabel(r"$\log_{10}(k_{abs}^{GT})$")
    ax.set_ylabel(r"$\log_{10}(k_{abs}^{Recon})$")
    ax.set_title("Absolute Permeability Parity")
    corr = float(np.corrcoef(lx, ly)[0, 1]) if lx.size > 2 else 0.0
    ax.text(0.03, 0.97, f"r={corr:.3f}", ha="left", va="top", transform=ax.transAxes)
    ax.legend(frameon=True, loc="lower right")
    _save(fig, out_path)
    return True


def plot_kabs_success_rate(df: pd.DataFrame, out_path: Path) -> bool:
    rows = []
    for phase in ("gt", "pred"):
        label = "GT" if phase == "gt" else "Recon"
        for ax_tag in ("x", "y", "z"):
            col = f"kabs_{phase}_{ax_tag}_ok"
            if col not in df.columns:
                continue
            vals = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=np.float64)
            m = np.isfinite(vals)
            if not np.any(m):
                continue
            rows.append(
                {
                    "Axis": ax_tag.upper(),
                    "Data": label,
                    "SuccessRate": float(np.mean(np.clip(vals[m], 0.0, 1.0))),
                    "N": int(np.sum(m)),
                }
            )

    if not rows:
        return False

    plot_df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    sns.barplot(data=plot_df, x="Axis", y="SuccessRate", hue="Data", ax=ax)
    ax.set_ylim(0.0, 1.02)
    ax.set_ylabel("Solver success rate")
    ax.set_title("Absolute Permeability Solver Success")
    ax.legend(frameon=True, loc="upper right")
    _save(fig, out_path)
    return True


def main():
    apply_thesis_style()

    parser = argparse.ArgumentParser(description="Plot thesis-ready KLVAE evaluation figures.")
    parser.add_argument("--eval-dir", default=str(Path(__file__).resolve().parent / "experiments" / "exp04_cube_structure_v1" / "eval_klvae_192"))
    parser.add_argument("--use-subset", action="store_true", help="Use subset files (recommended for your thesis figures).")
    parser.add_argument("--out-dir", default="")
    args = parser.parse_args()

    eval_dir = Path(args.eval_dir)
    out_dir = Path(args.out_dir) if args.out_dir else (eval_dir / "figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_name = "per_sample_metrics_subset.csv" if args.use_subset else "per_sample_metrics.csv"
    summary_name = "summary_subset.json" if args.use_subset else "summary_all.json"
    csv_path = eval_dir / csv_name
    summary_path = eval_dir / summary_name
    tp2_path = eval_dir / "tp2_curve_aggregate.npz"

    if not csv_path.exists():
        raise FileNotFoundError(f"metrics csv not found: {csv_path}")
    if not summary_path.exists():
        raise FileNotFoundError(f"summary json not found: {summary_path}")

    df = pd.read_csv(csv_path)
    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    plot_porosity_parity(df, out_dir / "fig_porosity_parity.png")
    plot_metric_box(df, out_dir / "fig_core_metric_boxplot.png")
    plot_phi_multiscale(summary, out_dir / "fig_phi_multiscale_mae.png")
    plot_tp2_curve(tp2_path, out_dir / "fig_tp2_mean_curve.png")
    has_kabs_success = plot_kabs_success_rate(df, out_dir / "fig_kabs_success_rate.png")
    has_kabs_parity = plot_kabs_parity(df, out_dir / "fig_kabs_parity.png")

    # Export concise table for direct thesis usage.
    rows = []
    for k in [
        "voxel_dice_mean",
        "voxel_iou_mean",
        "pore_porosity_abs_err_mean",
        "phi16_pore_mae_mean",
        "phi16_pore_corr_mean",
        "tp2_corr_mean",
        "tp2_mae_mean",
        "kabs_gt_x_ok_mean",
        "kabs_gt_y_ok_mean",
        "kabs_gt_z_ok_mean",
        "kabs_pred_x_ok_mean",
        "kabs_pred_y_ok_mean",
        "kabs_pred_z_ok_mean",
        "kabs_abs_err_mean_mean",
        "kabs_rel_err_mean_mean",
        "time_sec_mean",
    ]:
        if k in summary:
            rows.append({"metric": k, "value": float(summary[k])})
    pd.DataFrame(rows).to_csv(out_dir / "table_main_metrics.csv", index=False)

    print("[done] thesis figures generated")
    print(f"  eval_dir: {eval_dir}")
    print(f"  out_dir:  {out_dir}")
    if not has_kabs_success:
        print("  [info] skip fig_kabs_success_rate.png (no kabs_*_ok columns)")
    if not has_kabs_parity:
        print("  [info] skip fig_kabs_parity.png (insufficient valid GT/Recon kabs pairs)")


if __name__ == "__main__":
    main()

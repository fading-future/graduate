import argparse
import csv
import os
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def apply_thesis_style():
    # ================= 论文统一绘图配置 =================
    plt.rcParams["figure.dpi"] = 300  # 高分辨率，满足打印要求
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman"]
    plt.rcParams["mathtext.fontset"] = "stix"  # 数学公式字体与 Times 接近
    plt.rcParams["font.size"] = 10.5  # 五号字 (约 10.5pt)
    plt.rcParams["axes.labelsize"] = 10.5  # 轴标签大小
    plt.rcParams["xtick.labelsize"] = 9  # 刻度标签略小
    plt.rcParams["ytick.labelsize"] = 9
    plt.rcParams["legend.fontsize"] = 9
    plt.rcParams["axes.titlesize"] = 11  # 标题略大

    # 线条与布局
    plt.rcParams["axes.linewidth"] = 1.0  # 边框线宽
    plt.rcParams["lines.linewidth"] = 1.5  # 折线线宽
    plt.rcParams["grid.alpha"] = 0.3  # 网格透明度
    plt.rcParams["axes.grid"] = True  # 默认开启网格，方便读数
    plt.rcParams["xtick.direction"] = "in"  # 刻度向内 (很多期刊偏好)
    plt.rcParams["ytick.direction"] = "in"

    # 推荐配色方案 (Hex Codes)
    thesis_colors = [
        "#0072B2",  # 深蓝
        "#D55E00",  # 赭红
        "#E69F00",  # 橙黄
        "#009E73",  # 蓝绿
        "#56B4E9",  # 天蓝
        "#F0E442",  # 亮黄
        "#CC79A7",  # 紫粉
        "#333333",  # 深灰
    ]
    plt.rcParams["axes.prop_cycle"] = plt.cycler(color=thesis_colors)
    # ================= 配置结束 =================


def _load_csv(path: str) -> Dict[str, np.ndarray]:
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if len(rows) == 0:
        raise ValueError("Empty log file")

    cols = reader.fieldnames or []
    out: Dict[str, np.ndarray] = {}
    for c in cols:
        vals: List[float] = []
        for r in rows:
            try:
                vals.append(float(r[c]))
            except Exception:
                vals.append(float("nan"))
        out[c] = np.asarray(vals, dtype=np.float64)
    return out


def _moving_average_nan(y: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return y.copy()
    w = int(max(1, window))
    k = np.ones(w, dtype=np.float64)
    valid = np.isfinite(y).astype(np.float64)
    y0 = np.where(np.isfinite(y), y, 0.0)
    num = np.convolve(y0, k, mode="same")
    den = np.convolve(valid, k, mode="same")
    out = np.full_like(y, np.nan, dtype=np.float64)
    m = den > 1e-12
    out[m] = num[m] / den[m]
    return out


def _plot_series(
    ax: plt.Axes,
    x: np.ndarray,
    data: Dict[str, np.ndarray],
    keys: List[str],
    smooth: int,
    logy: bool,
    title: str,
    ylabel: str,
):
    plotted = 0
    eps = 1e-12
    for k in keys:
        if k not in data:
            continue
        y = data[k]
        if np.all(~np.isfinite(y)):
            continue
        ys = _moving_average_nan(y, smooth)
        if logy:
            ys = np.where(np.isfinite(ys), np.maximum(ys, eps), np.nan)
        # 移除了 hardcode 的 linewidth=1.8，让其自动继承 apply_thesis_style 的设定
        ax.plot(x, ys, label=k)
        plotted += 1
        
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    # 移除了 hardcode 的 ax.grid(alpha=0.25)，因为 rcParams 已经设定了 grid
    
    if plotted > 0:
        ax.legend(loc="best")
    else:
        ax.text(0.5, 0.5, "No available columns", ha="center", va="center", transform=ax.transAxes)
    if logy:
        ax.set_yscale("log")


def main():
    apply_thesis_style()

    parser = argparse.ArgumentParser(description="Plot detailed Stage07 training losses in thesis style.")
    parser.add_argument("--log", required=True, help="training_log_detailed.csv path")
    parser.add_argument("--out", default="loss_detailed.png", help="output png path")
    parser.add_argument("--smooth", type=int, default=25, help="moving average window size")
    parser.add_argument("--logy", action="store_true", help="use log scale on y-axis where applicable")
    args = parser.parse_args()

    print(f"正在加载日志文件: {args.log} ... (如果文件很大，请耐心等待)") # 加这句
    data = _load_csv(args.log)
    print("数据加载完成！正在绘制图表...") # 加这句
    if "Step" in data:
        x = data["Step"]
    else:
        n = len(next(iter(data.values())))
        x = np.arange(n, dtype=np.float64)

    # 布局改为 2x2，尺寸设为 7 x 5.25 英寸 (正好是两个半栏图宽度)
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.25), sharex=True)
    axes = axes.reshape(2, 2)

    # 图 1 (左上): 总体 Loss (解决原本线太多的问题)
    _plot_series(
        axes[0, 0],
        x,
        data,
        ["Loss"],
        smooth=args.smooth,
        logy=args.logy,
        title="Total Core Loss",
        ylabel="Loss",
    )
    
    # 图 2 (右上): 其余 Core 组件 Loss (拆分出来单独展示)
    _plot_series(
        axes[0, 1],
        x,
        data,
        ["DiffTarget", "X0Target", "StatsTarget"],
        smooth=args.smooth,
        logy=args.logy,
        title="Core Loss Components",
        ylabel="Loss",
    )
    
    # 图 3 (左下): 仅保留 Weighted 的 Phi Auxiliary Loss
    _plot_series(
        axes[1, 0],
        x,
        data,
        ["PhiDecodeWeighted", "PhiProxyWeighted"],
        smooth=args.smooth,
        logy=args.logy,
        title="Phi Auxiliary Loss (Weighted)",
        ylabel="Weighted Loss",
    )
    
    # 图 4 (右下): 保留 Phi Hyperparameters
    _plot_series(
        axes[1, 1],
        x,
        data,
        ["PhiConsistencyWeight", "PhiProxyWeight"],
        smooth=1,
        logy=False,
        title="Phi Hyperparameters",
        ylabel="Value",
    )

    # 只给最下面一排加 X 轴标签
    axes[1, 0].set_xlabel("Step")
    axes[1, 1].set_xlabel("Step")
    
    fig.suptitle("Stage 07 Training Curves", fontsize=12) # 调整了总标题字号以适配整体风格
    fig.tight_layout(rect=[0, 0, 1, 0.96]) # 留出 suptitle 的空间

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=300, bbox_inches="tight") # 增加 bbox_inches 防止边缘被裁
    plt.close(fig)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
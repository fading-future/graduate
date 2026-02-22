import argparse
import csv
import os
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


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
        ax.plot(x, ys, linewidth=1.8, label=k)
        plotted += 1
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    if plotted > 0:
        ax.legend(loc="best", fontsize=8)
    else:
        ax.text(0.5, 0.5, "No available columns", ha="center", va="center", transform=ax.transAxes)
    if logy:
        ax.set_yscale("log")


def main():
    parser = argparse.ArgumentParser(description="Plot detailed Stage07 training losses.")
    parser.add_argument("--log", required=True, help="training_log_detailed.csv path")
    parser.add_argument("--out", default="loss_detailed.png", help="output png path")
    parser.add_argument("--smooth", type=int, default=25, help="moving average window size")
    parser.add_argument("--logy", action="store_true", help="use log scale on y-axis where applicable")
    args = parser.parse_args()

    data = _load_csv(args.log)
    if "Step" in data:
        x = data["Step"]
    else:
        n = len(next(iter(data.values())))
        x = np.arange(n, dtype=np.float64)

    fig, axes = plt.subplots(3, 2, figsize=(14, 11), sharex=True)
    axes = axes.reshape(3, 2)

    _plot_series(
        axes[0, 0],
        x,
        data,
        # ["Loss", "DiffTarget", "X0Target", "StatsTarget"],
        ["Loss", "DiffTarget", "StatsTarget"],
        smooth=args.smooth,
        logy=args.logy,
        title="Core Loss Terms",
        ylabel="Loss",
    )
    _plot_series(
        axes[0, 1],
        x,
        data,
        ["PhiDecode", "PhiProxy"],
        smooth=args.smooth,
        logy=args.logy,
        title="Phi Auxiliary Loss (Raw)",
        ylabel="Loss",
    )
    _plot_series(
        axes[1, 0],
        x,
        data,
        ["PhiDecodeWeighted", "PhiProxyWeighted"],
        smooth=args.smooth,
        logy=args.logy,
        title="Phi Auxiliary Loss (Weighted in Total Loss)",
        ylabel="Weighted Loss",
    )
    _plot_series(
        axes[1, 1],
        x,
        data,
        ["PhiConsistencyWeight", "PhiProxyWeight", "PhiLossEverySteps", "PhiLossMaxBatch"],
        smooth=1,
        logy=False,
        title="Phi Hyperparameters (from Log)",
        ylabel="Value",
    )
    _plot_series(
        axes[2, 0],
        x,
        data,
        ["PhiTargetMean", "PhiDecodeApplied"],
        smooth=args.smooth,
        logy=False,
        title="Phi Target / Decode Applied Flag",
        ylabel="Value",
    )
    _plot_series(
        axes[2, 1],
        x,
        data,
        ["LR"],
        smooth=1,
        logy=True,
        title="Learning Rate",
        ylabel="LR",
    )

    axes[2, 0].set_xlabel("Step")
    axes[2, 1].set_xlabel("Step")
    fig.suptitle("Stage07 Detailed Training Curves", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=220)
    plt.close(fig)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()

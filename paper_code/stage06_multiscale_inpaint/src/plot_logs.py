import os
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.utils import get_root


def read_csv(csv_path: str):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(csv_path)
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"CSV is empty: {csv_path}")
    return df


def smooth(series, window=50):
    if window <= 1:
        return series
    return series.rolling(window=window, min_periods=1).mean()


def plot_one(csv_path, out_path, title, comp_cols, smooth_window=50, logy=True):
    df = read_csv(csv_path)

    steps = df["Step"]
    loss = df["Loss"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=150, constrained_layout=True)
    ax1, ax2 = axes

    ax1.plot(steps, loss, alpha=0.15, linewidth=0.7, label="Loss (Raw)")
    ax1.plot(steps, smooth(loss, smooth_window), linewidth=2.0, label=f"Loss (Smoothed, w={smooth_window})")
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Loss")
    ax1.grid(True, which="both", ls="--", alpha=0.3)
    if logy:
        ax1.set_yscale("log")

    if "LR" in df.columns:
        ax1b = ax1.twinx()
        ax1b.plot(steps, df["LR"], linestyle="--", linewidth=1.4, label="LR")
        ax1b.set_ylabel("LR")
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax1b.get_legend_handles_labels()
        ax1.legend().remove() if ax1.get_legend() else None
        ax1b.legend().remove() if ax1b.get_legend() else None
        ax1._combined_legend = (lines1 + lines2, labels1 + labels2)
    else:
        ax1._combined_legend = ax1.get_legend_handles_labels()

    for col in comp_cols:
        if col in df.columns:
            ax2.plot(steps, smooth(df[col], smooth_window), linewidth=1.8, label=col)
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Component Loss")
    ax2.grid(True, which="both", ls="--", alpha=0.3)
    if logy:
        ax2.set_yscale("log")

    lines, labels = getattr(ax1, "_combined_legend", ax1.get_legend_handles_labels())
    ax1.legend(lines, labels, loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8, frameon=True)

    lines, labels = ax2.get_legend_handles_labels()
    ax2.legend(lines, labels, loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8, frameon=True)

    ax1.set_title(f"{title} - Training Loss")
    ax2.set_title(f"{title} - Components")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path)
    plt.close()
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["coarse", "refine", "all"], default="all")
    parser.add_argument("--smooth", type=int, default=50)
    parser.add_argument("--logy", action="store_true")
    args = parser.parse_args()

    root = get_root()

    if args.stage in ("coarse", "all"):
        csv_path = os.path.join(root, "exp_results", "coarse", "logs", "training_log.csv")
        out_path = os.path.join(root, "exp_results", "coarse", "logs", "training_curves.png")
        plot_one(csv_path, out_path, "Coarse", ["LossUnknown", "LossKnown", "LossBoundary"], args.smooth, args.logy)

    if args.stage in ("refine", "all"):
        csv_path = os.path.join(root, "exp_results", "refine", "logs", "training_log.csv")
        out_path = os.path.join(root, "exp_results", "refine", "logs", "training_curves.png")
        plot_one(csv_path, out_path, "Refine", ["LossUnknown", "LossKnown", "LossBoundary", "LossCoarse", "LossGrad"], args.smooth, args.logy)


if __name__ == "__main__":
    main()

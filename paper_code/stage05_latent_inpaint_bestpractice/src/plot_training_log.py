#%%
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import CONFIG
from src.utils_path import get_root


def read_csv(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"CSV is empty: {csv_path}")
    return df


def smooth(series: pd.Series, window: int) -> pd.Series:
    window = int(window)
    if window <= 1:
        return series
    return series.rolling(window=window, min_periods=1).mean()


def sanitize_steps(df: pd.DataFrame) -> pd.DataFrame:
    # If step resets (due to restart without checkpoint), keep the last segment
    steps = df["Step"].values
    if len(steps) > 1:
        drops = np.where(np.diff(steps) < 0)[0]
        if len(drops) > 0:
            df = df.iloc[drops[-1] + 1 :].copy()

    # Drop duplicate steps, keep last
    df = df.sort_values("Step").drop_duplicates(subset=["Step"], keep="last")
    return df


def plot_loss(ax, df: pd.DataFrame, smooth_window: int, logy: bool):
    steps = df["Step"]
    loss = df["Loss"]
    ax.plot(steps, loss, alpha=0.15, linewidth=0.7, label="Loss (Raw)")
    ax.plot(steps, smooth(loss, smooth_window), linewidth=2.0, label=f"Loss (Smoothed, w={smooth_window})")
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.grid(True, which="both", ls="--", alpha=0.3)
    if logy:
        ax.set_yscale("log")

    if "LR" in df.columns:
        ax2 = ax.twinx()
        ax2.plot(steps, df["LR"], linestyle="--", linewidth=1.4, label="LR")
        ax2.set_ylabel("Learning Rate")
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend().remove() if ax.get_legend() else None
        ax2.legend().remove() if ax2.get_legend() else None
        ax._combined_legend = (lines1 + lines2, labels1 + labels2)
    else:
        ax._combined_legend = ax.get_legend_handles_labels()


def plot_components(ax, df: pd.DataFrame, smooth_window: int, logy: bool):
    cols = [
        ("DiffUnknown", "DiffUnknown"),
        ("DiffKnown", "DiffKnown"),
        ("X0Unknown", "X0Unknown"),
        ("X0Boundary", "X0Boundary"),
        ("LowFreq", "LowFreq"),
    ]
    steps = df["Step"]
    for col, label in cols:
        if col in df.columns:
            ax.plot(steps, smooth(df[col], smooth_window), linewidth=1.8, label=label)
    ax.set_xlabel("Step")
    ax.set_ylabel("Component Loss")
    ax.grid(True, which="both", ls="--", alpha=0.3)
    if logy:
        ax.set_yscale("log")
    ax._combined_legend = ax.get_legend_handles_labels()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default=None, help="path to training_log.csv")
    parser.add_argument("--out", type=str, default=None, help="output png path")
    parser.add_argument("--smooth", type=int, default=50, help="rolling window for smoothing")
    parser.add_argument("--logy", action="store_true", help="use log scale for y")
    args = parser.parse_args()

    root = get_root()
    default_csv = os.path.join(root, "exp_results", CONFIG["experiment_name"], "logs", "training_log.csv")
    csv_path = args.csv or default_csv

    out_path = args.out
    if out_path is None:
        out_path = os.path.join(root, "exp_results", CONFIG["experiment_name"], "logs", "training_curves.png")

    df = read_csv(csv_path)
    df = sanitize_steps(df)

    plt.style.use("default")
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=150, constrained_layout=True)
    ax1, ax2 = axes

    plot_loss(ax1, df, smooth_window=args.smooth, logy=args.logy)
    ax1.set_title("Training Loss")

    plot_components(ax2, df, smooth_window=args.smooth, logy=args.logy)
    ax2.set_title("Loss Components")

    # place legends outside right
    lines, labels = getattr(ax1, "_combined_legend", ax1.get_legend_handles_labels())
    ax1.legend(lines, labels, loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8, frameon=True)

    lines, labels = getattr(ax2, "_combined_legend", ax2.get_legend_handles_labels())
    ax2.legend(lines, labels, loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8, frameon=True)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path)
    plt.close()

    print(f"Saved plot: {out_path}")


if __name__ == "__main__":
    main()

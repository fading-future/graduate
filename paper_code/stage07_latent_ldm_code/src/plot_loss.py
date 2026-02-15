import argparse
import csv
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_csv(path: str):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows


def to_float_list(rows, key):
    out = []
    for r in rows:
        try:
            out.append(float(r[key]))
        except Exception:
            out.append(float("nan"))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", required=True, help="training_log.csv path")
    parser.add_argument("--out", default="loss.png", help="output png path")
    parser.add_argument("--logy", action="store_true", help="use log scale on y-axis")
    args = parser.parse_args()

    rows = load_csv(args.log)
    if len(rows) == 0:
        raise ValueError("Empty log file")

    steps = to_float_list(rows, "Step")
    loss = to_float_list(rows, "Loss")
    diff = to_float_list(rows, "DiffTarget")
    x0 = to_float_list(rows, "X0Target")

    # avoid log(0)
    eps = 1e-8
    loss_plot = [max(v, eps) for v in loss]
    diff_plot = [max(v, eps) for v in diff]
    x0_plot = [max(v, eps) for v in x0]

    fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
    axes[0].plot(steps, loss_plot, label="loss", color="C0")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(steps, diff_plot, label="diff_target", color="C1")
    axes[1].set_ylabel("Diff")
    axes[1].legend()

    axes[2].plot(steps, x0_plot, label="x0_target", color="C2")
    axes[2].set_ylabel("X0")
    axes[2].set_xlabel("Step")
    axes[2].legend()

    if args.logy:
        for ax in axes:
            ax.set_yscale("log")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    plt.tight_layout()
    plt.savefig(args.out, dpi=200)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()

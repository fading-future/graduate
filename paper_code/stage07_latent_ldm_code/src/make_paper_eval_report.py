import argparse
import csv
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SUMMARY_KEYS = [
    "num_samples",
    "voxel_dice_mean",
    "voxel_iou_mean",
    "voxel_precision_mean",
    "voxel_recall_mean",
    "porosity_abs_err_mean",
    "bin_phi_corr_mean",
    "bin_phi_mae_mean",
    "latent_mae_mean",
    "z_head_porosity_gap_mean",
    "z_tail_porosity_gap_mean",
    "time_sec_mean",
]

PER_SAMPLE_KEYS = [
    "voxel_dice",
    "voxel_iou",
    "porosity_abs_err",
    "bin_phi_corr",
    "latent_mae",
    "z_head_porosity_gap",
    "z_tail_porosity_gap",
]


def _safe_float(v):
    try:
        return float(v)
    except Exception:
        return float("nan")


def _load_summary(eval_dir: Path) -> Dict:
    path = eval_dir / "summary.json"
    if not path.exists():
        raise FileNotFoundError(f"summary.json not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_per_sample(eval_dir: Path) -> List[Dict]:
    path = eval_dir / "per_sample_metrics.csv"
    if not path.exists():
        raise FileNotFoundError(f"per_sample_metrics.csv not found: {path}")
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            out = dict(row)
            for k in PER_SAMPLE_KEYS:
                if k in out:
                    out[k] = _safe_float(out[k])
            rows.append(out)
    return rows


def _save_summary_table(
    summaries: List[Dict],
    labels: List[str],
    out_csv: Path,
    out_md: Path,
):
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["label"] + SUMMARY_KEYS)
        for label, s in zip(labels, summaries):
            writer.writerow([label] + [s.get(k, "") for k in SUMMARY_KEYS])

    lines = []
    lines.append("| label | " + " | ".join(SUMMARY_KEYS) + " |")
    lines.append("|" + "---|" * (len(SUMMARY_KEYS) + 1))
    for label, s in zip(labels, summaries):
        vals = []
        for k in SUMMARY_KEYS:
            v = s.get(k, "")
            if isinstance(v, float):
                vals.append(f"{v:.6f}")
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join([label] + vals) + " |")
    out_md.write_text("\n".join(lines), encoding="utf-8")


def _boxplot_metric(rows_by_exp: List[List[Dict]], labels: List[str], metric: str, out_path: Path):
    values = []
    for rows in rows_by_exp:
        arr = np.array([r.get(metric, float("nan")) for r in rows], dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        values.append(arr)

    plt.figure(figsize=(max(8, 1.5 * len(labels)), 5))
    plt.boxplot(values, labels=labels, showmeans=True)
    plt.ylabel(metric)
    plt.title(f"Per-sample distribution: {metric}")
    plt.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=220)
    plt.close()


def _plot_porosity_scatter(rows_by_exp: List[List[Dict]], labels: List[str], out_path: Path):
    n = len(labels)
    cols = min(3, n)
    rows_n = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows_n, cols, figsize=(5 * cols, 4 * rows_n))
    axes = np.array(axes).reshape(-1)

    for i, (rows, label) in enumerate(zip(rows_by_exp, labels)):
        ax = axes[i]
        gt = np.array([r.get("porosity_gt", float("nan")) for r in rows], dtype=np.float64)
        pred = np.array([r.get("porosity_pred", float("nan")) for r in rows], dtype=np.float64)
        ok = np.isfinite(gt) & np.isfinite(pred)
        gt = gt[ok]
        pred = pred[ok]
        ax.scatter(gt, pred, s=14, alpha=0.75)
        mn = min(gt.min(initial=0.0), pred.min(initial=0.0))
        mx = max(gt.max(initial=1.0), pred.max(initial=1.0))
        ax.plot([mn, mx], [mn, mx], linestyle="--", linewidth=1.0)
        ax.set_title(label)
        ax.set_xlabel("GT porosity")
        ax.set_ylabel("Pred porosity")
        ax.grid(alpha=0.3)

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=220)
    plt.close()


def _plot_boundary_bias_bars(summaries: List[Dict], labels: List[str], out_path: Path):
    head = np.array([_safe_float(s.get("z_head_porosity_gap_mean", np.nan)) for s in summaries], dtype=np.float64)
    tail = np.array([_safe_float(s.get("z_tail_porosity_gap_mean", np.nan)) for s in summaries], dtype=np.float64)

    x = np.arange(len(labels))
    w = 0.35
    plt.figure(figsize=(max(8, 1.6 * len(labels)), 5))
    plt.bar(x - w / 2, head, width=w, label="z_head_porosity_gap_mean")
    plt.bar(x + w / 2, tail, width=w, label="z_tail_porosity_gap_mean")
    plt.axhline(0.0, color="black", linewidth=1.0)
    plt.xticks(x, labels, rotation=20, ha="right")
    plt.ylabel("Gap (pred - gt)")
    plt.title("Boundary bias (mean porosity gap)")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=220)
    plt.close()


def _save_paired_delta_table(
    rows_by_exp: List[List[Dict]],
    labels: List[str],
    out_path: Path,
):
    # Only compare first two experiments when basename overlap exists.
    if len(rows_by_exp) < 2:
        return
    a_map = {r.get("basename", f"row_{i}"): r for i, r in enumerate(rows_by_exp[0])}
    b_map = {r.get("basename", f"row_{i}"): r for i, r in enumerate(rows_by_exp[1])}
    overlap = sorted(set(a_map.keys()) & set(b_map.keys()))
    if len(overlap) == 0:
        return

    metrics = ["voxel_dice", "voxel_iou", "bin_phi_corr", "porosity_abs_err", "latent_mae"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "n_overlap", f"{labels[0]}_mean", f"{labels[1]}_mean", "delta(new-old)"])
        for m in metrics:
            a = np.array([_safe_float(a_map[k].get(m, np.nan)) for k in overlap], dtype=np.float64)
            b = np.array([_safe_float(b_map[k].get(m, np.nan)) for k in overlap], dtype=np.float64)
            ok = np.isfinite(a) & np.isfinite(b)
            if ok.sum() == 0:
                continue
            writer.writerow([m, int(ok.sum()), float(a[ok].mean()), float(b[ok].mean()), float((b[ok] - a[ok]).mean())])


def main():
    parser = argparse.ArgumentParser(
        description="Generate paper-ready comparison figures from multiple eval_batch folders."
    )
    parser.add_argument(
        "--eval-dirs",
        nargs="+",
        required=True,
        help="List of eval_batch directories, each containing summary.json and per_sample_metrics.csv",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        default=None,
        help="Optional labels for each eval dir. If omitted, folder names are used.",
    )
    parser.add_argument("--out-dir", default="paper_figures", help="Output directory for tables and plots.")
    args = parser.parse_args()

    eval_dirs = [Path(p) for p in args.eval_dirs]
    labels = args.labels if args.labels else [p.name for p in eval_dirs]
    if len(labels) != len(eval_dirs):
        raise ValueError("Number of labels must match number of eval dirs.")

    summaries = [_load_summary(p) for p in eval_dirs]
    rows_by_exp = [_load_per_sample(p) for p in eval_dirs]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    _save_summary_table(
        summaries=summaries,
        labels=labels,
        out_csv=out_dir / "summary_table.csv",
        out_md=out_dir / "summary_table.md",
    )

    for m in ["voxel_dice", "voxel_iou", "porosity_abs_err", "bin_phi_corr", "latent_mae"]:
        _boxplot_metric(rows_by_exp, labels, m, out_dir / f"boxplot_{m}.png")

    _plot_porosity_scatter(rows_by_exp, labels, out_dir / "scatter_porosity_pred_vs_gt.png")
    _plot_boundary_bias_bars(summaries, labels, out_dir / "bar_boundary_gap_mean.png")
    _save_paired_delta_table(rows_by_exp, labels, out_dir / "paired_delta_first_two.csv")

    print("[done] paper eval report generated")
    print(f"  out_dir: {out_dir.resolve()}")
    print(f"  summary table: {(out_dir / 'summary_table.csv').resolve()}")
    print(f"  markdown table: {(out_dir / 'summary_table.md').resolve()}")


if __name__ == "__main__":
    main()

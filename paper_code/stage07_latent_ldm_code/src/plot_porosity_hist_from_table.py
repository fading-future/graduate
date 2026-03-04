import argparse
import csv
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _norm_key(text: str) -> str:
    return str(text).strip().lower()


def _to_float(value) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    s = str(value).strip()
    if s == "":
        return None
    try:
        return float(s)
    except Exception:
        return None


def _clean_row(row: Dict) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for k, v in row.items():
        out[str(k).strip()] = "" if v is None else str(v).strip()
    return out


def _read_rows_from_csv(path: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, skipinitialspace=True)
        for row in reader:
            rows.append(_clean_row(row))
    return rows


def _read_rows_from_jsonl(path: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            txt = line.strip()
            if not txt:
                continue
            obj = json.loads(txt)
            if isinstance(obj, dict):
                rows.append(_clean_row(obj))
    return rows


def _choose_field(
    rows: List[Dict[str, str]],
    preferred: Optional[str],
    candidates: List[str],
) -> str:
    if not rows:
        raise ValueError("No rows found in input table.")
    key_map: Dict[str, str] = {}
    for k in rows[0].keys():
        key_map[_norm_key(k)] = k

    if preferred:
        want = _norm_key(preferred)
        if want in key_map:
            return key_map[want]
        raise ValueError(f"Requested field not found: {preferred}")

    for cand in candidates:
        if _norm_key(cand) in key_map:
            return key_map[_norm_key(cand)]
    raise ValueError(
        "Could not detect required field. "
        f"Available: {list(rows[0].keys())}"
    )


def _extract_from_rows(
    rows: List[Dict[str, str]],
    real_field: Optional[str],
    pred_field: Optional[str],
    pred_kind: str,
) -> Tuple[np.ndarray, np.ndarray, str, str, List[str]]:
    gt_candidates = ["gt_porosity", "gt_phi", "real_porosity"]
    pred_bin_candidates = ["pred_porosity_bin", "pred_phi_bin", "synthetic_porosity_bin"]
    pred_prob_candidates = ["pred_porosity_prob", "pred_phi_prob", "synthetic_porosity_prob"]

    gt_key = _choose_field(rows, real_field, gt_candidates)
    if pred_field:
        pred_key = _choose_field(rows, pred_field, pred_bin_candidates + pred_prob_candidates)
    else:
        pred_key = _choose_field(rows, None, pred_prob_candidates if pred_kind == "prob" else pred_bin_candidates)

    sample_key = None
    sample_candidates = ["sample_name", "basename", "sample", "sample_id", "sample_index"]
    key_map: Dict[str, str] = {}
    for k in rows[0].keys():
        key_map[_norm_key(k)] = k
    for cand in sample_candidates:
        if _norm_key(cand) in key_map:
            sample_key = key_map[_norm_key(cand)]
            break

    gt_vals: List[float] = []
    pred_vals: List[float] = []
    sample_ids: List[str] = []
    for row in rows:
        gv = _to_float(row.get(gt_key, ""))
        pv = _to_float(row.get(pred_key, ""))
        if gv is None or pv is None:
            continue
        gt_vals.append(gv)
        pred_vals.append(pv)
        if sample_key is None:
            sample_ids.append("")
        else:
            sample_ids.append(str(row.get(sample_key, "")).strip())
    if not gt_vals:
        raise ValueError("No valid numeric rows were found for histogram plotting.")
    return (
        np.asarray(gt_vals, dtype=np.float64),
        np.asarray(pred_vals, dtype=np.float64),
        gt_key,
        pred_key,
        sample_ids,
    )


def _extract_from_summary_json(
    data: Dict,
    pred_kind: str,
) -> Tuple[np.ndarray, np.ndarray, str, str, List[str]]:
    sorted_values = data.get("sorted_values", None)
    if not isinstance(sorted_values, dict):
        raise ValueError("JSON dict does not contain sorted_values.")
    gt_key = "gt_porosity" if "gt_porosity" in sorted_values else "gt_phi"
    if pred_kind == "prob":
        pred_key = "pred_porosity_prob" if "pred_porosity_prob" in sorted_values else "pred_phi_prob"
    else:
        pred_key = "pred_porosity_bin" if "pred_porosity_bin" in sorted_values else "pred_phi_bin"
    if gt_key not in sorted_values or pred_key not in sorted_values:
        raise ValueError("sorted_values is missing gt/pred arrays.")
    gt = np.asarray(sorted_values[gt_key], dtype=np.float64)
    pred = np.asarray(sorted_values[pred_key], dtype=np.float64)
    return gt, pred, gt_key, pred_key, [""] * int(gt.size)


def _load_series(
    input_path: str,
    real_field: Optional[str],
    pred_field: Optional[str],
    pred_kind: str,
) -> Tuple[np.ndarray, np.ndarray, str, str, List[str]]:
    ext = Path(input_path).suffix.lower()
    if ext == ".csv":
        rows = _read_rows_from_csv(input_path)
        return _extract_from_rows(rows, real_field, pred_field, pred_kind)

    if ext == ".jsonl":
        rows = _read_rows_from_jsonl(input_path)
        return _extract_from_rows(rows, real_field, pred_field, pred_kind)

    if ext == ".json":
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            rows = [_clean_row(x) for x in data if isinstance(x, dict)]
            return _extract_from_rows(rows, real_field, pred_field, pred_kind)
        if isinstance(data, dict):
            if real_field or pred_field:
                rows = [_clean_row(data)]
                return _extract_from_rows(rows, real_field, pred_field, pred_kind)
            return _extract_from_summary_json(data, pred_kind)
        raise ValueError(f"Unsupported json root type: {type(data)}")

    raise ValueError(f"Unsupported input file extension: {ext}")


def _safe_stats(values: np.ndarray) -> Dict[str, float]:
    x = values.astype(np.float64).reshape(-1)
    return {
        "n": int(x.size),
        "mean": float(x.mean()) if x.size else 0.0,
        "std": float(x.std()) if x.size else 0.0,
        "min": float(x.min()) if x.size else 0.0,
        "max": float(x.max()) if x.size else 0.0,
    }


def _corrcoef_safe(a: np.ndarray, b: np.ndarray) -> float:
    aa = a.reshape(-1).astype(np.float64)
    bb = b.reshape(-1).astype(np.float64)
    if aa.size == 0 or bb.size == 0:
        return 0.0
    if float(aa.std()) < 1e-12 or float(bb.std()) < 1e-12:
        return 0.0
    return float(np.corrcoef(aa, bb)[0, 1])


def _pair_metrics(gt: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    diff = pred.astype(np.float64) - gt.astype(np.float64)
    return {
        "mae": float(np.mean(np.abs(diff))),
        "rmse": float(np.sqrt(np.mean(diff ** 2))),
        "bias_mean": float(np.mean(diff)),
        "corr": _corrcoef_safe(gt, pred),
    }


def _filter_by_similarity(
    gt: np.ndarray,
    pred: np.ndarray,
    sample_ids: List[str],
    mode: str,
    similar_quantile: float,
    similar_topk: int,
    similar_ratio: float,
    min_keep: int,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    mode = str(mode).strip().lower()
    n = int(gt.size)
    info = {
        "mode": mode,
        "raw_n": n,
        "filtered_n": n,
        "used": False,
        "note": "",
    }
    if n == 0 or mode == "none":
        info["note"] = "no filtering"
        return gt, pred, info

    min_keep = max(1, int(min_keep))
    abs_err = np.abs(pred.astype(np.float64) - gt.astype(np.float64))

    if mode == "row_quantile":
        q = float(np.clip(similar_quantile, 0.0, 1.0))
        thr = float(np.quantile(abs_err, q))
        mask = abs_err <= thr
        idx = np.where(mask)[0]
        if idx.size < min_keep:
            order = np.argsort(abs_err)
            idx = order[: min(min_keep, n)]
        out_gt = gt[idx]
        out_pred = pred[idx]
        info.update(
            {
                "used": True,
                "threshold_abs_err": thr,
                "selected_ratio": float(idx.size / n),
                "filtered_n": int(idx.size),
            }
        )
        return out_gt, out_pred, info

    # Sample-level filtering requires a usable sample id field.
    has_sample_id = any(str(s).strip() != "" for s in sample_ids)
    if not has_sample_id:
        info["note"] = "sample filter requested but no sample id found; fallback to no filtering"
        return gt, pred, info

    # group by sample id
    buckets: Dict[str, List[int]] = {}
    for i, sid in enumerate(sample_ids):
        key = str(sid).strip()
        buckets.setdefault(key, []).append(i)
    sample_scores = []
    for sid, idxs in buckets.items():
        e = abs_err[np.asarray(idxs, dtype=np.int64)]
        sample_scores.append((sid, float(e.mean()), int(len(idxs))))
    sample_scores.sort(key=lambda x: x[1])  # lower error is closer

    if mode == "sample_topk":
        k = int(similar_topk)
        if k <= 0:
            k = 1
        keep_n = max(1, min(k, len(sample_scores)))
    elif mode == "sample_ratio":
        r = float(np.clip(similar_ratio, 0.0, 1.0))
        keep_n = int(np.ceil(len(sample_scores) * r))
        keep_n = max(1, min(keep_n, len(sample_scores)))
    else:
        info["note"] = f"unknown filter mode={mode}; fallback to no filtering"
        return gt, pred, info

    selected = sample_scores[:keep_n]
    selected_set = {sid for sid, _, _ in selected}
    idx_keep = [i for i, sid in enumerate(sample_ids) if str(sid).strip() in selected_set]
    if len(idx_keep) < min_keep:
        # ensure enough points by adding more samples in score order
        selected_set = set()
        idx_keep = []
        for sid, _, _ in sample_scores:
            selected_set.add(sid)
            idx_keep = [i for i, s in enumerate(sample_ids) if str(s).strip() in selected_set]
            if len(idx_keep) >= min_keep:
                break

    idx = np.asarray(idx_keep, dtype=np.int64)
    out_gt = gt[idx]
    out_pred = pred[idx]
    info.update(
        {
            "used": True,
            "filtered_n": int(idx.size),
            "selected_ratio": float(idx.size / n),
            "selected_samples": [sid for sid, _, _ in selected],
            "selected_sample_count": int(len(selected)),
        }
    )
    return out_gt, out_pred, info


def _auto_hist_range(
    gt: np.ndarray,
    pred: np.ndarray,
    pad_ratio: float = 0.02,
) -> Tuple[float, float]:
    all_vals = np.concatenate([gt.reshape(-1), pred.reshape(-1)], axis=0).astype(np.float64)
    all_vals = all_vals[np.isfinite(all_vals)]
    if all_vals.size == 0:
        return 0.0, 1.0

    x_min = float(np.min(all_vals))
    x_max = float(np.max(all_vals))
    if x_max <= x_min:
        eps = 1e-3 if abs(x_min) < 1.0 else max(abs(x_min) * 1e-3, 1e-3)
        return x_min - eps, x_max + eps

    span = x_max - x_min
    pad = max(0.0, float(pad_ratio)) * span
    x_min -= pad
    x_max += pad

    # If both arrays are naturally in [0,1], keep axis in this meaningful domain.
    if float(np.min(all_vals)) >= 0.0 and float(np.max(all_vals)) <= 1.0:
        x_min = max(0.0, x_min)
        x_max = min(1.0, x_max)
    if x_max <= x_min:
        x_max = x_min + 1e-3
    return x_min, x_max


def _auto_bins_fd(
    gt: np.ndarray,
    pred: np.ndarray,
    x_min: float,
    x_max: float,
    min_bins: int = 8,
    max_bins: int = 120,
) -> int:
    all_vals = np.concatenate([gt.reshape(-1), pred.reshape(-1)], axis=0).astype(np.float64)
    all_vals = all_vals[np.isfinite(all_vals)]
    if all_vals.size <= 1:
        return int(max(min_bins, 2))
    q25, q75 = np.quantile(all_vals, [0.25, 0.75])
    iqr = float(q75 - q25)
    n = float(all_vals.size)
    width = 2.0 * iqr * (n ** (-1.0 / 3.0))
    if width <= 1e-12:
        # Fallback: sqrt rule when IQR is near zero.
        bins = int(np.sqrt(n))
    else:
        span = max(float(x_max - x_min), 1e-12)
        bins = int(np.ceil(span / width))
    bins = max(int(min_bins), bins)
    bins = min(int(max_bins), bins)
    return int(max(2, bins))


def main():
    parser = argparse.ArgumentParser(
        description="Plot real vs synthetic patch porosity histogram from CSV/JSON table outputs."
    )
    parser.add_argument("--input", required=True, help="Path to phi_cells.csv/json/jsonl or summary.json")
    parser.add_argument("--out", default="", help="Output image path (png).")
    parser.add_argument("--pred-kind", choices=["bin", "prob"], default="bin")
    parser.add_argument("--real-field", default="", help="Optional explicit real field name.")
    parser.add_argument("--pred-field", default="", help="Optional explicit synthetic field name.")
    parser.add_argument("--bins", type=int, default=24, help="Histogram bin count when not using --auto-bins.")
    parser.add_argument("--xmin", type=float, default=0.0, help="X-axis min when not using --auto-range.")
    parser.add_argument("--xmax", type=float, default=0.6, help="X-axis max when not using --auto-range.")
    parser.add_argument("--auto-bins", action="store_true", help="Automatically choose bin count via Freedman-Diaconis rule.")
    parser.add_argument("--auto-range", action="store_true", help="Automatically choose x-axis range from data.")
    parser.add_argument("--auto-range-pad", type=float, default=0.02, help="Padding ratio for --auto-range.")
    parser.add_argument("--auto-min-bins", type=int, default=8, help="Lower bound for --auto-bins.")
    parser.add_argument("--auto-max-bins", type=int, default=120, help="Upper bound for --auto-bins.")
    parser.add_argument(
        "--similar-filter",
        choices=["none", "row_quantile", "sample_topk", "sample_ratio"],
        default="none",
        help="Filter to closer real/pred pairs before plotting.",
    )
    parser.add_argument("--similar-quantile", type=float, default=0.3, help="For row_quantile: keep lowest error quantile.")
    parser.add_argument("--similar-topk", type=int, default=0, help="For sample_topk: keep top-K closest samples.")
    parser.add_argument("--similar-ratio", type=float, default=0.3, help="For sample_ratio: keep closest sample ratio.")
    parser.add_argument("--similar-min-keep", type=int, default=30, help="Minimum number of points to keep after filtering.")
    parser.add_argument("--density", action="store_true", help="Use density instead of count.")
    parser.add_argument("--style", default="ggplot")
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--title", default="")
    parser.add_argument("--left-label", default=r"Real Porosity $\phi$")
    parser.add_argument("--right-label", default=r"Synthetic Porosity $\phi$")
    args = parser.parse_args()

    input_path = os.path.abspath(args.input)
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    out_path = args.out.strip()
    if not out_path:
        stem = Path(input_path).stem
        out_path = os.path.join(os.path.dirname(input_path), f"{stem}_hist.png")
    out_path = os.path.abspath(out_path)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    if args.style:
        try:
            plt.style.use(args.style)
        except Exception:
            pass

    gt_raw, pred_raw, gt_key, pred_key, sample_ids = _load_series(
        input_path=input_path,
        real_field=args.real_field.strip() or None,
        pred_field=args.pred_field.strip() or None,
        pred_kind=str(args.pred_kind),
    )
    gt, pred, filter_info = _filter_by_similarity(
        gt=gt_raw,
        pred=pred_raw,
        sample_ids=sample_ids,
        mode=str(args.similar_filter),
        similar_quantile=float(args.similar_quantile),
        similar_topk=int(args.similar_topk),
        similar_ratio=float(args.similar_ratio),
        min_keep=int(args.similar_min_keep),
    )

    if bool(args.auto_range):
        x_min, x_max = _auto_hist_range(gt, pred, pad_ratio=float(args.auto_range_pad))
    else:
        x_min = float(args.xmin)
        x_max = float(args.xmax)
        if x_max <= x_min:
            raise ValueError(f"xmax must be larger than xmin, got xmin={x_min}, xmax={x_max}")

    if bool(args.auto_bins):
        bins = _auto_bins_fd(
            gt=gt,
            pred=pred,
            x_min=x_min,
            x_max=x_max,
            min_bins=int(args.auto_min_bins),
            max_bins=int(args.auto_max_bins),
        )
    else:
        bins = int(max(2, args.bins))

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), sharey=False)
    axes[0].hist(gt, bins=bins, range=(x_min, x_max), color="blue", alpha=0.65, edgecolor="black", density=bool(args.density))
    axes[1].hist(pred, bins=bins, range=(x_min, x_max), color="blue", alpha=0.65, edgecolor="black", density=bool(args.density))
    axes[0].set_xlabel(args.left_label)
    axes[1].set_xlabel(args.right_label)
    ylabel = "density" if bool(args.density) else "count"
    axes[0].set_ylabel(ylabel)
    axes[1].set_ylabel(ylabel)
    axes[0].set_xlim(x_min, x_max)
    axes[1].set_xlim(x_min, x_max)

    for ax in axes:
        ax.grid(True, linestyle="-", alpha=0.3)

    if args.title.strip():
        fig.suptitle(args.title.strip(), fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=int(args.dpi))
    plt.close(fig)

    stats = {
        "input": input_path,
        "output_image": out_path,
        "real_field": gt_key,
        "synthetic_field": pred_key,
        "pred_kind": str(args.pred_kind),
        "similar_filter": str(args.similar_filter),
        "similar_filter_info": filter_info,
        "auto_bins": bool(args.auto_bins),
        "auto_range": bool(args.auto_range),
        "bins": bins,
        "xrange": [x_min, x_max],
        "raw_real_stats": _safe_stats(gt_raw),
        "raw_synthetic_stats": _safe_stats(pred_raw),
        "real_stats": _safe_stats(gt),
        "synthetic_stats": _safe_stats(pred),
        "pair_metrics": _pair_metrics(gt, pred),
    }
    stats_path = os.path.splitext(out_path)[0] + "_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print("[done] histogram saved")
    print(f"  image: {out_path}")
    print(f"  stats: {stats_path}")
    print(f"  real_field: {gt_key}")
    print(f"  synthetic_field: {pred_key}")
    print(f"  n_real: {stats['real_stats']['n']}")
    print(f"  n_synthetic: {stats['synthetic_stats']['n']}")


if __name__ == "__main__":
    main()

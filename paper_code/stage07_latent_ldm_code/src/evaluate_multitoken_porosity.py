import argparse
import csv
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from src.analysis_metrics import to_phase_mask
from src.config import CONFIG
from src.diffusion import DiffusionHelper
from src.evaluate_generated_sample import (
    align_pair,
    compute_phi_map,
    decode_to_prob_and_bin,
    load_latent,
    load_vae,
)
from src.infer import generate_volume, load_model


def _resolve_path(path_str: str, repo_root: Path) -> str:
    if not path_str:
        return ""
    p = Path(path_str)
    if p.is_absolute():
        return str(p)
    p_cwd = Path.cwd() / p
    if p_cwd.exists():
        return str(p_cwd.resolve())
    p_repo = repo_root / p
    if p_repo.exists():
        return str(p_repo.resolve())
    return str(p_cwd.resolve())


def _corrcoef_safe(a: np.ndarray, b: np.ndarray) -> float:
    aa = a.reshape(-1).astype(np.float64)
    bb = b.reshape(-1).astype(np.float64)
    if aa.size == 0 or bb.size == 0:
        return 0.0
    if float(aa.std()) < 1e-12 or float(bb.std()) < 1e-12:
        return 0.0
    return float(np.corrcoef(aa, bb)[0, 1])


def _stats(arr: np.ndarray) -> Dict[str, float]:
    x = arr.reshape(-1).astype(np.float64)
    if x.size == 0:
        return {
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "q05": 0.0,
            "q25": 0.0,
            "q50": 0.0,
            "q75": 0.0,
            "q95": 0.0,
        }
    return {
        "mean": float(x.mean()),
        "std": float(x.std()),
        "min": float(x.min()),
        "max": float(x.max()),
        "q05": float(np.quantile(x, 0.05)),
        "q25": float(np.quantile(x, 0.25)),
        "q50": float(np.quantile(x, 0.50)),
        "q75": float(np.quantile(x, 0.75)),
        "q95": float(np.quantile(x, 0.95)),
    }


def _pair_metrics(pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
    diff = pred.astype(np.float64) - gt.astype(np.float64)
    return {
        "mae": float(np.mean(np.abs(diff))),
        "mse": float(np.mean(diff ** 2)),
        "rmse": float(np.sqrt(np.mean(diff ** 2))),
        "bias_mean": float(np.mean(diff)),
        "corr": _corrcoef_safe(pred, gt),
    }


def _prepare_gt_porosity(phi_map: np.ndarray, semantic: str) -> np.ndarray:
    sem = str(semantic).lower().strip()
    if sem not in ("porosity", "rock_rate"):
        raise ValueError(f"Unsupported gt phi semantic: {semantic}")
    if sem == "porosity":
        out = phi_map.astype(np.float32)
    else:
        out = (1.0 - phi_map).astype(np.float32)
    return np.clip(out, 0.0, 1.0)


def _flatten_cells(
    sample_index: int,
    basename: str,
    gt_phi: np.ndarray,
    pred_phi_bin: np.ndarray,
    pred_phi_prob: np.ndarray,
) -> List[Dict]:
    rows: List[Dict] = []
    g_d, g_h, g_w = gt_phi.shape
    center = (g_d // 2, g_h // 2, g_w // 2)
    idx = 0
    for i in range(g_d):
        for j in range(g_h):
            for k in range(g_w):
                gt = float(gt_phi[i, j, k])
                pb = float(pred_phi_bin[i, j, k])
                pp = float(pred_phi_prob[i, j, k])
                rows.append(
                    {
                        "sample_index": int(sample_index),
                        "sample_name": basename,
                        "cell_index": int(idx),
                        "cell_i": int(i),
                        "cell_j": int(j),
                        "cell_k": int(k),
                        "is_center_cell": int((i, j, k) == center),
                        "gt_porosity": gt,
                        "pred_porosity_bin": pb,
                        "pred_porosity_prob": pp,
                        "bin_abs_err": float(abs(pb - gt)),
                        "bin_bias": float(pb - gt),
                        "prob_abs_err": float(abs(pp - gt)),
                        "prob_bias": float(pp - gt),
                    }
                )
                idx += 1
    return rows


def _write_csv(rows: List[Dict], out_csv: str, keys: Optional[List[str]] = None):
    if not rows:
        if keys is None:
            keys = []
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            if keys:
                writer.writeheader()
        return
    if keys is None:
        seen = set()
        keys = []
        for row in rows:
            for key in row.keys():
                if key not in seen:
                    seen.add(key)
                    keys.append(key)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in keys})


def _run_infer_once(
    model,
    diffusion,
    phi_map_path: str,
    pred_latent_path: str,
    ddim_steps: int,
    seed: Optional[int],
    output_unscaled: bool,
    scale_factor: float,
):
    if not os.path.exists(phi_map_path):
        raise FileNotFoundError(f"phi map not found: {phi_map_path}")

    phi_map = np.load(phi_map_path).astype(np.float32)
    if phi_map.ndim != 3:
        raise ValueError(f"Expected phi map as 3D array, got shape={phi_map.shape}")

    z_full = generate_volume(
        phi_map=phi_map,
        model=model,
        diffusion=diffusion,
        patch_size=int(CONFIG["patch_size"]),
        window_size=int(CONFIG["window_size"]),
        steps=int(ddim_steps),
        seed=seed,
    )

    if bool(output_unscaled):
        sf = float(scale_factor) if float(scale_factor) != 0.0 else 1.0
        if sf != 1.0:
            z_full = z_full / sf

    os.makedirs(os.path.dirname(pred_latent_path) or ".", exist_ok=True)
    np.save(pred_latent_path, z_full)
    print(f"[infer] saved latent to: {pred_latent_path}")


def _collect_npy_files(phi_dir: str) -> List[str]:
    if not os.path.isdir(phi_dir):
        raise FileNotFoundError(f"phi dir not found: {phi_dir}")
    files = sorted(
        os.path.join(phi_dir, fn)
        for fn in os.listdir(phi_dir)
        if fn.lower().endswith(".npy")
    )
    if not files:
        raise ValueError(f"No .npy files found in {phi_dir}")
    return files


def _select_files(
    all_files: List[str],
    sample_mode: str,
    offset: int,
    num_samples: int,
    sample_seed: int,
) -> List[str]:
    offset = max(0, int(offset))
    n = int(num_samples)
    if sample_mode == "random":
        rng = np.random.default_rng(int(sample_seed))
        perm = rng.permutation(len(all_files)).tolist()
        sel = [all_files[int(i)] for i in perm]
    else:
        sel = list(all_files)
    sel = sel[offset:]
    if n > 0:
        sel = sel[:n]
    return sel


def _summary_stats_from_rows(rows: List[Dict]) -> Dict:
    if not rows:
        return {}
    out: Dict = {}
    numeric_keys = []
    for k in rows[0].keys():
        if isinstance(rows[0][k], (int, float, np.integer, np.floating)):
            numeric_keys.append(k)
    for k in numeric_keys:
        vals = []
        for row in rows:
            v = row.get(k, None)
            if isinstance(v, (int, float, np.integer, np.floating)):
                vals.append(float(v))
        if not vals:
            continue
        arr = np.array(vals, dtype=np.float64)
        out[f"{k}_mean"] = float(arr.mean())
        out[f"{k}_std"] = float(arr.std())
        out[f"{k}_min"] = float(arr.min())
        out[f"{k}_max"] = float(arr.max())
    return out


def _porosity_csv_keys() -> List[str]:
    return [
        "row_type",
        "sample_index",
        "sample_name",
        "num_cells",
        "gt_porosity_overall",
        "pred_porosity_bin_overall",
        "pred_porosity_prob_overall",
        "bin_abs_err",
        "bin_bias",
        "prob_abs_err",
        "prob_bias",
    ]


def _porosity_row_from_sample_summary(row: Dict) -> Dict:
    num_cells = int(row.get("num_cells", 0))
    gt = float(row.get("gt_mean", 0.0))
    pred_bin = float(row.get("pred_bin_mean", 0.0))
    pred_prob = float(row.get("pred_prob_mean", 0.0))
    return {
        "row_type": "sample",
        "sample_index": int(row.get("sample_index", 0)),
        "sample_name": str(row.get("sample_name", "")),
        "num_cells": num_cells,
        "gt_porosity_overall": gt,
        "pred_porosity_bin_overall": pred_bin,
        "pred_porosity_prob_overall": pred_prob,
        "bin_abs_err": float(abs(pred_bin - gt)),
        "bin_bias": float(pred_bin - gt),
        "prob_abs_err": float(abs(pred_prob - gt)),
        "prob_bias": float(pred_prob - gt),
    }


def _append_csv_rows(rows: List[Dict], out_csv: str, keys: List[str]):
    if not rows:
        return
    write_header = (not os.path.exists(out_csv)) or os.path.getsize(out_csv) == 0
    with open(out_csv, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in keys})


def _build_porosity_analysis(per_sample_rows: List[Dict]) -> Dict:
    per_sample = [_porosity_row_from_sample_summary(row) for row in per_sample_rows]
    total_cells = int(sum(int(r.get("num_cells", 0)) for r in per_sample))

    def _aggregate(rows: List[Dict], weighted: bool) -> Dict:
        if not rows:
            return {
                "gt_porosity_overall": 0.0,
                "pred_porosity_bin_overall": 0.0,
                "pred_porosity_prob_overall": 0.0,
                "bin_abs_err": 0.0,
                "bin_bias": 0.0,
                "prob_abs_err": 0.0,
                "prob_bias": 0.0,
            }

        if weighted:
            ws = np.array([float(max(0, int(r.get("num_cells", 0)))) for r in rows], dtype=np.float64)
            if float(ws.sum()) <= 0.0:
                ws = np.ones(len(rows), dtype=np.float64)
            ws = ws / float(ws.sum())
            gt = float(np.sum(np.array([r["gt_porosity_overall"] for r in rows], dtype=np.float64) * ws))
            pred_bin = float(np.sum(np.array([r["pred_porosity_bin_overall"] for r in rows], dtype=np.float64) * ws))
            pred_prob = float(np.sum(np.array([r["pred_porosity_prob_overall"] for r in rows], dtype=np.float64) * ws))
        else:
            gt = float(np.mean([r["gt_porosity_overall"] for r in rows]))
            pred_bin = float(np.mean([r["pred_porosity_bin_overall"] for r in rows]))
            pred_prob = float(np.mean([r["pred_porosity_prob_overall"] for r in rows]))

        return {
            "gt_porosity_overall": gt,
            "pred_porosity_bin_overall": pred_bin,
            "pred_porosity_prob_overall": pred_prob,
            "bin_abs_err": float(abs(pred_bin - gt)),
            "bin_bias": float(pred_bin - gt),
            "prob_abs_err": float(abs(pred_prob - gt)),
            "prob_bias": float(pred_prob - gt),
        }

    weighted_stats = _aggregate(per_sample, weighted=True)
    unweighted_stats = _aggregate(per_sample, weighted=False)

    csv_rows = list(per_sample)
    csv_rows.append(
        {
            "row_type": "batch_weighted",
            "sample_index": -1,
            "sample_name": "__BATCH_WEIGHTED__",
            "num_cells": int(total_cells),
            **weighted_stats,
        }
    )
    csv_rows.append(
        {
            "row_type": "batch_unweighted",
            "sample_index": -1,
            "sample_name": "__BATCH_UNWEIGHTED__",
            "num_cells": int(total_cells),
            **unweighted_stats,
        }
    )

    return {
        "csv_rows": csv_rows,
        "summary": {
            "num_samples": int(len(per_sample)),
            "num_total_cells": int(total_cells),
            "weighted_by_num_cells": weighted_stats,
            "unweighted_over_samples": unweighted_stats,
        },
    }


def _evaluate_one(
    sample_index: int,
    phi_map_path: str,
    pred_latent_path: str,
    ckpt_path: str,
    vae,
    model,
    diffusion,
    device: torch.device,
    args,
    patch_voxel: int,
) -> Dict:
    t0 = time.time()
    basename = Path(phi_map_path).stem

    if bool(args.run_infer):
        seed_i = int(args.seed) + int(sample_index) if args.seed is not None else None
        _run_infer_once(
            model=model,
            diffusion=diffusion,
            phi_map_path=phi_map_path,
            pred_latent_path=pred_latent_path,
            ddim_steps=int(args.ddim_steps),
            seed=seed_i,
            output_unscaled=bool(args.output_unscaled),
            scale_factor=float(args.scale_factor),
        )

    if not os.path.exists(pred_latent_path):
        raise FileNotFoundError(f"pred latent not found: {pred_latent_path}")

    gt_phi_raw = np.load(phi_map_path).astype(np.float32)
    if gt_phi_raw.ndim != 3:
        raise ValueError(f"Expected gt phi map as 3D array, got shape={gt_phi_raw.shape}")
    gt_phi = _prepare_gt_porosity(gt_phi_raw, args.gt_phi_semantic)

    pred_lat = load_latent(pred_latent_path)
    sf = float(args.scale_factor) if float(args.scale_factor) != 0.0 else 1.0
    pred_lat_unscaled = pred_lat / sf if bool(args.pred_latent_scaled) else pred_lat

    pred_prob, pred_bin = decode_to_prob_and_bin(pred_lat_unscaled, vae, device, float(args.threshold))

    pore_bin = to_phase_mask(pred_bin, int(args.pore_value)).astype(np.float32)
    pore_prob = pred_prob.astype(np.float32) if int(args.pore_value) == 1 else (1.0 - pred_prob.astype(np.float32))

    pred_phi_bin = compute_phi_map(pore_bin, patch_voxel)
    pred_phi_prob = compute_phi_map(pore_prob, patch_voxel)

    pred_phi_bin_al, gt_phi_al = align_pair(pred_phi_bin, gt_phi)
    pred_phi_prob_al, _ = align_pair(pred_phi_prob, gt_phi)

    rows = _flatten_cells(
        sample_index=sample_index,
        basename=basename,
        gt_phi=gt_phi_al,
        pred_phi_bin=pred_phi_bin_al,
        pred_phi_prob=pred_phi_prob_al,
    )

    gt_stats = _stats(gt_phi_al)
    pred_bin_stats = _stats(pred_phi_bin_al)
    pred_prob_stats = _stats(pred_phi_prob_al)
    pair_bin = _pair_metrics(pred_phi_bin_al, gt_phi_al)
    pair_prob = _pair_metrics(pred_phi_prob_al, gt_phi_al)

    out = {
        "sample_index": int(sample_index),
        "sample_name": basename,
        "phi_map_path": phi_map_path,
        "pred_latent_path": pred_latent_path,
        "ckpt_path": ckpt_path if bool(args.run_infer) else "",
        "num_cells": int(gt_phi_al.size),
        "gt_shape_d": int(gt_phi.shape[0]),
        "gt_shape_h": int(gt_phi.shape[1]),
        "gt_shape_w": int(gt_phi.shape[2]),
        "aligned_shape_d": int(gt_phi_al.shape[0]),
        "aligned_shape_h": int(gt_phi_al.shape[1]),
        "aligned_shape_w": int(gt_phi_al.shape[2]),
        "gt_mean": float(gt_stats["mean"]),
        "pred_bin_mean": float(pred_bin_stats["mean"]),
        "pred_prob_mean": float(pred_prob_stats["mean"]),
        "bin_mae": float(pair_bin["mae"]),
        "bin_rmse": float(pair_bin["rmse"]),
        "bin_bias": float(pair_bin["bias_mean"]),
        "bin_corr": float(pair_bin["corr"]),
        "prob_mae": float(pair_prob["mae"]),
        "prob_rmse": float(pair_prob["rmse"]),
        "prob_bias": float(pair_prob["bias_mean"]),
        "prob_corr": float(pair_prob["corr"]),
        "time_sec": float(time.time() - t0),
    }
    return {
        "rows": rows,
        "summary_row": out,
        "summary_detail": {
            "sample_name": basename,
            "phi_map_path": phi_map_path,
            "pred_latent_path": pred_latent_path,
            "distribution": {
                "gt_porosity": gt_stats,
                "pred_porosity_bin": pred_bin_stats,
                "pred_porosity_prob": pred_prob_stats,
            },
            "comparison_to_gt": {
                "bin": pair_bin,
                "prob": pair_prob,
            },
            "sorted_values": {
                "gt_porosity": sorted(float(x) for x in gt_phi_al.reshape(-1)),
                "pred_porosity_bin": sorted(float(x) for x in pred_phi_bin_al.reshape(-1)),
                "pred_porosity_prob": sorted(float(x) for x in pred_phi_prob_al.reshape(-1)),
            },
        },
    }


def main():
    repo_root = Path(__file__).resolve().parents[1]
    default_pred_scaled = not bool(CONFIG.get("output_unscaled", True))

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate multi-token porosity distribution from one sample. "
            "Exports per-cell (typically 3x3x3=27) porosity tables in CSV/JSON."
        )
    )
    parser.add_argument("--phi-map", default=CONFIG.get("phi_map_path", ""))
    parser.add_argument("--phi-dir", default="")
    parser.add_argument("--pred-latent", default=CONFIG.get("output_latent_path", "generated_latent.npy"))
    parser.add_argument("--pred-latent-dir", default="")
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--device", default=CONFIG.get("device", "cpu"))

    parser.add_argument("--vae-config", default=CONFIG.get("eval_vae_config_path", ""))
    parser.add_argument("--vae-ckpt", default=CONFIG.get("eval_vae_ckpt_path", ""))
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--scale-factor", type=float, default=float(CONFIG.get("scale_factor", 1.0)))
    parser.add_argument("--pred-latent-scaled", action="store_true", default=default_pred_scaled)

    parser.add_argument(
        "--patch-voxel",
        type=int,
        default=0,
        help="Voxel size of one phi cell. 0 means patch_size*downsample_factor from CONFIG.",
    )
    parser.add_argument(
        "--pore-value",
        type=int,
        choices=[0, 1],
        default=0,
        help="Binary value representing pore phase in decoded volume.",
    )
    parser.add_argument(
        "--gt-phi-semantic",
        choices=["porosity", "rock_rate"],
        default="porosity",
        help="Meaning of values in --phi-map.",
    )

    parser.add_argument("--run-infer", action="store_true", help="Run infer first to produce --pred-latent.")
    parser.add_argument("--ckpt", default=CONFIG.get("ckpt_path", ""))
    parser.add_argument("--ddim-steps", type=int, default=int(CONFIG.get("ddim_steps", 200)))
    parser.add_argument("--seed", type=int, default=int(CONFIG.get("seed", 6666)))
    parser.add_argument("--infer-use-ema", action="store_true", default=bool(CONFIG.get("infer_use_ema", False)))
    parser.add_argument("--sample-mode", choices=["sequential", "random"], default="sequential")
    parser.add_argument("--sample-seed", type=int, default=1234)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--num-samples", type=int, default=0, help="0 means all selected samples.")
    parser.add_argument("--save-per-sample", action="store_true", help="In batch mode, save per-sample files.")
    parser.add_argument(
        "--append-overall-porosity",
        action="store_true",
        help="In batch mode, append one overall-porosity row after each sample is evaluated.",
    )
    parser.add_argument(
        "--output-unscaled",
        action="store_true",
        default=bool(CONFIG.get("output_unscaled", True)),
        help="If --run-infer is used: divide generated latent by scale_factor before saving.",
    )
    args = parser.parse_args()

    phi_map_path = _resolve_path(args.phi_map, repo_root)
    phi_dir = _resolve_path(args.phi_dir, repo_root)
    pred_latent_path = _resolve_path(args.pred_latent, repo_root)
    pred_latent_dir = _resolve_path(args.pred_latent_dir, repo_root)
    vae_cfg = _resolve_path(args.vae_config, repo_root)
    vae_ckpt = _resolve_path(args.vae_ckpt, repo_root)
    ckpt_path = _resolve_path(args.ckpt, repo_root)

    is_batch = bool(phi_dir)
    if is_batch:
        if not os.path.isdir(phi_dir):
            raise FileNotFoundError(f"phi dir not found: {phi_dir}")
    else:
        if not phi_map_path or not os.path.exists(phi_map_path):
            raise FileNotFoundError(f"phi map not found: {phi_map_path}")
    if not os.path.exists(vae_cfg):
        raise FileNotFoundError(f"vae config not found: {vae_cfg}")
    if not os.path.exists(vae_ckpt):
        raise FileNotFoundError(f"vae checkpoint not found: {vae_ckpt}")

    if is_batch:
        out_dir = _resolve_path(args.out_dir, repo_root) if args.out_dir else str(repo_root / "eval_multitoken_phi_batch")
    else:
        basename = Path(phi_map_path).stem
        out_dir = _resolve_path(args.out_dir, repo_root) if args.out_dir else str(repo_root / f"eval_multitoken_phi_{basename}")
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device(args.device)
    patch_voxel = int(args.patch_voxel)
    if patch_voxel <= 0:
        patch_voxel = int(CONFIG.get("patch_size", 8)) * int(CONFIG.get("downsample_factor", 8))
    if patch_voxel <= 0:
        raise ValueError(f"Invalid patch_voxel={patch_voxel}")

    CONFIG["infer_use_ema"] = bool(args.infer_use_ema)
    model = None
    diffusion = None
    if bool(args.run_infer):
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"ckpt not found: {ckpt_path}")
        model = load_model(ckpt_path, device)
        diffusion = DiffusionHelper(int(CONFIG["timesteps"]), device)

    vae = load_vae(vae_cfg, vae_ckpt, device)

    sample_specs = []
    if is_batch:
        all_phi = _collect_npy_files(phi_dir)
        selected_phi = _select_files(
            all_files=all_phi,
            sample_mode=str(args.sample_mode),
            offset=int(args.offset),
            num_samples=int(args.num_samples),
            sample_seed=int(args.sample_seed),
        )
        if not selected_phi:
            raise ValueError("No phi files selected in batch mode.")
        if bool(args.run_infer):
            latent_out_dir = pred_latent_dir if pred_latent_dir else os.path.join(out_dir, "generated_latents")
            os.makedirs(latent_out_dir, exist_ok=True)
            for phi_path in selected_phi:
                name = Path(phi_path).stem
                pred_path = os.path.join(latent_out_dir, f"{name}.npy")
                sample_specs.append((phi_path, pred_path))
        else:
            if not pred_latent_dir:
                raise ValueError("Batch mode without --run-infer requires --pred-latent-dir.")
            for phi_path in selected_phi:
                name = Path(phi_path).stem
                pred_path = os.path.join(pred_latent_dir, f"{name}.npy")
                sample_specs.append((phi_path, pred_path))
    else:
        sample_specs.append((phi_map_path, pred_latent_path))

    all_rows: List[Dict] = []
    per_sample_rows: List[Dict] = []
    per_sample_details: List[Dict] = []
    porosity_csv_keys = _porosity_csv_keys()
    out_porosity_csv = os.path.join(out_dir, "overall_porosity.csv") if is_batch else ""
    out_porosity_summary = os.path.join(out_dir, "overall_porosity_summary.json") if is_batch else ""

    if is_batch and bool(args.append_overall_porosity):
        # Initialize file with header, then append one row per finished sample.
        _write_csv([], out_porosity_csv, keys=porosity_csv_keys)

    for i, (phi_path_i, pred_path_i) in enumerate(sample_specs):
        out_i = _evaluate_one(
            sample_index=i,
            phi_map_path=phi_path_i,
            pred_latent_path=pred_path_i,
            ckpt_path=ckpt_path,
            vae=vae,
            model=model,
            diffusion=diffusion,
            device=device,
            args=args,
            patch_voxel=patch_voxel,
        )
        rows_i = out_i["rows"]
        summary_row_i = out_i["summary_row"]
        detail_i = out_i["summary_detail"]

        all_rows.extend(rows_i)
        per_sample_rows.append(summary_row_i)
        per_sample_details.append(detail_i)

        if is_batch and bool(args.append_overall_porosity):
            sample_porosity_row = _porosity_row_from_sample_summary(summary_row_i)
            _append_csv_rows([sample_porosity_row], out_porosity_csv, porosity_csv_keys)
            running_summary = _build_porosity_analysis(per_sample_rows)["summary"]
            with open(out_porosity_summary, "w", encoding="utf-8") as f:
                json.dump(running_summary, f, indent=2, ensure_ascii=False)

        if bool(args.save_per_sample) and is_batch:
            sample_dir = os.path.join(out_dir, "samples", summary_row_i["sample_name"])
            os.makedirs(sample_dir, exist_ok=True)
            _write_csv(rows_i, os.path.join(sample_dir, "phi_cells.csv"))
            with open(os.path.join(sample_dir, "phi_cells.json"), "w", encoding="utf-8") as f:
                json.dump(rows_i, f, indent=2, ensure_ascii=False)
            with open(os.path.join(sample_dir, "summary.json"), "w", encoding="utf-8") as f:
                json.dump(detail_i, f, indent=2, ensure_ascii=False)

        print(
            f"[sample {i + 1}/{len(sample_specs)}] {summary_row_i['sample_name']} "
            f"cells={summary_row_i['num_cells']} bin_mae={summary_row_i['bin_mae']:.6f}"
        )

    if is_batch:
        out_cells_csv = os.path.join(out_dir, "phi_cells_all.csv")
        out_cells_json = os.path.join(out_dir, "phi_cells_all.json")
        out_sample_csv = os.path.join(out_dir, "per_sample_summary.csv")
        out_sample_jsonl = os.path.join(out_dir, "per_sample_summary.jsonl")
        out_summary = os.path.join(out_dir, "batch_summary.json")

        _write_csv(all_rows, out_cells_csv)
        with open(out_cells_json, "w", encoding="utf-8") as f:
            json.dump(all_rows, f, indent=2, ensure_ascii=False)
        _write_csv(per_sample_rows, out_sample_csv)
        with open(out_sample_jsonl, "w", encoding="utf-8") as f:
            for r in per_sample_rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        porosity_analysis = _build_porosity_analysis(per_sample_rows)
        if bool(args.append_overall_porosity):
            tail_rows = [r for r in porosity_analysis["csv_rows"] if str(r.get("row_type")) in ("batch_weighted", "batch_unweighted")]
            _append_csv_rows(tail_rows, out_porosity_csv, porosity_csv_keys)
        else:
            _write_csv(porosity_analysis["csv_rows"], out_porosity_csv)
        with open(out_porosity_summary, "w", encoding="utf-8") as f:
            json.dump(porosity_analysis["summary"], f, indent=2, ensure_ascii=False)

        batch_summary = {
            "mode": "batch",
            "num_samples": int(len(per_sample_rows)),
            "num_total_cells": int(len(all_rows)),
            "phi_dir": phi_dir,
            "pred_latent_dir": pred_latent_dir,
            "run_infer": bool(args.run_infer),
            "ckpt_path": ckpt_path if bool(args.run_infer) else "",
            "vae_config": vae_cfg,
            "vae_ckpt": vae_ckpt,
            "device": str(device),
            "patch_voxel": int(patch_voxel),
            "threshold": float(args.threshold),
            "pore_value": int(args.pore_value),
            "gt_phi_semantic": str(args.gt_phi_semantic),
            "sample_mode": str(args.sample_mode),
            "sample_seed": int(args.sample_seed),
            "offset": int(args.offset),
            "num_samples_arg": int(args.num_samples),
            "per_sample_stats": _summary_stats_from_rows(per_sample_rows),
            "cell_stats": _summary_stats_from_rows(all_rows),
        }
        with open(out_summary, "w", encoding="utf-8") as f:
            json.dump(batch_summary, f, indent=2, ensure_ascii=False)

        print("[done] batch multi-token porosity evaluation finished")
        print(f"  per-sample csv: {out_sample_csv}")
        print(f"  per-sample jsonl: {out_sample_jsonl}")
        print(f"  cell csv: {out_cells_csv}")
        print(f"  cell json: {out_cells_json}")
        print(f"  overall porosity csv: {out_porosity_csv}")
        print(f"  overall porosity summary: {out_porosity_summary}")
        print(f"  batch summary: {out_summary}")
    else:
        out_csv = os.path.join(out_dir, "phi_cells.csv")
        out_json = os.path.join(out_dir, "phi_cells.json")
        out_summary = os.path.join(out_dir, "summary.json")
        out_porosity_summary = os.path.join(out_dir, "overall_porosity_summary.json")

        _write_csv(all_rows, out_csv)
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(all_rows, f, indent=2, ensure_ascii=False)
        with open(out_summary, "w", encoding="utf-8") as f:
            json.dump(per_sample_details[0], f, indent=2, ensure_ascii=False)
        porosity_analysis = _build_porosity_analysis(per_sample_rows)
        with open(out_porosity_summary, "w", encoding="utf-8") as f:
            json.dump(porosity_analysis["summary"], f, indent=2, ensure_ascii=False)

        print("[done] multi-token porosity evaluation finished")
        print(f"  csv:     {out_csv}")
        print(f"  json:    {out_json}")
        print(f"  summary: {out_summary}")
        print(f"  overall porosity summary: {out_porosity_summary}")
        print(f"  cells:   {per_sample_rows[0]['num_cells']}")
        print(f"  gt_mean: {per_sample_rows[0]['gt_mean']:.6f}")
        print(f"  pred_bin_mean:  {per_sample_rows[0]['pred_bin_mean']:.6f}")
        print(f"  pred_prob_mean: {per_sample_rows[0]['pred_prob_mean']:.6f}")
        print(f"  bin_mae:  {per_sample_rows[0]['bin_mae']:.6f}")
        print(f"  prob_mae: {per_sample_rows[0]['prob_mae']:.6f}")


if __name__ == "__main__":
    main()

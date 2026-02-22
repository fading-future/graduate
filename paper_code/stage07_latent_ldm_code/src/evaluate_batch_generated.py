import argparse
import csv
import json
import logging
import os
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

from src.config import CONFIG
from src.diffusion import DiffusionHelper
from src.infer import generate_volume, load_model
from src.evaluate_generated_sample import (
    align_pair,
    compute_phi_map,
    decode_to_prob_and_bin,
    latent_metrics,
    load_latent,
    load_vae,
    phi_metrics,
    to_binary_volume,
    voxel_metrics,
)
from src.analysis_metrics import (
    absolute_permeability_openpnm,
    compare_two_point_probability,
    maybe_center_crop_3d,
    porosity_from_phase_mask,
    to_phase_mask,
)


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


def _gather_phi_files(phi_dir: str) -> List[str]:
    if not os.path.isdir(phi_dir):
        raise FileNotFoundError(f"phi_dir not found: {phi_dir}")
    files = sorted(
        os.path.join(phi_dir, fn)
        for fn in os.listdir(phi_dir)
        if fn.endswith(".npy")
    )
    if len(files) == 0:
        raise ValueError(f"No .npy files found in phi_dir: {phi_dir}")
    return files


def _deterministic_seed(base_seed: Optional[int], mode: str, sample_idx: int, basename: str) -> Optional[int]:
    if base_seed is None:
        return None
    if mode == "fixed":
        return int(base_seed)
    if mode == "offset":
        return int(base_seed) + int(sample_idx)
    # mode == "name_hash"
    h = 2166136261
    for ch in basename:
        h ^= ord(ch)
        h = (h * 16777619) & 0xFFFFFFFF
    return int(base_seed) + int(h % 1000003)


def _safe_float(v):
    if isinstance(v, (np.floating, np.integer)):
        return float(v)
    return v


def _parity_stats_from_xy(x: np.ndarray, y: np.ndarray, prefix: str) -> Dict:
    out: Dict = {}
    if x.size == 0 or y.size == 0:
        return out
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m].astype(np.float64)
    y = y[m].astype(np.float64)
    if x.size == 0:
        return out

    xm = float(x.mean())
    ym = float(y.mean())
    dx = x - xm
    dy = y - ym
    vx = float(np.sum(dx * dx))
    vy = float(np.sum(dy * dy))
    cov = float(np.sum(dx * dy))

    corr = cov / (np.sqrt(vx * vy) + 1e-12) if vx > 0.0 and vy > 0.0 else 0.0
    slope = cov / (vx + 1e-12) if vx > 0.0 else 0.0
    intercept = ym - slope * xm
    bias = ym - xm
    mae = float(np.mean(np.abs(y - x)))
    rmse = float(np.sqrt(np.mean((y - x) ** 2)))

    out[f"{prefix}_corr"] = float(corr)
    out[f"{prefix}_slope"] = float(slope)
    out[f"{prefix}_intercept"] = float(intercept)
    out[f"{prefix}_bias"] = float(bias)
    out[f"{prefix}_mae"] = float(mae)
    out[f"{prefix}_rmse"] = float(rmse)
    out[f"{prefix}_slope_gap"] = float(slope - 1.0)
    out[f"{prefix}_n"] = int(x.size)
    return out


def _axes_from_text(text: str) -> List[int]:
    t = str(text).lower()
    out: List[int] = []
    for ch in t:
        if ch == "x" and 0 not in out:
            out.append(0)
        elif ch == "y" and 1 not in out:
            out.append(1)
        elif ch == "z" and 2 not in out:
            out.append(2)
    return out


def _normalize_infer_direction_token(token: str) -> str:
    """
    Normalize direction token to +/- form for (i, j, k).
    Supported:
      - '+-+' / '-+-'
      - 'pmp' where p='+', m='-'
    """
    t = str(token).strip().lower().replace(" ", "").replace(",", "")
    if len(t) != 3:
        raise ValueError("infer direction must have exactly 3 symbols, e.g. '+-+' or 'pmp'.")
    if set(t).issubset({"+", "-"}):
        return t
    if set(t).issubset({"p", "m"}):
        return "".join("+" if c == "p" else "-" for c in t)
    raise ValueError("invalid infer direction. Use '+/-' (e.g. '+-+') or 'p/m' (e.g. 'pmp').")


def _configure_third_party_warnings(quiet: bool):
    if not quiet:
        return
    # PoreSpy currently emits many deprecation warnings from skimage internals.
    warnings.filterwarnings("ignore", category=FutureWarning, module=r"porespy\..*")
    warnings.filterwarnings("ignore", category=FutureWarning, module=r"skimage\..*")

    # Reduce OpenPNM informational/warning logs (e.g. PARDISO availability notice).
    logging.getLogger("openpnm").setLevel(logging.ERROR)
    logging.getLogger("openpnm.utils").setLevel(logging.ERROR)
    logging.getLogger("openpnm.utils._workspace").setLevel(logging.ERROR)


def _flatten_phi_cells(
    sample_index: int,
    basename: str,
    pred_phi_bin: np.ndarray,
    pred_phi_prob: np.ndarray,
    gt_phi: np.ndarray,
) -> List[Dict]:
    rows: List[Dict] = []
    g_d, g_h, g_w = gt_phi.shape
    for i in range(g_d):
        for j in range(g_h):
            for k in range(g_w):
                gt = float(gt_phi[i, j, k])
                pb = float(pred_phi_bin[i, j, k])
                pp = float(pred_phi_prob[i, j, k])
                rows.append(
                    {
                        "sample_index": int(sample_index),
                        "basename": basename,
                        "cell_i": int(i),
                        "cell_j": int(j),
                        "cell_k": int(k),
                        "gt_phi": gt,
                        "pred_phi_bin": pb,
                        "pred_phi_prob": pp,
                        "bin_abs_err": float(abs(pb - gt)),
                        "bin_bias": float(pb - gt),
                        "prob_abs_err": float(abs(pp - gt)),
                        "prob_bias": float(pp - gt),
                    }
                )
    return rows


def _write_phi_cell_reports(rows: List[Dict], out_dir: str) -> Tuple[str, str, str]:
    cell_csv = os.path.join(out_dir, "phi_cell_metrics.csv")
    cell_jsonl = os.path.join(out_dir, "phi_cell_metrics.jsonl")
    cell_summary_csv = os.path.join(out_dir, "phi_cell_summary.csv")

    # per-cell-per-sample table
    keys = [
        "sample_index",
        "basename",
        "cell_i",
        "cell_j",
        "cell_k",
        "gt_phi",
        "pred_phi_bin",
        "pred_phi_prob",
        "bin_abs_err",
        "bin_bias",
        "prob_abs_err",
        "prob_bias",
    ]
    with open(cell_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: _safe_float(r.get(k, "")) for k in keys})

    with open(cell_jsonl, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps({k: _safe_float(v) for k, v in r.items()}, ensure_ascii=False) + "\n")

    # aggregate by cell location (across samples)
    buckets = {}
    for r in rows:
        key = (int(r["cell_i"]), int(r["cell_j"]), int(r["cell_k"]))
        buckets.setdefault(key, []).append(r)

    summary_rows: List[Dict] = []
    for (i, j, k), rs in sorted(buckets.items()):
        gt = np.array([float(x["gt_phi"]) for x in rs], dtype=np.float64)
        pb = np.array([float(x["pred_phi_bin"]) for x in rs], dtype=np.float64)
        pp = np.array([float(x["pred_phi_prob"]) for x in rs], dtype=np.float64)
        summary_rows.append(
            {
                "cell_i": i,
                "cell_j": j,
                "cell_k": k,
                "n": int(len(rs)),
                "gt_phi_mean": float(gt.mean()),
                "pred_phi_bin_mean": float(pb.mean()),
                "pred_phi_prob_mean": float(pp.mean()),
                "bin_mae_mean": float(np.mean(np.abs(pb - gt))),
                "bin_bias_mean": float(np.mean(pb - gt)),
                "prob_mae_mean": float(np.mean(np.abs(pp - gt))),
                "prob_bias_mean": float(np.mean(pp - gt)),
            }
        )

    with open(cell_summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()) if summary_rows else [])
        if summary_rows:
            writer.writeheader()
            for r in summary_rows:
                writer.writerow({k: _safe_float(v) for k, v in r.items()})

    return cell_csv, cell_jsonl, cell_summary_csv


def _evaluate_one(
    phi_path: str,
    sample_idx: int,
    model,
    diffusion: DiffusionHelper,
    vae,
    args,
    do_physics: bool = False,
) -> Tuple[Dict, List[Dict]]:
    basename = os.path.basename(phi_path)
    sample_seed = _deterministic_seed(args.seed, args.seed_mode, sample_idx, basename)

    phi_map = np.load(phi_path).astype(np.float32)
    t0 = time.time()
    pred_latent = generate_volume(
        phi_map=phi_map,
        model=model,
        diffusion=diffusion,
        patch_size=int(CONFIG["patch_size"]),
        window_size=int(CONFIG["window_size"]),
        steps=int(args.ddim_steps),
        seed=sample_seed,
    )
    if bool(CONFIG.get("output_unscaled", True)):
        scale = float(CONFIG.get("scale_factor", 1.0))
        if scale not in (0.0, 1.0):
            pred_latent = pred_latent / scale

    pred_prob, pred_bin = decode_to_prob_and_bin(
        pred_latent.astype(np.float32),
        vae=vae,
        device=torch.device(args.device),
        threshold=float(args.threshold),
    )
    phase_value = int(args.pore_value)
    pred_phase_bin = to_phase_mask(pred_bin, phase_value=phase_value)
    pred_phase_prob = (1.0 - pred_prob).astype(np.float32) if phase_value == 0 else pred_prob.astype(np.float32)
    t1 = time.time()

    gt_lat_path = os.path.join(args.latent_dir, basename) if args.latent_dir else ""
    if not os.path.exists(gt_lat_path):
        gt_lat_path = ""
    gt_vox_path = os.path.join(args.raw_dir, basename) if args.raw_dir else ""
    if not os.path.exists(gt_vox_path):
        gt_vox_path = ""
    gt_phi_path = os.path.join(args.phi_dir, basename)
    if not os.path.exists(gt_phi_path):
        gt_phi_path = ""

    out = {
        "sample_index": int(sample_idx),
        "basename": basename,
        "seed": sample_seed if sample_seed is not None else "None",
        "time_sec": float(t1 - t0),
        "pred_latent_mean": float(pred_latent.mean()),
        "pred_latent_std": float(pred_latent.std()),
        "pred_prob_mean": float(pred_prob.mean()),
        "pred_prob_std": float(pred_prob.std()),
        "pred_bin_porosity": float(pred_bin.mean()),
        "phase_value": phase_value,
        "target_phase_fraction_pred": float(porosity_from_phase_mask(pred_phase_bin)),
        "gt_latent_path": gt_lat_path,
        "gt_voxel_path": gt_vox_path,
        "gt_phi_path": gt_phi_path,
    }
    if phase_value == 0:
        out["pore_porosity_pred"] = out["target_phase_fraction_pred"]
    phi_cell_rows: List[Dict] = []
    tp2_pred_curve = None
    tp2_gt_curve = None

    if gt_lat_path:
        gt_lat = load_latent(gt_lat_path)
        gt_lat_unscaled = gt_lat.astype(np.float32)
        pred_lat_cmp, gt_lat_cmp = align_pair(pred_latent.astype(np.float32), gt_lat_unscaled)
        out.update(latent_metrics(pred_lat_cmp, gt_lat_cmp))

    gt_bin = None
    if gt_vox_path:
        gt_vox = np.load(gt_vox_path)
        gt_bin = to_binary_volume(gt_vox)
        pred_bin_al, gt_bin_al = align_pair(pred_bin, gt_bin)
        pred_phase_al, gt_phase_al = align_pair(pred_phase_bin, to_phase_mask(gt_bin, phase_value=phase_value))
        out.update(voxel_metrics(pred_bin_al, gt_bin_al))
        out["target_phase_fraction_pred_aligned"] = float(porosity_from_phase_mask(pred_phase_al))
        out["target_phase_fraction_gt_aligned"] = float(porosity_from_phase_mask(gt_phase_al))
        out["target_phase_fraction_abs_err"] = float(
            abs(out["target_phase_fraction_pred_aligned"] - out["target_phase_fraction_gt_aligned"])
        )
        if phase_value == 0:
            out["pore_porosity_pred_aligned"] = out["target_phase_fraction_pred_aligned"]
            out["pore_porosity_gt_aligned"] = out["target_phase_fraction_gt_aligned"]
            out["pore_porosity_abs_err"] = out["target_phase_fraction_abs_err"]
        if int(args.tp2_max_lag) > 0:
            if str(args.tp2_phase).lower() == "phase":
                tp2 = compare_two_point_probability(pred_phase_al, gt_phase_al, max_lag=int(args.tp2_max_lag))
            else:
                tp2 = compare_two_point_probability(pred_bin_al, gt_bin_al, max_lag=int(args.tp2_max_lag))
            out.update(tp2["metrics"])
            tp2_pred_curve = tp2["pred_curve"]
            tp2_gt_curve = tp2["gt_curve"]
        # boundary diagnostics: check whether early-z slices are systematically biased
        head = min(16, pred_bin_al.shape[0])
        tail = min(16, pred_bin_al.shape[0])
        out["z_head_porosity_pred"] = float(pred_bin_al[:head].mean())
        out["z_head_porosity_gt"] = float(gt_bin_al[:head].mean())
        out["z_head_porosity_gap"] = float(out["z_head_porosity_pred"] - out["z_head_porosity_gt"])
        out["z_tail_porosity_pred"] = float(pred_bin_al[-tail:].mean())
        out["z_tail_porosity_gt"] = float(gt_bin_al[-tail:].mean())
        out["z_tail_porosity_gap"] = float(out["z_tail_porosity_pred"] - out["z_tail_porosity_gt"])
        out["z_head_phase_gap"] = float(pred_phase_al[:head].mean() - gt_phase_al[:head].mean())
        out["z_tail_phase_gap"] = float(pred_phase_al[-tail:].mean() - gt_phase_al[-tail:].mean())
        if phase_value == 0:
            out["z_head_pore_gap"] = out["z_head_phase_gap"]
            out["z_tail_pore_gap"] = out["z_tail_phase_gap"]

        if do_physics:
            axes = _axes_from_text(args.physics_axes)
            pred_phys = maybe_center_crop_3d(pred_phase_al.astype(np.uint8), int(args.physics_crop))
            gt_phys = maybe_center_crop_3d(gt_phase_al.astype(np.uint8), int(args.physics_crop))
            k_pred_vals = []
            k_gt_vals = []
            for ax in axes:
                k_pred = absolute_permeability_openpnm(
                    pred_phys,
                    axis=ax,
                    mu=float(args.physics_mu),
                    dp=float(args.physics_dp),
                )
                k_gt = absolute_permeability_openpnm(
                    gt_phys,
                    axis=ax,
                    mu=float(args.physics_mu),
                    dp=float(args.physics_dp),
                )
                ax_tag = "xyz"[ax]
                out[f"kabs_pred_{ax_tag}_ok"] = bool(k_pred.get("ok", False))
                out[f"kabs_gt_{ax_tag}_ok"] = bool(k_gt.get("ok", False))
                if bool(k_pred.get("ok", False)):
                    out[f"kabs_pred_{ax_tag}"] = float(k_pred["k_abs_voxel2"])
                    k_pred_vals.append(float(k_pred["k_abs_voxel2"]))
                else:
                    out[f"kabs_pred_{ax_tag}_err"] = str(k_pred.get("error", "unknown"))
                if bool(k_gt.get("ok", False)):
                    out[f"kabs_gt_{ax_tag}"] = float(k_gt["k_abs_voxel2"])
                    k_gt_vals.append(float(k_gt["k_abs_voxel2"]))
                else:
                    out[f"kabs_gt_{ax_tag}_err"] = str(k_gt.get("error", "unknown"))

                if bool(k_pred.get("ok", False)) and bool(k_gt.get("ok", False)):
                    kp = float(k_pred["k_abs_voxel2"])
                    kg = float(k_gt["k_abs_voxel2"])
                    out[f"kabs_abs_err_{ax_tag}"] = float(abs(kp - kg))
                    out[f"kabs_rel_err_{ax_tag}"] = float(abs(kp - kg) / (abs(kg) + 1e-12))
                    out[f"kabs_ratio_pred_over_gt_{ax_tag}"] = float(kp / (kg + 1e-12))

            if len(k_pred_vals) > 0:
                out["kabs_pred_mean"] = float(np.mean(k_pred_vals))
            if len(k_gt_vals) > 0:
                out["kabs_gt_mean"] = float(np.mean(k_gt_vals))
            if ("kabs_pred_mean" in out) and ("kabs_gt_mean" in out):
                out["kabs_abs_err_mean"] = float(abs(out["kabs_pred_mean"] - out["kabs_gt_mean"]))
                out["kabs_rel_err_mean"] = float(
                    abs(out["kabs_pred_mean"] - out["kabs_gt_mean"]) / (abs(out["kabs_gt_mean"]) + 1e-12)
                )

    patch_voxel = int(CONFIG.get("patch_size", 8)) * int(CONFIG.get("downsample_factor", 8))
    pred_phi_bin = compute_phi_map(pred_bin.astype(np.float32), patch_voxel)
    pred_phi_prob = compute_phi_map(pred_prob.astype(np.float32), patch_voxel)
    pred_phi_bin_phase = compute_phi_map(pred_phase_bin.astype(np.float32), patch_voxel)
    pred_phi_prob_phase = compute_phi_map(pred_phase_prob.astype(np.float32), patch_voxel)

    gt_phi = None
    if gt_phi_path:
        gt_phi = np.load(gt_phi_path).astype(np.float32)
    elif gt_bin is not None:
        gt_phi = compute_phi_map(gt_bin.astype(np.float32), patch_voxel)

    if gt_phi is not None:
        pred_phi_bin_al, gt_phi_al = align_pair(pred_phi_bin, gt_phi)
        pred_phi_prob_al, _ = align_pair(pred_phi_prob, gt_phi)
        out.update(phi_metrics(pred_phi_bin_al, gt_phi_al, prefix="bin"))
        out.update(phi_metrics(pred_phi_prob_al, gt_phi_al, prefix="prob"))
        out["phi_layer0_mae"] = float(np.mean(np.abs(pred_phi_bin_al[0] - gt_phi_al[0])))
        out["phi_layer0_bias"] = float(np.mean(pred_phi_bin_al[0] - gt_phi_al[0]))
        out["phi_layer_last_mae"] = float(np.mean(np.abs(pred_phi_bin_al[-1] - gt_phi_al[-1])))
        out["phi_layer_last_bias"] = float(np.mean(pred_phi_bin_al[-1] - gt_phi_al[-1]))

        gt_phi_phase = gt_phi.astype(np.float32)
        gt_sem = str(args.gt_phi_semantic).lower()
        if gt_sem == "rock_rate":
            gt_phi_phase = 1.0 - gt_phi_phase if phase_value == 0 else gt_phi_phase
        else:  # gt_sem == "porosity"
            gt_phi_phase = gt_phi_phase if phase_value == 0 else 1.0 - gt_phi_phase

        pred_phi_bin_phase_al, gt_phi_phase_al = align_pair(pred_phi_bin_phase, gt_phi_phase)
        pred_phi_prob_phase_al, _ = align_pair(pred_phi_prob_phase, gt_phi_phase)
        out.update(phi_metrics(pred_phi_bin_phase_al, gt_phi_phase_al, prefix="phase_bin"))
        out.update(phi_metrics(pred_phi_prob_phase_al, gt_phi_phase_al, prefix="phase_prob"))
        if phase_value == 0:
            out["pore_phi_corr"] = float(out.get("phase_bin_phi_corr", 0.0))
            out["pore_phi_mae"] = float(out.get("phase_bin_phi_mae", 0.0))
            out["pore_phi_rmse"] = float(out.get("phase_bin_phi_rmse", 0.0))

        if bool(args.export_phi_cells):
            phi_cell_rows = _flatten_phi_cells(
                sample_index=sample_idx,
                basename=basename,
                pred_phi_bin=pred_phi_bin_phase_al,
                pred_phi_prob=pred_phi_prob_phase_al,
                gt_phi=gt_phi_phase_al,
            )

    if args.save_each:
        sample_dir = os.path.join(args.out_dir, "samples", os.path.splitext(basename)[0])
        os.makedirs(sample_dir, exist_ok=True)
        np.save(os.path.join(sample_dir, "pred_latent.npy"), pred_latent.astype(np.float32))
        np.save(os.path.join(sample_dir, "pred_voxel_prob.npy"), pred_prob.astype(np.float32))
        np.save(os.path.join(sample_dir, "pred_voxel_bin.npy"), pred_bin.astype(np.uint8))
        np.save(os.path.join(sample_dir, "pred_phi_from_bin.npy"), pred_phi_bin.astype(np.float32))
        np.save(os.path.join(sample_dir, "pred_phi_from_prob.npy"), pred_phi_prob.astype(np.float32))
        np.save(os.path.join(sample_dir, "pred_phase_bin.npy"), pred_phase_bin.astype(np.uint8))
        np.save(os.path.join(sample_dir, "pred_phi_from_phase_bin.npy"), pred_phi_bin_phase.astype(np.float32))
        np.save(os.path.join(sample_dir, "pred_phi_from_phase_prob.npy"), pred_phi_prob_phase.astype(np.float32))
        if tp2_pred_curve is not None and tp2_gt_curve is not None and bool(args.tp2_save_curves):
            np.savez(
                os.path.join(sample_dir, "tp2_curves.npz"),
                lag=tp2_pred_curve["lag"],
                pred_x=tp2_pred_curve["x"],
                pred_y=tp2_pred_curve["y"],
                pred_z=tp2_pred_curve["z"],
                pred_mean=tp2_pred_curve["mean"],
                gt_x=tp2_gt_curve["x"],
                gt_y=tp2_gt_curve["y"],
                gt_z=tp2_gt_curve["z"],
                gt_mean=tp2_gt_curve["mean"],
            )
        with open(os.path.join(sample_dir, "metrics.json"), "w", encoding="utf-8") as f:
            json.dump({k: _safe_float(v) for k, v in out.items()}, f, indent=2, ensure_ascii=False)

    return out, phi_cell_rows


def _write_csv(rows: List[Dict], out_csv: str):
    if len(rows) == 0:
        return
    all_keys = sorted(set().union(*(r.keys() for r in rows)))
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: _safe_float(r.get(k, "")) for k in all_keys})


def _summary(rows: List[Dict]) -> Dict:
    if len(rows) == 0:
        return {"num_samples": 0}
    summary = {"num_samples": len(rows)}
    numeric_keys = []
    for k in rows[0].keys():
        vals = [r.get(k, None) for r in rows]
        if all(isinstance(v, (int, float, np.floating, np.integer)) for v in vals if v is not None):
            numeric_keys.append(k)
    for k in sorted(numeric_keys):
        vals = [float(r[k]) for r in rows if r.get(k, None) is not None]
        if len(vals) == 0:
            continue
        arr = np.array(vals, dtype=np.float64)
        summary[f"{k}_mean"] = float(arr.mean())
        summary[f"{k}_std"] = float(arr.std())
        summary[f"{k}_min"] = float(arr.min())
        summary[f"{k}_max"] = float(arr.max())
    return summary


def main():
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Batch evaluate Stage07 generation: phi_map -> infer -> decode -> metrics."
    )
    parser.add_argument("--phi-dir", default=CONFIG.get("phi_map_dir", ""))
    parser.add_argument("--latent-dir", default=CONFIG.get("latent_dir", ""))
    parser.add_argument("--raw-dir", default=CONFIG.get("raw_data_dir", ""))
    parser.add_argument("--ckpt", default=CONFIG.get("ckpt_path", ""))
    parser.add_argument("--vae-config", default=CONFIG.get("eval_vae_config_path", ""))
    parser.add_argument("--vae-ckpt", default=CONFIG.get("eval_vae_ckpt_path", ""))
    parser.add_argument("--out-dir", default=str(repo_root / "eval_batch"))
    parser.add_argument("--device", default=CONFIG.get("device", "cpu"))
    parser.add_argument(
        "--show-third-party-warnings",
        action="store_true",
        help="Show PoreSpy/OpenPNM warning logs (default is quiet mode).",
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--pore-value",
        type=int,
        choices=[0, 1],
        default=0,
        help="Value that represents pore/target phase in binary volumes. Default 0.",
    )
    parser.add_argument(
        "--gt-phi-semantic",
        choices=["rock_rate", "porosity"],
        default="rock_rate",
        help="Semantic of gt phi map values: rock_rate means phi=1 is rock fraction.",
    )
    parser.add_argument("--ddim-steps", type=int, default=int(CONFIG.get("ddim_steps", 200)))
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--num-samples", type=int, default=0, help="0 means all from offset.")
    parser.add_argument(
        "--sample-mode",
        choices=["sequential", "random"],
        default="sequential",
        help="How to select samples from phi-dir.",
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=1234,
        help="Random seed used when sample-mode=random.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=int(CONFIG.get("seed", 1234)) if CONFIG.get("seed", None) is not None else None,
    )
    parser.add_argument(
        "--seed-mode",
        choices=["fixed", "offset", "name_hash"],
        default="name_hash",
        help="fixed: same seed for all; offset: seed+index; name_hash: seed+hash(filename).",
    )
    parser.add_argument("--save-each", action="store_true", help="Save per-sample npy and metrics.")
    parser.add_argument(
        "--export-phi-cells",
        action="store_true",
        help="Export per-cell phi comparison table (gt vs pred) across all evaluated samples.",
    )
    parser.add_argument(
        "--tp2-max-lag",
        type=int,
        default=0,
        help="If >0, compute two-point probability metrics on voxel binaries up to this lag.",
    )
    parser.add_argument(
        "--tp2-phase",
        choices=["raw", "phase"],
        default="phase",
        help="raw: use original binary values; phase: use mask converted by --pore-value.",
    )
    parser.add_argument(
        "--tp2-save-curves",
        action="store_true",
        help="When --save-each is enabled, also save per-sample tp2 curves (.npz).",
    )
    parser.add_argument(
        "--physics-abs-k",
        action="store_true",
        help="Compute absolute permeability (OpenPNM StokesFlow) on target phase mask.",
    )
    parser.add_argument(
        "--physics-max-samples",
        type=int,
        default=0,
        help="Compute physics metrics only for first N evaluated samples. 0 means all.",
    )
    parser.add_argument(
        "--physics-crop",
        type=int,
        default=0,
        help="Center-crop size before physics simulation. 0 means no crop.",
    )
    parser.add_argument(
        "--physics-axes",
        default="xyz",
        help="Axes for absolute permeability simulation, e.g. x / yz / xyz.",
    )
    parser.add_argument("--physics-mu", type=float, default=1.0, help="Dynamic viscosity used in StokesFlow.")
    parser.add_argument("--physics-dp", type=float, default=1.0, help="Pressure drop used in StokesFlow.")
    parser.add_argument(
        "--infer-weight-source",
        choices=["config", "ema", "model"],
        default="config",
        help="Which checkpoint weights to use in inference.",
    )
    parser.add_argument(
        "--infer-order",
        default=None,
        help="Override fixed causal order for inference (one of ijk/ikj/jik/jki/kij/kji).",
    )
    parser.add_argument(
        "--infer-direction",
        default=None,
        help=(
            "Override fixed traversal direction for (i,j,k), e.g. +++, --+, +-+. "
            "If value starts with '-', pass as --infer-direction=-+-."
        ),
    )
    parser.add_argument(
        "--infer-direction-code",
        default=None,
        help="Alternative direction encoding without leading '-': use p/m, e.g. pmp -> +-+, mpm -> -+-.",
    )
    parser.add_argument(
        "--infer-random-order",
        action="store_true",
        help="Randomize axis order per sample (deterministic under sample seed).",
    )
    parser.add_argument(
        "--infer-random-direction",
        action="store_true",
        help="Randomize traversal direction (+/- for each axis) per sample.",
    )
    args = parser.parse_args()
    _configure_third_party_warnings(quiet=not bool(args.show_third_party_warnings))

    args.phi_dir = _resolve_path(args.phi_dir, repo_root)
    args.latent_dir = _resolve_path(args.latent_dir, repo_root)
    args.raw_dir = _resolve_path(args.raw_dir, repo_root)
    args.ckpt = _resolve_path(args.ckpt, repo_root)
    args.vae_config = _resolve_path(args.vae_config, repo_root)
    args.vae_ckpt = _resolve_path(args.vae_ckpt, repo_root)
    args.out_dir = _resolve_path(args.out_dir, repo_root)
    os.makedirs(args.out_dir, exist_ok=True)

    if args.infer_weight_source == "ema":
        CONFIG["infer_use_ema"] = True
    elif args.infer_weight_source == "model":
        CONFIG["infer_use_ema"] = False
    if args.infer_order is not None:
        CONFIG["order"] = str(args.infer_order)
    if args.infer_direction is not None or args.infer_direction_code is not None:
        direction_token = args.infer_direction if args.infer_direction is not None else args.infer_direction_code
        try:
            CONFIG["infer_direction"] = _normalize_infer_direction_token(direction_token)
        except ValueError as exc:
            parser.error(str(exc))

    # Keep CLI behavior intuitive:
    # - Explicit random flags enable random traversal.
    # - If fixed order/direction is explicitly provided, disable the corresponding random mode.
    # - Otherwise, preserve config defaults.
    if bool(args.infer_random_order):
        CONFIG["infer_random_order"] = True
    elif args.infer_order is not None:
        CONFIG["infer_random_order"] = False

    if bool(args.infer_random_direction):
        CONFIG["infer_random_direction"] = True
    elif args.infer_direction is not None or args.infer_direction_code is not None:
        CONFIG["infer_random_direction"] = False
    print(
        "[eval] infer settings:"
        f" random_order={bool(CONFIG.get('infer_random_order', False))},"
        f" random_direction={bool(CONFIG.get('infer_random_direction', False))},"
        f" order={CONFIG.get('order', 'ijk')},"
        f" direction={CONFIG.get('infer_direction', '+++')}"
    )

    if not os.path.exists(args.ckpt):
        raise FileNotFoundError(f"ckpt not found: {args.ckpt}")
    if not os.path.exists(args.vae_config):
        raise FileNotFoundError(f"vae config not found: {args.vae_config}")
    if not os.path.exists(args.vae_ckpt):
        raise FileNotFoundError(f"vae ckpt not found: {args.vae_ckpt}")

    all_phi_files = _gather_phi_files(args.phi_dir)
    if args.sample_mode == "sequential":
        phi_files = all_phi_files[int(args.offset):]
        if args.num_samples > 0:
            phi_files = phi_files[: int(args.num_samples)]
    else:
        rng = np.random.default_rng(int(args.sample_seed))
        perm = rng.permutation(len(all_phi_files))
        perm = perm[int(args.offset):]
        if args.num_samples > 0:
            perm = perm[: int(args.num_samples)]
        phi_files = [all_phi_files[int(i)] for i in perm]
    if len(phi_files) == 0:
        raise ValueError("No phi files selected for evaluation.")

    device = torch.device(args.device)
    model = load_model(args.ckpt, device)
    diffusion = DiffusionHelper(CONFIG["timesteps"], device)
    vae = load_vae(args.vae_config, args.vae_ckpt, device)

    rows: List[Dict] = []
    phi_cell_rows_all: List[Dict] = []
    pbar = tqdm(range(len(phi_files)), desc="BatchEval")
    for i in pbar:
        phi_path = phi_files[i]
        do_physics = bool(args.physics_abs_k) and (
            int(args.physics_max_samples) <= 0 or i < int(args.physics_max_samples)
        )
        row, phi_rows = _evaluate_one(
            phi_path=phi_path,
            sample_idx=i + int(args.offset),
            model=model,
            diffusion=diffusion,
            vae=vae,
            args=args,
            do_physics=do_physics,
        )
        rows.append(row)
        if bool(args.export_phi_cells) and len(phi_rows) > 0:
            phi_cell_rows_all.extend(phi_rows)
        if "voxel_dice" in row:
            pbar.set_postfix(
                dice=f"{row['voxel_dice']:.3f}",
                phi_corr=f"{row.get('bin_phi_corr', np.nan):.3f}",
                por_err=f"{row.get('porosity_abs_err', np.nan):.3f}",
            )
        else:
            pbar.set_postfix(por=row.get("pred_bin_porosity", np.nan))

    out_csv = os.path.join(args.out_dir, "per_sample_metrics.csv")
    _write_csv(rows, out_csv)
    out_jsonl = os.path.join(args.out_dir, "per_sample_metrics.jsonl")
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps({k: _safe_float(v) for k, v in r.items()}, ensure_ascii=False) + "\n")

    phi_cell_csv = ""
    phi_cell_jsonl = ""
    phi_cell_summary_csv = ""
    if bool(args.export_phi_cells) and len(phi_cell_rows_all) > 0:
        phi_cell_csv, phi_cell_jsonl, phi_cell_summary_csv = _write_phi_cell_reports(phi_cell_rows_all, args.out_dir)

    summ = _summary(rows)
    summ["ckpt"] = args.ckpt
    summ["phi_dir"] = args.phi_dir
    summ["num_selected"] = len(phi_files)
    summ["sample_mode"] = args.sample_mode
    summ["sample_seed"] = int(args.sample_seed)
    summ["seed_mode"] = args.seed_mode
    summ["threshold"] = float(args.threshold)
    summ["pore_value"] = int(args.pore_value)
    summ["gt_phi_semantic"] = str(args.gt_phi_semantic)
    summ["ddim_steps"] = int(args.ddim_steps)
    summ["infer_use_ema"] = bool(CONFIG.get("infer_use_ema", False))
    summ["infer_order"] = str(CONFIG.get("order", "ijk"))
    summ["infer_direction"] = str(CONFIG.get("infer_direction", "+++"))
    summ["infer_random_order"] = bool(CONFIG.get("infer_random_order", False))
    summ["infer_random_direction"] = bool(CONFIG.get("infer_random_direction", False))
    summ["tp2_max_lag"] = int(args.tp2_max_lag)
    summ["tp2_phase"] = str(args.tp2_phase)
    summ["export_phi_cells"] = bool(args.export_phi_cells)
    summ["physics_abs_k"] = bool(args.physics_abs_k)
    summ["physics_max_samples"] = int(args.physics_max_samples)
    summ["physics_crop"] = int(args.physics_crop)
    summ["physics_axes"] = str(args.physics_axes)

    # Batch-level pore parity diagnostics: y=Pred pore fraction, x=GT pore fraction.
    px_vals: List[float] = []
    py_vals: List[float] = []
    for r in rows:
        gx = _safe_float(r.get("pore_porosity_gt_aligned", None))
        gy = _safe_float(r.get("pore_porosity_pred_aligned", None))
        if gx is None or gy is None:
            continue
        px_vals.append(float(gx))
        py_vals.append(float(gy))
    px = np.array(px_vals, dtype=np.float64)
    py = np.array(py_vals, dtype=np.float64)
    summ.update(_parity_stats_from_xy(px, py, prefix="pore_parity"))

    if len(phi_cell_rows_all) > 0:
        bin_abs = np.array([float(r["bin_abs_err"]) for r in phi_cell_rows_all], dtype=np.float64)
        bin_bias = np.array([float(r["bin_bias"]) for r in phi_cell_rows_all], dtype=np.float64)
        prob_abs = np.array([float(r["prob_abs_err"]) for r in phi_cell_rows_all], dtype=np.float64)
        prob_bias = np.array([float(r["prob_bias"]) for r in phi_cell_rows_all], dtype=np.float64)
        summ["phi_cell_bin_abs_err_mean"] = float(bin_abs.mean())
        summ["phi_cell_bin_bias_mean"] = float(bin_bias.mean())
        summ["phi_cell_prob_abs_err_mean"] = float(prob_abs.mean())
        summ["phi_cell_prob_bias_mean"] = float(prob_bias.mean())
    summ_path = os.path.join(args.out_dir, "summary.json")
    with open(summ_path, "w", encoding="utf-8") as f:
        json.dump(summ, f, indent=2, ensure_ascii=False)

    print("[done] batch evaluation finished")
    print(f"  per-sample csv:   {out_csv}")
    print(f"  per-sample jsonl: {out_jsonl}")
    print(f"  summary:          {summ_path}")
    if phi_cell_csv:
        print(f"  phi-cell csv:     {phi_cell_csv}")
        print(f"  phi-cell jsonl:   {phi_cell_jsonl}")
        print(f"  phi-cell summary: {phi_cell_summary_csv}")
    for key in [
        "voxel_dice_mean",
        "voxel_iou_mean",
        "porosity_abs_err_mean",
        "pore_porosity_abs_err_mean",
        "target_phase_fraction_abs_err_mean",
        "bin_phi_corr_mean",
        "phase_bin_phi_corr_mean",
        "bin_phi_mae_mean",
        "phase_bin_phi_mae_mean",
        "pore_parity_corr",
        "pore_parity_slope",
        "pore_parity_slope_gap",
        "pore_parity_bias",
        "tp2_corr_mean",
        "tp2_mae_mean",
        "kabs_abs_err_mean_mean",
        "latent_mae_mean",
        "time_sec_mean",
    ]:
        if key in summ:
            print(f"  {key}: {summ[key]:.6f}")


if __name__ == "__main__":
    main()

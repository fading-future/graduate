import argparse
import csv
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

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


def _evaluate_one(
    phi_path: str,
    sample_idx: int,
    model,
    diffusion: DiffusionHelper,
    vae,
    args,
) -> Dict:
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
        "gt_latent_path": gt_lat_path,
        "gt_voxel_path": gt_vox_path,
        "gt_phi_path": gt_phi_path,
    }

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
        out.update(voxel_metrics(pred_bin_al, gt_bin_al))
        # boundary diagnostics: check whether early-z slices are systematically biased
        head = min(16, pred_bin_al.shape[0])
        tail = min(16, pred_bin_al.shape[0])
        out["z_head_porosity_pred"] = float(pred_bin_al[:head].mean())
        out["z_head_porosity_gt"] = float(gt_bin_al[:head].mean())
        out["z_head_porosity_gap"] = float(out["z_head_porosity_pred"] - out["z_head_porosity_gt"])
        out["z_tail_porosity_pred"] = float(pred_bin_al[-tail:].mean())
        out["z_tail_porosity_gt"] = float(gt_bin_al[-tail:].mean())
        out["z_tail_porosity_gap"] = float(out["z_tail_porosity_pred"] - out["z_tail_porosity_gt"])

    patch_voxel = int(CONFIG.get("patch_size", 8)) * int(CONFIG.get("downsample_factor", 8))
    pred_phi_bin = compute_phi_map(pred_bin.astype(np.float32), patch_voxel)
    pred_phi_prob = compute_phi_map(pred_prob.astype(np.float32), patch_voxel)

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

    if args.save_each:
        sample_dir = os.path.join(args.out_dir, "samples", os.path.splitext(basename)[0])
        os.makedirs(sample_dir, exist_ok=True)
        np.save(os.path.join(sample_dir, "pred_latent.npy"), pred_latent.astype(np.float32))
        np.save(os.path.join(sample_dir, "pred_voxel_prob.npy"), pred_prob.astype(np.float32))
        np.save(os.path.join(sample_dir, "pred_voxel_bin.npy"), pred_bin.astype(np.uint8))
        np.save(os.path.join(sample_dir, "pred_phi_from_bin.npy"), pred_phi_bin.astype(np.float32))
        np.save(os.path.join(sample_dir, "pred_phi_from_prob.npy"), pred_phi_prob.astype(np.float32))
        with open(os.path.join(sample_dir, "metrics.json"), "w", encoding="utf-8") as f:
            json.dump({k: _safe_float(v) for k, v in out.items()}, f, indent=2, ensure_ascii=False)

    return out


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
    parser.add_argument("--threshold", type=float, default=0.5)
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
        help="Override fixed traversal direction for (i,j,k), e.g. +++, --+, +-+.",
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
    if args.infer_direction is not None:
        CONFIG["infer_direction"] = str(args.infer_direction)
    if bool(args.infer_random_order):
        CONFIG["infer_random_order"] = True
    if bool(args.infer_random_direction):
        CONFIG["infer_random_direction"] = True

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
    pbar = tqdm(range(len(phi_files)), desc="BatchEval")
    for i in pbar:
        phi_path = phi_files[i]
        row = _evaluate_one(
            phi_path=phi_path,
            sample_idx=i + int(args.offset),
            model=model,
            diffusion=diffusion,
            vae=vae,
            args=args,
        )
        rows.append(row)
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

    summ = _summary(rows)
    summ["ckpt"] = args.ckpt
    summ["phi_dir"] = args.phi_dir
    summ["num_selected"] = len(phi_files)
    summ["sample_mode"] = args.sample_mode
    summ["sample_seed"] = int(args.sample_seed)
    summ["seed_mode"] = args.seed_mode
    summ["threshold"] = float(args.threshold)
    summ["ddim_steps"] = int(args.ddim_steps)
    summ["infer_use_ema"] = bool(CONFIG.get("infer_use_ema", False))
    summ["infer_order"] = str(CONFIG.get("order", "ijk"))
    summ["infer_direction"] = str(CONFIG.get("infer_direction", "+++"))
    summ["infer_random_order"] = bool(CONFIG.get("infer_random_order", False))
    summ["infer_random_direction"] = bool(CONFIG.get("infer_random_direction", False))
    summ_path = os.path.join(args.out_dir, "summary.json")
    with open(summ_path, "w", encoding="utf-8") as f:
        json.dump(summ, f, indent=2, ensure_ascii=False)

    print("[done] batch evaluation finished")
    print(f"  per-sample csv:   {out_csv}")
    print(f"  per-sample jsonl: {out_jsonl}")
    print(f"  summary:          {summ_path}")
    for key in [
        "voxel_dice_mean",
        "voxel_iou_mean",
        "porosity_abs_err_mean",
        "bin_phi_corr_mean",
        "bin_phi_mae_mean",
        "latent_mae_mean",
        "time_sec_mean",
    ]:
        if key in summ:
            print(f"  {key}: {summ[key]:.6f}")


if __name__ == "__main__":
    main()

import os
import glob
import json
import re
from typing import Dict, Tuple

import numpy as np
import torch
import yaml
from tqdm import tqdm

from src.config import CONFIG
from src.models.vae import KLVAE3D
from src.utils_path import get_root


def build_porosity_map_from_csv(csv_path: str) -> Dict[str, float]:
    por_map: Dict[str, float] = {}
    if not csv_path or not os.path.exists(csv_path):
        return por_map
    # CSV header: file,rel_path,porosity,scale_factor,orig_peak,clip_ratio,status
    # We only use file -> porosity
    with open(csv_path, "r", encoding="utf-8") as f:
        # skip header
        header = f.readline()
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 3:
                continue
            fname = parts[0].strip()
            por = parts[2].strip()
            try:
                por_map[fname] = float(por)
            except ValueError:
                continue
    return por_map


def load_vae(cfg_path: str, ckpt_path: str, device: torch.device) -> Tuple[KLVAE3D, dict]:
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"VAE config not found: {cfg_path}")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"VAE checkpoint not found: {ckpt_path}")

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    vae = KLVAE3D(cfg).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)

    if isinstance(ckpt, dict) and "vae_state_dict" in ckpt:
        state = ckpt["vae_state_dict"]
    elif isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
    else:
        state = ckpt

    # strip _orig_mod if compiled
    new_state = {}
    for k, v in state.items():
        new_state[k.replace("_orig_mod.", "")] = v

    vae.load_state_dict(new_state)
    vae.eval()
    return vae, cfg


def make_mask_pixel(shape, axis: str, ratio: float, jitter_ratio: float) -> Tuple[np.ndarray, int]:
    # shape: (D, H, W)
    d, h, w = shape
    if axis.upper() == "D":
        size = d
    elif axis.upper() == "H":
        size = h
    else:
        size = w

    base_cut = int(size * ratio)
    if jitter_ratio > 0:
        jitter = int(size * jitter_ratio)
        base_cut = base_cut + np.random.randint(-jitter, jitter + 1)
    cut = max(1, min(base_cut, size - 1))

    mask = np.zeros((1, d, h, w), dtype=np.float32)
    if axis.upper() == "D":
        mask[:, :cut, :, :] = 1.0
    elif axis.upper() == "H":
        mask[:, :, :cut, :] = 1.0
    else:
        mask[:, :, :, :cut] = 1.0
    return mask, cut


def downsample_factor_from_cfg(cfg: dict) -> int:
    # Encoder downsamples for each stage except the last one
    # (see KLVAE3D Encoder: downsample when i != len(ch_mult)-1)
    ch_mult = cfg["model"]["ch_mult"]
    return 2 ** max(0, (len(ch_mult) - 1))


def main():
    device = torch.device(CONFIG["device"])

    raw_dir = CONFIG["raw_data_dir"]
    latent_dir = CONFIG["latent_dir"]
    porosity_csv = CONFIG.get("porosity_csv", "")
    out_dir = CONFIG["paired_data_dir"]
    os.makedirs(out_dir, exist_ok=True)

    por_map = build_porosity_map_from_csv(porosity_csv)
    if len(por_map) == 0:
        print("⚠️ porosity map is empty. Will skip samples without porosity.")

    vae, vae_cfg = load_vae(CONFIG["vae_config_path"], CONFIG["vae_ckpt_path"], device)
    downsample = downsample_factor_from_cfg(vae_cfg)

    files = sorted(glob.glob(os.path.join(raw_dir, "*.npy")))
    if len(files) == 0:
        raise ValueError(f"No .npy files found in {raw_dir}")

    # stats
    sum_v = 0.0
    sum_sq = 0.0
    count = 0
    min_v = float("inf")
    max_v = float("-inf")

    axis = CONFIG["axis"]
    ratio = float(CONFIG["ratio"])
    jitter = float(CONFIG["jitter_ratio"])

    print(f"Found {len(files)} raw files. Start preprocessing...")

    for fp in tqdm(files, desc="Preprocess"):
        base = os.path.basename(fp)
        por = por_map.get(base, None)
        if por is None:
            # skip if no porosity
            continue

        data = np.load(fp, mmap_mode="r").astype(np.float32)
        # normalize to [-1, 1]
        data = (data / 65535.0) * 2.0 - 1.0

        mask_pixel, cut_pixel = make_mask_pixel(data.shape, axis, ratio, jitter)
        cond_pixel = data[None, ...] * mask_pixel  # [1, D, H, W]

        # to torch
        x_full = torch.from_numpy(data).unsqueeze(0).unsqueeze(0).to(device)  # [1,1,D,H,W]
        x_cond = torch.from_numpy(cond_pixel).unsqueeze(0).to(device)        # [1,1,D,H,W]

        with torch.no_grad():
            if device.type == "cuda":
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    z_full = vae.encode(x_full).mean
                    z_cond = vae.encode(x_cond).mean
            else:
                z_full = vae.encode(x_full).mean
                z_cond = vae.encode(x_cond).mean

        z_full_np = z_full.detach().cpu().float().numpy()[0]
        z_cond_np = z_cond.detach().cpu().float().numpy()[0]

        # latent mask (downsampled cut)
        zc, zd, zh, zw = z_full_np.shape
        # safer: compute downsample from actual shapes (robust to config mismatch)
        inferred_down = int(round(data.shape[0] / float(zd)))
        if inferred_down != downsample:
            downsample = inferred_down  # override with actual ratio
        cut_latent = max(1, min(int(round(cut_pixel / downsample)), zd - 1))
        mask_latent = np.zeros((1, zd, zh, zw), dtype=np.float32)
        if axis.upper() == "D":
            mask_latent[:, :cut_latent, :, :] = 1.0
        elif axis.upper() == "H":
            mask_latent[:, :, :cut_latent, :] = 1.0
        else:
            mask_latent[:, :, :, :cut_latent] = 1.0

        out_name = os.path.splitext(base)[0] + ".npz"
        out_path = os.path.join(out_dir, out_name)
        np.savez_compressed(
            out_path,
            z_full=z_full_np,
            z_cond=z_cond_np,
            mask=mask_latent,
            porosity=np.array([por], dtype=np.float32),
        )

        # update stats
        flat = z_full_np.reshape(-1)
        sum_v += float(flat.sum())
        sum_sq += float((flat * flat).sum())
        count += int(flat.size)
        min_v = min(min_v, float(flat.min()))
        max_v = max(max_v, float(flat.max()))

    if count == 0:
        raise RuntimeError("No samples processed. Check porosity mapping and file names.")

    mean = sum_v / count
    var = max(sum_sq / count - mean * mean, 1e-12)
    std = float(np.sqrt(var))
    scale = float(1.0 / std)

    stats = {
        "count": count,
        "mean": mean,
        "std": std,
        "min": min_v,
        "max": max_v,
        "suggested_scale_factor": scale,
        "downsample_factor": downsample,
    }

    stats_path = os.path.join(out_dir, "stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(f"✅ Done. Saved paired data to: {out_dir}")
    print(f"✅ Stats saved to: {stats_path}")
    print(f"Suggested scale_factor: {scale:.6f}")


if __name__ == "__main__":
    main()

import os
import glob
import numpy as np
from tqdm import tqdm

from src.config import CONFIG


def otsu_threshold(vol: np.ndarray, nbins: int = 256) -> float:
    # vol in [0,1]
    hist, bin_edges = np.histogram(vol, bins=nbins, range=(0.0, 1.0))
    hist = hist.astype(np.float64)
    total = hist.sum()
    if total <= 0:
        return 0.5
    prob = hist / total
    omega = np.cumsum(prob)
    mu = np.cumsum(prob * np.arange(nbins))
    mu_t = mu[-1]
    sigma_b = (mu_t * omega - mu) ** 2 / (omega * (1.0 - omega) + 1e-8)
    idx = np.argmax(sigma_b)
    # map bin index to threshold in [0,1]
    thr = (bin_edges[idx] + bin_edges[idx + 1]) * 0.5
    return float(thr)


def compute_phi_map(vol: np.ndarray, patch_voxel: int) -> np.ndarray:
    D, H, W = vol.shape
    gD, gH, gW = D // patch_voxel, H // patch_voxel, W // patch_voxel
    vol = vol[: gD * patch_voxel, : gH * patch_voxel, : gW * patch_voxel]
    # reshape to blocks
    vol = vol.reshape(gD, patch_voxel, gH, patch_voxel, gW, patch_voxel)
    phi = vol.mean(axis=(1, 3, 5))
    return phi.astype(np.float32)


def normalize_to_unit(vol: np.ndarray) -> np.ndarray:
    vol = vol.astype(np.float32)
    mn, mx = float(vol.min()), float(vol.max())

    # already [0,1]
    if mn >= 0.0 and mx <= 1.0 + 1e-6:
        return np.clip(vol, 0.0, 1.0)

    # already [-1,1]
    if mn >= -1.01 and mx <= 1.01:
        return np.clip((vol + 1.0) * 0.5, 0.0, 1.0)

    # common integer ranges
    if mx <= 255.5:
        return np.clip(vol / 255.0, 0.0, 1.0)
    return np.clip(vol / 65535.0, 0.0, 1.0)


def center_crop_cube(vol: np.ndarray, target_size: int) -> np.ndarray:
    if target_size is None or int(target_size) <= 0:
        return vol
    if vol.ndim != 3:
        raise ValueError(f"Expected 3D volume, got shape={vol.shape}.")
    target_size = int(target_size)
    D, H, W = vol.shape
    if target_size > D or target_size > H or target_size > W:
        raise ValueError(f"target_size={target_size} is larger than input shape {vol.shape}.")
    if D == target_size and H == target_size and W == target_size:
        return vol
    d0 = (D - target_size) // 2
    h0 = (H - target_size) // 2
    w0 = (W - target_size) // 2
    return vol[d0:d0 + target_size, h0:h0 + target_size, w0:w0 + target_size]


def main():
    raw_dir = CONFIG["raw_data_dir"]
    out_dir = CONFIG["phi_map_dir"]
    os.makedirs(out_dir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(raw_dir, "*.npy")))
    if len(files) == 0:
        raise ValueError(f"No .npy files found in {raw_dir}")

    patch_size_latent = int(CONFIG["patch_size"])
    downsample = int(CONFIG.get("downsample_factor", 8))
    patch_voxel = patch_size_latent * downsample

    mode = str(CONFIG.get("binarize_mode", "fixed")).lower()
    fixed_thr = float(CONFIG.get("binarize_threshold", 0.5))
    phi_target = int(CONFIG.get("phi_input_target_size", 0))

    for fp in tqdm(files, desc="PhiMap"):
        base = os.path.basename(fp)
        vol = np.load(fp).astype(np.float32)
        vol = center_crop_cube(vol, phi_target)
        vol = normalize_to_unit(vol)

        if mode == "otsu":
            thr = otsu_threshold(vol)
            bin_vol = (vol >= thr).astype(np.float32)
        elif mode == "none":
            # use grayscale mean as soft-porosity
            bin_vol = vol
        else:
            bin_vol = (vol >= fixed_thr).astype(np.float32)

        phi = compute_phi_map(bin_vol, patch_voxel)
        out_path = os.path.join(out_dir, os.path.splitext(base)[0] + ".npy")
        np.save(out_path, phi)

    print(f"Saved phi maps to {out_dir}")


if __name__ == "__main__":
    main()

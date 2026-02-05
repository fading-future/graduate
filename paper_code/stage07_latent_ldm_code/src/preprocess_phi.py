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

    for fp in tqdm(files, desc="PhiMap"):
        base = os.path.basename(fp)
        vol = np.load(fp).astype(np.float32)

        # normalize to [0,1] if 16-bit
        if vol.max() > 1.0:
            vol = np.clip(vol / 65535.0, 0.0, 1.0)

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

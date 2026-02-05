import os
import glob
import re
import random
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from src.config import CONFIG


def _strip_porosity_prefix(name: str) -> str:
    # support: porosity_0.123456_xxx.npy -> xxx.npy
    return re.sub(r"^porosity_[0-9]*\\.?[0-9]+_", "", name)


def _load_npy(path: str) -> np.ndarray:
    return np.load(path).astype(np.float32)


def _repeat_phi(phi_patch: np.ndarray, patch_size: int) -> np.ndarray:
    # phi_patch: (w, w, w) -> (w*ps, w*ps, w*ps)
    out = np.repeat(phi_patch, patch_size, axis=0)
    out = np.repeat(out, patch_size, axis=1)
    out = np.repeat(out, patch_size, axis=2)
    return out


def _is_prev(a: Tuple[int, int, int], b: Tuple[int, int, int], order: str) -> bool:
    ai, aj, ak = a
    bi, bj, bk = b
    if order.lower() == "ijk":
        if ai < bi:
            return True
        if ai == bi and aj < bj:
            return True
        if ai == bi and aj == bj and ak < bk:
            return True
        return False
    # fallback: treat as full context
    return False


def _load_porosity_map(csv_path: str) -> dict:
    por_map = {}
    if not csv_path or not os.path.exists(csv_path):
        return por_map
    with open(csv_path, "r", encoding="utf-8") as f:
        _ = f.readline()  # header
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


class PatchLatentDataset(Dataset):
    def __init__(self, latent_dir: str, phi_map_dir: str, augment: bool = True):
        self.latent_dir = latent_dir
        self.phi_map_dir = phi_map_dir
        self.augment = augment

        files = sorted(glob.glob(os.path.join(latent_dir, "*.npy")))
        if len(files) == 0:
            raise ValueError(f"No .npy files found in {latent_dir}")

        pairs: List[Tuple[str, str, str]] = []
        for fp in files:
            base = os.path.basename(fp)
            base2 = _strip_porosity_prefix(base)
            phi_path = os.path.join(phi_map_dir, base2)
            if not os.path.exists(phi_path):
                # try without extension change
                phi_path = os.path.join(phi_map_dir, os.path.splitext(base2)[0] + ".npy")
            if not os.path.exists(phi_path):
                continue
            pairs.append((fp, phi_path, base2))

        if len(pairs) == 0:
            raise ValueError("No (latent, phi_map) pairs found. Check phi_map_dir.")

        self.pairs = pairs
        self.scale_factor = float(CONFIG.get("scale_factor", 1.0))
        self.safe_threshold = float(CONFIG.get("safe_threshold", 8.0))
        self.patch_size = int(CONFIG["patch_size"])
        self.window_size = int(CONFIG["window_size"])
        self.context_mode = str(CONFIG.get("context_mode", "causal")).lower()
        self.order = str(CONFIG.get("order", "ijk")).lower()
        self.porosity_mode = str(CONFIG.get("porosity_mode", "local")).lower()
        self.porosity_map = _load_porosity_map(CONFIG.get("porosity_csv", ""))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        latent_path, phi_path, base_name = self.pairs[idx]

        z_full = _load_npy(latent_path)  # (C, D, H, W)
        phi_map = _load_npy(phi_path)    # (gD, gH, gW)

        if z_full.ndim == 5:
            z_full = z_full.squeeze(0)

        # scale + clamp
        z_full = z_full * self.scale_factor
        z_full = np.clip(z_full, -self.safe_threshold, self.safe_threshold)

        C, D, H, W = z_full.shape
        gD, gH, gW = phi_map.shape

        # sanity: derive grid from latent shape
        p = self.patch_size
        exp_gD, exp_gH, exp_gW = D // p, H // p, W // p
        if (gD, gH, gW) != (exp_gD, exp_gH, exp_gW):
            raise ValueError(f"phi_map shape {phi_map.shape} != latent grid {(exp_gD, exp_gH, exp_gW)}")

        # sample target patch index
        ti = random.randint(0, gD - 1)
        tj = random.randint(0, gH - 1)
        tk = random.randint(0, gW - 1)

        w = self.window_size
        r = w // 2

        # pad latent and phi to allow window at edges
        pad_p = r * p
        z_pad = np.pad(z_full, ((0, 0), (pad_p, pad_p), (pad_p, pad_p), (pad_p, pad_p)), mode="constant")
        phi_pad = np.pad(phi_map, ((r, r), (r, r), (r, r)), mode="constant")

        ci, cj, ck = ti + r, tj + r, tk + r

        # window slices (patch units)
        wi0, wi1 = ci - r, ci + r + 1
        wj0, wj1 = cj - r, cj + r + 1
        wk0, wk1 = ck - r, ck + r + 1

        phi_win = phi_pad[wi0:wi1, wj0:wj1, wk0:wk1]  # (w,w,w)

        # latent window (voxel units)
        zi0, zi1 = wi0 * p, wi1 * p
        zj0, zj1 = wj0 * p, wj1 * p
        zk0, zk1 = wk0 * p, wk1 * p
        z_win = z_pad[:, zi0:zi1, zj0:zj1, zk0:zk1]  # (C, w*p, w*p, w*p)

        # build patch-level known mask
        mask_patch = np.zeros((w, w, w), dtype=np.float32)
        for di in range(w):
            for dj in range(w):
                for dk in range(w):
                    gi = ti - r + di
                    gj = tj - r + dj
                    gk = tk - r + dk
                    if gi < 0 or gj < 0 or gk < 0 or gi >= gD or gj >= gH or gk >= gW:
                        continue
                    if self.context_mode == "full":
                        known = not (gi == ti and gj == tj and gk == tk)
                    else:
                        known = _is_prev((gi, gj, gk), (ti, tj, tk), self.order)
                    if known:
                        mask_patch[di, dj, dk] = 1.0

        # voxel mask & target mask
        mask = _repeat_phi(mask_patch, p)[None, ...]  # (1, w*p, w*p, w*p)
        target_patch = np.zeros_like(mask_patch)
        target_patch[r, r, r] = 1.0
        target_mask = _repeat_phi(target_patch, p)[None, ...]

        cond = z_win * mask
        phi_vol = _repeat_phi(phi_win, p)[None, ...]
        if self.porosity_mode == "global":
            por = self.porosity_map.get(base_name, None)
            if por is None:
                por = float(phi_map.mean())
            porosity = np.array([por], dtype=np.float32)
        else:
            porosity = np.array([phi_map[ti, tj, tk]], dtype=np.float32)

        # augment (flip)
        if self.augment:
            if random.random() > 0.5:
                z_win = np.flip(z_win, axis=1)
                cond = np.flip(cond, axis=1)
                mask = np.flip(mask, axis=1)
                target_mask = np.flip(target_mask, axis=1)
                phi_vol = np.flip(phi_vol, axis=1)
            if random.random() > 0.5:
                z_win = np.flip(z_win, axis=2)
                cond = np.flip(cond, axis=2)
                mask = np.flip(mask, axis=2)
                target_mask = np.flip(target_mask, axis=2)
                phi_vol = np.flip(phi_vol, axis=2)
            if random.random() > 0.5:
                z_win = np.flip(z_win, axis=3)
                cond = np.flip(cond, axis=3)
                mask = np.flip(mask, axis=3)
                target_mask = np.flip(target_mask, axis=3)
                phi_vol = np.flip(phi_vol, axis=3)

        # contiguous
        z_win = np.ascontiguousarray(z_win)
        cond = np.ascontiguousarray(cond)
        mask = np.ascontiguousarray(mask)
        target_mask = np.ascontiguousarray(target_mask)
        phi_vol = np.ascontiguousarray(phi_vol)

        return {
            "GT": torch.from_numpy(z_win).float(),
            "Condition": torch.from_numpy(cond).float(),
            "Mask": torch.from_numpy(mask).float(),
            "TargetMask": torch.from_numpy(target_mask).float(),
            "Phi": torch.from_numpy(phi_vol).float(),
            "Porosity": torch.from_numpy(porosity).float(),
        }

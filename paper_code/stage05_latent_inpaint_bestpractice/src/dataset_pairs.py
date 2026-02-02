import os
import glob
import json
import random
import numpy as np
import torch
from torch.utils.data import Dataset

from src.config import CONFIG


def _load_scale_factor(paired_dir: str):
    if CONFIG.get("scale_factor") is not None:
        return float(CONFIG["scale_factor"])
    stats_path = os.path.join(paired_dir, "stats.json")
    if os.path.exists(stats_path):
        with open(stats_path, "r", encoding="utf-8") as f:
            stats = json.load(f)
        return float(stats.get("suggested_scale_factor", 1.0))
    return 1.0


class PairedLatentDataset(Dataset):
    def __init__(self, paired_dir: str, augment: bool = True):
        self.paired_dir = paired_dir
        self.files = sorted(glob.glob(os.path.join(paired_dir, "*.npz")))
        if len(self.files) == 0:
            raise ValueError(f"No .npz files found in {paired_dir}")
        self.augment = augment
        self.scale_factor = _load_scale_factor(paired_dir)
        self.safe_threshold = float(CONFIG.get("safe_threshold", 8.0))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fp = self.files[idx]
        data = np.load(fp)
        z_full = data["z_full"].astype(np.float32)
        z_cond = data["z_cond"].astype(np.float32)
        mask = data["mask"].astype(np.float32)  # (1, D, H, W)
        porosity = data["porosity"].astype(np.float32)

        # scale + clamp
        z_full = z_full * self.scale_factor
        z_cond = z_cond * self.scale_factor
        z_full = np.clip(z_full, -self.safe_threshold, self.safe_threshold)
        z_cond = np.clip(z_cond, -self.safe_threshold, self.safe_threshold)

        # augment: only flip H/W to keep cut direction stable
        if self.augment:
            if random.random() > 0.5:
                z_full = np.flip(z_full, axis=2)  # H
                z_cond = np.flip(z_cond, axis=2)
                mask = np.flip(mask, axis=2)
            if random.random() > 0.5:
                z_full = np.flip(z_full, axis=3)  # W
                z_cond = np.flip(z_cond, axis=3)
                mask = np.flip(mask, axis=3)

        z_full_t = torch.from_numpy(z_full).float()
        z_cond_t = torch.from_numpy(z_cond).float()
        mask_t = torch.from_numpy(mask).float()
        porosity_t = torch.from_numpy(porosity).float()

        return {
            "GT": z_full_t,
            "Condition": z_cond_t,
            "Mask": mask_t,
            "Porosity": porosity_t,
        }

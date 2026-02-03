import os
import glob
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from src.config import CONFIG
from src.utils import build_porosity_map


def normalize_volume(vol: np.ndarray) -> np.ndarray:
    vol = vol.astype(np.float32)
    vol = vol / 65535.0
    vol = (vol * 2.0) - 1.0
    return vol


def make_mask(shape, axis: str, ratio: float, erosion: int):
    D, H, W = shape
    axis = axis.upper()
    if axis == "D":
        size = D
    elif axis == "H":
        size = H
    else:
        size = W
    cut = int(size * ratio)
    cut = max(1, min(cut, size - 1))
    cut = max(0, cut - erosion)

    mask = np.zeros((1, D, H, W), dtype=np.float32)
    if axis == "D":
        mask[:, :cut, :, :] = 1.0
    elif axis == "H":
        mask[:, :, :cut, :] = 1.0
    else:
        mask[:, :, :, :cut] = 1.0
    return mask, cut


class RefineDataset(Dataset):
    def __init__(self, data_dir: str, patch_size: int, coarse_cache_dir: str = "", augment: bool = True):
        self.files = sorted(glob.glob(os.path.join(data_dir, "*.npy")))
        if len(self.files) == 0:
            raise ValueError(f"No files found in {data_dir}")
        self.patch = patch_size
        self.augment = augment
        self.por_map = build_porosity_map(CONFIG["PATHS"]["porosity_csv"])
        self.coarse_cache = coarse_cache_dir

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fp = self.files[idx]
        base = os.path.basename(fp)
        por = self.por_map.get(base, 0.15)

        raw = np.load(fp, mmap_mode="r")
        vol = normalize_volume(raw)  # (256,256,256)

        # build full mask and condition
        mask_full, cut = make_mask(vol.shape, CONFIG["TASK"]["axis"], CONFIG["TASK"]["ratio"], CONFIG["TASK"]["erosion_px"])
        cond_full = vol[None, ...] * mask_full

        # load or compute coarse guidance (full resolution)
        if self.coarse_cache and os.path.exists(self.coarse_cache):
            coarse_path = os.path.join(self.coarse_cache, base.replace('.npy', '_coarse.npy'))
            if os.path.exists(coarse_path):
                coarse_full = np.load(coarse_path).astype(np.float32)
            else:
                coarse_full = None
        else:
            coarse_full = None

        if coarse_full is None:
            # fallback: use downsample+upsample of GT as guidance
            gt_t = torch.from_numpy(vol).unsqueeze(0).unsqueeze(0).float()
            coarse_size = CONFIG["COARSE"]["coarse_size"]
            gt_c = F.interpolate(gt_t, size=(coarse_size,)*3, mode="trilinear", align_corners=False)
            coarse_full = F.interpolate(gt_c, size=vol.shape, mode="trilinear", align_corners=False)[0,0].numpy()

        # random crop
        D, H, W = vol.shape
        p = self.patch
        z0 = random.randint(0, D - p)
        y0 = random.randint(0, H - p)
        x0 = random.randint(0, W - p)

        gt_patch = vol[z0:z0+p, y0:y0+p, x0:x0+p]
        cond_patch = cond_full[0, z0:z0+p, y0:y0+p, x0:x0+p]
        mask_patch = mask_full[0, z0:z0+p, y0:y0+p, x0:x0+p]
        coarse_patch = coarse_full[z0:z0+p, y0:y0+p, x0:x0+p]

        # augment flips
        if self.augment:
            if random.random() > 0.5:
                gt_patch = np.flip(gt_patch, axis=0)
                cond_patch = np.flip(cond_patch, axis=0)
                mask_patch = np.flip(mask_patch, axis=0)
                coarse_patch = np.flip(coarse_patch, axis=0)
            if random.random() > 0.5:
                gt_patch = np.flip(gt_patch, axis=1)
                cond_patch = np.flip(cond_patch, axis=1)
                mask_patch = np.flip(mask_patch, axis=1)
                coarse_patch = np.flip(coarse_patch, axis=1)
            if random.random() > 0.5:
                gt_patch = np.flip(gt_patch, axis=2)
                cond_patch = np.flip(cond_patch, axis=2)
                mask_patch = np.flip(mask_patch, axis=2)
                coarse_patch = np.flip(coarse_patch, axis=2)

        # to tensor
        gt_t = torch.from_numpy(np.ascontiguousarray(gt_patch)).unsqueeze(0).float()
        cond_t = torch.from_numpy(np.ascontiguousarray(cond_patch)).unsqueeze(0).float()
        mask_t = torch.from_numpy(np.ascontiguousarray(mask_patch)).unsqueeze(0).float()
        coarse_t = torch.from_numpy(np.ascontiguousarray(coarse_patch)).unsqueeze(0).float()

        return {
            "GT": gt_t,
            "Condition": cond_t,
            "Mask": mask_t,
            "Coarse": coarse_t,
            "Porosity": torch.tensor([por], dtype=torch.float32),
        }

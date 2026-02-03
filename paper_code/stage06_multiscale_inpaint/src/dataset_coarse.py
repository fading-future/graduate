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
    return mask


class CoarseDataset(Dataset):
    def __init__(self, data_dir: str, coarse_size: int, augment: bool = True):
        self.files = sorted(glob.glob(os.path.join(data_dir, "*.npy")))[:1600]
        if len(self.files) == 0:
            raise ValueError(f"No files found in {data_dir}")
        self.coarse_size = coarse_size
        self.augment = augment
        self.por_map = build_porosity_map(CONFIG["PATHS"]["porosity_csv"])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fp = self.files[idx]
        base = os.path.basename(fp)
        por = self.por_map.get(base, 0.15)

        raw = np.load(fp, mmap_mode="r")
        vol = normalize_volume(raw)

        # mask at full resolution
        mask_full = make_mask(vol.shape, CONFIG["TASK"]["axis"], CONFIG["TASK"]["ratio"], CONFIG["TASK"]["erosion_px"])
        cond_full = vol[None, ...] * mask_full

        # to torch
        gt = torch.from_numpy(vol).unsqueeze(0).unsqueeze(0).float()        # [1,1,256,256,256]
        cond = torch.from_numpy(cond_full).unsqueeze(0).float()            # [1,1,256,256,256]
        mask = torch.from_numpy(mask_full).unsqueeze(0).float()            # [1,1,256,256,256]

        # downsample to coarse size
        gt_c = F.interpolate(gt, size=(self.coarse_size,)*3, mode="trilinear", align_corners=False)
        cond_c = F.interpolate(cond, size=(self.coarse_size,)*3, mode="trilinear", align_corners=False)
        mask_c = F.interpolate(mask, size=(self.coarse_size,)*3, mode="nearest")

        if self.augment:
            # random flips on coarse
            if random.random() > 0.5:
                gt_c = torch.flip(gt_c, dims=[2])
                cond_c = torch.flip(cond_c, dims=[2])
                mask_c = torch.flip(mask_c, dims=[2])
            if random.random() > 0.5:
                gt_c = torch.flip(gt_c, dims=[3])
                cond_c = torch.flip(cond_c, dims=[3])
                mask_c = torch.flip(mask_c, dims=[3])
            if random.random() > 0.5:
                gt_c = torch.flip(gt_c, dims=[4])
                cond_c = torch.flip(cond_c, dims=[4])
                mask_c = torch.flip(mask_c, dims=[4])

        return {
            "GT": gt_c[0],
            "Condition": cond_c[0],
            "Mask": mask_c[0],
            "Porosity": torch.tensor([por], dtype=torch.float32),
        }

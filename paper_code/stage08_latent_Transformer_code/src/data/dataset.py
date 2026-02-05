from __future__ import annotations

import os
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .patching import generate_patch_indices, extract_patch, split_into_2x2x2
from .porosity import compute_porosity, load_porosity_csv, PorosityLookup


class VQVaePatchDataset(Dataset):
    def __init__(
        self,
        files: List[str],
        patch_size: int = 64,
        stride: int = 32,
        pore_value: int = 1,
        max_samples: int | None = None,
    ):
        self.files = files
        self.patch_size = patch_size
        self.stride = stride
        self.pore_value = pore_value
        self.index = []  # list of (file, z, y, x)
        self._build_index(max_samples)

    def _build_index(self, max_samples: int | None):
        for fp in self.files:
            vol = np.load(fp, mmap_mode="r")
            if vol.ndim != 3:
                raise ValueError(f"Expected 3D volume, got {vol.shape} for {fp}")
            starts = generate_patch_indices(vol.shape, self.patch_size, self.stride)
            for s in starts:
                self.index.append((fp, s[0], s[1], s[2]))
                if max_samples is not None and len(self.index) >= max_samples:
                    return

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        fp, z, y, x = self.index[idx]
        vol = np.load(fp, mmap_mode="r")
        patch = extract_patch(vol, (z, y, x), self.patch_size)
        patch = patch.astype(np.float32)
        # normalize to 0/1 float
        patch = (patch == self.pore_value).astype(np.float32)
        patch = torch.from_numpy(patch)[None, ...]  # add channel dim
        return patch


class TransformerPatchDataset(Dataset):
    def __init__(
        self,
        files: List[str],
        patch_size: int = 128,
        stride: int = 64,
        pore_value: int = 1,
        porosity_source: str = "compute",
        porosity_csv: str | None = None,
        max_samples: int | None = None,
    ):
        self.files = files
        self.patch_size = patch_size
        self.stride = stride
        self.pore_value = pore_value
        self.porosity_source = porosity_source
        self.lookup: PorosityLookup | None = None
        if porosity_source == "csv":
            if not porosity_csv:
                raise ValueError("porosity_csv is required when porosity_source='csv'")
            self.lookup = load_porosity_csv(porosity_csv)
        self.index = []
        self._build_index(max_samples)

    def _build_index(self, max_samples: int | None):
        for fp in self.files:
            vol = np.load(fp, mmap_mode="r")
            if vol.ndim != 3:
                raise ValueError(f"Expected 3D volume, got {vol.shape} for {fp}")
            starts = generate_patch_indices(vol.shape, self.patch_size, self.stride)
            for s in starts:
                self.index.append((fp, s[0], s[1], s[2]))
                if max_samples is not None and len(self.index) >= max_samples:
                    return

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        fp, z, y, x = self.index[idx]
        vol = np.load(fp, mmap_mode="r")
        patch = extract_patch(vol, (z, y, x), self.patch_size)
        patch = (patch == self.pore_value).astype(np.float32)
        # split into 8 subvolumes of 64^3
        subvols = split_into_2x2x2(patch)
        if self.porosity_source == "csv" and self.lookup is not None:
            if self.lookup.index_cols:
                # CSV provides per-patch porosity with indices
                cond = []
                for i, (dz, dy, dx) in enumerate(
                    [(0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1), (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1)]
                ):
                    # convert to absolute patch indices
                    pz = z + dz * 64
                    py = y + dy * 64
                    px = x + dx * 64
                    val = self.lookup.get(fp, (pz, py, px))
                    if val is None:
                        val = compute_porosity(subvols[i], self.pore_value)
                    cond.append(val)
            else:
                # CSV provides per-file porosity; broadcast to all 8 subvolumes
                file_phi = self.lookup.get(fp)
                if file_phi is None:
                    cond = [compute_porosity(sv, self.pore_value) for sv in subvols]
                else:
                    cond = [float(file_phi)] * len(subvols)
        else:
            cond = [compute_porosity(sv, self.pore_value) for sv in subvols]

        patch = torch.from_numpy(patch)[None, ...]
        cond = torch.tensor(cond, dtype=torch.float32)
        return patch, cond

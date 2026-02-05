from __future__ import annotations

from typing import Iterable, List, Tuple


def generate_patch_indices(shape: Tuple[int, int, int], patch_size: int, stride: int) -> List[Tuple[int, int, int]]:
    d, h, w = shape
    indices = []
    for z in range(0, d - patch_size + 1, stride):
        for y in range(0, h - patch_size + 1, stride):
            for x in range(0, w - patch_size + 1, stride):
                indices.append((z, y, x))
    return indices


def extract_patch(volume, start: Tuple[int, int, int], patch_size: int):
    z, y, x = start
    return volume[z:z+patch_size, y:y+patch_size, x:x+patch_size]


def split_into_2x2x2(volume_128):
    """Split 128^3 volume into 8 subvolumes of 64^3 in (z,y,x) nested order."""
    subvols = []
    for z in (0, 64):
        for y in (0, 64):
            for x in (0, 64):
                subvols.append(volume_128[z:z+64, y:y+64, x:x+64])
    return subvols

import os
import glob
import re
import random
from typing import List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from src.config import CONFIG


Direction3D = Tuple[int, int, int]


def _strip_porosity_prefix(name: str) -> str:
    # 从文件名中提取孔隙率信息并返回去除孔隙率前缀后的名字
    # support: porosity_0.123456_xxx.npy -> xxx.npy
    return re.sub(r"^porosity_[0-9]*\\.?[0-9]+_", "", name)


def _load_npy(path: str) -> np.ndarray:
    # 安全加载npy文件，确保输出为float32类型
    return np.load(path).astype(np.float32)


def _repeat_phi(phi_patch: np.ndarray, patch_size: int) -> np.ndarray:
    # phi_patch: (w, w, w) -> (w*ps, w*ps, w*ps)
    out = np.repeat(phi_patch, patch_size, axis=0)
    out = np.repeat(out, patch_size, axis=1)
    out = np.repeat(out, patch_size, axis=2)
    return out


def _pad_with_mode(x: np.ndarray, pad_width, mode: str) -> np.ndarray:
    # numpy reflect mode requires pad < axis length; fallback to edge if invalid
    if mode == "reflect":
        for axis, (pad_before, pad_after) in enumerate(pad_width):
            axis_len = x.shape[axis]
            if axis_len <= 1 and (pad_before > 0 or pad_after > 0):
                return np.pad(x, pad_width, mode="edge")
            if pad_before >= axis_len or pad_after >= axis_len:
                return np.pad(x, pad_width, mode="edge")
    return np.pad(x, pad_width, mode=mode)


def _normalize_order(order: str) -> str:
    # 将顺序字符串标准化为 "ijk" 格式，默认返回 "ijk"
    order = str(order).lower().strip()
    if len(order) != 3 or set(order) != {"i", "j", "k"}:
        return "ijk"
    return order


def _value_to_sign(v) -> int:
    if isinstance(v, (int, np.integer, float, np.floating)):
        return 1 if float(v) >= 0 else -1
    txt = str(v).strip().lower()
    if txt in ("+", "1", "+1", "pos", "forward", "fwd"):
        return 1
    if txt in ("-", "-1", "neg", "reverse", "rev", "backward", "bwd"):
        return -1
    return 1


def _normalize_direction(direction) -> Direction3D:
    # Accept forms like "+++", "+-+", "1,-1,1", [1,-1,1], ("+","-","+")
    if isinstance(direction, str):
        txt = direction.strip().lower().replace(" ", "")
        if len(txt) == 3 and set(txt).issubset({"+", "-"}):
            return tuple(1 if ch == "+" else -1 for ch in txt)  # type: ignore[return-value]
        if "," in txt:
            parts = txt.split(",")
            if len(parts) == 3:
                return tuple(_value_to_sign(p) for p in parts)  # type: ignore[return-value]
        return (1, 1, 1)
    if isinstance(direction, Sequence) and not isinstance(direction, (bytes, bytearray)):
        parts = list(direction)
        if len(parts) == 3:
            return tuple(_value_to_sign(p) for p in parts)  # type: ignore[return-value]
    return (1, 1, 1)


def _normalize_context_mode(mode: str) -> str:
    mode = str(mode).lower().strip()
    if mode not in ("causal", "full", "wavefront"):
        return "causal"
    return mode


def _normalize_pad_mode(mode: str) -> str:
    mode = str(mode).lower().strip()
    if mode not in ("constant", "edge", "reflect"):
        return "edge"
    return mode


def _normalize_anchor_sampling_mode(mode: str) -> str:
    mode = str(mode).lower().strip()
    if mode not in ("uniform", "low_context_boost"):
        return "uniform"
    return mode


def _normalize_sampler_semantic(semantic: str) -> str:
    s = str(semantic).lower().strip()
    if s in ("pore", "porosity", "pore_rate"):
        return "pore"
    if s in ("rock", "rock_rate", "phi"):
        return "rock_rate"
    return "pore"


def _is_prev_lexicographic(
    a: Tuple[int, int, int],
    b: Tuple[int, int, int],
    order: str,
    direction: Direction3D,
) -> bool:
    ai, aj, ak = a
    bi, bj, bk = b
    axis_a = {"i": ai, "j": aj, "k": ak}
    axis_b = {"i": bi, "j": bj, "k": bk}
    axis_s = {"i": direction[0], "j": direction[1], "k": direction[2]}
    for axis in _normalize_order(order):
        va, vb = axis_a[axis] * axis_s[axis], axis_b[axis] * axis_s[axis]
        if va < vb:
            return True
        if va > vb:
            return False
    return False


def _is_prev_wavefront(
    a: Tuple[int, int, int],
    b: Tuple[int, int, int],
    direction: Direction3D,
) -> bool:
    ai, aj, ak = a
    bi, bj, bk = b
    si, sj, sk = direction
    cond_i = ai <= bi if si > 0 else ai >= bi
    cond_j = aj <= bj if sj > 0 else aj >= bj
    cond_k = ak <= bk if sk > 0 else ak >= bk
    return (cond_i and cond_j and cond_k) and (a != b)


def _is_prev_by_mode(
    a: Tuple[int, int, int],
    b: Tuple[int, int, int],
    context_mode: str,
    order: str,
    direction: Direction3D,
) -> bool:
    if context_mode == "wavefront":
        return _is_prev_wavefront(a, b, direction)
    return _is_prev_lexicographic(a, b, order, direction)


def _load_porosity_map(csv_path: str) -> dict:
    # 从CSV文件中加载全局孔隙率映射，返回一个字典 {base_name: porosity}
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
    def __init__(self, 
                 latent_dir: str,       # 被KLVAE 压缩的潜在空间文件夹路径
                 phi_map_dir: str,      # latent对应的phi map文件夹路径
                 augment: bool = True   # 是否进行数据增强（翻转）
                 ):
        self.latent_dir = latent_dir
        self.phi_map_dir = phi_map_dir
        self.augment = augment

        files = sorted(glob.glob(os.path.join(latent_dir, "*.npy")))
        if len(files) == 0:
            raise ValueError(f"No .npy files found in {latent_dir}")

        # pairs 中存储了：每一对数据的潜在空间文件路径、对应的phi map文件路径、以及去除孔隙率前缀后的基本文件名
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
        self.context_mode = _normalize_context_mode(CONFIG.get("context_mode", "causal"))
        self.order = _normalize_order(CONFIG.get("order", "ijk"))
        self.train_random_order = bool(CONFIG.get("train_random_order", False))
        self.order_candidates = ["ijk", "ikj", "jik", "jki", "kij", "kji"]
        self.train_random_direction = bool(CONFIG.get("train_random_direction", False))
        self.train_direction = _normalize_direction(CONFIG.get("train_direction", "+++"))
        self.anchor_sampling_mode = _normalize_anchor_sampling_mode(CONFIG.get("anchor_sampling_mode", "uniform"))
        self.anchor_boost_power = float(CONFIG.get("anchor_boost_power", 1.0))
        self.anchor_boost_min_weight = float(CONFIG.get("anchor_boost_min_weight", 0.05))
        self.pad_mode = _normalize_pad_mode(CONFIG.get("pad_mode", "edge"))
        self.porosity_mode = str(CONFIG.get("porosity_mode", "local")).lower()
        self.porosity_mix_alpha = float(CONFIG.get("porosity_mix_alpha", 0.7))
        self.use_global_phi_channel = bool(CONFIG.get("use_global_phi_channel", False))
        self.porosity_map = _load_porosity_map(CONFIG.get("porosity_csv", ""))
        self.porosity_sampler_semantic = _normalize_sampler_semantic(
            CONFIG.get("porosity_sampler_semantic", "pore")
        )
        self.sample_phi_means = None
        self._last_sampler_stats = {}

    def _get_sample_phi_means(self) -> np.ndarray:
        if self.sample_phi_means is not None:
            return self.sample_phi_means
        vals: List[float] = []
        for _, phi_path, _ in self.pairs:
            phi = _load_npy(phi_path)
            vals.append(float(phi.mean()))
        self.sample_phi_means = np.array(vals, dtype=np.float32)
        return self.sample_phi_means

    def build_porosity_sampling_weights(
        self,
        bin_edges: List[float],
        power: float = 1.0,
        min_weight: float = 0.1,
        max_weight: float = 10.0,
    ) -> np.ndarray:
        vals = self._get_sample_phi_means().astype(np.float64)
        if self.porosity_sampler_semantic == "pore":
            vals = 1.0 - vals
        vals = np.clip(vals, 0.0, 1.0)
        if len(bin_edges) < 2:
            raise ValueError("porosity bin_edges must contain at least 2 values.")
        edges = np.array(bin_edges, dtype=np.float64)
        if not np.all(np.diff(edges) > 0):
            raise ValueError("porosity bin_edges must be strictly increasing.")

        # bin_id in [0, num_bins-1]
        bin_ids = np.digitize(vals, edges[1:-1], right=False)
        num_bins = len(edges) - 1
        counts = np.bincount(bin_ids, minlength=num_bins).astype(np.float64)

        inv = np.zeros_like(counts)
        nz = counts > 0
        inv[nz] = 1.0 / counts[nz]
        inv = np.power(inv, max(power, 0.0))
        sample_w = inv[bin_ids]

        # normalize around 1.0 for stable gradients
        mean_w = float(sample_w.mean()) if sample_w.size > 0 else 1.0
        if mean_w > 0:
            sample_w = sample_w / mean_w
        sample_w = np.clip(sample_w, max(min_weight, 1e-6), max(max_weight, min_weight))

        self._last_sampler_stats = {
            "semantic": self.porosity_sampler_semantic,
            "bin_edges": edges.tolist(),
            "bin_counts": counts.tolist(),
            "value_min": float(vals.min()) if vals.size > 0 else 0.0,
            "value_max": float(vals.max()) if vals.size > 0 else 0.0,
            "value_mean": float(vals.mean()) if vals.size > 0 else 0.0,
            "weight_min": float(sample_w.min()) if sample_w.size > 0 else 0.0,
            "weight_max": float(sample_w.max()) if sample_w.size > 0 else 0.0,
            "weight_mean": float(sample_w.mean()) if sample_w.size > 0 else 0.0,
        }
        return sample_w.astype(np.float64)

    def __len__(self):
        return len(self.pairs)

    def _sample_order_for_item(self) -> str:
        if self.context_mode != "causal":
            return self.order
        if not self.train_random_order:
            return self.order
        return random.choice(self.order_candidates)

    def _sample_direction_for_item(self) -> Direction3D:
        if self.context_mode not in ("causal", "wavefront"):
            return self.train_direction
        if not self.train_random_direction:
            return self.train_direction
        return (
            random.choice((1, -1)),
            random.choice((1, -1)),
            random.choice((1, -1)),
        )

    def _sample_anchor(
        self,
        gD: int,
        gH: int,
        gW: int,
        r: int,
        order_used: str,
        direction_used: Direction3D,
    ) -> Tuple[int, int, int]:
        if self.anchor_sampling_mode == "uniform":
            return (
                random.randint(0, gD - 1),
                random.randint(0, gH - 1),
                random.randint(0, gW - 1),
            )

        coords = [(i, j, k) for i in range(gD) for j in range(gH) for k in range(gW)]
        weights: List[float] = []
        pwr = max(self.anchor_boost_power, 0.0)
        min_w = max(self.anchor_boost_min_weight, 0.0)

        for ti, tj, tk in coords:
            prev_count = 0
            for gi in range(max(0, ti - r), min(gD, ti + r + 1)):
                for gj in range(max(0, tj - r), min(gH, tj + r + 1)):
                    for gk in range(max(0, tk - r), min(gW, tk + r + 1)):
                        if gi == ti and gj == tj and gk == tk:
                            continue
                        if self.context_mode == "full":
                            is_prev = True
                        else:
                            is_prev = _is_prev_by_mode(
                                (gi, gj, gk),
                                (ti, tj, tk),
                                self.context_mode,
                                order_used,
                                direction_used,
                            )
                        if is_prev:
                            prev_count += 1
            w = 1.0 / ((prev_count + 1) ** pwr if pwr > 0 else 1.0)
            weights.append(max(w, min_w))

        pick = random.choices(coords, weights=weights, k=1)[0]
        return int(pick[0]), int(pick[1]), int(pick[2])

    def __getitem__(self, idx):
        # 1. 加载Latent NPY和对应的Phi Map NPY，做数据预处理（scale + clamp）
        latent_path, phi_path, base_name = self.pairs[idx]
        z_full = _load_npy(latent_path)  # (C, D, H, W) -> 比如 (4, 24, 24, 24)
        phi_map = _load_npy(phi_path)    # (gD, gH, gW) -> 比如 (3, 3, 3)

        if z_full.ndim == 5:
            z_full = z_full.squeeze(0)

        # 数据预处理：scale + clamp
        z_full = z_full * self.scale_factor
        z_full = np.clip(z_full, -self.safe_threshold, self.safe_threshold)

        C, D, H, W = z_full.shape
        gD, gH, gW = phi_map.shape

        # 2. 随机采样目标 (The Anchor)
        # sanity: derive grid from latent shape
        p = self.patch_size # latent voxels per patch，比如8，对应的原始体素数为 patch_size * downsample_factor
        exp_gD, exp_gH, exp_gW = D // p, H // p, W // p
        if (gD, gH, gW) != (exp_gD, exp_gH, exp_gW):
            raise ValueError(f"phi_map shape {phi_map.shape} != latent grid {(exp_gD, exp_gH, exp_gW)}")

        w = self.window_size
        r = w // 2
        order_used = self._sample_order_for_item()
        direction_used = self._sample_direction_for_item()

        # 采样得到目标patch的中心坐标（ti, tj, tk），范围在 [0, gD), [0, gH), [0, gW)
        ti, tj, tk = self._sample_anchor(gD, gH, gW, r, order_used, direction_used)

        # 窗口切片与填充 (Padding & Slicing)
        pad_p = r * p
        z_pad = _pad_with_mode(
            z_full,
            ((0, 0), (pad_p, pad_p), (pad_p, pad_p), (pad_p, pad_p)),
            self.pad_mode,
        )
        phi_pad = _pad_with_mode(phi_map, ((r, r), (r, r), (r, r)), self.pad_mode)

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

        # 构建patch 级别的已知/未知掩码（mask_patch），以及对应的条件输入（cond）和孔隙率体积（phi_vol）
        # 已知/未知定义：已知的patch是指在当前目标patch (ti,tj,tk) 的窗口范围内，且满足 context_mode 要求的patch。
        # 已知为1，未知为0。
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
                        known = _is_prev_by_mode(
                            (gi, gj, gk),
                            (ti, tj, tk),
                            self.context_mode,
                            order_used,
                            direction_used,
                        )
                    if known:
                        mask_patch[di, dj, dk] = 1.0

        # voxel mask & target mask
        mask = _repeat_phi(mask_patch, p)[None, ...]  # (1, w*p, w*p, w*p)
        target_patch = np.zeros_like(mask_patch)
        target_patch[r, r, r] = 1.0
        target_mask = _repeat_phi(target_patch, p)[None, ...]

        cond = z_win * mask
        local_phi_val = float(phi_map[ti, tj, tk])
        global_phi_val = float(phi_map.mean())

        local_phi_vol = _repeat_phi(phi_win, p)
        if self.use_global_phi_channel:
            global_phi_patch = np.full_like(phi_win, global_phi_val, dtype=np.float32)
            global_phi_vol = _repeat_phi(global_phi_patch, p)
            phi_vol = np.stack([local_phi_vol, global_phi_vol], axis=0)
        else:
            phi_vol = local_phi_vol[None, ...]

        if self.porosity_mode == "global":
            por = self.porosity_map.get(base_name, global_phi_val)
        elif self.porosity_mode in ("mix", "local_global_mix"):
            a = float(np.clip(self.porosity_mix_alpha, 0.0, 1.0))
            por = a * local_phi_val + (1.0 - a) * global_phi_val
        else:
            por = local_phi_val
        porosity = np.array([por], dtype=np.float32)

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

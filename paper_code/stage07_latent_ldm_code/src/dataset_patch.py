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
    """移除文件名中的 porosity 前缀，统一成可与 phi_map 对齐的基名。"""
    # 从文件名中提取孔隙率信息并返回去除孔隙率前缀后的名字
    # support: porosity_0.123456_xxx.npy -> xxx.npy
    return re.sub(r"^porosity_[0-9]*\\.?[0-9]+_", "", name)


def _load_npy(path: str) -> np.ndarray:
    """读取 .npy 并统一转为 float32，减少后续 dtype 分歧。"""
    # 安全加载npy文件，确保输出为float32类型
    return np.load(path).astype(np.float32)


def _repeat_phi(phi_patch: np.ndarray, patch_size: int) -> np.ndarray:
    """将 patch 级 phi 通过 repeat 扩展到 voxel 级分辨率。"""
    # phi_patch: (w, w, w) -> (w*ps, w*ps, w*ps)
    out = np.repeat(phi_patch, patch_size, axis=0)
    out = np.repeat(out, patch_size, axis=1)
    out = np.repeat(out, patch_size, axis=2)
    return out


def _masked_mean_with_fallback(values: np.ndarray, mask: np.ndarray, fallback: float) -> float:
    """返回 values 在 mask>0.5 区域的均值；若为空则回退 fallback。"""
    sel = mask > 0.5
    if not np.any(sel):
        return float(fallback)
    return float(values[sel].mean())


def _pad_with_mode(x: np.ndarray, pad_width, mode: str) -> np.ndarray:
    """按给定模式做 padding；当 reflect 非法时自动回退到 edge。"""
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
    """标准化遍历顺序字符串，非法输入回退为 'ijk'。"""
    # 将顺序字符串标准化为 "ijk" 格式，默认返回 "ijk"
    order = str(order).lower().strip()
    if len(order) != 3 or set(order) != {"i", "j", "k"}:
        return "ijk"
    return order


def _value_to_sign(v) -> int:
    """把多种方向表示（字符/数字）统一映射到 +1 或 -1。"""
    if isinstance(v, (int, np.integer, float, np.floating)):
        return 1 if float(v) >= 0 else -1
    txt = str(v).strip().lower()
    if txt in ("+", "1", "+1", "pos", "forward", "fwd"):
        return 1
    if txt in ("-", "-1", "neg", "reverse", "rev", "backward", "bwd"):
        return -1
    return 1


def _normalize_direction(direction) -> Direction3D:
    """标准化方向输入，返回 (si, sj, sk) 形式的 ±1 三元组。"""
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
    """标准化上下文模式，非法输入回退为 causal。"""
    mode = str(mode).lower().strip()
    if mode not in ("causal", "full", "wavefront"):
        return "causal"
    return mode


def _normalize_pad_mode(mode: str) -> str:
    """标准化 padding 模式，非法输入回退为 edge。"""
    mode = str(mode).lower().strip()
    if mode not in ("constant", "edge", "reflect"):
        return "edge"
    return mode


def _normalize_anchor_sampling_mode(mode: str) -> str:
    """标准化 anchor 采样模式，非法输入回退为 uniform。"""
    mode = str(mode).lower().strip()
    if mode not in ("uniform", "low_context_boost", "porosity_balanced"):
        return "uniform"
    return mode


def _normalize_sampler_semantic(semantic: str) -> str:
    """标准化语义标签，统一为 pore 或 rock_rate。"""
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
    """在给定 order+direction 下判断 a 是否位于 b 的“前序”位置。"""
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
    """在 wavefront 规则下判断 a 是否为 b 的前序点。"""
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
    """根据 context_mode 分发到对应的前序判定函数。"""
    if context_mode == "wavefront":
        return _is_prev_wavefront(a, b, direction)
    return _is_prev_lexicographic(a, b, order, direction)


def _load_porosity_map(csv_path: str) -> dict:
    """从 CSV 读取样本级全局孔隙率映射 {filename: porosity}。"""
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
    """Stage07 训练数据集：按样本动态采样中心 patch 并构造条件输入。"""

    def __init__(self, 
                 latent_dir: str,       # 被KLVAE 压缩的潜在空间文件夹路径
                 phi_map_dir: str,      # latent对应的phi map文件夹路径
                 augment: bool = True   # 是否进行数据增强（翻转）
                 ):
        """
        初始化数据集索引、采样策略和条件构造相关超参数。

        说明:
        - 这里只建立样本对（latent, phi_map）和配置缓存；
        - 具体的窗口切片与掩码构造在 __getitem__ 内执行。
        """
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
        self.anchor_porosity_semantic = _normalize_sampler_semantic(
            CONFIG.get("anchor_porosity_semantic", CONFIG.get("porosity_sampler_semantic", "pore"))
        )
        self.anchor_porosity_power = float(CONFIG.get("anchor_porosity_power", 1.0))
        self.anchor_porosity_min_weight = float(CONFIG.get("anchor_porosity_min_weight", 0.05))
        ap_edges_cfg = CONFIG.get("anchor_porosity_bin_edges", CONFIG.get("porosity_bin_edges", [0.0, 0.25, 0.5, 0.75, 1.0]))
        try:
            ap_edges = [float(v) for v in ap_edges_cfg]
        except Exception:
            ap_edges = [0.0, 0.25, 0.5, 0.75, 1.0]
        if len(ap_edges) < 2 or not np.all(np.diff(np.array(ap_edges, dtype=np.float64)) > 0):
            ap_edges = [0.0, 0.25, 0.5, 0.75, 1.0]
        self.anchor_porosity_edges = np.array(ap_edges, dtype=np.float64)
        self.pad_mode = _normalize_pad_mode(CONFIG.get("pad_mode", "edge"))
        self.porosity_mode = str(CONFIG.get("porosity_mode", "local")).lower()
        self.porosity_mix_alpha = float(CONFIG.get("porosity_mix_alpha", 0.7))
        self.use_global_phi_channel = bool(CONFIG.get("use_global_phi_channel", False))
        self.use_dynamic_porosity_condition = bool(CONFIG.get("use_dynamic_porosity_condition", False))
        self.dynamic_phi_include_target = bool(CONFIG.get("dynamic_phi_include_target", True))
        self.dynamic_global_phi_channel = bool(CONFIG.get("dynamic_global_phi_channel", True))
        self.porosity_map = _load_porosity_map(CONFIG.get("porosity_csv", ""))
        # Context Dropout：以一定概率将所有上下文 patch 置零，模拟推理时无上下文场景
        self.context_drop_prob = float(CONFIG.get("context_drop_prob", 0.0))
        self.porosity_sampler_semantic = _normalize_sampler_semantic(
            CONFIG.get("porosity_sampler_semantic", "pore")
        )
        self.sample_phi_means = None    # 保存了每个样本的 phi 均值，供采样权重计算使用
        self._last_sampler_stats = {}

    def _get_sample_phi_means(self) -> np.ndarray:
        """计算并缓存每个体样本的 phi 均值（样本级统计）。"""
        # 计算“样本级”phi均值（每个体样本一个数），并做缓存避免每个 epoch 重复读盘。
        # 这里的样本级统计用于 DataLoader 的重采样，不影响 __getitem__ 内部的 patch 抽取逻辑。
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
        """
        根据样本级孔隙率分箱，构建 WeightedRandomSampler 的样本权重。返回一个与样本数相同的权重数组，权重值越大表示该样本被采样的概率越高。

        计算流程（与 train.py 中的 weighted sampler 一一对应）：
        1) 取每个样本的 phi 均值 -> vals
        2) 若语义为 pore，则使用 1-phi（把“孔隙率高”映射到更大值）
        3) 按 bin_edges 分箱，统计每个箱子的样本数 count
        4) 每个箱子的基础权重 = (1 / count) ** power（稀有箱权重更大）
        5) 映射回每个样本，得到 sample_w
        6) 归一化到均值约 1，再裁剪到 [min_weight, max_weight]

        小例子：
        - vals = [0.02, 0.03, 0.04, 0.30]
        - bin_edges = [0.0, 0.1, 0.5]
        - 分箱计数 count = [3, 1]
        - 当 power=1 时，箱权重 = [1/3, 1]
        - 样本权重 = [1/3, 1/3, 1/3, 1]（最后再做均值归一化与裁剪）
        """
        vals = self._get_sample_phi_means().astype(np.float64)
        if self.porosity_sampler_semantic == "pore":
            vals = 1.0 - vals
        vals = np.clip(vals, 0.0, 1.0)
        if len(bin_edges) < 2:
            raise ValueError("porosity bin_edges must contain at least 2 values.")
        edges = np.array(bin_edges, dtype=np.float64)
        if not np.all(np.diff(edges) > 0):
            raise ValueError("porosity bin_edges must be strictly increasing.")

        # 每个样本映射到分箱 id（范围 [0, num_bins-1]）
        bin_ids = np.digitize(vals, edges[1:-1], right=False)
        num_bins = len(edges) - 1
        counts = np.bincount(bin_ids, minlength=num_bins).astype(np.float64)

        # 稀有分箱（count 小）获得更大的逆频率权重
        inv = np.zeros_like(counts)
        nz = counts > 0
        inv[nz] = 1.0 / counts[nz]
        inv = np.power(inv, max(power, 0.0))
        sample_w = inv[bin_ids]

        # 将样本权重均值归一到 1 附近，减少对整体学习率尺度的扰动
        mean_w = float(sample_w.mean()) if sample_w.size > 0 else 1.0
        if mean_w > 0:
            sample_w = sample_w / mean_w
        # 裁剪上下界，避免极端权重导致训练不稳定
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
        """返回样本对数量（等于可用的 (latent, phi_map) 对数）。"""
        return len(self.pairs)

    def _sample_order_for_item(self) -> str:
        """按配置返回当前样本使用的遍历顺序（可固定或随机）。"""
        if self.context_mode != "causal":
            return self.order
        if not self.train_random_order:
            return self.order
        return random.choice(self.order_candidates)

    def _sample_direction_for_item(self) -> Direction3D:
        """按配置返回当前样本使用的遍历方向（可固定或随机）。"""
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
        phi_map: np.ndarray = None,
    ) -> Tuple[int, int, int]:
        """
        采样“目标中心 patch”坐标 (ti, tj, tk)。

        这是 __getitem__ 的关键一步：
        - 先选一个中心 patch（anchor）。
        - 再围绕该中心截取 window（w x w x w）的上下文。
        - 网络最终重点预测 window 中心位置对应的目标 patch。

        三种模式：
        1) uniform: 在整张网格上均匀随机选点。
        2) porosity_balanced: 按 phi 分箱做均衡抽样（稀有孔隙率区间权重大）。
        3) 其他模式（当前实现主要是 low_context_boost）：
           按“可用前序上下文数量”加权，前序越少越容易被选中。
        """
        if self.anchor_sampling_mode == "uniform":
            # 完全均匀随机：每个网格点概率相同
            return (
                random.randint(0, gD - 1),
                random.randint(0, gH - 1),
                random.randint(0, gW - 1),
            )
        if self.anchor_sampling_mode == "porosity_balanced" and phi_map is not None:
            # 按 phi_map 分箱后的逆频率采样，优先覆盖稀有孔隙率区域
            return self._sample_anchor_porosity_balanced(phi_map)

        # 低上下文增强（low_context_boost）分支：
        # 遍历所有候选坐标，为每个候选点计算一个权重后再按权重抽样。
        coords = [(i, j, k) for i in range(gD) for j in range(gH) for k in range(gW)]
        weights: List[float] = []
        pwr = max(self.anchor_boost_power, 0.0)
        min_w = max(self.anchor_boost_min_weight, 0.0)

        for ti, tj, tk in coords:
            # prev_count: 在该候选中心的局部窗口内，被定义为“前序”的邻居个数
            # 直觉上：prev_count 越小，已知上下文越少，任务更难，应该被适度强化抽样。
            prev_count = 0
            for gi in range(max(0, ti - r), min(gD, ti + r + 1)):
                for gj in range(max(0, tj - r), min(gH, tj + r + 1)):
                    for gk in range(max(0, tk - r), min(gW, tk + r + 1)):
                        if gi == ti and gj == tj and gk == tk:
                            continue
                        if self.context_mode == "full":
                            # full 模式下，中心外所有邻居都视为可用前序
                            is_prev = True
                        else:
                            # causal / wavefront 模式下，按顺序与方向判断前后关系
                            is_prev = _is_prev_by_mode(
                                (gi, gj, gk),
                                (ti, tj, tk),
                                self.context_mode,
                                order_used,
                                direction_used,
                            )
                        if is_prev:
                            prev_count += 1
            # 权重公式：w = 1 / (prev_count + 1)^pwr
            # - pwr=0 时退化为均匀采样
            # - pwr 越大，越偏向 prev_count 小（低上下文）的位置
            w = 1.0 / ((prev_count + 1) ** pwr if pwr > 0 else 1.0)
            # 下限裁剪，避免某些点权重过低几乎永远不被采样
            weights.append(max(w, min_w))

        # 按权重抽取一个中心坐标，作为本次训练样本的目标 patch 中心
        pick = random.choices(coords, weights=weights, k=1)[0]
        return int(pick[0]), int(pick[1]), int(pick[2])

    def _sample_anchor_porosity_balanced(self, phi_map: np.ndarray) -> Tuple[int, int, int]:
        """
        按“孔隙率分箱逆频率”采样中心 patch。

        过程与样本级 WeightedRandomSampler 类似，但这里是“单个样本内部的网格点级别”：
        - 把 phi_map 展平为每个网格点一个值；
        - 分箱统计每箱数量；
        - 箱权重取 1/count（可再用 power 调节）；
        - 将箱权重映射回每个网格点，按概率采样一个点。
        """
        gD, gH, gW = phi_map.shape
        # vals: 每个网格点的局部 phi（shape = gD*gH*gW）
        vals = phi_map.reshape(-1).astype(np.float64)
        if self.anchor_porosity_semantic == "pore":
            # 若采用 pore 语义，则用 1-phi 作为“孔隙率”值
            vals = 1.0 - vals
        vals = np.clip(vals, 0.0, 1.0)
        edges = self.anchor_porosity_edges
        # bin_ids: 每个网格点所属分箱
        bin_ids = np.digitize(vals, edges[1:-1], right=False)
        num_bins = len(edges) - 1
        # counts: 每个分箱内有多少网格点
        counts = np.bincount(bin_ids, minlength=num_bins).astype(np.float64)
        inv = np.zeros_like(counts)
        nz = counts > 0
        # 逆频率：稀有分箱获得更高权重
        inv[nz] = 1.0 / counts[nz]
        inv = np.power(inv, max(self.anchor_porosity_power, 0.0))
        # 将分箱权重映射回每个网格点
        w = inv[bin_ids]
        w = np.clip(w, max(self.anchor_porosity_min_weight, 1e-8), None)
        w_sum = float(w.sum())
        if not np.isfinite(w_sum) or w_sum <= 0.0:
            # 数值异常时回退到均匀随机
            pick = random.randint(0, vals.shape[0] - 1)
        else:
            pick = int(np.random.choice(vals.shape[0], p=(w / w_sum)))
        # 将展平索引还原为 (ti, tj, tk)
        ti = int(pick // (gH * gW))
        rem = int(pick % (gH * gW))
        tj = int(rem // gW)
        tk = int(rem % gW)
        return ti, tj, tk

    def __getitem__(self, idx):
        """
        构造一个训练样本（以“单个中心 patch”为监督目标）。

        参数:
            idx: 数据集中的“体样本索引”（不是 patch 索引）。

        返回:
            dict，包含以下键（均为 torch.float32）:
            - GT:        当前窗口内完整 latent，形状 (C, w*p, w*p, w*p)
            - Condition: 已知上下文 latent（未知区域置 0），形状同 GT
            - Mask:      已知区域掩码，形状 (1, w*p, w*p, w*p)
            - TargetMask:仅中心目标 patch 为 1 的掩码，形状同 Mask
            - Phi:       孔隙率条件体（1 或 2 通道），形状 (phi_ch, w*p, w*p, w*p)
            - Porosity:  标量孔隙率条件，形状 (1,)

        核心流程:
            1) 读取一个体样本 (latent + phi_map)，并做 scale/clamp。
            2) 采样中心 patch 坐标 (ti, tj, tk)。
            3) 围绕中心截取 window，构造 GT / Condition / Mask / TargetMask。
            4) 构造 Phi 体条件与 Porosity 标量条件。
            5) 数据增强（随机翻转）并转为 contiguous。
        """
        # 1) 加载单个体样本
        latent_path, phi_path, base_name = self.pairs[idx]
        z_full = _load_npy(latent_path)  # (C, D, H, W)，例如 (4, 24, 24, 24)
        phi_map = _load_npy(phi_path)    # (gD, gH, gW)，例如 (3, 3, 3)

        # 兼容形状 (1, C, D, H, W) 的历史数据
        if z_full.ndim == 5:
            z_full = z_full.squeeze(0)

        # 训练前对 latent 做统一缩放和安全截断
        z_full = z_full * self.scale_factor
        z_full = np.clip(z_full, -self.safe_threshold, self.safe_threshold)

        C, D, H, W = z_full.shape
        gD, gH, gW = phi_map.shape

        # 2) 采样中心 patch（anchor）
        # p: latent 空间中每个 patch 的边长（单位：latent voxel）
        p = self.patch_size
        # 校验 phi_map 网格尺寸是否与 latent 可整除网格一致
        exp_gD, exp_gH, exp_gW = D // p, H // p, W // p
        if (gD, gH, gW) != (exp_gD, exp_gH, exp_gW):
            raise ValueError(f"phi_map shape {phi_map.shape} != latent grid {(exp_gD, exp_gH, exp_gW)}")

        w = self.window_size           # patch 级窗口边长
        r = w // 2                     # patch 级窗口半径
        order_used = self._sample_order_for_item()
        direction_used = self._sample_direction_for_item()

        # 中心 patch 坐标范围: ti in [0,gD), tj in [0,gH), tk in [0,gW)
        # 该坐标决定本次样本“预测哪个中心 patch”
        ti, tj, tk = self._sample_anchor(gD, gH, gW, r, order_used, direction_used, phi_map=phi_map)

        # 3) 按窗口截取数据（先 pad 再切，便于处理边界）
        pad_p = r * p
        z_pad = _pad_with_mode(
            z_full,
            ((0, 0), (pad_p, pad_p), (pad_p, pad_p), (pad_p, pad_p)),
            self.pad_mode,
        )
        phi_pad = _pad_with_mode(phi_map, ((r, r), (r, r), (r, r)), self.pad_mode)

        ci, cj, ck = ti + r, tj + r, tk + r

        # patch 单位下的窗口切片坐标
        wi0, wi1 = ci - r, ci + r + 1
        wj0, wj1 = cj - r, cj + r + 1
        wk0, wk1 = ck - r, ck + r + 1

        phi_win = phi_pad[wi0:wi1, wj0:wj1, wk0:wk1]  # (w, w, w)

        # 转换到 latent voxel 单位下的窗口切片坐标
        zi0, zi1 = wi0 * p, wi1 * p
        zj0, zj1 = wj0 * p, wj1 * p
        zk0, zk1 = wk0 * p, wk1 * p
        z_win = z_pad[:, zi0:zi1, zj0:zj1, zk0:zk1]  # (C, w*p, w*p, w*p)

        # 构建 patch 级已知/未知掩码:
        # - known=1: 满足 context_mode 前序条件的邻域 patch
        # - known=0: 需要模型预测的区域
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

        # ── Context Dropout：训练时以 context_drop_prob 概率清零全部上下文 ──
        # 目的：让模型学会在无上下文时仅凭 phi_map + porosity 独立生成，
        # 弥合推理时首批 patch（无已知邻居）的训练-推理分布差异。
        if self.augment and self.context_drop_prob > 0.0 and random.random() < self.context_drop_prob:
            mask_patch = np.zeros_like(mask_patch)  # 全部上下文置为未知

        # 将 patch 级掩码扩展为 voxel 级掩码
        mask = _repeat_phi(mask_patch, p)[None, ...]  # (1, w*p, w*p, w*p)

        # TargetMask 只监督中心 patch（中心外为 0）
        target_patch = np.zeros_like(mask_patch)
        target_patch[r, r, r] = 1.0
        target_mask = _repeat_phi(target_patch, p)[None, ...]

        # Condition: 仅保留已知上下文，未知区域置零
        cond = z_win * mask
        local_phi_val = float(phi_map[ti, tj, tk])
        global_phi_val = float(phi_map.mean())

        dynamic_mask_patch = mask_patch.copy()
        if self.dynamic_phi_include_target:
            dynamic_mask_patch[r, r, r] = 1.0
        dynamic_phi_val = _masked_mean_with_fallback(phi_win, dynamic_mask_patch, fallback=global_phi_val)
        global_like_phi_val = dynamic_phi_val if self.use_dynamic_porosity_condition else global_phi_val

        # 4) 构建 Phi 体条件（局部 phi，可选叠加全局 phi 通道）
        local_phi_vol = _repeat_phi(phi_win, p)
        if self.use_global_phi_channel:
            global_phi_ch_val = global_like_phi_val if self.dynamic_global_phi_channel else global_phi_val
            global_phi_patch = np.full_like(phi_win, global_phi_ch_val, dtype=np.float32)
            global_phi_vol = _repeat_phi(global_phi_patch, p)
            phi_vol = np.stack([local_phi_vol, global_phi_vol], axis=0)
        else:
            phi_vol = local_phi_vol[None, ...]

        # 构建 Porosity 标量条件（local / global / mix）
        if self.porosity_mode == "global":
            if self.use_dynamic_porosity_condition:
                por = global_like_phi_val
            else:
                por = self.porosity_map.get(base_name, global_phi_val)
        elif self.porosity_mode in ("mix", "local_global_mix"):
            a = float(np.clip(self.porosity_mix_alpha, 0.0, 1.0))
            por = a * local_phi_val + (1.0 - a) * global_like_phi_val
        else:
            por = local_phi_val
        porosity = np.array([por], dtype=np.float32)

        # 5) 数据增强：3 个轴独立随机翻转（GT/Condition/Mask/Phi 必须同步翻转）
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

        # 转为 contiguous，避免 from_numpy 后出现 stride 相关性能/兼容问题
        z_win = np.ascontiguousarray(z_win)
        cond = np.ascontiguousarray(cond)
        mask = np.ascontiguousarray(mask)
        target_mask = np.ascontiguousarray(target_mask)
        phi_vol = np.ascontiguousarray(phi_vol)

        # 返回训练所需全部输入/监督张量
        return {
            "GT": torch.from_numpy(z_win).float(),
            "Condition": torch.from_numpy(cond).float(),
            "Mask": torch.from_numpy(mask).float(),
            "TargetMask": torch.from_numpy(target_mask).float(),
            "Phi": torch.from_numpy(phi_vol).float(),
            "Porosity": torch.from_numpy(porosity).float(),
        }

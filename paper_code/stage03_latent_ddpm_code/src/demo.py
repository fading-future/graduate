import os
import re
import glob
import math
import json
import random
import argparse
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import numpy as np
import torch
import matplotlib.pyplot as plt

# ============================================================
# ✅ 你需要改的路径（也可以用命令行参数覆盖）
# ============================================================
LATENT_DIR = r"E:\stage2_latents_full_256"   # 你的 stage1 生成的 latent npy 文件夹（你已给）
RAW_DATA_ROOT = r""                          # 可选：原始 256^3 数据所在目录（用于 stage1 解码对比）
STAGE1_CONFIG = r""                          # 可选：stage1 KLVAE 的 yaml 配置路径（Windows）
STAGE1_CKPT = r""                            # 可选：stage1 KLVAE 的 checkpoint 路径（Windows）
OUT_DIR = r"./_check_outputs"                # 输出目录

# Stage2 相关：用于评估 scale/clamp 是否在制造伪影
# 你现在 stage2 dataset 里是：latent = latent * scale_factor; clamp(-safe_threshold, safe_threshold)
STAGE2_SCALE_FACTOR = None   # 如果你已经设置了 stage2 的 scale_factor，填数字；否则脚本会从 latent std 推一个建议值
STAGE2_SAFE_THRESHOLD = 6.0  # 你现在常用的 safe_threshold

# 随机抽样数量：越大统计越稳，但越慢（latent 很小 4*32^3，其实很快）
SAMPLE_FILES = 200
SEED = 1234


# ============================================================
# 工具函数：通用
# ============================================================
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def human_bytes(n: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    x = float(n)
    for u in units:
        if x < 1024 or u == units[-1]:
            return f"{x:.2f}{u}"
        x /= 1024.0


def list_npy_files(latent_dir: str) -> List[str]:
    files = sorted(glob.glob(os.path.join(latent_dir, "*.npy")))
    return files


def parse_original_name(latent_filename: str) -> str:
    """
    你的 latent 保存名：porosity_{por:.6f}_{orig_name}
    这里把 orig_name 取出来（保留原始扩展名 .npy）
    """
    base = os.path.basename(latent_filename)
    m = re.match(r"porosity_[0-9\.]+_(.+)$", base)
    if m:
        return m.group(1)
    # fallback：如果不是这个格式，就直接返回原名
    return base


def load_latent(path: str) -> np.ndarray:
    z = np.load(path)
    # 期望 shape: (4,32,32,32)
    return z.astype(np.float32)


def load_raw_volume(raw_root: str, orig_name: str) -> Optional[np.ndarray]:
    """
    尝试在 raw_root 中找到对应的原始 256^3 数据。
    你原始数据命名可能与 orig_name 完全一致，也可能在子目录里。
    """
    if not raw_root:
        return None
    cand = os.path.join(raw_root, orig_name)
    if os.path.exists(cand):
        return np.load(cand).astype(np.float32)

    # 子目录递归搜索（只在找不到时做，避免太慢）
    hits = glob.glob(os.path.join(raw_root, "**", orig_name), recursive=True)
    if len(hits) > 0:
        return np.load(hits[0]).astype(np.float32)

    return None


def robust_percentiles(x: np.ndarray, ps=(0.1, 1, 5, 50, 95, 99, 99.9)) -> Dict[str, float]:
    out = {}
    for p in ps:
        out[str(p)] = float(np.percentile(x, p))
    return out


# ============================================================
# 统计与诊断：latent 分布、截断风险、噪声/高频倾向
# ============================================================
@dataclass
class LatentStats:
    n_files: int
    shape: Tuple[int, int, int, int]
    mean: float
    std: float
    minv: float
    maxv: float
    pct: Dict[str, float]
    per_channel: List[Dict[str, float]]

    # 截断风险
    suggested_scale_factor: float
    clamp_ratio_if_scaled: float
    clamp_ratio_raw: float


def compute_latent_stats(latents: List[np.ndarray], safe_threshold: float, scale_factor: Optional[float]) -> LatentStats:
    # 合并统计（flatten）
    flat_all = np.concatenate([z.reshape(-1) for z in latents], axis=0)

    mean = float(flat_all.mean())
    std = float(flat_all.std() + 1e-12)
    minv = float(flat_all.min())
    maxv = float(flat_all.max())
    pct = robust_percentiles(flat_all)

    # 每通道统计
    per_channel = []
    C = latents[0].shape[0]
    for c in range(C):
        fc = np.concatenate([z[c].reshape(-1) for z in latents], axis=0)
        per_channel.append({
            "c": c,
            "mean": float(fc.mean()),
            "std": float(fc.std() + 1e-12),
            "min": float(fc.min()),
            "max": float(fc.max()),
            "p99.9": float(np.percentile(fc, 99.9)),
            "p0.1": float(np.percentile(fc, 0.1)),
        })

    # 建议 scale_factor：1/std（你生成脚本也是这么建议的）
    suggested_scale = float(1.0 / std)

    # raw clamp 比例（不乘 scale）
    clamp_ratio_raw = float((np.abs(flat_all) >= safe_threshold).mean())

    # scaled clamp 比例（乘 scale_factor 后再 clamp）
    if scale_factor is None:
        sf = suggested_scale
    else:
        sf = float(scale_factor)

    flat_scaled = flat_all * sf
    clamp_ratio_scaled = float((np.abs(flat_scaled) >= safe_threshold).mean())

    return LatentStats(
        n_files=len(latents),
        shape=latents[0].shape,
        mean=mean,
        std=std,
        minv=minv,
        maxv=maxv,
        pct=pct,
        per_channel=per_channel,
        suggested_scale_factor=suggested_scale,
        clamp_ratio_if_scaled=clamp_ratio_scaled,
        clamp_ratio_raw=clamp_ratio_raw,
    )


def high_freq_indicator(z: np.ndarray) -> Dict[str, float]:
    """
    一个非常轻量的“高频/噪声倾向”指标：用一阶差分的能量比例衡量。
    - 如果 latent 主要是平滑低频结构，差分能量会相对低
    - 如果 latent 充满碎纹理/噪声，差分能量会相对高
    """
    # z: (C,D,H,W)
    eps = 1e-12
    base = float(np.mean(z * z) + eps)

    dz = np.diff(z, axis=1)
    dy = np.diff(z, axis=2)
    dx = np.diff(z, axis=3)

    diff_energy = float(np.mean(dz * dz) + np.mean(dy * dy) + np.mean(dx * dx))
    ratio = diff_energy / base
    return {"base_energy": base, "diff_energy": diff_energy, "diff/base": float(ratio)}


def plot_histograms(latents: List[np.ndarray], out_dir: str, tag: str):
    """
    画整体 latent 直方图 + 每通道直方图（便于看是否重尾、是否偏移、是否有饱和趋势）
    """
    ensure_dir(out_dir)
    C = latents[0].shape[0]

    flat_all = np.concatenate([z.reshape(-1) for z in latents], axis=0)
    plt.figure(figsize=(10, 4), dpi=150)
    plt.hist(flat_all, bins=200)
    plt.title(f"Latent Histogram (All) - {tag}")
    plt.xlabel("value")
    plt.ylabel("count")
    plt.grid(True, alpha=0.3, linestyle="--")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"hist_all_{tag}.png"))
    plt.close()

    for c in range(C):
        fc = np.concatenate([z[c].reshape(-1) for z in latents], axis=0)
        plt.figure(figsize=(10, 4), dpi=150)
        plt.hist(fc, bins=200)
        plt.title(f"Latent Histogram (Channel {c}) - {tag}")
        plt.xlabel("value")
        plt.ylabel("count")
        plt.grid(True, alpha=0.3, linestyle="--")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"hist_c{c}_{tag}.png"))
        plt.close()


def plot_slices_from_latent(z: np.ndarray, out_path: str, title: str):
    """
    把 latent 的每个通道取一个中心切片（D/H/W 三个方向）看看是否有块状/振铃/棋盘/噪点纹理
    """
    C, D, H, W = z.shape
    cd, ch, cw = D // 2, H // 2, W // 2

    fig, axes = plt.subplots(C, 3, figsize=(10, 2.6 * C), dpi=150)
    if C == 1:
        axes = np.expand_dims(axes, axis=0)

    for c in range(C):
        xy = z[c, cd, :, :]
        xz = z[c, :, ch, :]
        yz = z[c, :, :, cw]
        axes[c, 0].imshow(xy, cmap="gray")
        axes[c, 0].set_title(f"c{c} XY (D={cd})")
        axes[c, 1].imshow(xz, cmap="gray")
        axes[c, 1].set_title(f"c{c} XZ (H={ch})")
        axes[c, 2].imshow(yz, cmap="gray")
        axes[c, 2].set_title(f"c{c} YZ (W={cw})")
        for j in range(3):
            axes[c, j].axis("off")

    fig.suptitle(title, fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


# ============================================================
# 可选：Stage1 模型 decode 重建对比（需要 KLVAE3D + cfg + ckpt）
# ============================================================
def try_load_stage1_vae(stage1_config: str, stage1_ckpt: str, device: str = "cuda"):
    """
    尝试加载 KLVAE3D（如果你 Windows 环境里有对应代码与权重）。
    成功返回 (vae_model, cfg_dict)，失败返回 None。
    """
    if (not stage1_config) or (not stage1_ckpt):
        return None

    if (not os.path.exists(stage1_config)) or (not os.path.exists(stage1_ckpt)):
        print(f"[SKIP] stage1 config/ckpt 路径不存在：\n  cfg={stage1_config}\n  ckpt={stage1_ckpt}")
        return None

    try:
        import yaml
        from models.vae import KLVAE3D
    except Exception as e:
        print(f"[SKIP] 无法 import KLVAE3D / yaml（你的 Windows 项目里可能没有 stage1 代码）：{e}")
        return None

    with open(stage1_config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    vae = KLVAE3D(cfg).to(device)
    ckpt = torch.load(stage1_ckpt, map_location=device)
    state_dict = ckpt.get("vae_state_dict", ckpt)

    # 兼容你之前的 _orig_mod. 前缀
    new_state = {}
    for k, v in state_dict.items():
        if k.startswith("_orig_mod."):
            new_state[k[10:]] = v
        else:
            new_state[k] = v
    vae.load_state_dict(new_state, strict=True)
    vae.eval()
    print("[OK] Loaded Stage1 KLVAE3D for decoding check.")
    return vae, cfg


@torch.no_grad()
def decode_latent_with_stage1(vae, z_np: np.ndarray, device: str = "cuda") -> np.ndarray:
    """
    用 stage1 decoder 把 latent decode 回 256^3（或 stage1 的输出尺度）
    兼容：decode 返回 Tensor / Distribution-like / tuple(list)
    """
    z = torch.from_numpy(z_np).unsqueeze(0).to(device)  # (1,4,32,32,32)

    out = vae.decode(z)

    # 1) 最优先：如果 out 本身就是 Tensor
    if torch.is_tensor(out):
        x = out

    # 2) 有些实现会返回 (tensor, ...) 或 [tensor, ...]
    elif isinstance(out, (tuple, list)) and len(out) > 0 and torch.is_tensor(out[0]):
        x = out[0]

    # 3) Distribution-like：mean / sample 可能是 Tensor 属性或可调用
    else:
        # mean 是 Tensor 属性的情况（常见）
        if hasattr(out, "mean") and torch.is_tensor(getattr(out, "mean")):
            x = out.mean
        # sample 可调用的情况（常见）
        elif hasattr(out, "sample") and callable(getattr(out, "sample")):
            x = out.sample()
        # 有些实现叫 mode
        elif hasattr(out, "mode") and torch.is_tensor(getattr(out, "mode")):
            x = out.mode
        else:
            raise TypeError(f"Unsupported decode output type: {type(out)}")

    x = x.detach().float().cpu().numpy()

    # squeeze batch
    if x.ndim >= 1:
        x = np.squeeze(x, axis=0)  # (1, D, H, W) -> (D,H,W) 或 (C,D,H,W)

    return x



def plot_recon_compare(raw: np.ndarray, recon: np.ndarray, out_path: str, title: str):
    """
    画 GT vs Recon vs Diff 的三方向切片对比。
    raw: 原始体数据（可能是 uint16 或 float），建议已经归一化到 [-1,1]
    recon: decoder 输出（通常也是 [-1,1]）
    """
    # 统一成 shape: (D,H,W)
    if raw.ndim == 4:
        raw_v = raw[0]
    else:
        raw_v = raw
    if recon.ndim == 4:
        recon_v = recon[0]
    else:
        recon_v = recon

    D, H, W = raw_v.shape
    cd, ch, cw = D // 2, H // 2, W // 2

    slices = [
        ("XY", raw_v[cd], recon_v[cd]),
        ("XZ", raw_v[:, ch, :], recon_v[:, ch, :]),
        ("YZ", raw_v[:, :, cw], recon_v[:, :, cw]),
    ]

    fig, axes = plt.subplots(3, 3, figsize=(10, 9), dpi=150)
    for i, (name, gt, rc) in enumerate(slices):
        diff = rc - gt
        axes[i, 0].imshow(gt, cmap="gray");  axes[i, 0].set_title(f"GT {name}")
        axes[i, 1].imshow(rc, cmap="gray");  axes[i, 1].set_title(f"Recon {name}")
        axes[i, 2].imshow(diff, cmap="gray"); axes[i, 2].set_title(f"Diff {name}")
        for j in range(3):
            axes[i, j].axis("off")

    fig.suptitle(title, fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


# ============================================================
# 主流程：一键跑完所有校验
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--latent_dir", type=str, default=LATENT_DIR)
    parser.add_argument("--raw_root", type=str, default=RAW_DATA_ROOT)
    parser.add_argument("--stage1_config", type=str, default=STAGE1_CONFIG)
    parser.add_argument("--stage1_ckpt", type=str, default=STAGE1_CKPT)
    parser.add_argument("--out_dir", type=str, default=OUT_DIR)
    parser.add_argument("--sample_files", type=int, default=SAMPLE_FILES)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--safe_threshold", type=float, default=STAGE2_SAFE_THRESHOLD)
    parser.add_argument("--scale_factor", type=float, default=-1.0)  # -1 表示不用用户提供，自动建议
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    set_seed(args.seed)
    ensure_dir(args.out_dir)

    files = list_npy_files(args.latent_dir)
    if len(files) == 0:
        raise ValueError(f"在 {args.latent_dir} 没找到 .npy latent 文件。")

    # 抽样（统计用）
    pick_n = min(args.sample_files, len(files))
    sampled_files = random.sample(files, pick_n)

    print("=" * 80)
    print("[1] Basic Info")
    print(f"Latent dir: {args.latent_dir}")
    print(f"Total latent files: {len(files)}")
    print(f"Sampled for stats: {pick_n}")
    print(f"Output dir: {args.out_dir}")
    print(f"Device: {args.device}")
    print("=" * 80)

    # 读取 latent
    latents = []
    bad = 0
    for p in sampled_files:
        try:
            z = load_latent(p)
            if z.ndim != 4:
                print(f"[WARN] unexpected latent ndim={z.ndim} path={p}")
                continue
            latents.append(z)
        except Exception as e:
            bad += 1
            print(f"[WARN] failed load {p}: {e}")

    if len(latents) == 0:
        raise ValueError("抽样 latent 全部读取失败。")

    # scale_factor：用户给则用，否则自动建议（1/std）
    user_sf = None if args.scale_factor < 0 else float(args.scale_factor)

    stats = compute_latent_stats(
        latents=latents,
        safe_threshold=args.safe_threshold,
        scale_factor=user_sf
    )

    # 额外：抽几个样本计算“高频指标”
    hf_list = []
    for z in random.sample(latents, min(10, len(latents))):
        hf_list.append(high_freq_indicator(z))
    hf_mean = {
        "base_energy": float(np.mean([h["base_energy"] for h in hf_list])),
        "diff_energy": float(np.mean([h["diff_energy"] for h in hf_list])),
        "diff/base": float(np.mean([h["diff/base"] for h in hf_list])),
    }

    # 输出统计
    print("\n" + "=" * 80)
    print("[2] Latent Global Stats (sampled)")
    print(f"latent shape: {stats.shape}")
    print(f"mean: {stats.mean:.6f}")
    print(f"std : {stats.std:.6f}")
    print(f"min : {stats.minv:.6f}")
    print(f"max : {stats.maxv:.6f}")
    print(f"percentiles: {json.dumps(stats.pct, indent=2)}")
    print("\nPer-channel stats:")
    for cst in stats.per_channel:
        print(cst)

    print("\n[Stage2 scaling/clamp risk]")
    print(f"safe_threshold: {args.safe_threshold}")
    print(f"raw clamp ratio (|z|>=thr): {stats.clamp_ratio_raw * 100:.4f}%")

    if user_sf is None:
        print(f"Suggested scale_factor (1/std): {stats.suggested_scale_factor:.6f}")
        print(f"Clamp ratio after scaled+clamp (using suggested scale): {stats.clamp_ratio_if_scaled * 100:.4f}%")
    else:
        print(f"Using provided scale_factor: {user_sf:.6f}")
        print(f"Clamp ratio after scaled+clamp (using provided scale): {stats.clamp_ratio_if_scaled * 100:.4f}%")

    print("\n[High-frequency tendency indicator (rough)]")
    print(hf_mean)
    print("=" * 80)

    # 保存统计 JSON
    out_json = {
        "n_total_files": len(files),
        "n_sampled_files": pick_n,
        "latent_shape": stats.shape,
        "mean": stats.mean,
        "std": stats.std,
        "min": stats.minv,
        "max": stats.maxv,
        "percentiles": stats.pct,
        "per_channel": stats.per_channel,
        "safe_threshold": args.safe_threshold,
        "suggested_scale_factor": stats.suggested_scale_factor,
        "raw_clamp_ratio": stats.clamp_ratio_raw,
        "scaled_clamp_ratio": stats.clamp_ratio_if_scaled,
        "high_freq_indicator_mean": hf_mean,
        "bad_load_count": bad,
    }
    with open(os.path.join(args.out_dir, "latent_stats.json"), "w", encoding="utf-8") as f:
        json.dump(out_json, f, indent=2, ensure_ascii=False)

    # 画直方图
    plot_histograms(latents, args.out_dir, tag="sampled")

    # 画几个 latent 的切片（看是否有块状/噪点/棋盘）
    for i, p in enumerate(random.sample(sampled_files, min(5, len(sampled_files)))):
        z = load_latent(p)
        out_png = os.path.join(args.out_dir, f"latent_slices_{i}.png")
        plot_slices_from_latent(z, out_png, title=f"Latent slices: {os.path.basename(p)}")

    # ============================================================
    # [3] 可选：Stage1 decode 重建对比（如果你提供 stage1 cfg+ckpt + raw_root）
    # ============================================================
    stage1 = try_load_stage1_vae(args.stage1_config, args.stage1_ckpt, device=args.device)
    if stage1 is None:
        print("\n[3] Stage1 decode check: SKIPPED (no usable cfg/ckpt in this environment)")
        print("    如果你想做“重建伪影归因”，请把 stage1_config/stage1_ckpt/raw_root 配上再跑一次。")
        return

    vae, cfg = stage1

    if not args.raw_root:
        print("\n[3] Stage1 decode check: SKIPPED (raw_root not provided)")
        print("    你需要提供原始 256^3 数据目录 raw_root 才能做 GT vs Recon 对比。")
        return

    print("\n[3] Stage1 decode check: running recon comparisons...")

    # 随机挑 3 个文件做对比
    test_files = random.sample(files, min(3, len(files)))
    for i, lat_path in enumerate(test_files):
        orig_name = parse_original_name(lat_path)
        raw = load_raw_volume(args.raw_root, orig_name)
        if raw is None:
            print(f"[WARN] raw not found for {orig_name}, skip recon compare.")
            continue

        # 你 stage1 压缩脚本里对 raw 做了归一化 [-1,1]：data=(data/65535)*2-1
        # 如果 raw 是 uint16，这里也做同样归一化
        if raw.dtype != np.float32:
            raw = raw.astype(np.float32)
        # 如果 raw 范围看起来像 0~65535，做归一化
        if raw.max() > 10.0:
            raw = (raw / 65535.0) * 2.0 - 1.0

        z_np = load_latent(lat_path)
        recon = decode_latent_with_stage1(vae, z_np, device=args.device)

        out_png = os.path.join(args.out_dir, f"recon_compare_{i}.png")
        plot_recon_compare(raw, recon, out_png, title=f"Stage1 Recon Compare: {os.path.basename(lat_path)}")

        # 简单数值指标（MAE/MSE）
        rv = raw[0] if raw.ndim == 4 else raw
        rc = recon[0] if recon.ndim == 4 else recon
        mae = float(np.mean(np.abs(rc - rv)))
        mse = float(np.mean((rc - rv) ** 2))
        print(f"[Recon metrics] {i}  MAE={mae:.6f}  MSE={mse:.6f}  file={os.path.basename(lat_path)}")

    print("\n✅ All checks done.")
    print(f"Outputs saved to: {os.path.abspath(args.out_dir)}")


if __name__ == "__main__":
    main()



# #%%
# import os
# import glob

# #%%
# data_dir = r"E:\stage2_latents_full_256"

# #%%
# file_list = []

# if isinstance(data_dir, str):
#     data_dir_list = [data_dir]
# else:
#     data_dir_list = data_dir
    
# print(f"Loading data from {len(data_dir_list)} directories...")
# for d in data_dir_list:
#     if not os.path.exists(d):
#         print(f"⚠️ Warning: Directory not found: {d}")
#         continue
#     files = sorted(glob.glob(os.path.join(d, "*.npy")))
#     file_list.extend(files)

# print(file_list[:5])
# # %%

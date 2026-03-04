import os
import csv
import json
import re
from contextlib import contextmanager, nullcontext
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import autocast, GradScaler
from tqdm import tqdm

from src.config import CONFIG
from src.dataset_patch import PatchLatentDataset
from src.model_unet3d import ConditionalLatentUNet
from src.diffusion import DiffusionHelper
from src.ema import EMA
from utils.utils_path import get_root
from src.infer import ddim_sample
from model.vae import KLVAE3D

import numpy as np
import random
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml


def _safe_torch_load(path: str, map_location):
    """Prefer restricted checkpoint loading when supported by local PyTorch."""
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)
    except Exception:
        # Compatibility fallback for checkpoints that include unsupported objects.
        return torch.load(path, map_location=map_location)


@contextmanager
def _preserve_rng_state(device: torch.device):
    py_state = random.getstate()
    np_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    cuda_state = None
    if device.type == "cuda" and torch.cuda.is_available():
        cuda_state = torch.cuda.get_rng_state_all()
    try:
        yield
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
        torch.random.set_rng_state(torch_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)


def setup_experiment():
    """
    setup_experiment 的 Docstring
    parameters:
    returns:
        exp_dir: str, 实验主目录路径
        model_dir: str, 模型保存目录路径
        log_dir: str, 日志保存目录路径
        csv_path: str, 训练日志 CSV 文件路径
        csv_detail_path: str, 训练详细日志 CSV 文件路径
    """
    root = get_root()
    exp_dir = os.path.join(root, "exp_results", CONFIG["experiment_name"])
    os.makedirs(exp_dir, exist_ok=True)

    model_dir = os.path.join(exp_dir, "models")
    log_dir = os.path.join(exp_dir, "logs")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    with open(os.path.join(exp_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(CONFIG, f, indent=2, ensure_ascii=False)

    csv_path = os.path.join(log_dir, "training_log.csv")
    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Epoch", "Step", "Loss", "LR", "DiffTarget", "X0Target"])

    csv_detail_path = os.path.join(log_dir, "training_log_detailed.csv")
    if not os.path.exists(csv_detail_path):
        with open(csv_detail_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "Epoch",
                    "Step",
                    "Loss",
                    "LR",
                    "DiffTarget",
                    "X0Target",
                    "StatsTarget",
                    "PhiDecode",
                    "PhiProxy",
                    "PhiDecodeWeighted",
                    "PhiProxyWeighted",
                    "PhiDecodeApplied",
                    "PhiConsistencyWeight",
                    "PhiProxyWeight",
                    "PhiLossEverySteps",
                    "PhiLossMaxBatch",
                    "PhiTargetMean",
                ]
            )

    return exp_dir, model_dir, log_dir, csv_path, csv_detail_path


def _phi_channels() -> int:
    """
    _phi_channels 的 Docstring
    
    :return: phi 相关的输入通道数，取决于 CONFIG 中是否启用 use_global_phi_channel
    :rtype: int
    """
    return 2 if bool(CONFIG.get("use_global_phi_channel", False)) else 1


def _resolve_in_channels() -> int:
    # model input = x_t(C) + cond(C) + mask(1) + phi(phi_channels)
    c = int(CONFIG.get("out_channels", CONFIG.get("latent_channels", 4)))
    expected = 2 * c + 1 + _phi_channels()
    cfg_in = int(CONFIG.get("in_channels", expected))
    if cfg_in != expected:
        print(f"[warn] CONFIG['in_channels']={cfg_in} != expected={expected}; using expected.")
    return expected


def load_latest_checkpoint(model_dir, device):
    latest_path = os.path.join(model_dir, "unet_latest.pth")
    if os.path.exists(latest_path):
        return latest_path
    ckpts = [f for f in os.listdir(model_dir) if f.startswith("unet_epoch_") and f.endswith(".pth")]
    if len(ckpts) == 0:
        return None
    ckpts.sort(key=lambda x: int(re.findall(r"\d+", x)[0]))
    return os.path.join(model_dir, ckpts[-1])


def _save_latent_slices(latent: np.ndarray, out_path: str, title: str = ""):
    # latent: (C, D, H, W)
    c = 0
    vol = latent[c]
    D, H, W = vol.shape
    cz, cy, cx = D // 2, H // 2, W // 2
    slices = [vol[cz], vol[:, cy, :], vol[:, :, cx]]
    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    for i, ax in enumerate(axes):
        ax.imshow(slices[i], cmap="gray")
        ax.axis("off")
    fig.suptitle(title, fontsize=10)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close(fig)


def _save_voxel_slices(vol: np.ndarray, out_path: str, title: str = ""):
    # vol: (D,H,W)
    D, H, W = vol.shape
    cz, cy, cx = D // 2, H // 2, W // 2
    slices = [vol[cz], vol[:, cy, :], vol[:, :, cx]]
    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    for i, ax in enumerate(axes):
        ax.imshow(slices[i], cmap="gray")
        ax.axis("off")
    fig.suptitle(title, fontsize=10)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close(fig)


def _load_klvae(cfg_path: str, ckpt_path: str, device: torch.device):
    if not cfg_path or not ckpt_path:
        return None
    if not os.path.exists(cfg_path) or not os.path.exists(ckpt_path):
        print("⚠️ eval_decode_voxel enabled but VAE config/ckpt not found.")
        return None
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    vae = KLVAE3D(cfg).to(device)
    ckpt = _safe_torch_load(ckpt_path, map_location=device)
    if isinstance(ckpt, dict) and "vae_state_dict" in ckpt:
        state = ckpt["vae_state_dict"]
    elif isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
    else:
        state = ckpt
    new_state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
    vae.load_state_dict(new_state)
    vae.eval()
    return vae


def _latent_phi_proxy_loss(
    pred_center: torch.Tensor,
    gt_center: torch.Tensor,
    phi_local_target: torch.Tensor,
    ridge: float = 1e-4,
) -> torch.Tensor:
    """
    Lightweight porosity proxy loss in latent space.

    Fit a tiny linear probe (closed-form ridge regression) from GT latent features
    to local target phi in current batch, then enforce prediction features to match.
    This avoids VAE decode in training loop.
    """
    # Ensure linear algebra stays in fp32 and bypasses outer autocast.
    if pred_center.device.type == "cuda":
        amp_off = torch.amp.autocast("cuda", enabled=False)
    elif pred_center.device.type == "cpu":
        amp_off = torch.amp.autocast("cpu", enabled=False)
    else:
        amp_off = nullcontext()

    with amp_off:
        feat_gt = gt_center.mean(dim=(2, 3, 4)).detach().to(torch.float32)   # (B, C)
        feat_pred = pred_center.mean(dim=(2, 3, 4)).to(torch.float32)        # (B, C), keeps grad
        y = phi_local_target.view(-1, 1).detach().to(torch.float32)          # (B, 1)

        B = feat_gt.shape[0]
        ones = torch.ones((B, 1), device=feat_gt.device, dtype=torch.float32)
        X = torch.cat([feat_gt, ones], dim=1)                   # (B, C+1)
        X_pred = torch.cat([feat_pred, ones], dim=1)            # (B, C+1)

        with torch.no_grad():
            xtx = (X.T @ X).to(torch.float32)
            xty = (X.T @ y).to(torch.float32)
            d = xtx.shape[0]
            reg = torch.eye(d, device=xtx.device, dtype=torch.float32) * float(max(ridge, 0.0))
            A = xtx + reg
            try:
                w = torch.linalg.solve(A, xty)      # (C+1, 1)
            except (RuntimeError, NotImplementedError):
                w = torch.linalg.pinv(A) @ xty

        pred_phi_proxy = (X_pred @ w).squeeze(1).clamp(0.0, 1.0)
        target = phi_local_target.view(-1).to(torch.float32)
        return torch.abs(pred_phi_proxy - target).mean()


def _low_noise_snr_weights(ab_t_scalar: torch.Tensor, gamma: float) -> torch.Tensor:
    # ab_t_scalar: (N,), values in (0,1)
    denom = torch.clamp(1.0 - ab_t_scalar, min=1e-8)
    snr = ab_t_scalar / denom
    g = max(float(gamma), 1e-6)
    return torch.clamp(snr / g, max=1.0)


def run_eval_step(model, diffusion, device, step, exp_dir):
    eval_every = int(CONFIG.get("eval_every_steps", 0))
    if eval_every <= 0:
        return

    eval_dir = os.path.join(exp_dir, CONFIG.get("eval_output_dir", "eval"))
    os.makedirs(eval_dir, exist_ok=True)

    with _preserve_rng_state(device):
        # deterministic sample
        seed = int(CONFIG.get("eval_seed", 1234))
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)

        dataset_eval = PatchLatentDataset(CONFIG["latent_dir"], CONFIG["phi_map_dir"], augment=False)
        idx = int(CONFIG.get("eval_index", 0)) % len(dataset_eval)
        sample = dataset_eval[idx]

        x0 = sample["GT"].unsqueeze(0).to(device)
        cond = sample["Condition"].unsqueeze(0).to(device)
        mask = sample["Mask"].unsqueeze(0).to(device)
        phi = sample["Phi"].unsqueeze(0).to(device)
        por = sample["Porosity"].unsqueeze(0).to(device)

        model_was_train = model.training
        model.eval()
        with torch.no_grad():
            x_pred = ddim_sample(
                model, cond, mask, phi, por,
                diffusion,
                steps=int(CONFIG.get("eval_ddim_steps", 50)),
                seed=seed,
                safe_thresh=float(CONFIG.get("safe_threshold", 8.0)),
                cfg_scale=float(CONFIG.get("cfg_scale", 1.0)),
                context_cfg_scale=float(CONFIG.get("context_cfg_scale", 1.0)),
            )
        if model_was_train:
            model.train()

    # save latents
    step_tag = f"step{step:07d}"
    np.save(os.path.join(eval_dir, f"{step_tag}_gt.npy"), x0.cpu().float().numpy()[0])
    np.save(os.path.join(eval_dir, f"{step_tag}_pred.npy"), x_pred.cpu().float().numpy()[0])
    np.save(os.path.join(eval_dir, f"{step_tag}_cond.npy"), cond.cpu().float().numpy()[0])

    # quick slice png
    if bool(CONFIG.get("eval_save_png", True)):
        _save_latent_slices(x_pred.cpu().float().numpy()[0], os.path.join(eval_dir, f"{step_tag}_pred.png"), "pred")
        _save_latent_slices(x0.cpu().float().numpy()[0], os.path.join(eval_dir, f"{step_tag}_gt.png"), "gt")

    # optional voxel decode + visualization
    if bool(CONFIG.get("eval_decode_voxel", False)):
        vae_cfg = CONFIG.get("eval_vae_config_path", "")
        vae_ckpt = CONFIG.get("eval_vae_ckpt_path", "")
        vae = getattr(run_eval_step, "_vae_cache", None)
        if vae is None:
            vae = _load_klvae(vae_cfg, vae_ckpt, device)
            run_eval_step._vae_cache = vae

        if vae is not None:
            scale = float(CONFIG.get("scale_factor", 1.0))
            if scale == 0.0:
                scale = 1.0
            # unscale latents before decode
            z_pred = x_pred / scale
            z_gt = x0 / scale

            with torch.no_grad():
                if device.type == "cuda":
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        vox_pred = vae.decode(z_pred)
                        vox_gt = vae.decode(z_gt)
                else:
                    vox_pred = vae.decode(z_pred)
                    vox_gt = vae.decode(z_gt)

            vox_pred = vox_pred.cpu().float().numpy()[0, 0]
            vox_gt = vox_gt.cpu().float().numpy()[0, 0]

            np.save(os.path.join(eval_dir, f"{step_tag}_pred_voxel.npy"), vox_pred)
            np.save(os.path.join(eval_dir, f"{step_tag}_gt_voxel.npy"), vox_gt)

            if bool(CONFIG.get("eval_voxel_save_png", True)):
                _save_voxel_slices(vox_pred, os.path.join(eval_dir, f"{step_tag}_pred_voxel.png"), "pred_voxel")
                _save_voxel_slices(vox_gt, os.path.join(eval_dir, f"{step_tag}_gt_voxel.png"), "gt_voxel")


def main():
    """
    Stage07 主训练入口。

    主要流程：
    1) 构建数据集/采样器/DataLoader
    2) 构建模型、扩散器、优化器、EMA 与混合精度工具
    3) （可选）从最近 checkpoint 恢复训练状态
    4) 训练循环：扩散噪声监督 + x0 监督 + 统计约束 + phi 约束
    5) 记录日志、周期评估、保存 checkpoint
    """
    # ------------------ 初始化实验与设备 ------------------
    device = torch.device(CONFIG["device"])
    exp_dir, model_dir, log_dir, csv_path, csv_detail_path = setup_experiment()

    # ------------------ 构建数据加载 ------------------
    dataset = PatchLatentDataset(CONFIG["latent_dir"], CONFIG["phi_map_dir"], augment=True)
    # 默认 DataLoader 行为：均匀随机打乱样本（每个样本被抽到的概率相同）
    sampler = None
    shuffle = True
    if bool(CONFIG.get("use_porosity_weighted_sampler", False)):
        # 按样本级孔隙率统计做“逆频率重采样”，让稀有孔隙率区间被更频繁抽到
        bin_edges = list(CONFIG.get("porosity_bin_edges", [0.0, 0.75, 0.9, 0.96, 0.985, 1.01]))
        power = float(CONFIG.get("porosity_sampler_power", 1.0))
        min_w = float(CONFIG.get("porosity_sampler_min_weight", 0.2))
        max_w = float(CONFIG.get("porosity_sampler_max_weight", 5.0))
        sample_w = dataset.build_porosity_sampling_weights(
            bin_edges=bin_edges,
            power=power,
            min_weight=min_w,
            max_weight=max_w,
        )
        sampler = WeightedRandomSampler(
            weights=torch.from_numpy(sample_w).double(),
            num_samples=len(sample_w),
            # 有放回采样：同一 epoch 内样本可重复出现，便于放大稀有分箱
            replacement=True,
        )
        # 使用 sampler 时必须关闭 shuffle（PyTorch 不允许同时指定）
        shuffle = False
        stats = getattr(dataset, "_last_sampler_stats", {})
        if stats:
            # 打印统计信息，便于确认重采样是否按预期生效
            print(f"[sampler] semantic={stats.get('semantic', 'pore')}")
            print(f"[sampler] bin_edges={stats.get('bin_edges')}")
            print(f"[sampler] bin_counts={stats.get('bin_counts')}")
            print(
                "[sampler] value range="
                f"[{stats.get('value_min', 0.0):.4f}, {stats.get('value_max', 0.0):.4f}], "
                f"mean={stats.get('value_mean', 0.0):.4f}"
            )
            print(
                "[sampler] weight range="
                f"[{stats.get('weight_min', 0.0):.4f}, {stats.get('weight_max', 0.0):.4f}], "
                f"mean={stats.get('weight_mean', 0.0):.4f}"
            )

    dataloader = DataLoader(
        dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=shuffle,
        sampler=sampler,
        num_workers=CONFIG["num_workers"],
        pin_memory=CONFIG.get("pin_memory", True),
    )

    # ------------------ 构建模型与优化组件 ------------------
    model = ConditionalLatentUNet(
        in_channels=_resolve_in_channels(),
        out_channels=CONFIG["out_channels"],
        base_channels=CONFIG["base_channels"],
        channel_mults=CONFIG["channel_mults"],
        use_attention=CONFIG["use_attention"],
        use_adagn=bool(CONFIG.get("use_adagn", False)),
        cfg_drop_prob=float(CONFIG.get("cfg_drop_prob", 0.0)),
    ).to(device)

    ema = EMA(model, decay=float(CONFIG.get("ema_decay", 0.9999))).to(device)
    optimizer = AdamW(model.parameters(), lr=CONFIG["lr"])
    scheduler = CosineAnnealingLR(optimizer, T_max=CONFIG["epochs"], eta_min=1e-6)
    diffusion = DiffusionHelper(CONFIG["timesteps"], device)
    scaler = GradScaler("cuda" if device.type == "cuda" else "cpu")

    # ------------------ 断点恢复（可选） ------------------
    start_epoch = 0
    global_step = 0
    if CONFIG.get("resume", True):
        latest = load_latest_checkpoint(model_dir, device)
        if latest:
            print(f"🔄 Loading checkpoint: {latest}")
            ckpt = _safe_torch_load(latest, map_location=device)
            model.load_state_dict(ckpt["model_state_dict"])
            if "ema_state_dict" in ckpt:
                ema.load_state_dict(ckpt["ema_state_dict"])
            if bool(CONFIG.get("resume_load_optimizer", True)) and "optimizer_state_dict" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            else:
                print("ℹ️ Resume without optimizer state (fresh optimizer).")
            if bool(CONFIG.get("resume_load_scheduler", True)) and "scheduler_state_dict" in ckpt:
                scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            else:
                print("ℹ️ Resume without scheduler state (fresh scheduler).")
            start_epoch = int(ckpt["epoch"])
            global_step = int(ckpt.get("global_step", 0))

    print(f"🚀 Start training at epoch {start_epoch}, step {global_step}")

    # ------------------ 读取训练策略超参数 ------------------
    # 这部分把 CONFIG 中的损失项和权重读出来，便于后续组合总损失。
    loss_type = CONFIG.get("loss_type", "l1").lower()
    use_min_snr = bool(CONFIG.get("use_min_snr", True))
    gamma = float(CONFIG.get("min_snr_gamma", 5.0))
    x0_w = float(CONFIG.get("x0_weight", 0.2))
    use_target_stats_loss = bool(CONFIG.get("use_target_stats_loss", False))
    target_stats_weight = float(CONFIG.get("target_stats_weight", 0.0))
    use_phi_consistency_loss = bool(CONFIG.get("use_phi_consistency_loss", False))
    phi_consistency_weight = float(CONFIG.get("phi_consistency_weight", 0.0))
    phi_loss_every_steps = max(1, int(CONFIG.get("phi_loss_every_steps", 1)))
    phi_loss_max_batch = int(CONFIG.get("phi_loss_max_batch", 0))
    phi_loss_t_min_ratio = float(CONFIG.get("phi_loss_t_min_ratio", 0.0))
    phi_loss_t_max_ratio = float(CONFIG.get("phi_loss_t_max_ratio", -1.0))
    phi_loss_use_low_noise_snr_weight = bool(CONFIG.get("phi_loss_use_low_noise_snr_weight", False))
    phi_loss_snr_gamma = float(CONFIG.get("phi_loss_snr_gamma", gamma))
    use_phi_proxy_loss = bool(CONFIG.get("use_phi_proxy_loss", False))
    phi_proxy_weight = float(CONFIG.get("phi_proxy_weight", 0.0))
    phi_proxy_ridge = float(CONFIG.get("phi_proxy_ridge", 1e-4))
    phi_proxy_use_phi_t_filter = bool(CONFIG.get("phi_proxy_use_phi_t_filter", True))
    phi_proxy_use_low_noise_snr_weight = bool(CONFIG.get("phi_proxy_use_low_noise_snr_weight", False))
    band_w = int(CONFIG.get("boundary_band_width", 0))
    band_weight = float(CONFIG.get("boundary_band_weight", 0.0))
    safe_thresh = float(CONFIG.get("safe_threshold", 8.0))
    patch_size = int(CONFIG.get("patch_size", 8))
    window_size = int(CONFIG.get("window_size", 3))
    center_start = (window_size // 2) * patch_size
    center_end = center_start + patch_size

    # phi 辅助损失只在低噪声时间步段启用（可通过 ratio 控制）
    # 直观上：低噪声步的 x0 估计更可信，更适合做 phi 一致性约束。
    t_total = int(CONFIG["timesteps"])
    phi_t_min = int(round(max(0.0, phi_loss_t_min_ratio) * max(t_total - 1, 1)))
    if phi_loss_t_max_ratio < 0.0:
        phi_t_max = t_total - 1
    else:
        phi_t_max = int(round(min(1.0, max(0.0, phi_loss_t_max_ratio)) * max(t_total - 1, 1)))
    phi_t_min = int(np.clip(phi_t_min, 0, max(t_total - 1, 0)))
    phi_t_max = int(np.clip(phi_t_max, 0, max(t_total - 1, 0)))
    if phi_t_max < phi_t_min:
        phi_t_max = phi_t_min

    print(
        "[phi] settings: "
        f"consistency_weight={phi_consistency_weight}, "            
        f"proxy_weight={phi_proxy_weight}, "
        f"loss_every_steps={phi_loss_every_steps}, "
        f"loss_max_batch={phi_loss_max_batch}, "
        f"t_range=[{phi_t_min},{phi_t_max}], "
        f"decode_snr_weight={phi_loss_use_low_noise_snr_weight}, "
        f"proxy_t_filter={phi_proxy_use_phi_t_filter}, "
        f"proxy_snr_weight={phi_proxy_use_low_noise_snr_weight}"
    )

    # ------------------ 载入 VAE（仅用于 decode 式 phi 损失） ------------------
    # 注意：这里只作为固定评估器使用，不参与参数更新。
    vae_phi = None
    if use_phi_consistency_loss and phi_consistency_weight > 0.0:
        vae_cfg = CONFIG.get("eval_vae_config_path", "")
        vae_ckpt = CONFIG.get("eval_vae_ckpt_path", "")
        vae_phi = _load_klvae(vae_cfg, vae_ckpt, device)
        if vae_phi is None:
            print("⚠️ phi consistency loss requested but VAE is unavailable; disabling.")
            use_phi_consistency_loss = False
            phi_consistency_weight = 0.0
        else:
            for p in vae_phi.parameters():
                p.requires_grad = False

    for epoch in range(start_epoch, CONFIG["epochs"]):
        # ------------------ 单个 epoch 训练 ------------------
        model.train()
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']}")

        for batch in pbar:
            global_step += 1
            # 批数据：GT 为完整窗口真值；Condition/Mask 给出可见上下文；
            # TargetMask 指定仅中心 patch 参与主要监督。
            x0 = batch["GT"].to(device, non_blocking=True)           # (B,C,D,H,W)
            cond = batch["Condition"].to(device, non_blocking=True)  # (B,C,D,H,W)
            mask = batch["Mask"].to(device, non_blocking=True)       # (B,1,D,H,W)
            tmask = batch["TargetMask"].to(device, non_blocking=True) # (B,1,D,H,W)
            phi = batch["Phi"].to(device, non_blocking=True)         # (B,1,D,H,W)
            por = batch["Porosity"].to(device, non_blocking=True)    # (B,1)

            B = x0.shape[0]
            # 扩散时间步按样本独立随机采样
            t = torch.randint(0, CONFIG["timesteps"], (B,), device=device).long()

            with autocast("cuda" if device.type == "cuda" else "cpu"):
                # q(x_t | x0): 将 x0 加噪到时间步 t
                ab_t = diffusion.alphas_cumprod[t].view(-1, 1, 1, 1, 1)
                sqrt_ab = torch.sqrt(ab_t)
                sqrt_om = torch.sqrt(1.0 - ab_t)

                noise = torch.randn_like(x0)
                x_t = sqrt_ab * x0 + sqrt_om * noise

                # 已知上下文区域强制与 cond 保持一致（使用同一份 noise 保证统计一致）
                known_xt = sqrt_ab * cond + sqrt_om * noise
                x_t = x_t * (1.0 - mask) + known_xt * mask

                # 模型输入由 [当前噪声态, 条件 latent, 上下文掩码, phi 体条件] 组成
                model_in = torch.cat([x_t, cond, mask, phi], dim=1)
                eps_pred = model(model_in, t, por)

                # ------------------ 主损失 1：噪声预测损失（仅中心目标 patch） ------------------
                if loss_type == "mse":
                    raw = (eps_pred - noise) ** 2
                else:
                    raw = torch.abs(eps_pred - noise)

                if use_min_snr:
                    snr = ab_t / (1.0 - ab_t)
                    w = torch.minimum(snr, torch.tensor(gamma, device=device)) / snr
                    raw = raw * w

                C = eps_pred.shape[1]
                tmask_b = tmask.expand(-1, C, -1, -1, -1)
                # boundary-weighted target loss (emphasize seams)
                if band_w > 0 and band_weight > 0:
                    # erosion: inner = 1 - maxpool(1 - tmask)
                    inner = 1.0 - F.max_pool3d(
                        1.0 - tmask, kernel_size=2 * band_w + 1, stride=1, padding=band_w
                    )
                    inner = inner.clamp(0.0, 1.0)
                    boundary = (tmask - inner).clamp(0.0, 1.0)
                    weight = tmask + boundary * band_weight
                else:
                    weight = tmask

                weight_b = weight.expand(-1, C, -1, -1, -1)
                loss_diff = (raw * weight_b).sum() / weight_b.sum().clamp_min(1.0)

                # ------------------ 主损失 2：x0 重建损失（仅中心目标 patch） ------------------
                pred_x0 = (x_t - sqrt_om * eps_pred) / (sqrt_ab + 1e-8)
                pred_x0 = torch.clamp(pred_x0, -safe_thresh, safe_thresh)
                # 与主损失保持一致：MSE 时用平方，L1 时用绝对值
                if loss_type == "mse":
                    x0_raw = (pred_x0 - x0) ** 2
                else:
                    x0_raw = torch.abs(pred_x0 - x0)
                loss_x0 = (x0_raw * weight_b).sum() / weight_b.sum().clamp_min(1.0)

                # ------------------ 可选损失：目标区域统计量匹配 ------------------
                if use_target_stats_loss and target_stats_weight > 0.0:
                    # Match mean/std inside target region to discourage collapsed predictions.
                    stat_w = tmask_b
                    denom = stat_w.sum(dim=(2, 3, 4), keepdim=True).clamp_min(1.0)
                    pred_mean = (pred_x0 * stat_w).sum(dim=(2, 3, 4), keepdim=True) / denom
                    gt_mean = (x0 * stat_w).sum(dim=(2, 3, 4), keepdim=True) / denom
                    pred_var = ((pred_x0 - pred_mean) ** 2 * stat_w).sum(dim=(2, 3, 4), keepdim=True) / denom
                    gt_var = ((x0 - gt_mean) ** 2 * stat_w).sum(dim=(2, 3, 4), keepdim=True) / denom
                    pred_std = torch.sqrt(pred_var + 1e-8)
                    gt_std = torch.sqrt(gt_var + 1e-8)
                    loss_stats = torch.abs(pred_mean - gt_mean).mean() + torch.abs(pred_std - gt_std).mean()
                else:
                    loss_stats = torch.zeros((), device=device, dtype=loss_x0.dtype)

                pred_center_all = pred_x0[
                    :,
                    :,
                    center_start:center_end,
                    center_start:center_end,
                    center_start:center_end,
                ]
                gt_center_all = x0[
                    :,
                    :,
                    center_start:center_end,
                    center_start:center_end,
                    center_start:center_end,
                ]
                phi_local_target_all = phi[
                    :,
                    0:1,
                    center_start:center_end,
                    center_start:center_end,
                    center_start:center_end,
                ].mean(dim=(1, 2, 3, 4))
                phi_target_mean = float(phi_local_target_all.detach().mean().item())
                phi_t_valid_mask_all = (t >= phi_t_min) & (t <= phi_t_max)

                # 仅在指定低噪声 t 区间计算 phi 相关损失，并可限制子批大小降低显存/耗时
                idx_valid_phi = torch.where(phi_t_valid_mask_all)[0]
                if phi_loss_max_batch > 0 and idx_valid_phi.numel() > phi_loss_max_batch:
                    perm = torch.randperm(idx_valid_phi.numel(), device=device)[:phi_loss_max_batch]
                    idx_phi = idx_valid_phi[perm]
                else:
                    idx_phi = idx_valid_phi
                if idx_phi.numel() > 0:
                    pred_center_phi = pred_center_all[idx_phi]
                    phi_local_target_phi = phi_local_target_all[idx_phi]
                else:
                    pred_center_phi = pred_center_all[:0]
                    phi_local_target_phi = phi_local_target_all[:0]

                do_decode_phi = (
                    use_phi_consistency_loss
                    and phi_consistency_weight > 0.0
                    and vae_phi is not None
                    and (global_step % phi_loss_every_steps == 0)
                    and idx_phi.numel() > 0
                )
                if do_decode_phi:
                    # ------------------ 可选损失：decode 式 phi 一致性 ------------------
                    # 用冻结的 VAE 解码中心 patch，约束其局部 phi 与目标 phi 一致。
                    if device.type == "cuda":
                        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                            pred_logits = vae_phi.decode(pred_center_phi)
                    else:
                        pred_logits = vae_phi.decode(pred_center_phi)
                    pred_prob = torch.sigmoid(pred_logits)
                    pred_phi_local = pred_prob.mean(dim=(1, 2, 3, 4))
                    phi_abs_err = torch.abs(pred_phi_local - phi_local_target_phi)
                    if phi_loss_use_low_noise_snr_weight:
                        ab_phi = ab_t[idx_phi, 0, 0, 0, 0].to(torch.float32)
                        w_phi = _low_noise_snr_weights(ab_phi, phi_loss_snr_gamma).to(phi_abs_err.dtype)
                        loss_phi = (phi_abs_err * w_phi).sum() / w_phi.sum().clamp_min(1e-8)
                    else:
                        loss_phi = phi_abs_err.mean()
                else:
                    loss_phi = torch.zeros((), device=device, dtype=loss_x0.dtype)

                # ------------------ 可选损失：latent 代理 phi 损失（不走解码） ------------------
                # 用轻量线性探针在 latent 空间近似约束 porosity，计算开销更小。
                if use_phi_proxy_loss and phi_proxy_weight > 0.0:
                    if phi_proxy_use_phi_t_filter:
                        idx_proxy = torch.where(phi_t_valid_mask_all)[0]
                    else:
                        idx_proxy = torch.arange(B, device=device)
                    if idx_proxy.numel() > 0:
                        loss_phi_proxy = _latent_phi_proxy_loss(
                            pred_center=pred_center_all[idx_proxy],
                            gt_center=gt_center_all[idx_proxy],
                            phi_local_target=phi_local_target_all[idx_proxy],
                            ridge=phi_proxy_ridge,
                        ).to(dtype=loss_x0.dtype)
                        if phi_proxy_use_low_noise_snr_weight:
                            ab_proxy = ab_t[idx_proxy, 0, 0, 0, 0].to(torch.float32)
                            w_proxy = _low_noise_snr_weights(ab_proxy, phi_loss_snr_gamma).to(loss_phi_proxy.dtype)
                            loss_phi_proxy = loss_phi_proxy * w_proxy.mean()
                    else:
                        loss_phi_proxy = torch.zeros((), device=device, dtype=loss_x0.dtype)
                else:
                    loss_phi_proxy = torch.zeros((), device=device, dtype=loss_x0.dtype)

                # 总损失 = 主损失 + 各辅助项（由权重控制）
                loss = (
                    loss_diff
                    + x0_w * loss_x0
                    + target_stats_weight * loss_stats
                    + phi_consistency_weight * loss_phi
                    + phi_proxy_weight * loss_phi_proxy
                )

            # ------------------ 反向传播与参数更新 ------------------
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            grad_clip_norm = float(CONFIG.get("grad_clip_norm", 0.0))
            if grad_clip_norm > 0.0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()

            # EMA 维护一份平滑权重，便于更稳定的评估/推理
            ema.update(model)

            postfix = {"loss": f"{loss.item():.4f}", "diff": f"{loss_diff.item():.4f}"}
            if use_target_stats_loss and target_stats_weight > 0.0:
                postfix["stats"] = f"{loss_stats.item():.4f}"
            if use_phi_consistency_loss and phi_consistency_weight > 0.0:
                postfix["phi"] = f"{loss_phi.item():.4f}"
            if use_phi_proxy_loss and phi_proxy_weight > 0.0:
                postfix["phi_proxy"] = f"{loss_phi_proxy.item():.4f}"
            pbar.set_postfix(postfix)

            # ------------------ 日志记录 ------------------
            if global_step % CONFIG.get("save_log_every", 1) == 0:
                with open(csv_path, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        epoch, global_step, f"{loss.item():.6f}",
                        optimizer.param_groups[0]["lr"],
                        f"{loss_diff.item():.6f}",
                        f"{loss_x0.item():.6f}",
                    ])
                with open(csv_detail_path, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(
                        [
                            epoch,
                            global_step,
                            f"{loss.item():.6f}",
                            optimizer.param_groups[0]["lr"],
                            f"{loss_diff.item():.6f}",
                            f"{loss_x0.item():.6f}",
                            f"{loss_stats.item():.6f}",
                            f"{loss_phi.item():.6f}",
                            f"{loss_phi_proxy.item():.6f}",
                            f"{(phi_consistency_weight * loss_phi).item():.6f}",
                            f"{(phi_proxy_weight * loss_phi_proxy).item():.6f}",
                            int(do_decode_phi),
                            f"{phi_consistency_weight:.6f}",
                            f"{phi_proxy_weight:.6f}",
                            int(phi_loss_every_steps),
                            int(phi_loss_max_batch),
                            f"{phi_target_mean:.6f}",
                        ]
                    )

            # ------------------ 训练中评估（可选） ------------------
            if int(CONFIG.get("eval_every_steps", 0)) > 0 and (global_step % int(CONFIG.get("eval_every_steps", 0)) == 0):
                use_ema_eval = bool(CONFIG.get("eval_use_ema", False))
                if use_ema_eval and hasattr(ema, "ema_model"):
                    eval_model = ema.ema_model
                else:
                    eval_model = model
                run_eval_step(eval_model, diffusion, device, global_step, exp_dir)

        # epoch 结束后更新学习率调度器
        scheduler.step()

        # ------------------ 周期保存 checkpoint ------------------
        if (epoch + 1) % CONFIG["save_model_every"] == 0:
            ckpt = {
                "epoch": epoch + 1,
                "global_step": global_step,
                "model_state_dict": model.state_dict(),
                "ema_state_dict": ema.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
            }
            torch.save(ckpt, os.path.join(model_dir, f"unet_epoch_{epoch+1}.pth"))
            torch.save(ckpt, os.path.join(model_dir, "unet_latest.pth"))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import yaml
import shutil
from tqdm import tqdm
import argparse
import time
import glob
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam
from torchvision.utils import save_image

from models.vae import KLVAE3D
from models.discriminator import NLayerDiscriminator3D
from data.dataset import CubeDataset
from utils.logger import CSVLogger  # 假设 CSVLogger 能接受任意字段写入

# A100 TF32
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def load_config(path: str):
    """加载 YAML 配置文件"""
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def hinge_d_loss(logits_real, logits_fake):
    """判别器 Hinge Loss"""
    return 0.5 * (torch.mean(F.relu(1.0 - logits_real)) + torch.mean(F.relu(1.0 + logits_fake)))


def fix_compile_state_dict(state_dict):
    """兼容 torch.compile 导致的 _orig_mod. 前缀"""
    new_state = {}
    for k, v in state_dict.items():
        if isinstance(k, str) and k.startswith("_orig_mod."):
            new_state[k[10:]] = v
        else:
            new_state[k] = v
    return new_state


def calc_latent_volume(cfg):
    """
    根据 config 的 image_size 与 ch_mult 计算 latent 空间体素数 (D*H*W)
    注意：本项目 Encoder 的下采样次数 = len(ch_mult) - 1（最后一层不再 stride=2）
    """
    image_size = cfg['data'].get('croped_size', cfg['data']['image_size'])
    ch_mult_len = len(cfg['model']['ch_mult'])
    downsample = 2 ** max(ch_mult_len - 1, 0)

    if image_size % downsample != 0:
        raise ValueError(f"image_size {image_size} 不能被下采样因子 {downsample} 整除")

    latent_edge = image_size // downsample
    return latent_edge ** 3


def human_time():
    """返回当前时间字符串"""
    return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())


def soft_dice_loss(prob, target, eps=1e-6):
    """
    Soft Dice Loss（适合二值结构/类别不平衡）
    prob/target: [B,1,D,H,W]，范围 [0,1]
    """
    dims = (1, 2, 3, 4)
    inter = torch.sum(prob * target, dim=dims)
    union = torch.sum(prob + target, dim=dims)
    dice = 1.0 - (2.0 * inter + eps) / (union + eps)
    return dice.mean()


def _center_crop_3d(vol: np.ndarray, target: int) -> np.ndarray:
    """把 3D 体 vol 中心裁剪为 [target,target,target]"""
    D, H, W = vol.shape
    if D < target or H < target or W < target:
        raise ValueError(f"体积太小：{vol.shape}，无法裁到 {target}^3")
    d0 = (D - target) // 2
    h0 = (H - target) // 2
    w0 = (W - target) // 2
    return vol[d0:d0 + target, h0:h0 + target, w0:w0 + target]


def _normalize_to_minus1_1(data_np: np.ndarray) -> np.ndarray:
    """
    把输入 npy 归一化到 [-1,1]，尽量兼容 0/1、0/255、0/65535 等情况
    """
    data_np = data_np.astype(np.float32)
    mn, mx = float(data_np.min()), float(data_np.max())

    # 已经是 [-1,1] 或 [0,1]
    if mn >= -1.01 and mx <= 1.01:
        if mn >= 0.0 and mx <= 1.01:
            data_np = data_np * 2.0 - 1.0
        return data_np

    # 常见整型范围
    if mx <= 1.5:
        data_np = data_np * 2.0 - 1.0
    elif mx <= 255.5:
        data_np = (data_np / 255.0) * 2.0 - 1.0
    else:
        data_np = (data_np / 65535.0) * 2.0 - 1.0
    return data_np


@torch.no_grad()
def run_periodic_inference(
    cfg: dict,
    vae: KLVAE3D,
    device: torch.device,
    exp_dir: str,
    global_step: int
):
    """
    每隔固定 step 做一次推理并保存对比图：
    - 上排：Real 的 XY/XZ/YZ 中间切片
    - 下排：Recon（概率图或二值图） 的 XY/XZ/YZ 中间切片
    """
    monitor_cfg = cfg.get("monitor", {})
    if not monitor_cfg.get("enable", False):
        return

    interval = int(monitor_cfg.get("interval_steps", 0))
    if interval <= 0:
        return

    # step=0 也可以推一次（可视化 sanity）
    if global_step % interval != 0:
        return

    # 1) 选择样本 npy
    sample_path = str(monitor_cfg.get("sample_npy_path", "") or "").strip()
    if sample_path and os.path.exists(sample_path):
        target_file = sample_path
    else:
        data_root = cfg['data']['data_root']
        files = glob.glob(os.path.join(data_root, f"*{cfg['data']['file_extension']}"))
        if len(files) == 0:
            print(f"[{human_time()}] 监控推理：data_root 下没有找到样本文件，跳过。")
            return
        target_file = files[0]
        if sample_path:
            print(f"[{human_time()}] 监控推理：sample_npy_path 不存在，自动选取 {target_file}")

    # 2) 加载并归一化到 [-1,1]
    data_np = np.load(target_file)
    data_np = _normalize_to_minus1_1(data_np)

    # 3) 推理输入大小（允许与训练不同）
    # sample_size：最终输入边长；crop_size：如果原始更大，中心裁剪到 crop_size（一般等于 sample_size）
    sample_size = int(monitor_cfg.get("sample_size", cfg['data'].get("croped_size", cfg['data']['image_size'])))
    crop_size = int(monitor_cfg.get("crop_size", sample_size))

    vol = data_np
    if vol.ndim == 4:
        # 有些数据可能是 [C,D,H,W]，我们取第 0 通道
        vol = vol[0]
    if vol.ndim != 3:
        print(f"[{human_time()}] 监控推理：样本维度不为 3D，shape={data_np.shape}，跳过。")
        return

    # 先裁到 crop_size，再确保等于 sample_size（一般两者相同）
    if vol.shape != (crop_size, crop_size, crop_size):
        vol = _center_crop_3d(vol, crop_size)

    if crop_size != sample_size:
        # 这里不做 resize（会引入插值灰度），只允许 crop_size>=sample_size 再裁一次
        if crop_size < sample_size:
            print(f"[{human_time()}] 监控推理：crop_size<{sample_size}，无法得到 sample_size，跳过。")
            return
        if vol.shape != (sample_size, sample_size, sample_size):
            vol = _center_crop_3d(vol, sample_size)

    # 4) 推理
    inp = torch.from_numpy(vol).unsqueeze(0).unsqueeze(0).to(device)  # [1,1,D,H,W]
    vae_was_training = vae.training
    vae.eval()

    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        recon_logits, _ = vae(inp, sample_posterior=False)  # 推理用 posterior mean 更稳定

    # 5) 可视化：real01 + recon01(prob 或 binary)
    show_binary = bool(monitor_cfg.get("show_binary", False))

    real = inp[0, 0].float().cpu()  # [-1,1]
    real01 = (real.clamp(-1, 1) + 1.0) / 2.0  # [0,1]

    recon_logits_ = recon_logits[0, 0].float().cpu()
    recon_prob = torch.sigmoid(recon_logits_)  # [0,1]
    recon01 = (recon_prob > 0.5).float() if show_binary else recon_prob

    D, H, W = real01.shape
    # 三个视角中间切片
    slice_xy_real = real01[D // 2, :, :]
    slice_xy_rec = recon01[D // 2, :, :]

    slice_xz_real = real01[:, H // 2, :]
    slice_xz_rec = recon01[:, H // 2, :]

    slice_yz_real = real01[:, :, W // 2]
    slice_yz_rec = recon01[:, :, W // 2]

    row1 = torch.cat([slice_xy_real, slice_xz_real, slice_yz_real], dim=1)
    row2 = torch.cat([slice_xy_rec, slice_xz_rec, slice_yz_rec], dim=1)
    grid = torch.cat([row1, row2], dim=0).unsqueeze(0)  # [1, H*2, W*3]

    # 6) 保存
    out_dir = os.path.join(exp_dir, "monitor_infer")
    os.makedirs(out_dir, exist_ok=True)

    mode = "binary" if show_binary else "prob"
    base = os.path.splitext(os.path.basename(target_file))[0]
    save_path = os.path.join(out_dir, f"step_{global_step:08d}_{mode}_{sample_size}_{base}.png")
    save_image(grid, save_path)
    print(f"[{human_time()}] 监控推理已保存：{save_path}")

    # 复原训练状态
    if vae_was_training:
        vae.train()


def main():
    # 1) 解析命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/train_config.yaml')
    parser.add_argument('--resume', type=str, default=None, help='checkpoint 路径，用于断点继续训练')
    parser.add_argument('--restart', action='store_true', help='忽略 resume，从头重新训练')
    parser.add_argument('--reset-kl', action='store_true', help='resume 时强制重置 KL warmup（从 0 重新 warmup）')
    args = parser.parse_args()

    # 2) 加载配置，准备实验目录与日志
    cfg = load_config(args.config)
    exp_dir = os.path.join(cfg['experiment']['save_dir'], cfg['experiment']['exp_name'])
    os.makedirs(exp_dir, exist_ok=True)

    if args.resume is None or args.restart:
        shutil.copy(args.config, os.path.join(exp_dir, 'config.yaml'))

    logger = CSVLogger(exp_dir)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[{human_time()}] 设备：{device}")

    # 3) 数据集与 DataLoader
    dataset = CubeDataset(
        cfg['data']['data_root'],
        cfg['data']['file_extension'],
        crop_size=cfg['data']['croped_size'],
        is_train=True
    )
    dataloader = DataLoader(
        dataset,
        batch_size=cfg['train']['batch_size'],
        shuffle=True,
        num_workers=cfg['data']['num_workers'],
        pin_memory=True,
        persistent_workers=False
    )

    # 4) 初始化模型：VAE + 判别器
    vae = KLVAE3D(cfg).to(device)
    discriminator = NLayerDiscriminator3D().to(device)

    # 5) 初始化优化器
    opt_vae = Adam(vae.parameters(), lr=float(cfg['train']['lr']), betas=(0.5, 0.9))
    opt_disc = Adam(discriminator.parameters(), lr=float(cfg['train']['lr']), betas=(0.5, 0.9))

    # 训练状态
    global_step = 0
    start_epoch = 0

    # KL/GAN 参数
    kl_weight_max = float(cfg['loss']['kl_weight_max'])
    disc_start = int(cfg['loss']['disc_start'])
    disc_weight = float(cfg['loss']['disc_weight'])
    kl_warmup_steps = int(cfg['train'].get('kl_warmup_steps', 20000))

    # Dice 权重（默认 0.5，也可在 config.loss.dice_weight 配）
    dice_weight = float(cfg.get('loss', {}).get('dice_weight', 0.5))

    # latent 信息（用于日志指标）
    latent_volume = calc_latent_volume(cfg)
    z_ch = int(cfg['model']['z_channels'])

    # KL warmup 起点（用于无缝 resume）
    kl_start_step = 0

    # 断点继续训练（兼容 _orig_mod）
    if args.resume is not None and not args.restart:
        if os.path.isfile(args.resume):
            print(f"[{human_time()}] 加载 checkpoint：{args.resume}")
            ckpt = torch.load(args.resume, map_location=device)

            try:
                vae.load_state_dict(fix_compile_state_dict(ckpt['vae_state_dict']))
            except Exception:
                vae.load_state_dict(ckpt['vae_state_dict'])

            try:
                discriminator.load_state_dict(fix_compile_state_dict(ckpt['disc_state_dict']))
            except Exception:
                discriminator.load_state_dict(ckpt['disc_state_dict'])

            try:
                opt_vae.load_state_dict(ckpt['optimizer_vae'])
            except Exception:
                print("警告：optimizer_vae 加载失败，将使用新初始化的优化器状态")

            if 'optimizer_disc' in ckpt:
                try:
                    opt_disc.load_state_dict(ckpt['optimizer_disc'])
                except Exception:
                    print("警告：optimizer_disc 加载失败，将使用新初始化的优化器状态")
            else:
                print("提示：checkpoint 不包含 optimizer_disc，判别器优化器将从头开始")

            start_epoch = int(ckpt.get('epoch', 0)) + 1
            global_step = int(ckpt.get('global_step', 0))

            if args.reset_kl:
                kl_start_step = global_step
                print(f"[{human_time()}] 按用户要求：KL warmup 重置（kl_start_step={kl_start_step}）")
            else:
                ckpt_kl_start = ckpt.get('kl_start_step', None)
                kl_start_step = int(ckpt_kl_start) if ckpt_kl_start is not None else 0

            print(f"[{human_time()}] Resume：start_epoch={start_epoch}, global_step={global_step}, kl_start_step={kl_start_step}")
        else:
            print(f"[{human_time()}] 警告：resume 文件不存在：{args.resume}，将从头训练")
            kl_start_step = 0
    else:
        kl_start_step = 0
        if args.restart and args.resume is not None:
            print(f"[{human_time()}] restart 已指定：忽略 resume={args.resume}，从头训练")

    from collections import deque
    kl_history = deque(maxlen=5)

    # 训练主循环
    for epoch in range(start_epoch, cfg['train']['epochs']):
        vae.train()
        discriminator.train()

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{cfg['train']['epochs']}", ncols=120)

        for batch_idx, images in enumerate(pbar):
            images = images.to(device, non_blocking=True)  # [B,1,D,H,W]，范围 [-1,1]

            # KL warmup：从 kl_start_step 开始计数
            effective_step = max(0, global_step - kl_start_step)
            kl_weight = min(kl_weight_max, kl_weight_max * (effective_step / max(1, kl_warmup_steps)))

            if global_step == disc_start:
                print(f"[{human_time()}] 判别器启动阈值到达：step={global_step}")

            # =========================
            # 训练 VAE（生成器）
            # =========================
            opt_vae.zero_grad()
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                recon_logits, posterior = vae(images)

                # [-1,1] -> [0,1] 目标：孔隙(-1)->0，骨架(+1)->1
                target = (images + 1.0) / 2.0

                # pos_weight：抵抗类别不平衡（骨架比例通常很高/或很低）
                p = target.mean().clamp(1e-4, 1 - 1e-4)
                pos_weight = ((1 - p) / p).detach()

                # 1) BCEWithLogits
                bce_loss = F.binary_cross_entropy_with_logits(
                    recon_logits, target, pos_weight=pos_weight
                )

                # 2) Soft Dice
                prob = torch.sigmoid(recon_logits)
                dice_loss = soft_dice_loss(prob, target)

                rec_loss = bce_loss + dice_weight * dice_loss

                # KL
                kl_loss = torch.mean(posterior.kl())

                # GAN：给判别器的 fake 输入使用 tanh(logits)，梯度更顺、边界更容易变硬
                recon_img = torch.tanh(recon_logits)  # [-1,1]

                if global_step > disc_start:
                    logits_fake_for_g = discriminator(recon_img)
                    g_adv_loss = -torch.mean(logits_fake_for_g)
                else:
                    g_adv_loss = torch.tensor(0.0, device=device)

                total_loss = rec_loss + kl_weight * kl_loss + disc_weight * g_adv_loss

            total_loss.backward()
            grad_norm_vae = torch.nn.utils.clip_grad_norm_(vae.parameters(), max_norm=1.0)
            opt_vae.step()

            # =========================
            # 训练判别器 D
            # =========================
            loss_d = torch.tensor(0.0, device=device)
            grad_norm_disc = 0.0
            if global_step > disc_start:
                opt_disc.zero_grad()
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    logits_real = discriminator(images)               # real：[-1,1]
                    logits_fake = discriminator(recon_img.detach())   # fake：[-1,1]
                    loss_d = hinge_d_loss(logits_real, logits_fake)

                loss_d.backward()
                grad_norm_disc = torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)
                opt_disc.step()

            # =========================
            # 周期性推理可视化（按 step）
            # =========================
            run_periodic_inference(cfg, vae, device, exp_dir, global_step)

            # =========================
            # 日志与监控
            # =========================
            kl_raw = float(kl_loss.item())
            avg_kl_per_latent = kl_raw / (z_ch * latent_volume)
            kl_contrib = kl_weight * kl_raw
            kl_contrib_ratio = (kl_contrib / (rec_loss.item() + 1e-12))

            kl_history.append(kl_raw)
            kl_trend = ""
            if len(kl_history) >= 2:
                prev = kl_history[-2]
                if prev > 0 and (kl_raw - prev) / prev > 0.25:
                    kl_trend = "KL 上升过快（相邻窗口 >25%）"

            if global_step % cfg['train']['log_interval'] == 0:
                log_dict = {
                    "Time": human_time(),
                    "Epoch": epoch,
                    "Step": global_step,
                    "Loss_Total": float(total_loss.item()),
                    "Loss_Recon": float(rec_loss.item()),
                    "Loss_Recon_BCE": float(bce_loss.item()),
                    "Loss_Recon_Dice": float(dice_loss.item()),
                    "Dice_Weight": float(dice_weight),
                    "Loss_KL": float(kl_raw),
                    "KL_Weight": float(kl_weight),
                    "Loss_KL_weighted": float(kl_contrib),
                    "KL_avg_per_latent": float(avg_kl_per_latent),
                    "KL_contrib_ratio": float(kl_contrib_ratio),
                    "Loss_G_Adv": float(g_adv_loss.item()),
                    "Loss_D": float(loss_d.item() if isinstance(loss_d, torch.Tensor) else float(loss_d)),
                    "GradNorm_VAE": float(grad_norm_vae),
                    "GradNorm_Disc": float(grad_norm_disc),
                    "LR_VAE": float(opt_vae.param_groups[0]['lr']),
                    "LR_Disc": float(opt_disc.param_groups[0]['lr']),
                    "KL_trend_flag": kl_trend,
                }
                logger.log(log_dict)

            pbar.set_postfix({
                "Rec": f"{rec_loss.item():.4f}",
                "BCE": f"{bce_loss.item():.4f}",
                "Dice": f"{dice_loss.item():.4f}",
                "KL": f"{kl_raw:.2f}",
                "KLw": f"{kl_weight:.2e}",
                "Gadv": f"{g_adv_loss.item():.4f}"
            })

            if avg_kl_per_latent > 0.5:
                print(f"[{human_time()}] 警告：avg_kl_per_latent={avg_kl_per_latent:.3f} > 0.5，step={global_step}")

            if kl_contrib_ratio > 0.5:
                print(f"[{human_time()}] 警告：KL 加权项占比过高 kl_contrib_ratio={kl_contrib_ratio:.3f}，step={global_step}")

            global_step += 1

        # =========================
        # 每个 epoch 保存 checkpoint
        # =========================
        if (epoch + 1) % cfg['train']['save_interval'] == 0:
            ckpt_path = os.path.join(exp_dir, f"ckpt_epoch_{epoch+1}.pt")
            torch.save({
                "epoch": epoch,
                "global_step": global_step,
                "vae_state_dict": vae.state_dict(),
                "disc_state_dict": discriminator.state_dict(),
                "optimizer_vae": opt_vae.state_dict(),
                "optimizer_disc": opt_disc.state_dict(),
                "kl_warmup_steps": kl_warmup_steps,
                "kl_start_step": kl_start_step,
            }, ckpt_path)
            print(f"[{human_time()}] 已保存 checkpoint：{ckpt_path}")

    print(f"[{human_time()}] 训练结束")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import os
# 推荐使用新环境变量名以避免版本 warning
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import yaml
import shutil
from tqdm import tqdm
import argparse
import math
import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam

from models.vae import KLVAE3D
from models.discriminator import NLayerDiscriminator3D
from data.dataset import CubeDataset
from utils.logger import CSVLogger  # 假设 CSVLogger 能接受任意字段写入

# A100 TF32
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def load_config(path):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def hinge_d_loss(logits_real, logits_fake):
    return 0.5 * (torch.mean(F.relu(1.0 - logits_real)) + torch.mean(F.relu(1.0 + logits_fake)))


def fix_compile_state_dict(state_dict):
    """
    Remove _orig_mod. prefix if present (from torch.compile)
    """
    new_state = {}
    for k, v in state_dict.items():
        if isinstance(k, str) and k.startswith("_orig_mod."):
            new_state[k[10:]] = v
        else:
            new_state[k] = v
    return new_state


def calc_latent_volume(cfg):
    """
    根据 config 的 image_size 与 ch_mult 计算 latent spatial volume (D*H*W)
    假设每次 ch_mult 表示下采样 /2（总下采样因子为 2**len(ch_mult)）
    """
    image_size = cfg['data']['image_size']
    ch_mult_len = len(cfg['model']['ch_mult'])
    downsample = 2 ** ch_mult_len
    if image_size % downsample != 0:
        raise ValueError(f"image_size {image_size} not divisible by downsample factor {downsample}")
    latent_edge = image_size // downsample
    return latent_edge ** 3  # D*H*W


def human_time():
    return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/train_config.yaml')
    parser.add_argument('--resume', type=str, default=None, help='path to checkpoint to resume from')
    parser.add_argument('--restart', action='store_true', help='ignore resume and restart training from scratch')
    parser.add_argument('--reset-kl', action='store_true', help='if set, reset KL warmup on resume (force warmup from 0)')
    args = parser.parse_args()

    cfg = load_config(args.config)
    exp_dir = os.path.join(cfg['experiment']['save_dir'], cfg['experiment']['exp_name'])
    os.makedirs(exp_dir, exist_ok=True)
    if args.resume is None or args.restart:
        # 只有当没有 resume 或明确 restart 时才备份 config（避免覆盖旧 config）
        shutil.copy(args.config, os.path.join(exp_dir, 'config.yaml'))

    logger = CSVLogger(exp_dir)  # 假设该 logger 会自动 append csv header
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[{human_time()}] Running on {device}")

    # Dataset / Dataloader
    dataset = CubeDataset(cfg['data']['data_root'],
                          cfg['data']['file_extension'],
                          crop_size=cfg['data']['croped_size'],
                          is_train=True)
    dataloader = DataLoader(dataset,
                            batch_size=cfg['train']['batch_size'],
                            shuffle=True,
                            num_workers=cfg['data']['num_workers'],
                            pin_memory=True,
                            persistent_workers=False)

    # Models
    vae = KLVAE3D(cfg).to(device)
    discriminator = NLayerDiscriminator3D().to(device)

    # Optimizers
    opt_vae = Adam(vae.parameters(), lr=float(cfg['train']['lr']), betas=(0.5, 0.9))
    opt_disc = Adam(discriminator.parameters(), lr=float(cfg['train']['lr']), betas=(0.5, 0.9))

    # Training state
    global_step = 0
    start_epoch = 0

    # KL/GAN params
    kl_weight_max = float(cfg['loss']['kl_weight'])
    disc_start = int(cfg['loss']['disc_start'])
    disc_weight = float(cfg['loss']['disc_weight'])
    kl_warmup_steps = int(cfg['train'].get('kl_warmup_steps', 20000))

    # latent geometry
    latent_volume = calc_latent_volume(cfg)
    z_ch = int(cfg['model']['z_channels'])
    batch_size = int(cfg['train']['batch_size'])

    # Default kl_start_step (where annealing counts from). If 0: anneal counted from step 0.
    kl_start_step = 0

    # resume logic (兼容 _orig_mod 前缀；如果 --restart 指定则忽略 resume)
    if args.resume is not None and not args.restart:
        if os.path.isfile(args.resume):
            print(f"[{human_time()}] Loading checkpoint from {args.resume} ...")
            ckpt = torch.load(args.resume, map_location=device)

            # load model states with prefix fix attempts
            try:
                vae_state = fix_compile_state_dict(ckpt['vae_state_dict'])
                vae.load_state_dict(vae_state)
            except Exception:
                try:
                    vae.load_state_dict(ckpt['vae_state_dict'])
                except Exception as e:
                    print("Warning: failed to load vae_state_dict:", e)

            try:
                disc_state = fix_compile_state_dict(ckpt['disc_state_dict'])
                discriminator.load_state_dict(disc_state)
            except Exception:
                try:
                    discriminator.load_state_dict(ckpt['disc_state_dict'])
                except Exception as e:
                    print("Warning: failed to load disc_state_dict:", e)

            try:
                opt_vae.load_state_dict(ckpt['optimizer_vae'])
            except Exception:
                print("Warning: failed to load optimizer_vae (will init new)")

            if 'optimizer_disc' in ckpt:
                try:
                    opt_disc.load_state_dict(ckpt['optimizer_disc'])
                except Exception:
                    print("Warning: failed to load optimizer_disc (will init new)")
            else:
                print("No optimizer_disc in checkpoint, disc optimizer will start from scratch.")

            start_epoch = int(ckpt.get('epoch', 0)) + 1
            global_step = int(ckpt.get('global_step', 0))

            # ====== 关键修改 ======
            # 如果 checkpoint 中保存了 kl_start_step（训练开始 warmup 时记录的值），就使用它，
            # 这样 resume 时 annealing 的计数点不会被重置/错位，从而实现无缝衔接。
            # 如果用户显式传入 --reset-kl，则强制把 kl_start_step 设为 global_step（即从 0 重新 warmup）。
            if args.reset_kl:
                kl_start_step = global_step  # 强制从 resume 时刻重新 warmup（用户要求）
                print(f"[{human_time()}] KL warmup forced reset at resume (kl_start_step set to current global_step={global_step}).")
            else:
                # 优先使用 checkpoint 中保存的 kl_start_step（如果有），否则采用 ckpt 中的 kl_start_step 或默认 0。
                ckpt_kl_start = ckpt.get('kl_start_step', None)
                if ckpt_kl_start is not None:
                    kl_start_step = int(ckpt_kl_start)
                    print(f"[{human_time()}] Restored kl_start_step={kl_start_step} from checkpoint (resume will be seamless).")
                else:
                    # 如果 checkpoint 不含 kl_start_step（早期 checkpoint），则：
                    # - 如果 global_step >= kl_warmup_steps: warmup 已经完成，保持 kl_start_step = 0（使 kl_weight=kl_max）
                    # - 否则我们假设训练是从 step 0 开始的（kl_start_step=0），可以继续计算 effective_step=global_step
                    kl_start_step = 0
                    print(f"[{human_time()}] No kl_start_step in checkpoint, defaulting kl_start_step=0 (will compute effective_step from global_step).")

            print(f"[{human_time()}] Resume: start_epoch={start_epoch}, global_step={global_step}, kl_start_step={kl_start_step}")
        else:
            print(f"[{human_time()}] Warning: resume file not found: {args.resume}. Starting from scratch.")
            kl_start_step = 0
    else:
        # no resume or restart requested
        kl_start_step = 0
        if args.restart and args.resume is not None:
            print(f"[{human_time()}] Restart requested: ignoring resume path {args.resume} and starting from scratch. kl_start_step=0")

    # Monitoring history for simple trend detection
    from collections import deque
    kl_history = deque(maxlen=5)

    # Training loop
    for epoch in range(start_epoch, cfg['train']['epochs']):
        vae.train()
        discriminator.train()

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{cfg['train']['epochs']}", ncols=120)

        for batch_idx, images in enumerate(pbar):
            images = images.to(device, non_blocking=True)  # shape [B,1,D,H,W]

            # effective step for annealing: start counting from kl_start_step
            effective_step = max(0, global_step - kl_start_step)
            kl_weight = min(kl_weight_max, kl_weight_max * (effective_step / max(1, kl_warmup_steps)))

            # optionally print when discriminator will activate
            if global_step == disc_start:
                print(f"[{human_time()}] Discriminator activation threshold reached at step {global_step}")

            # === Train VAE (Generator) ===
            opt_vae.zero_grad()
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                reconstructions, posterior = vae(images)

                rec_loss = torch.mean(torch.abs(images - reconstructions))  # L1
                kl_per_sample = posterior.kl()
                kl_loss = torch.mean(kl_per_sample)

                if global_step > disc_start:
                    logits_fake_for_g = discriminator(reconstructions)
                    g_adv_loss = -torch.mean(logits_fake_for_g)
                else:
                    g_adv_loss = torch.tensor(0.0, device=device)

                total_loss = rec_loss + kl_weight * kl_loss + disc_weight * g_adv_loss

                if global_step < 20 and (global_step % 1 == 0):
                    # posterior.logvar exists due to our Distribution class
                    lv = posterior.logvar.detach()
                    # 计算 batch 上的 min/max/mean (把 spatial+channel 累加为 scalar)
                    lv_min = float(lv.min().item())
                    lv_max = float(lv.max().item())
                    lv_mean = float(lv.mean().item())
                    print(f"[DEBUG] step={global_step} logvar min/max/mean = {lv_min:.4f} / {lv_max:.4f} / {lv_mean:.4f}")

            # backward / clip / step for VAE
            total_loss.backward()
            grad_norm_vae = torch.nn.utils.clip_grad_norm_(vae.parameters(), max_norm=1.0)
            opt_vae.step()

            # === Train Discriminator ===
            loss_d = torch.tensor(0.0, device=device)
            grad_norm_disc = 0.0
            if global_step > disc_start:
                opt_disc.zero_grad()
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    logits_real = discriminator(images)
                    logits_fake = discriminator(reconstructions.detach())
                    loss_d = hinge_d_loss(logits_real, logits_fake)
                loss_d.backward()
                grad_norm_disc = torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)
                opt_disc.step()

            # === Metrics for CSV ===
            kl_raw = float(kl_loss.item())
            # kl_loss here is per-sample mean; keep it consistent with prior CSV
            avg_kl_per_latent = kl_raw / (z_ch * latent_volume)
            kl_contrib = kl_weight * kl_raw
            kl_contrib_ratio = (kl_contrib / (rec_loss.item() + 1e-12))

            # push into history and simple trend warning
            kl_history.append(kl_raw)
            kl_trend = ""
            if len(kl_history) >= 2:
                prev = kl_history[-2]
                if prev > 0 and (kl_raw - prev) / prev > 0.25:
                    kl_trend = "KL rising fast (>25% vs prev window)"

            if global_step % cfg['train']['log_interval'] == 0:
                log_dict = {
                    "Time": human_time(),
                    "Epoch": epoch,
                    "Step": global_step,
                    "Loss_Total": float(total_loss.item()),
                    "Loss_Recon": float(rec_loss.item()),
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
                "KL": f"{kl_raw:.1f}",
                "KLw": f"{kl_weight:.2e}",
                "Gadv": f"{g_adv_loss.item():.4f}"
            })

            # Simple console warnings
            if avg_kl_per_latent > 0.5:
                print(f"[{human_time()}] WARNING: avg_kl_per_latent={avg_kl_per_latent:.3f} > 0.5 at step {global_step}")

            if kl_contrib_ratio > 0.5:
                print(f"[{human_time()}] WARNING: kl_contrib_ratio={kl_contrib_ratio:.3f} (KL contributes >50% of Rec) at step {global_step}")

            global_step += 1

        # End epoch operations: checkpointing
        if (epoch + 1) % cfg['train']['save_interval'] == 0:
            ckpt_path = os.path.join(exp_dir, f"ckpt_epoch_{epoch+1}.pt")
            torch.save({
                "epoch": epoch,
                "global_step": global_step,
                "vae_state_dict": vae.state_dict(),
                "disc_state_dict": discriminator.state_dict(),
                "optimizer_vae": opt_vae.state_dict(),
                "optimizer_disc": opt_disc.state_dict(),
                # 记录 kl_warmup/kl_start，方便 resume 时无缝衔接
                "kl_warmup_steps": kl_warmup_steps,
                "kl_start_step": kl_start_step
            }, ckpt_path)
            print(f"[{human_time()}] Saved checkpoint to {ckpt_path}")

    print(f"[{human_time()}] Training finished.")


if __name__ == "__main__":
    main()
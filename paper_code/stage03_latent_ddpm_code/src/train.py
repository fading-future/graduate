import os
import csv
import json
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.config import CONFIG
from src.ema import EMA
from src.diffusion_trainer import DiffusionTrainer
from src.dataset_latent import LatentDataset
from src.model_latent import ConditionalLatentUNet
from utils.get_root_path import get_project_root

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# ---------------- plotting util (unchanged) ----------------
def plot_paper_curves(csv_path, save_dir):
    if not os.path.exists(csv_path):
        return
    try:
        df = pd.read_csv(csv_path)
    except:
        return
    plt.style.use('default')
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
    fig, ax1 = plt.subplots(figsize=(10, 6), dpi=150)
    steps = df['Step']
    loss_raw = df['Loss']
    loss_smooth = loss_raw.rolling(window=50, min_periods=1).mean()
    color = '#D62728'
    ax1.plot(steps, loss_raw, color=color, alpha=0.15, linewidth=0.5, label='Raw Loss')
    ax1.plot(steps, loss_smooth, color=color, linewidth=2.0, label='Smoothed Loss')
    ax1.set_xlabel('Iteration Steps', fontsize=12, fontweight='bold')
    ax1.set_ylabel('MSE Loss (Log Scale)', fontsize=12, fontweight='bold', color=color)
    ax1.set_yscale('log')
    ax1.grid(True, which="both", ls="--", alpha=0.3)
    ax1.legend(loc='upper right')
    plt.title(f"Training Dynamics: {CONFIG['experiment_name']}", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "loss_curve.png"))
    plt.close()
    print(f"📈 Chart updated: {os.path.join(save_dir, 'loss_curve.png')}")

# ---------------- experiment setup ----------------
def setup_experiment():
    root = get_project_root()
    exp_dir = os.path.join(root, "exp_results", CONFIG["experiment_name"])
    os.makedirs(exp_dir, exist_ok=True)

    model_dir = os.path.join(exp_dir, CONFIG["model_output_dir"])
    log_dir = os.path.join(exp_dir, CONFIG["log_output_dir"])
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    config_path = os.path.join(exp_dir, "config.json")
    with open(str(config_path), "w", encoding="utf-8") as f:
        json.dump(CONFIG, f, indent=4, ensure_ascii=False)

    # main training csv (keep same columns to avoid breaking existing file)
    csv_path = os.path.join(log_dir, "training_log.csv")
    if not os.path.exists(csv_path):
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Epoch', 'Step', 'Loss', 'LR'])

    # pred_x0 logging csv (keep)
    pred_csv_path = os.path.join(log_dir, "pred_x0_log.csv")
    if not os.path.exists(pred_csv_path):
        with open(pred_csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Epoch', 'Step', 'Pred_x0_penalty'])

    region_csv_path = os.path.join(log_dir, "region_loss_log_v2.csv")

    # 初始化 region CSV（如果不存在就写表头）
    if not os.path.exists(region_csv_path):
        with open(region_csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "Epoch", "Step",
                "DiffLoss_unknown", "DiffLoss_known",
                "KnownConsistency", "BoundaryX0Cons",
                "UnknownRatio"
            ])

    return model_dir, log_dir, csv_path, pred_csv_path, region_csv_path

# ---------------- main ----------------
def main():
    print(f"🚀 Starting Stage 2 Training: {CONFIG['experiment_name']}")
    model_dir, log_dir, csv_path, pred_csv_path, region_csv_path = setup_experiment()

    # ============================================================
    # 0) 数据加载
    # ============================================================
    dataset = LatentDataset(
        data_dir=CONFIG['processed_data_dir'],
        augment=not CONFIG.get("overfit_num_samples", 0),  # 有 overfit 就关增强（默认策略）
        overfit_num_samples=int(CONFIG.get("overfit_num_samples", 0))
    )
    dataloader = DataLoader(
        dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=True,
        num_workers=CONFIG['num_workers'],
        pin_memory=CONFIG.get('pin_memory', True)
    )
    print(f"📦 Data loaded: {len(dataset)} samples.")

    # ============================================================
    # 1) 模型 / EMA / 优化器 / 调度器
    # ============================================================
    model = ConditionalLatentUNet(
        in_channels=CONFIG['in_channels'],
        out_channels=CONFIG['out_channels'],
        base_channels=CONFIG['base_channels'],
        channel_mults=CONFIG['channel_mults'],
        use_attention=CONFIG['use_attention']
    ).to(CONFIG['device'])

    ema = EMA(model, decay=0.9999).to(CONFIG['device'])
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['lr'])
    scheduler = CosineAnnealingLR(optimizer, T_max=CONFIG['epochs'], eta_min=1e-6)

    diffusion = DiffusionTrainer(model, CONFIG)  # 主要用它的 alphas_cumprod/schedule
    scaler = GradScaler('cuda' if torch.cuda.is_available() else 'cpu')

    # ============================================================
    # 2) 自动断点恢复（尽量不改你的逻辑，只加注释/整理）
    # ============================================================
    start_epoch = 0

    if CONFIG.get('resume', True):
        import re
        checkpoints = [f for f in os.listdir(model_dir) if f.startswith("unet_epoch_") and f.endswith(".pth")]
        if len(checkpoints) > 0:
            checkpoints.sort(key=lambda x: int(re.findall(r'\d+', x)[0]))
            latest_ckpt = checkpoints[-1]
            checkpoint_path = os.path.join(model_dir, latest_ckpt)

            print(f"🔄 Found latest checkpoint: {checkpoint_path}, loading...")
            checkpoint = torch.load(checkpoint_path, map_location=CONFIG['device'])

            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])

                # --- 关键：加载 optimizer 但强制重置 LR（保持你的策略）---
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                for pg in optimizer.param_groups:
                    pg['lr'] = CONFIG['lr']
                print(f"✅ Optimizer loaded, LR reset to {CONFIG['lr']}")

                if 'ema_state_dict' in checkpoint:
                    ema.load_state_dict(checkpoint['ema_state_dict'])
                    print("✅ EMA weights loaded.")
                else:
                    ema = EMA(model, decay=0.9999).to(CONFIG['device'])

                if 'scheduler_state_dict' in checkpoint:
                    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                    print("✅ Scheduler state loaded.")

                start_epoch = checkpoint['epoch']
                print(f"✅ Loaded Dict: Resuming from Epoch {start_epoch}")
            else:
                # 兼容旧格式：只有 state_dict
                model.load_state_dict(checkpoint)
                ema = EMA(model, decay=0.9999).to(CONFIG['device'])
                start_epoch = 16
                print(f"⚠️ Loaded Weights Only: Starting from Epoch {start_epoch + 1}")
        else:
            print("✨ No checkpoint found. Starting from scratch.")
    else:
        print("✨ Resume disabled. Starting from scratch.")
        start_epoch = 0

    # ============================================================
    # 3) global_step 对齐（保持你的 CSV 对齐策略）
    # ============================================================
    global_step = start_epoch * len(dataloader)

    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            if not df.empty:
                last_step_csv = int(df.iloc[-1]['Step'])
                last_epoch_csv = int(df.iloc[-1]['Epoch'])
                if last_epoch_csv == start_epoch:
                    print(f"📍 Aligned global_step from CSV: {last_step_csv}")
                    global_step = last_step_csv
                else:
                    print(f"⚠️ CSV epoch ({last_epoch_csv}) != Checkpoint epoch ({start_epoch}). Using calculated step: {global_step}")
        except Exception as e:
            print(f"⚠️ Failed to read step from CSV: {e}. Using calculated step.")

    print(f"🚀 Training starting at Global Step: {global_step}")
    pred_log_every = CONFIG.get('pred_log_every', 50)

    # ============================================================
    # 4) 训练主循环
    # ============================================================
    for epoch in range(start_epoch, CONFIG['epochs']):
        model.train()
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']}")
        epoch_loss = 0.0

        for step, batch in enumerate(pbar):
            global_step += 1

            # ------------------------------------------------------------
            # 4.1 取数据
            # ------------------------------------------------------------
            x_0 = batch["GT"].to(CONFIG['device'], non_blocking=True)          # (B,4,32,32,32)
            condition = batch["Condition"].to(CONFIG['device'], non_blocking=True)  # (B,4,32,32,32) = x0*mask
            mask = batch["Mask"].to(CONFIG['device'], non_blocking=True)      # (B,1,32,32,32) 1=known 0=unknown
            porosity = batch["Porosity"].to(CONFIG['device'], non_blocking=True)  # (B,1)

            B = x_0.shape[0]
            t = torch.randint(0, CONFIG['timesteps'], (B,), device=CONFIG['device']).long()

            # ------------------------------------------------------------
            # 4.2 前向 + loss（这里是你主要想改清楚的部分）
            # ------------------------------------------------------------
            with autocast('cuda'):
                # =========================
                # [L1-1] Masked Diffusion / Known-hard
                # 训练时就保证 known 区域的 x_t 永远等于 q(condition, t)
                # unknown 区域才来自 q(x0, t)
                # =========================
                ab_t = diffusion.alphas_cumprod[t].view(-1, 1, 1, 1, 1)   # alpha_bar(t)
                sqrt_ab = torch.sqrt(ab_t)
                sqrt_om = torch.sqrt(1.0 - ab_t)

                # 用同一份噪声，让 known_xt 与 unknown_xt 在噪声空间严格对齐（关键）
                noise = torch.randn_like(x_0)

                # unknown: 来自 GT 的 q(x0,t)
                x_t = sqrt_ab * x_0 + sqrt_om * noise

                # known: 强制来自 condition 的 q(cond,t)
                known_xt = sqrt_ab * condition + sqrt_om * noise

                # 拼成最终输入 x_t（known 硬替换）
                x_t = x_t * (1.0 - mask) + known_xt * mask

                # 模型输入：noisy + condition + mask
                model_input = torch.cat([x_t, condition, mask], dim=1)
                noise_pred = model(model_input, t, porosity)

                # =========================
                # 区域 mask（broadcast 到通道）
                # =========================
                unknown = (1.0 - mask)   # (B,1,D,H,W)
                known = mask

                C = noise_pred.shape[1]
                unknown_b = unknown.expand(-1, C, -1, -1, -1) if C != 1 else unknown
                known_b   = known.expand(-1, C, -1, -1, -1)   if C != 1 else known

                # =========================
                # diffusion 基础误差（噪声预测误差）
                # =========================
                if CONFIG.get('loss_type', 'l1').lower() == 'mse':
                    raw = (noise_pred - noise) ** 2
                else:
                    raw = torch.abs(noise_pred - noise)

                # =========================
                # 可选：Min-SNR weighting（保持你的实现）
                # =========================
                use_min_snr = CONFIG.get('use_min_snr', False)
                gamma = float(CONFIG.get('min_snr_gamma', 5.0))
                if use_min_snr:
                    snr = ab_t / (1.0 - ab_t)
                    w = torch.minimum(snr, torch.tensor(gamma, device=snr.device)) / snr
                    raw = raw * w

                # =========================
                # boundary band 加权（只加在 unknown 上）
                # =========================
                band_width = int(CONFIG.get('boundary_band_width', 2))
                band_weight = float(CONFIG.get('boundary_band_weight', 4.0))

                # 为了后面 boundary_x0_consistency 复用，这里统一定义 boundary_band
                boundary_band = torch.zeros_like(unknown)

                if band_width > 0 and band_weight > 0:
                    k = 2 * band_width + 1
                    known_dil = F.max_pool3d(known, kernel_size=k, stride=1, padding=band_width)
                    unk_dil   = F.max_pool3d(unknown, kernel_size=k, stride=1, padding=band_width)
                    boundary_band = (known_dil * unk_dil).clamp(0.0, 1.0)  # (B,1,D,H,W)

                    weight_map = (1.0 + band_weight * boundary_band) * unknown  # (B,1,D,H,W)
                    weight_map_b = weight_map.expand(-1, C, -1, -1, -1) if C != 1 else weight_map
                else:
                    weight_map_b = unknown_b

                den_u = weight_map_b.sum().clamp_min(1.0)
                loss_diff_unknown = (raw * weight_map_b).sum() / den_u

                den_k = known_b.sum().clamp_min(1.0)
                loss_diff_known = (raw * known_b).sum() / den_k  # 以前只做日志，现在 L1 要引入一点点权重

                # =========================
                # pred_x0（注意：这里必须用 x_t，不再是 x_noisy）
                # =========================
                pred_x0 = (x_t - sqrt_om * noise_pred) / (sqrt_ab + 1e-5)

                safe_thresh = float(CONFIG.get('safe_threshold', 6.0))
                pred_x0_clamped = torch.clamp(pred_x0, min=-safe_thresh, max=safe_thresh)

                # =========================
                # known-consistency（保持你原策略：对 known 区域约束 pred_x0）
                # =========================
                known_cons_w = float(CONFIG.get('known_consistency_weight', 0.0))
                known_cons_type = str(CONFIG.get('known_consistency_type', 'l1')).lower()

                if known_cons_w > 0:
                    if known_cons_type == 'mse':
                        cons_raw = (pred_x0_clamped - x_0) ** 2
                    else:
                        cons_raw = torch.abs(pred_x0_clamped - x_0)

                    # 你原来这里乘 ab_t：保持不动（它主要是稳定用）
                    cons_raw = cons_raw * ab_t
                    loss_known_cons = (cons_raw * known_b).sum() / den_k
                else:
                    loss_known_cons = torch.zeros((), device=CONFIG['device'], dtype=pred_x0.dtype)

                # =========================
                # pred_x0 penalty（保持你的写法）
                # =========================
                penalty_raw = (pred_x0 - pred_x0_clamped) ** 2
                pred_x0_penalty = (penalty_raw * ab_t).mean()

                reg_w = float(CONFIG.get('pred_x0_reg_weight', 0.1))
                if torch.isnan(pred_x0_penalty) or torch.isinf(pred_x0_penalty):
                    pred_x0_penalty = torch.tensor(0.0, device=CONFIG['device'])
                pred_x0_penalty = torch.clamp(pred_x0_penalty, max=10.0)

                # =========================
                # [L1-3] boundary x0 consistency（只在 unknown 且靠边界）
                # 关键：默认不乘 ab_t，避免边界监督在结构阶段被压没
                # =========================
                bx_w = float(CONFIG.get('boundary_x0_consistency_weight', 0.0))
                bx_type = str(CONFIG.get('boundary_x0_consistency_type', 'l1')).lower()

                if bx_w > 0:
                    boundary_unknown = boundary_band * unknown  # (B,1,D,H,W)
                    boundary_unknown_b = boundary_unknown.expand(-1, C, -1, -1, -1) if C != 1 else boundary_unknown
                    den_bx = boundary_unknown_b.sum().clamp_min(1.0)

                    if bx_type == 'mse':
                        bx_raw = (pred_x0_clamped - x_0) ** 2
                    else:
                        bx_raw = torch.abs(pred_x0_clamped - x_0)

                    # ✅ L1：不再乘 ab_t（你已经注释过了，这里固定成“不乘”版本）
                    loss_boundary_x0 = (bx_raw * boundary_unknown_b).sum() / den_bx
                else:
                    loss_boundary_x0 = torch.zeros((), device=CONFIG['device'], dtype=pred_x0.dtype)

                # =========================
                # [L1-2] 给 known 少量 diffusion loss（稳定特征，利于边界传递）
                # =========================
                known_diff_w = float(CONFIG.get('known_diff_weight', 0.05))

                # =========================
                # 总 loss（保持你的结构，只是把 L1 的项补齐）
                # =========================
                loss = (
                    loss_diff_unknown
                    + known_diff_w * loss_diff_known
                    + known_cons_w * loss_known_cons
                    + bx_w * loss_boundary_x0
                    + reg_w * pred_x0_penalty
                )

            # ------------------------------------------------------------
            # 4.3 反传更新（保持你的 AMP + clip）
            # ------------------------------------------------------------
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            ema.update(model)

            # ------------------------------------------------------------
            # 4.4 日志与 CSV（保持你的逻辑）
            # ------------------------------------------------------------
            curr_loss_val = float(loss.item())
            epoch_loss += curr_loss_val
            current_lr = optimizer.param_groups[0]['lr']

            pbar.set_postfix(loss=f"{curr_loss_val:.4f}", lr=f"{current_lr:.2e}")

            if os.path.exists(csv_path):
                with open(csv_path, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([epoch + 1, global_step, f"{curr_loss_val:.6f}", f"{current_lr:.8f}"])

            if global_step % pred_log_every == 0:
                with open(pred_csv_path, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([epoch + 1, global_step, float(pred_x0_penalty.detach().cpu().item())])

            # region-wise 记录
            unknown_ratio = float((unknown.sum() / unknown.numel()).detach().cpu().item())

            with open(region_csv_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    epoch + 1, global_step,
                    float(loss_diff_unknown.detach().cpu().item()),
                    float(loss_diff_known.detach().cpu().item()),
                    float(loss_known_cons.detach().cpu().item()),
                    float(loss_boundary_x0.detach().cpu().item()),
                    unknown_ratio
                ])

            pbar.set_postfix(
                loss=f"{curr_loss_val:.4f}",
                u=f"{float(loss_diff_unknown.detach().cpu()):.4f}",
                k=f"{float(loss_diff_known.detach().cpu()):.4f}",
                kc=f"{float(loss_known_cons.detach().cpu()):.4f}",
                lr=f"{current_lr:.2e}"
            )

        # ============================================================
        # 5) epoch 结束：scheduler + save（保持你的逻辑）
        # ============================================================
        scheduler.step(epoch + 1)
        avg_loss = epoch_loss / len(dataloader)
        print(f"✅ Epoch {epoch+1} Done. Avg Loss: {avg_loss:.6f}")

        if (epoch + 1) % CONFIG['save_model_every'] == 0:
            save_path = os.path.join(model_dir, f"unet_epoch_{epoch+1}.pth")
            save_dict = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'ema_state_dict': ema.state_dict(),
                'loss': avg_loss,
            }
            torch.save(save_dict, save_path)
            print(f"💾 Full checkpoint (with EMA) saved: {save_path}")
            try:
                plot_paper_curves(csv_path, log_dir)
            except Exception as e:
                print(f"⚠️ Plotting failed: {e}")

if __name__ == "__main__":
    main()
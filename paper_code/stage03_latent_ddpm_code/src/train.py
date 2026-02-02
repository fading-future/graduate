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

    # NEW: region-wise error logging csv (do not touch old csv header)
    # region_csv_path = os.path.join(log_dir, "region_loss_log.csv")
    # if not os.path.exists(region_csv_path):
    #     with open(region_csv_path, 'w', newline='') as f:
    #         writer = csv.writer(f)
    #         writer.writerow([
    #             'Epoch', 'Step',
    #             'DiffLoss_unknown', 'DiffLoss_known',
    #             'KnownConsistency', 'UnknownRatio'
    #         ])

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

    dataset = LatentDataset(data_dir=CONFIG['processed_data_dir'])
    dataloader = DataLoader(
        dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=True,
        num_workers=CONFIG['num_workers'],
        pin_memory=CONFIG.get('pin_memory', True)
    )
    print(f"📦 Data loaded: {len(dataset)} samples.")

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
    # criterion = nn.MSELoss()
    # criterion = nn.L1Loss()
    diffusion = DiffusionTrainer(model, CONFIG)
    scaler = GradScaler('cuda' if torch.cuda.is_available() else 'cpu')

    # ================= 🛡️ 自动断点重训 & CSV 对齐逻辑 & 忽略之前的学习率加载=================
    start_epoch = 0
    
    # 1. 自动寻找最新的 checkpoint
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
                
                # --- 🔴 关键修改 1: 加载 Optimizer 但强制重置 LR ---
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
                model.load_state_dict(checkpoint)
                ema = EMA(model, decay=0.9999).to(CONFIG['device'])
                start_epoch = 16 # 如果是纯权重，只能瞎猜或者手动指定
                print(f"⚠️ Loaded Weights Only: Starting from Epoch {start_epoch + 1}")
        else:
            print("✨ No checkpoint found. Starting from scratch.")
    else:
        print("✨ Resume disabled. Starting from scratch.")
        start_epoch = 0
        global_step = 0


    # 2. 核心修改：优先从 CSV 对齐 global_step
    # 默认值 (数学估算)
    global_step = start_epoch * len(dataloader)
    
    if os.path.exists(csv_path):
        try:
            # 读取 CSV 最后一行
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
    
    pred_log_every = CONFIG.get('pred_log_every', 50)  # default every 50 steps

    for epoch in range(start_epoch, CONFIG['epochs']):
        model.train()
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']}")
        epoch_loss = 0.0

        for step, batch in enumerate(pbar):
            global_step += 1

            x_0 = batch["GT"].to(CONFIG['device'], non_blocking=True)
            condition = batch["Condition"].to(CONFIG['device'], non_blocking=True)
            mask = batch["Mask"].to(CONFIG['device'], non_blocking=True)
            porosity = batch["Porosity"].to(CONFIG['device'], non_blocking=True)

            t = torch.randint(0, CONFIG['timesteps'], (x_0.shape[0],), device=CONFIG['device']).long()

            with autocast('cuda'):
                x_noisy, noise = diffusion.add_noise(x_0, t)
                model_input = torch.cat([x_noisy, condition, mask], dim=1)
                noise_pred = model(model_input, t, porosity)

                # ---------------- [NEW] region masks (broadcast to channels) ----------------
                # mask: 1=known, 0=unknown
                unknown = (1.0 - mask)  # (B,1,D,H,W)
                known = mask

                # broadcast to match noise_pred/noise shape: (B,C,D,H,W)
                C = noise_pred.shape[1]
                if C != 1:
                    unknown_b = unknown.expand(-1, C, -1, -1, -1)
                    known_b = known.expand(-1, C, -1, -1, -1)
                else:
                    unknown_b = unknown
                    known_b = known

                # ---------------- base raw loss per element ----------------
                if CONFIG.get('loss_type', 'l1').lower() == 'mse':
                    raw = (noise_pred - noise) ** 2
                else:
                    raw = torch.abs(noise_pred - noise)

                # ---------------- optional: min-SNR weighting (still supports) ----------------
                use_min_snr = CONFIG.get('use_min_snr', False)
                gamma = float(CONFIG.get('min_snr_gamma', 5.0))
                if use_min_snr:
                    alpha_bar_t = diffusion.alphas_cumprod[t].view(-1, 1, 1, 1, 1)
                    snr = alpha_bar_t / (1.0 - alpha_bar_t)
                    w = torch.minimum(snr, torch.tensor(gamma, device=snr.device)) / snr  # (B,1,1,1,1)
                    raw = raw * w  # broadcast ok

                band_width = int(CONFIG.get('boundary_band_width', 2))          # thickness in voxels, e.g. 2~4
                band_weight = float(CONFIG.get('boundary_band_weight', 4.0))    # extra weight on boundary, e.g. 2~8

                if band_width > 0 and band_weight > 0:
                    # dilation over known/unknown to get "near boundary" region
                    k = 2 * band_width + 1
                    known_dil = F.max_pool3d(known, kernel_size=k, stride=1, padding=band_width)
                    unk_dil   = F.max_pool3d(unknown, kernel_size=k, stride=1, padding=band_width)

                    # near-boundary voxels: close to both known and unknown
                    boundary_band = (known_dil * unk_dil).clamp(0.0, 1.0)     # (B,1,D,H,W)

                    # weights only for unknown region
                    weight_map = (1.0 + band_weight * boundary_band) * unknown   # (B,1,D,H,W)

                    # broadcast to channels
                    if C != 1:
                        weight_map_b = weight_map.expand(-1, C, -1, -1, -1)
                    else:
                        weight_map_b = weight_map
                else:
                    # no weighting
                    weight_map_b = unknown_b

                den_u = weight_map_b.sum().clamp_min(1.0)
                loss_diff_unknown = (raw * weight_map_b).sum() / den_u

                # for logging only (not used in backward)
                den_k = known_b.sum().clamp_min(1.0)
                loss_diff_known = (raw * known_b).sum() / den_k

                # ---------------- pred_x0 ----------------
                alpha_bar_t = diffusion.alphas_cumprod[t].view(-1, 1, 1, 1, 1)
                sqrt_alpha_bar = torch.sqrt(alpha_bar_t)
                sqrt_one_minus = torch.sqrt(1.0 - alpha_bar_t)
                pred_x0 = (x_noisy - sqrt_one_minus * noise_pred) / (sqrt_alpha_bar + 1e-5)

                # ---------------- clamp FIRST (so it exists) ----------------
                safe_thresh = CONFIG.get('safe_threshold', 6.0)
                pred_x0_clamped = torch.clamp(pred_x0, min=-safe_thresh, max=safe_thresh)

                # ---------------- [NEW] known-consistency (stable) ----------------
                known_cons_w = float(CONFIG.get('known_consistency_weight', 0.0))
                known_cons_type = str(CONFIG.get('known_consistency_type', 'l1')).lower()

                if known_cons_w > 0:
                    pred_for_cons = pred_x0_clamped  # now defined

                    if known_cons_type == 'mse':
                        cons_raw = (pred_for_cons - x_0) ** 2
                    else:
                        cons_raw = torch.abs(pred_for_cons - x_0)

                    # suppress large-t explosion
                    cons_raw = cons_raw * alpha_bar_t

                    loss_known_cons = (cons_raw * known_b).sum() / den_k
                else:
                    loss_known_cons = torch.zeros((), device=CONFIG['device'], dtype=pred_x0.dtype)

                # ---------------- pred_x0 penalty (same as before) ----------------
                penalty_raw = ((pred_x0 - pred_x0_clamped) ** 2)
                reg_weighting = alpha_bar_t
                pred_x0_penalty = (penalty_raw * reg_weighting).mean()

                reg_w = CONFIG.get('pred_x0_reg_weight', 0.1)
                if torch.isnan(pred_x0_penalty) or torch.isinf(pred_x0_penalty):
                    pred_x0_penalty = torch.tensor(0.0, device=CONFIG['device'])
                pred_x0_penalty = torch.clamp(pred_x0_penalty, max=10.0)


                # ---------------- [NEW] boundary-band x0 consistency on UNKNOWN ----------------
                bx_w = float(CONFIG.get('boundary_x0_consistency_weight', 0.0))
                bx_type = str(CONFIG.get('boundary_x0_consistency_type', 'l1')).lower()

                if bx_w > 0:
                    # boundary_band: (B,1,D,H,W) you already computed above in boundary loss part
                    # If band is disabled, boundary_band may not exist -> rebuild minimal boundary here
                    if 'boundary_band' not in locals():
                        band_width = int(CONFIG.get('boundary_band_width', 2))
                        if band_width > 0:
                            k = 2 * band_width + 1
                            known_dil = F.max_pool3d(known, kernel_size=k, stride=1, padding=band_width)
                            unk_dil   = F.max_pool3d(unknown, kernel_size=k, stride=1, padding=band_width)
                            boundary_band = (known_dil * unk_dil).clamp(0.0, 1.0)
                        else:
                            boundary_band = torch.zeros_like(unknown)

                    # only on UNKNOWN pixels near boundary
                    boundary_unknown = boundary_band * unknown  # (B,1,D,H,W)

                    # broadcast to C
                    if C != 1:
                        boundary_unknown_b = boundary_unknown.expand(-1, C, -1, -1, -1)
                    else:
                        boundary_unknown_b = boundary_unknown

                    den_bx = boundary_unknown_b.sum().clamp_min(1.0)

                    # x0 target: GT latent x_0
                    if bx_type == 'mse':
                        bx_raw = (pred_x0_clamped - x_0) ** 2
                    else:
                        bx_raw = torch.abs(pred_x0_clamped - x_0)

                    # optional: you can keep alpha_bar_t weighting, but boundary already helps.
                    # If you keep it, the term may become too small again.
                    bx_raw = bx_raw * alpha_bar_t

                    loss_boundary_x0 = (bx_raw * boundary_unknown_b).sum() / den_bx
                else:
                    loss_boundary_x0 = torch.zeros((), device=CONFIG['device'], dtype=pred_x0.dtype)


                # ---------------- total loss ----------------
                # loss = loss_diff_unknown + known_cons_w * loss_known_cons + reg_w * pred_x0_penalty

                loss = loss_diff_unknown + known_cons_w * loss_known_cons + bx_w * loss_boundary_x0 + reg_w * pred_x0_penalty

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            ema.update(model)

            curr_loss_val = loss.item()
            epoch_loss += curr_loss_val
            current_lr = optimizer.param_groups[0]['lr']
            pbar.set_postfix(loss=f"{curr_loss_val:.4f}", lr=f"{current_lr:.2e}")

            # write main training CSV
            if os.path.exists(csv_path):
                with open(csv_path, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([epoch+1, global_step, f"{curr_loss_val:.6f}", f"{current_lr:.8f}"])

            # write pred_x0 penalty csv every N steps
            if global_step % pred_log_every == 0:
                with open(pred_csv_path, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([epoch+1, global_step, float(pred_x0_penalty.detach().cpu().item())])

            # ---- NEW: write region-wise losses ----
            unknown_ratio = float((unknown.sum() / unknown.numel()).detach().cpu().item())  # ratio in [0,1]

            # 保证 loss_boundary_x0 即使 bx_w==0 也存在（否则这里会报错）
            bx_val = float(loss_boundary_x0.detach().cpu().item()) if 'loss_boundary_x0' in locals() else 0.0


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

        scheduler.step()
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
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

    # main training csv
    csv_path = os.path.join(log_dir, "training_log.csv")
    if not os.path.exists(csv_path):
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Epoch', 'Step', 'Loss', 'LR'])

    # pred_x0 logging csv (separate file to avoid header mismatch)
    pred_csv_path = os.path.join(log_dir, "pred_x0_log.csv")
    if not os.path.exists(pred_csv_path):
        with open(pred_csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Epoch', 'Step', 'Pred_x0_penalty'])

    return model_dir, log_dir, csv_path, pred_csv_path

# ---------------- main ----------------
def main():
    print(f"🚀 Starting Stage 2 Training: {CONFIG['experiment_name']}")
    model_dir, log_dir, csv_path, pred_csv_path = setup_experiment()

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
        channel_mults=(1, 2, 4),
        use_attention=CONFIG['use_attention']
    ).to(CONFIG['device'])

    ema = EMA(model, decay=0.9999).to(CONFIG['device'])
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['lr'])
    scheduler = CosineAnnealingLR(optimizer, T_max=CONFIG['epochs'], eta_min=1e-6)
    # criterion = nn.MSELoss()
    criterion = nn.L1Loss()
    diffusion = DiffusionTrainer(model, CONFIG)
    scaler = GradScaler('cuda' if torch.cuda.is_available() else 'cpu')

    # # ================= 🛡️ 自动断点重训 & CSV 对齐逻辑 (修改版) =================
    # start_epoch = 0
    
    # # 1. 自动寻找最新的 checkpoint (不再硬编码 epoch_17)
    # # 逻辑：列出所有 unet_epoch_*.pth，按数字排序，取最大的
    # import re
    # checkpoints = [f for f in os.listdir(model_dir) if f.startswith("unet_epoch_") and f.endswith(".pth")]
    # if len(checkpoints) > 0:
    #     # 提取数字并排序: unet_epoch_17.pth -> 17
    #     checkpoints.sort(key=lambda x: int(re.findall(r'\d+', x)[0]))
    #     latest_ckpt = checkpoints[-1]
    #     checkpoint_path = os.path.join(model_dir, latest_ckpt)
        
    #     print(f"🔄 Found latest checkpoint: {checkpoint_path}, loading...")
    #     checkpoint = torch.load(checkpoint_path, map_location=CONFIG['device'])
        
    #     if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
    #         model.load_state_dict(checkpoint['model_state_dict'])
    #         optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    #         if 'ema_state_dict' in checkpoint:
    #             ema.load_state_dict(checkpoint['ema_state_dict'])
    #             print("✅ EMA weights loaded.")
    #         else:
    #             ema = EMA(model, decay=0.9999).to(CONFIG['device'])
    #         if 'scheduler_state_dict' in checkpoint:
    #              scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
    #         # 读取 Checkpoint 里的 epoch
    #         start_epoch = checkpoint['epoch']
    #         print(f"✅ Loaded Dict: Resuming from Epoch {start_epoch} (Next is {start_epoch+1})")

    # ================= 🛡️ 自动断点重训 & CSV 对齐逻辑 & 忽略之前的学习率加载=================
    start_epoch = 0
    
    # 1. 自动寻找最新的 checkpoint
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
            # for param_group in optimizer.param_groups:
            #     param_group['lr'] = CONFIG['lr'] # 强行改回 1e-4
            # print(f"🔥 Force reset Learning Rate to {CONFIG['lr']} (Ignored checkpoint LR)")
            
            if 'ema_state_dict' in checkpoint:
                ema.load_state_dict(checkpoint['ema_state_dict'])
                print("✅ EMA weights loaded.")
            else:
                ema = EMA(model, decay=0.9999).to(CONFIG['device'])
            
            # --- 🔴 关键修改 2: 不要加载 Scheduler 状态 ---
            if 'scheduler_state_dict' in checkpoint:
                 scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            print("⚠️ Scheduler reset to start (New Stage Training)")
            
            start_epoch = checkpoint['epoch']
            print(f"✅ Loaded Dict: Resuming from Epoch {start_epoch}")
        else:
            model.load_state_dict(checkpoint)
            ema = EMA(model, decay=0.9999).to(CONFIG['device'])
            start_epoch = 16 # 如果是纯权重，只能瞎猜或者手动指定
            print(f"⚠️ Loaded Weights Only: Starting from Epoch {start_epoch + 1}")
    else:
        print("✨ No checkpoint found. Starting from scratch.")

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
                
                # 只有当 CSV 记录的 Epoch 和 Checkpoint 的 Epoch 一致时，才信任 CSV 的 Step
                # 防止 CSV 是旧的，而模型是新的（或反之）
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
                loss_mse = criterion(noise_pred, noise)

                # ---------------- [Fix] pred_x0 soft-clamp penalty ----------------
                alpha_bar_t = diffusion.alphas_cumprod[t].view(-1, 1, 1, 1, 1)
                sqrt_alpha_bar = torch.sqrt(alpha_bar_t)
                sqrt_one_minus = torch.sqrt(1.0 - alpha_bar_t)
                
                # 1. 反推 pred_x0 (注意分母极小值)
                pred_x0 = (x_noisy - sqrt_one_minus * noise_pred) / (sqrt_alpha_bar + 1e-5) # 稍微调大 epsilon
                
                # 2. 计算正则化项
                safe_thresh = CONFIG.get('safe_threshold', 6.0)
                
                # 惩罚超出范围的部分
                # 使用 Huber Loss 风格或 L2，这里用 L2
                pred_x0_clamped = torch.clamp(pred_x0, min=-safe_thresh, max=safe_thresh)
                penalty_raw = ((pred_x0 - pred_x0_clamped) ** 2)

                # 3. 【核心修复】SNR 加权 / 时间步加权
                # 当 t 很大时(全噪声)，sqrt_alpha_bar 接近0，pred_x0 计算极不稳定。
                # 我们乘以 sqrt_alpha_bar (或 alpha_bar) 来抵消分母的影响。
                # 这样在 t=1000 时，权重接近0，模型不会因为数学误差被惩罚。
                # 形状: (B, 1, 1, 1, 1)
                reg_weighting = alpha_bar_t  # 或者用 sqrt_alpha_bar，这里用 alpha_bar 更保守
                
                # 加权后的 Penalty (对 Batch 求平均)
                pred_x0_penalty = (penalty_raw * reg_weighting).mean()

                # 4. 计算总 Loss
                reg_w = CONFIG.get('pred_x0_reg_weight', 0.1) # 权重建议 0.1
                
                # 5. 【最后一道防线】防止极端情况下的 NaN 或 Infinity
                # 如果 penalty 依然过大（虽然加权后不太可能），将其截断
                if torch.isnan(pred_x0_penalty) or torch.isinf(pred_x0_penalty):
                     pred_x0_penalty = torch.tensor(0.0, device=CONFIG['device'])
                
                # 将正则化项限制在一个合理范围 (例如最大 10.0)，防止冲垮 MSE Loss
                pred_x0_penalty = torch.clamp(pred_x0_penalty, max=10.0)

                loss = loss_mse + reg_w * pred_x0_penalty
                # ---------------- end pred_x0 penalty ----------------

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
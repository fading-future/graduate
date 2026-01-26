# train.py (updated: adds pred_x0_penalty logging every N steps)
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

from config import CONFIG
from ema import EMA
from diffusion_trainer import DiffusionTrainer
from dataset_latent import LatentDataset
from model_latent import ConditionalLatentUNet
from utils.get_root_path import get_root_path

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
    root = get_root_path()
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
        use_attention=(False, True, True)
    ).to(CONFIG['device'])

    ema = EMA(model, decay=0.9999).to(CONFIG['device'])
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['lr'])
    scheduler = CosineAnnealingLR(optimizer, T_max=CONFIG['epochs'], eta_min=1e-6)
    criterion = nn.MSELoss()
    diffusion = DiffusionTrainer(model, CONFIG)
    scaler = GradScaler('cuda' if torch.cuda.is_available() else 'cpu')

    # resume logic (unchanged)
    start_epoch = 0
    checkpoint_path = os.path.join(model_dir, "unet_epoch_17.pth")
    if os.path.exists(checkpoint_path):
        print(f"🔄 Found checkpoint: {checkpoint_path}, loading...")
        checkpoint = torch.load(checkpoint_path, map_location=CONFIG['device'])
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'ema_state_dict' in checkpoint:
                ema.load_state_dict(checkpoint['ema_state_dict'])
                print("✅ EMA weights loaded.")
            else:
                ema = EMA(model, decay=0.9999).to(CONFIG['device'])
                print("⚠️ No EMA in checkpoint. Initialized from current model.")
            if 'scheduler_state_dict' in checkpoint:
                 scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch']
            print(f"✅ Loaded Dict: Resuming from Epoch {start_epoch + 1}")
        else:
            model.load_state_dict(checkpoint)
            ema = EMA(model, decay=0.9999).to(CONFIG['device'])
            start_epoch = 16
            print(f"⚠️ Loaded Weights Only: Starting from Epoch {start_epoch + 1}")

    global_step = start_epoch * len(dataloader)
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


# import os
# import csv
# import json
# import pandas as pd
# import matplotlib
# matplotlib.use('Agg') # 关键：禁用 GUI 后端
# import matplotlib.pyplot as plt
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader
# from tqdm import tqdm
# from torch.amp import autocast, GradScaler
# # 修改建议：在 Optimizer 后增加 Scheduler
# from torch.optim.lr_scheduler import CosineAnnealingLR

# # 项目内部引用
# from config import CONFIG
# from ema import EMA
# from diffusion_trainer import DiffusionTrainer
# from dataset_latent import LatentDataset
# from model_latent import ConditionalLatentUNet
# from utils.get_root_path import get_root_path

# # 开启 A100 的 TF32 加速 (只需加这两行，速度可能起飞)
# torch.backends.cuda.matmul.allow_tf32 = True
# torch.backends.cudnn.allow_tf32 = True

# # ================= 绘图工具函数 (直接内置在这里) =================
# def plot_paper_curves(csv_path, save_dir):
#     """
#     读取 CSV 并绘制科研风格的 Loss 曲线
#     """
#     if not os.path.exists(csv_path): return
    
#     try:
#         df = pd.read_csv(csv_path)
#     except:
#         return 

#     # 绘图设置
#     plt.style.use('default')
#     plt.rcParams['font.family'] = 'sans-serif'
#     plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
    
#     fig, ax1 = plt.subplots(figsize=(10, 6), dpi=150)
    
#     # 1. 处理 Loss 数据 (平滑)
#     steps = df['Step']
#     loss_raw = df['Loss']
#     # 窗口越大越平滑
#     loss_smooth = loss_raw.rolling(window=50, min_periods=1).mean()
    
#     # 2. 绘制
#     color = '#D62728' # 科研红
#     ax1.plot(steps, loss_raw, color=color, alpha=0.15, linewidth=0.5, label='Raw Loss')
#     ax1.plot(steps, loss_smooth, color=color, linewidth=2.0, label='Smoothed Loss')
    
#     ax1.set_xlabel('Iteration Steps', fontsize=12, fontweight='bold')
#     ax1.set_ylabel('MSE Loss (Log Scale)', fontsize=12, fontweight='bold', color=color)
#     ax1.set_yscale('log') # 对数坐标
#     ax1.grid(True, which="both", ls="--", alpha=0.3)
#     ax1.legend(loc='upper right')
    
#     plt.title(f"Training Dynamics: {CONFIG['experiment_name']}", fontsize=14)
#     plt.tight_layout()
    
#     # 保存
#     plt.savefig(os.path.join(save_dir, "loss_curve.png"))
#     plt.close()
#     print(f"📈 Chart updated: {os.path.join(save_dir, 'loss_curve.png')}")

# # ================= 实验设置辅助 =================
# def setup_experiment():
#     root = get_root_path()
#     exp_dir = os.path.join(root, "exp_results", CONFIG["experiment_name"])
#     os.makedirs(exp_dir, exist_ok=True)
    
#     model_dir = os.path.join(exp_dir, CONFIG["model_output_dir"])
#     log_dir = os.path.join(exp_dir, CONFIG["log_output_dir"])
#     os.makedirs(model_dir, exist_ok=True)
#     os.makedirs(log_dir, exist_ok=True)

#     # 保存配置文件：实验文件夹/config.json（绝对路径）
#     config_path = os.path.join(exp_dir, "config.json")
#     with open(str(config_path), "w", encoding="utf-8") as f:
#         json.dump(CONFIG, f, indent=4, ensure_ascii=False)
    
#     # 初始化 CSV
#     csv_path = os.path.join(log_dir, "training_log.csv")
#     if not os.path.exists(csv_path):
#         with open(csv_path, 'w', newline='') as f:
#             writer = csv.writer(f)
#             writer.writerow(['Epoch', 'Step', 'Loss', 'LR'])
    
#     # pred_x0 logging csv (separate file to avoid header mismatch)
#     pred_csv_path = os.path.join(log_dir, "pred_x0_log.csv")
#     if not os.path.exists(pred_csv_path):
#         with open(pred_csv_path, 'w', newline='') as f:
#             writer = csv.writer(f)
#             writer.writerow(['Epoch', 'Step', 'Pred_x0_penalty'])

#     return model_dir, log_dir, csv_path, pred_csv_path

# def main():
#     print(f"🚀 Starting Stage 2 Training: {CONFIG['experiment_name']}")
#     model_dir, log_dir, csv_path, pred_csv_path = setup_experiment()
    
#     # 1. Dataset & DataLoader
#     dataset = LatentDataset(data_dir=CONFIG['processed_data_dir'])
#     dataloader = DataLoader(
#         dataset, 
#         batch_size=CONFIG['batch_size'], 
#         shuffle=True, 
#         num_workers=CONFIG['num_workers'], 
#         pin_memory=True
#     )
#     print(f"📦 Data loaded: {len(dataset)} samples.")

#     # 2. Model
#     model = ConditionalLatentUNet(
#         in_channels=CONFIG['in_channels'],
#         out_channels=CONFIG['out_channels'],
#         base_channels=CONFIG['base_channels'],
#         channel_mults=(1, 2, 4),
#         use_attention=(False, True, True)
#     ).to(CONFIG['device'])

#     # ✨ 初始化 EMA (在模型移至 GPU 后)
#     # 0.9999 是 Diffusion 标配，能极大提升生成连贯性
#     ema = EMA(model, decay=0.9999).to(CONFIG['device'])
    
#     # 3. Optimizer & Tools
#     optimizer = optim.AdamW(model.parameters(), lr=CONFIG['lr'])
    
#     # ⚠️ 修正：Scheduler 配置
#     # T_max=CONFIG['epochs'] 意味着我们将在每个 Epoch 结束时 step，而不是每个 batch
#     scheduler = CosineAnnealingLR(optimizer, T_max=CONFIG['epochs'], eta_min=1e-6)
    
#     criterion = nn.MSELoss()
#     diffusion = DiffusionTrainer(model, CONFIG)
#     scaler = GradScaler('cuda' if torch.cuda.is_available() else 'cpu')
    
#     # ================= 🛡️ 加载权重/继续训练逻辑 (含 EMA) =================
#     start_epoch = 0
#     checkpoint_path = os.path.join(model_dir, "unet_epoch_17.pth") 
    
#     if os.path.exists(checkpoint_path):
#         print(f"🔄 Found checkpoint: {checkpoint_path}, loading...")
#         checkpoint = torch.load(checkpoint_path, map_location=CONFIG['device'])
        
#         # 兼容性判断
#         if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
#             model.load_state_dict(checkpoint['model_state_dict'])
#             optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
#             # ✨ 加载 EMA 权重
#             if 'ema_state_dict' in checkpoint:
#                 ema.load_state_dict(checkpoint['ema_state_dict'])
#                 print("✅ EMA weights loaded.")
#             else:
#                 # 如果旧模型没有 EMA，重新初始化（从当前模型复制）
#                 ema = EMA(model, decay=0.9999).to(CONFIG['device'])
#                 print("⚠️ No EMA in checkpoint. Initialized from current model.")

#             # 恢复 Scheduler (如果有保存)
#             if 'scheduler_state_dict' in checkpoint:
#                  scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

#             start_epoch = checkpoint['epoch']
#             print(f"✅ Loaded Dict: Resuming from Epoch {start_epoch + 1}")
#         else:
#             model.load_state_dict(checkpoint)
#             # 纯权重模式下，EMA 也要重新对齐
#             ema = EMA(model, decay=0.9999).to(CONFIG['device'])
#             start_epoch = 16 
#             print(f"⚠️ Loaded Weights Only: Starting from Epoch {start_epoch + 1}")
#     # =======================================================================

#     global_step = start_epoch * len(dataloader)
#     pred_log_every = CONFIG.get('pred_log_every', 50)  # default every 50 steps
    
#     # 4. Loop
#     for epoch in range(start_epoch, CONFIG['epochs']):
#         model.train()
#         pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']}")
#         epoch_loss = 0
        
#         for step, batch in enumerate(pbar):
#             global_step += 1
            
#             x_0 = batch["GT"].to(CONFIG['device'], non_blocking=True)
#             condition = batch["Condition"].to(CONFIG['device'], non_blocking=True)
#             mask = batch["Mask"].to(CONFIG['device'], non_blocking=True)
#             porosity = batch["Porosity"].to(CONFIG['device'], non_blocking=True)
            
#             t = torch.randint(0, CONFIG['timesteps'], (x_0.shape[0],), device=CONFIG['device']).long()
            
#             with autocast('cuda'):
#                 x_noisy, noise = diffusion.add_noise(x_0, t)
#                 model_input = torch.cat([x_noisy, condition, mask], dim=1)
#                 noise_pred = model(model_input, t, porosity)
#                 loss_mse = criterion(noise_pred, noise)

#                 # --- 新增: pred_x0 正则（soft-clamp penalty） ---
#                 # pred_x0 = (x_noisy - sqrt(1-alpha_bar)*noise_pred) / sqrt(alpha_bar)
#                 alpha_bar_t = diffusion.alphas_cumprod[t].view(-1, 1, 1, 1, 1)
#                 sqrt_alpha_bar = torch.sqrt(alpha_bar_t)
#                 sqrt_one_minus = torch.sqrt(1.0 - alpha_bar_t)
#                 pred_x0 = (x_noisy - sqrt_one_minus * noise_pred) / (sqrt_alpha_bar + 1e-12)

#                 safe_thresh = CONFIG.get('safe_threshold', 6.0)
#                 pred_x0_clamped = torch.clamp(pred_x0, min=-safe_thresh, max=safe_thresh)
#                 pred_x0_penalty = ((pred_x0 - pred_x0_clamped) ** 2).mean()

#                 reg_w = CONFIG.get('pred_x0_reg_weight', 0.12)
#                 loss = loss_mse + reg_w * pred_x0_penalty
#                 # --- end pred_x0 正则 ---
            
#             optimizer.zero_grad()
#             scaler.scale(loss).backward()
#             scaler.unscale_(optimizer)
#             torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
#             scaler.step(optimizer)
#             scaler.update()

#             # ✨ 关键：每次参数更新后，更新 EMA
#             ema.update(model)
            
#             curr_loss_val = loss.item()
#             epoch_loss += curr_loss_val
#             current_lr = optimizer.param_groups[0]['lr']
#             pbar.set_postfix(loss=f"{curr_loss_val:.4f}", lr=f"{current_lr:.2e}")
            
#             # write main training CSV
#             if os.path.exists(csv_path):
#                 with open(csv_path, 'a', newline='') as f:
#                     writer = csv.writer(f)
#                     writer.writerow([epoch+1, global_step, f"{curr_loss_val:.6f}", f"{current_lr:.8f}"])

#             # write pred_x0 penalty csv every N steps
#             if global_step % pred_log_every == 0:
#                 with open(pred_csv_path, 'a', newline='') as f:
#                     writer = csv.writer(f)
#                     writer.writerow([epoch+1, global_step, float(pred_x0_penalty.detach().cpu().item())])

#         # ⚠️ 修正：Scheduler Step 移至 Epoch 循环末尾
#         # 因为 T_max 是按 Epochs 设置的
#         scheduler.step()

#         avg_loss = epoch_loss / len(dataloader)
#         print(f"✅ Epoch {epoch+1} Done. Avg Loss: {avg_loss:.6f}")
        
#         # 5. 保存模型 (保存包含 EMA 的 dict)
#         if (epoch + 1) % CONFIG['save_model_every'] == 0:
#             save_path = os.path.join(model_dir, f"unet_epoch_{epoch+1}.pth")
#             save_dict = {
#                 'epoch': epoch + 1,
#                 'model_state_dict': model.state_dict(),
#                 'optimizer_state_dict': optimizer.state_dict(),
#                 'scheduler_state_dict': scheduler.state_dict(),
#                 'ema_state_dict': ema.state_dict(),
#                 'loss': avg_loss,
#             }
#             torch.save(save_dict, save_path)
#             print(f"💾 Full checkpoint (with EMA) saved: {save_path}")
#             try:
#                 plot_paper_curves(csv_path, log_dir)
#             except Exception as e:
#                 print(f"⚠️ Plotting failed: {e}")

# if __name__ == "__main__":
#     main()

# def main():
#     print(f"🚀 Starting Stage 2 Training: {CONFIG['experiment_name']}")
#     model_dir, log_dir, csv_path = setup_experiment()
    
#     # 1. Dataset & DataLoader
#     dataset = LatentDataset(data_dir=CONFIG['processed_data_dir'])
#     dataloader = DataLoader(
#         dataset, 
#         batch_size=CONFIG['batch_size'], 
#         shuffle=True, 
#         num_workers=CONFIG['num_workers'], 
#         pin_memory=True
#     )
#     print(f"📦 Data loaded: {len(dataset)} samples.")

#     # 2. Model
#     model = ConditionalLatentUNet(
#         in_channels=CONFIG['in_channels'],
#         out_channels=CONFIG['out_channels'],
#         base_channels=CONFIG['base_channels'],
#         channel_mults=(1, 2, 4),
#         use_attention=(False, True, True)
#     ).to(CONFIG['device'])
    
#     # 3. Optimizer & Tools
#     optimizer = optim.AdamW(model.parameters(), lr=CONFIG['lr'])
#     # T_max 设置为总 epoch 数
#     scheduler = CosineAnnealingLR(optimizer, T_max=CONFIG['epochs'], eta_min=1e-6)
#     criterion = nn.MSELoss()
#     diffusion = DiffusionTrainer(model, CONFIG)
#     scaler = GradScaler('cuda' if torch.cuda.is_available() else 'cpu')
    
#     # ================= 🛡️ 新增：加载权重/继续训练逻辑 =================
#     start_epoch = 0
#     # 你可以手动指定路径，或者让它自动找最新的
#     checkpoint_path = os.path.join(model_dir, "unet_epoch_16.pth") 
    
#     if os.path.exists(checkpoint_path):
#         print(f"🔄 Found checkpoint: {checkpoint_path}, loading...")
#         checkpoint = torch.load(checkpoint_path, map_location=CONFIG['device'])
        
#         # 兼容性判断：判断存的是纯权重还是字典
#         if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
#             model.load_state_dict(checkpoint['model_state_dict'])
#             optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#             start_epoch = checkpoint['epoch']
#             print(f"✅ Loaded Dict: Resuming from Epoch {start_epoch + 1}")
#         else:
#             model.load_state_dict(checkpoint)
#             start_epoch = 16 # 如果是纯权重，手动指定从16开始
#             print(f"⚠️ Loaded Weights Only: Starting from Epoch {start_epoch + 1}")
#     # =============================================================

#     global_step = start_epoch * len(dataloader)
    
#     # 4. Loop
#     for epoch in range(start_epoch, CONFIG['epochs']):
#         model.train()
#         pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']}")
#         epoch_loss = 0
        
#         for step, batch in enumerate(pbar):
#             global_step += 1
            
#             x_0 = batch["GT"].to(CONFIG['device'], non_blocking=True)
#             condition = batch["Condition"].to(CONFIG['device'], non_blocking=True)
#             mask = batch["Mask"].to(CONFIG['device'], non_blocking=True)
#             porosity = batch["Porosity"].to(CONFIG['device'], non_blocking=True)
            
#             t = torch.randint(0, CONFIG['timesteps'], (x_0.shape[0],), device=CONFIG['device']).long()
            
#             with autocast('cuda'):
#                 x_noisy, noise = diffusion.add_noise(x_0, t)
#                 model_input = torch.cat([x_noisy, condition, mask], dim=1)
#                 noise_pred = model(model_input, t, porosity)
#                 loss = criterion(noise_pred, noise)
            
#             optimizer.zero_grad()
#             scaler.scale(loss).backward()

#             scaler.unscale_(optimizer)
#             torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

#             scaler.step(optimizer)
#             scaler.update()

#             # 每个 Epoch 结束后更新 LR
#             scheduler.step()
            
#             current_loss = loss.item()
#             epoch_loss += current_loss
#             pbar.set_postfix(loss=f"{current_loss:.4f}")
            
#             # 写入 CSV
#             if os.path.exists(csv_path):
#                 with open(csv_path, 'a', newline='') as f:
#                     writer = csv.writer(f)
#                     writer.writerow([epoch+1, global_step, f"{current_loss:.6f}", f"{optimizer.param_groups[0]['lr']:.8f}"])

#         avg_loss = epoch_loss / len(dataloader)
#         print(f"✅ Epoch {epoch+1} Done. Avg Loss: {avg_loss:.6f}")
        
#         # 5. 保存模型 (保存完整的 dict)
#         if (epoch + 1) % CONFIG['save_model_every'] == 0:
#             save_path = os.path.join(model_dir, f"unet_epoch_{epoch+1}.pth")
#             save_dict = {
#                 'epoch': epoch + 1,
#                 'model_state_dict': model.state_dict(),
#                 'optimizer_state_dict': optimizer.state_dict(),
#                 'loss': avg_loss,
#             }
#             # 注意：这里保存的是 save_dict，不再是 model.state_dict()
#             torch.save(save_dict, save_path)
#             print(f"💾 Full checkpoint saved: {save_path}")
            
#             # 调用绘图
#             try:
#                 plot_paper_curves(csv_path, log_dir)
#                 print(f"📈 Chart updated.")
#             except Exception as e:
#                 print(f"⚠️ Plotting failed: {e}")

# if __name__ == "__main__":
#     main()

# # ================= 训练主循环 =================
# def main():
#     print(f"🚀 Starting Stage 2 Training: {CONFIG['experiment_name']}")
#     model_dir, log_dir, csv_path = setup_experiment()
    
#     # 1. Dataset & DataLoader
#     dataset = LatentDataset(data_dir=CONFIG['processed_data_dir'])
#     dataloader = DataLoader(
#         dataset, 
#         batch_size=CONFIG['batch_size'], 
#         shuffle=True, 
#         num_workers=CONFIG['num_workers'], 
#         pin_memory=True
#     )
#     print(f"📦 Data loaded: {len(dataset)} samples.")

#     # 2. Model
#     model = ConditionalLatentUNet(
#         in_channels=CONFIG['in_channels'],       # 64+64+1
#         out_channels=CONFIG['out_channels'],
#         base_channels=CONFIG['base_channels'],     # A100 显存大，直接上 128
#         channel_mults=(1, 2, 4), # 通道变成 128 -> 256 -> 512
#         use_attention=(False, True, True) # 在中间层和最底层开启 Attention
#     ).to(CONFIG['device'])
    
#     # 3. Optimizer & Tools
#     optimizer = optim.AdamW(model.parameters(), lr=CONFIG['lr'])
#     criterion = nn.MSELoss()
#     diffusion = DiffusionTrainer(model, CONFIG)
#     scaler = GradScaler('cuda' if torch.cuda.is_available() else 'cpu')

#     # ================= 🛡️ 新增：加载权重/继续训练逻辑 =================
#     start_epoch = 0
#     # 你可以手动指定路径，或者让它自动找最新的
#     checkpoint_path = os.path.join(model_dir, "unet_epoch_16.pth")

#     if os.path.exists(checkpoint_path):
#         print(f"🔄 Found checkpoint: {checkpoint_path}, loading...")
#         checkpoint = torch.load(checkpoint_path, map_location=CONFIG['device'])
        
#         # 兼容性判断：判断存的是纯权重还是字典
#         if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
#             model.load_state_dict(checkpoint['model_state_dict'])
#             optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#             start_epoch = checkpoint['epoch']
#             print(f"✅ Loaded Dict: Resuming from Epoch {start_epoch + 1}")
#         else:
#             model.load_state_dict(checkpoint)
#             start_epoch = 16 # 如果是纯权重，手动指定从16开始
#             print(f"⚠️ Loaded Weights Only: Starting from Epoch {start_epoch + 1}")
#     # =============================================================
    
#     global_step = start_epoch * len(dataloader)
    
#     # 4. Loop
#     for epoch in range(CONFIG['epochs']):
#         model.train()
#         pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']}")
#         epoch_loss = 0
        
#         for step, batch in enumerate(pbar):
#             global_step += 1
            
#             # 搬运数据
#             x_0 = batch["GT"].to(CONFIG['device'], non_blocking=True)
#             condition = batch["Condition"].to(CONFIG['device'], non_blocking=True)
#             mask = batch["Mask"].to(CONFIG['device'], non_blocking=True)
#             porosity = batch["Porosity"].to(CONFIG['device'], non_blocking=True)
            
#             # 生成时间步
#             t = torch.randint(0, CONFIG['timesteps'], (x_0.shape[0],), device=CONFIG['device']).long()
            
#             # 混合精度训练
#             with autocast('cuda'):
#                 # 加噪
#                 x_noisy, noise = diffusion.add_noise(x_0, t)
#                 # 拼接输入 (Inpainting 模式: Noisy + Condition + Mask)
#                 model_input = torch.cat([x_noisy, condition, mask], dim=1)
#                 # 预测
#                 noise_pred = model(model_input, t, porosity)
#                 # Loss
#                 loss = criterion(noise_pred, noise)
            
#             # 反向传播
#             optimizer.zero_grad()
#             scaler.scale(loss).backward()

#             # ==========================================
#             # 🛡️ 【核心新增】 梯度裁剪 (Gradient Clipping)
#             # 在 scaler.step 之前执行
#             # max_norm=1.0 是扩散模型的标准操作
#             # ==========================================
#             scaler.unscale_(optimizer) # 必须先 unscale 才能 clip
#             torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

#             scaler.step(optimizer)
#             scaler.update()
            
#             # 记录
#             current_loss = loss.item()
#             epoch_loss += current_loss
#             pbar.set_postfix(loss=f"{current_loss:.4f}")
            
#             # 写入 CSV (实时)
#             with open(csv_path, 'a', newline='') as f:
#                 writer = csv.writer(f)
#                 writer.writerow([epoch+1, global_step, f"{current_loss:.6f}", f"{optimizer.param_groups[0]['lr']:.8f}"])

#         # Epoch 结束总结
#         avg_loss = epoch_loss / len(dataloader)
#         print(f"✅ Epoch {epoch+1} Done. Avg Loss: {avg_loss:.6f}")
        
#         # 保存模型 & 刷新图表
#         if (epoch + 1) % CONFIG['save_model_every'] == 0:
#             save_path = os.path.join(model_dir, f"unet_epoch_{epoch+1}.pth")
#             save_dict = {
#                 'epoch': epoch+1,
#                 'model_state_dict': model.state_dict(),
#                 'optimizer_state_dict': optimizer.state_dict(),
#                 'loss': avg_loss,
#             }            
#             torch.save(model.state_dict(), save_path)
#             print(f"💾 Model saved: {save_path}")
            
#             # 调用内置画图函数
#             plot_paper_curves(csv_path, log_dir)

# if __name__ == "__main__":
#     main()
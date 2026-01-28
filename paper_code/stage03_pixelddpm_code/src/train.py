import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from utils.get_root_path import get_root_path

from tqdm import tqdm
import json
import matplotlib.pyplot as plt
from torch.amp import autocast, GradScaler
from config import CONFIG
from diffusion_trainer import DiffusionTrainer
from dataset_rev import MaskedREVDataset
from model_unet import ConditionalUNet3D

# =========== [修改 1] 导入 csv 和 time ===========
import csv
import datetime
# ===============================================

# 1. 实验文件夹与配置保存（绝对路径）
def setup_experiment(PROJECT_ROOT: Path, config: dict) -> Path:
    # 1. 结果根目录：项目根/exp_results（绝对路径）
    results_root = PROJECT_ROOT / "exp_results" 
    results_root.mkdir(exist_ok=True) 
    
    # 2. 当前实验文件夹：项目根/exp_results/实验名
    exp_dir = results_root / config["experiment_name"]
    try:
        exp_dir.mkdir(exist_ok=False) 
    except FileExistsError:
        # 如果是为了Resume，这里可能需要允许存在，但在当前逻辑里通过修改config名控制
        # 如果你希望自动支持同文件夹续训，可以将 exist_ok 改为 True
        print(f"⚠️ 警告：实验文件夹已存在 {exp_dir} (如果是续训请忽略)")
        # raise FileExistsError(f"实验文件夹已存在！请修改experiment_name：{exp_dir}")
    
    # 3. 保存配置文件
    config_path = exp_dir / "config.json"
    # 如果文件夹已存在且我们想续训，就不必覆盖配置，除非想强制更新
    if not config_path.exists():
        with open(str(config_path), "w", encoding="utf-8") as f:
            json.dump(config, f, indent=4, ensure_ascii=False)
    
    # 4. 模型保存目录
    model_dir = exp_dir / config["model_output_dir"]
    model_dir.mkdir(exist_ok=True)

    # 5. 日志保存目录
    log_dir = exp_dir / config["log_output_dir"]
    log_dir.mkdir(exist_ok=True)
    
    print(f"\n{'='*50}")
    print(f"结果根目录（绝对路径）：{results_root}")
    print(f"实验文件夹（绝对路径）：{exp_dir}")
    print(f"配置文件（绝对路径）：{config_path}")
    print(f"模型保存目录（绝对路径）：{model_dir}")
    print(f"日志保存目录（绝对路径）：{log_dir}")
    print(f"{'='*50}\n")
    
    return results_root, exp_dir, config_path, model_dir, log_dir

def main_optimized_v2():
    # 配置项目的各种路径
    _, _, _, model_dir, log_dir = setup_experiment(get_root_path(), CONFIG)
    print(f"Starting optimized training V2 on {CONFIG['device']}...")
    
    # --- 优化 1: 开启 Benchmark ---
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        print("Cudnn Benchmark enabled.")

    # --- 优化 2: 显存与Batch策略 ---
    real_batch_size = CONFIG['batch_size'] 
    accumulation_steps = CONFIG['accumulation_steps'] 
    
    dataset = MaskedREVDataset(data_dir=CONFIG['processed_data_dir'], volume_size=CONFIG['image_size'], limit_size=CONFIG['limit_dataset_size'])
    
    num_workers = CONFIG['num_workers']
    
    dataloader = DataLoader(
        dataset, 
        batch_size=real_batch_size, 
        shuffle=True, 
        num_workers=num_workers,     
        pin_memory=True if torch.cuda.is_available() else False
    )
    print(f"DataLoader initialized. Workers: {num_workers}, Batch: {real_batch_size}")

    model = ConditionalUNet3D(in_channels=3, out_channels=1, base_channels=CONFIG['base_channels'])
    model.to(CONFIG['device'])

    # =========== 断点续训逻辑 ============
    RESUME_EPOCH = 88 
    RESUME_PATH = os.path.join(CONFIG['model_checkpoint_path'], f"unet_epoch_{RESUME_EPOCH}.pth")
    
    if os.path.exists(RESUME_PATH):
        print(f"🔄 Found checkpoint: {RESUME_PATH}")
        print(f"🔄 Loading weights and resuming from Epoch {RESUME_EPOCH+1}...")
        state_dict = torch.load(RESUME_PATH, map_location=CONFIG['device'])
        model.load_state_dict(state_dict)
        start_epoch = RESUME_EPOCH 
    else:
        print("⚠️ No checkpoint found. Starting from scratch.")
        start_epoch = 0
    
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['lr'])
    criterion = nn.MSELoss()
    diffusion = DiffusionTrainer(model, CONFIG)
    
    device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
    scaler = GradScaler(device_type)
    print(f"Mixed Precision Scaler initialized for {device_type}.")
    
    loss_history = []

    # =========== [修改 2] 初始化 CSV 文件 ===========
    csv_path = log_dir / "training_log.csv"
    csv_headers = ["Epoch", "Current_Loss", "Learning_Rate", "Timestamp"]
    
    # 如果是从头开始(start_epoch=0)，使用 'w' 覆盖；如果是续训，使用 'a' 追加
    write_mode = 'a' if start_epoch > 0 else 'w'
    
    # 只有当文件不存在 或者 我们是覆盖模式时，才写入表头
    should_write_header = (not csv_path.exists()) or (write_mode == 'w')
    
    with open(csv_path, mode=write_mode, newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if should_write_header:
            writer.writerow(csv_headers)
    
    print(f"📄 Logging training data to: {csv_path}")
    # ===============================================

    print("Start Training Loop...")
    for epoch in range(start_epoch, CONFIG['epochs']):
        model.train()
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']}")
        epoch_loss = 0
        optimizer.zero_grad() 
        
        for step, batch in enumerate(pbar):
            # A. 数据搬运
            x_0 = batch["GT"].to(CONFIG['device'], non_blocking=True)
            condition = batch["Condition"].to(CONFIG['device'], non_blocking=True)
            mask = batch["Mask"].to(CONFIG['device'], non_blocking=True)
            
            # B. 扩散采样
            current_batch_size = x_0.shape[0]
            t = torch.randint(0, CONFIG['timesteps'], (current_batch_size,), device=CONFIG['device']).long()
            
            # C. 混合精度训练
            with autocast(device_type):
                x_noisy, noise = diffusion.add_noise(x_0, t)
                model_input = torch.cat([x_noisy, condition, mask], dim=1)
                noise_pred = model(model_input, t)
                loss = criterion(noise_pred, noise)
                loss = loss / accumulation_steps

            # D. 反向传播
            scaler.scale(loss).backward()
            
            # 记录 Loss (还原为真实Loss大小)
            current_loss_val = loss.item() * accumulation_steps
            epoch_loss += current_loss_val
            pbar.set_postfix(loss=f"{current_loss_val:.4f}")

            # E. 梯度更新
            if (step + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            # 获取当前学习率
            current_lr = optimizer.param_groups[0]['lr']

            # =========== [修改 3] 写入 CSV ===========
            with open(csv_path, mode='a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                # 写入: [Epoch数, 平均Loss, 学习率, 时间戳]
                writer.writerow([epoch + 1, current_loss_val, current_lr, timestamp])
            # ========================================

        # 处理 epoch 末尾剩余梯度
        if len(dataloader) % accumulation_steps != 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        avg_loss = epoch_loss / len(dataloader)
        loss_history.append(avg_loss)
        
        print(f"Epoch {epoch+1} Average Loss: {avg_loss:.6f}")

        if (epoch + 1) % CONFIG['save_model_every'] == 0:
            save_path = os.path.join(model_dir, f"unet_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), save_path)
            print(f"Model saved to {save_path}")

    # 训练结束后绘图
    plt.figure(figsize=(10, 5))
    plt.plot(range(start_epoch + 1, start_epoch + 1 + len(loss_history)), loss_history, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(log_dir, "loss_curve.png"))
    print("Training finished.")

if __name__ == "__main__":
    main_optimized_v2()
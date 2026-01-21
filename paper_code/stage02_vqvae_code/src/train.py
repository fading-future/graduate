import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import json
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

# 自定义模块
from src.dataset_rev import VQVAEDataset 
from src.model_vqvae import VQVAE3D
from src.config import CONFIG
from utils.get_root_path import get_project_root

ROOT_DIR = get_project_root()

def smooth_curve(points, factor=0.9):
    """
    平滑曲线函数（用于 Tensorboard 风格的平滑）
    factor 越大越平滑 (0.0 - 0.99)
    """
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points

def plot_paper_curves(log_path, output_dir):
    """
    读取 CSV 并绘制适合论文的 Loss 曲线
    """
    try:
        df = pd.read_csv(log_path)
    except:
        return # 文件可能还不存在

    # 样式设置
    plt.style.use('default') # 或者 'seaborn-whitegrid'
    fig, ax1 = plt.subplots(figsize=(10, 6), dpi=300) # 高分辨率

    steps = df['Step']
    
    # --- 绘制 Loss (左轴) ---
    color_total = 'tab:red'
    color_recon = 'tab:blue'
    
    # 原始数据太抖，画淡一点作为背景
    ax1.plot(steps, df['Total_Loss'], color=color_total, alpha=0.15, linewidth=1)
    ax1.plot(steps, df['Recon_Loss'], color=color_recon, alpha=0.15, linewidth=1)
    
    # 平滑数据画深色实线
    # 窗口大小根据你的总步数调整，比如总步数几万，窗口可以是 100-500
    smooth_total = df['Total_Loss'].rolling(window=50, min_periods=1).mean()
    smooth_recon = df['Recon_Loss'].rolling(window=50, min_periods=1).mean()
    
    lns1 = ax1.plot(steps, smooth_total, color=color_total, label='Total Loss (Smoothed)', linewidth=2)
    lns2 = ax1.plot(steps, smooth_recon, color=color_recon, label='Recon Loss (Smoothed)', linewidth=1.5, linestyle='--')
    
    ax1.set_xlabel('Iteration Steps', fontsize=12)
    ax1.set_ylabel('Loss Value', fontsize=12)
    ax1.tick_params(axis='y')
    ax1.grid(True, linestyle='--', alpha=0.5)

    # --- 绘制 Perplexity (右轴，监控码本使用率) ---
    ax2 = ax1.twinx()  
    color_perp = 'tab:green'
    
    # 平滑 Perplexity
    smooth_perp = df['Perplexity'].rolling(window=50, min_periods=1).mean()
    lns3 = ax2.plot(steps, smooth_perp, color=color_perp, label='Perplexity (Codebook Usage)', linewidth=2)
    
    ax2.set_ylabel('Perplexity', fontsize=12, color=color_perp)
    ax2.tick_params(axis='y', labelcolor=color_perp)

    # 合并图例
    lns = lns1 + lns2 + lns3
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc='upper right', frameon=True, fancybox=True, shadow=True)

    plt.title('VQ-VAE Training Dynamics', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_curve_paper.png"))
    plt.close()

def setup_experiment():
    results_root = ROOT_DIR / "exp_results" 
    results_root.mkdir(exist_ok=True)
    
    exp_dir = results_root / CONFIG["experiment_name"]
    # 如果存在就报错，防止覆盖
    if exp_dir.exists():
         print(f"⚠️ 警告: 实验目录 {exp_dir} 已存在")
    
    exp_dir.mkdir(exist_ok=True)
    
    config_path = exp_dir / "config.json"
    with open(str(config_path), "w", encoding="utf-8") as f:
        json.dump(CONFIG, f, indent=4, ensure_ascii=False)
    
    model_dir = exp_dir / CONFIG["model_output_dir"]
    model_dir.mkdir(exist_ok=True)

    log_dir = exp_dir / CONFIG["log_output_dir"]
    log_dir.mkdir(exist_ok=True)
    
    log_csv_path = log_dir / "training_log.csv"
    if not log_csv_path.exists():
        with open(str(log_csv_path), 'w') as f:
            f.write("Epoch,Step,Total_Loss,Recon_Loss,VQ_Loss,Perplexity\n")

    return model_dir, log_dir, log_csv_path

def train_vqvae():
    # 1. 自动检测设备
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_count = torch.cuda.device_count()
        print(f"🚀 Training on {gpu_count} GPUs!")
    else:
        device = torch.device("cpu")
        print("⚠️ Warning: Training on CPU!")
        gpu_count = 0

    # 2. 准备实验目录
    save_dir, log_dir, log_csv_path = setup_experiment()

    # 3. 数据集
    dataset = VQVAEDataset(
        data_dir=CONFIG['processed_data_dir'], 
        volume_size=CONFIG['image_size'],
        augment=True
    )
    
    dataloader = DataLoader(
        dataset, 
        batch_size=CONFIG['batch_size'], 
        shuffle=True, 
        num_workers=CONFIG['num_workers'], 
        pin_memory=True
    )
    
    # 4. 模型初始化
    model = VQVAE3D(
        in_channels=1, 
        embedding_dim=CONFIG['embedding_dim'], 
        num_embeddings=CONFIG['num_embeddings']
    )
    
    # --- 关键修改：启用 DataParallel (DP) ---
    # 只要显卡数量 > 1，就自动用多卡
    if gpu_count > 1:
        model = nn.DataParallel(model)
    
    model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['lr'])
    
    global_step = 0
    
    # 5. 训练循环
    for epoch in range(CONFIG['epochs']):
        model.train()
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']}")
        
        for batch in pbar:
            global_step += 1
            
            # 手动把数据搬到 GPU (DataParallel 会自动切分分发)
            img = batch["GT"].to(device)
            
            optimizer.zero_grad()
            
            # Forward
            img_recon, vq_loss, perplexity = model(img)
            
            # Loss 计算 
            # DataParallel 返回的 Loss 是 [Batch_Size] 大小的向量，或者 [GPU_Count] 大小
            # 所以这里必须做一次 mean()，否则反向传播会报错
            recon_loss = torch.mean((img_recon - img)**2)
            vq_loss = vq_loss.mean() 
            perplexity = perplexity.mean()

            loss = recon_loss + vq_loss
            
            loss.backward()
            optimizer.step()
            
            # 记录日志
            with open(str(log_csv_path), 'a') as f:
                f.write(f"{epoch+1},{global_step},{loss.item():.6f},{recon_loss.item():.6f},{vq_loss.item():.6f},{perplexity.item():.6f}\n")

            pbar.set_postfix(
                Recon=f"{recon_loss.item():.4f}", 
                VQ=f"{vq_loss.item():.4f}",
                Perp=f"{perplexity.item():.1f}"
            )
            
        # 保存模型
        if (epoch+1) % 5 == 0:
            print(f"Epoch {epoch+1} Done. Saving model...")
            # 如果用了 DataParallel，真实模型在 model.module 里
            if isinstance(model, nn.DataParallel):
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()
                
            torch.save(state_dict, save_dir / f"vqvae_epoch_{epoch+1}.pth")
            
            # 简单的可视化 (为了避免多卡推理麻烦，我们简单地取一个样本做一次单次推理)
            with torch.no_grad():
                # 构造一个临时的单卡模型来跑这张图，或者直接用 model(img) 取第一个结果
                # 这里为了简单，直接取 batch 里的第一张图
                sample_input = img[0:1] # [1, 1, D, H, W]
                # model 依然是多卡的，它会处理这 1 个样本（虽然有点浪费）
                recon_out, _, _ = model(sample_input)
                
                save_visualization(sample_input, recon_out, epoch+1, log_dir)

def save_visualization(orig, recon, epoch, log_dir):
    with torch.no_grad():
        mid_slice_idx = orig.shape[2] // 2
        img_orig = orig[0, 0, mid_slice_idx].cpu().numpy()
        img_recon = recon[0, 0, mid_slice_idx].cpu().numpy()
        
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].imshow(img_orig, cmap='gray', vmin=-1, vmax=1)
        ax[0].set_title("Original")
        ax[0].axis('off')
        ax[1].imshow(img_recon, cmap='gray', vmin=-1, vmax=1)
        ax[1].set_title(f"Recon Epoch {epoch}")
        ax[1].axis('off')
        plt.tight_layout()
        plt.savefig(log_dir / f"viz_epoch_{epoch}.png")
        plt.close()

if __name__ == "__main__":
    train_vqvae()
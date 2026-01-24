import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  # 【新增】需要 pandas 来处理数据和保存 CSV

from src.dataset_rev import VQVAEDataset 
from src.model_vqvae import VQVAE3D
from src.config import CONFIG
from utils.get_root_path import get_project_root

DIR_ROOT = get_project_root()
EXP_ROOT = os.path.join(DIR_ROOT, CONFIG['experiment_name'])
MODEL_DIR = os.path.join(EXP_ROOT, CONFIG['model_output_dir'])
LOG_DIR = os.path.join(EXP_ROOT, CONFIG['log_output_dir'])

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

def train_vqvae():
    # 1. 准备目录
    save_dir = MODEL_DIR
    log_dir = LOG_DIR
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # 【新增】初始化 CSV 日志文件
    log_csv_path = os.path.join(log_dir, "training_log.csv")
    if not os.path.exists(log_csv_path):
        with open(log_csv_path, 'w') as f:
            f.write("Epoch,Step,Total_Loss,Recon_Loss,VQ_Loss,Perplexity\n")

    # 2. 数据集
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
    
    print(f"Training on Device: {CONFIG['device']}")
    
    # 3. 模型
    model = VQVAE3D(
        in_channels=1, 
        embedding_dim=CONFIG['embedding_dim'], 
        num_embeddings=CONFIG['num_embeddings']
    ).to(CONFIG['device'])
    
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['lr'])
    
    # 全局步数计数器
    global_step = 0
    
    # 4. 训练循环
    for epoch in range(CONFIG['epochs']):
        model.train()
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']}")
        for batch in pbar:
            global_step += 1
            
            img = batch["GT"].to(CONFIG['device']) 
            
            optimizer.zero_grad()
            img_recon, vq_loss, perplexity = model(img)
            
            recon_loss = torch.mean((img_recon - img)**2)
            loss = recon_loss + vq_loss
            
            loss.backward()
            optimizer.step()
            
            # 【核心修改：实时写入 CSV】
            # 每一步都记录，这样你的曲线数据点就非常密集，平滑后很好看
            with open(log_csv_path, 'a') as f:
                f.write(f"{epoch+1},{global_step},{loss.item():.6f},{recon_loss.item():.6f},{vq_loss.item():.6f},{perplexity.item():.6f}\n")

            pbar.set_postfix(
                Recon=f"{recon_loss.item():.4f}", 
                VQ=f"{vq_loss.item():.4f}",
                Perp=f"{perplexity.item():.1f}"
            )
            
        print(f"Epoch {epoch+1} Done.")
        
        # 保存模型
        if (epoch+1) % 5 == 0:
            torch.save(model.state_dict(), os.path.join(save_dir, f"vqvae_epoch_{epoch+1}.pth"))
            save_visualization(img, img_recon, epoch+1, log_dir)
            
            # 【每5个epoch刷新一次 Loss 曲线图】
            plot_paper_curves(log_csv_path, log_dir)

def save_visualization(orig, recon, epoch, log_dir):
    """保存中间切片对比图（保持原样即可）"""
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
        plt.savefig(os.path.join(log_dir, f"viz_epoch_{epoch}.png"))
        plt.close()

if __name__ == "__main__":
    train_vqvae()
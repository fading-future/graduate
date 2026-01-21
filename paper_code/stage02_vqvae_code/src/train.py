import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from accelerate import Accelerator
from accelerate.utils import set_seed

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
    # 注意：在多卡模式下，只有主进程需要创建文件夹
    results_root = ROOT_DIR / "exp_results"
    exp_dir = results_root / CONFIG["experiment_name"]
    model_dir = exp_dir / CONFIG["model_output_dir"]
    log_dir = exp_dir / CONFIG["log_output_dir"]
    config_path = exp_dir / "config.json"

    # 只让主进程创建目录
    # 我们会在 train_vqvae 里判断 is_main_process 再调用这个
    return results_root, exp_dir, config_path, model_dir, log_dir

def train_vqvae():
    # 1. 初始化 Accelerator
    # mixed_precision='fp16' 或 'bf16' (A100 建议用 bf16，如果代码报错则改回 fp16)
    accelerator = Accelerator(mixed_precision="bf16", log_with="all")

    # 设置随机种子，保证多卡同步
    set_seed(42)

    # 只有主进程负责创建目录和日志
    if accelerator.is_main_process:
        results_root, exp_dir, config_path, save_dir, log_dir = setup_experiment()
        # 创建目录
        results_root.mkdir(exist_ok=True)
        try:
            exp_dir.mkdir(exist_ok=False) 
        except FileExistsError:
            print(f"警告：实验目录 {exp_dir} 已存在。")
            
        model_dir = save_dir # setup_experiment 返回的是 Path
        model_dir.mkdir(exist_ok=True, parents=True)
        log_dir.mkdir(exist_ok=True, parents=True)

        # 保存配置
        with open(str(config_path), "w", encoding="utf-8") as f:
            json.dump(CONFIG, f, indent=4, ensure_ascii=False)
        
        # 初始化 CSV
        log_csv_path = os.path.join(log_dir, "training_log.csv")
        with open(log_csv_path, 'w') as f:
            f.write("Epoch,Step,Total_Loss,Recon_Loss,VQ_Loss,Perplexity\n")
    
    # 等待主进程创建好目录，防止其他进程报错
    accelerator.wait_for_everyone()
    
    # 如果不是主进程，变量需要定义一下防止报错，虽然用不到
    if not accelerator.is_main_process:
        save_dir = None
        log_dir = None
        log_csv_path = None

    # 2. 数据集
    dataset = VQVAEDataset(
        data_dir=CONFIG['processed_data_dir'], 
        volume_size=CONFIG['image_size'],
        augment=True
    )
    
    # 注意：accelerate 会自动处理多卡采样，shuffle=True 即可
    dataloader = DataLoader(
        dataset, 
        batch_size=CONFIG['batch_size'], 
        shuffle=True, 
        num_workers=CONFIG['num_workers'], 
        pin_memory=True
    )
    
    # 3. 模型
    model = VQVAE3D(
        in_channels=1, 
        embedding_dim=CONFIG['embedding_dim'], 
        num_embeddings=CONFIG['num_embeddings']
    )
    # 注意：不要手动 .to(device)，也不要手动 nn.DataParallel
    
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['lr'])
    
    # --- 关键步骤：Prepare ---
    # accelerate 会自动把模型、优化器、数据加载器分配到不同的 GPU 上
    model, optimizer, dataloader = accelerator.prepare(
        model, optimizer, dataloader
    )
    
    print(f"Process {accelerator.process_index} ready.")

    global_step = 0
    
    # 4. 训练循环
    for epoch in range(CONFIG['epochs']):
        model.train()
        
        # 只有主进程显示进度条
        if accelerator.is_main_process:
            pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']}", disable=False)
        else:
            pbar = dataloader # 其他进程静默运行
            
        for batch in pbar:
            global_step += 1
            
            # 不需要 batch["GT"].to(device)，accelerator 已处理 dataloader
            img = batch["GT"]
            
            # --- Forward ---
            # 这里的 model 已经被 wrap 成了 DDP 模型
            img_recon, vq_loss, perplexity = model(img)
            
            recon_loss = torch.mean((img_recon - img)**2)
            loss = recon_loss + vq_loss
            
            # --- Backward (修改点) ---
            optimizer.zero_grad()
            accelerator.backward(loss) # 代替 loss.backward()
            optimizer.step()
            
            # --- Logging (只在主进程) ---
            if accelerator.is_main_process:
                # 记录到 CSV
                with open(log_csv_path, 'a') as f:
                    f.write(f"{epoch+1},{global_step},{loss.item():.6f},{recon_loss.item():.6f},{vq_loss.item():.6f},{perplexity.item():.6f}\n")

                pbar.set_postfix(
                    Recon=f"{recon_loss.item():.4f}", 
                    VQ=f"{vq_loss.item():.4f}",
                    Perp=f"{perplexity.item():.1f}"
                )
            
        # --- 保存模型 (只在主进程) ---
        # 等待所有 GPU 跑完这个 epoch
        accelerator.wait_for_everyone()
        
        if accelerator.is_main_process and (epoch+1) % 5 == 0:
            print(f"Epoch {epoch+1} Done. Saving model...")
            # unwrap_model 取出原始模型，防止保存 DDP 包装层的参数前缀
            unwrapped_model = accelerator.unwrap_model(model)
            torch.save(unwrapped_model.state_dict(), os.path.join(save_dir, f"vqvae_epoch_{epoch+1}.pth"))
            
            # 保存可视化图 (需要取一个样本)
            # 这是一个小技巧：我们需要构造一个简单的推理过程来画图
            unwrapped_model.eval()
            with torch.no_grad():
                # 从当前 batch 取一张图画一下即可，注意要把它转到 CPU
                sample_img = img[0:1].detach() # [1, 1, D, H, W]
                # 因为 unwrapped_model 在 GPU 上，所以可以直接推理
                recon_img, _, _ = unwrapped_model(sample_img)
                save_visualization(sample_img, recon_img, epoch+1, log_dir)
            
            # 绘制曲线
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

    # 1. 准备目录,初始化 CSV 日志文件
    results_root, exp_dir, config_path, save_dir, log_dir = setup_experiment()
    
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

            plot_paper_curves(log_csv_path, log_dir)

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 引入上面的模块
from src.dataset_rev import VQVAEDataset 
from src.model_vqvae_v2 import VQVAE3D
from src.config import CONFIG
from utils.get_root_path import get_project_root

DIR_ROOT = get_project_root()
EXP_ROOT = os.path.join(DIR_ROOT, CONFIG['experiment_name'])
MODEL_DIR = os.path.join(EXP_ROOT, CONFIG['model_output_dir'])
LOG_DIR = os.path.join(EXP_ROOT, CONFIG['log_output_dir'])

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

RESUME_WEIGHT_PATH = os.path.join(MODEL_DIR, "vqvae_finetune_epoch_2.pth") 

def train_vqvae_finetune():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    log_csv_path = os.path.join(LOG_DIR, "training_log_finetune.csv")
    
    # Dataset ... (保持不变)
    dataset = VQVAEDataset(
        data_dir=CONFIG['processed_data_dir'], 
        volume_size=CONFIG['image_size'],
        augment=True
    )
    dataloader = DataLoader(dataset, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=4, pin_memory=True)
    
    # Model ... (保持不变)
    model = VQVAE3D(
        in_channels=1, 
        embedding_dim=CONFIG['embedding_dim'], 
        num_embeddings=CONFIG['num_embeddings']
    ).to(CONFIG['device'])
    
    # --- 加载旧权重逻辑 (Seamless Resume) ---
    start_epoch = 0
    if os.path.exists(RESUME_WEIGHT_PATH):
        print(f"Loading weights from {RESUME_WEIGHT_PATH}...")
        state_dict = torch.load(RESUME_WEIGHT_PATH, map_location=CONFIG['device'])
        model.load_state_dict(state_dict, strict=False) # strict=False 兼容我们刚改的属性
        
        # 尝试从文件名推断当前是第几个 epoch (可选)
        try:
            # 假设文件名是 vqvae_finetune_epoch_5.pth
            start_epoch = int(RESUME_WEIGHT_PATH.split('_')[-1].split('.')[0])
            print(f"Resuming from Epoch {start_epoch}")
        except:
            print("Could not infer epoch from filename, starting from 0 (display only).")
            
    else:
        print("No checkpoint found. Please check RESUME_WEIGHT_PATH.")
        return # 如果找不到权重，直接退出，防止误操作从头跑

    # 初始学习率
    base_lr = 1e-4
    optimizer = optim.Adam(model.parameters(), lr=base_lr)
    
    # 全局步数 (如果是追加写入 CSV，最好读取一下 CSV 最后的 Step，这里简化处理)
    global_step = 0 
    
    # 总共计划跑 20 个 epoch (包含之前已经跑过的)
    TOTAL_EPOCHS = 20
    
    # --- 训练循环 ---
    for epoch in range(start_epoch, TOTAL_EPOCHS): 
        current_epoch_display = epoch + 1
        
        # ================= Stage 控制逻辑 (核心) =================
        
        # Stage 2: 冷却期 (Epoch 6 - 15) -> 关闭重启
        if current_epoch_display >= 6:
            model.quantizer.restart_threshold = -1.0 # 关闭重启
            stage_msg = "[Stage 2: Cooldown - Restart OFF]"
        else:
            model.quantizer.restart_threshold = 0.1  # 开启重启
            stage_msg = "[Stage 1: Exploration - Restart ON]"
            
        # Stage 3: 精调期 (Epoch 16 - 20) -> 降低 LR
        if current_epoch_display >= 16:
            new_lr = base_lr * 0.1 # 1e-5
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lr
            stage_msg = f"[Stage 3: Finetuning - LR {new_lr}]"
        
        print(f"\n=== Start Epoch {current_epoch_display} {stage_msg} ===")
        # ========================================================

        model.train()
        pbar = tqdm(dataloader, desc=f"Ep {current_epoch_display}")
        
        for batch in pbar:
            global_step += 1
            img = batch["GT"].to(CONFIG['device'])
            
            optimizer.zero_grad()
            img_recon, vq_loss, perplexity = model(img)
            
            recon_loss = torch.mean((img_recon - img)**2)
            loss = recon_loss + vq_loss
            
            loss.backward()
            optimizer.step()
            
            # Logging
            with open(log_csv_path, 'a') as f:
                f.write(f"{current_epoch_display},{global_step},{loss.item():.6f},{recon_loss.item():.6f},{vq_loss.item():.6f},{perplexity.item():.6f}\n")

            pbar.set_postfix(
                Recon=f"{recon_loss.item():.4f}", 
                Perp=f"{perplexity.item():.1f}",
                Stage=stage_msg.split(':')[0] # 简短显示
            )
            
        # 保存
        torch.save(model.state_dict(), os.path.join(MODEL_DIR, f"vqvae_finetune_epoch_{current_epoch_display}.pth"))
        save_visualization(img, img_recon, current_epoch_display, LOG_DIR)

if __name__ == "__main__":
    train_vqvae_finetune()

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from tqdm import tqdm
import os
import json
import warnings
import matplotlib.pyplot as plt
import pandas as pd
import math
from accelerate import Accelerator
from accelerate.utils import set_seed

from src.dataset_rev import VQVAEDataset 
from src.model_vqvae import VQVAE3D
from src.config import CONFIG
from utils.get_root_path import get_project_root

warnings.filterwarnings("ignore", message="Can't initialize NVML")
ROOT_DIR = get_project_root()

# --- 绘图工具函数 ---
def plot_paper_curves(log_path, output_dir):
    try:
        df = pd.read_csv(log_path)
    except:
        return 
    
    plt.style.use('default')
    fig, ax1 = plt.subplots(figsize=(10, 6), dpi=300)
    steps = df['Step']
    
    color_total = 'tab:red'
    color_recon = 'tab:blue'
    
    ax1.plot(steps, df['Total_Loss'], color=color_total, alpha=0.15, linewidth=1)
    ax1.plot(steps, df['Recon_Loss'], color=color_recon, alpha=0.15, linewidth=1)
    
    smooth_total = df['Total_Loss'].rolling(window=50, min_periods=1).mean()
    smooth_recon = df['Recon_Loss'].rolling(window=50, min_periods=1).mean()
    
    lns1 = ax1.plot(steps, smooth_total, color=color_total, label='Total Loss (Smoothed)', linewidth=2)
    lns2 = ax1.plot(steps, smooth_recon, color=color_recon, label='Recon Loss (Smoothed)', linewidth=1.5, linestyle='--')
    
    ax1.set_xlabel('Iteration Steps', fontsize=12)
    ax1.set_ylabel('Loss Value', fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.5)

    ax2 = ax1.twinx()  
    color_perp = 'tab:green'
    smooth_perp = df['Perplexity'].rolling(window=50, min_periods=1).mean()
    lns3 = ax2.plot(steps, smooth_perp, color=color_perp, label='Perplexity', linewidth=2)
    ax2.set_ylabel('Perplexity', fontsize=12, color=color_perp)

    lns = lns1 + lns2 + lns3
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc='upper right')
    
    plt.title('VQ-VAE Training Dynamics', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_curve_paper.png"))
    plt.close()

def save_visualization(orig, recon, epoch, log_dir):
    with torch.no_grad():
        mid_slice_idx = orig.shape[2] // 2
        img_orig = orig[0, 0, mid_slice_idx].cpu().float().numpy() # Ensure float
        img_recon = recon[0, 0, mid_slice_idx].cpu().float().numpy()
        
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

def setup_experiment():
    results_root = ROOT_DIR / "exp_results"
    exp_dir = results_root / CONFIG["experiment_name"]
    model_dir = exp_dir / CONFIG["model_output_dir"]
    log_dir = exp_dir / CONFIG["log_output_dir"]
    config_path = exp_dir / "config.json"
    return results_root, exp_dir, config_path, model_dir, log_dir

def train_vqvae():
    accelerator = Accelerator(mixed_precision="bf16", log_with="all", gradient_accumulation_steps=4)
    set_seed(42)

    if accelerator.is_main_process:
        results_root, exp_dir, config_path, save_dir, log_dir = setup_experiment()
        results_root.mkdir(exist_ok=True)
        try:
            exp_dir.mkdir(exist_ok=False) 
        except FileExistsError:
            print(f"Warning: Directory {exp_dir} exists. Cleaning up...")
            import shutil
            shutil.rmtree(exp_dir) # 自动清理，防止你忘了删
            exp_dir.mkdir(exist_ok=False)

        model_dir = save_dir 
        model_dir.mkdir(exist_ok=True, parents=True)
        log_dir.mkdir(exist_ok=True, parents=True)

        with open(str(config_path), "w", encoding="utf-8") as f:
            json.dump(CONFIG, f, indent=4, ensure_ascii=False)
        
        log_csv_path = os.path.join(log_dir, "training_log.csv")
        with open(log_csv_path, 'w') as f:
            f.write("Epoch,Step,Total_Loss,Recon_Loss,VQ_Loss,Perplexity,LR\n")
    
    accelerator.wait_for_everyone()
    
    if not accelerator.is_main_process:
        save_dir = None
        log_dir = None
        log_csv_path = None

    dataset = VQVAEDataset(data_dir=CONFIG['processed_data_dir'], volume_size=CONFIG['image_size'], augment=True)
    dataloader = DataLoader(dataset, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=CONFIG['num_workers'], pin_memory=True)
    
    model = VQVAE3D(in_channels=1, embedding_dim=CONFIG['embedding_dim'], num_embeddings=CONFIG['num_embeddings'])
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['lr'])

    # Scheduler Setup
    num_epochs = CONFIG['epochs']
    grad_accum_steps = accelerator.gradient_accumulation_steps
    steps_per_epoch = math.ceil(len(dataloader) / grad_accum_steps)
    warmup_epochs = 10 
    warmup_steps = warmup_epochs * steps_per_epoch
    total_steps = num_epochs * steps_per_epoch 

    scheduler_warmup = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps)
    scheduler_cosine = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps, eta_min=1e-6)
    scheduler = SequentialLR(optimizer, schedulers=[scheduler_warmup, scheduler_cosine], milestones=[warmup_steps])
    
    model, optimizer, dataloader, scheduler = accelerator.prepare(model, optimizer, dataloader, scheduler)
    
    if accelerator.is_main_process:
        print(f"Start Training. Total Steps: {total_steps}. Device: {accelerator.device}")

    global_step = 0
    
    for epoch in range(num_epochs):
        model.train()
        if accelerator.is_main_process:
            pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", disable=False)
        else:
            pbar = dataloader

        for batch in pbar:
            with accelerator.accumulate(model):
                img = batch["GT"]
                
                # =======================================================
                # 【核清洗】最后一道防线：不管 Dataset 做了什么，
                # 训练前强制把 Batch 里所有的 NaN 和 Inf 全部干掉！
                # 这样计算 Loss 时，Target (img) 就绝对是干净的了。
                # =======================================================
                if torch.isnan(img).any() or torch.isinf(img).any():
                    # 只有主进程打印警告，避免刷屏
                    if accelerator.is_main_process and global_step % 10 == 0:
                        print(f"Warning: Batch {global_step} contains NaN/Inf! Sanitizing inside training loop...")
                    
                    # 将 NaN 变 0，正无穷变 1，负无穷变 -1 (因为我们归一化范围是 -1~1)
                    img = torch.nan_to_num(img, nan=0.0, posinf=1.0, neginf=-1.0)
                # =======================================================

                # 现在的 img 是 100% 干净的，进去跑模型
                img_recon, vq_loss, perplexity = model(img)
                
                # 计算 Loss：干净 - 干净 = 干净！
                recon_loss = torch.mean((img_recon - img)**2)
                loss = recon_loss + vq_loss
                
                optimizer.zero_grad()
                accelerator.backward(loss)
                
                # 梯度裁剪 (保持不动)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                    
                optimizer.step()
                
                if accelerator.sync_gradients:
                    scheduler.step()
                    global_step += 1

            if accelerator.is_main_process and accelerator.sync_gradients:
                current_lr = scheduler.get_last_lr()[0]
                
                # 检查是否出现 NaN
                if math.isnan(loss.item()):
                    print(f"!!! CRITICAL WARNING: Loss is NaN at step {global_step} !!!")
                    print(f"Recon: {recon_loss.item()}, VQ: {vq_loss.item()}")
                    # 遇到 NaN 强行退出，方便 debug，而不是空转
                    # raise ValueError("Training diverged with NaN.") 

                with open(log_csv_path, 'a') as f:
                    f.write(f"{epoch+1},{global_step},{loss.item():.6f},{recon_loss.item():.6f},{vq_loss.item():.6f},{perplexity.item():.6f},{current_lr:.8f}\n")

                pbar.set_postfix(Recon=f"{recon_loss.item():.4f}", VQ=f"{vq_loss.item():.4f}", Perp=f"{perplexity.item():.1f}", LR=f"{current_lr:.6f}")
            
        accelerator.wait_for_everyone()
        
        if accelerator.is_main_process and (epoch+1) % 2 == 0:
            unwrapped_model = accelerator.unwrap_model(model)
            torch.save(unwrapped_model.state_dict(), os.path.join(save_dir, f"vqvae_epoch_{epoch+1}.pth"))
            
            unwrapped_model.eval()
            with torch.no_grad():
                sample_img = img[0:1].detach().float()
                recon_img, _, _ = unwrapped_model(sample_img)
                save_visualization(sample_img, recon_img, epoch+1, log_dir)
            
            plot_paper_curves(log_csv_path, log_dir)

if __name__ == "__main__":
    train_vqvae()
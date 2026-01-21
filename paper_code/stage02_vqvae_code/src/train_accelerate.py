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

# --- 新增引入 ---
from accelerate import Accelerator
from accelerate.utils import set_seed

# 自定义模块
from src.dataset_rev import VQVAEDataset 
from src.model_vqvae import VQVAE3D
from src.config import CONFIG
from utils.get_root_path import get_project_root

# 过滤掉包含特定关键词的警告
warnings.filterwarnings("ignore", message="Can't initialize NVML")

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
    accelerator = Accelerator(
        mixed_precision="bf16", 
        log_with="all",
        gradient_accumulation_steps=4 # <--- 建议在这里显式指定，或者在 launch 命令中指定
    )
    
    set_seed(42)

    # 只有主进程负责创建目录和日志
    if accelerator.is_main_process:
        results_root, exp_dir, config_path, save_dir, log_dir = setup_experiment()
        results_root.mkdir(exist_ok=True)
        try:
            exp_dir.mkdir(exist_ok=False) 
        except FileExistsError:
            print(f"警告：实验目录 {exp_dir} 已存在。")
        model_dir = save_dir 
        model_dir.mkdir(exist_ok=True, parents=True)
        log_dir.mkdir(exist_ok=True, parents=True)

        with open(str(config_path), "w", encoding="utf-8") as f:
            json.dump(CONFIG, f, indent=4, ensure_ascii=False)
        
        # --- 修改 Log Header: 增加 LR 列 ---
        log_csv_path = os.path.join(log_dir, "training_log.csv")
        with open(log_csv_path, 'w') as f:
            f.write("Epoch,Step,Total_Loss,Recon_Loss,VQ_Loss,Perplexity,LR\n")
    
    accelerator.wait_for_everyone()
    
    if not accelerator.is_main_process:
        save_dir = None
        log_dir = None
        log_csv_path = None

    # 2. 数据集 & DataLoader
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
    
    # 3. 模型 & 优化器
    model = VQVAE3D(
        in_channels=1, 
        embedding_dim=CONFIG['embedding_dim'], 
        num_embeddings=CONFIG['num_embeddings']
    )
    
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['lr'])

    # --- 新增: 学习率调度器 (Warmup + Cosine) ---
    
    import math # 确保引入 math 库
    
    num_epochs = CONFIG['epochs']
    
    # === 核心修正开始 ===
    # 获取累积步数 (例如 4)
    grad_accum_steps = accelerator.gradient_accumulation_steps
    
    # 计算实际的 "优化步数" (Optimization Steps)
    # 逻辑：总 Batch 数 / 累积步数 = 实际参数更新次数
    # 如果不除以这个，Scheduler 会以为步数很多，导致 LR 降不下去
    steps_per_epoch = math.ceil(len(dataloader) / grad_accum_steps)
    # === 核心修正结束 ===
    
    # 定义 Warmup 的 Epoch 数 (例如总 Epoch 的 5% 或固定 10 个)
    warmup_epochs = 10 
    warmup_steps = warmup_epochs * steps_per_epoch
    total_steps = num_epochs * steps_per_epoch # 总迭代次数 (iteration)

    # 打印一下确认算对了没
    if accelerator.is_main_process:
        print(f"DEBUG: Batches per epoch: {len(dataloader)}")
        print(f"DEBUG: Accumulation steps: {grad_accum_steps}")
        print(f"DEBUG: Actual optimization steps per epoch: {steps_per_epoch}")
        print(f"DEBUG: Total optimization steps: {total_steps}")

    # 1. 预热调度器: 从 lr * 0.01 线性增加到 lr
    scheduler_warmup = LinearLR(
        optimizer, 
        start_factor=0.01, 
        end_factor=1.0, 
        total_iters=warmup_steps
    )

    # 2. 余弦退火调度器: 从 lr 降到 min_lr (这里设为 0 或者 1e-6)
    scheduler_cosine = CosineAnnealingLR(
        optimizer, 
        T_max=total_steps - warmup_steps, 
        eta_min=1e-6
    )

    # 3. 串联起来
    scheduler = SequentialLR(
        optimizer, 
        schedulers=[scheduler_warmup, scheduler_cosine], 
        milestones=[warmup_steps]
    )
    # --------------------------------------------
    
    # --- Prepare (加入 scheduler) ---
    model, optimizer, dataloader, scheduler = accelerator.prepare(
        model, optimizer, dataloader, scheduler
    )
    
    print(f"Process {accelerator.process_index} ready. Total training steps: {total_steps}")

    global_step = 0 # 记录总 Update 次数
    
    # 4. 训练循环
    for epoch in range(num_epochs):
        model.train()
        
        if accelerator.is_main_process:
            pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", disable=False)
        else:
            pbar = dataloader

        for batch in pbar:
            # 使用 accumulate 上下文
            with accelerator.accumulate(model):
                img = batch["GT"]
                img_recon, vq_loss, perplexity = model(img)
                
                recon_loss = torch.mean((img_recon - img)**2)
                loss = recon_loss + vq_loss
                
                optimizer.zero_grad()
                accelerator.backward(loss)
                optimizer.step()
                
                # --- 关键：Scheduler Step ---
                # 只有在梯度真正更新的时候（sync_gradients=True）才更新 LR
                # 这样可以正确处理 Gradient Accumulation
                if accelerator.sync_gradients:
                    scheduler.step()
                    global_step += 1

            # --- Logging ---
            if accelerator.is_main_process and accelerator.sync_gradients:
                # 获取当前 LR
                current_lr = scheduler.get_last_lr()[0]
                
                # 写入 CSV
                with open(log_csv_path, 'a') as f:
                    f.write(f"{epoch+1},{global_step},{loss.item():.6f},{recon_loss.item():.6f},{vq_loss.item():.6f},{perplexity.item():.6f},{current_lr:.8f}\n")

                pbar.set_postfix(
                    Recon=f"{recon_loss.item():.4f}", 
                    VQ=f"{vq_loss.item():.4f}",
                    Perp=f"{perplexity.item():.1f}",
                    LR=f"{current_lr:.6f}"
                )
            
        # --- 保存模型 ---
        accelerator.wait_for_everyone()
        
        if accelerator.is_main_process and (epoch+1) % 5 == 0:
            print(f"Epoch {epoch+1} Done. Saving model...")
            unwrapped_model = accelerator.unwrap_model(model)
            torch.save(unwrapped_model.state_dict(), os.path.join(save_dir, f"vqvae_epoch_{epoch+1}.pth"))
            
            # 可视化
            unwrapped_model.eval()
            with torch.no_grad():
                sample_img = img[0:1].detach().float() # 确保类型匹配
                # 这里的 unwrapped_model 在 GPU 上，直接推
                recon_img, _, _ = unwrapped_model(sample_img)
                save_visualization(sample_img, recon_img, epoch+1, log_dir)
            
            # 绘制曲线
            plot_paper_curves(log_csv_path, log_dir)

# --- 把 save_visualization 函数移到 train_vqvae 后面，主程序运行之前 ---
def save_visualization(orig, recon, epoch, log_dir):
    """保持不变"""
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

# --- 确保只有一个入口点，且在最后 ---
if __name__ == "__main__":
    train_vqvae()
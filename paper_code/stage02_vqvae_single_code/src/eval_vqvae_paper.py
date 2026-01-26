import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from tqdm import tqdm
import pandas as pd

# 引入你的项目模块
from src.dataset_rev import VQVAEDataset
from src.model_vqvae_v2 import VQVAE3D # 确保这是你最新的带restart的模型文件
from src.config import CONFIG
from utils.get_root_path import get_project_root

# ================= 配置区 =================
# 评估哪一个权重？建议选 Epoch 20 的最终权重
CHECKPOINT_PATH = os.path.join(get_project_root(), CONFIG['experiment_name'], CONFIG['model_output_dir'], "vqvae_finetune_epoch_2.pth")
OUTPUT_DIR = os.path.join(get_project_root(), CONFIG['experiment_name'], "evaluation_results")
DEVICE = CONFIG['device']
BATCH_SIZE = 1 # 评估时建议 Batch Size = 1，方便逐个分析
# =========================================

def setup_plotting_style():
    """设置论文级别的绘图风格 (Times New Roman + 高级配色)"""
    sns.set_theme(style="whitegrid")
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['savefig.dpi'] = 300

def calculate_metrics(vol_true, vol_recon):
    """
    计算单个 3D 体数据的指标
    vol_true, vol_recon: numpy array, range [0, 1]
    """
    # 1. MSE
    mse = np.mean((vol_true - vol_recon) ** 2)
    
    # 2. PSNR
    # data_range=1.0 因为我们要把数据归一化到 0-1
    val_psnr = psnr(vol_true, vol_recon, data_range=1.0)
    
    # 3. SSIM (3D)
    # win_size 必须小于图像最小边长，channel_axis=None 表示输入是 (D,H,W)
    val_ssim = ssim(vol_true, vol_recon, data_range=1.0, channel_axis=None)
    
    # 4. 物理属性：孔隙率/平均密度误差 (Porosity Error)
    # 假设归一化后，值越小越接近孔隙，值越大越接近骨架
    # 这里简单比较平均密度，作为孔隙率的代理指标
    porosity_true = np.mean(vol_true)
    porosity_recon = np.mean(vol_recon)
    porosity_error = abs(porosity_true - porosity_recon) / (porosity_true + 1e-8)
    
    return mse, val_psnr, val_ssim, porosity_error

def evaluate():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    setup_plotting_style()
    
    print(f"Loading model from {CHECKPOINT_PATH}...")
    
    # 1. 加载模型
    model = VQVAE3D(
        in_channels=1, 
        embedding_dim=CONFIG['embedding_dim'], 
        num_embeddings=CONFIG['num_embeddings']
    ).to(DEVICE)
    
    state_dict = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    
    # 2. 加载数据
    dataset = VQVAEDataset(
        data_dir=CONFIG['processed_data_dir'], 
        volume_size=CONFIG['image_size'],
        augment=False # 评估时关闭增强！我们要看原汁原味的效果
    )
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    metrics_list = []
    
    print("Starting evaluation loop...")
    
    # 用于可视化的样本
    vis_sample_true = None
    vis_sample_recon = None
    
    # 3. 循环计算
    with torch.no_grad():
        for i, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
            img = batch["GT"].to(DEVICE)
            
            # Forward
            img_recon, _, _ = model(img)
            
            # 转为 Numpy 并归一化到 [0, 1] 用于指标计算
            # 原始数据是 [-1, 1]
            vol_true = (img.cpu().numpy().squeeze() + 1.0) / 2.0
            vol_recon = (img_recon.cpu().numpy().squeeze() + 1.0) / 2.0
            
            # 防止越界
            vol_true = np.clip(vol_true, 0, 1)
            vol_recon = np.clip(vol_recon, 0, 1)
            
            # 计算指标
            mse, p, s, poro_err = calculate_metrics(vol_true, vol_recon)
            
            metrics_list.append({
                "Sample_ID": i,
                "MSE": mse,
                "PSNR": p,
                "SSIM": s,
                "Porosity_Error": poro_err * 100 # 转百分比
            })
            
            # 保存第一个样本用于画图
            if i == 0:
                vis_sample_true = vol_true
                vis_sample_recon = vol_recon
                # 同时也获取一下码本的使用情况用于画图
                # 我们需要重新跑一次 encode 获取 indices
                _, _, _, indices = model.quantizer(model.encoder(img))
                code_indices = indices.cpu().numpy().flatten()

    # 4. 保存 CSV 结果
    df = pd.DataFrame(metrics_list)
    csv_path = os.path.join(OUTPUT_DIR, "quantitative_metrics.csv")
    df.to_csv(csv_path, index=False)
    
    print(f"\nEvaluation Complete!")
    print(f"Average PSNR: {df['PSNR'].mean():.2f} +/- {df['PSNR'].std():.2f}")
    print(f"Average SSIM: {df['SSIM'].mean():.4f} +/- {df['SSIM'].std():.4f}")
    print(f"Metrics saved to {csv_path}")
    
    # ================= 5. 绘制论文图表 =================
    
    # --- 图表 1: 指标箱线图 (Quantitative Distribution) ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    sns.boxplot(y=df['PSNR'], ax=axes[0], color='#8da0cb')
    axes[0].set_title("Reconstruction PSNR (dB)")
    axes[0].set_ylabel("PSNR")
    
    sns.boxplot(y=df['SSIM'], ax=axes[1], color='#fc8d62')
    axes[1].set_title("Structural Similarity (SSIM)")
    axes[1].set_ylabel("SSIM")
    
    sns.boxplot(y=df['Porosity_Error'], ax=axes[2], color='#66c2a5')
    axes[2].set_title("Density/Porosity Error (%)")
    axes[2].set_ylabel("Error %")
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fig1_metrics_distribution.png"))
    plt.close()
    
    # --- 图表 2: 切片对比 + 误差热力图 (Visual Quality) ---
    # 取中间切片
    mid_idx = vis_sample_true.shape[0] // 2
    slice_true = vis_sample_true[mid_idx]
    slice_recon = vis_sample_recon[mid_idx]
    diff_map = np.abs(slice_true - slice_recon)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    im0 = axes[0].imshow(slice_true, cmap='gray', vmin=0, vmax=1)
    axes[0].set_title("Original Ground Truth")
    axes[0].axis('off')
    
    im1 = axes[1].imshow(slice_recon, cmap='gray', vmin=0, vmax=1)
    axes[1].set_title(f"Reconstruction (Epoch 20)")
    axes[1].axis('off')
    
    # 误差图用 'inferno' 或 'hot' 配色，突出差异
    im2 = axes[2].imshow(diff_map, cmap='inferno', vmin=0, vmax=0.2) # vmax设小一点，突出微小误差
    axes[2].set_title("Difference Heatmap (|Orig - Recon|)")
    axes[2].axis('off')
    
    # 加 Colorbar
    cbar = fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel('Absolute Error', rotation=270, labelpad=15)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fig2_visual_comparison.png"))
    plt.close()
    
    # --- 图表 3: 灰度分布直方图 (Physical Property - Density) ---
    plt.figure(figsize=(8, 6))
    sns.kdeplot(vis_sample_true.flatten(), fill=True, label='Ground Truth', color='black', alpha=0.3)
    sns.kdeplot(vis_sample_recon.flatten(), fill=False, label='Reconstruction', color='red', linestyle='--', linewidth=2)
    plt.title("Voxel Intensity Distribution (Density Profile)")
    plt.xlabel("Normalized Intensity")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(OUTPUT_DIR, "fig3_density_histogram.png"))
    plt.close()
    
    # --- 图表 4: 码本使用率 (Codebook Utilization) ---
    # 统计最后一张图的码本使用情况 (或者你可以累积所有batch的来画)
    plt.figure(figsize=(10, 5))
    plt.hist(code_indices, bins=100, color='#1f77b4', alpha=0.8)
    plt.title("Codebook Index Usage (Single Volume)")
    plt.xlabel("Code Index (0-2047)")
    plt.ylabel("Frequency")
    plt.xlim(0, CONFIG['num_embeddings'])
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(OUTPUT_DIR, "fig4_codebook_usage.png"))
    plt.close()
    
    print(f"All figures saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    evaluate()
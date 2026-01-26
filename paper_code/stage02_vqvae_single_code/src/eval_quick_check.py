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
from src.model_vqvae_v2 import VQVAE3D 
from src.config import CONFIG
from utils.get_root_path import get_project_root

# ================= 配置区 =================
# 1. 想要评估的模型权重路径
TARGET_EPOCH = 3 # <--- 确保这里是你想要评估的 epoch
CHECKPOINT_PATH = os.path.join(get_project_root(), CONFIG['experiment_name'], CONFIG['model_output_dir'], f"vqvae_finetune_epoch_{TARGET_EPOCH}.pth")

# 2. 结果保存路径
OUTPUT_DIR = os.path.join(get_project_root(), CONFIG['experiment_name'], "paper_plots_final")

# 3. 只评估多少个样本
MAX_SAMPLES = 20

DEVICE = CONFIG['device']
# =========================================

def setup_plotting_style():
    """
    修改版：移除 Times New Roman 强制要求，防止卡死
    """
    sns.set_theme(style="whitegrid")
    # 移除字体强制设置，使用 matplotlib 默认字体 (通常是 DejaVu Sans)
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['savefig.dpi'] = 300
    print("🎨 Style set to default (safe mode)")

def calculate_metrics(vol_true, vol_recon):
    mse = np.mean((vol_true - vol_recon) ** 2)
    val_psnr = psnr(vol_true, vol_recon, data_range=1.0)
    val_ssim = ssim(vol_true, vol_recon, data_range=1.0, channel_axis=None)
    
    # 简单的孔隙率/密度误差
    porosity_true = np.mean(vol_true)
    porosity_recon = np.mean(vol_recon)
    porosity_error = abs(porosity_true - porosity_recon) / (porosity_true + 1e-8)
    
    return mse, val_psnr, val_ssim, porosity_error

def generate_plots():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    setup_plotting_style()
    
    print(f"Loading checkpoint: {CHECKPOINT_PATH}")
    
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"❌ Error: 权重文件找不到！")
        return

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
        augment=False 
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)
    
    metrics_list = []
    
    # 缓存
    vis_sample_true = None
    vis_sample_recon = None
    code_indices = None
    
    print(f"🚀 Sampling {MAX_SAMPLES} volumes...")
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader, total=MAX_SAMPLES)):
            if i >= MAX_SAMPLES:
                break
            
            img = batch["GT"].to(DEVICE)
            
            # Forward
            img_recon, _, _ = model(img)
            
            # Normalize
            vol_true = (img.cpu().numpy().squeeze() + 1.0) / 2.0
            vol_recon = (img_recon.cpu().numpy().squeeze() + 1.0) / 2.0
            vol_true = np.clip(vol_true, 0, 1)
            vol_recon = np.clip(vol_recon, 0, 1)
            
            # Metrics
            mse, p, s, poro_err = calculate_metrics(vol_true, vol_recon)
            
            metrics_list.append({
                "Sample_ID": i,
                "MSE": mse,
                "PSNR": p,
                "SSIM": s,
                "Porosity_Error": poro_err * 100
            })
            
            # Save first sample for vis
            if i == 0:
                vis_sample_true = vol_true
                vis_sample_recon = vol_recon
                
                # Get indices for Histogram
                z = model.encoder(img)
                _, _, _, indices = model.quantizer(z)
                code_indices = indices.cpu().numpy().flatten()

    # Save CSV
    df = pd.DataFrame(metrics_list)
    df.to_csv(os.path.join(OUTPUT_DIR, "quick_metrics.csv"), index=False)
    print(f"✅ Metrics calculated. Avg PSNR: {df['PSNR'].mean():.2f}")

    # ================= 绘图 =================
    print("📊 Drawing figures...")

    # Figure 1: Metrics
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    sns.boxplot(y=df['PSNR'], ax=axes[0], color='#8da0cb')
    axes[0].set_title("PSNR Distribution")
    sns.boxplot(y=df['SSIM'], ax=axes[1], color='#fc8d62')
    axes[1].set_title("SSIM Distribution")
    sns.boxplot(y=df['Porosity_Error'], ax=axes[2], color='#66c2a5')
    axes[2].set_title("Porosity Error (%)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fig1_metrics_boxplot.png"))
    plt.close()
    print("  -> Saved Fig 1")

    # Figure 2: Visual Comparison
    mid_idx = vis_sample_true.shape[0] // 2
    slice_true = vis_sample_true[mid_idx]
    slice_recon = vis_sample_recon[mid_idx]
    diff_map = np.abs(slice_true - slice_recon)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(slice_true, cmap='gray', vmin=0, vmax=1)
    axes[0].set_title("Original")
    axes[0].axis('off')
    axes[1].imshow(slice_recon, cmap='gray', vmin=0, vmax=1)
    axes[1].set_title(f"Reconstruction")
    axes[1].axis('off')
    im2 = axes[2].imshow(diff_map, cmap='inferno', vmin=0, vmax=0.1) # 高亮误差
    axes[2].set_title("Difference Heatmap")
    axes[2].axis('off')
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fig2_visual_comparison.png"))
    plt.close()
    print("  -> Saved Fig 2")

    # Figure 3: Density
    plt.figure(figsize=(8, 6))
    sns.kdeplot(vis_sample_true.flatten(), fill=True, label='GT', color='black', alpha=0.3)
    sns.kdeplot(vis_sample_recon.flatten(), fill=False, label='Recon', color='red', linestyle='--', linewidth=2)
    plt.title("Voxel Intensity Distribution")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(OUTPUT_DIR, "fig3_density_histogram.png"))
    plt.close()
    print("  -> Saved Fig 3")

    # Figure 4: Codebook
    plt.figure(figsize=(10, 5))
    if code_indices is not None:
        plt.hist(code_indices, bins=100, color='#1f77b4', alpha=0.8)
        plt.title(f"Codebook Usage")
        plt.xlabel("Code Index")
        plt.xlim(0, CONFIG['num_embeddings'])
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(OUTPUT_DIR, "fig4_codebook_usage.png"))
        plt.close()
    print("  -> Saved Fig 4")
    
    print(f"🎉 Done! Files saved in: {OUTPUT_DIR}")

if __name__ == "__main__":
    generate_plots()
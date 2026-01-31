import torch
import numpy as np
import os
import matplotlib.pyplot as plt
import yaml
import glob
import random

# 引用你的 VAE 模型定义
from src.models.vae import KLVAE3D 

# ================= 配置区域 =================
# VAE 的配置和权重路径
VAE_CONFIG_PATH = "/chendou_space/chendou/paper_code/stage02_KLvae_single_code_v2/config/train_config copy.yaml"
VAE_CHECKPOINT = "/chendou_space/chendou/paper_code/stage02_KLvae_single_code_v2/experiments/exp01_cube_structure_v1/checkpoint_epoch_26.pt"

# 原始数据路径 (直接读 NPY)
DATA_ROOT = "/chendou_space/data/aligned_Training_Data"

# 输出图片保存位置
SAVE_PATH = "vae_reconstruction_check.png"

DEVICE = "cuda"

# 是否裁剪？ (256^3 显存可能吃紧，建议设为 128 或 160 看细节，设为 None 则跑全图)
CROP_SIZE = 160  # 填整数(如 128) 或 None

def load_vae():
    print(f"Loading VAE from {os.path.basename(VAE_CHECKPOINT)}...")
    with open(VAE_CONFIG_PATH, 'r') as f:
        vae_cfg = yaml.safe_load(f)
    
    vae = KLVAE3D(vae_cfg).to(DEVICE)
    
    # 加载权重 (处理 _orig_mod 前缀)
    ckpt = torch.load(VAE_CHECKPOINT, map_location=DEVICE)
    state_dict = {}
    for k, v in ckpt['vae_state_dict'].items():
        state_dict[k.replace('_orig_mod.', '')] = v
    vae.load_state_dict(state_dict)
    vae.eval()
    return vae

def get_sample_data():
    files = glob.glob(os.path.join(DATA_ROOT, "*.npy"))
    if not files:
        raise FileNotFoundError(f"No .npy files found in {DATA_ROOT}")
    
    path = random.choice(files)
    print(f"Selected Sample: {os.path.basename(path)}")
    
    # 加载数据 [D, H, W]
    data = np.load(path)
    
    # 归一化到 [-1, 1]
    data = data.astype(np.float32)
    data = (data / 65535.0) * 2.0 - 1.0
    
    # 裁剪逻辑
    if CROP_SIZE is not None:
        d, h, w = data.shape
        ds = (d - CROP_SIZE) // 2
        hs = (h - CROP_SIZE) // 2
        ws = (w - CROP_SIZE) // 2
        data = data[ds:ds+CROP_SIZE, hs:hs+CROP_SIZE, ws:ws+CROP_SIZE]
        print(f"Cropped to {data.shape}")
    else:
        print(f"Full size {data.shape}")

    # 转 Tensor [1, 1, D, H, W]
    tensor = torch.from_numpy(data).unsqueeze(0).unsqueeze(0).to(DEVICE)
    return tensor

def visualize_vae():
    vae = load_vae()
    real_img = get_sample_data()
    
    # === VAE 推理 ===
    with torch.no_grad():
        with torch.amp.autocast('cuda'):
            # 1. Encode -> Latent
            # VAE 返回的是分布，我们要 sample 或者取 mean
            posterior = vae.encode(real_img)
            latent = posterior.sample() # 或者 posterior.mode()
            
            # 2. Decode -> Reconstruction
            recon_img = vae.decode(latent)
    
    # === 数据准备 ===
    # 转回 CPU numpy
    real = real_img.squeeze().cpu().float().numpy() # [D, H, W]
    recon = recon_img.squeeze().cpu().float().numpy()
    
    # 归一化回 [0, 1] 方便显示
    real = (real + 1) / 2
    recon = (recon + 1) / 2
    
    # 截断一下防止显示异常
    real = np.clip(real, 0, 1)
    recon = np.clip(recon, 0, 1)

    # === 切片可视化 ===
    # 取中心切片
    D, H, W = real.shape
    idx_d = D // 2
    idx_h = H // 2
    idx_w = W // 2
    
    # 创建 2行3列 的图
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 第一排：原始数据 (Original)
    # XY Plane (Top View) - 切 Z 轴
    axes[0, 0].imshow(real[idx_d, :, :], cmap='gray')
    axes[0, 0].set_title("Original (XY Plane / Z-cut)")
    axes[0, 0].axis('off')

    # XZ Plane (Side View) - 切 Y 轴
    # 注意 imshow 坐标系，转置一下可能更符合直觉
    axes[0, 1].imshow(real[:, idx_h, :], cmap='gray')
    axes[0, 1].set_title("Original (XZ Plane / Y-cut)")
    axes[0, 1].axis('off')

    # YZ Plane (Front View) - 切 X 轴
    axes[0, 2].imshow(real[:, :, idx_w], cmap='gray')
    axes[0, 2].set_title("Original (YZ Plane / X-cut)")
    axes[0, 2].axis('off')

    # 第二排：重建数据 (Reconstructed)
    axes[1, 0].imshow(recon[idx_d, :, :], cmap='gray')
    axes[1, 0].set_title("VAE Recon (XY Plane)")
    axes[1, 0].axis('off')

    axes[1, 1].imshow(recon[:, idx_h, :], cmap='gray')
    axes[1, 1].set_title("VAE Recon (XZ Plane)")
    axes[1, 1].axis('off')

    axes[1, 2].imshow(recon[:, :, idx_w], cmap='gray')
    axes[1, 2].set_title("VAE Recon (YZ Plane)")
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(SAVE_PATH, dpi=150)
    print(f"✅ VAE Reconstruction result saved to {SAVE_PATH}")
    print(f"Latent Stats: Mean={latent.mean().item():.4f}, Std={latent.std().item():.4f}")

if __name__ == "__main__":
    visualize_vae()
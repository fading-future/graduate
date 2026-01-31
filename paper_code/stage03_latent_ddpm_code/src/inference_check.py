import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import yaml

# 引用你的模块
from src.config import CONFIG
from src.dataset_latent import LatentDataset
from src.model_latent import ConditionalLatentUNet
from src.diffusion_trainer import DiffusionTrainer
from src.models.vae import KLVAE3D 

# ================= 配置 =================
# Stage 1 VAE 的配置和权重
VAE_CONFIG_PATH = "/chendou_space/chendou/paper_code/stage02_KLvae_single_code_v2/config/train_config copy.yaml"
VAE_CHECKPOINT = "/chendou_space/chendou/paper_code/stage02_KLvae_single_code_v2/experiments/exp01_cube_structure_v1/checkpoint_epoch_26.pt"

# Stage 2 UNet 的权重
EPOCH_TO_LOAD = 110
UNET_CHECKPOINT = f"/chendou_space/chendou/paper_code/stage03_latent_ddpm_code/exp_results/exp_final_stage2_graduation/models/unet_epoch_{EPOCH_TO_LOAD}.pth"

DEVICE = "cuda"
SAVE_DIR = "inference_results"
os.makedirs(SAVE_DIR, exist_ok=True)

# === 裁剪配置 ===
# 建议设为 16 或 20。
# 16 -> 解码出 128^3 像素
# 20 -> 解码出 160^3 像素
LATENT_CROP_SIZE = 16

def print_stats(name, x):
    print(f"[{name}] Mean: {x.mean().item():.3f} | Std: {x.std().item():.3f}")

def load_models():
    print(f"Loading Stage 2 UNet from {os.path.basename(UNET_CHECKPOINT)}...")
    unet = ConditionalLatentUNet(
        in_channels=CONFIG['in_channels'],
        out_channels=CONFIG['out_channels'],
        base_channels=CONFIG['base_channels'],
        channel_mults=CONFIG['channel_mults'],
        use_attention=(False, True, True)
    ).to(DEVICE)
    
    ckpt = torch.load(UNET_CHECKPOINT, map_location=DEVICE)
    if 'ema_state_dict' in ckpt:
        print("Using EMA weights (Smoother!)")
        unet.load_state_dict(ckpt['ema_state_dict'])
    else:
        print("Using standard weights")
        unet.load_state_dict(ckpt['model_state_dict'])
    unet.eval()

    print("Loading Stage 1 VAE...")
    with open(VAE_CONFIG_PATH, 'r') as f:
        vae_cfg = yaml.safe_load(f)
    vae = KLVAE3D(vae_cfg).to(DEVICE)
    
    vae_ckpt = torch.load(VAE_CHECKPOINT, map_location=DEVICE)
    state_dict = {}
    for k, v in vae_ckpt['vae_state_dict'].items():
        state_dict[k.replace('_orig_mod.', '')] = v
    vae.load_state_dict(state_dict)
    vae.eval()
    
    return unet, vae

def sample_demo():
    unet, vae = load_models()
    diffusion = DiffusionTrainer(unet, CONFIG) 
    
    # 1. 拿一个测试数据
    dataset = LatentDataset(data_dir=CONFIG['processed_data_dir'], augment=False)
    idx = np.random.randint(0, len(dataset))
    batch = dataset[idx]
    
    gt_latent_full = batch['GT'].unsqueeze(0).to(DEVICE)       
    porosity = batch['Porosity'].unsqueeze(0).to(DEVICE)
    
    # === 裁剪逻辑 ===
    crop_s = LATENT_CROP_SIZE
    _, _, d, h, w = gt_latent_full.shape
    d_start = (d - crop_s) // 2
    h_start = (h - crop_s) // 2
    w_start = (w - crop_s) // 2
    gt_latent = gt_latent_full[:, :, d_start:d_start+crop_s, h_start:h_start+crop_s, w_start:w_start+crop_s]
    
    print(f"Latent shape: {gt_latent.shape} (Decoder output: {crop_s*8}^3)")

    # 2. 制造 "切掉一半" 的任务 (Z轴)
    mask = torch.ones_like(gt_latent[:, :1, ...]) 
    split_idx = int(crop_s * 0.5)
    mask[:, :, split_idx:, :, :] = 0.0  # 后半部分未知
    
    condition = gt_latent * mask
    print_stats("Condition", condition)
    print_stats("GT Input (Scaled)", gt_latent)
    
    print(f"Start Sampling... Target Porosity: {porosity.item():.4f}")
    
    unet.eval()
    with torch.no_grad():
        x = torch.randn_like(gt_latent)
        
        for t in reversed(range(0, CONFIG['timesteps'])):
            t_tensor = torch.full((1,), t, device=DEVICE, dtype=torch.long)
            model_input = torch.cat([x, condition, mask], dim=1)
            noise_pred = unet(model_input, t_tensor, porosity)
            
            alpha = diffusion.alphas[t]
            alpha_cumprod = diffusion.alphas_cumprod[t]
            beta = diffusion.betas[t]
            
            if t > 0:
                noise = torch.randn_like(x)
            else:
                noise = 0
            
            # DDPM Step
            x = (1 / torch.sqrt(alpha)) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_cumprod))) * noise_pred) + torch.sqrt(beta) * noise
            
            # RePaint 策略 (关键步骤：强行把已知区域修正回 GT)
            if t > 0:
                noise_gt = torch.randn_like(gt_latent)
                gt_noisy = torch.sqrt(diffusion.alphas_cumprod[t-1]) * gt_latent + torch.sqrt(1 - diffusion.alphas_cumprod[t-1]) * noise_gt
                x = x * (1 - mask) + gt_noisy * mask

    # === 最后一步强制覆盖 ===
    x = x * (1 - mask) + gt_latent * mask 
    
    print("Sampling Done. Decoding...")
    
    # 4. 解码
    rec_latent = x / CONFIG['scale_factor']
    
    # 【建议新增】: 稍微截断一下 Latent，防止极个别噪点(比如 > 5.0) 导致 VAE 解码后的图片整体对比度异常
    # 你的数据标准差是1，所以截断在 +/- 4 或 5 是安全的
    rec_latent = torch.clamp(rec_latent, min=-5.0, max=5.0) 
    
    gt_latent_raw = gt_latent / CONFIG['scale_factor']
    
    print("Sampling Done. Decoding...")
    
    with torch.no_grad():
        with torch.amp.autocast('cuda'):
            pred_volume = vae.decode(rec_latent) 
            gt_volume = vae.decode(gt_latent_raw)
    
    # 5. 转 Numpy & 监控数值
    pred_vol = pred_volume.squeeze().cpu().float().numpy()
    gt_vol = gt_volume.squeeze().cpu().float().numpy()
    
    # 【DEBUG】打印一下数值范围，看看有没有爆炸
    print(f"DEBUG: Pred Range [{pred_vol.min():.2f}, {pred_vol.max():.2f}]")
    print(f"DEBUG: GT Range   [{gt_vol.min():.2f}, {gt_vol.max():.2f}]")

    # 归一化显示
    pred_vol = (pred_vol + 1) / 2
    gt_vol = (gt_vol + 1) / 2
    
    # 选取切片
    final_size = pred_vol.shape[0]
    slice_idx = final_size // 2
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Ground Truth
    axes[0].imshow(gt_vol[:, :, slice_idx], cmap='gray', vmin=0, vmax=1)
    axes[0].set_title(f"Ground Truth ({final_size}^3)")
    axes[0].axhline(y=slice_idx, color='r', linestyle='--') 
    
    # Input
    masked_img = gt_vol[:, :, slice_idx].copy()
    masked_img[slice_idx:, :] = 0 
    axes[1].imshow(masked_img, cmap='gray', vmin=0, vmax=1)
    axes[1].set_title("Input (Masked)")
    
    # Result
    axes[2].imshow(pred_vol[:, :, slice_idx], cmap='gray', vmin=0, vmax=1)
    axes[2].set_title("Inpainted Result")
    axes[2].axhline(y=slice_idx, color='r', linestyle='--')
    
    plt.tight_layout()
    save_path = os.path.join(SAVE_DIR, f"check_epoch{EPOCH_TO_LOAD}_crop_{LATENT_CROP_SIZE}.png")
    plt.savefig(save_path)
    print(f"✅ Result saved to {save_path}")

if __name__ == "__main__":
    sample_demo()
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg') # 关键：禁用 GUI 后端
import matplotlib.pyplot as plt
import os
import glob
import sys
import random
import re
import nibabel as nib
from tqdm import tqdm

# --- 项目依赖 ---
from model_latent import ConditionalLatentUNet
from model_vqvae import VQVAE3D 
from config import CONFIG
from utils.get_root_path import get_root_path

# ==========================================
# 1. 辅助与可视化模块 (Visualization)
# ==========================================

def get_beta_schedule(timesteps):
    return torch.linspace(1e-4, 0.02, timesteps)

def load_models(model_path, device):
    """加载 Stage1(VQVAE) 和 Stage2(Diffusion)"""
    print(f"🔹 Loading Diffusion Model: {os.path.basename(model_path)}")
    
    # 1. 实例化 Stage 2 模型
    diffusion_model = ConditionalLatentUNet(
        in_channels=CONFIG['in_channels'],       # 64+64+1
        out_channels=CONFIG['out_channels'],
        base_channels=CONFIG['base_channels'],    
        channel_mults=(1, 2, 4), 
        use_attention=(False, True, True) 
    ).to(device)

    # 2. 智能加载权重 (处理 EMA 和 Dict 格式)
    checkpoint = torch.load(model_path, map_location=device)
    
    if isinstance(checkpoint, dict):
        # 优先加载 EMA 权重，因为 EMA 权重的生成质量通常更稳定
        if 'ema_state_dict' in checkpoint:
            print("✨ Successfully loaded EMA weights for Stage 2.")
            diffusion_model.load_state_dict(checkpoint['ema_state_dict'])
        elif 'model_state_dict' in checkpoint:
            print("⚠️ EMA weights not found, using raw model weights.")
            diffusion_model.load_state_dict(checkpoint['model_state_dict'])
        else:
            # 兼容某些只存了字典但 key 不对的情况
            diffusion_model.load_state_dict(checkpoint)
    else:
        # 兼容旧版本直接保存的 state_dict (纯权重文件)
        print("📜 Loading raw state_dict (Old format).")
        diffusion_model.load_state_dict(checkpoint)
    
    diffusion_model.eval()

    # 3. 加载 Stage 1 VQ-VAE
    print(f"🔹 Loading VQ-VAE Model: {os.path.basename(CONFIG['stage1_model_path'])}")
    vqvae_model = VQVAE3D(
        in_channels=1,
        embedding_dim=CONFIG['latent_channels'], 
        num_embeddings=2048 
    ).to(device)
    
    # 注意：如果你的 Stage 1 也是字典格式，这里也需要像上面一样做判断
    vqvae_checkpoint = torch.load(CONFIG['stage1_model_path'], map_location=device)
    if isinstance(vqvae_checkpoint, dict) and 'model_state_dict' in vqvae_checkpoint:
        vqvae_model.load_state_dict(vqvae_checkpoint['model_state_dict'])
    else:
        vqvae_model.load_state_dict(vqvae_checkpoint)
        
    vqvae_model.eval()
    
    return diffusion_model, vqvae_model

# def load_models(model_path, device):
#     """加载 Stage1(VQVAE) 和 Stage2(Diffusion)"""
#     print(f"🔹 Loading Diffusion Model: {os.path.basename(model_path)}")
#     # diffusion_model = ConditionalLatentUNet(
#     #     in_channels=CONFIG['in_channels'],
#     #     out_channels=CONFIG['out_channels'],
#     #     base_channels=CONFIG['base_channels']
#     # ).to(device)
#     diffusion_model = ConditionalLatentUNet(
#         in_channels=CONFIG['in_channels'],       # 64+64+1
#         out_channels=CONFIG['out_channels'],
#         base_channels=CONFIG['base_channels'],     # A100 显存大，直接上 128
#         channel_mults=(1, 2, 4), # 通道变成 128 -> 256 -> 512
#         use_attention=(False, True, True) # 在中间层和最底层开启 Attention
#     ).to(CONFIG['device'])
#     diffusion_model.load_state_dict(torch.load(model_path, map_location=device))
#     diffusion_model.eval()

#     print(f"🔹 Loading VQ-VAE Model: {os.path.basename(CONFIG['stage1_model_path'])}")
#     vqvae_model = VQVAE3D(
#         in_channels=1,
#         embedding_dim=CONFIG['latent_channels'], 
#         num_embeddings=2048 
#     ).to(device)
#     vqvae_model.load_state_dict(torch.load(CONFIG['stage1_model_path'], map_location=device))
#     vqvae_model.eval()
    
#     return diffusion_model, vqvae_model

def visualize_three_planes(gt_volume, input_cond_volume, gen_volume, split_point, save_path, sample_name):
    """
    可视化三个正交平面 (XY, XZ, ZY) 的切片，并保存为单个图片。
    数据形状应为 (D, H, W)，即 (Z, Y, X)。
    """
    D, H, W = gt_volume.shape
    slice_idx_z = D // 2
    slice_idx_y = H // 2
    slice_idx_x = W // 2
    
    volumes = {
        "Ground Truth": gt_volume,
        "Input Condition": input_cond_volume,
        "Generated": gen_volume
    }
    
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(12, 12))
    plt.suptitle(f"Sample: {sample_name} | Orthogonal Plane Visualizations")

    for i, (title, vol) in enumerate(volumes.items()):
        # XY 平面 (Z轴切片)
        axes[0, i].imshow(vol[slice_idx_z, :, :], cmap='gray', vmin=-1, vmax=1)
        axes[0, i].set_title(f"{title} (XY plane)")
        # XZ 平面 (Y轴切片)
        axes[1, i].imshow(vol[:, slice_idx_y, :], cmap='gray', vmin=-1, vmax=1, origin='lower')
        axes[1, i].set_title(f"{title} (XZ plane)")
        axes[1, i].axhline(y=split_point, color='r', linestyle='--') # 标记 Z 轴分割线
        # ZY 平面 (X轴切片)
        axes[2, i].imshow(vol[:, :, slice_idx_x], cmap='gray', vmin=-1, vmax=1, origin='lower')
        axes[2, i].set_title(f"{title} (ZY plane)")
        axes[2, i].axhline(y=split_point, color='r', linestyle='--') # 标记 Z 轴分割线

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path)
    plt.close(fig) # 关闭图形，防止内存堆积
    print(f"Saved visualization to {save_path}")


def visualize_inference_results(gt_vol, input_cond_vol, gen_vol, mask_pixel_np, save_path, fname):
    """
    统一可视化函数：3x3 布局 (GT, Condition, Generated) x (XY, XZ, ZY)
    自动根据 Mask 边缘绘制红线
    """
    # 1. 动态对比度 (避免灰色)
    flat_gt = gt_vol.flatten()
    vmin, vmax = np.percentile(flat_gt, 1), np.percentile(flat_gt, 99)
    dist = vmax - vmin
    if dist < 1e-5: dist = 1.0
    vmin -= dist * 0.1
    vmax += dist * 0.1
    
    # 2. 寻找 Mask 边界 (为了画红线)
    # 假设 Mask 是 Z 轴方向的扩展 (Core Extension)，我们找 Z 轴上 0 和 1 的交界处
    # mask_pixel_np: (256, 256, 256), 1=Known, 0=Unknown
    z_profile = mask_pixel_np.mean(axis=(1, 2)) # 沿 Z 轴的平均值
    # 找到从 1 变成 0 的那个索引
    split_indices = np.where(np.diff(z_profile) != 0)[0]
    split_idx = split_indices[0] if len(split_indices) > 0 else -1

    # 3. 切片位置 (默认取中心)
    D, H, W = gt_vol.shape
    cz, cy, cx = D // 2, H // 2, W // 2

    # 4. 绘图
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    plt.suptitle(f"Inference: {fname}", fontsize=16, y=0.98)
    
    cols = ["Ground Truth", "Input Condition", "Generated"]
    data_list = [gt_vol, input_cond_vol, gen_vol]

    # 定义画线辅助函数
    def draw_split_line(ax, axis_name):
        if split_idx > 0:
            if axis_name == 'Z': # 纵轴是 Z
                ax.axhline(y=split_idx, color='red', linestyle='--', linewidth=2, alpha=0.9)
                ax.text(5, split_idx - 5, 'Known', color='red', fontsize=9, fontweight='bold', va='bottom')
                ax.text(5, split_idx + 5, 'Gen', color='yellow', fontsize=9, fontweight='bold', va='top')

    for col_idx, (title, vol) in enumerate(zip(cols, data_list)):
        # --- Row 1: XY Plane (Top View, Slice Z) ---
        ax = axes[0, col_idx]
        ax.imshow(vol[cz, :, :], cmap='gray', vmin=vmin, vmax=vmax)
        ax.set_title(f"{title}\nXY (Z={cz})")
        ax.axis('off')

        # --- Row 2: XZ Plane (Side View, Slice Y) ---
        # 纵轴是 Z，横轴是 X
        ax = axes[1, col_idx]
        ax.imshow(vol[:, cy, :], cmap='gray', vmin=vmin, vmax=vmax, origin='lower')
        ax.set_title(f"{title}\nXZ (Y={cy})")
        ax.set_ylabel("Z (Depth)")
        ax.set_xticks([])
        ax.set_yticks([])
        draw_split_line(ax, 'Z') # 画线

        # --- Row 3: ZY Plane (Side View, Slice X) ---
        # 纵轴是 Z，横轴是 Y
        ax = axes[2, col_idx]
        ax.imshow(vol[:, :, cx], cmap='gray', vmin=vmin, vmax=vmax, origin='lower')
        ax.set_title(f"{title}\nZY (X={cx})")
        ax.set_ylabel("Z (Depth)")
        ax.set_xticks([])
        ax.set_yticks([])
        draw_split_line(ax, 'Z') # 画线

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  🖼️ Saved visualization: {save_path}")

# ==========================================
# 2. 核心采样策略 (Sampling Strategies)
# ==========================================

# 设定推理时的安全截断阈值
# 训练时 Scale=4.22, Clamp=5.0
# 推理时稍微放宽一点，防止硬截断导致纹理死锁，同时遏制指数级漂移
SAFE_LIMIT = CONFIG['safe_threshold'] 

def simple_sample(model, condition, mask, porosity, device):
    """
    策略 A: 标准 DDPM 采样 (修正版：加入数值稳定截断)
    """
    print("  🚀 Strategy: Robust DDPM Sampling...")
    timesteps = CONFIG['timesteps']
    betas = get_beta_schedule(timesteps).to(device)
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)

    # 1. 初始噪声
    x = torch.randn((1, CONFIG['latent_channels'], CONFIG['image_size'], CONFIG['image_size'], CONFIG['image_size'])).to(device)
    
    # 设定安全边界 (必须和 DDIM 保持一致)
    SAFE_LIMIT = CONFIG['safe_threshold'] 

    for i in tqdm(reversed(range(timesteps)), total=timesteps, desc="DDPM Sampling"):
        t = torch.full((1,), i, device=device, dtype=torch.long)
        
        # 2. 预测噪声
        model_input = torch.cat([x, condition, mask], dim=1)
        noise_pred = model(model_input, t, porosity)
        
        # 3. 计算 x_{t-1}
        beta_t = betas[i]
        alpha_t = alphas[i]
        alpha_bar_t = alphas_cumprod[i]
        
        if i > 0:
            noise = torch.randn_like(x)
        else:
            noise = torch.zeros_like(x)
            
        # DDPM 逆向公式
        x = (1 / torch.sqrt(alpha_t)) * (x - (beta_t / torch.sqrt(1 - alpha_bar_t)) * noise_pred) + torch.sqrt(beta_t) * noise
        
        # 【核心修正】立即截断，防止单步数值爆炸
        x = torch.clamp(x, -SAFE_LIMIT, SAFE_LIMIT)
        
        # 4. 强制替换已知区域
        if i > 0:
            noise_new = torch.randn_like(condition)
            alpha_bar_prev = alphas_cumprod[i-1]
            noisy_gt = torch.sqrt(alpha_bar_prev) * condition + torch.sqrt(1 - alpha_bar_prev) * noise_new
            x = x * (1 - mask) + noisy_gt * mask
        else:
            x = x * (1 - mask) + condition * mask
            
        # 【核心修正】融合后再次截断
        x = torch.clamp(x, -SAFE_LIMIT, SAFE_LIMIT)

    return x

def ddim_sample(model, condition, mask, porosity, device, ddim_steps=50, eta=0.0):
    """
    策略 B: 优化版 DDIM 采样 (Pred_x0 Clamping)
    最稳健的方案，强制截断预测出的 x0，从根源上消除数值爆炸。
    """
    ddim_steps = ddim_steps if ddim_steps is not None else CONFIG.get('ddim_steps_infer', 200)
    print(f"  🚀 Strategy: Optimized DDIM Sampling ({ddim_steps} steps)...")
    
    total_timesteps = CONFIG['timesteps']
    betas = get_beta_schedule(total_timesteps).to(device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)

    # 时间步序列
    times = torch.linspace(0, total_timesteps - 1, steps=ddim_steps).long().to(device)
    times = list(reversed(times.tolist()))
    
    x = torch.randn((1, CONFIG['latent_channels'], CONFIG['image_size'], CONFIG['image_size'], CONFIG['image_size'])).to(device)

    with torch.no_grad():
        for i, t in enumerate(tqdm(times, desc="DDIM Sampling")):
            t_tensor = torch.full((1,), t, device=device, dtype=torch.long)
            t_prev = times[i + 1] if i < len(times) - 1 else -1
            
            # --- A. 预测 ---
            model_input = torch.cat([x, condition, mask], dim=1)
            noise_pred = model(model_input, t_tensor, porosity)
            
            # --- B. 数学推导 (x_t -> pred_x0 -> x_{t-1}) ---
            alpha_bar_t = alphas_cumprod[t]
            alpha_bar_t_prev = alphas_cumprod[t_prev] if t_prev >= 0 else torch.tensor(1.0).to(device)
            
            # 1. 预测 x0 (Denoised)
            pred_x0 = (x - torch.sqrt(1.0 - alpha_bar_t) * noise_pred) / torch.sqrt(alpha_bar_t)
            
            # 【防护 1】核心：对预测的 x0 进行截断
            # 这是防止雪花屏最关键的一步
            pred_x0 = torch.clamp(pred_x0, -SAFE_LIMIT, SAFE_LIMIT)
            
            # 2. 指向 x_{t-1} 的方向
            pred_dir_xt = torch.sqrt(1.0 - alpha_bar_t_prev) * noise_pred
            
            # 3. 重构 x_{t-1}
            x_prev = torch.sqrt(alpha_bar_t_prev) * pred_x0 + pred_dir_xt
            
            # --- C. 注入已知区域 ---
            if t_prev >= 0:
                # 给 GT 加上 t_prev 时刻的噪声
                noise_real = torch.randn_like(condition)
                known_x_prev = torch.sqrt(alpha_bar_t_prev) * condition + torch.sqrt(1.0 - alpha_bar_t_prev) * noise_real
                
                x = x_prev * (1 - mask) + known_x_prev * mask
            else:
                # 最后一步：直接使用最清晰的 pred_x0 和 condition
                x = pred_x0 * (1 - mask) + condition * mask
                
            # 【防护 2】防止融合后出现微小溢出
            x = torch.clamp(x, -SAFE_LIMIT, SAFE_LIMIT)
                
    return x

def repaint_sample(model, condition, mask, porosity, device, n_resample=5):
    """
    策略 C: RePaint (Fixed)
    修正了方差爆炸问题：
    1. Step 1 改为确定性采样 (不加噪)，防止与 Step 3 的加噪叠加导致能量溢出。
    2. 修正了回溯步骤的 beta 索引。
    """
    print(f"  🚀 Strategy: Robust RePaint Sampling (Resample={n_resample})...")
    timesteps = CONFIG['timesteps']
    betas = get_beta_schedule(timesteps).to(device)
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)

    # 1. 初始化噪声
    x = torch.randn((1, CONFIG['latent_channels'], CONFIG['image_size'], CONFIG['image_size'], CONFIG['image_size'])).to(device)
    
    # 进度条
    pbar = tqdm(reversed(range(timesteps)), total=timesteps, desc="RePaint Sampling")
    
    for t in pbar:
        # 动态调整重采样次数
        # 只有在生成中间纹理的关键阶段才多次采样，节省时间
        current_resample = n_resample if (50 < t < 800) else 1
        
        for r in range(current_resample):
            # -----------------------------------------------------------
            # Step 1: 逆向 (Reverse) x_t -> x_{t-1} 
            # 【关键修改】使用确定性采样 (Deterministic)，不加随机噪声！
            # -----------------------------------------------------------
            t_tensor = torch.full((1,), t, device=device, dtype=torch.long)
            model_input = torch.cat([x, condition, mask], dim=1)
            noise_pred = model(model_input, t_tensor, porosity)
            
            # 获取参数
            beta_t = betas[t]
            alpha_t = alphas[t]
            alpha_bar_t = alphas_cumprod[t]
            
            # 计算 pred_x0 (用于指导方向)
            pred_x0 = (x - torch.sqrt(1 - alpha_bar_t) * noise_pred) / torch.sqrt(alpha_bar_t)
            pred_x0 = torch.clamp(pred_x0, -SAFE_LIMIT, SAFE_LIMIT) # 截断防护
            
            # 计算 x_{t-1} 的均值 (Mean only, NO random noise here!)
            # 这是一个指向 x_{t-1} 的确定性向量
            # 公式推导自 DDIM (eta=0) 或 Posterior Mean
            coef1 = torch.sqrt(1.0 - alpha_bar_t - beta_t) # 这里简化处理，近似 DDIM
            # 或者使用标准的 DDPM 均值公式:
            # mean = (1 / sqrt(alpha_t)) * (x - (beta_t / sqrt(1-alpha_bar_t)) * noise_pred)
            # 我们使用 DDPM 均值公式更准确：
            x_prev_unknown = (1 / torch.sqrt(alpha_t)) * (x - (beta_t / torch.sqrt(1 - alpha_bar_t)) * noise_pred)
            
            # 【核心区别】这里不要加 "+ sigma * noise"，保持确定性！
            
            # 截断
            x_prev_unknown = torch.clamp(x_prev_unknown, -SAFE_LIMIT, SAFE_LIMIT)
            
            # -----------------------------------------------------------
            # Step 2: 注入已知 (Inject Condition)
            # -----------------------------------------------------------
            if t > 0:
                alpha_bar_prev = alphas_cumprod[t-1]
                # 每次重新采样 GT 的噪声，保证多样性
                noise_gt = torch.randn_like(condition)
                x_prev_known = torch.sqrt(alpha_bar_prev) * condition + torch.sqrt(1 - alpha_bar_prev) * noise_gt
                
                # 融合
                x_prev = x_prev_unknown * (1 - mask) + x_prev_known * mask
            else:
                x_prev = x_prev_unknown * (1 - mask) + condition * mask

            # -----------------------------------------------------------
            # Step 3: 时间回溯 (Forward / Resample) x_{t-1} -> x_t
            # 只有当不是最后一次重采样，且 t > 0 时才回溯
            # -----------------------------------------------------------
            if r < current_resample - 1 and t > 0:
                # 【关键修改】使用当前的 beta_t，而不是 beta_{t-1}
                # 因为我们要模拟从 t-1 到 t 的过程，方差由 beta_t 控制
                beta_forward = betas[t] 
                
                noise_add = torch.randn_like(x_prev)
                
                # Forward Process 公式: x_t = sqrt(1-beta)*x_{t-1} + sqrt(beta)*noise
                # 注意：这里我们用近似公式，或者直接用 RePaint 论文公式
                # RePaint 论文: x_t ~ N(sqrt(1-beta_t)*x_{t-1}, beta_t*I)
                x = torch.sqrt(1 - beta_forward) * x_prev + torch.sqrt(beta_forward) * noise_add
                
                # 截断，防止噪声叠加溢出
                x = torch.clamp(x, -SAFE_LIMIT, SAFE_LIMIT)
            else:
                # 循环结束或最后一步，保留 x_{t-1} 进入下一个 t
                x = x_prev
                
    return x

# def repaint_sample(model, condition, mask, porosity, device, n_resample=5):
#     """
#     策略 C: RePaint (Resampling)
#     效果最好，通过“进两步退一步”的方式增强边界融合。
#     已加入数值稳定保护。
#     """
#     print(f"  🚀 Strategy: Robust RePaint Sampling (Resample={n_resample})...")
#     timesteps = CONFIG['timesteps']
#     betas = get_beta_schedule(timesteps).to(device)
#     alphas = 1. - betas
#     alphas_cumprod = torch.cumprod(alphas, dim=0)

#     x = torch.randn((1, CONFIG['latent_channels'], CONFIG['image_size'], CONFIG['image_size'], CONFIG['image_size'])).to(device)
    
#     sys.stdout.flush()
#     pbar = tqdm(reversed(range(timesteps)), total=timesteps, desc="RePaint Sampling")
    
#     for t in pbar:
#         # 动态调整重采样次数 (中间阶段多采几次，两头少采)
#         current_resample = n_resample if (100 < t < 900) else 1
        
#         for r in range(current_resample):
#             # --- Step 1: 逆向 (Reverse) x_t -> x_{t-1} ---
#             t_tensor = torch.full((1,), t, device=device, dtype=torch.long)
#             model_input = torch.cat([x, condition, mask], dim=1)
#             noise_pred = model(model_input, t_tensor, porosity)
            
#             beta_t = betas[t]
#             alpha_t = alphas[t]
#             alpha_bar_t = alphas_cumprod[t]
            
#             # 为了更好的稳定性，我们先算出 pred_x0 (仅用于截断参考，实际计算仍用 DDPM 公式)
#             # pred_x0_guess = (x - torch.sqrt(1 - alpha_bar_t) * noise_pred) / torch.sqrt(alpha_bar_t)
#             # pred_x0_guess = torch.clamp(pred_x0_guess, -SAFE_LIMIT, SAFE_LIMIT)
#             # 这里的 RePaint 实现采用标准 DDPM 步进，我们直接截断结果
            
#             if t > 0:
#                 noise = torch.randn_like(x)
#             else:
#                 noise = torch.zeros_like(x)
            
#             # DDPM Update
#             x_prev_unknown = (1 / torch.sqrt(alpha_t)) * (x - (beta_t / torch.sqrt(1 - alpha_bar_t)) * noise_pred) + torch.sqrt(beta_t) * noise
            
#             # 【防护 1】
#             x_prev_unknown = torch.clamp(x_prev_unknown, -SAFE_LIMIT, SAFE_LIMIT)
            
#             # --- Step 2: 注入已知 (Condition) ---
#             if t > 0:
#                 noise_gt = torch.randn_like(condition)
#                 alpha_bar_prev = alphas_cumprod[t-1]
#                 # GT 的 x_{t-1} (带噪)
#                 x_prev_known = torch.sqrt(alpha_bar_prev) * condition + torch.sqrt(1 - alpha_bar_prev) * noise_gt
                
#                 # 融合
#                 x_prev = x_prev_unknown * (1 - mask) + x_prev_known * mask
#             else:
#                 # 最后一步
#                 x_prev = x_prev_unknown * (1 - mask) + condition * mask

#             # --- Step 3: 时间回溯 (Forward) x_{t-1} -> x_t ---
#             # 如果还需要重采样，就把 x_{t-1} 再加噪变回 x_t
#             if r < current_resample - 1 and t > 0:
#                 beta_next = betas[t-1] # 这里的 beta 对应的是前一步的噪声水平
#                 noise_add = torch.randn_like(x_prev)
#                 # Forward Process: q(x_t | x_{t-1})
#                 x = torch.sqrt(1 - beta_next) * x_prev + torch.sqrt(beta_next) * noise_add
                
#                 # 【防护 2】回溯后的值也要截断，防止噪声叠加导致溢出
#                 x = torch.clamp(x, -SAFE_LIMIT, SAFE_LIMIT)
#             else:
#                 x = x_prev
                
#     return x

# ==========================================
# 2. 核心采样策略 (Sampling Strategies)
# ==========================================

# def simple_sample(model, condition, mask, porosity, device):
#     """
#     策略 A: 标准 DDPM 采样 + 强制替换 (Replacement)
#     最慢，最基础，用于验证模型能力。
#     """
#     print("  🚀 Strategy: Simple DDPM Sampling...")
#     timesteps = CONFIG['timesteps']
#     betas = get_beta_schedule(timesteps).to(device)
#     alphas = 1. - betas
#     alphas_cumprod = torch.cumprod(alphas, dim=0)

#     # 初始噪声
#     x = torch.randn((1, CONFIG['latent_channels'], CONFIG['image_size'], CONFIG['image_size'], CONFIG['image_size'])).to(device)

#     # for i in tqdm(reversed(range(timesteps))):
#     sys.stdout.flush() # 强制刷新输出缓冲区，给 tqdm 腾位置

#     # 使用 tqdm 时显式指定 leave=True
#     for i in tqdm(reversed(range(timesteps)), total=timesteps, desc="Generating"):
#         t = torch.full((1,), i, device=device, dtype=torch.long)
        
#         # 1. 预测噪声
#         model_input = torch.cat([x, condition, mask], dim=1)
#         noise_pred = model(model_input, t, porosity)
        
#         # 2. 计算 x_{t-1}
#         beta_t = betas[i]
#         alpha_t = alphas[i]
#         alpha_bar_t = alphas_cumprod[i]
        
#         if i > 0:
#             noise = torch.randn_like(x)
#         else:
#             noise = torch.zeros_like(x)
            
#         x = (1 / torch.sqrt(alpha_t)) * (x - (beta_t / torch.sqrt(1 - alpha_bar_t)) * noise_pred) + torch.sqrt(beta_t) * noise
        
#         # 3. 【核心】强制替换已知区域 (Inpainting/Outpainting Constraint)
#         # 将 x_{t-1} 中已知的部分，替换为 Ground Truth 加噪后的版本
#         if i > 0:
#             noise_new = torch.randn_like(condition)
#             alpha_bar_prev = alphas_cumprod[i-1]
#             # 计算 q(x_{t-1} | x_0)
#             noisy_gt = torch.sqrt(alpha_bar_prev) * condition + torch.sqrt(1 - alpha_bar_prev) * noise_new
#             x = x * (1 - mask) + noisy_gt * mask
#         else:
#             x = x * (1 - mask) + condition * mask

#     return x

# def ddim_sample(model, condition, mask, porosity, device, ddim_steps=50, eta=0.0):
#     """
#     策略 B: DDIM 加速采样
#     速度快，适合调试。
#     """
#     print(f"  🚀 Strategy: DDIM Sampling ({ddim_steps} steps)...")
#     total_timesteps = CONFIG['timesteps']
#     # 生成时间步序列 (e.g., [999, 979, ..., 0])
#     times = torch.linspace(0, total_timesteps - 1, steps=ddim_steps).long().to(device)
#     times = list(reversed(times.int().tolist()))
    
#     betas = get_beta_schedule(total_timesteps).to(device)
#     alphas = 1.0 - betas
#     alphas_cumprod = torch.cumprod(alphas, dim=0)

#     x = torch.randn((1, CONFIG['latent_channels'], CONFIG['image_size'], CONFIG['image_size'], CONFIG['image_size'])).to(device)
    
#     with torch.no_grad():
#         for i, t in enumerate(tqdm(times)):
#             t_prev = times[i + 1] if i < len(times) - 1 else -1
            
#             # 1. 预测
#             model_input = torch.cat([x, condition, mask], dim=1)
#             t_tensor = torch.full((1,), t, device=device, dtype=torch.long)
#             noise_pred = model(model_input, t_tensor, porosity)
            
#             # 2. DDIM 公式
#             alpha_bar_t = alphas_cumprod[t]
#             alpha_bar_t_prev = alphas_cumprod[t_prev] if t_prev >= 0 else torch.tensor(1.0).to(device)
            
#             # 预测 x0
#             pred_x0 = (x - torch.sqrt(1 - alpha_bar_t) * noise_pred) / torch.sqrt(alpha_bar_t)
#             # 指向 xt 的方向
#             pred_dir_xt = torch.sqrt(1 - alpha_bar_t_prev) * noise_pred # eta=0 for deterministic
            
#             x_prev = torch.sqrt(alpha_bar_t_prev) * pred_x0 + pred_dir_xt
            
#             # 3. 强制替换
#             if t_prev >= 0:
#                 noise_real = torch.randn_like(condition)
#                 known_x_prev = torch.sqrt(alpha_bar_t_prev) * condition + torch.sqrt(1 - alpha_bar_t_prev) * noise_real
#                 x = x_prev * (1 - mask) + known_x_prev * mask
#             else:
#                 x = pred_x0 * (1 - mask) + condition * mask # 最后一步直接用 x0
                
#     return x

# def ddim_sample(model, condition, mask, porosity, device, ddim_steps=50, eta=0.0):
#     """
#     重构后的 DDIM 采样逻辑：加入数值稳定控制与 pred_x0 截断。
#     """
#     print(f"   🚀 Strategy: Optimized DDIM Sampling ({ddim_steps} steps)...")
    
#     # 1. 获取 Beta/Alpha 序列
#     total_timesteps = CONFIG['timesteps']
#     betas = get_beta_schedule(total_timesteps).to(device)
#     alphas = 1.0 - betas
#     alphas_cumprod = torch.cumprod(alphas, dim=0)

#     # 2. 准备时间序列 (从 T-1 到 0)
#     times = torch.linspace(0, total_timesteps - 1, steps=ddim_steps).long().to(device)
#     times = list(reversed(times.tolist()))
    
#     # 初始噪声：由模型从标准正态分布生成
#     # 注意：这里的 x 已经是放大 4.22 倍后对应的空间了 (Std 约为 1)
#     x = torch.randn((1, CONFIG['latent_channels'], CONFIG['image_size'], CONFIG['image_size'], CONFIG['image_size'])).to(device)
    
#     # 设定安全边界：既然你用了 4.22 缩放，GT 的范围大概在 [-6, 6]
#     # 我们把截断设在 5.5 到 6 之间
#     SAFE_LIMIT = 6.0 

#     with torch.no_grad():
#         for i, t in enumerate(tqdm(times)):
#             t_tensor = torch.full((1,), t, device=device, dtype=torch.long)
#             t_prev = times[i + 1] if i < len(times) - 1 else -1
            
#             # --- A. 准备模型输入 ---
#             # 确保 x, condition, mask 在 concat 时的数值范围是一致的
#             model_input = torch.cat([x, condition, mask], dim=1)
            
#             # --- B. 预测噪声 ---
#             noise_pred = model(model_input, t_tensor, porosity)
            
#             # --- C. 数学推导与稳定截断 ---
#             alpha_bar_t = alphas_cumprod[t]
#             alpha_bar_t_prev = alphas_cumprod[t_prev] if t_prev >= 0 else torch.tensor(1.0).to(device)
            
#             # 1. 核心公式：从当前 x 预测 pred_x0
#             # pred_x0 = (x_t - sqrt(1-alpha_t)*noise) / sqrt(alpha_t)
#             pred_x0 = (x - torch.sqrt(1.0 - alpha_bar_t) * noise_pred) / torch.sqrt(alpha_bar_t)
            
#             # 【防护层 1】：对预测的 x0 进行强制截断
#             # 如果不截断，误差会在这里呈指数级放大，导致最终 Std 变成 30+
#             pred_x0 = torch.clamp(pred_x0, -SAFE_LIMIT, SAFE_LIMIT)
            
#             # 2. 计算方向指向 xt (Dirichlet 方向)
#             # 在 η=0 (DDIM) 时，预测是确定性的
#             pred_dir_xt = torch.sqrt(1.0 - alpha_bar_t_prev) * noise_pred
            
#             # 3. 计算前一步的 x (x_prev)
#             x_prev = torch.sqrt(alpha_bar_t_prev) * pred_x0 + pred_dir_xt
            
#             # --- D. 强力约束 (Inpainting 逻辑) ---
#             if t_prev >= 0:
#                 # 【防护层 2】：对于 Mask 为 1 的区域，我们已知其内容 (Condition)
#                 # 但我们需要给已知内容加上对应时间步的噪声，以保证和生成区域在同一分布
#                 noise_real = torch.randn_like(condition)
#                 known_x_prev = torch.sqrt(alpha_bar_t_prev) * condition + torch.sqrt(1.0 - alpha_bar_t_prev) * noise_real
                
#                 # 融合生成区域与已知区域
#                 x = x_prev * (1.0 - mask) + known_x_prev * mask
#             else:
#                 # 最后一步：直接使用最清晰的 pred_x0
#                 x = pred_x0 * (1.0 - mask) + condition * mask
            
#             # 【防护层 3】：防止数值在融合后出现极小比例的溢出
#             x = torch.clamp(x, -SAFE_LIMIT, SAFE_LIMIT)
                
#     return x

# def repaint_sample(model, condition, mask, porosity, device, n_resample=5):
#     """
#     策略 C: RePaint (Resampling)
#     效果最好，通过“进两步退一步”的方式增强边界融合。
#     """
#     print(f"  🚀 Strategy: RePaint Sampling (Resample={n_resample})...")
#     timesteps = CONFIG['timesteps']
#     betas = get_beta_schedule(timesteps).to(device)
#     alphas = 1. - betas
#     alphas_cumprod = torch.cumprod(alphas, dim=0)

#     x = torch.randn((1, CONFIG['latent_channels'], CONFIG['image_size'], CONFIG['image_size'], CONFIG['image_size'])).to(device)
    
#     # 我们只在中间阶段进行重采样，节省时间
#     pbar = tqdm(reversed(range(timesteps)), total=timesteps)
    
#     for t in pbar:
#         # 动态调整重采样次数
#         current_resample = n_resample if (100 < t < 900) else 1
        
#         for r in range(current_resample):
#             # --- Step 1: 逆向 (Reverse) x_t -> x_{t-1} ---
#             t_tensor = torch.full((1,), t, device=device, dtype=torch.long)
#             model_input = torch.cat([x, condition, mask], dim=1)
#             noise_pred = model(model_input, t_tensor, porosity)
            
#             beta_t = betas[t]
#             alpha_t = alphas[t]
#             alpha_bar_t = alphas_cumprod[t]
            
#             if t > 0:
#                 noise = torch.randn_like(x)
#             else:
#                 noise = torch.zeros_like(x)
            
#             # 预测出的未知区域的 x_{t-1}
#             x_prev_unknown = (1 / torch.sqrt(alpha_t)) * (x - (beta_t / torch.sqrt(1 - alpha_bar_t)) * noise_pred) + torch.sqrt(beta_t) * noise
            
#             # --- Step 2: 注入已知 (Condition) ---
#             if t > 0:
#                 noise_gt = torch.randn_like(condition)
#                 alpha_bar_prev = alphas_cumprod[t-1]
#                 # GT 的 x_{t-1}
#                 x_prev_known = torch.sqrt(alpha_bar_prev) * condition + torch.sqrt(1 - alpha_bar_prev) * noise_gt
                
#                 # 融合
#                 x_prev = x_prev_unknown * (1 - mask) + x_prev_known * mask
#             else:
#                 x_prev = x_prev_unknown * (1 - mask) + condition * mask

#             # --- Step 3: 时间回溯 (Forward) x_{t-1} -> x_t ---
#             # 如果还需要重采样，就把 x_{t-1} 再加噪变回 x_t，让模型重画一次
#             if r < current_resample - 1 and t > 0:
#                 beta_next = betas[t-1] # 注意这里的索引对应的 beta
#                 noise_add = torch.randn_like(x_prev)
#                 x = torch.sqrt(1 - beta_next) * x_prev + torch.sqrt(beta_next) * noise_add
#             else:
#                 x = x_prev
                
#     return x

# ==========================================
# 3. 主程序 (Main Pipeline)
# ==========================================

def main():
    # --- 配置与环境 ---
    device = CONFIG['device']
    root = get_root_path()
    
    # 1. 确定模型路径
    models_dir = os.path.join(root, "exp_results", CONFIG['experiment_name'], "models")
    model_files = sorted(glob.glob(os.path.join(models_dir, "unet_epoch_*.pth")), key=os.path.getmtime)
    if not model_files: print("❌ No models found"); return
    # print(model_files)
    model_path = model_files[-1] 
    
    # 2. 加载模型
    diffusion_model, vqvae_model = load_models(model_path, device)
    
    # 3. 准备输出目录
    save_dir = os.path.join(root, "exp_results", CONFIG['experiment_name'], "inference_outputs", "final_test")
    os.makedirs(save_dir, exist_ok=True)
    
    # 4. 获取测试数据
    data_files = glob.glob(os.path.join(CONFIG['processed_data_dir'][1], "*.npy"))
    if not data_files: print("❌ No test data found"); return
    sample_file = random.choice(data_files) # 随机取一个
    fname = os.path.basename(sample_file)
    print(f"\n📄 Processing: {fname}")

    # ================= 数据流转 (Data Flow) =================

    # Step A: 加载原始数据
    gt_raw = np.load(sample_file)
    gt_tensor = torch.from_numpy(gt_raw).float().to(device)

    # Step B: 缩放 (Entering Diffusion Space) -> ⚠️ 唯一一次放大
    scale = CONFIG['scale_factor']
    gt_scaled = gt_tensor * scale
    safe_thresh = CONFIG.get('safe_threshold', 4.0)
    gt_scaled = torch.clamp(gt_scaled, min=-safe_thresh, max=safe_thresh)
    # gt_scaled = gt_tensor / 2.0
    
    # Step C: 创建 Mask (自定义逻辑)
    # 这里演示：Core Extension (保留 Top 50% / 32层)
    D = 64
    mask = torch.zeros_like(gt_scaled) # (1, C, D, H, W)
    split_point = int(D * 0.5) # 自定义 Mask 大小：50%
    mask[..., :split_point, :, :] = 1.0 
    
    # 只有 Mask 是单通道的，需要切一下或者保持一致
    # LatentUNet 的 mask 输入通常是 (B, 1, D, H, W)
    mask_input = mask[:, 0:1, ...] 

    # Condition = Scaled Latent * Mask
    condition = gt_scaled * mask_input

    # Step D: 提取孔隙度
    match = re.search(r'porosity_(\d+\.\d+)', fname)
    porosity_val = float(match.group(1)) if match else 0.15
    porosity = torch.tensor([porosity_val]).to(device).view(1,1)

    # Step E: 采样 (Sampling) -> 保持 Scaled 状态
    # 切换这里来测试不同策略: 'simple', 'ddim', 'repaint'
    # MODE = 'simple'
    MODE = 'ddim'
    # MODE = 'repaint' 
    
    with torch.no_grad():
        if MODE == 'simple':
            gen_scaled = simple_sample(diffusion_model, condition, mask_input, porosity, device)
        elif MODE == 'ddim':
            gen_scaled = ddim_sample(diffusion_model, condition, mask_input, porosity, device, ddim_steps=200)
        elif MODE == 'repaint':
            gen_scaled = repaint_sample(diffusion_model, condition, mask_input, porosity, device, n_resample=5)

        
        # Step F: 还原 (Leaving Diffusion Space) -> ⚠️ 唯一一次缩小
        gen_restored = gen_scaled / scale
        # gen_restored = gen_scaled * 2
        gt_restored = gt_tensor # 原始的 tensor 本来就是没缩放的，或者 gt_scaled / scale
        

        # === DEBUG 专用 ===
        print("\n🐛 Debugging Values before Decode:")
        print(f"  GT (Restored)  -> Min: {gt_restored.min():.4f}, Max: {gt_restored.max():.4f}, Mean: {gt_restored.mean():.4f}, Std: {gt_restored.std():.4f}")
        print(f"  Gen (Restored) -> Min: {gen_restored.min():.4f}, Max: {gen_restored.max():.4f}, Mean: {gen_restored.mean():.4f}, Std: {gen_restored.std():.4f}")
        
        # 强制检查：如果 Gen 的范围远远超过 GT (比如 GT是-1到1，Gen是-10到10)
        # 那么一定是 Scale 没除对，或者模型发散了
        if gen_restored.std() > gt_restored.std() * 3:
            print("⚠️ WARNING: Generated values are way too high! Trying to force rescale...")
            # 死马当活马医：强制把 Gen 的分布拉回 GT 的分布
            gen_restored = (gen_restored - gen_restored.mean()) / gen_restored.std() * gt_restored.std() + gt_restored.mean()
            print(f"  Fixed Gen      -> Min: {gen_restored.min():.4f}, Max: {gen_restored.max():.4f}")
        # ==================
        
        # Step G: 解码 (Decoding to Pixel)
        print("  🎨 Decoding to Pixel Space...")
        recon_gen = vqvae_model.decode(gen_restored) # (1, 1, 256, 256, 256)
        recon_gt = vqvae_model.decode(gt_restored)
        
        

    # ================= 结果保存与可视化 =================
    # 转 Numpy
    vol_gen = recon_gen[0, 0].cpu().numpy()
    vol_gt = recon_gt[0, 0].cpu().numpy()
    
    # 生成 Pixel 级的 Condition (用于可视化)
    # Mask 插值: Latent(64) -> Pixel(256)
    mask_pixel = torch.nn.functional.interpolate(mask_input, scale_factor=4, mode='nearest')
    mask_pixel_np = mask_pixel[0, 0].cpu().numpy()
    
    vol_cond = vol_gt.copy()
    vol_cond[mask_pixel_np == 0] = 0 # 抹去未知区域

    # 1. 保存 NifTI
    nib.save(nib.Nifti1Image(vol_gen, np.eye(4)), os.path.join(save_dir, f"{fname}_{MODE}_gen.nii.gz"))
    
    # 2. 统一可视化
    viz_path = os.path.join(save_dir, f"{fname}_{MODE}_LDM.png")
    visualize_inference_results(vol_gt, vol_cond, vol_gen, mask_pixel_np, viz_path, fname)
    vis_path = os.path.join(save_dir, f"{fname}_{MODE}_Pixel.png")
    visualize_three_planes(vol_gt, vol_cond, vol_gen, D*2, vis_path, sample_name=f"{fname}_{MODE}")

if __name__ == "__main__":
    main()
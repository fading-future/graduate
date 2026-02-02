import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg') # 关键：禁用 GUI 后端
import matplotlib.pyplot as plt
import os
import glob
import math
import random
import re
import yaml
import nibabel as nib
from tqdm import tqdm

# --- 项目依赖 ---
from src.model_latent import ConditionalLatentUNet
from src.models.vae import KLVAE3D  # <--- 修改：导入 KLVAE
from src.config import CONFIG
from utils.get_root_path import get_project_root

# ================= 配置区域 =================
# 【请务必修改这里】Stage 1 KL-VAE 的配置文件路径
VAE_CONFIG_PATH = r"/chendou_space/chendou/paper_code/stage02_KLvae_single_code_v2/config/train_config.yaml" 
# FIXED_SAMPLE = "/chendou_space/data/stage2_latents_full_256/porosity_0.160917_6-6-20 全部_z4032_y506_x585.npy"
FIXED_SAMPLE = None
SAFE_LIMIT = CONFIG['safe_threshold'] 
# 0) Sanity 开关：None / "all1" / "all0"
# SANITY = "all1"
SANITY = None
# 0) 是否启用直方图匹配（默认关：先看结构）
FORCE_DISABLE_HIST_MATCH = False
# ===========================================


# 1. 辅助与可视化模块 (Visualization)
def get_cosine_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)

def load_models(model_path, device):
    """加载 Stage1(KL-VAE) 和 Stage2(Diffusion)"""
    print(f"🔹 Loading Diffusion Model: {os.path.basename(model_path)}")
    
    # 1. 实例化 Stage 2 模型 (Latent Diffusion)
    diffusion_model = ConditionalLatentUNet(
        in_channels=CONFIG['in_channels'],       
        out_channels=CONFIG['out_channels'],
        base_channels=CONFIG['base_channels'],    
        channel_mults=CONFIG['channel_mults'],
        use_attention=CONFIG['use_attention'],
    ).to(device)

    # 2. 智能加载 Diffusion 权重
    checkpoint = torch.load(model_path, map_location=device)
    
    if isinstance(checkpoint, dict):
        if 'ema_state_dict' in checkpoint:
            print("✨ Successfully loaded EMA weights for Stage 2.")
            diffusion_model.load_state_dict(checkpoint['ema_state_dict'])
        elif 'model_state_dict' in checkpoint:
            print("⚠️ EMA weights not found, using raw model weights.")
            diffusion_model.load_state_dict(checkpoint['model_state_dict'])
        else:
            diffusion_model.load_state_dict(checkpoint)
    else:
        print("📜 Loading raw state_dict (Old format).")
        diffusion_model.load_state_dict(checkpoint)
    
    diffusion_model.eval()

    # 3. 加载 Stage 1 KL-VAE
    print(f"🔹 Loading KL-VAE Model from: {CONFIG['stage1_model_path']}")
    
    # [关键修改] 读取 YAML 配置来初始化 KL-VAE
    if not os.path.exists(VAE_CONFIG_PATH):
        raise FileNotFoundError(f"❌ VAE Config not found: {VAE_CONFIG_PATH}")
        
    with open(VAE_CONFIG_PATH, 'r') as f:
        vae_config = yaml.safe_load(f)
    
    # 初始化模型
    vae_model = KLVAE3D(vae_config).to(device)
    
    # 加载权重
    vae_checkpoint = torch.load(CONFIG['stage1_model_path'], map_location=device)
    
    # 处理可能的权重格式差异
    state_dict = None
    if isinstance(vae_checkpoint, dict) and 'vae_state_dict' in vae_checkpoint:
        state_dict = vae_checkpoint['vae_state_dict']
    elif isinstance(vae_checkpoint, dict) and 'model_state_dict' in vae_checkpoint:
        state_dict = vae_checkpoint['model_state_dict']
    else:
        state_dict = vae_checkpoint

    # 处理 _orig_mod 前缀 (如果用了 torch.compile)
    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k.replace('_orig_mod.', '')
        new_state_dict[new_key] = v
        
    vae_model.load_state_dict(new_state_dict)     
    vae_model.eval()
    
    return diffusion_model, vae_model

def visualize_inference_results(gt_vol, input_cond_vol, gen_vol, mask_pixel_np, save_path, fname):
    """
    可视化函数：保持不变，通用逻辑
    """
    flat_gt = gt_vol.flatten()
    vmin, vmax = np.percentile(flat_gt, 1), np.percentile(flat_gt, 99)
    dist = vmax - vmin
    if dist < 1e-5: dist = 1.0
    vmin -= dist * 0.1
    vmax += dist * 0.1
    
    # 寻找 Mask 边界
    z_profile = mask_pixel_np.mean(axis=(1, 2))
    split_indices = np.where(np.diff(z_profile) != 0)[0]
    split_idx = split_indices[0] if len(split_indices) > 0 else -1

    D, H, W = gt_vol.shape
    cz, cy, cx = D // 2, H // 2, W // 2

    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    plt.suptitle(f"Inference: {fname}", fontsize=16, y=0.98)
    
    cols = ["Ground Truth", "Input Condition", "Generated"]
    data_list = [gt_vol, input_cond_vol, gen_vol]

    def draw_split_line(ax, axis_name):
        if split_idx > 0:
            if axis_name == 'Z':
                ax.axhline(y=split_idx, color='red', linestyle='--', linewidth=2, alpha=0.9)

    for col_idx, (title, vol) in enumerate(zip(cols, data_list)):
        # XY (Z-cut)
        ax = axes[0, col_idx]
        ax.imshow(vol[cz, :, :], cmap='gray', vmin=vmin, vmax=vmax)
        ax.set_title(f"{title}\nXY (Z={cz})")
        ax.axis('off')

        # XZ (Y-cut)
        ax = axes[1, col_idx]
        ax.imshow(vol[:, cy, :], cmap='gray', vmin=vmin, vmax=vmax, origin='lower')
        ax.set_title(f"{title}\nXZ (Y={cy})")
        draw_split_line(ax, 'Z')
        ax.axis('off')

        # ZY (X-cut)
        ax = axes[2, col_idx]
        ax.imshow(vol[:, :, cx], cmap='gray', vmin=vmin, vmax=vmax, origin='lower')
        ax.set_title(f"{title}\nZY (X={cx})")
        draw_split_line(ax, 'Z')
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  🖼️ Saved visualization: {save_path}")

import torch
from tqdm import tqdm

def ddim_sample(model, condition, mask, porosity, device, ddim_steps=50, seed: int | None = 1234):
    """
    DDIM sampling for conditional inpainting in latent space.
    condition: (B, C, D, H, W)  -> known-region latent (GT * mask)
    mask:      (B, 1, D, H, W)  -> 1=known, 0=unknown
    porosity:  (B, 1) or (B,)   -> conditioning scalar
    seed:      int or None      -> make sampling deterministic for fair comparison
    """
    model.eval()

    # steps
    ddim_steps = ddim_steps if ddim_steps is not None else CONFIG.get('ddim_steps_infer', 200)
    total_timesteps = int(CONFIG['timesteps'])
    SAFE_LIMIT = float(CONFIG.get('safe_threshold', 6.0))

    print(f"  🚀 Strategy: DDIM Sampling ({ddim_steps} steps), total T={total_timesteps}, seed={seed}")

    # --- schedule (must match training) ---
    betas = get_cosine_schedule(total_timesteps).to(device)          # (T,)
    alphas = 1.0 - betas                                             # (T,)
    alphas_cumprod = torch.cumprod(alphas, dim=0)                    # (T,)

    # --- timesteps for DDIM ---
    times = torch.linspace(0, total_timesteps - 1, steps=ddim_steps, device=device)
    times = torch.unique(torch.round(times).long(), sorted=True)
    times = list(reversed(times.tolist()))  # e.g. [999, ..., 0]

    B = condition.shape[0]

    # =======================
    # Deterministic noises
    # =======================
    # Use a per-call generator so this doesn't affect global randomness elsewhere
    if seed is not None:
        g = torch.Generator(device=device)
        g.manual_seed(int(seed))
        x = torch.randn(condition.shape, device=device, dtype=condition.dtype, generator=g)
        fixed_noise = torch.randn(condition.shape, device=device, dtype=condition.dtype, generator=g)
    else:
        # fallback: random each call
        x = torch.randn_like(condition)
        fixed_noise = torch.randn_like(condition)

    # --- IMPORTANT: initialize known region at the starting timestep ---
    t_start = times[0]
    alpha_bar_start = alphas_cumprod[t_start]     # scalar tensor
    known_xt = torch.sqrt(alpha_bar_start) * condition + torch.sqrt(1.0 - alpha_bar_start) * fixed_noise
    x = x * (1.0 - mask) + known_xt * mask
    x = torch.clamp(x, -SAFE_LIMIT, SAFE_LIMIT)

    with torch.no_grad():
        for i, t in enumerate(tqdm(times, desc="DDIM Sampling")):
            t_tensor = torch.full((B,), t, device=device, dtype=torch.long)
            t_prev = times[i + 1] if i < len(times) - 1 else -1

            # model forward
            model_input = torch.cat([x, condition, mask], dim=1)
            noise_pred = model(model_input, t_tensor, porosity)

            # alpha bars
            alpha_bar_t = alphas_cumprod[t]
            alpha_bar_prev = alphas_cumprod[t_prev] if t_prev >= 0 else torch.tensor(1.0, device=device)

            # pred x0
            pred_x0 = (x - torch.sqrt(1.0 - alpha_bar_t) * noise_pred) / (torch.sqrt(alpha_bar_t) + 1e-8)
            pred_x0 = torch.clamp(pred_x0, -SAFE_LIMIT, SAFE_LIMIT)

            # DDIM update (eta=0)
            x_prev = torch.sqrt(alpha_bar_prev) * pred_x0 + torch.sqrt(1.0 - alpha_bar_prev) * noise_pred

            # known-region injection (use SAME fixed_noise every step)
            if t_prev >= 0:
                known_x_prev = torch.sqrt(alpha_bar_prev) * condition + torch.sqrt(1.0 - alpha_bar_prev) * fixed_noise
                x = x_prev * (1.0 - mask) + known_x_prev * mask
            else:
                x = pred_x0 * (1.0 - mask) + condition * mask

            x = torch.clamp(x, -SAFE_LIMIT, SAFE_LIMIT)

    return x

def repaint_sample(model, condition, mask, porosity, device, n_resample=5):
    """
    策略 C: RePaint (KL-VAE Optimized Version)
    修复了灰白问题，增强了边界融合。
    """
    # 增加 Resample 次数能显著提升大孔隙的连通性
    # 建议 n_resample 设为 10 或 20
    print(f"  🚀 Strategy: Optimized RePaint (Resample={n_resample})...")
    
    total_timesteps = CONFIG['timesteps']
    betas = get_cosine_schedule(total_timesteps).to(device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)

    # 1. 从纯噪声开始
    x = torch.randn_like(condition)  # (1, 4, D, H, W) 由数据决定
    fixed_noise = torch.randn_like(condition)  # 固定一次
    
    # 定义需要进行 Time-Travel 的时间段
    # 我们不需要全过程都回溯，只在生成“结构”的关键阶段回溯
    # 对于 Latent Diffusion，通常是 t=400 到 t=50 的区间决定了孔隙形状
    resample_start = CONFIG.get('repaint_resample_start', 800)
    resample_end   = CONFIG.get('repaint_resample_end', 50)
    
    pbar = tqdm(reversed(range(total_timesteps)), total=total_timesteps, desc="RePaint")
    
    for t in pbar:
        # 动态决定当前步是否回溯
        is_resample_zone = (t < resample_start and t > resample_end)
        current_iter = n_resample if is_resample_zone else 1
        
        for r in range(current_iter):
            # -----------------------------------------------------------
            # A. 逆向一步 (Reverse Step): x_t -> x_{t-1}
            # -----------------------------------------------------------
            t_tensor = torch.full((1,), t, device=device, dtype=torch.long)
            model_input = torch.cat([x, condition, mask], dim=1)
            noise_pred = model(model_input, t_tensor, porosity)
            
            # 使用标准的 DDPM 采样公式 (保留随机性，这对生成纹理很重要)
            alpha_t = alphas[t]
            alpha_bar_t = alphas_cumprod[t]
            beta_t = betas[t]
            
            # 1. 预测 x0
            pred_x0 = (x - torch.sqrt(1 - alpha_bar_t) * noise_pred) / torch.sqrt(alpha_bar_t)
            # 软截断：允许偶尔跳出范围，但限制极端值
            pred_x0 = torch.clamp(pred_x0, -SAFE_LIMIT, SAFE_LIMIT)
            
            # 2. 计算 x_{t-1} 的均值
            posterior_mean = (1 / torch.sqrt(alpha_t)) * (x - (beta_t / torch.sqrt(1 - alpha_bar_t)) * noise_pred)
            
            # 3. 添加噪声 (除非是最后一步)
            if t > 0:
                noise = torch.randn_like(x)
                # sigma_t = sqrt(beta_t) # 标准 DDPM
                sigma_t = torch.sqrt(beta_t * (1. - alphas_cumprod[t-1]) / (1. - alphas_cumprod[t])) # 后验方差 (更稳)
                x_prev = posterior_mean + sigma_t * noise
            else:
                x_prev = posterior_mean

            # -----------------------------------------------------------
            # B. 注入已知区域 (Inject Known) - 解决拼接生硬的核心
            # -----------------------------------------------------------
            if t > 0:
                # 获取 t-1 时刻的 GT 噪声分布
                alpha_bar_prev = alphas_cumprod[t-1]
                known_part = torch.sqrt(alpha_bar_prev) * condition + torch.sqrt(1.0 - alpha_bar_prev) * fixed_noise
                
                # 融合
                x_prev = x_prev * (1 - mask) + known_part * mask
            else:
                # 最后一步直接贴
                x_prev = x_prev * (1 - mask) + condition * mask

            # C. 时间回溯 (Time Travel): x_{t-1} -> x_t
            # 使用标准 forward transition: q(x_t | x_{t-1}) = N(sqrt(alpha_t)*x_{t-1}, (1-alpha_t)I)
            if r < current_iter - 1 and t > 0:
                alpha_forward = alphas[t]  # 对应从 x_{t-1} -> x_t 的那一步
                noise_forward = torch.randn_like(x_prev)
                x = torch.sqrt(alpha_forward) * x_prev + torch.sqrt(1.0 - alpha_forward) * noise_forward
            else:
                x = x_prev

            x = torch.clamp(x, -SAFE_LIMIT, SAFE_LIMIT)
                
    return x


def decode_tiled_klvae(vae_model, z, latent_tile=16, latent_overlap=4, up_factor=8):
    """
    z: [1, C, 32, 32, 32]  (你的 KL-VAE latent)
    输出: [1, 1, 256, 256, 256] (假设 decoder 输出 1 通道)
    latent_tile: 每块 latent 的边长（16 -> 输出 128³）
    latent_overlap: latent 的重叠（建议 4）
    up_factor: 256/32 = 8
    """
    assert z.dim() == 5 and z.size(0) == 1
    B, C, D, H, W = z.shape
    device = z.device

    step = latent_tile - latent_overlap
    assert step > 0

    out_D, out_H, out_W = D * up_factor, H * up_factor, W * up_factor
    out = torch.zeros((1, 1, out_D, out_H, out_W), device=device, dtype=torch.float32)
    wgt = torch.zeros_like(out)

    # 简单的加权窗（避免拼接硬边）
    # 这里用三角窗，够用了
    def tri(n):
        x = torch.linspace(0, 1, n, device=device)
        w = 1.0 - (2.0 * (x - 0.5)).abs()
        return w.clamp_min(0.0)

    # decode 一个 patch，拿到输出通道数（避免你 decoder 不是 1 通道时挂）
    # 但为了少占显存，这里不预跑；假设输出 1 通道，若不是你再改 out 的通道数即可

    for dz in range(0, D, step):
        for dy in range(0, H, step):
            for dx in range(0, W, step):
                z0, y0, x0 = dz, dy, dx
                z1 = min(z0 + latent_tile, D)
                y1 = min(y0 + latent_tile, H)
                x1 = min(x0 + latent_tile, W)

                # 向后对齐，保证 tile 尺寸一致（尤其到边界）
                z0 = max(0, z1 - latent_tile)
                y0 = max(0, y1 - latent_tile)
                x0 = max(0, x1 - latent_tile)

                patch = z[:, :, z0:z1, y0:y1, x0:x1]

                with torch.no_grad():
                    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                        dec = vae_model.decode(patch)  # [1, 1, tile*8, tile*8, tile*8] 期望

                dec = dec.float()

                # 对应到像素坐标
                oz0, oy0, ox0 = z0 * up_factor, y0 * up_factor, x0 * up_factor
                oz1, oy1, ox1 = z1 * up_factor, y1 * up_factor, x1 * up_factor

                # 生成权重窗（按实际 patch 大小）
                pD, pH, pW = dec.shape[-3:]
                wz = tri(pD).view(1, 1, pD, 1, 1)
                wy = tri(pH).view(1, 1, 1, pH, 1)
                wx = tri(pW).view(1, 1, 1, 1, pW)
                ww = (wz * wy * wx).float()

                out[:, :, oz0:oz1, oy0:oy1, ox0:ox1] += dec * ww
                wgt[:, :, oz0:oz1, oy0:oy1, ox0:ox1] += ww

                # 及时释放 patch 的 decoder 输出缓存（减少峰值）
                del dec, patch
                torch.cuda.empty_cache()

    out = out / (wgt + 1e-8)
    return out


# 3. 主程序 (Main Pipeline)
def main():
    device = CONFIG['device']
    root = get_project_root()

    # 1. 确定 Stage 2 模型路径
    models_dir = os.path.join(root, "exp_results", CONFIG['experiment_name'], "models")
    model_files = sorted(glob.glob(os.path.join(models_dir, "unet_epoch_*.pth")), key=os.path.getmtime)
    if not model_files:
        print("❌ No models found")
        return
    model_path = model_files[-1]

    # 2. 加载模型 (KL-VAE + Diffusion)
    diffusion_model, vae_model = load_models(model_path, device)

    # 3. 准备输出目录
    save_dir = os.path.join(root, "exp_results", CONFIG['experiment_name'], "inference_outputs", "final_test_klvae")
    os.makedirs(save_dir, exist_ok=True)

    # 4. 获取测试数据 (Latent .npy)
    data_files = glob.glob(os.path.join(CONFIG['processed_data_dir'], "*.npy"))
    if not data_files and isinstance(CONFIG['processed_data_dir'], list):
        for d in CONFIG['processed_data_dir']:
            data_files += glob.glob(os.path.join(d, "*.npy"))

    if not data_files:
        print("❌ No test data found")
        return

    if FIXED_SAMPLE is not None:
        sample_file = FIXED_SAMPLE
        if not os.path.exists(sample_file):
            print(f"❌ FIXED_SAMPLE not found: {sample_file}")
            return
    else:
        sample_file = random.choice(data_files)

    fname = os.path.basename(sample_file)
    print(f"\n📄 Processing: {fname}")

    # Step A: 加载原始数据 (Latent)
    gt_raw = np.load(sample_file)
    gt_tensor = torch.from_numpy(gt_raw).float().to(device)
    if gt_tensor.dim() == 4:
        gt_tensor = gt_tensor.unsqueeze(0)

    # Step B: 缩放 (Entering Diffusion Space)
    scale = CONFIG['scale_factor']
    gt_scaled = gt_tensor * scale
    # gt_scaled = gt_tensor

    safe_thresh = CONFIG.get('safe_threshold', 6.0)
    gt_scaled = torch.clamp(gt_scaled, min=-safe_thresh, max=safe_thresh)

    # Step C: 创建 Mask
    D = gt_scaled.shape[2]
    mask = torch.zeros_like(gt_scaled)
    split_point = int(D * 0.5)
    mask[..., :split_point, :, :] = 1.0

    mask_input = mask[:, 0:1, ...]

    if SANITY == "all1":
        mask_input[:] = 1.0
    elif SANITY == "all0":
        mask_input[:] = 0.0

    condition = gt_scaled * mask_input

    # Step D: 提取孔隙度
    match = re.search(r'porosity_([0-9]*\.?[0-9]+)', fname)
    porosity_val = float(match.group(1)) if match else 0.15
    porosity = torch.tensor([porosity_val]).to(device).view(1, 1)

    print("gt_tensor", gt_tensor.mean().item(), gt_tensor.std().item(), gt_tensor.min().item(), gt_tensor.max().item())
    print("gt_scaled", gt_scaled.mean().item(), gt_scaled.std().item(), gt_scaled.min().item(), gt_scaled.max().item())
    print("mask", mask_input.mean().item(), mask_input.min().item(), mask_input.max().item())
    print("condition", condition.mean().item(), condition.std().item(), condition.min().item(), condition.max().item())


    # Step E: 采样
    MODE = 'ddim'  # 或 'repaint'

    with torch.no_grad():
        if MODE == 'ddim':
            gen_scaled = ddim_sample(diffusion_model, condition, mask_input, porosity, device, ddim_steps=600)
        elif MODE == 'repaint':
            gen_scaled = repaint_sample(diffusion_model, condition, mask_input, porosity, device, n_resample=10)
        else:
            print(f"❌ Unknown MODE: {MODE}")
            return

        # Step F: 还原 (Leaving Diffusion Space)
        gen_restored = gen_scaled / scale
        gt_restored = gt_tensor

        # Step G: KL-VAE 解码 (Latent -> Pixel) - 使用分块解码
        print("  🎨 Decoding with KL-VAE...")
        with torch.amp.autocast('cuda'):
            recon_gen = decode_tiled_klvae(vae_model, gen_restored, latent_tile=16, latent_overlap=4, up_factor=8)
            recon_gt  = decode_tiled_klvae(vae_model, gt_restored,  latent_tile=16, latent_overlap=4, up_factor=8)

    # 结果转 numpy
    vol_gen = recon_gen[0, 0].cpu().float().numpy()
    vol_gt  = recon_gt[0, 0].cpu().float().numpy()

    # Mask 插值到 Pixel 尺寸（用于可视化边界）
    latent_dim = gt_tensor.shape[2]
    pixel_dim = vol_gen.shape[0]
    upscale_factor = pixel_dim // latent_dim
    mask_pixel = torch.nn.functional.interpolate(mask_input, scale_factor=upscale_factor, mode='nearest')
    mask_pixel_np = mask_pixel[0, 0].cpu().numpy()

    # Input Condition 可视化：用 GT 的像素域遮罩（避免 decode masked-latent 产生棋盘伪影）
    vol_cond_viz = vol_gt.copy()
    vol_cond_viz[mask_pixel_np == 0] = vol_gt.min()

    # 直方图匹配（默认关闭，先看结构）
    print("  🎨 Applying Histogram Matching...")
    def match_histogram(source, template):
        oldshape = source.shape
        source = source.ravel()
        template = template.ravel()
        s_values, bin_idx, s_counts = np.unique(source, return_inverse=True, return_counts=True)
        t_values, t_counts = np.unique(template, return_counts=True)
        s_quantiles = np.cumsum(s_counts).astype(np.float64)
        s_quantiles /= s_quantiles[-1]
        t_quantiles = np.cumsum(t_counts).astype(np.float64)
        t_quantiles /= t_quantiles[-1]
        interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)
        return interp_t_values[bin_idx].reshape(oldshape)

    vol_gen_matched = vol_gen
    if (not FORCE_DISABLE_HIST_MATCH) and CONFIG.get('enable_hist_match', False):
        vol_gen_matched = match_histogram(vol_gen, vol_gt)

    # 可视化输出
    viz_path = os.path.join(save_dir, f"{fname}_{MODE}_LDM_matched.png")
    visualize_inference_results(vol_gt, vol_cond_viz, vol_gen_matched, mask_pixel_np, viz_path, fname)

if __name__ == "__main__":
    main()
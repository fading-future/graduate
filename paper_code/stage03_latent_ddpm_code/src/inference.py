import torch
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
VAE_CONFIG_PATH = r"/chendou_space/chendou/paper_code/stage02_KLvae_single_code_v2/config/train_config copy.yaml" 
# ===========================================

# ==========================================
# 1. 辅助与可视化模块 (Visualization)
# ==========================================

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

# ==========================================
# 2. 核心采样策略 (DDIM & RePaint)
# ==========================================

SAFE_LIMIT = CONFIG['safe_threshold'] 

def ddim_sample(model, condition, mask, porosity, device, ddim_steps=50):
    """
    DDIM 采样，保持通用逻辑
    """
    ddim_steps = ddim_steps if ddim_steps is not None else CONFIG.get('ddim_steps_infer', 200)
    print(f"  🚀 Strategy: Optimized DDIM Sampling ({ddim_steps} steps)...")
    
    total_timesteps = CONFIG['timesteps']
    betas = get_cosine_schedule(total_timesteps).to(device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    times = torch.linspace(0, total_timesteps - 1, steps=ddim_steps, device=device)
    times = torch.unique(torch.round(times).long(), sorted=True)
    times = list(reversed(times.tolist()))
    
    # [关键] 这里的 latent_channels 要和 KLVAE 压缩后的通道数一致
    x = torch.randn_like(condition)  # (1, 4, D, H, W) 由数据决定

    fixed_noise = torch.randn_like(condition)  # 固定一次

    with torch.no_grad():
        for i, t in enumerate(tqdm(times, desc="DDIM Sampling")):
            t_tensor = torch.full((1,), t, device=device, dtype=torch.long)
            t_prev = times[i + 1] if i < len(times) - 1 else -1
            
            model_input = torch.cat([x, condition, mask], dim=1)
            noise_pred = model(model_input, t_tensor, porosity)
            
            alpha_bar_t = alphas_cumprod[t]
            alpha_bar_t_prev = alphas_cumprod[t_prev] if t_prev >= 0 else torch.tensor(1.0).to(device)
            
            pred_x0 = (x - torch.sqrt(1.0 - alpha_bar_t) * noise_pred) / torch.sqrt(alpha_bar_t)
            pred_x0 = torch.clamp(pred_x0, -SAFE_LIMIT, SAFE_LIMIT)
            
            pred_dir_xt = torch.sqrt(1.0 - alpha_bar_t_prev) * noise_pred
            x_prev = torch.sqrt(alpha_bar_t_prev) * pred_x0 + pred_dir_xt
            
            if t_prev >= 0:
                known_x_prev = torch.sqrt(alpha_bar_t_prev) * condition + torch.sqrt(1.0 - alpha_bar_t_prev) * fixed_noise
                x = x_prev * (1 - mask) + known_x_prev * mask
            else:
                x = pred_x0 * (1 - mask) + condition * mask
                
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


# ==========================================
# 3. 主程序 (Main Pipeline)
# ==========================================

def main():
    device = CONFIG['device']
    root = get_project_root()
    
    # 1. 确定 Stage 2 模型路径
    models_dir = os.path.join(root, "exp_results", CONFIG['experiment_name'], "models")
    model_files = sorted(glob.glob(os.path.join(models_dir, "unet_epoch_*.pth")), key=os.path.getmtime)
    if not model_files: print("❌ No models found"); return
    model_path = model_files[-1] 
    
    # 2. 加载模型 (KL-VAE + Diffusion)
    diffusion_model, vae_model = load_models(model_path, device)
    
    # 3. 准备输出目录
    save_dir = os.path.join(root, "exp_results", CONFIG['experiment_name'], "inference_outputs", "final_test_klvae")
    os.makedirs(save_dir, exist_ok=True)
    
    # 4. 获取测试数据 (Latent .npy)
    # 注意：这里我们直接读取处理好的 Latent NPY，这样保证了和训练时一致
    data_files = glob.glob(os.path.join(CONFIG['processed_data_dir'], "*.npy"))
    if not data_files: 
         # 尝试从 config 的列表里找
         if isinstance(CONFIG['processed_data_dir'], list):
             for d in CONFIG['processed_data_dir']:
                 data_files += glob.glob(os.path.join(d, "*.npy"))
    
    if not data_files: print("❌ No test data found"); return
    sample_file = random.choice(data_files)
    fname = os.path.basename(sample_file)
    print(f"\n📄 Processing: {fname}")

    # ================= 数据流转 (Data Flow) =================

    # Step A: 加载原始数据 (Latent)
    gt_raw = np.load(sample_file)
    gt_tensor = torch.from_numpy(gt_raw).float().to(device)
    if gt_tensor.dim() == 4: # (C, D, H, W) -> (1, C, D, H, W)
        gt_tensor = gt_tensor.unsqueeze(0)

    # Step B: 缩放 (Entering Diffusion Space)
    scale = CONFIG['scale_factor']
    gt_scaled = gt_tensor * scale
    
    # 截断保护
    safe_thresh = CONFIG.get('safe_threshold', 6.0)
    gt_scaled = torch.clamp(gt_scaled, min=-safe_thresh, max=safe_thresh)
    
    # Step C: 创建 Mask
    D = gt_scaled.shape[2]
    mask = torch.zeros_like(gt_scaled) # (1, C, D, H, W)
    split_point = int(D * 0.5) 
    # Core Extension Task: 已知 Top 50%
    mask[..., :split_point, :, :] = 1.0 
    
    # Mask 输入只需要单通道
    mask_input = mask[:, 0:1, ...] 

    # Condition
    condition = gt_scaled * mask_input

    # Step D: 提取孔隙度
    match = re.search(r'porosity_([0-9]*\.?[0-9]+)', fname)
    porosity_val = float(match.group(1)) if match else 0.15
    porosity = torch.tensor([porosity_val]).to(device).view(1,1)

    # Step E: 采样
    MODE = 'ddim' # 或 'repaint'
    # MODE = 'repaint' # 或 'repaint'
    
    with torch.no_grad():
        if MODE == 'ddim':
            gen_scaled = ddim_sample(diffusion_model, condition, mask_input, porosity, device, ddim_steps=500)
        elif MODE == 'repaint':
            gen_scaled = repaint_sample(diffusion_model, condition, mask_input, porosity, device, n_resample=10)

        # Step F: 还原 (Leaving Diffusion Space)
        gen_restored = gen_scaled / scale
        gt_restored = gt_tensor
        
        # 强制修正：如果生成的 std 异常大，强制拉回正常范围
        if gen_restored.std() > gt_restored.std() * 3:
            print("⚠️ WARNING: Generated values too high! Rescaling...")
            gen_restored = (gen_restored - gen_restored.mean()) / gen_restored.std() * gt_restored.std() + gt_restored.mean()

        # Step G: KL-VAE 解码 (Latent -> Pixel)
        print("  🎨 Decoding with KL-VAE...")
        
        # ⚠️ 注意: KL-VAE decode 输入通常不需要特别的归一化，只要维度对即可
        # 记得 autocast，因为 VAE 显存占用大
        with torch.amp.autocast('cuda'):
            recon_gen = vae_model.decode(gen_restored) 
            recon_gt = vae_model.decode(gt_restored)
            
            # 创建仅包含 Condition 的 Latent 进行解码 (用于可视化)
            # 注意: 直接解码 mask 过的 Latent 可能会有伪影，但为了可视化 Condition 是必要的
            cond_restored = condition / scale
            recon_cond = vae_model.decode(cond_restored)

    # ================= 结果保存与可视化 =================
    vol_gen = recon_gen[0, 0].cpu().float().numpy()
    vol_gt = recon_gt[0, 0].cpu().float().numpy()
    vol_cond_viz = recon_cond[0, 0].cpu().float().numpy()
    
    # 计算 Latent 到 Pixel 的缩放倍数 (用于可视化 Mask 边界)
    # 比如 Latent 64, Pixel 256 -> factor 4
    latent_dim = gt_tensor.shape[2]
    pixel_dim = vol_gen.shape[0]
    upscale_factor = pixel_dim // latent_dim
    
    # Mask 插值到 Pixel 尺寸
    mask_pixel = torch.nn.functional.interpolate(mask_input, scale_factor=upscale_factor, mode='nearest')
    mask_pixel_np = mask_pixel[0, 0].cpu().numpy()
    
    # 让 Condition 可视化更直观：未知区域抹黑
    # vol_cond_viz[mask_pixel_np == 0] = -1 # 或者设为背景色

    # 保存 NifTI
    nib.save(nib.Nifti1Image(vol_gen, np.eye(4)), os.path.join(save_dir, f"{fname}_{MODE}_gen.nii.gz"))
    
    # 可视化
    viz_path = os.path.join(save_dir, f"{fname}_{MODE}_LDM.png")
    visualize_inference_results(vol_gt, vol_cond_viz, vol_gen, mask_pixel_np, viz_path, fname)

    # =========== 🔴 新增：直方图匹配 (Histogram Matching) ===========
    # 这就像给生成的图片加了一个“滤镜”，强制它的黑白分布和 GT 一样
    # 如果生成的结构是对的，这一步会让它瞬间变清晰
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

    # 使用 GT 的分布来修正生成的分布
    vol_gen_matched = vol_gen  # 默认：不做匹配就用原结果
    if CONFIG.get('enable_hist_match', False):
        vol_gen_matched = match_histogram(vol_gen, vol_gt)


    # 保存 NifTI (保存修正后的版本)
    nib.save(nib.Nifti1Image(vol_gen_matched, np.eye(4)), os.path.join(save_dir, f"{fname}_{MODE}_gen_matched.nii.gz"))
    
    # 可视化 (使用修正后的版本进行画图)
    viz_path = os.path.join(save_dir, f"{fname}_{MODE}_LDM_matched.png")
    # 注意：这里传入 vol_gen_matched
    visualize_inference_results(vol_gt, vol_cond_viz, vol_gen_matched, mask_pixel_np, viz_path, fname)

if __name__ == "__main__":
    main()
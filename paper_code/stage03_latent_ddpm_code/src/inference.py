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

def get_beta_schedule(timesteps):
    return torch.linspace(1e-4, 0.02, timesteps)

def load_models(model_path, device):
    """加载 Stage1(KL-VAE) 和 Stage2(Diffusion)"""
    print(f"🔹 Loading Diffusion Model: {os.path.basename(model_path)}")
    
    # 1. 实例化 Stage 2 模型 (Latent Diffusion)
    diffusion_model = ConditionalLatentUNet(
        in_channels=CONFIG['in_channels'],       
        out_channels=CONFIG['out_channels'],
        base_channels=CONFIG['base_channels'],    
        channel_mults=(1, 2, 4), 
        use_attention=(False, True, True) 
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
    betas = get_beta_schedule(total_timesteps).to(device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    times = torch.linspace(0, total_timesteps - 1, steps=ddim_steps).long().to(device)
    times = list(reversed(times.tolist()))
    
    # [关键] 这里的 latent_channels 要和 KLVAE 压缩后的通道数一致
    x = torch.randn((1, CONFIG['latent_channels'], CONFIG['image_size'], CONFIG['image_size'], CONFIG['image_size'])).to(device)

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
                noise_real = torch.randn_like(condition)
                known_x_prev = torch.sqrt(alpha_bar_t_prev) * condition + torch.sqrt(1.0 - alpha_bar_t_prev) * noise_real
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
    betas = get_beta_schedule(total_timesteps).to(device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)

    # 1. 从纯噪声开始
    x = torch.randn((1, CONFIG['latent_channels'], CONFIG['image_size'], CONFIG['image_size'], CONFIG['image_size'])).to(device)
    
    # 定义需要进行 Time-Travel 的时间段
    # 我们不需要全过程都回溯，只在生成“结构”的关键阶段回溯
    # 对于 Latent Diffusion，通常是 t=400 到 t=50 的区间决定了孔隙形状
    resample_start = 800
    resample_end = 50
    
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
                noise_gt = torch.randn_like(condition)
                
                # 【关键修复】不要每次都重新采样 noise_gt！
                # 在同一次 Time-Travel 中，应该尽量保持噪声的一致性，否则会变灰
                # 但为了简单，我们先保证能量守恒：
                known_part = torch.sqrt(alpha_bar_prev) * condition + torch.sqrt(1 - alpha_bar_prev) * noise_gt
                
                # 融合
                x_prev = x_prev * (1 - mask) + known_part * mask
            else:
                # 最后一步直接贴
                x_prev = x_prev * (1 - mask) + condition * mask

            # -----------------------------------------------------------
            # C. 时间回溯 (Time Travel): x_{t-1} -> x_t
            # -----------------------------------------------------------
            # 只有当：在重采样区间内 且 不是该时间步的最后一次循环 且 t>0
            if r < current_iter - 1 and t > 0:
                # 加噪回去
                # x_t = sqrt(1-beta)*x_{t-1} + sqrt(beta)*noise
                # 这里使用 forward process 公式
                beta_forward = betas[t] # 或者 betas[t-1]
                noise_forward = torch.randn_like(x_prev)
                
                x = torch.sqrt(1 - beta_forward) * x_prev + torch.sqrt(beta_forward) * noise_forward
            else:
                # 推进到下一步
                x = x_prev
                
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
    match = re.search(r'porosity_(\d+\.\d+)', fname)
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
    vol_gen_matched = match_histogram(vol_gen, vol_gt)
    # ==============================================================

    # 保存 NifTI (保存修正后的版本)
    nib.save(nib.Nifti1Image(vol_gen_matched, np.eye(4)), os.path.join(save_dir, f"{fname}_{MODE}_gen_matched.nii.gz"))
    
    # 可视化 (使用修正后的版本进行画图)
    viz_path = os.path.join(save_dir, f"{fname}_{MODE}_LDM_matched.png")
    # 注意：这里传入 vol_gen_matched
    visualize_inference_results(vol_gt, vol_cond_viz, vol_gen_matched, mask_pixel_np, viz_path, fname)


if __name__ == "__main__":
    main()

# import torch
# import numpy as np
# import matplotlib
# matplotlib.use('Agg') # 关键：禁用 GUI 后端
# import matplotlib.pyplot as plt
# import os
# import glob
# import sys
# import random
# import re
# import nibabel as nib
# from tqdm import tqdm

# # --- 项目依赖 ---
# from model_latent import ConditionalLatentUNet
# from model_vqvae import VQVAE3D 
# from config import CONFIG
# from utils.get_root_path import get_project_root

# # ==========================================
# # 1. 辅助与可视化模块 (Visualization)
# # ==========================================

# def get_beta_schedule(timesteps):
#     return torch.linspace(1e-4, 0.02, timesteps)

# def load_models(model_path, device):
#     """加载 Stage1(VQVAE) 和 Stage2(Diffusion)"""
#     print(f"🔹 Loading Diffusion Model: {os.path.basename(model_path)}")
    
#     # 1. 实例化 Stage 2 模型
#     diffusion_model = ConditionalLatentUNet(
#         in_channels=CONFIG['in_channels'],       # 64+64+1
#         out_channels=CONFIG['out_channels'],
#         base_channels=CONFIG['base_channels'],    
#         channel_mults=(1, 2, 4), 
#         use_attention=(False, True, True) 
#     ).to(device)

#     # 2. 智能加载权重 (处理 EMA 和 Dict 格式)
#     checkpoint = torch.load(model_path, map_location=device)
    
#     if isinstance(checkpoint, dict):
#         # 优先加载 EMA 权重，因为 EMA 权重的生成质量通常更稳定
#         if 'ema_state_dict' in checkpoint:
#             print("✨ Successfully loaded EMA weights for Stage 2.")
#             diffusion_model.load_state_dict(checkpoint['ema_state_dict'])
#         elif 'model_state_dict' in checkpoint:
#             print("⚠️ EMA weights not found, using raw model weights.")
#             diffusion_model.load_state_dict(checkpoint['model_state_dict'])
#         else:
#             # 兼容某些只存了字典但 key 不对的情况
#             diffusion_model.load_state_dict(checkpoint)
#     else:
#         # 兼容旧版本直接保存的 state_dict (纯权重文件)
#         print("📜 Loading raw state_dict (Old format).")
#         diffusion_model.load_state_dict(checkpoint)
    
#     diffusion_model.eval()

#     # 3. 加载 Stage 1 VQ-VAE
#     print(f"🔹 Loading VQ-VAE Model: {os.path.basename(CONFIG['stage1_model_path'])}")
#     vqvae_model = VQVAE3D(
#         in_channels=1,
#         embedding_dim=CONFIG['latent_channels'], 
#         num_embeddings=2048 
#     ).to(device)
    
#     # 注意：如果你的 Stage 1 也是字典格式，这里也需要像上面一样做判断
#     vqvae_checkpoint = torch.load(CONFIG['stage1_model_path'], map_location=device)
#     if isinstance(vqvae_checkpoint, dict) and 'model_state_dict' in vqvae_checkpoint:
#         vqvae_model.load_state_dict(vqvae_checkpoint['model_state_dict'])
#     else:
#         vqvae_model.load_state_dict(vqvae_checkpoint)
        
#     vqvae_model.eval()
    
#     return diffusion_model, vqvae_model

# def visualize_inference_results(gt_vol, input_cond_vol, gen_vol, mask_pixel_np, save_path, fname):
#     """
#     统一可视化函数：3x3 布局 (GT, Condition, Generated) x (XY, XZ, ZY)
#     自动根据 Mask 边缘绘制红线
#     """
#     # 1. 动态对比度 (避免灰色)
#     flat_gt = gt_vol.flatten()
#     vmin, vmax = np.percentile(flat_gt, 1), np.percentile(flat_gt, 99)
#     dist = vmax - vmin
#     if dist < 1e-5: dist = 1.0
#     vmin -= dist * 0.1
#     vmax += dist * 0.1
    
#     # 2. 寻找 Mask 边界 (为了画红线)
#     # 假设 Mask 是 Z 轴方向的扩展 (Core Extension)，我们找 Z 轴上 0 和 1 的交界处
#     # mask_pixel_np: (256, 256, 256), 1=Known, 0=Unknown
#     z_profile = mask_pixel_np.mean(axis=(1, 2)) # 沿 Z 轴的平均值
#     # 找到从 1 变成 0 的那个索引
#     split_indices = np.where(np.diff(z_profile) != 0)[0]
#     split_idx = split_indices[0] if len(split_indices) > 0 else -1

#     # 3. 切片位置 (默认取中心)
#     D, H, W = gt_vol.shape
#     cz, cy, cx = D // 2, H // 2, W // 2

#     # 4. 绘图
#     fig, axes = plt.subplots(3, 3, figsize=(15, 15))
#     plt.suptitle(f"Inference: {fname}", fontsize=16, y=0.98)
    
#     cols = ["Ground Truth", "Input Condition", "Generated"]
#     data_list = [gt_vol, input_cond_vol, gen_vol]

#     # 定义画线辅助函数
#     def draw_split_line(ax, axis_name):
#         if split_idx > 0:
#             if axis_name == 'Z': # 纵轴是 Z
#                 ax.axhline(y=split_idx, color='red', linestyle='--', linewidth=2, alpha=0.9)
#                 ax.text(5, split_idx - 5, 'Known', color='red', fontsize=9, fontweight='bold', va='bottom')
#                 ax.text(5, split_idx + 5, 'Gen', color='yellow', fontsize=9, fontweight='bold', va='top')

#     for col_idx, (title, vol) in enumerate(zip(cols, data_list)):
#         # --- Row 1: XY Plane (Top View, Slice Z) ---
#         ax = axes[0, col_idx]
#         ax.imshow(vol[cz, :, :], cmap='gray', vmin=vmin, vmax=vmax)
#         ax.set_title(f"{title}\nXY (Z={cz})")
#         ax.axis('off')

#         # --- Row 2: XZ Plane (Side View, Slice Y) ---
#         # 纵轴是 Z，横轴是 X
#         ax = axes[1, col_idx]
#         ax.imshow(vol[:, cy, :], cmap='gray', vmin=vmin, vmax=vmax, origin='lower')
#         ax.set_title(f"{title}\nXZ (Y={cy})")
#         ax.set_ylabel("Z (Depth)")
#         ax.set_xticks([])
#         ax.set_yticks([])
#         draw_split_line(ax, 'Z') # 画线

#         # --- Row 3: ZY Plane (Side View, Slice X) ---
#         # 纵轴是 Z，横轴是 Y
#         ax = axes[2, col_idx]
#         ax.imshow(vol[:, :, cx], cmap='gray', vmin=vmin, vmax=vmax, origin='lower')
#         ax.set_title(f"{title}\nZY (X={cx})")
#         ax.set_ylabel("Z (Depth)")
#         ax.set_xticks([])
#         ax.set_yticks([])
#         draw_split_line(ax, 'Z') # 画线

#     plt.tight_layout()
#     plt.savefig(save_path, dpi=150)
#     plt.close()
#     print(f"  🖼️ Saved visualization: {save_path}")

# # ==========================================
# # 2. 核心采样策略 (Sampling Strategies)
# # ==========================================

# # 设定推理时的安全截断阈值
# # 训练时 Scale=4.22, Clamp=5.0
# # 推理时稍微放宽一点，防止硬截断导致纹理死锁，同时遏制指数级漂移
# SAFE_LIMIT = CONFIG['safe_threshold'] 

# def ddim_sample(model, condition, mask, porosity, device, ddim_steps=50, eta=0.0):
#     """
#     策略 B: 优化版 DDIM 采样 (Pred_x0 Clamping)
#     最稳健的方案，强制截断预测出的 x0，从根源上消除数值爆炸。
#     """
#     ddim_steps = ddim_steps if ddim_steps is not None else CONFIG.get('ddim_steps_infer', 200)
#     print(f"  🚀 Strategy: Optimized DDIM Sampling ({ddim_steps} steps)...")
    
#     total_timesteps = CONFIG['timesteps']
#     betas = get_beta_schedule(total_timesteps).to(device)
#     alphas = 1.0 - betas
#     alphas_cumprod = torch.cumprod(alphas, dim=0)

#     # 时间步序列
#     times = torch.linspace(0, total_timesteps - 1, steps=ddim_steps).long().to(device)
#     times = list(reversed(times.tolist()))
    
#     x = torch.randn((1, CONFIG['latent_channels'], CONFIG['image_size'], CONFIG['image_size'], CONFIG['image_size'])).to(device)

#     with torch.no_grad():
#         for i, t in enumerate(tqdm(times, desc="DDIM Sampling")):
#             t_tensor = torch.full((1,), t, device=device, dtype=torch.long)
#             t_prev = times[i + 1] if i < len(times) - 1 else -1
            
#             # --- A. 预测 ---
#             model_input = torch.cat([x, condition, mask], dim=1)
#             noise_pred = model(model_input, t_tensor, porosity)
            
#             # --- B. 数学推导 (x_t -> pred_x0 -> x_{t-1}) ---
#             alpha_bar_t = alphas_cumprod[t]
#             alpha_bar_t_prev = alphas_cumprod[t_prev] if t_prev >= 0 else torch.tensor(1.0).to(device)
            
#             # 1. 预测 x0 (Denoised)
#             pred_x0 = (x - torch.sqrt(1.0 - alpha_bar_t) * noise_pred) / torch.sqrt(alpha_bar_t)
            
#             # 【防护 1】核心：对预测的 x0 进行截断
#             # 这是防止雪花屏最关键的一步
#             pred_x0 = torch.clamp(pred_x0, -SAFE_LIMIT, SAFE_LIMIT)
            
#             # 2. 指向 x_{t-1} 的方向
#             pred_dir_xt = torch.sqrt(1.0 - alpha_bar_t_prev) * noise_pred
            
#             # 3. 重构 x_{t-1}
#             x_prev = torch.sqrt(alpha_bar_t_prev) * pred_x0 + pred_dir_xt
            
#             # --- C. 注入已知区域 ---
#             if t_prev >= 0:
#                 # 给 GT 加上 t_prev 时刻的噪声
#                 noise_real = torch.randn_like(condition)
#                 known_x_prev = torch.sqrt(alpha_bar_t_prev) * condition + torch.sqrt(1.0 - alpha_bar_t_prev) * noise_real
                
#                 x = x_prev * (1 - mask) + known_x_prev * mask
#             else:
#                 # 最后一步：直接使用最清晰的 pred_x0 和 condition
#                 x = pred_x0 * (1 - mask) + condition * mask
                
#             # 【防护 2】防止融合后出现微小溢出
#             x = torch.clamp(x, -SAFE_LIMIT, SAFE_LIMIT)
                
#     return x

# def repaint_sample(model, condition, mask, porosity, device, n_resample=5):
#     """
#     策略 C: RePaint (Fixed)
#     修正了方差爆炸问题：
#     1. Step 1 改为确定性采样 (不加噪)，防止与 Step 3 的加噪叠加导致能量溢出。
#     2. 修正了回溯步骤的 beta 索引。
#     """
#     print(f"  🚀 Strategy: Robust RePaint Sampling (Resample={n_resample})...")
#     timesteps = CONFIG['timesteps']
#     betas = get_beta_schedule(timesteps).to(device)
#     alphas = 1. - betas
#     alphas_cumprod = torch.cumprod(alphas, dim=0)

#     # 1. 初始化噪声
#     x = torch.randn((1, CONFIG['latent_channels'], CONFIG['image_size'], CONFIG['image_size'], CONFIG['image_size'])).to(device)
    
#     # 进度条
#     pbar = tqdm(reversed(range(timesteps)), total=timesteps, desc="RePaint Sampling")
    
#     for t in pbar:
#         # 动态调整重采样次数
#         # 只有在生成中间纹理的关键阶段才多次采样，节省时间
#         current_resample = n_resample if (50 < t < 800) else 1
        
#         for r in range(current_resample):
#             # -----------------------------------------------------------
#             # Step 1: 逆向 (Reverse) x_t -> x_{t-1} 
#             # 【关键修改】使用确定性采样 (Deterministic)，不加随机噪声！
#             # -----------------------------------------------------------
#             t_tensor = torch.full((1,), t, device=device, dtype=torch.long)
#             model_input = torch.cat([x, condition, mask], dim=1)
#             noise_pred = model(model_input, t_tensor, porosity)
            
#             # 获取参数
#             beta_t = betas[t]
#             alpha_t = alphas[t]
#             alpha_bar_t = alphas_cumprod[t]
            
#             # 计算 pred_x0 (用于指导方向)
#             pred_x0 = (x - torch.sqrt(1 - alpha_bar_t) * noise_pred) / torch.sqrt(alpha_bar_t)
#             pred_x0 = torch.clamp(pred_x0, -SAFE_LIMIT, SAFE_LIMIT) # 截断防护
            
#             # 计算 x_{t-1} 的均值 (Mean only, NO random noise here!)
#             # 这是一个指向 x_{t-1} 的确定性向量
#             # 公式推导自 DDIM (eta=0) 或 Posterior Mean
#             coef1 = torch.sqrt(1.0 - alpha_bar_t - beta_t) # 这里简化处理，近似 DDIM
#             # 或者使用标准的 DDPM 均值公式:
#             # mean = (1 / sqrt(alpha_t)) * (x - (beta_t / sqrt(1-alpha_bar_t)) * noise_pred)
#             # 我们使用 DDPM 均值公式更准确：
#             x_prev_unknown = (1 / torch.sqrt(alpha_t)) * (x - (beta_t / torch.sqrt(1 - alpha_bar_t)) * noise_pred)
            
#             # 【核心区别】这里不要加 "+ sigma * noise"，保持确定性！
            
#             # 截断
#             x_prev_unknown = torch.clamp(x_prev_unknown, -SAFE_LIMIT, SAFE_LIMIT)
            
#             # -----------------------------------------------------------
#             # Step 2: 注入已知 (Inject Condition)
#             # -----------------------------------------------------------
#             if t > 0:
#                 alpha_bar_prev = alphas_cumprod[t-1]
#                 # 每次重新采样 GT 的噪声，保证多样性
#                 noise_gt = torch.randn_like(condition)
#                 x_prev_known = torch.sqrt(alpha_bar_prev) * condition + torch.sqrt(1 - alpha_bar_prev) * noise_gt
                
#                 # 融合
#                 x_prev = x_prev_unknown * (1 - mask) + x_prev_known * mask
#             else:
#                 x_prev = x_prev_unknown * (1 - mask) + condition * mask

#             # -----------------------------------------------------------
#             # Step 3: 时间回溯 (Forward / Resample) x_{t-1} -> x_t
#             # 只有当不是最后一次重采样，且 t > 0 时才回溯
#             # -----------------------------------------------------------
#             if r < current_resample - 1 and t > 0:
#                 # 【关键修改】使用当前的 beta_t，而不是 beta_{t-1}
#                 # 因为我们要模拟从 t-1 到 t 的过程，方差由 beta_t 控制
#                 beta_forward = betas[t] 
                
#                 noise_add = torch.randn_like(x_prev)
                
#                 # Forward Process 公式: x_t = sqrt(1-beta)*x_{t-1} + sqrt(beta)*noise
#                 # 注意：这里我们用近似公式，或者直接用 RePaint 论文公式
#                 # RePaint 论文: x_t ~ N(sqrt(1-beta_t)*x_{t-1}, beta_t*I)
#                 x = torch.sqrt(1 - beta_forward) * x_prev + torch.sqrt(beta_forward) * noise_add
                
#                 # 截断，防止噪声叠加溢出
#                 x = torch.clamp(x, -SAFE_LIMIT, SAFE_LIMIT)
#             else:
#                 # 循环结束或最后一步，保留 x_{t-1} 进入下一个 t
#                 x = x_prev
                
#     return x

# # ==========================================
# # 3. 主程序 (Main Pipeline)
# # ==========================================

# def main():
#     # --- 配置与环境 ---
#     device = CONFIG['device']
#     root = get_project_root()
    
#     # 1. 确定模型路径
#     models_dir = os.path.join(root, "exp_results", CONFIG['experiment_name'], "models")
#     model_files = sorted(glob.glob(os.path.join(models_dir, "unet_epoch_*.pth")), key=os.path.getmtime)
#     if not model_files: print("❌ No models found"); return
#     # print(model_files)
#     model_path = model_files[-1] 
    
#     # 2. 加载模型
#     diffusion_model, vqvae_model = load_models(model_path, device)
    
#     # 3. 准备输出目录
#     save_dir = os.path.join(root, "exp_results", CONFIG['experiment_name'], "inference_outputs", "final_test")
#     os.makedirs(save_dir, exist_ok=True)
    
#     # 4. 获取测试数据
#     data_files = glob.glob(os.path.join(CONFIG['processed_data_dir'][1], "*.npy"))
#     if not data_files: print("❌ No test data found"); return
#     sample_file = random.choice(data_files) # 随机取一个
#     fname = os.path.basename(sample_file)
#     print(f"\n📄 Processing: {fname}")

#     # ================= 数据流转 (Data Flow) =================

#     # Step A: 加载原始数据
#     gt_raw = np.load(sample_file)
#     gt_tensor = torch.from_numpy(gt_raw).float().to(device)

#     # Step B: 缩放 (Entering Diffusion Space) -> ⚠️ 唯一一次放大
#     scale = CONFIG['scale_factor']
#     gt_scaled = gt_tensor * scale
#     safe_thresh = CONFIG.get('safe_threshold', 4.0)
#     gt_scaled = torch.clamp(gt_scaled, min=-safe_thresh, max=safe_thresh)
#     # gt_scaled = gt_tensor / 2.0
    
#     # Step C: 创建 Mask (自定义逻辑)
#     # 这里演示：Core Extension (保留 Top 50% / 32层)
#     D = 64
#     mask = torch.zeros_like(gt_scaled) # (1, C, D, H, W)
#     split_point = int(D * 0.5) # 自定义 Mask 大小：50%
#     mask[..., :split_point, :, :] = 1.0 
    
#     # 只有 Mask 是单通道的，需要切一下或者保持一致
#     # LatentUNet 的 mask 输入通常是 (B, 1, D, H, W)
#     mask_input = mask[:, 0:1, ...] 

#     # Condition = Scaled Latent * Mask
#     condition = gt_scaled * mask_input

#     # Step D: 提取孔隙度
#     match = re.search(r'porosity_(\d+\.\d+)', fname)
#     porosity_val = float(match.group(1)) if match else 0.15
#     porosity = torch.tensor([porosity_val]).to(device).view(1,1)

#     # Step E: 采样 (Sampling) -> 保持 Scaled 状态
#     # 切换这里来测试不同策略: 'ddim', 'repaint'
#     MODE = 'ddim'
#     # MODE = 'repaint' 
    
#     with torch.no_grad():
#         if MODE == 'ddim':
#             gen_scaled = ddim_sample(diffusion_model, condition, mask_input, porosity, device, ddim_steps=200)
#         elif MODE == 'repaint':
#             gen_scaled = repaint_sample(diffusion_model, condition, mask_input, porosity, device, n_resample=5)

        
#         # Step F: 还原 (Leaving Diffusion Space) -> ⚠️ 唯一一次缩小
#         gen_restored = gen_scaled / scale
#         # gen_restored = gen_scaled * 2
#         gt_restored = gt_tensor # 原始的 tensor 本来就是没缩放的，或者 gt_scaled / scale
        

#         # === DEBUG 专用 ===
#         print("\n🐛 Debugging Values before Decode:")
#         print(f"  GT (Restored)  -> Min: {gt_restored.min():.4f}, Max: {gt_restored.max():.4f}, Mean: {gt_restored.mean():.4f}, Std: {gt_restored.std():.4f}")
#         print(f"  Gen (Restored) -> Min: {gen_restored.min():.4f}, Max: {gen_restored.max():.4f}, Mean: {gen_restored.mean():.4f}, Std: {gen_restored.std():.4f}")
        
#         # 强制检查：如果 Gen 的范围远远超过 GT (比如 GT是-1到1，Gen是-10到10)
#         # 那么一定是 Scale 没除对，或者模型发散了
#         if gen_restored.std() > gt_restored.std() * 3:
#             print("⚠️ WARNING: Generated values are way too high! Trying to force rescale...")
#             # 死马当活马医：强制把 Gen 的分布拉回 GT 的分布
#             gen_restored = (gen_restored - gen_restored.mean()) / gen_restored.std() * gt_restored.std() + gt_restored.mean()
#             print(f"  Fixed Gen      -> Min: {gen_restored.min():.4f}, Max: {gen_restored.max():.4f}")
#         # ==================
        
#         # Step G: 解码 (Decoding to Pixel)
#         print("  🎨 Decoding to Pixel Space...")
#         recon_gen = vqvae_model.decode(gen_restored) # (1, 1, 256, 256, 256)
#         recon_gt = vqvae_model.decode(gt_restored)
        
        

#     # ================= 结果保存与可视化 =================
#     # 转 Numpy
#     vol_gen = recon_gen[0, 0].cpu().numpy()
#     vol_gt = recon_gt[0, 0].cpu().numpy()
    
#     # 生成 Pixel 级的 Condition (用于可视化)
#     # Mask 插值: Latent(64) -> Pixel(256)
#     mask_pixel = torch.nn.functional.interpolate(mask_input, scale_factor=4, mode='nearest')
#     mask_pixel_np = mask_pixel[0, 0].cpu().numpy()
    
#     vol_cond = vol_gt.copy()
#     vol_cond[mask_pixel_np == 0] = 0 # 抹去未知区域

#     # 1. 保存 NifTI
#     nib.save(nib.Nifti1Image(vol_gen, np.eye(4)), os.path.join(save_dir, f"{fname}_{MODE}_gen.nii.gz"))
    
#     # 2. 统一可视化
#     viz_path = os.path.join(save_dir, f"{fname}_{MODE}_LDM.png")
#     visualize_inference_results(vol_gt, vol_cond, vol_gen, mask_pixel_np, viz_path, fname)

# if __name__ == "__main__":
#     main()
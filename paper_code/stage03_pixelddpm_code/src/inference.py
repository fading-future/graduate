import torch
import numpy as np
import matplotlib.pyplot as plt
from model_unet import ConditionalUNet3D # 确保 model_unet.py 可用
import os
import glob
from pathlib import Path
import random
import time
import nibabel as nib
from utils.get_root_path import get_root_path
from config import CONFIG

# ================= 辅助函数 =================
def get_beta_schedule(timesteps):
    return torch.linspace(1e-4, 0.02, timesteps)

def simple_sample(model, condition, mask, device):
    """
    DDPM 采样核心逻辑 (Inpainting 模式)
    """
    model.eval()
    betas = get_beta_schedule(CONFIG['timesteps']).to(device)
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    x = torch.randn((1, 1, CONFIG['image_size'], CONFIG['image_size'], CONFIG['image_size'])).to(device)
    
    print("Start Sampling...")
    with torch.no_grad():
        for t in reversed(range(CONFIG['timesteps'])):
            # 简化日志输出，避免循环中打印过多信息
            if t % 20 == 0: print(f"Step {t}...") 
            
            model_input = torch.cat([x, condition, mask], dim=1)
            t_tensor = torch.full((1,), t, device=device, dtype=torch.long)
            predicted_noise = model(model_input, t_tensor)
            
            beta_t = betas[t]
            alpha_t = alphas[t]
            alpha_bar_t = alphas_cumprod[t]
            sigma_t = torch.sqrt(beta_t) if t > 0 else 0
            noise_z = torch.randn_like(x) if t > 0 else 0
            x = (1 / torch.sqrt(alpha_t)) * (x - (beta_t / torch.sqrt(1 - alpha_bar_t)) * predicted_noise) + sigma_t * noise_z
            
            if t > 0:
                noise_real = torch.randn_like(condition)
                alpha_bar_prev = alphas_cumprod[t-1]
                noisy_condition = torch.sqrt(alpha_bar_prev) * condition + torch.sqrt(1 - alpha_bar_prev) * noise_real
                x = x * (1 - mask) + noisy_condition * mask
            else:
                x = x * (1 - mask) + condition * mask
    return x

def repaint_sample(model, condition, mask, device):
    """
    RePaint 采样策略 (v2.0): 
    强力解决 '生成全黑' 和 '边界断层' 问题。
    原理：在生成过程中，反复将已知区域的真实纹理 '注入' 到画面中，
    并进行 '时间回溯' (Resampling)，强迫模型去匹配边界。
    """
    model.eval()
    timesteps = CONFIG['timesteps']
    betas = get_beta_schedule(timesteps).to(device)
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)

    # 1. 纯噪声开始
    x = torch.randn((1, 1, CONFIG['image_size'], CONFIG['image_size'], CONFIG['image_size'])).to(device)
    
    # ★ 关键参数：重采样次数
    # 次数越多，融合越好，但速度越慢。建议 5-10 次。
    n_resample = CONFIG['n_resample']
    
    print(f"Start RePaint Sampling (Resample={n_resample})...")
    with torch.no_grad():
        # 倒序时间步 T -> 0
        for t in reversed(range(timesteps)):
            if t % 20 == 0: print(f"Step {t}...")
            
            # 只在中间关键阶段 (t=200~800) 进行多轮打磨
            # 早期全是噪声没必要磨，晚期基本定型了也没必要磨
            current_resample = n_resample if (100 < t < 900) else 1
            
            for r in range(current_resample):
                # --- A. 正常预测逆向一步 x_{t-1} ---
                # 拼接输入：把当前的 x 和条件拼起来喂给模型
                model_input = torch.cat([x, condition, mask], dim=1)
                t_tensor = torch.full((1,), t, device=device, dtype=torch.long)
                
                # 预测噪声
                noise_pred = model(model_input, t_tensor)
                
                # DDPM 公式计算 x_{t-1}
                beta_t = betas[t]
                alpha_t = alphas[t]
                alpha_bar_t = alphas_cumprod[t]
                sigma_t = torch.sqrt(beta_t) if t > 0 else 0
                noise_z = torch.randn_like(x) if t > 0 else 0
                
                # x_{t-1} = 1/sqrt(alpha) * (x_t - coeff * eps) + sigma * z
                x_prev = (1 / torch.sqrt(alpha_t)) * (x - (beta_t / torch.sqrt(1 - alpha_bar_t)) * noise_pred) + sigma_t * noise_z
                
                # --- B. 注入已知信息 (Masked Area Replacement) ---
                # 这是 RePaint 的灵魂：强制把已知区域替换回 Ground Truth 的加噪版本
                if t > 0:
                    noise_real = torch.randn_like(condition)
                    alpha_bar_prev = alphas_cumprod[t-1]
                    # 计算 GT 在 t-1 时刻的样子: q(x_{t-1} | x_0)
                    noisy_gt = torch.sqrt(alpha_bar_prev) * condition + torch.sqrt(1 - alpha_bar_prev) * noise_real
                    
                    # 融合：已知区域用 noisy_gt，未知区域用模型预测的 x_prev
                    x = x_prev * (1 - mask) + noisy_gt * mask
                else:
                    # 最后一步直接用无噪的 GT
                    x = x_prev * (1 - mask) + condition * mask
                
                # --- C. 时间回溯 (Time Travel) ---
                # 如果还没磨完 (r < current_resample - 1)，就重新加噪回到 t 时刻，再生成一次
                # 这样模型就有机会根据“注入的已知信息”修正它的预测
                if r < current_resample - 1 and t > 0:
                    beta_next = betas[t-1] # 这里取 t-1 的 beta 近似
                    noise_add = torch.randn_like(x)
                    # 前向加噪一步：x_t = sqrt(1-beta)*x_{t-1} + sqrt(beta)*noise
                    x = torch.sqrt(1 - beta_next) * x + torch.sqrt(beta_next) * noise_add

    return x

def ddim_sample(model, condition, mask, device, ddim_steps=500, eta=0.0):
    """
    DDIM 加速采样
    ddim_steps: 采样步数，推荐 50 或 100 (比原来的 1000 快 10-20 倍)
    eta: 0 代表确定性采样 (DDIM), 1 代表 DDPM
    """
    model.eval()
    
    # 生成时间步序列：[0, 20, 40, ..., 980]
    total_timesteps = CONFIG['timesteps'] # 1000
    times = torch.linspace(0, total_timesteps - 1, steps=ddim_steps).long().to(device)
    times = list(reversed(times.int().tolist())) # 倒序: [980, ..., 0]
    
    # 获取预计算参数
    # 注意：这里需要你 diffusion_trainer 里的 alphas_cumprod
    # 简单起见，这里重新计算一遍
    betas = torch.linspace(1e-4, 0.02, total_timesteps).to(device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)

    # 初始噪声
    x = torch.randn((1, 1, CONFIG['image_size'], CONFIG['image_size'], CONFIG['image_size'])).to(device)
    
    print(f"Start DDIM Sampling ({ddim_steps} steps)...")
    
    with torch.no_grad():
        for i, t in enumerate(times):
            # 简化日志输出，避免循环中打印过多信息
            if t % 20 == 0: print(f"Step {t}...")

            # 获取当前 t 和下一个 t_prev
            t_prev = times[i + 1] if i < len(times) - 1 else -1
            
            # 1. 构造输入
            model_input = torch.cat([x, condition, mask], dim=1)
            t_tensor = torch.full((1,), t, device=device, dtype=torch.long)
            
            # 2. 预测噪声 epsilon_theta
            noise_pred = model(model_input, t_tensor)
            
            # 3. 计算 alpha 参数
            alpha_bar_t = alphas_cumprod[t]
            alpha_bar_t_prev = alphas_cumprod[t_prev] if t_prev >= 0 else torch.tensor(1.0).to(device)
            
            # 4. 预测 x0 (Denoised Image)
            # x0 = (xt - sqrt(1-at) * eps) / sqrt(at)
            pred_x0 = (x - torch.sqrt(1 - alpha_bar_t) * noise_pred) / torch.sqrt(alpha_bar_t)
            
            # === 关键：Inpainting 强制修正 ===
            # 在预测出 x0 后，强行把已知区域替换为 GT
            # 注意：DDIM 的 Inpainting 逻辑通常是在 x0 层面做替换，或者在 xt 层面做
            # 这里我们在 xt 层面做，更稳定
            if t > 0:
                 noise_real = torch.randn_like(condition)
                 noisy_condition = torch.sqrt(alpha_bar_t) * condition + torch.sqrt(1 - alpha_bar_t) * noise_real
                 # 此时的 x 是这一步优化前的，我们需要计算这一步优化后的 x_prev 再混合吗？
                 # 通常做法：计算出 x_prev 后再混合。
            
            # 5. DDIM 更新公式计算 x_prev
            sigma_t = eta * torch.sqrt((1 - alpha_bar_t_prev) / (1 - alpha_bar_t) * (1 - alpha_bar_t / alpha_bar_t_prev))
            
            # 指向 x_t 的方向
            dir_xt = torch.sqrt(1 - alpha_bar_t_prev - sigma_t**2) * noise_pred
            
            x_prev = torch.sqrt(alpha_bar_t_prev) * pred_x0 + dir_xt + sigma_t * torch.randn_like(x)
            
            # === Mask 混合 (放在得到 x_prev 后) ===
            if t_prev >= 0:
                # 计算 t_prev 时刻的真实条件噪声图
                noise_real = torch.randn_like(condition)
                noisy_cond_prev = torch.sqrt(alpha_bar_t_prev) * condition + torch.sqrt(1 - alpha_bar_t_prev) * noise_real
                
                # 混合
                x = x_prev * (1 - mask) + noisy_cond_prev * mask
            else:
                x = x_prev * (1 - mask) + condition * mask

    return x

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

# 实验文件夹与配置保存（绝对路径）
def setup_experiment(PROJECT_ROOT: Path, config: dict) -> Path:
    # 1. 结果根目录：项目根/exp_results（绝对路径）
    results_root = PROJECT_ROOT / "exp_results"  # pathlib的/运算符拼接路径，跨平台兼容
    results_root.mkdir(exist_ok=True)  # 不存在则创建，存在则跳过
    
    # 2. 当前实验文件夹：项目根/exp_results/实验名
    exp_dir = results_root / config["experiment_name"]
    exp_dir.mkdir(exist_ok=True)
    
    # 3. 模型保存目录：实验文件夹/models
    # model_dir = PROJECT_ROOT / config["model_output_dir"]
    # model_dir.mkdir(exist_ok=True)
    model_dir = Path(r"/chendou_space/chendou/paper_code/stage03_pixelddpm_code/exp_results/exp_01/models")

    # 4. 推理结果的保存目录：实验文件夹/img_output_dir
    # img_output_dir = exp_dir / config["inference_output_dir"]
    # img_output_dir.mkdir(exist_ok=True)
    img_output_dir = Path(r"/chendou_space/chendou/paper_code/stage03_pixelddpm_code/exp_results/exp_01/logs") / config["inference_output_dir"]
    
    return results_root, exp_dir, model_dir, img_output_dir

# 获取本次推理涉及到的绝对路径
_, _, model_dir, img_output_dir = setup_experiment(get_root_path(), CONFIG)

def normalize_volume(volume):
    """
    将 16-bit 数据归一化到 [-1, 1] 区间，适合 DDPM/GAN
    param volume: 输入的 3D 体数据，numpy 数组格式
    return: 归一化后的 3D 体数据
    """

    volume = volume.astype(np.float32)
    volume = volume / 65535.0 # 归一化到 [0, 1]
    volume = (volume * 2.0) - 1.0 # 归一化到 [-1, 1]
    return volume

# ================= 主流程 =================
def main(model_path, sampling_method, condition_size, inference_save_dir, npy_save_dir):
    device = CONFIG['device']

    # 1. 准备输出目录
    os.makedirs(img_output_dir, exist_ok=True)
    
    # 2. 加载模型
    print(f"Loading model from {model_dir}...")
    model = ConditionalUNet3D(in_channels=3, out_channels=1, base_channels=CONFIG['base_channels']).to(device)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval() # 确保模型处于评估模式
    except FileNotFoundError:
        print("Model file not found! Please run training first or check the path.")
        return

    # 3. 获取并随机选择测试数据
    npy_files = glob.glob(os.path.join(CONFIG['processed_data_dir'], "*.npy"))
    if not npy_files:
        print("No test data found.")
        return
    
    # 随机选择 20 个文件索引
    if len(npy_files) >= CONFIG['num_samples_to_generate']:
        selected_files = random.sample(npy_files, CONFIG['num_samples_to_generate'])
    else:
        selected_files = npy_files
        print(f"Only found {len(npy_files)} files, generating all of them.")

    # 4. 循环生成和可视化
    split_point = condition_size # 定义Z轴的分割点
    
    for i, file_path in enumerate(selected_files):
        print(f"\n--- Processing sample {i+1}/{len(selected_files)}: {os.path.basename(file_path)} ---")
        
        # 加载数据并准备 Tensor
        gt_np = np.load(file_path) # [-1, 1]

        gt_np = normalize_volume(gt_np)

        # ================== 新增代码：裁剪逻辑 ==================
        # 获取当前模型需要的尺寸 (128)
        target_size = CONFIG['image_size'] 
        
        # 如果原始数据比目标大，就进行“中心裁剪”
        if gt_np.shape[0] > target_size:
            print(f"Cropping from {gt_np.shape} to {target_size}^3 (Center Crop)")
            
            # 计算起始位置，保证切在正中间
            start_z = (gt_np.shape[0] - target_size) // 2
            start_y = (gt_np.shape[1] - target_size) // 2
            start_x = (gt_np.shape[2] - target_size) // 2
            
            end_z = start_z + target_size
            end_y = start_y + target_size
            end_x = start_x + target_size
            
            # 执行切片
            gt_np = gt_np[start_z:end_z, start_y:end_y, start_x:end_x]
        # =======================================================


        gt_tensor = torch.from_numpy(gt_np).unsqueeze(0).unsqueeze(0).float().to(device) # (1, 1, 64, 64, 64)
        
        # 构造 Mask (Z轴前32已知)
        mask = torch.zeros_like(gt_tensor)
        mask[..., :split_point, :, :] = 1.0 
        condition = gt_tensor * mask
        
        # 运行生成
        if sampling_method == 'simple_sample':
            generated = simple_sample(model, condition, mask, device)
        elif sampling_method == 'repaint_sample':
            generated = repaint_sample(model, condition, mask, device)
        elif sampling_method == 'ddim_sample':
            generated = ddim_sample(model, condition, mask, device, ddim_steps=200, eta=0.0)
        
        # 准备 NumPy 数据用于可视化
        gt_volume = gt_tensor[0, 0].cpu().numpy()
        input_cond_volume = gt_volume.copy()
        input_cond_volume[mask[0, 0].cpu().numpy() == 0] = 0 # 将未知区域设为 0
        gen_volume = generated[0, 0].cpu().numpy()

        # ... 在 plt.show() 之前 ...
        # 保存 3D 体积数据供 analyze_quality.py 读取
        # save_dir = Path(r"C:\Users\Administrator\Desktop\graduation_thesis_code\exp_results\exp_04\logs\analysis_samples") / npy_save_dir
        # time_stamp = time.time()
        # os.makedirs(save_dir, exist_ok=True)
        # np.save(os.path.join(save_dir, f"sample_{time_stamp}_gt.npy"), gt_volume)
        # np.save(os.path.join(save_dir, f"sample_{time_stamp}_gen.npy"), gen_volume)
        # print(f"Volume data saved to {save_dir}/")

        # 可视化并保存
        visualization_dir = img_output_dir / inference_save_dir
        os.makedirs(visualization_dir, exist_ok=True)
        save_path = os.path.join(visualization_dir, f"sample_{i+1}_visualization.png")
        visualize_three_planes(gt_volume, input_cond_volume, gen_volume, split_point, save_path, os.path.basename(file_path))

        # 保存 GT
        # gt_nii = nib.Nifti1Image(gt_volume, np.eye(4))
        # nib.save(gt_nii, os.path.join(visualization_dir, f"sample_{i+1}_gt.nii.gz"))

        # 保存 Generated
        # gen_nii = nib.Nifti1Image(gen_volume, np.eye(4))
        # nib.save(gen_nii, os.path.join(visualization_dir, f"sample_{i+1}_gen.nii.gz"))
    print("\nAll generations complete.")

if __name__ == "__main__":
    
    # 使用模型参数a，选取不同的采样策略b，设置不同的已经条件大小c，推理采样d 个数据，保存到指定目录
    # a: model_dir/unet_epoch_{epoch_id}.pth
    for epoch_id in reversed(range(35, 82, 1)):
        model_path = os.path.join(model_dir, f"unet_epoch_{epoch_id}.pth")

        # b: simple_sample / repaint_sample / ddim_sample
        # for sampling_method in ['ddim_sample', 'simple_sample', 'repaint_sample']:
        for sampling_method in ['ddim_sample', 'simple_sample']:
            # c: mask定义部分
            for condition_size in [64]:
                # d: CONFIG['num_samples_to_generate']
                CONFIG['num_samples_to_generate'] = 1
                print(f"\n=== Inference with Model Epoch {epoch_id}, Method: {sampling_method}, Condition Size: {condition_size} ===")
                inference_save_dir = f"inference_outputs_epoch{epoch_id}_{sampling_method}_cond{condition_size}"
                npy_save_dir = f"npy_outputs_epoch{epoch_id}_{sampling_method}_cond{condition_size}"
                # 运行主推理流程
                main(model_path=model_path, 
                     sampling_method=sampling_method, 
                     condition_size=condition_size, 
                     inference_save_dir=inference_save_dir,
                     npy_save_dir=npy_save_dir)
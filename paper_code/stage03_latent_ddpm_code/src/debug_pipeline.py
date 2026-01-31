import torch
import numpy as np
import os
import matplotlib.pyplot as plt
import yaml

# 引用你的模块
from src.config import CONFIG
from src.dataset_latent import LatentDataset
from src.model_latent import ConditionalLatentUNet
from src.models.vae import KLVAE3D 

# ================= 配置 =================
VAE_CONFIG_PATH = "/chendou_space/chendou/paper_code/stage02_KLvae_single_code_v2/config/train_config copy.yaml"
VAE_CHECKPOINT = "/chendou_space/chendou/paper_code/stage02_KLvae_single_code_v2/experiments/exp01_cube_structure_v1/checkpoint_epoch_26.pt"
UNET_CHECKPOINT = "/chendou_space/chendou/paper_code/stage03_latent_ddpm_code/exp_results/exp_final_stage2_graduation/models/unet_epoch_15.pth"

DEVICE = "cuda"
SAVE_PATH = "debug_process_viz_v2.png"
CROP_SIZE = 16 

def load_models():
    print(">>> Loading Models...")
    unet = ConditionalLatentUNet(
        in_channels=CONFIG['in_channels'],
        out_channels=CONFIG['out_channels'],
        base_channels=CONFIG['base_channels'],
        channel_mults=CONFIG['channel_mults'],
        use_attention=(False, True, True)
    ).to(DEVICE)
    ckpt = torch.load(UNET_CHECKPOINT, map_location=DEVICE)
    unet.load_state_dict(ckpt.get('ema_state_dict', ckpt.get('model_state_dict')))
    unet.eval()

    with open(VAE_CONFIG_PATH, 'r') as f:
        vae_cfg = yaml.safe_load(f)
    vae = KLVAE3D(vae_cfg).to(DEVICE)
    vae_ckpt = torch.load(VAE_CHECKPOINT, map_location=DEVICE)
    state_dict = {k.replace('_orig_mod.', ''): v for k, v in vae_ckpt['vae_state_dict'].items()}
    vae.load_state_dict(state_dict)
    vae.eval()
    return unet, vae

def visualize_pipeline():
    unet, vae = load_models()
    
    # 1. 获取数据
    dataset = LatentDataset(data_dir=CONFIG['processed_data_dir'], augment=False)
    batch = dataset[0]
    
    gt_latent_full = batch['GT'].unsqueeze(0).to(DEVICE)
    porosity = batch['Porosity'].unsqueeze(0).to(DEVICE)
    
    # === Crop ===
    d, h, w = gt_latent_full.shape[2:]
    ds, hs, ws = (d-CROP_SIZE)//2, (h-CROP_SIZE)//2, (w-CROP_SIZE)//2
    gt_latent = gt_latent_full[:, :, ds:ds+CROP_SIZE, hs:hs+CROP_SIZE, ws:ws+CROP_SIZE]
    
    # 2. 构造 Mask (沿 Z 轴切断)
    # Mask Shape: [1, 1, 16, 16, 16]
    mask = torch.ones((1, 1, CROP_SIZE, CROP_SIZE, CROP_SIZE), device=DEVICE)
    split_idx = CROP_SIZE // 2
    # 后半截 (Z > 8) 设为 0
    mask[:, :, split_idx:, :, :] = 0.0
    
    condition = gt_latent * mask
    
    noise = torch.randn_like(gt_latent)
    noisy_latent = 0.7 * gt_latent + 0.7 * noise

    # 3. 解码
    scale = CONFIG['scale_factor']
    
    with torch.no_grad():
        with torch.amp.autocast('cuda'):
            # 解码后的 Shape: [1, 1, 128, 128, 128]
            img_gt = vae.decode(gt_latent / scale)
            img_cond = vae.decode(condition / scale)
            
            # Mask 不需要 VAE，直接放大
            img_mask = torch.nn.functional.interpolate(mask, scale_factor=8, mode='nearest')

            # UNet 预测
            model_input = torch.cat([noisy_latent, condition, mask], dim=1)
            t = torch.tensor([500], device=DEVICE).long()
            noise_pred = unet(model_input, t, porosity)

    # 4. 绘图 (关键修改：换视角！)
    # 我们要看侧面 (YZ平面)，所以切 X 轴的中间
    # 这样横轴是 Y，纵轴是 Z，就能看到 Z 轴上下的变化了
    slice_idx = (CROP_SIZE * 8) // 2
    
    imgs = {
        "1. GT (Side View)": img_gt,
        "2. Condition (Look here!)": img_cond,
        "3. Mask (0=Black, 1=White)": img_mask,
        "4. Noisy Latent": noisy_latent,
        "5. Pred Noise": noise_pred
    }

    fig, axes = plt.subplots(1, 5, figsize=(25, 5))
    
    for i, (title, tensor) in enumerate(imgs.items()):
        ax = axes[i]
        arr = tensor.squeeze().cpu().float().numpy()
        
        # === 核心修改 ===
        # arr shape is [128, 128, 128] (Z, H, W) or Latent [16, 16, 16]
        # 之前是 arr[mid] -> 取 Z 切片 (XY平面) -> 看不到 Z 轴变化
        # 现在改成 arr[:, :, mid] -> 取 W 切片 (ZH平面) -> 纵轴是 Z，横轴是 H
        
        mid = arr.shape[2] // 2 # 取宽度(W)中间的切片，看侧面
        
        if "Channel 0" in title or "Latent" in title or "Noise" in title:
            # Latent: [4, 16, 16, 16] -> 取第0通道
            if arr.ndim == 4: arr = arr[0]
            mid = arr.shape[2] // 2
            # 这里的切片是 [Z, H]，所以能看到 Z 轴的断层
            im = ax.imshow(arr[:, :, mid], cmap='viridis') 
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        else:
            # Image: [128, 128, 128]
            # imshow 默认原点在左上角，第一维是Y(纵轴)，第二维是X(横轴)
            # 我们的 arr 是 [Z, H, W]。
            # arr[:, :, mid] -> 纵轴是 Z，横轴是 H。
            # 这样应该能看到上半截有图，下半截黑的。
            ax.imshow(arr[:, :, mid], cmap='gray', vmin=-1, vmax=1)
            
        ax.set_title(title)
        ax.set_xlabel("Height (Y)")
        ax.set_ylabel("Depth (Z)") # 标注一下轴

    plt.tight_layout()
    plt.savefig(SAVE_PATH)
    print(f"✅ Visualization saved to {SAVE_PATH} (Side View)")

if __name__ == "__main__":
    visualize_pipeline()

# import torch
# import numpy as np
# import os
# import matplotlib.pyplot as plt
# import yaml

# # 引用你的模块
# from src.config import CONFIG
# from src.dataset_latent import LatentDataset
# from src.model_latent import ConditionalLatentUNet
# from src.models.vae import KLVAE3D 

# # ================= 配置 =================
# VAE_CONFIG_PATH = "/chendou_space/chendou/paper_code/stage02_KLvae_single_code_v2/config/train_config copy.yaml"
# VAE_CHECKPOINT = "/chendou_space/chendou/paper_code/stage02_KLvae_single_code_v2/experiments/exp01_cube_structure_v1/checkpoint_epoch_26.pt"
# UNET_CHECKPOINT = "/chendou_space/chendou/paper_code/stage03_latent_ddpm_code/exp_results/exp_final_stage2_graduation/models/unet_epoch_10.pth"

# DEVICE = "cuda"
# SAVE_PATH = "debug_process_viz.png"
# CROP_SIZE = 16  # 裁剪 Latent 大小，16对应128像素，20对应160像素

# def load_models():
#     print(">>> Loading Models...")
#     unet = ConditionalLatentUNet(
#         in_channels=CONFIG['in_channels'],
#         out_channels=CONFIG['out_channels'],
#         base_channels=CONFIG['base_channels'],
#         channel_mults=CONFIG['channel_mults'],
#         use_attention=(False, True, True)
#     ).to(DEVICE)
#     ckpt = torch.load(UNET_CHECKPOINT, map_location=DEVICE)
#     unet.load_state_dict(ckpt.get('ema_state_dict', ckpt.get('model_state_dict')))
#     unet.eval()

#     with open(VAE_CONFIG_PATH, 'r') as f:
#         vae_cfg = yaml.safe_load(f)
#     vae = KLVAE3D(vae_cfg).to(DEVICE)
#     vae_ckpt = torch.load(VAE_CHECKPOINT, map_location=DEVICE)
#     state_dict = {k.replace('_orig_mod.', ''): v for k, v in vae_ckpt['vae_state_dict'].items()}
#     vae.load_state_dict(state_dict)
#     vae.eval()
#     return unet, vae

# def visualize_pipeline():
#     unet, vae = load_models()
    
#     # 1. 获取数据
#     dataset = LatentDataset(data_dir=CONFIG['processed_data_dir'], augment=False)
#     batch = dataset[0] # 取第一个样本
    
#     # [1, 4, 32, 32, 32]
#     gt_latent_full = batch['GT'].unsqueeze(0).to(DEVICE)
#     porosity = batch['Porosity'].unsqueeze(0).to(DEVICE)
    
#     # === 关键：裁剪 Latent 以节省显存 ===
#     d, h, w = gt_latent_full.shape[2:]
#     ds, hs, ws = (d-CROP_SIZE)//2, (h-CROP_SIZE)//2, (w-CROP_SIZE)//2
#     gt_latent = gt_latent_full[:, :, ds:ds+CROP_SIZE, hs:hs+CROP_SIZE, ws:ws+CROP_SIZE]
#     print(f"✅ Cropped Latent Shape: {gt_latent.shape}")

#     # 2. 构造各种中间变量
#     # Mask: 切掉后一半 (Z轴)
#     mask = torch.ones((1, 1, CROP_SIZE, CROP_SIZE, CROP_SIZE), device=DEVICE)
#     split_idx = CROP_SIZE // 2
#     mask[:, :, split_idx:, :, :] = 0.0
    
#     # Condition: 只有前半部分有值
#     condition = gt_latent * mask
    
#     # Noisy Latent: 模拟 UNet 看到的噪声图 (t=500)
#     noise = torch.randn_like(gt_latent)
#     # 简单模拟加噪: 0.7*原图 + 0.7*噪声 (修正了这里的变量名)
#     noisy_latent = 0.7 * gt_latent + 0.7 * noise

#     # 3. 解码所有变量用于可视化
#     # 注意：解码前必须除以 scale_factor
#     scale = CONFIG['scale_factor']
    
#     with torch.no_grad():
#         with torch.amp.autocast('cuda'):
#             # A. 解码 GT
#             img_gt = vae.decode(gt_latent / scale)
            
#             # B. 解码 Condition (带 Mask)
#             # 同样除以 scale，这能检查 Condition 在像素空间长什么样
#             img_cond = vae.decode(condition / scale)
            
#             # C. 解码 Mask (本身就是 0/1，不需要 VAE，直接插值放大方便看)
#             img_mask = torch.nn.functional.interpolate(mask, scale_factor=8, mode='nearest')

#             # D. UNet 预测一次 (修正：使用 noisy_latent)
#             model_input = torch.cat([noisy_latent, condition, mask], dim=1)
#             t = torch.tensor([500], device=DEVICE).long()
#             noise_pred = unet(model_input, t, porosity)

#     # 4. 绘图 (取中间切片)
#     slice_idx = (CROP_SIZE * 8) // 2
    
#     imgs = {
#         "1. Ground Truth (Latent->VAE)": img_gt,
#         "2. Condition (Masked->VAE)": img_cond,
#         "3. Mask (0=Missing, 1=Keep)": img_mask,
#         "4. Noisy Latent (Channel 0)": noisy_latent, # 直接看 Latent 通道
#         "5. Predicted Noise (Channel 0)": noise_pred # 直接看 Latent 通道
#     }

#     fig, axes = plt.subplots(1, 5, figsize=(25, 5))
    
#     for i, (title, tensor) in enumerate(imgs.items()):
#         ax = axes[i]
        
#         # 针对 Latent 和 Image 分别处理
#         arr = tensor.squeeze().cpu().float().numpy()
        
#         if "Channel 0" in title:
#             # 如果是 Latent，画第0个通道的切片
#             if arr.ndim == 4: arr = arr[0] 
#             mid = arr.shape[0] // 2
#             im = ax.imshow(arr[mid], cmap='viridis') # Latent 用热力图看
#             plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
#         else:
#             # 如果是 Image，画中间切片
#             if arr.ndim == 4: arr = arr[0] # VAE 输出可能是 [1, 256, 256, 256] -> [256...]
#             mid = arr.shape[0] // 2
#             ax.imshow(arr[mid], cmap='gray', vmin=-1, vmax=1)
            
#         ax.set_title(title)
#         ax.axis('off')

#     plt.tight_layout()
#     plt.savefig(SAVE_PATH)
#     print(f"✅ Visualization saved to {SAVE_PATH}")

# if __name__ == "__main__":
#     visualize_pipeline()

# import torch
# import numpy as np
# import os
# import matplotlib.pyplot as plt
# import yaml

# # 引用你的模块
# from src.config import CONFIG
# from src.dataset_latent import LatentDataset
# from src.model_latent import ConditionalLatentUNet
# from src.models.vae import KLVAE3D 

# # ================= 配置 =================
# # 必须确保这三个路径是对的
# VAE_CONFIG_PATH = "/chendou_space/chendou/paper_code/stage02_KLvae_single_code_v2/config/train_config copy.yaml"
# VAE_CHECKPOINT = "/chendou_space/chendou/paper_code/stage02_KLvae_single_code_v2/experiments/exp01_cube_structure_v1/checkpoint_epoch_26.pt"
# # 你的 Stage 2 模型
# UNET_CHECKPOINT = "/chendou_space/chendou/paper_code/stage03_latent_ddpm_code/exp_results/exp_final_stage2_graduation/models/unet_epoch_10.pth"

# DEVICE = "cuda"

# def print_stats(name, tensor):
#     """打印 Tensor 的统计信息，用于抓内鬼"""
#     tensor = tensor.float()
#     print(f"[{name}] Shape: {list(tensor.shape)}")
#     print(f"    Min: {tensor.min().item():.4f} | Max: {tensor.max().item():.4f}")
#     print(f"    Mean: {tensor.mean().item():.4f} | Std: {tensor.std().item():.4f}")
#     if torch.isnan(tensor).any():
#         print("    🚨 WARNING: Contains NaNs!")
#     if torch.isinf(tensor).any():
#         print("    🚨 WARNING: Contains Infs!")
#     print("-" * 40)

# def load_models():
#     print(">>> 1. Loading Models...")
#     # 加载 Stage 2 UNet
#     unet = ConditionalLatentUNet(
#         in_channels=CONFIG['in_channels'],
#         out_channels=CONFIG['out_channels'],
#         base_channels=CONFIG['base_channels'],
#         channel_mults=CONFIG['channel_mults'],
#         use_attention=(False, True, True)
#     ).to(DEVICE)
#     ckpt = torch.load(UNET_CHECKPOINT, map_location=DEVICE)
#     unet.load_state_dict(ckpt.get('ema_state_dict', ckpt.get('model_state_dict')))
#     unet.eval()

#     # 加载 Stage 1 VAE
#     with open(VAE_CONFIG_PATH, 'r') as f:
#         vae_cfg = yaml.safe_load(f)
#     vae = KLVAE3D(vae_cfg).to(DEVICE)
#     vae_ckpt = torch.load(VAE_CHECKPOINT, map_location=DEVICE)
#     state_dict = {k.replace('_orig_mod.', ''): v for k, v in vae_ckpt['vae_state_dict'].items()}
#     vae.load_state_dict(state_dict)
#     vae.eval()
    
#     return unet, vae

# def test_pipeline():
#     unet, vae = load_models()
    
#     # === 第一步：检查数据源 (Dataset) ===
#     print("\n>>> 2. Checking Dataset (Stage 2 Input)...")
#     dataset = LatentDataset(data_dir=CONFIG['processed_data_dir'], augment=False)
#     batch = dataset[0] # 取第一个样本
    
#     gt_latent = batch['GT'].unsqueeze(0).to(DEVICE)
#     porosity = batch['Porosity'].unsqueeze(0).to(DEVICE)
    
#     print_stats("Loaded Latent from NPY", gt_latent)
    
#     # 判断 Scale Factor 是否正确
#     if gt_latent.std() < 0.5:
#         print("⚠️ 警告：Latent 标准差过小 (<0.5)。")
#         print("    可能原因：Dataset 里没有乘 scale_factor，或者 scale_factor 设置太小。")
#     elif gt_latent.std() > 2.0:
#         print("⚠️ 警告：Latent 标准差过大 (>2.0)。")
#         print("    可能原因：scale_factor 设置过大，导致数据分布发散。")
#     else:
#         print("✅ 数据分布看起来正常 (接近 N(0,1))。")

#     # === 第二步：检查 VAE 解码能力 (Stage 1 Sanity Check) ===
#     print("\n>>> 3. Checking VAE Decoding (Stage 1)...")
#     # 尝试把加载进来的 Latent 直接解码回去，看看是不是 VAE 坏了
#     with torch.no_grad():
#         # 记得除以 scale factor 还原回 VAE 的原始空间
#         latent_raw = gt_latent / CONFIG['scale_factor']
#         print_stats("Latent Input to VAE (Raw)", latent_raw)
        
#         recon_img = vae.decode(latent_raw)
#         print_stats("VAE Reconstructed Image", recon_img)
        
#     if recon_img.min() < -1.5 or recon_img.max() > 1.5:
#          print("⚠️ 警告：VAE 输出范围异常，可能模型权重加载错误或 Latent 维度不对。")

#     # === 第三步：检查 UNet 预测 (Stage 2 Model Check) ===
#     print("\n>>> 4. Checking UNet Prediction (Stage 2)...")
    
#     # 造一个输入
#     noisy_latent = torch.randn_like(gt_latent) # [1, 4, 32, 32, 32]
    
#     # --- 🔴 修改这里 ---
#     # mask = torch.ones_like(gt_latent) # <--- 错误代码：这会生成 4 通道 mask
    
#     # ✅ 正确代码：强制生成 1 通道 mask
#     mask = torch.ones((1, 1, 32, 32, 32), device=DEVICE) 
#     # ------------------
    
#     mask[:, :, 16:, :, :] = 0 # 切一半
#     condition = gt_latent * mask # 这里的广播机制会自动处理 [1,4] * [1,1]
    
#     # 现在的拼接：4 + 4 + 1 = 9 Channels
#     model_input = torch.cat([noisy_latent, condition, mask], dim=1)
#     t = torch.tensor([500], device=DEVICE).long() # 取中间时间步
    
#     with torch.no_grad():
#         print_stats("UNet Input", model_input)
#         noise_pred = unet(model_input, t, porosity)
#         print_stats("UNet Output (Predicted Noise)", noise_pred)
        
#     if noise_pred.abs().max() > 10:
#         print("🚨 严重警告：UNet 输出的数值极其巨大！")
#         print("    原因A：Scale Factor 填错了 (太大)。")
#         print("    原因B：Input Channels 不对齐 (比如 9 通道层读到了 129 通道的权重)。")
#         print("    原因C：学习率太大导致模型炸了。")

#     # === 第四步：模拟一步 RePaint 融合 ===
#     print("\n>>> 5. Checking RePaint Logic (Integration)...")
#     # 模拟最后一步 t=0
#     # 假设预测的 x0 就是 gt_latent (理想情况)
#     x_pred = gt_latent.clone() 
    
#     # 强制融合逻辑
#     # x_final = x_pred * (1 - mask) + gt_latent * mask
#     # 理论上应该等于 gt_latent
#     x_final = x_pred * (1 - mask) + gt_latent * mask
#     print_stats("RePaint Output (Should match GT)", x_final)
    
#     # 解码这个融合后的结果
#     with torch.no_grad():
#         x_final_raw = x_final / CONFIG['scale_factor']
#         final_img = vae.decode(x_final_raw)
#         print_stats("Final Decoded Image", final_img)
        
#         # 保存一下这张图看看
#         plt.figure()
#         plt.imshow(final_img[0, 0, 16, :, :].cpu().numpy(), cmap='gray', vmin=-1, vmax=1)
#         plt.title("Debug Reconstructed Image")
#         plt.savefig("debug_output.png")
#         print("✅ Saved debug_output.png")

# if __name__ == "__main__":
#     test_pipeline()
import torch
import yaml
import numpy as np
import os
import argparse
from torchvision.utils import save_image
from models.vae import KLVAE3D

# === 配置部分 ===
EPOCH_TO_TEST = 10  # 你想测试的 epoch 数
# 你的 npy 数据路径 (请修改这里)
# SAMPLE_NPY_PATH = r"E:\aligned_Training_Data\6-6-20 全部_z640_y530_x449.npy" 
SAMPLE_NPY_PATH = r"D:\多尺度岩心数据集\Final_Result_Sorted\6-6-21_Global_Consistency_z320_y192_x448.npy"
# SAMPLE_NPY_PATH = r""
# 你的训练好的模型路径 (请修改这里)
CHECKPOINT_PATH = rf"E:\chendou\paper_code\stage02_KLvae_single_code_v2\experiments\exp04_cube_structure_v1\ckpt_epoch_{EPOCH_TO_TEST}.pt" # 举例
CONFIG_PATH = r"E:\chendou\paper_code\stage02_KLvae_single_code_v2\config\train_config.yaml"

def load_config(path):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def fix_state_dict(state_dict):
    new_state = {}
    for k, v in state_dict.items():
        if k.startswith("_orig_mod."):
            new_state[k[10:]] = v
        else:
            new_state[k] = v
    return new_state

def inference_single_volume():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. Load Config & Model
    print(f"Loading config from {CONFIG_PATH}...")
    cfg = load_config(CONFIG_PATH)
    
    model = KLVAE3D(cfg).to(device)
    
    print(f"Loading checkpoint from {CHECKPOINT_PATH}...")
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(fix_state_dict(ckpt['vae_state_dict']))
    model.eval()

    # 2. Load Data (256^3)
    if not os.path.exists(SAMPLE_NPY_PATH):
        # 为了方便演示，如果没有指定具体文件，我们自动去 data_root 找一个
        data_root = cfg['data']['data_root']
        import glob
        files = glob.glob(os.path.join(data_root, "*.npy"))
        if len(files) > 0:
            target_file = files[0]
            print(f"Auto-selected file: {target_file}")
        else:
            print("Error: No .npy files found.")
            return
    else:
        target_file = SAMPLE_NPY_PATH

    print("Loading NPY data...")
    
    data_np = np.load(target_file).astype(np.float32)

    # 建议：先打印“真正原始值域”（放在归一化前）
    print("RAW(before norm) min/max/unique:", data_np.min(), data_np.max(), np.unique(data_np)[:10])

    mn, mx = float(data_np.min()), float(data_np.max())
    if mn >= -1.01 and mx <= 1.01:
        if mn >= 0.0 and mx <= 1.01:
            data_np = data_np * 2.0 - 1.0
    else:
        if mx <= 1.5:
            data_np = data_np * 2.0 - 1.0
        elif mx <= 255.5:
            data_np = (data_np / 255.0) * 2.0 - 1.0
        else:
            data_np = (data_np / 65535.0) * 2.0 - 1.0

    print("AFTER(norm) min/max/unique:", data_np.min(), data_np.max(), np.unique(data_np)[:10])

    vol = data_np  # [256,256,256]
    s = 64
    vol = vol[128-s:128+s, 128-s:128+s, 128-s:128+s]  # 128^3
    input_tensor = torch.from_numpy(vol).unsqueeze(0).unsqueeze(0).to(device)

    # [D, H, W] -> [1, 1, D, H, W]
    # input_tensor = torch.from_numpy(data_np).unsqueeze(0).unsqueeze(0).to(device)
    print(f"Input shape: {input_tensor.shape}")

    # 3. Inference
    print("Running VAE inference (this might take memory)...")
    with torch.no_grad():
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            # 直接输入 256^3，VAE 是全卷积的，会自动适应
            # 输出 latent 大小应为 32x32x32 (如果是 8 倍下采样)
            reconstruction, posterior = model(input_tensor, sample_posterior=False) # 测试时通常用 mean (sample=False)
    
    print(f"Recon shape: {reconstruction.shape}")

    # 4. Save Slices
    real = input_tensor[0, 0].float().cpu()                 # [-1,1]
    recon_logits = reconstruction[0, 0].float().cpu()        # logits

    r = recon_logits.detach().cpu().numpy()
    print("recon logits min/max/mean:", r.min(), r.max(), r.mean())

    # real: [-1,1] -> [0,1]
    real = (real.clamp(-1, 1) + 1) / 2

    # recon: logits -> prob [0,1]
    recon = torch.sigmoid(recon_logits)

    # （可选）如果你想看二值效果：
    recon = (recon > 0.5).float()

    D, H, W = real.shape
    
    # 取三个视角的中间切片
    # XY Plane (Axial)
    slice_xy_real = real[D//2, :, :]
    slice_xy_recon = recon[D//2, :, :]
    
    # XZ Plane (Coronal)
    slice_xz_real = real[:, H//2, :]
    slice_xz_recon = recon[:, H//2, :]

    # YZ Plane (Sagittal)
    slice_yz_real = real[:, :, W//2]
    slice_yz_recon = recon[:, :, W//2]

    # 拼接大图: 
    # Row 1: Real XY, Real XZ, Real YZ
    # Row 2: Rec  XY, Rec  XZ, Rec  YZ
    
    row1 = torch.cat([slice_xy_real, slice_xz_real, slice_yz_real], dim=1)
    row2 = torch.cat([slice_xy_recon, slice_xz_recon, slice_yz_recon], dim=1)
    final_grid = torch.cat([row1, row2], dim=0)

    save_path = f"inference_result_256_{EPOCH_TO_TEST}pt_1.png"
    save_image(final_grid, save_path)
    print(f"Saved comparison to {save_path}")

if __name__ == "__main__":
    inference_single_volume()
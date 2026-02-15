import torch
import yaml
import numpy as np
import os
from torchvision.utils import save_image
from models.vae import KLVAE3D

# === 配置部分 ===
EPOCH_TO_TEST = 11 # 你想测试的 epoch 数

# 你的 npy 数据路径 (请修改这里)
# SAMPLE_NPY_PATH = r"E:\aligned_Training_Data\6-6-20 全部_z640_y530_x449.npy"
SAMPLE_NPY_PATH = r"D:\\多尺度岩心数据集\\window_slide_result\\6-6-24\\Final_Result_Sorted_w300_s50\\6-6-24_z550_y450_x636.npy"
# SAMPLE_NPY_PATH = r""

# 你的训练好的模型路径 (请修改这里)
CHECKPOINT_PATH = rf"E:\chendou\paper_code\stage02_KLvae_single_code_v2\experiments\exp04_cube_structure_v1\ckpt_epoch_{EPOCH_TO_TEST}.pt"  # 举例
CONFIG_PATH = r"E:\chendou\paper_code\stage02_KLvae_single_code_v2\config\train_config copy.yaml"

# 是否将重建结果二值化显示（True 会掩盖“接近 0/1”的细节；建议先 False 看概率图）
SHOW_BINARY = True


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


def _center_crop_3d(vol: np.ndarray, target: int) -> np.ndarray:
    """Center crop a 3D volume to [target, target, target]."""
    D, H, W = vol.shape
    if D < target or H < target or W < target:
        raise ValueError(f"Volume too small: {vol.shape}, cannot crop to {target}^3")
    d0 = (D - target) // 2
    h0 = (H - target) // 2
    w0 = (W - target) // 2
    return vol[d0:d0 + target, h0:h0 + target, w0:w0 + target]


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

    # 2. Load Data
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

    # === Normalize to [-1,1] (保持与你 dataset.py 的语义一致) ===
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

    # === 关键修改：输入体积自适应裁剪到训练输入大小（你训练是 64^3）===
    # 不改变其它逻辑：仅把原来写死的 256->中心裁 128 的逻辑改成“如果是 64 就直接用；大于 64 就中心裁到 64”
    target = int(cfg.get('data', {}).get('croped_size', cfg.get('data', {}).get('image_size', 64)))
    vol = data_np  # [D,H,W]

    if vol.shape != (target, target, target):
        # 只要体积 >= target，就中心裁到 target^3
        vol = _center_crop_3d(vol, target)

    input_tensor = torch.from_numpy(vol).unsqueeze(0).unsqueeze(0).to(device)
    print(f"Input shape: {input_tensor.shape}")

    # 3. Inference
    print("Running VAE inference...")
    with torch.no_grad():
        # 保持你原来的 autocast 设置；若你想完全对齐训练 dtype，可自行改 dtype
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            reconstruction, posterior = model(input_tensor, sample_posterior=False)  # 测试时通常用 mean

    print(f"Recon shape: {reconstruction.shape}")

    # 4. Save Slices
    real = input_tensor[0, 0].float().cpu()                 # [-1,1]
    recon_logits = reconstruction[0, 0].float().cpu()        # logits

    r = recon_logits.detach().cpu().numpy()
    print("recon logits min/max/mean:", r.min(), r.max(), r.mean())

    # real: [-1,1] -> [0,1]
    real01 = (real.clamp(-1, 1) + 1) / 2

    # recon: logits -> prob [0,1]
    recon_prob = torch.sigmoid(recon_logits)
    recon01 = (recon_prob > 0.5).float() if SHOW_BINARY else recon_prob

    D, H, W = real01.shape

    # 反转0,1 像素值
    # real01 = 1 - real01
    # recon01 = 1 - recon01

    # 取三个视角的中间切片
    # XY Plane (Axial)
    slice_xy_real = real01[D // 2, :, :]
    slice_xy_recon = recon01[D // 2, :, :]

    # XZ Plane (Coronal)
    slice_xz_real = real01[:, H // 2, :]
    slice_xz_recon = recon01[:, H // 2, :]

    # YZ Plane (Sagittal)
    slice_yz_real = real01[:, :, W // 2]
    slice_yz_recon = recon01[:, :, W // 2]

    # 拼接大图:
    # Row 1: Real XY, Real XZ, Real YZ
    # Row 2: Rec  XY, Rec  XZ, Rec  YZ
    row1 = torch.cat([slice_xy_real, slice_xz_real, slice_yz_real], dim=1)
    row2 = torch.cat([slice_xy_recon, slice_xz_recon, slice_yz_recon], dim=1)
    final_grid = torch.cat([row1, row2], dim=0)

    save_path = f"inference_result_{target}_{EPOCH_TO_TEST}pt.png"
    save_image(final_grid, save_path)
    print(f"Saved comparison to {save_path}")


if __name__ == "__main__":
    inference_single_volume()

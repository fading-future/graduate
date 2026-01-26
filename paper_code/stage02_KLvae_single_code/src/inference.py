#%%
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import random
from vae_model import VAE3D
from config import CONFIG

# ================= 配置部分 =================
# 1. 指定模型路径 (训练生成的 checkpoint)
# 暂时还没有保存模型的话，等有了 checkpoint_ep8.pth 再填这里
CHECKPOINT_PATH = "/chendou_space/chendou/paper_code/stage02_KLvae_single_code/src/experiments/checkpoint_ep10.pth" 
    
# 2. 指定一个测试数据的路径 (.npy)
DATA_PATH = "/chendou_space/data/aligned_Training_Data/6-6-21_z2240_y587_x567.npy"
# 3. 输出图片保存文件夹
OUTPUT_DIR = "./results_visual"
os.makedirs(OUTPUT_DIR, exist_ok=True)
# ===========================================

#%%
def denormalize(tensor, g_min, g_max):
    """将 [-1, 1] 还原回 [0, 1] 用于可视化"""
    # 实际数据分布还原: x = (x_norm + 1) / 2 * (max - min) + min
    # 但为了可视化，我们只需要映射到 0-1 即可
    return (tensor.clamp(-1, 1) + 1) / 2.0

def plot_slices(orig, recon, filename):
    """
    画出 3D 数据的正交切片对比图
    orig, recon: (D, H, W) numpy array
    """
    # 选取中间切片
    D, H, W = orig.shape
    slice_d = D // 2
    slice_h = H // 2
    slice_w = W // 2

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # --- 第一行：原始数据 (Ground Truth) ---
    axes[0, 0].imshow(orig[slice_d, :, :], cmap='gray', vmin=0, vmax=1)
    axes[0, 0].set_title(f"GT - Axial (Z={slice_d})")
    
    axes[0, 1].imshow(orig[:, slice_h, :], cmap='gray', vmin=0, vmax=1)
    axes[0, 1].set_title(f"GT - Coronal (Y={slice_h})")
    
    axes[0, 2].imshow(orig[:, :, slice_w], cmap='gray', vmin=0, vmax=1)
    axes[0, 2].set_title(f"GT - Sagittal (X={slice_w})")

    # --- 第二行：重建数据 (Reconstruction) ---
    axes[1, 0].imshow(recon[slice_d, :, :], cmap='gray', vmin=0, vmax=1)
    axes[1, 0].set_title("Recon - Axial")
    
    axes[1, 1].imshow(recon[:, slice_h, :], cmap='gray', vmin=0, vmax=1)
    axes[1, 1].set_title("Recon - Coronal")
    
    axes[1, 2].imshow(recon[:, :, slice_w], cmap='gray', vmin=0, vmax=1)
    axes[1, 2].set_title("Recon - Sagittal")

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, f"{filename}.png")
    # plt.savefig(save_path)
    plt.show()
    # print(f"✅ Saved visualization to: {save_path}")
    plt.close()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. 初始化模型 (结构必须和训练时完全一致)
    print("Building model...")
    # 强制覆盖 Config 以匹配你当前的训练设置 (防止 import 的 config 不一致)
    CONFIG['model']['use_checkpoint'] = False 
    model = VAE3D(CONFIG['model']).to(device)
    model.eval()

    # 2. 加载权重
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"⚠️ Checkpoint not found at {CHECKPOINT_PATH}. \nWaiting for training to produce first checkpoint...")
        return

    print(f"Loading checkpoint: {CHECKPOINT_PATH}")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    
    # 兼容处理：有些代码保存时多包了一层 'vae' key，有些直接存 state_dict
    if 'vae' in checkpoint:
        state_dict = checkpoint['vae']
    else:
        state_dict = checkpoint
        
    # 处理可能的 key 前缀问题 (比如 DDP 训练会有 module. 前缀)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
            
    model.load_state_dict(new_state_dict)
    print("Model loaded successfully.")

    # 3. 加载并处理数据
    # 既然你训练是用 Patch (128)，推理我们也先切 Patch 看看细节
    # # 或者如果显存够，可以直接推全图 (256)
    # print(f"Loading data: {DATA_PATH}")
    # # 这里我们临时搜一个文件，如果你没填路径
    # if not os.path.exists(DATA_PATH):
    #     import glob
    #     files = glob.glob(os.path.join(CONFIG['data_path'], "*.npy"))
    #     if len(files) > 0:
    #         DATA_PATH = files[0]
    #         print(f"⚠️ Auto-selected data: {DATA_PATH}")
    
    # data_numpy = np.load(DATA_PATH).astype(np.float32)

    # === 修改开始 ===
    # 1. 先把全局变量赋值给一个局部变量
    target_path = DATA_PATH 

    print(f"Loading data: {target_path}")
    
    # 2. 对局部变量进行检查和修改
    if not os.path.exists(target_path):
        import glob
        # 假设 config 在这里是可用的，或者你需要从 CONFIG 字典取
        files = glob.glob(os.path.join(CONFIG['data_path'], "*.npy")) # 注意这里可能需要根据你的实际 CONFIG 调整
        if len(files) > 0:
            target_path = files[0]
            print(f"⚠️ Auto-selected data: {target_path}")
    
    # 3. 加载数据时使用 target_path
    data_numpy = np.load(target_path).astype(np.float32)
    # === 修改结束 ===
    
    # 归一化 (使用 Config 中的全局极值)
    g_min = CONFIG['global_min']
    g_max = CONFIG['global_max']
    scale = g_max - g_min
    
    data_norm = (data_numpy - g_min) / scale
    data_norm = data_norm * 2.0 - 1.0
    
    # 转 Tensor
    inputs = torch.from_numpy(data_norm).unsqueeze(0).unsqueeze(0) # (1, 1, D, H, W)
    
    # 如果全图 256^3 爆显存，这里做一个 Center Crop 到 128^3 进行测试
    # 你可以尝试注释掉下面这几行来跑全图
    D, H, W = inputs.shape[2:]
    crop_size = 160
    if D > crop_size:
        print(f"Cropping center {crop_size}^3 for visualization...")
        sz = (D - crop_size) // 2
        sy = (H - crop_size) // 2
        sx = (W - crop_size) // 2
        inputs = inputs[:, :, sz:sz+crop_size, sy:sy+crop_size, sx:sx+crop_size]
    
    inputs = inputs.to(device)

    # 4. 推理
    print("Running inference...")
    with torch.no_grad():
        # 如果训练用了 bfloat16，推理也建议开，省显存
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            recons, _, _ = model(inputs)

    # 5. 后处理与画图
    inputs = inputs.cpu().float()
    recons = recons.cpu().float()
    
    # 还原到 0-1 区间
    img_orig = denormalize(inputs[0, 0], g_min, g_max).numpy()
    img_recon = denormalize(recons[0, 0], g_min, g_max).numpy()
    
    plot_slices(img_orig, img_recon, filename="test_result")
    
    # 计算简单的误差指标
    l1_err = np.mean(np.abs(img_orig - img_recon))
    print(f"Mean L1 Error: {l1_err:.6f}")

if __name__ == "__main__":
    main()
# %%

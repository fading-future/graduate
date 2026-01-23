import torch
import numpy as np
import os
import glob
from tqdm import tqdm
from src.model_vae_3d import AutoencoderKL
from src.dataset_raw import RawCoreDataset
from torch.utils.data import DataLoader

# --- Config ---
CHECKPOINT_PATH = "./vae_results_exp01/vae_epoch_100.pth" # 训练好的权重
DATA_ROOTS = ["/path/to/raw_data"]
OUTPUT_DIR = "/path/to/save/latent_npy"
DEVICE = "cuda"

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. Load Model
    vae = AutoencoderKL(z_channels=4).to(DEVICE)
    vae.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    vae.eval()
    
    # 2. Data
    dataset = RawCoreDataset(DATA_ROOTS)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    print("🚀 Start Processing Latents...")
    
    # 3. Statistics for Stage 2 Scaling
    all_means = []
    all_stds = []

    with torch.no_grad():
        for i, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
            batch = batch.to(DEVICE)
            
            # Encode -> Get Mean
            # 我们通常使用 Mean 作为 Stage 2 的 Ground Truth，而不是采样的 z
            mean, logvar = vae.encode(batch)
            
            # 保存到硬盘
            # 原始文件名假设我们不知道，这里简单按序号保存，或者你需要修改 Dataset 返回文件名
            latent_numpy = mean.cpu().numpy().squeeze(0) # (4, 64, 64, 64)
            
            # 保存
            save_name = os.path.join(OUTPUT_DIR, f"latent_{i:05d}.npy")
            np.save(save_name, latent_numpy)
            
            # 统计分布 (用于 Stage 2 的 scale_factor)
            if i < 1000: # 抽样统计前1000个
                all_means.append(latent_numpy.mean())
                all_stds.append(latent_numpy.std())

    # 计算全局 Scale Factor
    global_std = np.mean(all_stds)
    print(f"✅ Processing Done!")
    print(f"📊 Suggested Scale Factor for Stage 2: {1.0 / global_std:.4f}")
    print(f"   (Use this value in Stage 2 config['scale_factor'])")

if __name__ == "__main__":
    main()
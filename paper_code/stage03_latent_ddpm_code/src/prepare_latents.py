import torch
import numpy as np
import os
from glob import glob
from tqdm import tqdm
import sys

from model_vqvae import VQVAE3D 

"""
准备 Stage 2 Latent Diffusion Model 所需的 VQ-VAE Latents 数据
打印转换之前和转换之后的npy 数据信息

Note: 使用时将该文件移动到src 路径下面
Result: 成功将(256, 256, 256)的体数据编码为(1, 64, 64, 64, 64)的潜在表示，并保存为 .npy 文件
"""

CONFIG = {
    'device': "cuda" if torch.cuda.is_available() else "cpu",
    'src_voxel_dir': r"d:\chendou\data\Cleaned_NPY_Dataset_21",  # 源数据路径 (256 大小的 .npy)
    'dst_latent_dir': r"d:\chendou\data\Latent_Cleaned_NPY_Dataset_21",  # 目标输出路径， Stage 2 专用 (64 大小的 .npy)
    'stage1_weight_path': r"C:\Users\vipuser\Desktop\chendou\vqvae_code\src\stage1_models\vqvae_epoch_110.pth", # VQ-VAE 模型路径 (Epoch 110)
    'porosity_threshold': 103,  # 全局物理阈值 (Triangle Method 统计得出)
}



def calculate_porosity_fixed(volume_numpy, threshold=85):
    """
    计算孔隙度：使用全局固定的物理阈值
    """
    # 1. 鲁棒归一化 (Map to 0-255)
    v_min, v_max = volume_numpy.min(), volume_numpy.max()
    if v_max - v_min > 1e-6:
        vol_norm = (volume_numpy - v_min) / (v_max - v_min)
        vol_uint8 = (vol_norm * 255).astype(np.uint8)
    else:
        vol_uint8 = np.zeros_like(volume_numpy, dtype=np.uint8)

    # 2. 阈值分割
    # 假设：数值小(暗)的是孔隙，数值大(亮)的是岩石
    voids = vol_uint8 < threshold
    porosity = np.sum(voids) / voids.size
    return porosity

def encode_dataset():
    device = CONFIG['device']
    os.makedirs(CONFIG['dst_latent_dir'], exist_ok=True)
    
    print(f"🚀 Loading VQ-VAE from: {CONFIG['stage1_weight_path']}")
    # 加载 Stage 1 模型
    vqvae = VQVAE3D(in_channels=1, embedding_dim=64, num_embeddings=2048).to(device)
    if os.path.exists(CONFIG['stage1_weight_path']):
        vqvae.load_state_dict(torch.load(CONFIG['stage1_weight_path'], map_location=device))
    else:
        raise FileNotFoundError(f"❌ 找不到权重文件: {CONFIG['stage1_weight_path']}")
    
    vqvae.eval()
    
    files = glob(os.path.join(CONFIG['src_voxel_dir'], "*.npy"))
    print(f"📂 Source Data: {len(files)} files")
    print(f"🎯 Porosity Threshold: {CONFIG['porosity_threshold']}/255")
    
    success_count = 0
    with torch.no_grad():
        for fpath in tqdm(files):
            try:
                # 1. 加载原始数据 [256, 256, 256]
                data = np.load(fpath)
                
                # 2. 计算孔隙度 (写入文件名用)
                porosity = calculate_porosity_fixed(data, threshold=CONFIG['porosity_threshold'])
                
                # 3. VQ-VAE 预处理 (归一化到 -1 ~ 1)
                data_tensor = torch.from_numpy(data).float().to(device)
                d_min, d_max = data_tensor.min(), data_tensor.max()
                
                if d_max - d_min > 1e-6:
                    data_norm = (data_tensor - d_min) / (d_max - d_min) * 2.0 - 1.0
                else:
                    data_norm = torch.zeros_like(data_tensor)
                    
                data_norm = data_norm.unsqueeze(0).unsqueeze(0) # [1, 1, 256, 256, 256]

                # 4. 编码 (Encode) -> 得到 64^3 的特征向量
                # 注意：这里保存的是 quantizer 输出前的 z_e 还是输出后的 z_q 都可以
                # 通常为了训练稳定性，保存 z_q (Quantized)
                z_q, _, _ = vqvae.encode(data_norm) 
                
                # 5. 保存 (文件名携带孔隙度信息)
                # 格式: 6-6-9_z256..._porosity_0.098.npy
                basename = os.path.basename(fpath).replace(".npy", "")
                save_name = f"{basename}_porosity_{porosity:.4f}.npy"
                
                np.save(os.path.join(CONFIG['dst_latent_dir'], save_name), z_q.cpu().numpy())
                success_count += 1
                
            except Exception as e:
                print(f"⚠️ Error processing {fpath}: {e}")

    print(f"\n🎉 转换完成！成功处理: {success_count}/{len(files)}")
    print(f"💾 Latents 已保存在: {CONFIG['dst_latent_dir']}")
    print("👉 请记得更新 config.py 中的 'processed_data_dir' 为上述路径！")

if __name__ == "__main__":
    # 1. 执行数据转换
    encode_dataset()

    # # 2. 打印某个转换之前的npy 文件的信息
    # test_npy_path = r"D:\多尺度岩心数据集\Final_Dataset_NPY_12\6-6-12_z256_y853_x1137.npy"
    # if os.path.exists(test_npy_path):
    #     data = np.load(test_npy_path)
    #     print(f"Test NPY Shape: {data.shape}, Dtype: {data.dtype}, Min: {data.min()}, Max: {data.max()}")

    # # 3. 打印某个转换之后的npy 文件的信息
    # test_npy_path = r"C:\Users\Administrator\Desktop\paper\stage2_latentddpm_code\data\6-6-12_z1024_y271_x498_porosity_0.2421.npy"
    # if os.path.exists(test_npy_path):
    #     data = np.load(test_npy_path)
    #     print(f"Test NPY Shape: {data.shape}, Dtype: {data.dtype}, Min: {data.min()}, Max: {data.max()}")
import torch
import numpy as np
import os
import glob
import pandas as pd
import yaml
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from models.vae import KLVAE3D

# ================= 配置区域 =================
CONFIG_PATH = "/chendou_space/chendou/paper_code/stage02_KLvae_single_code_v2/config/train_config copy.yaml"
# 请确保这里替换成你最新的 checkpoint
CHECKPOINT_PATH = "/chendou_space/chendou/paper_code/stage02_KLvae_single_code_v2/experiments/exp01_cube_structure_v1/checkpoint_epoch_26.pt" 
# CSV 文件路径
CSV_PATH = "/chendou_space/data/aligned_Training_Data/processing_report.csv"
# 输出路径 (建议改个名，区分 crop 版和 full 版)
SAVE_DIR = "/chendou_space/data/stage2_latents_full_256" 

# 显卡设置
DEVICE = "cuda"
# A100 80G 处理 256^3 的数据，Batch Size 建议设为 1 或 2，大了容易 OOM
BATCH_SIZE = 8 

# ================= 1. 修改后的 Dataset (不裁剪) =================
class InferenceDataset(Dataset):
    def __init__(self, data_root, ext=".npy"):
        # 移除了 crop_size 参数，因为不需要了
        self.files = sorted(glob.glob(os.path.join(data_root, f"*{ext}")))
        print(f"Found {len(self.files)} files for FULL FRAME inference.")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        file_name = os.path.basename(path) 
        
        # 加载完整数据 [256, 256, 256]
        # 注意：这里去掉了 mmap_mode，直接读入内存。
        # A100 机器内存通常很大，直接读更快且不易出错。
        data = np.load(path) 
        
        # 归一化 [-1, 1]
        data = data.astype(np.float32)
        data = (data / 65535.0) * 2.0 - 1.0

        # === 核心修改：不再裁剪 ===
        # 直接增加 Channel 维度 -> [1, 256, 256, 256]
        data_tensor = torch.from_numpy(data).unsqueeze(0)
        
        # 返回：数据，文件名
        return data_tensor, file_name

def load_porosity_map(csv_path):
    print(f"Loading porosity labels from {csv_path}...")
    df = pd.read_csv(csv_path)
    porosity_map = dict(zip(df['file'], df['porosity']))
    return porosity_map

def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    # 1. 加载配置
    with open(CONFIG_PATH, 'r') as f:
        cfg = yaml.safe_load(f)

    # 2. 加载 CSV 映射表
    porosity_map = load_porosity_map(CSV_PATH)

    # 3. 初始化模型
    vae = KLVAE3D(cfg).to(DEVICE)
    
    # 加载权重
    print(f"Loading model from {CHECKPOINT_PATH}...")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    state_dict = checkpoint['vae_state_dict']
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("_orig_mod."):
            new_state_dict[k[10:]] = v
        else:
            new_state_dict[k] = v
    vae.load_state_dict(new_state_dict)
    
    # 开启编译加速 (如果在 A100 上跑不通，可以注释掉)
    # vae = torch.compile(vae)
    vae.eval()

    # 4. 准备数据
    # 这里不再传递 crop_size
    dataset = InferenceDataset(cfg['data']['data_root'], cfg['data']['file_extension'])
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True)

    print("Start generating latents from FULL 256^3 images...")
    
    all_latents_stats = [] 
    save_count = 0

    with torch.no_grad():
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            for batch_data, batch_filenames in tqdm(dataloader):
                batch_data = batch_data.to(DEVICE)
                
                # VAE 编码 -> [B, 4, 32, 32, 32]
                posterior = vae.encode(batch_data)
                z = posterior.sample() 
                
                z_np = z.cpu().float().numpy()
                
                # 遍历 Batch 保存
                for i in range(z_np.shape[0]):
                    orig_name = batch_filenames[i]
                    latent_data = z_np[i]
                    
                    if orig_name in porosity_map:
                        por = porosity_map[orig_name]
                        
                        # 文件名格式: porosity_{孔隙率}_{原文件名}.npy
                        # 不需要 id 计数器了，因为每个文件只处理一次
                        new_name = f"porosity_{por:.6f}_{orig_name}"
                        
                        np.save(os.path.join(SAVE_DIR, new_name), latent_data)
                        save_count += 1
                        
                        # 收集统计信息 (收集前 500 个就够了，全尺寸数据比较大)
                        if len(all_latents_stats) < 500:
                            all_latents_stats.append(latent_data.flatten())
                    else:
                        print(f"Warning: {orig_name} not found in CSV, skipping...")

    # 5. 计算 Scale Factor
    print("Calculating statistics...")
    if len(all_latents_stats) > 0:
        all_data = np.concatenate(all_latents_stats)
        std = np.std(all_data)
        mean = np.mean(all_data)
        
        print("\n" + "="*40)
        print("✅ Data Preparation Finished!")
        print(f"Total files saved: {save_count}")
        print(f"Saved to: {SAVE_DIR}")
        print("-" * 40)
        print(f"⚠️  STAGE 2 CONFIGURATION INFO ⚠️")
        print(f"Latent Mean: {mean:.6f}")
        print(f"Latent Std:  {std:.6f}")
        print(f"🚀 Set 'scale_factor' in Stage 2 config to: {1.0 / std:.6f}")
        print("="*40)
    else:
        print("Error: No data saved.")

if __name__ == "__main__":
    main()


# import torch
# import numpy as np
# import os
# import glob
# import pandas as pd  # 需要安装 pandas: pip install pandas
# import yaml
# import argparse
# from tqdm import tqdm
# from torch.utils.data import DataLoader, Dataset
# from models.vae import KLVAE3D

# # ================= 配置区域 =================
# CONFIG_PATH = "/chendou_space/chendou/paper_code/stage02_KLvae_single_code_v2/config/train_config copy.yaml"
# CHECKPOINT_PATH = "/chendou_space/chendou/paper_code/stage02_KLvae_single_code_v2/experiments/exp01_cube_structure_v1/checkpoint_epoch_26.pt" # 替换成你最好的那个模型
# # CSV 文件路径
# CSV_PATH = "/chendou_space/data/aligned_Training_Data/processing_report.csv"
# # 输出路径
# SAVE_DIR = "/chendou_space/data/stage2_latents_with_cond" 

# # 显卡设置
# DEVICE = "cuda"
# BATCH_SIZE = 2 # A100 推理可以开很大，为了快

# # ================= 1. 定义一个能返回文件名的 Dataset =================
# class InferenceDataset(Dataset):
#     def __init__(self, data_root, ext=".npy", crop_size=96): # crop_size 和训练保持一致
#         self.files = sorted(glob.glob(os.path.join(data_root, f"*{ext}")))
#         self.crop_size = crop_size
#         print(f"Found {len(self.files)} files for inference.")

#     def __len__(self):
#         return len(self.files)

#     def __getitem__(self, idx):
#         path = self.files[idx]
#         file_name = os.path.basename(path) # 获取文件名，如 "6-6-20 全部_z3840.npy"
        
#         # 使用 mmap 加速读取
#         data = np.load(path, mmap_mode='r') 
        
#         # 归一化 [-1, 1]
#         data = data.astype(np.float32)
#         data = (data / 65535.0) * 2.0 - 1.0

#         # === 核心逻辑：推理时如何裁剪？===
#         # 方案 A：随机裁剪 (适合增加数据量)
#         # 方案 B：中心裁剪 (适合测试)
#         # 这里为了给 Stage 2 准备丰富的训练数据，我们依然使用【随机裁剪】
#         # 这样同一个原始大文件，可以切出很多个不同的 Latent 块
#         d, h, w = data.shape
#         d_s = np.random.randint(0, max(0, d - self.crop_size))
#         h_s = np.random.randint(0, max(0, h - self.crop_size))
#         w_s = np.random.randint(0, max(0, w - self.crop_size))
        
#         data_crop = data[d_s:d_s+self.crop_size, h_s:h_s+self.crop_size, w_s:w_s+self.crop_size].copy()
        
#         data_tensor = torch.from_numpy(data_crop).unsqueeze(0)
        
#         # 返回：数据，文件名
#         return data_tensor, file_name

# def load_porosity_map(csv_path):
#     print(f"Loading porosity labels from {csv_path}...")
#     df = pd.read_csv(csv_path)
#     # 建立字典: { "文件名.npy": 0.000397 }
#     # 确保 CSV 里的 file 列和磁盘上的文件名一致
#     porosity_map = dict(zip(df['file'], df['porosity']))
#     return porosity_map

# def main():
#     os.makedirs(SAVE_DIR, exist_ok=True)
    
#     # 1. 加载配置
#     with open(CONFIG_PATH, 'r') as f:
#         cfg = yaml.safe_load(f)

#     # 2. 加载 CSV 映射表
#     porosity_map = load_porosity_map(CSV_PATH)

#     # 3. 初始化模型
#     vae = KLVAE3D(cfg).to(DEVICE)
    
#     # 加载权重 (自动处理前缀)
#     print(f"Loading model from {CHECKPOINT_PATH}...")
#     checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
#     state_dict = checkpoint['vae_state_dict']
#     new_state_dict = {}
#     for k, v in state_dict.items():
#         if k.startswith("_orig_mod."):
#             new_state_dict[k[10:]] = v
#         else:
#             new_state_dict[k] = v
#     vae.load_state_dict(new_state_dict)
    
#     # 开启编译加速推理 (可选，如果报错就注释掉)
#     # vae = torch.compile(vae)
#     vae.eval()

#     # 4. 准备数据
#     # 注意：这里我们为了生成更多数据，可以多次遍历 Dataset，或者让 Dataset 长度翻倍
#     # 为了简单，这里只遍历一次。如果数据不够，你可以运行这个脚本多次（因为是随机裁剪）
#     dataset = InferenceDataset(cfg['data']['data_root'], cfg['data']['file_extension'], crop_size=cfg['data']['croped_size'])
#     dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True)

#     print("Start generating latents...")
    
#     all_latents_stats = [] # 用于计算 std
#     save_count = 0

#     with torch.no_grad():
#         with torch.amp.autocast('cuda', dtype=torch.bfloat16):
#             for batch_data, batch_filenames in tqdm(dataloader):
#                 batch_data = batch_data.to(DEVICE)
                
#                 # VAE 编码 -> 得到分布 -> 采样
#                 posterior = vae.encode(batch_data)
#                 z = posterior.sample() # [B, 4, 12, 12, 12] (假设下采样8倍)
                
#                 z_np = z.cpu().float().numpy()
                
#                 # 遍历 Batch 保存
#                 for i in range(z_np.shape[0]):
#                     orig_name = batch_filenames[i]
#                     latent_data = z_np[i]
                    
#                     # 查表获取孔隙率
#                     if orig_name in porosity_map:
#                         por = porosity_map[orig_name]
                        
#                         # === 构造新文件名 ===
#                         # 格式: porosity_{孔隙率:.6f}_{原文件名}.npy
#                         # 例如: porosity_0.000397_6-6-20_z3840.npy
#                         # 这里的 save_count 用于防止重名覆盖（如果多次运行或同名文件切多次）
#                         new_name = f"porosity_{por:.6f}_id{save_count}_{orig_name}"
                        
#                         np.save(os.path.join(SAVE_DIR, new_name), latent_data)
#                         save_count += 1
                        
#                         # 收集统计信息 (只收集前 2000 个以节省内存)
#                         if len(all_latents_stats) < 2000:
#                             all_latents_stats.append(latent_data.flatten())
#                     else:
#                         print(f"Warning: {orig_name} not found in CSV, skipping...")

#     # 5. 计算 Scale Factor (用于 Stage 2 缩放)
#     print("Calculating statistics...")
#     if len(all_latents_stats) > 0:
#         all_data = np.concatenate(all_latents_stats)
#         std = np.std(all_data)
#         mean = np.mean(all_data)
        
#         print("\n" + "="*40)
#         print("✅ Data Preparation Finished!")
#         print(f"Total files saved: {save_count}")
#         print(f"Saved to: {SAVE_DIR}")
#         print("-" * 40)
#         print(f"⚠️  STAGE 2 CONFIGURATION INFO ⚠️")
#         print(f"Latent Mean: {mean:.6f}")
#         print(f"Latent Std:  {std:.6f}")
#         print(f"🚀 Set 'scale_factor' in Stage 2 config to: {1.0 / std:.6f}")
#         print("="*40)
#     else:
#         print("Error: No data saved.")

# if __name__ == "__main__":
#     main()


# import torch
# import numpy as np
# import os
# import glob
# from tqdm import tqdm
# from torch.utils.data import DataLoader
# from models.vae import KLVAE3D
# from data.dataset import CubeDataset
# import yaml

# # === 配置 ===
# CONFIG_PATH = "/chendou_space/chendou/paper_code/stage02_KLvae_single_code_v2/config/train_config copy.yaml"
# CHECKPOINT_PATH = "/chendou_space/chendou/paper_code/stage02_KLvae_single_code_v2/experiments/exp01_cube_structure_v1/checkpoint_epoch_26.pt" # 替换成你最好的那个模型
# SAVE_DIR = "/chendou_space/data/stage2_latents_data" # 预处理后的数据存这
# BATCH_SIZE = 1 # 推理时可以用大一点
# DEVICE = "cuda"

# def main():
#     # 1. 加载配置和模型
#     with open(CONFIG_PATH, 'r') as f:
#         cfg = yaml.safe_load(f)
    
#     os.makedirs(SAVE_DIR, exist_ok=True)
    
#     # 初始化模型
#     vae = KLVAE3D(cfg).to(DEVICE)
    
#     # 加载权重 (处理 _orig_mod 前缀)
#     checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
#     state_dict = checkpoint['vae_state_dict']
#     new_state_dict = {}
#     for k, v in state_dict.items():
#         if k.startswith("_orig_mod."):
#             new_state_dict[k[10:]] = v
#         else:
#             new_state_dict[k] = v
#     vae.load_state_dict(new_state_dict)
#     vae.eval()
    
#     # 2. 加载数据
#     # 注意：这里不用 crop，最好是让 VAE 处理同样大小的块，或者按需处理
#     dataset = CubeDataset(cfg['data']['data_root'], cfg['data']['file_extension'], crop_size=cfg['data']['croped_size'], is_train=False) 
#     dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8)

#     print(f"Start processing {len(dataset)} files...")
    
#     all_latents = []

#     # 3. 推理并保存
#     with torch.no_grad():
#         for i, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
#             batch = batch.to(DEVICE)
            
#             # 编码得到分布
#             posterior = vae.encode(batch)
#             # 关键：取分布的均值作为 Latent (比采样更稳)，或者 sample() 也可以
#             # 为了 Stage 2 训练稳定，通常取 sample() 能够增加鲁棒性，或者取 mode()
#             # 这里建议取 sample()，保留 VAE 的特性
#             z = posterior.sample() 
            
#             # 转回 CPU 存起来
#             z_np = z.cpu().numpy() # [B, C, d, h, w]
            
#             # 逐个保存文件
#             for j in range(z_np.shape[0]):
#                 file_name = f"latent_{i*BATCH_SIZE + j:05d}.npy"
#                 np.save(os.path.join(SAVE_DIR, file_name), z_np[j])
                
#             # 收集一部分数据用于计算 Scale Factor (不用全部，太占内存)
#             if len(all_latents) * BATCH_SIZE < 1000: 
#                 all_latents.append(z_np)

#     # 4. 计算 Scale Factor
#     all_latents = np.concatenate(all_latents, axis=0)
#     std = np.std(all_latents)
#     mean = np.mean(all_latents)
    
#     print(f"✅ Data processing done!")
#     print(f"Saved to {SAVE_DIR}")
#     print(f"⚠️ Scale Factor Calculation (Very Important for Stage 2):")
#     print(f"   Mean: {mean:.6f}")
#     print(f"   Std:  {std:.6f}")
#     print(f"🚀 In Stage 2 config, set 'scale_factor' to: {1.0 / std:.6f}")

# if __name__ == "__main__":
#     main()
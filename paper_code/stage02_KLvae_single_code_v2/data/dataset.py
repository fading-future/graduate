import torch
from torch.utils.data import Dataset
import numpy as np
import os
import glob
import random

class CubeDataset(Dataset):
    def __init__(self, data_root, ext=".npy", crop_size=128, is_train=True):
        self.files = sorted(glob.glob(os.path.join(data_root, f"*{ext}")))[:8000]
        self.crop_size = crop_size
        self.is_train = is_train
        print(f"Dataset: {len(self.files)} files. Crop size: {crop_size}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]

        data = np.load(path, mmap_mode='r').astype(np.float32)
        data = data / 127.5 - 1.0   # 等价于 (data/255)*2 - 1

        # 2. 随机切块 (Random Crop)
        if self.is_train:
            # 只有训练时才切块
            d, h, w = data.shape
            d_s = random.randint(0, max(0, d - self.crop_size))
            h_s = random.randint(0, max(0, h - self.crop_size))
            w_s = random.randint(0, max(0, w - self.crop_size))
            
            data_crop = data[d_s:d_s+self.crop_size, h_s:h_s+self.crop_size, w_s:w_s+self.crop_size].copy()
            
            # === 新增：3D 数据增强 (Data Augmentation) ===
            # 仅在训练模式下启用
            
            # A. 随机翻转 (Flip): 针对 D, H, W 三个轴独立随机翻转
            # 相当于增加了 2*2*2 = 8 倍变化
            if random.random() < 0.5:
                data_crop = np.flip(data_crop, axis=0) # Flip Depth
            if random.random() < 0.5:
                data_crop = np.flip(data_crop, axis=1) # Flip Height
            if random.random() < 0.5:
                data_crop = np.flip(data_crop, axis=2) # Flip Width
            
            # B. 随机 90度 旋转 (Rotate): 在 H-W 平面上随机旋转 0, 90, 180, 270 度
            # 你也可以改为随机选择轴 (axis=(0,1) 或 (0,2))，但在 H-W 上旋转最快且够用了
            k = random.randint(0, 3)
            if k > 0:
                data_crop = np.rot90(data_crop, k=k, axes=(1, 2))
            
            # 此时 data_crop 仍然是 numpy array，但在内存中可能不连续了，为了保险起见：
            data_crop = data_crop.copy()
            
        else:
            # 验证/测试时保持原样
            data_crop = data

        # 3. 增加 Channel 维度 -> [1, D, H, W]
        data_tensor = torch.from_numpy(data_crop).unsqueeze(0)
        return data_tensor

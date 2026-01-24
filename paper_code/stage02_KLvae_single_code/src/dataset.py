import os
import glob
import torch
import numpy as np
from torch.utils.data import Dataset

class Core3DDataset(Dataset):
    def __init__(self, data_dir, global_min, global_max):
        """
        Args:
            data_dir (str): 数据文件夹路径
            global_min (float): 也就是 calculate_global_stats.py 算出来的最小值
            global_max (float): 也就是 calculate_global_stats.py 算出来的最大值
        """
        self.files = sorted(glob.glob(os.path.join(data_dir, "*.npy")))
        self.global_min = float(global_min)
        self.global_max = float(global_max)
        
        # 预计算分母，避免每次 getitem 都算一次除法，稍微提速
        self.scale = self.global_max - self.global_min
        if self.scale <= 0:
            raise ValueError(f"❌ Invalid stats: Max ({self.global_max}) must be greater than Min ({self.global_min})")

        print(f"Dataset initialized. Norm range: [{self.global_min}, {self.global_max}] -> [-1, 1]")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        
        try:
            # 1. Load Data
            # 你的原始数据是 uint16, shape (256, 256, 256)
            data_numpy = np.load(file_path).astype(np.float32)
            
            # 2. Normalize to [-1, 1] using Global Stats
            # 公式: (x - min) / (max - min) * 2 - 1
            data_norm = (data_numpy - self.global_min) / self.scale
            data_norm = data_norm * 2.0 - 1.0
            
            # 3. To Tensor & Add Channel Dimension
            # Output Shape: (1, 256, 256, 256)
            data_tensor = torch.from_numpy(data_norm).unsqueeze(0)
            
            return data_tensor
            
        except Exception as e:
            print(f"⚠️ Error loading {file_path}: {e}")
            # 简单的容错机制：如果读取失败，随机返回另一个样本
            new_idx = np.random.randint(0, len(self.files))
            return self.__getitem__(new_idx)
import os
import glob
import random
import numpy as np
import torch
from torch.utils.data import Dataset

class VQVAEDataset(Dataset):
    def __init__(self, data_dir, volume_size=256, augment=True):
        """
        Stage 1 专用 Dataset (强力清洗版)
        """
        self.file_list = sorted(glob.glob(os.path.join(data_dir, "**", "*.npy"), recursive=True))
        
        if len(self.file_list) == 0:
            raise ValueError(f"在路径 {data_dir} 下未找到任何 .npy 文件，请检查路径设置。")
            
        self.volume_size = volume_size
        self.augment = augment 
        
        print(f"Stage 1 Dataset loaded: {len(self.file_list)} files.")

    def __len__(self):
        return len(self.file_list)

    def normalize(self, volume):
        """归一化到 [-1, 1]"""
        volume = volume.float()
        volume = (volume / 127.5) - 1.0
        return torch.clamp(volume, -1.0, 1.0) # 钳制防止溢出

    def random_transform(self, volume):
        # ... (保持原本的增强逻辑不变) ...
        for dim in [0, 1, 2]: 
            if random.random() > 0.5:
                volume = torch.flip(volume, dims=[dim])
        k = random.randint(0, 3)
        if k > 0:
            volume = torch.rot90(volume, k, dims=(1, 2)) 
        return volume

    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        
        try:
            # --- 1. 加载数据 ---
            data_numpy = np.load(file_path)
            
            # --- [核心修复] 源头强力清洗 ---
            # 无论原来是什么，只要有 NaN/Inf，直接变 0
            if np.isnan(data_numpy).any() or np.isinf(data_numpy).any():
                data_numpy = np.nan_to_num(data_numpy, nan=0.0, posinf=255.0, neginf=0.0)
            
            # 强制类型转换，防止 float16/uint16 溢出
            data_numpy = data_numpy.astype(np.float32)
            
            # 兼容性处理：如果数值范围是 0-65535，压缩到 0-255
            if data_numpy.max() > 255.0:
                data_numpy = (data_numpy / 65535.0 * 255.0)
            
            # 转 Tensor
            gt_volume = torch.from_numpy(data_numpy).float()
            
            # --- [核心修复] 处理全黑/损坏数据 ---
            # 如果最大值最小值一样，为了防止后续计算出问题，直接返回全0
            if gt_volume.max() == gt_volume.min():
                 gt_volume = torch.zeros_like(gt_volume)

            # --- 2. 随机裁剪 ---
            D, H, W = gt_volume.shape
            # Pad 如果太小
            if D < self.volume_size or H < self.volume_size or W < self.volume_size:
                 pad_d = max(0, self.volume_size - D)
                 pad_h = max(0, self.volume_size - H)
                 pad_w = max(0, self.volume_size - W)
                 gt_volume = torch.nn.functional.pad(gt_volume, (0, pad_w, 0, pad_h, 0, pad_d))
                 D, H, W = gt_volume.shape

            # Crop 如果太大
            if D > self.volume_size:
                z = random.randint(0, D - self.volume_size)
                gt_volume = gt_volume[z:z+self.volume_size, :, :]
            if H > self.volume_size:
                y = random.randint(0, H - self.volume_size)
                gt_volume = gt_volume[:, y:y+self.volume_size, :]
            if W > self.volume_size:
                x = random.randint(0, W - self.volume_size)
                gt_volume = gt_volume[:, :, x:x+self.volume_size]
            
            # --- 3. 归一化 ---
            gt_volume = self.normalize(gt_volume)
            
            # --- 4. 增强 ---
            if self.augment:
                gt_volume = self.random_transform(gt_volume)
            
            gt_volume = gt_volume.unsqueeze(0)
            
            # --- [最后一道防线] ---
            # 确保出去的 Tensor 绝对没有 NaN
            if torch.isnan(gt_volume).any():
                return {"GT": torch.zeros((1, self.volume_size, self.volume_size, self.volume_size))}

            return {"GT": gt_volume}
            
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return {"GT": torch.zeros((1, self.volume_size, self.volume_size, self.volume_size))}
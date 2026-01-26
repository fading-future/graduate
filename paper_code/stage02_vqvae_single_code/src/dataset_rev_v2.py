import os
import glob
import random
import numpy as np
import torch
from torch.utils.data import Dataset

class VQVAEDataset(Dataset):
    def __init__(self, data_dir, volume_size=64, augment=True):
        """
        Stage 1 Dataset
        """
        # 递归搜索所有子目录下的 .npy
        self.file_list = sorted(glob.glob(os.path.join(data_dir, "**", "*.npy"), recursive=True))
        
        if len(self.file_list) == 0:
            raise ValueError(f"错误：在 {data_dir} 及其子目录下未找到 .npy 文件。")
            
        self.volume_size = volume_size
        self.augment = augment 
        
        print(f"[Dataset] Loaded {len(self.file_list)} files. Size: {volume_size}^3, Augment: {augment}")

    def __len__(self):
        return len(self.file_list)

    def normalize(self, volume):
        """
        归一化到 [-1, 1]。
        增加极小值保护，防止除以0。
        """
        volume = volume.float()
        v_min = volume.min()
        v_max = volume.max()
        
        # 如果数据块是纯色的（例如全是背景或全是石头），直接返回全0或全-1
        if v_max - v_min < 1e-6:
            return torch.zeros_like(volume)
            
        # 线性映射
        return (volume - v_min) / (v_max - v_min) * 2.0 - 1.0

    def random_transform(self, volume):
        # 1. 随机翻转 (D, H, W 三轴)
        if random.random() > 0.5:
            volume = torch.flip(volume, dims=[0]) # D
        if random.random() > 0.5:
            volume = torch.flip(volume, dims=[1]) # H
        if random.random() > 0.5:
            volume = torch.flip(volume, dims=[2]) # W
            
        # 2. 随机旋转 (90度倍数，保持体素对齐)
        # 随机选择旋转轴：(0,1), (0,2), (1,2)
        dims = [(0, 1), (0, 2), (1, 2)]
        k = random.randint(0, 3)
        if k > 0:
            axis = random.choice(dims)
            volume = torch.rot90(volume, k, dims=axis)
            
        return volume

    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        
        try:
            # 加载
            data_numpy = np.load(file_path)
            gt_volume = torch.from_numpy(data_numpy) # [D, H, W]
            
            # 随机裁剪 (在归一化之前裁剪，这样归一化更贴合局部特征，或者归一化后裁剪也可，取决于物理意义)
            # 这里建议：先裁剪，再归一化 (Instance Norm 针对局部块)
            D, H, W = gt_volume.shape
            
            # 只有当数据大于目标尺寸时才裁剪
            if D > self.volume_size:
                z = random.randint(0, D - self.volume_size)
                y = random.randint(0, H - self.volume_size)
                x = random.randint(0, W - self.volume_size)
                gt_volume = gt_volume[z:z+self.volume_size, y:y+self.volume_size, x:x+self.volume_size]
            
            # 归一化
            gt_volume = self.normalize(gt_volume)
            
            # 数据增强
            if self.augment:
                gt_volume = self.random_transform(gt_volume)
            
            # 增加 Channel 维度 -> [1, D, H, W]
            return {"GT": gt_volume.unsqueeze(0)}
            
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            # 容错：返回一个全黑的数据块，避免 Crash
            return {"GT": torch.zeros((1, self.volume_size, self.volume_size, self.volume_size))}
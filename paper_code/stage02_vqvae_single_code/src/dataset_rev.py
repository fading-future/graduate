import os
import glob
import random
import numpy as np
import torch
from torch.utils.data import Dataset

class VQVAEDataset(Dataset):
    def __init__(self, data_dir, volume_size=256, augment=True):
        """
        Stage 1 专用 Dataset
        """
        # 【修改点 2：递归读取子文件夹】
        # "**/ *.npy" 配合 recursive=True 可以搜索所有子目录
        self.file_list = sorted(glob.glob(os.path.join(data_dir, "**", "*.npy"), recursive=True))
        
        # 建议加一个检查，防止路径写错没读到文件
        if len(self.file_list) == 0:
            raise ValueError(f"在路径 {data_dir} 下未找到任何 .npy 文件，请检查路径设置。")
            
        self.volume_size = volume_size
        self.augment = augment 
        
        print(f"Stage 1 Dataset loaded: {len(self.file_list)} files.")
        print(f"Target Size: {volume_size}^3 | Augmentation: {augment}")

    def __len__(self):
        return len(self.file_list)

    def normalize(self, volume):
        """
        【关键】将 16-bit 数据 (0-65535) 或 其他范围数据 归一化到 [-1, 1]
        """
        # 1. 转换类型
        volume = volume.float()
        
        # 2. 获取当前数据的最大最小值 (Instance Normalization)
        # 既然是 CT，如果不确定全局最大值，用当前块的 min/max 最稳妥
        v_min = volume.min()
        v_max = volume.max()
        
        # 防止除以 0
        if v_max - v_min < 1e-5:
            return torch.zeros_like(volume)
            
        # 3. 线性映射到 [0, 1] -> [-1, 1]
        # 公式: (x - min) / (max - min) * 2 - 1
        volume = (volume - v_min) / (v_max - v_min) * 2.0 - 1.0
        
        return volume

    def random_transform(self, volume):
        # 随机翻转
        for dim in [1, 2, 3]: # Skip batch/channel dim if exists, here input is [D,H,W]
            if random.random() > 0.5:
                volume = torch.flip(volume, dims=[dim-1]) # dim-1 because volume is [D,H,W]
        # 随机旋转 (XY平面)
        k = random.randint(0, 3)
        if k > 0:
            volume = torch.rot90(volume, k, dims=(1, 2)) 
        return volume

    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        
        try:
            # 加载数据 [256, 256, 256]
            data_numpy = np.load(file_path)
            gt_volume = torch.from_numpy(data_numpy) # [D, H, W]
            
            # --- 1. 归一化 (必须在进入网络前完成) ---
            gt_volume = self.normalize(gt_volume)
            
            # --- 2. 随机裁剪 (如果需要) ---
            # 如果原始数据就是 256，且 volume_size=256，这一步其实就是原样返回
            # 为了兼容性，保留逻辑
            D, H, W = gt_volume.shape
            if D > self.volume_size:
                z = random.randint(0, D - self.volume_size)
                y = random.randint(0, H - self.volume_size)
                x = random.randint(0, W - self.volume_size)
                gt_volume = gt_volume[z:z+self.volume_size, y:y+self.volume_size, x:x+self.volume_size]
            
            # --- 3. 数据增强 ---
            if self.augment:
                gt_volume = self.random_transform(gt_volume)
            
            # 增加 Channel 维度: [D, H, W] -> [1, D, H, W]
            gt_volume = gt_volume.unsqueeze(0)
            
            return {
                "GT": gt_volume
            }
            
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            # 返回随机噪声或者下一个数据，防止训练中断 (简单处理：报错)
            return torch.zeros((1, self.volume_size, self.volume_size, self.volume_size))
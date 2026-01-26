import os
import glob
import random
import re
import numpy as np
import torch
from torch.utils.data import Dataset
from config import CONFIG

class LatentDataset(Dataset):
    def __init__(self, data_dir, augment=True):
        self.augment = augment 
        self.file_list = []

        if isinstance(data_dir, str):
            data_dir_list = [data_dir]
        else:
            data_dir_list = data_dir
            
        print(f"Loading data from {len(data_dir_list)} directories...")
        for d in data_dir_list:
            if not os.path.exists(d):
                print(f"⚠️ Warning: Directory not found: {d}")
                continue
            files = sorted(glob.glob(os.path.join(d, "*.npy")))
            self.file_list.extend(files)
            
        if len(self.file_list) == 0:
            raise ValueError("❌ No .npy files found! Check config paths.")
        print(f"Total LatentDataset size: {len(self.file_list)} files.")

    def __len__(self):
        return len(self.file_list)

    def extract_porosity(self, filename):
        # 匹配 porosity_0.123456_...
        match = re.search(r'porosity_(\d+\.\d+)', filename)
        if match:
            return float(match.group(1))
        return 0.5 # 默认值

    def apply_mask(self, latent):
        """
        你的原始 Mask 逻辑非常棒，完美契合 '切掉一半补一半' 的论文故事。
        这里稍作适配以兼容 32^3 尺寸。
        """
        C, D, H, W = latent.shape
        mask = torch.zeros((1, D, H, W), dtype=torch.float32)

        # 动态计算厚度 (针对 32 尺寸适配)
        # 32 * 0.25 = 8, 32 * 0.7 = 22
        min_thick = max(4, int(D * 0.25))
        max_thick = max(10, int(D * 0.7))
        thickness = random.randint(min_thick, max_thick)
        
        # 模式选择
        mode = random.choice(['one_face', 'corner', 'random_box'])
        
        if mode == 'one_face':
            # 单面保留 (模拟切掉另一半)
            face = random.choice(['top', 'bottom', 'left', 'right', 'front', 'back'])
            if face == 'top': mask[..., :thickness, :, :] = 1
            elif face == 'bottom': mask[..., -thickness:, :, :] = 1
            elif face == 'left': mask[..., :, :thickness, :] = 1
            elif face == 'right': mask[..., :, -thickness:, :] = 1
            elif face == 'front': mask[..., :, :, :thickness] = 1
            elif face == 'back': mask[..., :, :, -thickness:] = 1
            
        elif mode == 'corner':
            # 角落保留
            mask[..., :thickness, :, :] = 1
            mask[..., :, :thickness, :] = 1
            mask[..., :, :, :thickness] = 1
            
        elif mode == 'random_box':
            # 随机挖孔 (Inpainting)
            hole_size = random.randint(int(D*0.3), int(D*0.6))
            z = random.randint(0, D - hole_size)
            y = random.randint(0, H - hole_size)
            x = random.randint(0, W - hole_size)
            mask.fill_(1.0)
            mask[..., z:z+hole_size, y:y+hole_size, x:x+hole_size] = 0 

        masked_latent = latent * mask 
        return masked_latent, mask

    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        filename = os.path.basename(file_path)
        
        try:
            # 加载数据 [4, 32, 32, 32]
            data_numpy = np.load(file_path)
        except:
            return self.__getitem__(random.randint(0, len(self.file_list)-1))
        
        latent = torch.from_numpy(data_numpy).float()
        
        # === 你的核心逻辑修正 ===
        # 1. 缩放: 让数据变成 N(0, 1)
        latent = latent * CONFIG['scale_factor']
        
        # 2. 截断: 防止极端值 (离群点) 破坏训练
        # 你的 config 里 safe_threshold 设为 6.0 或 10.0 都行
        limit = CONFIG['safe_threshold']
        latent = torch.clamp(latent, min=-limit, max=limit)

        if latent.dim() == 5: latent = latent.squeeze(0)
            
        if self.augment:
            if random.random() > 0.5: latent = torch.flip(latent, dims=[1]) # Z
            if random.random() > 0.5: latent = torch.flip(latent, dims=[2]) # Y
            if random.random() > 0.5: latent = torch.flip(latent, dims=[3]) # X
        
        # 提取孔隙率
        porosity_val = self.extract_porosity(filename)
        porosity = torch.tensor([porosity_val], dtype=torch.float32)
        
        # 生成 Mask 和 Condition
        condition_latent, mask = self.apply_mask(latent)
        
        return {
            "GT": latent,
            "Condition": condition_latent,
            "Mask": mask,
            "Porosity": porosity
        }
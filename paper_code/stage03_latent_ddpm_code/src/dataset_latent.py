import os
import glob
import random
import re
import numpy as np
import torch
from torch.utils.data import Dataset
from src.config import CONFIG

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
            raise ValueError("No .npy files found! Check config paths.")
        
        # self.file_list = self.file_list[:3200]
        print(f"Total LatentDataset size: {len(self.file_list)} files.")

    def __len__(self):
        return len(self.file_list)

    def extract_porosity(self, filename):
        # 支持 porosity_0.123 / porosity_1 / porosity_.5 等
        match = re.search(r'porosity_([0-9]*\.?[0-9]+)', filename)
        if match:
            try:
                return float(match.group(1))
            except:
                pass
        return 0.5

    def apply_mask(self, latent):
        """
        mask 语义：
        - 1.0 = known / observed
        - 0.0 = unknown / to inpaint

        你的策略：
        A) 已知一半（上/下/左/右/前/后），推理剩余一半
        B) 已知两半（左右 or 上下 or 前后），推理中间部分
        C) 已知三面（缺一个角），推理缺失角
        D) 随机挖空一块，已知剩余，推理挖空部分
        """
        C, D, H, W = latent.shape
        mask = torch.zeros((1, D, H, W), dtype=torch.float32)  # 默认全未知

        # ---- 模式选择权重（只影响训练分布，不影响其它逻辑）----
        mode = random.choices(
            ['half', 'two_halves', 'corner_missing', 'random_box'],
            weights=[80, 10, 5, 5],
            k=1
        )[0]

        # ========== A) 已知一半 -> 推理另一半 ==========
        if mode == 'half':
            # 让切分位置在 45%~55% 附近波动，避免固定一刀切
            # axis = random.choice(['D', 'H', 'W'])
            axis = 'D'  # 强制 Z 轴切分，符合实际任务
            # side = random.choice(['low', 'high'])
            side = random.choice(['low'])

            if axis == 'D':
                cut = random.randint(int(D * 0.45), int(D * 0.55))
                if side == 'low':
                    mask[:, :cut, :, :] = 1.0
                else:
                    mask[:, cut:, :, :] = 1.0

            elif axis == 'H':
                cut = random.randint(int(H * 0.45), int(H * 0.55))
                if side == 'low':
                    mask[:, :, :cut, :] = 1.0
                else:
                    mask[:, :, cut:, :] = 1.0

            else:  # 'W'
                cut = random.randint(int(W * 0.45), int(W * 0.55))
                if side == 'low':
                    mask[:, :, :, :cut] = 1.0
                else:
                    mask[:, :, :, cut:] = 1.0

        # ========== B) 已知两半（两端）-> 推理中间 ==========
        elif mode == 'two_halves':
            # 两端已知，中间未知。middle_ratio 控制中间缺失宽度
            axis = random.choice(['D', 'H', 'W'])
            middle_ratio = random.uniform(0.25, 0.50)  # 中间缺 25%~50%
            if axis == 'D':
                mid = int(D * middle_ratio)
                start = (D - mid) // 2
                end = start + mid
                mask.fill_(1.0)
                mask[:, start:end, :, :] = 0.0

            elif axis == 'H':
                mid = int(H * middle_ratio)
                start = (H - mid) // 2
                end = start + mid
                mask.fill_(1.0)
                mask[:, :, start:end, :] = 0.0

            else:  # 'W'
                mid = int(W * middle_ratio)
                start = (W - mid) // 2
                end = start + mid
                mask.fill_(1.0)
                mask[:, :, :, start:end] = 0.0

        # ========== C) 已知三面（缺一个角）-> 推理角落缺失 ==========
        elif mode == 'corner_missing':
            # 整体已知，只挖掉一个角落立方体
            mask.fill_(1.0)
            corner_ratio = random.uniform(0.25, 0.55)  # 缺角大小 25%~55%
            cd = max(2, int(D * corner_ratio))
            ch = max(2, int(H * corner_ratio))
            cw = max(2, int(W * corner_ratio))

            # 8 个角随机一个
            d_side = random.choice(['low', 'high'])
            h_side = random.choice(['low', 'high'])
            w_side = random.choice(['low', 'high'])

            d_slice = slice(0, cd) if d_side == 'low' else slice(D - cd, D)
            h_slice = slice(0, ch) if h_side == 'low' else slice(H - ch, H)
            w_slice = slice(0, cw) if w_side == 'low' else slice(W - cw, W)

            mask[:, d_slice, h_slice, w_slice] = 0.0

        # ========== D) 随机挖空一块 -> 推理挖空部分 ==========
        else:  # 'random_box'
            mask.fill_(1.0)

            # 盒子尺寸：占边长 25%~70%（避免过小，也避免直接全挖没）
            hole_d = random.randint(max(2, int(D * 0.25)), max(3, int(D * 0.70)))
            hole_h = random.randint(max(2, int(H * 0.25)), max(3, int(H * 0.70)))
            hole_w = random.randint(max(2, int(W * 0.25)), max(3, int(W * 0.70)))

            hole_d = min(hole_d, D - 1)
            hole_h = min(hole_h, H - 1)
            hole_w = min(hole_w, W - 1)

            z0 = random.randint(0, D - hole_d)
            y0 = random.randint(0, H - hole_h)
            x0 = random.randint(0, W - hole_w)

            mask[:, z0:z0 + hole_d, y0:y0 + hole_h, x0:x0 + hole_w] = 0.0

        masked_latent = latent * mask  # (C,D,H,W) * (1,D,H,W) 广播
        return masked_latent, mask
 
    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        filename = os.path.basename(file_path)
        
        try:
            data_numpy = np.load(file_path)
        except:
            return self.__getitem__(random.randint(0, len(self.file_list)-1))
        
        latent = torch.from_numpy(data_numpy).float()
        
        latent = latent * CONFIG['scale_factor']
        latent = torch.clamp(latent, min=-CONFIG['safe_threshold'], max=CONFIG['safe_threshold'])

        if latent.dim() == 5: latent = latent.squeeze(0)
            
        if self.augment:
            if random.random() > 0.5: latent = torch.flip(latent, dims=[1])
            if random.random() > 0.5: latent = torch.flip(latent, dims=[2])
            if random.random() > 0.5: latent = torch.flip(latent, dims=[3])
        
        porosity = self.extract_porosity(filename)
        porosity = torch.tensor([porosity], dtype=torch.float32)
        
        condition_latent, mask = self.apply_mask(latent)
        
        return {
            "GT": latent,
            "Condition": condition_latent,
            "Mask": mask,
            "Porosity": porosity
        }
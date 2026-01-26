import os
import glob
import torch
import numpy as np
import random
from torch.utils.data import Dataset

class Core3DDataset(Dataset):
    def __init__(self, data_dir, global_min, global_max, patch_size=128, train=True):
        """
        Args:
            patch_size (int): 训练时的切块大小，建议 128
            train (bool): 训练模式下开启随机切块，验证模式下可能中心裁剪
        """
        self.files = sorted(glob.glob(os.path.join(data_dir, "*.npy")))
        self.global_min = float(global_min)
        self.global_max = float(global_max)
        self.patch_size = patch_size
        self.train = train
        
        self.scale = self.global_max - self.global_min
        print(f"Dataset init. Norm: [{self.global_min}, {self.global_max}]. Patch Size: {patch_size}^3")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        try:
            # 1. Load Data (256, 256, 256)
            # mmap_mode='r' 对于大文件切块读取非常快，不需要加载整个文件到内存
            data_numpy = np.load(file_path, mmap_mode='r') 
            
            # 2. Random Crop (关键步骤)
            # 原始尺寸
            D, H, W = data_numpy.shape
            target_s = self.patch_size
            
            if self.train:
                # 随机选择起始点
                z_start = random.randint(0, D - target_s)
                y_start = random.randint(0, H - target_s)
                x_start = random.randint(0, W - target_s)
            else:
                # 中心裁剪 (Center Crop)
                z_start = (D - target_s) // 2
                y_start = (H - target_s) // 2
                x_start = (W - target_s) // 2
            
            # 切片 (这一步因为用了 mmap，只会读取这一小块内存，速度极快)
            crop_data = data_numpy[z_start:z_start+target_s, 
                                   y_start:y_start+target_s, 
                                   x_start:x_start+target_s]
            
            # 转为 float32 并拷贝到内存 (断开 mmap 连接)
            crop_data = crop_data.astype(np.float32)

            # 3. Normalize
            data_norm = (crop_data - self.global_min) / self.scale
            data_norm = data_norm * 2.0 - 1.0
            
            # 4. To Tensor (1, 128, 128, 128)
            data_tensor = torch.from_numpy(data_norm).unsqueeze(0)
            
            return data_tensor
            
        except Exception as e:
            print(f"⚠️ Error loading {file_path}: {e}")
            return self.__getitem__(random.randint(0, len(self.files)-1))
import torch
import numpy as np
import glob
import os
from torch.utils.data import Dataset

class RawCoreDataset(Dataset):
    def __init__(self, data_root_list, crop_size=256):
        self.files = []
        for root in data_root_list:
            self.files.extend(sorted(glob.glob(os.path.join(root, "*.npy"))))
        
        self.crop_size = crop_size
        print(f"📦 Found {len(self.files)} raw samples.")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        try:
            # 1. Load Data (uint16)
            data = np.load(path) # Shape: (256, 256, 256)
            
            # 2. Normalize: [0, 38748] -> [-1, 1]
            # 你的数据最大值约 38748，为了鲁棒性，我们可以除以 40000 或者 65535
            # 但既然你做过 P99 归一化，建议直接按最大值归一化，或者使用你统计的 38748
            data = data.astype(np.float32) / 38748.0 
            data = data * 2.0 - 1.0
            
            # 3. Handle Channels
            # PyTorch 3D Conv expects (C, D, H, W)
            data = torch.from_numpy(data).unsqueeze(0) # (1, 256, 256, 256)
            
            return data
            
        except Exception as e:
            print(f"Error loading {path}: {e}")
            return torch.zeros((1, 256, 256, 256))
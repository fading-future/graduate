import os
import glob
import random
import numpy as np
import torch
from torch.utils.data import Dataset

class MaskedREVDataset(Dataset):
    def __init__(self, data_dir, volume_size=128, limit_size=None, augment=True):
        """
        Args:
            data_dir: 预处理后的 .npy 文件目录 (文件本身可以是 256x256x256)
            volume_size: 训练时实际喂给模型的尺寸 (建议 128)
        """
        self.file_list = sorted(glob.glob(os.path.join(data_dir, "*.npy")))

        # 2. 隨機挑選 n 個文件
        # 注意：如果文件總數小於 n，sample 會報錯，建議加個長度判斷
        sample_size = min(len(self.file_list), 2000)
        self.file_list = random.sample(self.file_list, sample_size)
        
        if limit_size is not None and len(self.file_list) > limit_size:
            print(f"Dataset limit applied: using {limit_size} out of {len(self.file_list)} images.")
            self.file_list = self.file_list[:limit_size]
            
        self.volume_size = volume_size
        self.augment = augment 
        
        print(f"Dataset loaded: {len(self.file_list)} files. Training Patch Size: {volume_size}^3")

    def __len__(self):
        return len(self.file_list)

    def random_crop(self, volume, target_size):
        """
        从大体积中随机切出一个 target_size 的块
        """
        D, H, W = volume.shape[-3:]
        
        # 如果原始数据比目标尺寸小，就报错或者Padding（这里假设原始数据够大）
        if D < target_size or H < target_size or W < target_size:
            raise ValueError(f"Data size ({D},{H},{W}) is smaller than target size ({target_size})")

        # 随机选择起始点
        z = random.randint(0, D - target_size)
        y = random.randint(0, H - target_size)
        x = random.randint(0, W - target_size)
        
        return volume[..., z:z+target_size, y:y+target_size, x:x+target_size]

    def random_transform(self, volume):
        # 随机翻转
        for dim in [1, 2, 3]:
            if random.random() > 0.5:
                volume = torch.flip(volume, dims=[dim])
        # 随机旋转
        k = random.randint(0, 3)
        if k > 0:
            plane = random.choice([(1, 2), (1, 3), (2, 3)])
            volume = torch.rot90(volume, k, dims=plane)
        return volume

    def apply_boundary_mask(self, volume):
        """
        生成 Mask 并构建条件输入
        """

        # 动态计算厚度和挖空大小 (根据 128 尺寸适配)
        # 边界厚度：10% - 20%
        D, H, W = volume.shape[-3:]
        mask = torch.zeros((1, D, H, W), dtype=torch.float32)
        
        # 这里的 thickness 根据当前的 crop size (128) 动态计算
        min_thick = max(8, int(D * 0.1)) 
        max_thick = max(16, int(D * 0.2))
        thickness = random.randint(min_thick, max_thick)
        
        mode = random.choice(['one_face', 'two_faces', 'corner', 'random_box'])
        
        if mode == 'one_face':
            face = random.choice(['top', 'bottom', 'left', 'right', 'front', 'back'])
            if face == 'top': mask[..., :thickness, :, :] = 1
            elif face == 'bottom': mask[..., -thickness:, :, :] = 1
            elif face == 'left': mask[..., :, :thickness, :] = 1
            elif face == 'right': mask[..., :, -thickness:, :] = 1
            elif face == 'front': mask[..., :, :, :thickness] = 1
            elif face == 'back': mask[..., :, :, -thickness:] = 1
        elif mode == 'two_faces':
            mask[..., :thickness, :, :] = 1 
            mask[..., :, :thickness, :] = 1 
        elif mode == 'corner':
            mask[..., :thickness, :, :] = 1
            mask[..., :, :thickness, :] = 1
            mask[..., :, :, :thickness] = 1
        elif mode == 'random_box':
            min_hole = int(D * 0.3)
            max_hole = int(D * 0.7)
            hole_size = random.randint(min_hole, max_hole)
            z_start = random.randint(0, D - hole_size)
            y_start = random.randint(0, H - hole_size)
            x_start = random.randint(0, W - hole_size)
            mask.fill_(1.0) 
            mask[..., z_start:z_start+hole_size, y_start:y_start+hole_size, x_start:x_start+hole_size] = 0 

        masked_volume = volume * mask 
        return masked_volume, mask
    
    def normalize_volume(self, volume):
        """
        将 16-bit 数据归一化到 [-1, 1] 区间，适合 DDPM/GAN
        param volume: 输入的 3D 体数据，numpy 数组格式
        return: 归一化后的 3D 体数据
        """

        volume = volume.astype(np.float32)
        volume = volume / 65535.0 # 归一化到 [0, 1]
        volume = (volume * 2.0) - 1.0 # 归一化到 [-1, 1]
        return volume

    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        
        # 1. 使用 mmap 打开，不加载到 RAM
        # 注意：mmap 模式下数据是只读的，不能直接修改，后续复制出来即可
        raw_data = np.load(file_path, mmap_mode='r') # shape: (256, 256, 256)
        
        # 2. 先计算切片的坐标 (直接复用 random_crop 的逻辑算出 x,y,z)
        # 这里需要把 random_crop 拆解一下，只返回坐标
        d, h, w = raw_data.shape
        target_size = self.volume_size
        
        z = random.randint(0, d - target_size)
        y = random.randint(0, h - target_size)
        x = random.randint(0, w - target_size)
        
        # 3. 只加载这一小块到内存 (128^3)
        crop_data = raw_data[z:z+target_size, y:y+target_size, x:x+target_size]
        
        # 4. 此时转为 float32 并拷贝到内存（解除了 mmap 锁定）
        crop_data = crop_data.astype(np.float32)
        
        # 5. 再进行归一化（只计算 128^3 的量，速度快 8 倍）
        crop_data = self.normalize_volume(crop_data) # 使用上面优化后的归一化
        
        # 转 Tensor
        gt_volume = torch.from_numpy(crop_data).unsqueeze(0).float()
        
        # --- 关键修改：先切块，把 256 变成 128 ---
        # 这样进入显存的数据只有原来的 1/8
        # gt_volume = self.random_crop(gt_volume, self.volume_size)
        
        # 数据增强
        if self.augment:
            gt_volume = self.random_transform(gt_volume)
        
        # 生成 Mask
        masked_volume, mask = self.apply_boundary_mask(gt_volume)
        
        return {
            "GT": gt_volume,
            "Condition": masked_volume,
            "Mask": mask
        }


# import os
# import glob
# import random
# import numpy as np
# import torch
# from torch.utils.data import Dataset

# import os
# import glob
# import random
# import numpy as np
# import torch
# from torch.utils.data import Dataset

# class MaskedREVDataset(Dataset):
#     def __init__(self, data_dir, volume_size=128, limit_size=None, augment=True):
#         """
#         Args:
#             data_dir: 预处理后的 .npy 文件目录
#             volume_size: REV 的边长 (128)
#             limit_size: 限制数据集大小 (调试用)
#             augment: 是否开启数据增强 (训练集True, 验证集False)
#         """
#         self.file_list = sorted(glob.glob(os.path.join(data_dir, "*.npy")))
        
#         # 数据截断 (调试用)
#         if limit_size is not None and len(self.file_list) > limit_size:
#             print(f"Dataset limit applied: using {limit_size} out of {len(self.file_list)} images.")
#             self.file_list = self.file_list[:limit_size]
            
#         self.volume_size = volume_size
#         self.augment = augment # 开关数据增强
        
#         print(f"Dataset loaded: Found {len(self.file_list)} REV blocks. Augmentation={'ON' if augment else 'OFF'}")

#     def __len__(self):
#         return len(self.file_list)

#     def random_transform(self, volume):
#         """
#         随机数据增强：翻转 + 旋转
#         volume: Tensor (1, D, H, W)
#         """
#         # 1. 随机翻转 (Flip)
#         # 针对 D, H, W 维度 (即 dim 1, 2, 3) 进行随机翻转
#         for dim in [1, 2, 3]:
#             if random.random() > 0.5:
#                 volume = torch.flip(volume, dims=[dim])
        
#         # 2. 随机旋转 (Rotate90)
#         # 随机选择旋转次数 (0, 1, 2, 3)
#         k = random.randint(0, 3)
#         if k > 0:
#             # 随机选择旋转平面: (1,2)XY面, (1,3)XZ面, (2,3)YZ面
#             plane = random.choice([(1, 2), (1, 3), (2, 3)])
#             volume = torch.rot90(volume, k, dims=plane)
            
#         return volume

#     def apply_boundary_mask(self, volume):
#         """
#         生成 Mask 并构建条件输入
#         """
#         D, H, W = volume.shape[-3:]
#         mask = torch.zeros((1, D, H, W), dtype=torch.float32)
        
#         # 动态计算厚度和挖空大小 (根据 128 尺寸适配)
#         # 边界厚度：10% - 20%
#         min_thick = max(8, int(D * 0.1)) 
#         max_thick = max(16, int(D * 0.2))
#         thickness = random.randint(min_thick, max_thick)
        
#         mode = random.choice(['one_face', 'two_faces', 'corner', 'random_box'])
        
#         if mode == 'one_face':
#             face = random.choice(['top', 'bottom', 'left', 'right', 'front', 'back'])
#             if face == 'top': mask[..., :thickness, :, :] = 1
#             elif face == 'bottom': mask[..., -thickness:, :, :] = 1
#             elif face == 'left': mask[..., :, :thickness, :] = 1
#             elif face == 'right': mask[..., :, -thickness:, :] = 1
#             elif face == 'front': mask[..., :, :, :thickness] = 1
#             elif face == 'back': mask[..., :, :, -thickness:] = 1
            
#         elif mode == 'two_faces':
#             mask[..., :thickness, :, :] = 1 # Top
#             mask[..., :, :thickness, :] = 1 # Left
            
#         elif mode == 'corner':
#             mask[..., :thickness, :, :] = 1
#             mask[..., :, :thickness, :] = 1
#             mask[..., :, :, :thickness] = 1
            
#         elif mode == 'random_box':
#             # 挖掉中间 30% - 70%
#             min_hole = int(D * 0.3)
#             max_hole = int(D * 0.7)
            
#             hole_size = random.randint(min_hole, max_hole)
            
#             z_start = random.randint(0, D - hole_size)
#             y_start = random.randint(0, H - hole_size)
#             x_start = random.randint(0, W - hole_size)
            
#             mask.fill_(1.0) # 先全已知
#             mask[..., z_start:z_start+hole_size, y_start:y_start+hole_size, x_start:x_start+hole_size] = 0 # 挖空

#         masked_volume = volume * mask 
#         return masked_volume, mask

#     def __getitem__(self, idx):
#         file_path = self.file_list[idx]
#         data_numpy = np.load(file_path)
        
#         # 转为 Tensor (1, 128, 128, 128)
#         gt_volume = torch.from_numpy(data_numpy).unsqueeze(0).float()
        
#         # --- 关键：先做增强，再做 Mask ---
#         # 这样模型每次见到的不仅是 Mask 不同，连石头本身的角度都不同
#         if self.augment:
#             gt_volume = self.random_transform(gt_volume)
        
#         # 动态生成 Mask
#         masked_volume, mask = self.apply_boundary_mask(gt_volume)
        
#         return {
#             "GT": gt_volume,
#             "Condition": masked_volume,
#             "Mask": mask
#         }

# class MaskedREVDataset(Dataset):
#     def __init__(self, data_dir, volume_size=64, limit_size=None):
#         """
#         data_dir: 预处理后的 .npy 文件目录路径
#         volume_size: REV 的边长
#         """
#         self.file_list = glob.glob(os.path.join(data_dir, "*.npy"))

#         if limit_size is not None and len(self.file_list) > limit_size:
#             self.file_list = self.file_list[:limit_size]

#         self.volume_size = volume_size
#         print(f"Dataset loaded: Found {len(self.file_list)} REV blocks.")

#     def __len__(self):
#         return len(self.file_list)

#     def apply_boundary_mask(self, volume):
#         """
#         核心函数：生成 Mask 并构建条件输入。
#         模拟边界补全任务。
        
#         volume: Tensor (1, D, H, W) 值在 [-1, 1]
#         返回: 
#             masked_volume: 被挖空的体积 (未知区域填 -1 或 0)
#             mask: 0/1 掩码 (1表示已知/保留，0表示未知/待生成)
#         """
#         D, H, W = volume.shape[-3:]
#         mask = torch.zeros((1, D, H, W), dtype=torch.float32)
        
#         # ====== 策略：随机选择已知面 (Outpainting 模拟) ======
#         # 1. 随机决定我们要保留多少个面 (1个面, 2个面-棱, 3个面-角)
#         # 为了让模型足够鲁棒，我们随机混合这些情况
        
#         mode = random.choice(['one_face', 'two_faces', 'corner', 'random_box'])
        
#         # 设定已知边界的厚度 (例如已知 10-30 个像素层)
#         thickness = random.randint(10, 40)
        
#         if mode == 'one_face':
#             # 随机选6个面中的一个
#             face = random.choice(['top', 'bottom', 'left', 'right', 'front', 'back'])
#             if face == 'top': mask[..., :thickness, :, :] = 1
#             elif face == 'bottom': mask[..., -thickness:, :, :] = 1
#             elif face == 'left': mask[..., :, :thickness, :] = 1
#             elif face == 'right': mask[..., :, -thickness:, :] = 1
#             elif face == 'front': mask[..., :, :, :thickness] = 1
#             elif face == 'back': mask[..., :, :, -thickness:] = 1
            
#         elif mode == 'two_faces':
#             # 模拟棱边连接 (例如已知 Top 和 Left)
#             mask[..., :thickness, :, :] = 1 # Top
#             mask[..., :, :thickness, :] = 1 # Left (简单叠加)
            
#         elif mode == 'corner':
#             # 模拟角点连接 (例如已知 Top, Left, Front)
#             mask[..., :thickness, :, :] = 1
#             mask[..., :, :thickness, :] = 1
#             mask[..., :, :, :thickness] = 1
            
#         elif mode == 'random_box':
#             # === 修正 2: 动态计算挖空大小 (约为边长的 30% 到 70%) ===
#             min_hole = int(self.volume_size * 0.3)
#             max_hole = int(self.volume_size * 0.7)
            
#             # 确保不会因为尺寸太小导致 min > max
#             if min_hole >= max_hole:
#                 min_hole = max_hole - 1
            
#             hole_size = random.randint(min_hole, max_hole)
            
#             # 现在的 D 是 64，hole_size 约 20-40，相减一定是正数，不会报错了
#             z_start = random.randint(0, D - hole_size)
#             y_start = random.randint(0, H - hole_size)
#             x_start = random.randint(0, W - hole_size)
            
#             mask.fill_(1.0)
#             mask[..., z_start:z_start+hole_size, y_start:y_start+hole_size, x_start:x_start+hole_size] = 0

#         masked_volume = volume * mask 
#         return masked_volume, mask

#     def __getitem__(self, index):
#         # 1. 加载 Ground Truth (C, D, H, W)
#         file_path = self.file_list[index]
#         data_numpy = np.load(file_path)
#         gt_volume = torch.from_numpy(data_numpy).unsqueeze(0).float() # 转为 Tensor 并增加 Channel 维度
        
#         # 2. 动态生成 Mask
#         masked_volume, mask = self.apply_boundary_mask(gt_volume)
        
#         # 3. 返回训练对
#         # input: 模型需要看到的条件
#         # target: 模型需要预测的真值 (或者用于计算 Loss)
#         return {
#             "GT": gt_volume,           # 完整的真值 [-1, 1]
#             "Condition": masked_volume, # 被挖空的图 (未知区域为0)
#             "Mask": mask               # 0/1 指示图
#         }
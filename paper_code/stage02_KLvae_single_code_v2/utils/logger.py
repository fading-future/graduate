import torch
import os
import csv
from torchvision.utils import save_image

class CSVLogger:
    def __init__(self, log_dir, filename="train_log.csv"):
        self.log_dir = log_dir
        self.file_path = os.path.join(log_dir, filename)
        self.columns = ["Epoch", "Step", "Loss_Total", "Loss_Recon", "Loss_KL", "Loss_G_Adv", "Loss_D"]
        
        # 只有文件不存在时才写入表头
        if not os.path.exists(self.file_path):
            with open(self.file_path, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(self.columns)

    def log(self, data_dict):
        row = [data_dict.get(col, "") for col in self.columns]
        with open(self.file_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)

    # === 新增：读取 CSV 最后一行获取 Step ===
    def get_last_step(self):
        if not os.path.exists(self.file_path):
            return 0
        
        try:
            with open(self.file_path, 'r') as f:
                # 使用 seek 高效读取最后部分，避免加载整个大文件
                try:
                    f.seek(-2, os.SEEK_END)
                    while f.read(1) != "\n":
                        f.seek(-2, os.SEEK_CUR)
                except OSError:
                    # 文件太小（只有一行或是空的），直接回到开头
                    f.seek(0)
                
                last_line = f.readline().strip()
                
                # 如果是表头或空行，返回 0
                if not last_line or "Epoch" in last_line:
                    return 0
                
                # 解析最后一行
                # 假设 Step 在第二个位置 (index 1)
                parts = last_line.split(',')
                if len(parts) >= 2:
                    return int(parts[1])
                return 0
        except Exception as e:
            print(f"[Logger Warning] Failed to read last step from CSV: {e}")
            return 0

    def save_comparison_grid(self, reals, recons, epoch, step):
        with torch.no_grad():
            depth_idx = reals.shape[2] // 2
            slice_real = reals[:, :, depth_idx, :, :]
            slice_recon = recons[:, :, depth_idx, :, :]
            img_grid = torch.cat([slice_real, slice_recon], dim=0) 
            img_grid = (img_grid + 1.0) / 2.0
            
            save_name = os.path.join(self.log_dir, f"vis_epoch_{epoch}_step_{step}.png")
            save_image(img_grid, save_name, nrow=reals.shape[0])



# 没有断点续训功能，新增可视化功能
# import torch
# import os
# import csv
# from torchvision.utils import save_image

# class CSVLogger:
#     def __init__(self, log_dir, filename="train_log.csv"):
#         self.log_dir = log_dir # 记住目录，方便存图
#         self.file_path = os.path.join(log_dir, filename)
#         self.columns = ["Epoch", "Step", "Loss_Total", "Loss_Recon", "Loss_KL", "Loss_G_Adv", "Loss_D"]
        
#         if not os.path.exists(self.file_path):
#             with open(self.file_path, mode='w', newline='') as f:
#                 writer = csv.writer(f)
#                 writer.writerow(self.columns)

#     def log(self, data_dict):
#         row = [data_dict.get(col, "") for col in self.columns]
#         with open(self.file_path, mode='a', newline='') as f:
#             writer = csv.writer(f)
#             writer.writerow(row)

#     # === 新增可视化函数 ===
#     def save_comparison_grid(self, reals, recons, epoch, step):
#         """
#         reals, recons: [B, C, D, H, W] 范围在 [-1, 1]
#         我们取 Depth 维度的中间切片
#         """
#         with torch.no_grad():
#             # 1. 取中间切片 -> [B, C, H, W]
#             depth_idx = reals.shape[2] // 2
#             slice_real = reals[:, :, depth_idx, :, :]
#             slice_recon = recons[:, :, depth_idx, :, :]

#             # 2. 拼接：上面是原图，下面是重建图
#             # cat dim=0 会把 batch 里的图竖着拼或者横着拼，这里我们构造成 Comparison
#             # 建议：取 Batch 中的第一张图来对比即可，或者拼成网格
#             # 为了简单直观，我们把 Batch 里所有的图拼成一排 Real，下面一排 Recon
#             img_grid = torch.cat([slice_real, slice_recon], dim=0) 

#             # 3. 反归一化：从 [-1, 1] -> [0, 1]
#             img_grid = (img_grid + 1.0) / 2.0
            
#             # 4. 保存
#             save_name = os.path.join(self.log_dir, f"vis_epoch_{epoch}_step_{step}.png")
#             # nrow=B 表示每一行显示 B 张图（即上面一行原图，下面一行重建）
#             save_image(img_grid, save_name, nrow=reals.shape[0])


# 没有可视化的功能
# import csv
# import os

# class CSVLogger:
#     def __init__(self, log_dir, filename="train_log.csv"):
#         self.file_path = os.path.join(log_dir, filename)
#         self.columns = ["Epoch", "Step", "Loss_Total", "Loss_Recon", "Loss_KL", "Loss_G_Adv", "Loss_D"]
        
#         # 如果文件不存在，写入表头
#         if not os.path.exists(self.file_path):
#             with open(self.file_path, mode='w', newline='') as f:
#                 writer = csv.writer(f)
#                 writer.writerow(self.columns)

#     def log(self, data_dict):
#         # 确保数据顺序与表头一致
#         row = [data_dict.get(col, "") for col in self.columns]
#         with open(self.file_path, mode='a', newline='') as f:
#             writer = csv.writer(f)
#             writer.writerow(row)
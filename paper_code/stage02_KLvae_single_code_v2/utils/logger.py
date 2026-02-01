import torch
import os
import csv
from torchvision.utils import save_image

class CSVLogger:
    def __init__(self, log_dir, filename="train_log.csv"):
        self.log_dir = log_dir
        self.file_path = os.path.join(log_dir, filename)
        
        # === 修改处：扩充表头以匹配 train.py 中的 log_dict ===
        self.columns = [
            "Time", "Epoch", "Step", 
            "Loss_Total", "Loss_Recon", "Loss_KL", 
            "KL_Weight", "Loss_KL_weighted", 
            "KL_avg_per_latent", "KL_contrib_ratio", 
            "Loss_G_Adv", "Loss_D", 
            "GradNorm_VAE", "GradNorm_Disc", 
            "LR_VAE", "LR_Disc", "KL_trend_flag"
        ]
        
        # 只有文件不存在时才写入表头
        if not os.path.exists(self.file_path):
            with open(self.file_path, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(self.columns)

    def log(self, data_dict):
        # 根据 columns 的顺序从 data_dict 提取数据，没有的填空字符串
        row = [data_dict.get(col, "") for col in self.columns]
        with open(self.file_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)

    def get_last_step(self):
        if not os.path.exists(self.file_path):
            return 0
        try:
            with open(self.file_path, 'r') as f:
                try:
                    f.seek(-2, os.SEEK_END)
                    while f.read(1) != "\n":
                        f.seek(-2, os.SEEK_CUR)
                except OSError:
                    f.seek(0)
                last_line = f.readline().strip()
                if not last_line or "Epoch" in last_line:
                    return 0
                
                # Step 对应 columns 的第 3 列 (index 2)
                parts = last_line.split(',')
                # 简单判断：如果分割后长度不够，或者表头不匹配，可能需要根据实际情况调整 index
                # 只要 csv 格式对齐，取 index 2 就是 Step
                if len(parts) > 2:
                    return int(parts[2]) 
                return 0
        except Exception as e:
            print(f"[Logger Warning] Failed to read last step: {e}")
            return 0
        
    def save_comparison_grid(self, reals, recons, epoch, step, max_images=8):
        with torch.no_grad():
            # === 修复处：应该用 min 而不是 max，否则 batch_size 小于 max_images 会越界 ===
            num_to_show = min(max_images, reals.shape[0])
            
            reals = reals[:num_to_show]
            recons = recons[:num_to_show]
            
            # 取 Depth 维度的中间切片 -> [N, 1, H, W]
            depth_idx = reals.shape[2] // 2
            slice_real = reals[:, :, depth_idx, :, :]
            slice_recon = recons[:, :, depth_idx, :, :]
            
            # 拼接：第一行原图，第二行重建
            img_grid = torch.cat([slice_real, slice_recon], dim=0) 
            
            # 反归一化 [-1, 1] -> [0, 1]
            img_grid = (img_grid + 1.0) / 2.0
            
            save_name = os.path.join(self.log_dir, f"vis_epoch_{epoch}_step_{step}.png")
            # nrow=num_to_show 意味着一行显示 N 张（即第一行全是原图，第二行全是重建）
            save_image(img_grid, save_name, nrow=num_to_show)

    # def save_comparison_grid(self, reals, recons, epoch, step):
    #     with torch.no_grad():
    #         depth_idx = reals.shape[2] // 2
    #         slice_real = reals[:, :, depth_idx, :, :]
    #         slice_recon = recons[:, :, depth_idx, :, :]
    #         img_grid = torch.cat([slice_real, slice_recon], dim=0) 
    #         img_grid = (img_grid + 1.0) / 2.0
            
    #         save_name = os.path.join(self.log_dir, f"vis_epoch_{epoch}_step_{step}.png")
    #         save_image(img_grid, save_name, nrow=reals.shape[0])



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
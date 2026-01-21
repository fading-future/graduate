import os
import random
import numpy as np
import matplotlib.pyplot as plt
import glob

"""
检查npy数据（REV 立方体）的三个切面，确认数据的有效性和质量。
"""

CONFIG = {
    # 可以根据需要调整这里的参数
    'num_samples': 15,  # 随机抽检的样本数量
    'src_root': r"D:\多尺度岩心数据集\Cleaned_NPY_Dataset_24",  # npy文件所在目录
}


def inspect_npy_files(data_dir, num_samples=3):
    """
    随机读取几个 .npy 文件并可视化其三个切面
    """
    files = glob.glob(os.path.join(data_dir, "*.npy"))
    
    if len(files) == 0:
        print(f"错误：在 {data_dir} 中没有找到 .npy 文件。请先确保提取脚本运行成功。")
        return

    print(f"检查目录: {data_dir}")
    print(f"共找到 {len(files)} 个数据块。正在随机抽检 {num_samples} 个...\n")
    
    # 随机选择文件
    samples = random.sample(files, min(len(files), num_samples))
    
    for filepath in samples:
        filename = os.path.basename(filepath)
        print(f"--- 正在检查: {filename} ---")
        
        # 1. 加载数据
        try:
            volume = np.load(filepath)
        except Exception as e:
            print(f"无法加载 {filename}: {e}")
            continue
            
        # 2. 打印统计信息
        print(f"  尺寸: {volume.shape}") # 应该是 (128, 128, 128)
        print(f"  数据类型: {volume.dtype}")
        print(f"  数值范围: min={volume.min()}, max={volume.max()}, mean={volume.mean():.2f}")
        
        # 3. 简单的质量判断
        if volume.min() == volume.max():
            print("  [警告] 图像是纯色的（全黑或全白），无效数据！")
        elif volume.mean() < 500: # 假设值很低
            print("  [警告] 图像整体过暗，可能截取到了背景区域！")
        else:
            print("  [通过] 数据统计特征正常。")

        # 4. 可视化三视图 (XY, XZ, YZ)
        # 取立方体的中心切片
        mid_z, mid_y, mid_x = volume.shape[0]//2, volume.shape[1]//2, volume.shape[2]//2
        
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        
        # XY Plane (Top view)
        axes[0].imshow(volume[mid_z, :, :], cmap='gray')
        axes[0].set_title(f"XY Plane (Z={mid_z})")
        axes[0].axis('off')
        
        # XZ Plane (Side view 1) - 展示纵向连续性
        axes[1].imshow(volume[:, mid_y, :], cmap='gray')
        axes[1].set_title(f"XZ Plane (Y={mid_y})")
        axes[1].axis('off')
        
        # YZ Plane (Side view 2)
        axes[2].imshow(volume[:, :, mid_x], cmap='gray')
        axes[2].set_title(f"YZ Plane (X={mid_x})")
        axes[2].axis('off')
        
        plt.suptitle(f"Sample: {filename}", fontsize=14)
        plt.tight_layout()
        plt.show() # 这会弹出一个窗口显示图片


if __name__ == "__main__":
    inspect_npy_files(CONFIG['src_root'], num_samples=CONFIG['num_samples'])
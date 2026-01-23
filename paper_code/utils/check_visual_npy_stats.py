#%%
import os
import numpy as np
import matplotlib.pyplot as plt

# 设置中文字体
# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP', 'DejaVu Sans'] # 优先使用中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

"""
检查npy 文件的元数据和基本统计信息
绘制npy 文件（REV）体素值的分布直方图
"""

#%%
# 文件路径
file_path = r"/chendou_space/data/aligned_Training_Data_Interactive/6-6-20 全部_z5184_y448_x659.npy"

try:
    # 加载数据
    data = np.load(file_path)

    # 修复 SyntaxError：先提取文件名
    file_name = os.path.basename(file_path)
    print(f"--- 文件信息: {file_name} ---")
    
    print(f"数据形状 (Shape): {data.shape}")
    print(f"数据维度 (Dimensions): {data.ndim}")
    print(f"数据类型 (Dtype): {data.dtype}")
    print(f"元素总数 (Size): {data.size}")
    
    # 统计信息
    print(f"最大值 (Max): {np.max(data)}")
    print(f"最小值 (Min): {np.min(data)}")
    print(f"平均值 (Mean): {np.mean(data):.4f}")

    # 3. 绘制直方图
    plt.figure(figsize=(12, 6))
    
    # 针对 uint16 的优化设置：
    # 如果数据是 uint16 且数值跨度很大，建议增加 bins 数量
    # 或者使用 bins='auto' 自动选择最优间隔
    bins_count = 1024 if data.dtype == np.uint16 else 256
    
    print(f"正在绘制直方图 (Bins={bins_count})...")
    plt.hist(data.ravel(), bins=bins_count, color='forestgreen', alpha=0.75)
    
    plt.title(f"Voxel Distribution - {file_name} ({data.dtype})")
    plt.xlabel("Voxel Value")
    plt.ylabel("Frequency")
    
    # 如果 uint16 数据集中在某个小区间，可以手动设置横坐标范围
    # plt.xlim(np.min(data), np.max(data)) 
    
    plt.grid(axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.show()

except FileNotFoundError:
    print("错误：未找到文件，请检查路径。")
except Exception as e:
    print(f"发生错误: {e}")
# %%

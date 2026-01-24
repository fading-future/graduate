import os
import glob
import numpy as np
from tqdm import tqdm

def calculate_stats(data_dir):
    # 1. 获取所有 .npy 文件
    files = sorted(glob.glob(os.path.join(data_dir, "*.npy")))
    if len(files) == 0:
        raise ValueError(f"❌ No .npy files found in {data_dir}")
    
    print(f"🔎 Found {len(files)} files. Starting statistics calculation...")
    
    # 初始化全局极值
    # uint16 范围是 0-65535，初始值设为相反的极端即可
    global_min = float('inf')
    global_max = float('-inf')
    
    # 用于计算 Mean/Std 的累加器 (可选，若只需要 Min/Max 可忽略这部分)
    # 考虑到数据量太大，计算精确的 Global Mean/Std 需要两轮遍历或 Welford 算法
    # 这里我们优先保证 Min/Max 的准确性
    
    for file_path in tqdm(files, desc="Scanning dataset"):
        try:
            # 加载数据 (mmap_mode='r' 可以避免一次性把大文件读入内存，但对 npy 读取速度稍有影响，视内存情况而定)
            # 鉴于 A100 机器内存通常很大，直接 load 应该没问题
            data = np.load(file_path)
            
            # 获取当前文件的极值
            current_min = data.min()
            current_max = data.max()
            
            # 更新全局极值
            if current_min < global_min:
                global_min = current_min
            
            if current_max > global_max:
                global_max = current_max
                
        except Exception as e:
            print(f"⚠️ Error reading {file_path}: {e}")

    return global_min, global_max

if __name__ == "__main__":
    # 修改这里为你的数据路径
    DATA_DIR = "./data/train" 
    
    try:
        g_min, g_max = calculate_stats(DATA_DIR)
        
        print("\n" + "="*40)
        print("✅ Global Statistics Calculated")
        print("="*40)
        print(f"📂 Data Directory: {DATA_DIR}")
        print(f"⬇️  Global Min: {g_min}")
        print(f"⬆️  Global Max: {g_max}")
        print("="*40)
        print("\n📝 Please update your config.py with these values!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
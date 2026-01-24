#%%
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

# ================= 魔法参数配置 =================
CONFIG = {
    # 要检查的数据文件夹
    "data_root": r"D:\多尺度岩心数据集\cleaned_npy_dataset", 
    # 随机抽样文件数
    "sample_file_count": 40,  
    # 每个文件随机抽样像素点数 (太多会慢且卡内存)
    "pixels_per_file": 256*256*256, 
    # 绘图区间 (uint16)
    "plot_range": (0, 65535),
    # 直方图精细度
    "bins": 200,
    # 保存路径
    "save_path": "./histogram_check.png"
}
# ===============================================

#%%
def load_and_sample(path):
    """读取并随机采样像素"""
    try:
        data = np.load(path)
        flat = data.ravel()
        # 随机采样，避免数据量过大
        if len(flat) > CONFIG["pixels_per_file"]:
            sampled = np.random.choice(flat, CONFIG["pixels_per_file"], replace=False)
            return sampled
        return flat
    except:
        return None

def plot_distribution_curves():
    # 1. 获取文件列表
    all_npy = glob.glob(os.path.join(CONFIG["data_root"], "**/*.npy"), recursive=True)
    if not all_npy:
        print("❌ 未找到NPY文件")
        return

    # 2. 随机抽取文件
    import random
    selected_files = random.sample(all_npy, min(len(all_npy), CONFIG["sample_file_count"]))
    
    print(f"📉 正在加载 {len(selected_files)} 个文件进行分布分析...")

    # 3. 并行读取
    pixel_batches = []
    with ThreadPoolExecutor(max_workers=16) as executor:
        results = list(tqdm(executor.map(load_and_sample, selected_files), total=len(selected_files)))
        pixel_batches = [r for r in results if r is not None]

    # 4. 绘图
    plt.figure(figsize=(12, 8), dpi=150)
    plt.style.use('seaborn-v0_8-whitegrid') # 或者 'ggplot'

    print("🎨 正在绘制曲线...")
    
    # 绘制每一条单文件的曲线 (细线，半透明)
    for pixels in pixel_batches:
        # 使用直方图模拟曲线 (step模式)
        counts, bin_edges = np.histogram(pixels, bins=CONFIG["bins"], range=CONFIG["plot_range"], density=True)
        centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        plt.plot(centers, counts, color='blue', alpha=0.1, linewidth=1)

    # 绘制所有数据的平均曲线 (粗红线)
    all_pixels = np.concatenate(pixel_batches)
    counts_all, bin_edges_all = np.histogram(all_pixels, bins=CONFIG["bins"], range=CONFIG["plot_range"], density=True)
    centers_all = (bin_edges_all[:-1] + bin_edges_all[1:]) / 2
    plt.plot(centers_all, counts_all, color='red', alpha=1.0, linewidth=2, label='Mean Distribution')

    # 装饰
    plt.title(f"Voxel Intensity Distribution ({len(selected_files)} samples)", fontsize=14)
    plt.xlabel("Pixel Intensity (uint16)", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.xlim(CONFIG["plot_range"])
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 标注峰值
    peak_x = centers_all[np.argmax(counts_all)]
    plt.axvline(peak_x, color='green', linestyle='--', alpha=0.8)
    plt.text(peak_x, np.max(counts_all), f' Peak: {int(peak_x)}', color='green', fontweight='bold')

    plt.savefig(CONFIG["save_path"])
    print(f"✅ 分布图已保存: {CONFIG['save_path']}")
    # plt.show() # 如果在远程服务器上，注释掉这行

if __name__ == "__main__":
    plot_distribution_curves()
#%%
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import warnings

# ================= 配置魔法参数 =================
CONFIG = {
    # 你的数据路径 (建议先用少量数据测试)
    "data_root": r"E:\aligned_Training_Data", 
    
    # 采样设置
    "sample_ratio": 0.5,        # 采样比例 (0.1 表示随机抽 10% 的文件画图，1.0 表示全画)
    "max_files": 500,           # 最大文件数限制 (防止几万张图把内存撑爆)
    
    # 直方图参数
    "bins": 1000,               # 将 65536 压缩到 1000 个区间进行绘图 (太密了看不清)
    "x_range": (0, 65535),      # uint16 范围
    
    # 绘图风格
    "plot_alpha": 0.05,         # 线条透明度 (越低越适合看密集重叠)
    "plot_color": "tab:blue",   # 曲线颜色
    "figure_size": (12, 8),
    "dpi": 150,
    
    # 并行核心数
    "num_workers": 24
}
# ===============================================

#%%
def calculate_hist_worker(file_path):
    """
    子进程：计算单个文件的直方图数据
    同时统计 0 和 65535 的精确数量
    """
    try:
        data = np.load(file_path)
        flat = data.ravel()
        
        # 1. 计算用于绘图的直方图 (降低分辨率以减少数据量)
        hist, bin_edges = np.histogram(flat, bins=CONFIG['bins'], range=CONFIG['x_range'])
        
        # 2. 精确统计两端截断情况
        count_0 = np.sum(flat == 0)
        count_max = np.sum(flat == 65535)
        total_pixels = flat.size
        
        return {
            "hist": hist,
            "edges": bin_edges,
            "zeros": count_0,
            "maxs": count_max,
            "total": total_pixels,
            "status": "ok"
        }
    except Exception as e:
        return {"status": "error"}

def main():
    # 1. 搜集文件
    print(f"🔍 正在扫描文件: {CONFIG['data_root']}")
    all_files = list(Path(CONFIG['data_root']).rglob("*.npy"))
    
    if not all_files:
        print("❌ 未找到 .npy 文件")
        return

    # 随机采样
    import random
    num_to_sample = min(int(len(all_files) * CONFIG['sample_ratio']), CONFIG['max_files'])
    if num_to_sample < 1: num_to_sample = 1
    selected_files = random.sample(all_files, num_to_sample)
    
    print(f"📊 将分析 {len(selected_files)} / {len(all_files)} 个文件...")

    # 2. 并行计算
    results = []
    with ProcessPoolExecutor(max_workers=CONFIG['num_workers']) as executor:
        for res in tqdm(executor.map(calculate_hist_worker, selected_files), 
                        total=len(selected_files), desc="Computing Histograms"):
            if res['status'] == 'ok':
                results.append(res)
    
    if not results:
        print("❌ 没有有效数据")
        return

    # 3. 开始绘图
    print("🎨 正在绘制曲线...")
    plt.figure(figsize=CONFIG['figure_size'], dpi=CONFIG['dpi'])
    
    # 用于计算平均分布
    sum_hist = np.zeros_like(results[0]['hist'], dtype=np.float64)
    
    # 统计截断率
    zero_ratios = []
    max_ratios = []

    # 绘制每一条线
    x_axis = (results[0]['edges'][:-1] + results[0]['edges'][1:]) / 2
    
    for res in results:
        # 归一化频率 (PDF)
        pdf = res['hist'] / res['total']
        sum_hist += pdf
        
        # 绘制单条曲线 (透明叠加)
        plt.plot(x_axis, pdf, color=CONFIG['plot_color'], alpha=CONFIG['plot_alpha'], linewidth=1)
        
        # 记录截断数据
        zero_ratios.append(res['zeros'] / res['total'])
        max_ratios.append(res['maxs'] / res['total'])

    # 绘制平均曲线 (高亮)
    avg_hist = sum_hist / len(results)
    plt.plot(x_axis, avg_hist, color='red', linewidth=2, linestyle='--', label='Average Distribution')

    # 4. 设置图表细节
    plt.title(f"Voxel Intensity Distribution (N={len(results)})", fontsize=14)
    plt.xlabel("Pixel Value (uint16)", fontsize=12)
    plt.ylabel("Normalized Frequency", fontsize=12)
    plt.xlim(0, 65535)
    plt.grid(True, alpha=0.3)
    
    # 5. 在图中添加统计文本 (关于截断)
    avg_zero = np.mean(zero_ratios) * 100
    max_zero = np.max(zero_ratios) * 100
    avg_max = np.mean(max_ratios) * 100
    max_max = np.max(max_ratios) * 100
    
    stats_text = (
        f"Clipping Statistics:\n"
        f"-------------------\n"
        f"Value = 0 (Black Clipping):\n"
        f"  Avg: {avg_zero:.2f}%\n"
        f"  Max: {max_zero:.2f}%\n\n"
        f"Value = 65535 (White Clipping):\n"
        f"  Avg: {avg_max:.2f}%\n"
        f"  Max: {max_max:.2f}%"
    )
    
    # 将统计数据放在图的右上角
    plt.text(0.95, 0.95, stats_text, transform=plt.gca().transAxes, 
             fontsize=10, verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.legend()
    
    # 保存
    save_path = "dataset_histogram_analysis.png"
    plt.savefig(save_path)
    print(f"\n✅ 分析完成！图片已保存至: {save_path}")
    print(f"   (请检查图中右侧文本框，如果 Clipping Max > 1% 说明可能有问题)")

if __name__ == "__main__":
    main()
# %%

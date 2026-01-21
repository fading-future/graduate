import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

# 导入自定义模块
from utils.get_root_path import get_project_root

"""
计算REV 数据的孔隙率的脚本 - 可视化部分
用于生成论文中的图表，展示数据处理前后的效果对比
"""

ROOT_DIR = get_project_root()

CONFIG = {
    "data_dir": r"D:\多尺度岩心数据集\Cleaned_NPY_Dataset_18", # 数据路径
    "offset": 20,  # 与处理脚本中使用的偏移量保持一致
    "manual_samples": [
        # Cleaned_NPY_Dataset_18 数据集中手动指定的样本路径
        # r"D:\多尺度岩心数据集\Cleaned_NPY_Dataset_18\6-6-18_z448_y1077_x820.npy",
        # r"D:\多尺度岩心数据集\Cleaned_NPY_Dataset_18\6-6-18_z448_y1013_x948.npy",
        # r"D:\多尺度岩心数据集\Cleaned_NPY_Dataset_18\6-6-18_z448_y821_x1076.npy",
        # r"D:\多尺度岩心数据集\Cleaned_NPY_Dataset_18\6-6-18_z384_y492_x626.npy",

        # Cleaned_NPY_Dataset_20 数据集中手动指定的样本路径
        r"D:\多尺度岩心数据集\Cleaned_NPY_Dataset_20\6-6-20 全部_z4608_y1006_x1072.npy",
        r"D:\多尺度岩心数据集\Cleaned_NPY_Dataset_20\6-6-20 全部_z4096_y1037_x1047.npy",
        r"D:\多尺度岩心数据集\Cleaned_NPY_Dataset_20\6-6-20 全部_z2944_y980_x841.npy",
        r"D:\多尺度岩心数据集\Cleaned_NPY_Dataset_20\6-6-20 全部_z960_y1041_x1022.npy",

        # Cleaned_NPY_Dataset_21 数据集中手动指定的样本路径
        # Cleaned_NPY_Dataset_22 全部 数据集中手动指定的样本路径
    ], # 手动指定样本路径列表，留空则随机选取
    "sample_count": 4,  # 随机选取样本可视化的数量
    "target_peak_sample_count": 180,  # 用于自动计算目标峰值的样本数量

    # 图像保存路径
    "img_data_dir": str(ROOT_DIR / "src" / "visualization" / "img_data"),
    "figure_1_path": "Step2_Figure_1_Distribution_Wide.png",
    "figure_2_path": "Step2_Figure_2_Processing_Flow_Reordered.png",
}

def normalize_fixed(data):
    return (data.astype(np.float32) / 65535.0 * 255.0).astype(np.uint8)

def get_peak(data_u8):
    hist, _ = np.histogram(data_u8.ravel(), bins=256, range=(0, 255))
    hist[:40] = 0
    return np.argmax(hist)

def auto_find_peak(files):
    sample_files = np.random.choice(files, min(len(files), CONFIG["target_peak_sample_count"]), replace=False)
    peaks = [get_peak(normalize_fixed(np.load(f))) for f in sample_files]
    return int(np.mean(peaks))

def plot_figure_1_wide(files, target_peak, threshold):
    """
    修改版 Figure 1: 更宽的画布，更舒展的直方图
    """
    print("正在绘制 Figure 1 (Wide Distribution)...")
    # 【修改点1】调整画布大小，变宽 (16, 5)
    plt.figure(figsize=(16, 5), dpi=300)
    
    gs = gridspec.GridSpec(1, 2, wspace=0.15)
    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])
    
    plot_files = np.random.choice(files, min(len(files), 100), replace=False)
    
    for f in plot_files:
        d = normalize_fixed(np.load(f))
        h, _ = np.histogram(d.ravel(), bins=256, range=(0, 255))
        peak = get_peak(d)
        shift = target_peak - peak
        
        # 左图：原始
        ax1.plot(h, color='gray', alpha=0.2, linewidth=0.8)
        
        # 右图：对齐
        x_shifted = np.arange(256) + shift
        valid = (x_shifted >= 0) & (x_shifted <= 255)
        ax2.plot(x_shifted[valid], h[valid], color='#2E86C1', alpha=0.25, linewidth=0.8)

    # 装饰调整
    for ax in [ax1, ax2]:
        ax.set_xlim(0, 255)
        # 【修改点2】稍微压低一点Y轴上限，让峰看起来不那么尖
        ax.set_ylim(0, ax.get_ylim()[1]*0.9) 

    ax1.set_title("(a) Raw Intensity Distribution (Inconsistent)", loc='left', fontweight='bold')
    ax1.set_xlabel("Gray Value")
    ax1.set_ylabel("Voxel Count")
    
    ax2.set_title(f"(b) Aligned Distribution (Target Peak={target_peak})", loc='left', fontweight='bold')
    ax2.set_xlabel("Gray Value")
    ax2.set_yticks([])
    
    ax2.axvline(target_peak, color='k', linestyle='--', linewidth=1.2, label='Target Alignment')
    ax2.axvline(threshold, color='#E74C3C', linestyle='-', linewidth=2, label='Global Threshold')
    
    # 调整标注位置，避免拥挤
    ax2.text(target_peak+5, ax2.get_ylim()[1]*0.7, "Matrix Aligned", color='k', fontsize=10)
    ax2.text(threshold-5, ax2.get_ylim()[1]*0.7, "Pore Region", color='#E74C3C', ha='right', fontsize=10)
    ax2.legend(loc='upper right', frameon=True)
    
    plt.tight_layout()
    plt.savefig(str(Path(CONFIG["img_data_dir"]) / CONFIG["figure_1_path"]), bbox_inches='tight')
    plt.close()

def plot_figure_2_reordered(files, target_peak, threshold):
    """
    修改版 Figure 2: 
    1. 列顺序调整: Raw -> Aligned -> Segmentation -> Histogram
    2. 直方图标注美化: 避免重叠和换行
    """
    print("正在绘制 Figure 2 (Reordered Flow)...")
    
    if CONFIG["manual_samples"]:
        selected_files = [Path(f) for f in CONFIG["manual_samples"]]
    else:
        selected_files = np.random.choice(files, CONFIG["sample_count"], replace=False)
        
    rows = len(selected_files)
    cols = 4
    
    # 【修改点3】调整画布宽度比例，最后一列直方图稍微窄一点
    fig = plt.figure(figsize=(16, 4 * rows), dpi=300)
    gs = gridspec.GridSpec(rows, cols, wspace=0.15, hspace=0.2, width_ratios=[1, 1, 1, 0.8])
    
    for i, f_path in enumerate(selected_files):
        # 数据准备
        data_u8 = normalize_fixed(np.load(f_path))
        peak = get_peak(data_u8)
        shift = target_peak - peak
        
        data_aligned = np.clip(data_u8.astype(np.int16) + shift, 0, 255).astype(np.uint8)
        mask = data_aligned < threshold
        porosity = np.sum(mask) / mask.size
        
        idx = data_u8.shape[0] // 2
        sl_raw = data_u8[idx]
        sl_align = data_aligned[idx]
        sl_mask = mask[idx]
        
        # === Col 1: Raw Image ===
        ax1 = plt.subplot(gs[i, 0])
        ax1.imshow(sl_raw, cmap='gray', vmin=0, vmax=255)
        ax1.set_xticks([]); ax1.set_yticks([])
        ax1.set_ylabel(f"Sample {i+1}", fontweight='bold', fontsize=12)
        if i == 0: ax1.set_title("Step 1: Original REV", fontweight='bold')
        ax1.text(5, 25, f"Peak={peak}", color='yellow', fontweight='bold', fontsize=9, 
                 bbox=dict(facecolor='black', edgecolor='none', alpha=0.6))
        
        # === Col 2: Aligned Image (原 Col 3) ===
        ax2 = plt.subplot(gs[i, 1])
        ax2.imshow(sl_align, cmap='gray', vmin=0, vmax=255)
        ax2.set_xticks([]); ax2.set_yticks([])
        if i == 0: ax2.set_title("Step 2: Aligned REV", fontweight='bold')
        ax2.text(5, 25, "Corrected", color='#00FF00', fontweight='bold', fontsize=9,
                 bbox=dict(facecolor='black', edgecolor='none', alpha=0.6))

        # === Col 3: Segmentation (原 Col 4) ===
        ax3 = plt.subplot(gs[i, 2])
        ax3.imshow(~sl_mask, cmap='gray', vmin=0, vmax=1) 
        ax3.set_xticks([]); ax3.set_yticks([])
        if i == 0: ax3.set_title("Step 3: Segmentation", fontweight='bold')
        ax3.text(5, 25, f"Phi={porosity:.2%}", color='red', fontweight='bold', fontsize=9,
                 bbox=dict(facecolor='white', edgecolor='none', alpha=0.8))

        # === Col 4: Histogram Schematic (原 Col 2, 美化版) ===
        ax4 = plt.subplot(gs[i, 3])
        h, _ = np.histogram(data_u8.ravel(), bins=256, range=(0, 255))
        
        # 画曲线
        ax4.plot(h, color='gray', alpha=0.6, lw=1)
        ax4.fill_between(range(256), h, color='gray', alpha=0.1)
        
        # 画峰值线
        ax4.axvline(peak, color='gray', linestyle=':', label='Original Peak')
        ax4.axvline(target_peak, color='#2E86C1', linestyle='--', label='Target Peak')
        
        # 【修改点4】美化标注：使用高位的横向箭头和文字，避免重叠
        text_y = np.max(h) * 0.85 # 文字高度
        arrow_y = text_y * 0.9    # 箭头高度
        
        # 绘制横向箭头示意移动方向
        if abs(shift) > 0:
            ax4.annotate('', xy=(target_peak, arrow_y), xytext=(peak, arrow_y),
                         arrowprops=dict(arrowstyle='->', color='#2E86C1', lw=1.5))
            # 在箭头上方标注 shift 值，保证不换行
            sign = "+" if shift > 0 else ""
            ax4.text((peak + target_peak) / 2, text_y, f"Shift: {sign}{shift}", 
                     ha='center', color='#2E86C1', fontweight='bold', fontsize=9,
                     bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=1))

        # 装饰
        ax4.set_xlim(0, 255)
        ax4.set_yticks([])
        ax4.spines['left'].set_visible(False)
        ax4.spines['top'].set_visible(False)
        ax4.spines['right'].set_visible(False)
        
        if i == 0: ax4.set_title("Calibration Principle", fontweight='bold')
        else: ax4.set_xlabel("Gray Value")
        
        # 只在第一个子图显示图例，避免冗余
        if i == 0:
            ax4.legend(loc='upper right', fontsize=8, frameon=False)

    plt.tight_layout()
    plt.savefig(str(Path(CONFIG["img_data_dir"]) / CONFIG["figure_2_path"]), bbox_inches='tight')
    plt.close()

def main():
    all_files = list(Path(CONFIG["data_dir"]).rglob("*.npy"))
    if not all_files:
        print("未找到文件！")
        return

    target_peak = auto_find_peak(all_files)
    threshold = target_peak - CONFIG["offset"]
    print(f"绘图基准: Target Peak={target_peak}, Threshold={threshold}")

    # 绘制新版图 1
    plot_figure_1_wide(all_files, target_peak, threshold)
    
    # 绘制新版图 2
    plot_figure_2_reordered(all_files, target_peak, threshold)
    
    print("\n图片绘制完成！检查生成的 PNG 文件。")

if __name__ == "__main__":
    main()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from pathlib import Path

# ================= 配置区域 =================
# CSV 路径
CSV_PATH = r"E:\chendou\paper_data\processing_report.csv"
# 原始数据根目录
SRC_ROOT = r"D:\多尺度岩心数据集\Raw_Data"
# 你的 Target Peak (根据处理报告填写)
TARGET_PEAK = 152
# 你的 Offset (阈值 = Target - Offset)
OFFSET = 30
THRESHOLD = TARGET_PEAK - OFFSET
# ===========================================

def normalize_fixed(data):
    return (data.astype(np.float32) / 65535.0 * 255.0).astype(np.uint8)

def plot_thesis_verification(csv_path, src_root, target_peak, threshold):
    print("正在生成毕业论文验证图...")
    
    # 1. 数据准备
    df = pd.read_csv(csv_path)
    df = df[df['status'] == 'ok'].copy()
    df_sorted = df.sort_values(by='porosity')
    
    # 选出最极端的样本用于画直方图组 (各选10个，体现统计规律)
    dense_group = df_sorted.head(20)
    porous_group = df_sorted.tail(20)
    
    # 选出最极端的1个样本用于画切片展示
    sample_dense = df_sorted.iloc[20]
    sample_porous = df_sorted.iloc[-10]
    
    # 2. 设置画布
    fig = plt.figure(figsize=(12, 10), dpi=300)
    gs = gridspec.GridSpec(3, 3, height_ratios=[1.2, 1, 1], hspace=0.4, wspace=0.1)
    
    # ================= Panel A: 直方图物理规律验证 =================
    ax_hist = plt.subplot(gs[0, :])
    
    # 绘制 Target Line
    ax_hist.axvline(target_peak, color='black', linestyle='--', linewidth=2, label=f'Target Alignment ({target_peak})')
    
    # 绘制致密组 (Dense) - 蓝色
    print("绘制直方图规律...")
    for _, row in dense_group.iterrows():
        try:
            full_path = os.path.join(src_root, row['rel_path'])
            d = normalize_fixed(np.load(full_path))
            h, _ = np.histogram(d.ravel(), bins=256, range=(0, 255))
            ax_hist.plot(h, color='#2E86C1', alpha=0.4, linewidth=1)
        except: pass
        
    # 绘制疏松组 (Porous) - 红色
    for _, row in porous_group.iterrows():
        try:
            full_path = os.path.join(src_root, row['rel_path'])
            d = normalize_fixed(np.load(full_path))
            h, _ = np.histogram(d.ravel(), bins=256, range=(0, 255))
            ax_hist.plot(h, color='#E74C3C', alpha=0.4, linewidth=1)
        except: pass

    # 伪造图例
    ax_hist.plot([], [], color='#2E86C1', linewidth=2, label='Low Porosity (Dense) - Raw Data')
    ax_hist.plot([], [], color='#E74C3C', linewidth=2, label='High Porosity (Porous) - Raw Data')
    
    ax_hist.set_title('(a) Physical Validation: Raw Intensity Distributions of Extreme Samples', fontsize=12, fontweight='bold', loc='left')
    ax_hist.set_xlabel('Raw Gray Value (Before Alignment)')
    ax_hist.set_ylabel('Frequency')
    ax_hist.set_xlim(0, 255)
    ax_hist.legend(loc='upper right')
    
    # 添加文字解释箭头
    ax_hist.annotate('Dense samples are darker\n(Need positive shift)', 
                     xy=(target_peak-20, ax_hist.get_ylim()[1]*0.5), 
                     xytext=(target_peak-80, ax_hist.get_ylim()[1]*0.5),
                     arrowprops=dict(arrowstyle='->', color='#2E86C1'), color='#2E86C1', fontweight='bold')
                     
    ax_hist.annotate('Porous samples are lighter\n(Need negative shift)', 
                     xy=(target_peak+20, ax_hist.get_ylim()[1]*0.5), 
                     xytext=(target_peak+60, ax_hist.get_ylim()[1]*0.5),
                     arrowprops=dict(arrowstyle='->', color='#E74C3C'), color='#E74C3C', fontweight='bold')

    # ================= Panel B: 视觉全流程验证 =================
    
    def plot_sample_row(ax_list, row_data, title_prefix):
        # 读取与处理
        full_path = os.path.join(src_root, row_data['rel_path'])
        raw_u8 = normalize_fixed(np.load(full_path))
        
        # 复现处理过程
        shift = row_data['shift']
        aligned_u8 = np.clip(raw_u8.astype(np.int16) + shift, 0, 255).astype(np.uint8)
        mask = aligned_u8 < threshold
        
        # 取中间切片
        idx = raw_u8.shape[0] // 2
        sl_raw = raw_u8[idx]
        sl_align = aligned_u8[idx]
        sl_mask = mask[idx]
        
        # 1. Raw
        ax_list[0].imshow(sl_raw, cmap='gray', vmin=0, vmax=255)
        ax_list[0].set_ylabel(title_prefix, fontweight='bold', fontsize=11)
        ax_list[0].set_xticks([]); ax_list[0].set_yticks([])
        ax_list[0].text(5, 20, f"Raw", color='yellow', fontweight='bold', bbox=dict(facecolor='black', alpha=0.5))
        
        # 2. Aligned
        ax_list[1].imshow(sl_align, cmap='gray', vmin=0, vmax=255)
        ax_list[1].set_xticks([]); ax_list[1].set_yticks([])
        sign = "+" if shift > 0 else ""
        ax_list[1].text(5, 20, f"Aligned ({sign}{int(shift)})", color='cyan', fontweight='bold', bbox=dict(facecolor='black', alpha=0.5))
        
        # 3. Segmented (注意 vmin=0, vmax=1 防止全黑)
        # 显示 ~mask 以符合 白色=骨架，黑色=孔隙
        ax_list[2].imshow(~sl_mask, cmap='gray', vmin=0, vmax=1)
        ax_list[2].set_xticks([]); ax_list[2].set_yticks([])
        ax_list[2].text(5, 20, f"Phi={row_data['porosity']:.2%}", color='red', fontweight='bold', bbox=dict(facecolor='white', alpha=0.7))

    # --- Row 1: 最致密样本 ---
    ax_r1 = [plt.subplot(gs[1, 0]), plt.subplot(gs[1, 1]), plt.subplot(gs[1, 2])]
    plot_sample_row(ax_r1, sample_dense, "Dense Sample\n(Low Porosity)")
    
    # --- Row 2: 最疏松样本 ---
    ax_r2 = [plt.subplot(gs[2, 0]), plt.subplot(gs[2, 1]), plt.subplot(gs[2, 2])]
    plot_sample_row(ax_r2, sample_porous, "Porous Sample\n(High Porosity)")
    
    # 列标题
    ax_r1[0].set_title("(b) Step 1: Raw Image", fontweight='bold')
    ax_r1[1].set_title("Step 2: After Alignment", fontweight='bold')
    ax_r1[2].set_title("Step 3: Segmentation", fontweight='bold')
    
    plt.tight_layout()
    plt.savefig("Thesis_Verification_Chart.png", bbox_inches='tight')
    print("图表已生成: Thesis_Verification_Chart.png")
    plt.show()

if __name__ == "__main__":
    plot_thesis_verification(CSV_PATH, SRC_ROOT, TARGET_PEAK, THRESHOLD)
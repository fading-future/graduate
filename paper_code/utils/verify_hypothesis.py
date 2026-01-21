import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path

# ================= 配置 =================
# 必须指向你刚才生成的那个包含所有样本信息的 CSV
CSV_PATH = r"E:\chendou\paper_data\processing_report.csv"
SRC_ROOT = r"D:\多尺度岩心数据集\Raw_Data" # 原始数据路径，用于读取原始图像
TARGET_PEAK = 140 # 填入你处理报告中最终确定的那个 Target Peak (看图1或图2的标题)
# =======================================

def normalize_fixed(data):
    return (data.astype(np.float32) / 65535.0 * 255.0).astype(np.uint8)

def verify_hypothesis(csv_path, src_root, target_peak):
    # 1. 读取报告
    df = pd.read_csv(csv_path)
    df = df[df['status'] == 'ok'].copy()
    
    # 2. 挑选两组极端样本
    # 按孔隙度排序
    df_sorted = df.sort_values(by='porosity')
    
    # 选出孔隙度最低的 5 个 (Dense Group)
    low_porosity_samples = df_sorted.head(5)
    # 选出孔隙度最高的 5 个 (Porous Group)
    high_porosity_samples = df_sorted.tail(5)
    
    print(f"低孔隙度组均值: {low_porosity_samples['porosity'].mean():.2%}")
    print(f"高孔隙度组均值: {high_porosity_samples['porosity'].mean():.2%}")

    # 3. 绘图：直方图对比
    plt.figure(figsize=(12, 8), dpi=300)
    
    # --- 画布局：上图画直方图堆叠，下图画对应切片 ---
    ax_hist = plt.subplot2grid((3, 5), (0, 0), colspan=5)
    
    # 绘制 Target Line
    ax_hist.axvline(target_peak, color='black', linestyle='--', linewidth=2, label=f'Target Peak ({target_peak})')
    
    # A. 绘制低孔隙度组 (Dense) - 预期偏暗(偏左) -> 需要正Shift
    print("正在绘制低孔隙度组...")
    for idx, row in low_porosity_samples.iterrows():
        # 构建完整路径 (注意：这里需要根据你的 rel_path 拼凑)
        # 如果 rel_path 已经是相对路径，直接拼
        full_path = os.path.join(src_root, row['rel_path'])
        
        try:
            data = np.load(full_path)
            d_u8 = normalize_fixed(data)
            h, _ = np.histogram(d_u8.ravel(), bins=256, range=(0, 255))
            
            # 画线：用蓝色表示致密
            ax_hist.plot(h, color='blue', alpha=0.6, linewidth=1.5)
        except Exception as e:
            print(f"无法读取 {row['file']}: {e}")

    # B. 绘制高孔隙度组 (Porous) - 预期偏亮(偏右) -> 需要负Shift
    print("正在绘制高孔隙度组...")
    for idx, row in high_porosity_samples.iterrows():
        full_path = os.path.join(src_root, row['rel_path'])
        try:
            data = np.load(full_path)
            d_u8 = normalize_fixed(data)
            h, _ = np.histogram(d_u8.ravel(), bins=256, range=(0, 255))
            
            # 画线：用红色表示疏松
            ax_hist.plot(h, color='red', alpha=0.6, linewidth=1.5)
        except:
            pass

    # 伪造图例句柄
    ax_hist.plot([], [], color='blue', label='Low Porosity (Dense) - Raw Hist')
    ax_hist.plot([], [], color='red', label='High Porosity (Porous) - Raw Hist')
    
    ax_hist.set_title(f"Visual Verification: Do Porosity Extremes Have Distinct Raw Distributions?", fontsize=14, fontweight='bold')
    ax_hist.set_xlabel("Raw Gray Value (Before Alignment)")
    ax_hist.set_ylabel("Frequency")
    ax_hist.legend()
    ax_hist.set_xlim(0, 255)
    
    # 添加文字说明
    ax_hist.text(0.02, 0.8, "If hypothesis is true:\nBlue lines should be LEFT of Target\nRed lines should be RIGHT of Target", 
                 transform=ax_hist.transAxes, bbox=dict(facecolor='white', alpha=0.8))

    # --- 下半部分：展示具体的 Shift 值 ---
    # 为了直观，我们列出这几个样本的 Shift 值
    # 绘制低孔隙度样本的 Shift
    shifts_low = low_porosity_samples['shift'].values
    shifts_high = high_porosity_samples['shift'].values
    
    txt_low = f"Low Porosity Group Shifts:\n{shifts_low}\n(Expect Positive +)"
    txt_high = f"High Porosity Group Shifts:\n{shifts_high}\n(Expect Negative -)"
    
    plt.figtext(0.15, 0.6, txt_low, fontsize=12, color='blue', fontweight='bold', ha='left')
    plt.figtext(0.65, 0.6, txt_high, fontsize=12, color='red', fontweight='bold', ha='left')
    
    # --- 画几个切片看看 (选最极端的各1个) ---
    # 最致密
    ax_dense = plt.subplot2grid((3, 5), (1, 1), rowspan=2, colspan=1)
    f_dense = os.path.join(src_root, low_porosity_samples.iloc[0]['rel_path'])
    d_dense = normalize_fixed(np.load(f_dense))
    ax_dense.imshow(d_dense[d_dense.shape[0]//2], cmap='gray', vmin=0, vmax=255)
    ax_dense.set_title(f"Densest Sample\nShift={low_porosity_samples.iloc[0]['shift']}", color='blue')
    ax_dense.axis('off')
    
    # 最疏松
    ax_porous = plt.subplot2grid((3, 5), (1, 3), rowspan=2, colspan=1)
    f_porous = os.path.join(src_root, high_porosity_samples.iloc[0]['rel_path'])
    d_porous = normalize_fixed(np.load(f_porous))
    ax_porous.imshow(d_porous[d_porous.shape[0]//2], cmap='gray', vmin=0, vmax=255)
    ax_porous.set_title(f"Most Porous Sample\nShift={high_porosity_samples.iloc[0]['shift']}", color='red')
    ax_porous.axis('off')

    plt.tight_layout()
    plt.savefig("Verification_Hypothesis.png")
    print("验证图已生成: Verification_Hypothesis.png")
    plt.show()

if __name__ == "__main__":
    verify_hypothesis(CSV_PATH, SRC_ROOT, TARGET_PEAK)
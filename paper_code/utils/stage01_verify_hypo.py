#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from pathlib import Path

# ================= 配置区域 =================
# CSV 路径 (请确保该文件存在)
CSV_PATH = "/chendou_space/data/aligned_Training_Data_Interactive/processing_report.csv"
# 原始 NPY 数据所在的文件夹路径
SRC_ROOT = "/chendou_space/data/cleaned_npy_dataset" 

# 你的 Target Peak (根据上一轮代码输出填写)
TARGET_PEAK = 38748
# 你的 Offset 
OFFSET = 8200
THRESHOLD = TARGET_PEAK - OFFSET

# 绘图配置
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans'] # 论文常用字体
plt.rcParams['axes.unicode_minus'] = False
# ===========================================

#%%
def get_peak_uint16(data):
    """辅助函数：计算单张图的峰值"""
    hist, _ = np.histogram(data.ravel(), bins=256, range=(0, 65535))
    hist[:200] = 0 # 忽略背景噪点
    return np.argmax(hist) * (65535/256) # 粗略估算

def plot_thesis_verification(csv_path, src_root, target_peak, threshold):
    print(f"正在生成毕业论文验证图... (Target={target_peak}, Th={threshold})")
    
    # 1. 数据加载与清洗
    if not os.path.exists(csv_path):
        print(f"错误: 找不到CSV文件 {csv_path}")
        return

    df = pd.read_csv(csv_path)
    # 过滤掉报错的样本
    df = df[df['status'] == 'ok'].copy()
    
    if len(df) == 0:
        print("CSV中没有有效数据！")
        return

    # 按孔隙度排序
    df_sorted = df.sort_values(by='porosity')
    
    # 选出最极端的样本组 (用于画直方图趋势)
    dense_group = df_sorted.head(10) # 孔隙度最小的10个 (致密)
    porous_group = df_sorted.tail(10) # 孔隙度最大的10个 (疏松)
    
    # 选出用于可视化的代表性样本 (避免选第一个防止是坏数据，选第5个和倒数第5个)
    # 也就是“典型致密”和“典型疏松”
    try:
        sample_dense = df_sorted.iloc[15] 
        sample_porous = df_sorted.iloc[-15]
    except:
        sample_dense = df_sorted.iloc[0]
        sample_porous = df_sorted.iloc[-1]
    
    # 2. 设置画布
    fig = plt.figure(figsize=(14, 10), dpi=300)
    gs = gridspec.GridSpec(3, 3, height_ratios=[1.2, 1, 1], hspace=0.4, wspace=0.1)
    
    # ================= Panel A: 直方图物理规律验证 =================
    ax_hist = plt.subplot(gs[0, :])
    
    # 绘制 Target Line
    ax_hist.axvline(target_peak, color='black', linestyle='--', linewidth=2, label=f'Target Alignment ({target_peak})')
    
    print("绘制直方图规律...")
    # 绘制致密组 (Dense) - 蓝色
    # 注意：这里我们使用 range=(0, 65535) 适配 uint16
    for _, row in dense_group.iterrows():
        try:
            # 路径拼接逻辑：如果 CSV 里 rel_path 已经是文件名，直接拼接
            full_path = os.path.join(src_root, os.path.basename(row['rel_path']))
            if not os.path.exists(full_path): continue
            
            d = np.load(full_path)
            # 使用 fewer bins (200) 使曲线平滑
            h, edges = np.histogram(d.ravel(), bins=200, range=(0, 65535))
            centers = (edges[:-1] + edges[1:]) / 2
            ax_hist.plot(centers, h, color='#2E86C1', alpha=0.3, linewidth=1)
        except Exception as e: pass
        
    # 绘制疏松组 (Porous) - 红色
    for _, row in porous_group.iterrows():
        try:
            full_path = os.path.join(src_root, os.path.basename(row['rel_path']))
            if not os.path.exists(full_path): continue
            
            d = np.load(full_path)
            h, edges = np.histogram(d.ravel(), bins=200, range=(0, 65535))
            centers = (edges[:-1] + edges[1:]) / 2
            ax_hist.plot(centers, h, color='#E74C3C', alpha=0.3, linewidth=1)
        except: pass

    # 伪造图例 (为了Legend好看)
    ax_hist.plot([], [], color='#2E86C1', linewidth=2, label='Low Porosity Group (Original Darker)')
    ax_hist.plot([], [], color='#E74C3C', linewidth=2, label='High Porosity Group (Original Brighter)')
    
    ax_hist.set_title('(a) Physical Validation: Intensity Distribution Before Alignment', fontsize=14, fontweight='bold', loc='left')
    ax_hist.set_xlabel('Raw Gray Value (uint16)', fontsize=12)
    ax_hist.set_ylabel('Pixel Frequency', fontsize=12)
    ax_hist.set_xlim(0, 65535) # 关键修改：适配 uint16
    ax_hist.legend(loc='upper right', frameon=True)
    
    # 添加文字解释箭头 (坐标也要适配 65535)
    # 致密样本通常偏左(暗)，需要向右移
    ax_hist.annotate('Dense/Darker Samples\nNeed Positive Shift (+)', 
                     xy=(target_peak - 5000, ax_hist.get_ylim()[1]*0.6), 
                     xytext=(target_peak - 15000, ax_hist.get_ylim()[1]*0.6),
                     arrowprops=dict(arrowstyle='->', color='#2E86C1', lw=2), 
                     color='#2E86C1', fontweight='bold', ha='right')
    
    # 疏松样本通常偏右(亮)，需要向左移
    ax_hist.annotate('Porous/Brighter Samples\nNeed Negative Shift (-)', 
                     xy=(target_peak + 5000, ax_hist.get_ylim()[1]*0.6), 
                     xytext=(target_peak + 15000, ax_hist.get_ylim()[1]*0.6),
                     arrowprops=dict(arrowstyle='->', color='#E74C3C', lw=2), 
                     color='#E74C3C', fontweight='bold', ha='left')

    # ================= Panel B: 视觉全流程验证 =================
    
    def plot_sample_row(ax_list, row_data, title_prefix):
        # 读取
        full_path = os.path.join(src_root, os.path.basename(row_data['rel_path']))
        if not os.path.exists(full_path):
            print(f"无法找到文件用于绘图: {full_path}")
            return
            
        raw = np.load(full_path) # uint16
        
        # 复现处理过程
        shift = int(row_data['shift'])
        # 模拟对齐，防止溢出
        aligned = np.clip(raw.astype(np.int32) + shift, 0, 65535).astype(np.uint16)
        
        # 分割
        mask = aligned < threshold
        
        # 取中间切片
        idx = raw.shape[0] // 2
        sl_raw = raw[idx]
        sl_align = aligned[idx]
        sl_mask = mask[idx]
        
        # 1. Raw
        ax_list[0].imshow(sl_raw, cmap='gray', vmin=0, vmax=65535) # 适配 uint16
        ax_list[0].set_ylabel(title_prefix, fontweight='bold', fontsize=12)
        ax_list[0].set_xticks([]); ax_list[0].set_yticks([])
        # 在图中添加文字标签
        ax_list[0].text(10, 30, f"Raw Peak: {int(target_peak - shift)}", color='yellow', fontsize=10, fontweight='bold')
        
        # 2. Aligned
        ax_list[1].imshow(sl_align, cmap='gray', vmin=0, vmax=65535) # 适配 uint16
        ax_list[1].set_xticks([]); ax_list[1].set_yticks([])
        sign = "+" if shift >= 0 else ""
        ax_list[1].text(10, 30, f"Shift: {sign}{shift}", color='cyan', fontsize=10, fontweight='bold')
        
        # 3. Segmented 
        # 使用 ~sl_mask (取反): 这样 骨架(False->True->1)是白, 孔隙(True->False->0)是黑
        ax_list[2].imshow(~sl_mask, cmap='gray', vmin=0, vmax=1)
        ax_list[2].set_xticks([]); ax_list[2].set_yticks([])
        ax_list[2].text(10, 30, f"Phi: {row_data['porosity']:.2%}", color='red', fontsize=10, fontweight='bold', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

    print("绘制切片对比图...")
    # --- Row 1: 典型致密样本 ---
    ax_r1 = [plt.subplot(gs[1, 0]), plt.subplot(gs[1, 1]), plt.subplot(gs[1, 2])]
    plot_sample_row(ax_r1, sample_dense, "Low Porosity\n(Dense)")
    
    # --- Row 2: 典型疏松样本 ---
    ax_r2 = [plt.subplot(gs[2, 0]), plt.subplot(gs[2, 1]), plt.subplot(gs[2, 2])]
    plot_sample_row(ax_r2, sample_porous, "High Porosity\n(Porous)")
    
    # 列标题
    ax_r1[0].set_title("(b) Original Image", fontweight='bold', fontsize=12)
    ax_r1[1].set_title("Aligned to Target", fontweight='bold', fontsize=12)
    ax_r1[2].set_title("Segmentation Result", fontweight='bold', fontsize=12)
    
    plt.tight_layout()
    save_path = "Thesis_Verification_Chart.png"
    plt.savefig(save_path, bbox_inches='tight')
    print(f"✅ 图表已生成: {save_path}")

if __name__ == "__main__":
    plot_thesis_verification(CSV_PATH, SRC_ROOT, TARGET_PEAK, THRESHOLD)
# %%

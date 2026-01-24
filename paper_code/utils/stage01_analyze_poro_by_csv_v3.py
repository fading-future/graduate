import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
import re

from utils.get_root_path import get_project_root

DIR_ROOT = get_project_root()

# ================= 配置区域 =================
# 输入：新的 CSV 报告路径
CSV_PATH = r"E:\aligned_Training_Data\processing_report.csv"
# 输出：分析图表保存路径
OUTPUT_DIR = DIR_ROOT / "utils" / "output_data" / "thesis_visualization"
# 图像风格配置
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
# ===========================================

def extract_group_name(filename):
    """
    从文件名中提取分组名称。
    逻辑：提取 '_z' 之前的所有字符。
    例如: '6-6-20 全部_z3840_y595_x528.npy' -> '6-6-20 全部'
    """
    match = re.match(r"^(.*?)_z\d+", filename)
    if match:
        return match.group(1)
    return "Unknown"

def analyze_and_plot(csv_path, output_dir):
    # 1. 准备目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # 2. 加载数据
    print(f"正在加载数据: {csv_path} ...")
    if not os.path.exists(csv_path):
        print(f"错误: 文件不存在 {csv_path}")
        return

    df = pd.read_csv(csv_path)
    
    # 3. 数据清洗与特征工程
    total_count = len(df)
    # 过滤掉处理失败的样本
    if 'status' in df.columns:
        df = df[df['status'] == 'ok'].copy()
    valid_count = len(df)
    print(f"总样本数: {total_count}, 有效样本数: {valid_count}")
    
    # 转换数值
    df['porosity_pct'] = df['porosity'] * 100  # 转换为百分比
    df['clip_ratio_pct'] = df['clip_ratio'] * 100 # 转换为百分比
    
    # 提取分组信息 (适配新文件名结构)
    df['group'] = df['file'].apply(extract_group_name)
    
    # 4. 打印统计报告
    print("-" * 50)
    print("【数据统计摘要】")
    print(f"孔隙度 (Porosity): {df['porosity_pct'].mean():.2f}% ± {df['porosity_pct'].std():.2f}% (Range: {df['porosity_pct'].min():.2f}% - {df['porosity_pct'].max():.2f}%)")
    print(f"缩放因子 (Scale Factor): {df['scale_factor'].mean():.4f} ± {df['scale_factor'].std():.4f}")
    print(f"原始峰值 (Original Peak): {df['orig_peak'].mean():.0f} (Range: {df['orig_peak'].min()} - {df['orig_peak'].max()})")
    print(f"高光截断率 (Clip Ratio): 平均 {df['clip_ratio_pct'].mean():.4f}% (Max: {df['clip_ratio_pct'].max():.4f}%)")
    print("-" * 50)

    # ================= 绘图部分 =================
    
    # --- 图1: 孔隙度总体分布 (Porosity Distribution) ---
    plt.figure(figsize=(10, 6), dpi=300)
    sns.histplot(data=df, x='porosity_pct', kde=True, bins=60, 
                 color='#4A90E2', edgecolor='black', alpha=0.7)
    
    mean_val = df['porosity_pct'].mean()
    median_val = df['porosity_pct'].median()
    plt.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2f}%')
    plt.axvline(median_val, color='green', linestyle='-', linewidth=2, label=f'Median: {median_val:.2f}%')
    
    plt.title('Distribution of Calculated Porosity (All Samples)', fontsize=14, fontweight='bold')
    plt.xlabel('Porosity (%)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'Fig1_Porosity_Distribution.png'))
    print("已生成: Fig1_Porosity_Distribution.png")

    # --- 图2: 原始峰值 vs 缩放因子 (Normalization Logic) ---
    # 这张图展示了你的对齐算法是如何工作的：峰值越低，缩放因子越大，呈现完美的反比关系
    fig, ax1 = plt.subplots(figsize=(10, 6), dpi=300)

    sns.histplot(data=df, x='orig_peak', bins=50, color='gray', alpha=0.5, ax=ax1, label='Original Peak Dist.')
    ax1.set_xlabel('Original Gray Peak Value (uint16)', fontsize=12)
    ax1.set_ylabel('Sample Count', fontsize=12)

    ax2 = ax1.twinx()
    sns.scatterplot(data=df, x='orig_peak', y='scale_factor', color='#E74C3C', s=10, alpha=0.5, ax=ax2, label='Scale Factor')
    ax2.set_ylabel('Applied Scale Factor', fontsize=12, color='#E74C3C')
    ax2.tick_params(axis='y', labelcolor='#E74C3C')
    
    plt.title('Data Normalization: Original Peak Variance & Scale Factor', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'Fig2_Normalization_Logic.png'))
    print("已生成: Fig2_Normalization_Logic.png")

    # --- 图3: 质量控制 - 高光截断率 (Quality Control) ---
    # 这是证明你使用了“软压缩/动态范围保留”算法有效的关键证据
    plt.figure(figsize=(10, 6), dpi=300)
    # 使用 log scale 因为截断率通常非常低，接近0
    sns.histplot(data=df, x='clip_ratio_pct', bins=50, color='#2ECC71', element="step", fill=True)
    plt.yscale('log') 
    
    plt.title('Quality Control: Pixel Clipping Ratio (Log Scale)', fontsize=14, fontweight='bold')
    plt.xlabel('Clipped Pixels Ratio (%) (> 65535)', fontsize=12)
    plt.ylabel('Count (Log Scale)', fontsize=12)
    plt.axvline(1.0, color='red', linestyle='--', label='Warning Threshold (1%)')
    
    # 标注绝大多数数据是安全的
    safe_count = len(df[df['clip_ratio_pct'] < 1.0])
    safe_pct = (safe_count / total_count) * 100
    plt.text(0.5, 0.8, f'{safe_pct:.1f}% samples have < 1% clipping', 
             transform=plt.gca().transAxes, fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
    
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'Fig3_QC_Clipping_Ratio.png'))
    print("已生成: Fig3_QC_Clipping_Ratio.png")

    # --- 图4: 分组孔隙度对比 (Group Comparison) ---
    # 检查提取的组名是否过多
    unique_groups = df['group'].nunique()
    if unique_groups > 20:
        print(f"注意: 分组数量过多 ({unique_groups})，仅展示样本量前15名的分组。")
        top_groups = df['group'].value_counts().nlargest(15).index
        df_plot = df[df['group'].isin(top_groups)]
    else:
        df_plot = df

    plt.figure(figsize=(12, 6), dpi=300)
    sns.boxplot(data=df_plot, x='group', y='porosity_pct', hue='group', palette="viridis", legend=False)
    plt.title('Porosity Heterogeneity Across Different Samples', fontsize=14, fontweight='bold')
    plt.xlabel('Sample ID', fontsize=12)
    plt.ylabel('Porosity (%)', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'Fig4_Porosity_by_Group.png'))
    print("已生成: Fig4_Porosity_by_Group.png")

    # --- 图5: 相关性检查 (Scale Factor vs Porosity) ---
    # 验证对齐操作（缩放）是否人为改变了孔隙度。理想情况是相关性极低。
    plt.figure(figsize=(8, 8), dpi=300)
    sns.scatterplot(data=df, x='scale_factor', y='porosity_pct', alpha=0.3, color='purple', s=15)
    sns.regplot(data=df, x='scale_factor', y='porosity_pct', scatter=False, color='black', line_kws={'linestyle':'--'})
    
    corr = df['scale_factor'].corr(df['porosity_pct'])
    plt.title(f'Correlation Check: Scale Factor vs Porosity\n(Pearson r={corr:.3f})', fontsize=14, fontweight='bold')
    plt.xlabel('Applied Scale Factor', fontsize=12)
    plt.ylabel('Porosity (%)', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'Fig5_Scale_Correlation.png'))
    print("已生成: Fig5_Scale_Correlation.png")
    
    # 5. 输出汇总 CSV
    group_stats = df.groupby('group')[['porosity_pct', 'scale_factor', 'clip_ratio_pct']].agg(['mean', 'std', 'count'])
    # 扁平化列名
    group_stats.columns = ['_'.join(col).strip() for col in group_stats.columns.values]
    group_stats = group_stats.reset_index()
    group_stats.to_csv(os.path.join(output_dir, 'Group_Statistics_Summary.csv'), index=False)
    print(f"已生成分组统计表: Group_Statistics_Summary.csv")

if __name__ == "__main__":
    analyze_and_plot(CSV_PATH, OUTPUT_DIR)
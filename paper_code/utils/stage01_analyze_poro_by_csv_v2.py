import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
import re

from utils.get_root_path import get_project_root

DIR_ROOT = get_project_root() # 输出paper_code的路径

# ================= 配置区域 =================
# 输入：生成的 CSV 报告路径
CSV_PATH = "/chendou_space/data/aligned_Training_Data_Interactive/processing_report.csv"
# 输出：分析图表保存路径
OUTPUT_DIR = DIR_ROOT / "utils" / "output_data" / "visualization_v2"

# 过滤阈值：孔隙度大于95%通常是纯空气背景，小于0.01%通常是纯骨架或无效数据
FILTER_EXTREME = True 
# ===========================================

def extract_group_from_filename(filename):
    """
    针对新文件名的分组逻辑
    输入: "6-6-20 全部_z3840_y649_x528.npy"
    输出: "6-6-20"
    """
    # 逻辑：取第一个空格前的部分，或者第一个下划线前的部分
    # 你可以根据实际文件名格式调整这里
    if ' ' in filename:
        return filename.split(' ')[0]
    elif '_' in filename:
        return filename.split('_')[0]
    else:
        return "Unknown"

def analyze_and_plot(csv_path, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    print(f"正在加载数据: {csv_path} ...")
    df = pd.read_csv(csv_path)
    
    # --- 1. 数据清洗与预处理 ---
    # 过滤掉处理失败的
    df = df[df['status'] == 'ok'].copy()
    
    # 转换孔隙度为百分比
    df['porosity_pct'] = df['porosity'] * 100 

    # [关键修改] 提取分组信息：不再使用文件夹路径，而是解析文件名
    df['group'] = df['file'].apply(extract_group_from_filename)
    
    # [关键修改] 标记并过滤极端异常值 (可选，但推荐)
    # 理由：孔隙度 > 90% 的通常是切到了岩石外面的空气，不具备统计学意义
    if FILTER_EXTREME:
        initial_count = len(df)
        df_valid = df[(df['porosity_pct'] < 95) & (df['porosity_pct'] > 0.05)].copy()
        filtered_count = initial_count - len(df_valid)
        print(f"已过滤 {filtered_count} 个极端样本 (Porosity > 95% 或 < 0.05%)")
    else:
        df_valid = df.copy()

    # --- 2. 打印修正后的统计报告 ---
    print("-" * 50)
    print("【优化后的数据统计摘要】")
    print(f"有效样本数 (REV): {len(df_valid)}")
    print(f"孔隙度均值: {df_valid['porosity_pct'].mean():.2f}%")
    print(f"孔隙度中位数: {df_valid['porosity_pct'].median():.2f}%")
    print(f"孔隙度 SD: {df_valid['porosity_pct'].std():.2f}%")
    # 重点关注 range，看是否还存在 98% 这种离谱数据
    print(f"Range: {df_valid['porosity_pct'].min():.2f}% - {df_valid['porosity_pct'].max():.2f}%")
    print(f"样本分组数: {df_valid['group'].nunique()} (组名示例: {df_valid['group'].unique()[:3]})")
    print("-" * 50)

    # ================= 绘图部分 =================
    
    # 图1: 孔隙度分布 (去除极端值后，分布应该更符合正态或对数正态)
    plt.figure(figsize=(10, 6), dpi=150)
    sns.histplot(data=df_valid, x='porosity_pct', kde=True, bins=50, 
                 color='#4A90E2', edgecolor='black', alpha=0.7)
    plt.axvline(df_valid['porosity_pct'].mean(), color='red', linestyle='--', label=f'Mean: {df_valid["porosity_pct"].mean():.2f}%')
    plt.title('Porosity Distribution (Filtered)', fontsize=14)
    plt.xlabel('Porosity (%)')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'Fig1_Porosity_Distribution.png'))

    # 图2: 分组箱线图 (现在应该能看到不同样本组的区别了)
    # 按照每组的平均孔隙度排序，让图表更好看
    order = df_valid.groupby('group')['porosity_pct'].median().sort_values().index
    
    plt.figure(figsize=(14, 6), dpi=150) # 加宽画布
    sns.boxplot(data=df_valid, x='group', y='porosity_pct', order=order, palette="viridis", hue='group', legend=False)
    plt.title('Porosity by Sample Group', fontsize=14)
    plt.xticks(rotation=45) # 旋转标签防止重叠
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'Fig2_Porosity_by_Group.png'))

    # 图3: Shift 分布 (保持不变)
    plt.figure(figsize=(10, 6), dpi=150)
    sns.histplot(data=df_valid, x='shift', kde=True, bins=50, color='#E74C3C')
    plt.title('Intensity Shift Distribution', fontsize=14)
    plt.savefig(os.path.join(output_dir, 'Fig3_Shift_Distribution.png'))

    # 图4: Shift vs Porosity (加上解释)
    plt.figure(figsize=(8, 8), dpi=150)
    sns.scatterplot(data=df_valid, x='shift', y='porosity_pct', alpha=0.3, color='purple', s=15)
    sns.regplot(data=df_valid, x='shift', y='porosity_pct', scatter=False, color='black', line_kws={'linestyle':'--'})
    
    corr = df_valid['shift'].corr(df_valid['porosity_pct'])
    plt.title(f'Shift vs Porosity (r={corr:.3f})', fontsize=14)
    plt.xlabel('Shift (Positive = Original was Darker)')
    plt.ylabel('Porosity (%)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'Fig4_Shift_Correlation.png'))

    # 5. 保存统计表
    stats = df_valid.groupby('group')['porosity_pct'].agg(['mean', 'std', 'count']).reset_index()
    stats.to_csv(os.path.join(output_dir, 'Group_Stats.csv'), index=False)
    print("分析完成。")

if __name__ == "__main__":
    analyze_and_plot(CSV_PATH, OUTPUT_DIR)
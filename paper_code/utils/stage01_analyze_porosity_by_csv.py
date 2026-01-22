import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os

from utils.get_root_path import get_project_root

DIR_ROOT = get_project_root() # 输出paper_code的路径

# ================= 配置区域 =================
# 输入：生成的 CSV 报告路径
CSV_PATH = "/chendou_space/data/aligned_Training_Data_Interactive/processing_report.csv"
# 输出：分析图表保存路径
OUTPUT_DIR = DIR_ROOT / "utils" / "output_data" / "visualization"
# 图像风格配置
plt.style.use('seaborn-v0_8-whitegrid') # 使用整洁的科研风格
# 设置全局字体 (如果有Times New Roman建议使用，没有则使用默认sans-serif)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False # 解决负号显示问题
# ===========================================

def analyze_and_plot(csv_path, output_dir):
    # 1. 准备目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # 2. 加载数据
    print(f"正在加载数据: {csv_path} ...")
    df = pd.read_csv(csv_path)
    
    # 3. 数据清洗
    total_count = len(df)
    df = df[df['status'] == 'ok'].copy()
    valid_count = len(df)
    print(f"总样本数: {total_count}, 有效样本数: {valid_count} (过滤失败样本)")
    
    # 数据转换
    df['porosity_pct'] = df['porosity'] * 100  # 转换为百分比
    # 从 rel_path 提取分组信息 (假设结构为 GroupName/filename.npy)
    # 如果你的结构不同，可能需要调整 lambda 函数，例如 p.parts[0]
    df['group'] = df['rel_path'].apply(lambda x: Path(x).parent.name) 
    
    # 4. 打印统计报告
    print("-" * 50)
    print("【数据统计摘要】")
    print(f"孔隙度均值: {df['porosity_pct'].mean():.2f}%")
    print(f"孔隙度中位数: {df['porosity_pct'].median():.2f}%")
    print(f"孔隙度标准差: {df['porosity_pct'].std():.2f}%")
    print(f"孔隙度范围: {df['porosity_pct'].min():.2f}% - {df['porosity_pct'].max():.2f}%")
    print(f"平均灰度偏移(Shift): {df['shift'].mean():.2f}")
    print("-" * 50)

    # ================= 绘图部分 =================
    
    # --- 图1: 孔隙度总体分布直方图 (Distribution) ---
    plt.figure(figsize=(10, 6), dpi=300)
    sns.histplot(data=df, x='porosity_pct', kde=True, bins=50, 
                 color='#4A90E2', edgecolor='black', alpha=0.7)
    
    # 添加统计标注
    mean_val = df['porosity_pct'].mean()
    plt.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2f}%')
    plt.title('Overall Porosity Distribution', fontsize=14, fontweight='bold')
    plt.xlabel('Porosity (%)', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'Fig1_Porosity_Distribution.png'))
    print("已生成: Fig1_Porosity_Distribution.png")

    # --- 图2: 不同分组(样本)的孔隙度箱线图 (Boxplot) ---
    # 如果组别太多，取数量最多的前10组
    top_groups = df['group'].value_counts().nlargest(10).index
    df_top = df[df['group'].isin(top_groups)]
    
    plt.figure(figsize=(12, 6), dpi=300)
    sns.boxplot(data=df_top, x='group', y='porosity_pct', hue='group', palette="viridis", legend=False)
    plt.title('Porosity Variation by Sample Group', fontsize=14, fontweight='bold')
    plt.xlabel('Sample Group', fontsize=12)
    plt.ylabel('Porosity (%)', fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'Fig2_Porosity_by_Group.png'))
    print("已生成: Fig2_Porosity_by_Group.png")

    # --- 图3: 原始灰度偏移量分布 (Shift Statistics) ---
    # 这张图很有科研价值，它展示了原始数据的不一致性，证明了你“对齐工作”的必要性
    plt.figure(figsize=(10, 6), dpi=300)
    sns.histplot(data=df, x='shift', kde=True, bins=50, color='#E74C3C', alpha=0.6)
    plt.title('Distribution of Intensity Shifts Applied', fontsize=14, fontweight='bold')
    plt.xlabel('Gray Value Shift (Target - Original Peak)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    
    # 标注意义
    plt.text(0.05, 0.9, 'Positive Shift: Original was Darker', transform=plt.gca().transAxes, color='black')
    plt.text(0.05, 0.85, 'Negative Shift: Original was Brighter', transform=plt.gca().transAxes, color='black')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'Fig3_Shift_Distribution.png'))
    print("已生成: Fig3_Shift_Distribution.png")

    # --- 图4: 偏移量 vs 孔隙度 相关性分析 (Correlation Check) ---
    # 用于验证：对齐操作是否引入了系统性偏差？理想情况下应该是散乱分布，无明显相关性
    plt.figure(figsize=(8, 8), dpi=300)
    sns.scatterplot(data=df, x='shift', y='porosity_pct', alpha=0.3, color='purple', s=20)
    # 添加回归线
    sns.regplot(data=df, x='shift', y='porosity_pct', scatter=False, color='black', line_kws={'linestyle':'--'})
    
    corr = df['shift'].corr(df['porosity_pct'])
    plt.title(f'Correlation Analysis: Shift vs Porosity (r={corr:.3f})', fontsize=14, fontweight='bold')
    plt.xlabel('Applied Intensity Shift', fontsize=12)
    plt.ylabel('Calculated Porosity (%)', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'Fig4_Shift_Correlation.png'))
    print("已生成: Fig4_Shift_Correlation.png")
    
    # 5. 输出汇总 CSV (按组统计)
    group_stats = df.groupby('group')['porosity_pct'].agg(['mean', 'std', 'min', 'max', 'count']).reset_index()
    group_stats.columns = ['Sample Group', 'Mean Porosity(%)', 'Std Dev', 'Min', 'Max', 'REV Count']
    group_stats.to_csv(os.path.join(output_dir, 'Group_Statistics_Summary.csv'), index=False)
    print(f"已生成分组统计表: Group_Statistics_Summary.csv")

if __name__ == "__main__":
    analyze_and_plot(CSV_PATH, OUTPUT_DIR)
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import re
import os
from scipy.stats import pearsonr

# ================= 配置 =================
# 指向 quick_infer_ema.py 生成的 csv 路径
CSV_PATH = r"C:\Users\vipuser\Desktop\chendou\latent_ddpm_code\exp_results\exp_08_latent_diffusion\quick_infer\quick_infer_summary.csv" 
# 结果保存路径
OUTPUT_DIR = os.path.dirname(CSV_PATH)
# ========================================

def extract_porosity(fname):
    """从文件名提取孔隙度"""
    match = re.search(r'porosity_(\d+\.\d+)', str(fname))
    if match:
        return float(match.group(1))
    return None

def analyze_csv(csv_path):
    if not os.path.exists(csv_path):
        print(f"❌ File not found: {csv_path}")
        return

    print(f"🚀 Analyzing: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # 1. 提取 Porosity
    df['Porosity_Cond'] = df['fname'].apply(extract_porosity)
    
    # 2. 基础统计打印
    print("\n" + "="*40)
    print("📊 核心指标诊断 (Global Diagnostics)")
    print("="*40)
    
    # --- 纹理一致性诊断 (Std) ---
    avg_gt_std = df['gt_std'].mean()
    avg_gen_std = df['gen_std'].mean()
    std_diff_ratio = (avg_gen_std - avg_gt_std) / avg_gt_std
    
    print(f"[Texture Energy]")
    print(f"  GT Std  : {avg_gt_std:.4f}")
    print(f"  Gen Std : {avg_gen_std:.4f}")
    if abs(std_diff_ratio) < 0.1:
        print("  ✅ 状态: 完美 (Perfect) - 纹理丰富度一致")
    elif std_diff_ratio > 0.1:
        print(f"  ⚠️ 状态: 过曝 (Oversaturated) - 偏高 {std_diff_ratio*100:.1f}%，可能有噪点")
    else:
        print(f"  ⚠️ 状态: 欠曝 (Undersaturated) - 偏低 {abs(std_diff_ratio)*100:.1f}%，纹理模糊")

    # --- 结构跟随性诊断 (Mean) ---
    print(f"\n[Structure/Density]")
    avg_gt_mean = df['gt_mean'].mean()
    avg_gen_mean = df['gen_mean'].mean()
    print(f"  GT Mean : {avg_gt_mean:.4f}")
    print(f"  Gen Mean: {avg_gen_mean:.4f}")
    
    # 检查均值坍缩 (Mode Collapse)
    gen_mean_std = df['gen_mean'].std()
    if gen_mean_std < 0.02:
        print("  ❌ 警告: 均值坍缩 (Mean Collapse) - 模型生成的岩石密度几乎不变，忽略了 Condition！")
    else:
        print(f"  ✅ 状态: 正常 - 生成的多样性尚可 (Std={gen_mean_std:.4f})")

    # --- 条件相关性诊断 (Correlation) ---
    print(f"\n[Condition Adherence]")
    df_valid = df.dropna(subset=['Porosity_Cond'])
    if len(df_valid) > 2:
        # 我们期望 Gen Mean (密度) 与 Porosity (孔隙率) 成反比 (通常孔隙是低值/黑，基质是高值/白)
        # 或者成正比，取决于你的数据归一化方式。
        # 这里只看相关性强度
        corr, _ = pearsonr(df_valid['Porosity_Cond'], df_valid['gen_mean'])
        print(f"  Porosity vs Gen_Mean Corr: {corr:.4f}")
        
        if abs(corr) > 0.5:
            print("  ✅ 状态: 听话 - 模型遵循了孔隙度条件")
        else:
            print("  ⚠️ 状态: 耳聋 - 模型似乎忽略了孔隙度条件")
    else:
        print("  ⚠️ 样本太少，无法计算相关性")

    # ================= 绘图 =================
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    plt.suptitle(f"Model Health Report: {os.path.basename(os.path.dirname(csv_path))}", fontsize=16)

    # 图1: Std 对比 (纹理能量)
    # 理想情况是落在红线上
    ax = axes[0, 0]
    sns.scatterplot(data=df, x='gt_std', y='gen_std', ax=ax, color='blue', s=100, alpha=0.7)
    # 画 y=x 参考线
    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
        np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
    ]
    ax.plot(lims, lims, 'r--', alpha=0.75, label='Ideal (y=x)')
    ax.set_title("Texture Consistency (Std Dev)", fontweight='bold')
    ax.set_xlabel("Ground Truth Std")
    ax.set_ylabel("Generated Std")
    ax.legend()

    # 图2: Porosity 控制力
    ax = axes[0, 1]
    if len(df_valid) > 0:
        sns.regplot(data=df_valid, x='Porosity_Cond', y='gen_mean', ax=ax, color='green', scatter_kws={'s':100})
        ax.set_title(f"Condition Control (Corr: {corr:.2f})", fontweight='bold')
        ax.set_xlabel("Input Porosity (%)")
        ax.set_ylabel("Generated Mean Intensity")
    else:
        ax.text(0.5, 0.5, "No Porosity Info", ha='center')

    # 图3: 均值分布对比 (检查坍缩)
    ax = axes[1, 0]
    sns.kdeplot(data=df, x='gt_mean', fill=True, label='GT Mean Dist', ax=ax, color='gray')
    sns.kdeplot(data=df, x='gen_mean', fill=True, label='Gen Mean Dist', ax=ax, color='orange')
    ax.set_title("Density Distribution (Mean Value)", fontweight='bold')
    ax.legend()
    # 如果 Gen 的峰非常尖，说明坍缩

    # 图4: 误差分布 (MSE & MaxAbs)
    ax = axes[1, 1]
    # 双轴图
    ax2 = ax.twinx()
    x_idx = range(len(df))
    ax.bar(x_idx, df['unknown_mse'], alpha=0.5, color='purple', label='MSE (Bar)')
    ax2.plot(x_idx, df['unknown_maxabs'], color='red', marker='o', linestyle='-', linewidth=2, label='MaxAbs (Line)')
    
    ax.set_ylabel("MSE (Mean Square Error)", color='purple')
    ax2.set_ylabel("Max Abs Error", color='red')
    ax.set_title("Reconstruction Error per Sample", fontweight='bold')
    ax.set_xticks(x_idx)
    ax.set_xticklabels([s[:10]+".." for s in df['fname']], rotation=45, ha='right', fontsize=8)
    
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, "analysis_report.png")
    plt.savefig(save_path, dpi=150)
    print(f"\n🖼️ Report saved to: {save_path}")
    # plt.show() # 如果在服务器上跑，注释掉这行

if __name__ == "__main__":
    analyze_csv(CSV_PATH)
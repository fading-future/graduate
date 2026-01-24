import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 1. 读取数据
# 请确保文件名和路径正确
csv_path = r"C:\Users\vipuser\Desktop\chendou\vqvae_code\src\stage1_logs\training_log.csv"  # 你的 CSV 文件路径
try:
    df = pd.read_csv(csv_path)
except FileNotFoundError:
    print("找不到文件，请确认 CSV 文件路径！")
    exit()

# 2. 设置“论文级”绘图风格
# 使用无衬线字体 (Arial/Helvetica)，线条加粗，字体加大
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'SimHei'] # SimHei 防止中文乱码
plt.rcParams['axes.linewidth'] = 1.
plt.rcParams['xtick.major.width'] = 1.
plt.rcParams['ytick.major.width'] = 1.
plt.rcParams['font.size'] = 14

# ==========================================
# 绘图方案：双轴 + 对数坐标 (Log Scale)
# ==========================================

fig, ax1 = plt.subplots(figsize=(10, 6), dpi=300) # 300 DPI 是打印标准

# 准备平滑数据 (Window size 可根据数据量调整，数据量大就设大点，比如 50-100)
window_size = 50 
df['Loss_Smooth'] = df['Total_Loss'].rolling(window=window_size, min_periods=1).mean()
df['Perp_Smooth'] = df['Perplexity'].rolling(window=window_size, min_periods=1).mean()

# --- 左轴：Loss (红色系) ---
color_loss = '#D62728' # 经典的科研红
ax1.set_xlabel('Iteration Steps', fontsize=14, fontweight='bold')
ax1.set_ylabel('Total Loss (Log Scale)', color=color_loss, fontsize=14, fontweight='bold')

# 1. 画原始数据的阴影 (透明度 alpha=0.15)
ax1.plot(df['Step'], df['Total_Loss'], color=color_loss, alpha=0.3, linewidth=0.5)
# 2. 画平滑曲线 (实线)
lns1 = ax1.plot(df['Step'], df['Loss_Smooth'], color=color_loss, linewidth=1.5, label='Total Loss (Smoothed)')

# 【关键点】开启对数坐标，解决 "Loss 变成直线" 的问题
ax1.set_yscale('log') 

ax1.tick_params(axis='y', labelcolor=color_loss)
ax1.grid(True, which="both", ls="--", alpha=0.3) # 开启精细网格

# --- 右轴：Perplexity (绿色系) ---
ax2 = ax1.twinx()
color_perp = '#2CA02C' # 经典的科研绿
ax2.set_ylabel('Perplexity (Codebook Usage)', color=color_perp, fontsize=14, fontweight='bold')

# 1. 画原始数据的阴影
ax2.plot(df['Step'], df['Perplexity'], color=color_perp, alpha=0.3, linewidth=0.5)
# 2. 画平滑曲线
lns2 = ax2.plot(df['Step'], df['Perp_Smooth'], color=color_perp, linewidth=1.5, label='Perplexity')

ax2.tick_params(axis='y', labelcolor=color_perp)

# --- 合并图例 ---
lns = lns1 + lns2
labs = [l.get_label() for l in lns]
ax1.legend(lns, labs, loc='center right', frameon=True, fancybox=True, shadow=True, fontsize=12)

plt.title('VQ-VAE Training Dynamics', fontsize=16, pad=15, fontweight='bold')
plt.tight_layout()

# 保存高清图
plt.savefig('vqvae_training_log_scale.png', dpi=300, bbox_inches='tight')
plt.savefig('vqvae_training_log_scale.pdf', bbox_inches='tight') # PDF 格式适合插入 LaTeX

print("绘图完成！图片已保存为 vqvae_training_log_scale.png")
plt.show()
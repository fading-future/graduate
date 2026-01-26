#%%
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

#%%
# ================= 配置区域 =================
# 替换成你实际的 CSV 文件路径
csv_path = "/chendou_space/chendou/paper_code/stage02_KLvae_single_code_v2/experiments/exp02_cube_structure_v1/train_log.csv" 

# 平滑窗口大小 (Window Size)
# 如果数据抖动厉害，把这个数字改大 (比如 50 或 100)
# 如果想看原始数据的每一个跳动，设为 1
SMOOTH_WINDOW = 10 

# 是否开启对数坐标 (Log Scale)
# 针对 KL 散度这种容易爆炸的 Loss，建议开启
USE_LOG_SCALE_KL = True 
USE_LOG_SCALE_RECON = True # 重建损失通常不需要 Log，除非初期下降极快

# ================= 数据加载 =================
if not os.path.exists(csv_path):
    # 如果找不到文件，创建一个假的 DataFrame 用于演示效果
    print(f"⚠️ 警告: 找不到文件 {csv_path}，正在生成模拟数据用于演示...")
    data = {
        'Step': np.arange(100),
        'Loss_Total': np.random.rand(100) * 10 + 100,
        'Loss_Recon': np.linspace(0.3, 0.05, 100) + np.random.rand(100) * 0.01,
        'Loss_KL': np.linspace(5000, 100, 100) + np.random.rand(100) * 500,
        'Loss_G_Adv': np.zeros(100),
        'Loss_D': np.zeros(100)
    }
    # 模拟 50 步之后开启对抗
    data['Loss_G_Adv'][50:] = np.random.rand(50) * 0.5
    data['Loss_D'][50:] = np.random.rand(50) * 0.5 + 0.5
    df = pd.DataFrame(data)
else:
    df = pd.read_csv(csv_path)
    print(f"✅ 成功加载日志，共 {len(df)} 条记录")

# ================= 数据处理 (平滑) =================
# 使用 rolling mean 进行平滑，min_periods=1 保证开头的数据也能显示
df_smooth = df.rolling(window=SMOOTH_WINDOW, min_periods=1).mean()
# Step 不需要平滑，保持原样
df_smooth['Step'] = df['Step']

# ================= 绘图逻辑 =================
# 创建 3 行 1 列的图表，共享 X 轴 (Step)
fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

# --- 子图 1: 重建损失 (核心指标) ---
ax1 = axes[0]
ax1.plot(df['Step'], df['Loss_Recon'], color='lightgray', alpha=0.4, label='Raw') # 原始数据(浅色)
ax1.plot(df_smooth['Step'], df_smooth['Loss_Recon'], color='blue', linewidth=2, label=f'Smoothed ({SMOOTH_WINDOW})')
ax1.set_title('Reconstruction Loss (L1) - Should Decrease', fontsize=12, fontweight='bold')
ax1.set_ylabel('Loss Value')
ax1.grid(True, linestyle='--', alpha=0.6)
ax1.legend()
if USE_LOG_SCALE_RECON:
    ax1.set_yscale('log')

# --- 子图 2: KL 散度 (正则化指标) ---
ax2 = axes[1]
ax2.plot(df['Step'], df['Loss_KL'], color='lightgray', alpha=0.4)
ax2.plot(df_smooth['Step'], df_smooth['Loss_KL'], color='orange', linewidth=2, label='KL Loss')
ax2.set_title('KL Divergence - Check for Stability', fontsize=12, fontweight='bold')
ax2.set_ylabel('Loss Value')
ax2.grid(True, linestyle='--', alpha=0.6)
ax2.legend()

# 重点：KL 散度通常数值很大，开启对数坐标更容易看清下降趋势
if USE_LOG_SCALE_KL:
    ax2.set_yscale('log')
    ax2.set_ylabel('Loss Value (Log Scale)')

# --- 子图 3: 对抗损失 (GAN指标) ---
ax3 = axes[2]
# G_Adv
ax3.plot(df_smooth['Step'], df_smooth['Loss_G_Adv'], color='green', label='Generator Adv Loss')
# D Loss
ax3.plot(df_smooth['Step'], df_smooth['Loss_D'], color='red', linestyle='--', label='Discriminator Loss')

ax3.set_title('Adversarial Losses (GAN) - Active after warmup', fontsize=12, fontweight='bold')
ax3.set_ylabel('Loss Value')
ax3.set_xlabel('Global Steps')
ax3.grid(True, linestyle='--', alpha=0.6)
ax3.legend()

# 自动调整布局，防止文字重叠
plt.tight_layout()
plt.show()


# import pandas as pd
# import matplotlib.pyplot as plt
# import argparse
# import os

# def plot_training_logs(csv_path, output_path=None, smooth_window=50):
#     """
#     读取 CSV 并绘制训练曲线
#     :param csv_path: CSV 文件路径
#     :param output_path: 保存图片的路径，如果为 None 则自动生成
#     :param smooth_window: 平滑窗口大小（多少个 step 取平均），设为 1 则不平滑
#     """
#     if not os.path.exists(csv_path):
#         print(f"Error: File not found at {csv_path}")
#         return

#     # 1. 读取数据
#     df = pd.read_csv(csv_path)
#     print(f"Loaded {len(df)} steps from {csv_path}")

#     # 2. 设置绘图风格
#     plt.style.use('ggplot') # 使用比较好看的 ggplot 风格
#     fig, axes = plt.subplots(2, 2, figsize=(16, 12))
#     fig.suptitle(f'Training Metrics (Smoothed window={smooth_window})', fontsize=16)

#     # 辅助函数：绘制平滑曲线
#     def plot_metric(ax, x, y, label, color, use_log=False):
#         # 原始数据（透明度高，作为背景）
#         ax.plot(x, y, color=color, alpha=0.15, linewidth=1)
        
#         # 平滑数据（实线）
#         if smooth_window > 1:
#             y_smooth = y.rolling(window=smooth_window, min_periods=1).mean()
#             ax.plot(x, y_smooth, color=color, label=f'{label} (avg)', linewidth=2)
#         else:
#             ax.plot(x, y, color=color, label=label, linewidth=2)
            
#         if use_log:
#             ax.set_yscale('log')
#             ax.set_ylabel("Log Scale")
        
#         ax.set_title(label)
#         ax.set_xlabel("Steps")
#         ax.grid(True, which="both", ls="-", alpha=0.5)
#         ax.legend()

#     # --- 子图 1: Reconstruction Loss (线性坐标) ---
#     # 这是最重要的指标，通常不需要对数坐标，除非初期下降太快
#     plot_metric(axes[0, 0], df['Step'], df['Loss_Recon'], 'Reconstruction Loss', 'tab:blue', use_log=False)

#     # --- 子图 2: KL Divergence (对数坐标) ---
#     # KL 散度初期会非常大，必须用 Log 坐标才能看清后期的微小变化
#     plot_metric(axes[0, 1], df['Step'], df['Loss_KL'], 'KL Divergence', 'tab:orange', use_log=True)

#     # --- 子图 3: Total Loss (对数坐标) ---
#     # 因为包含了 KL，Total Loss 通常也需要 Log 坐标
#     plot_metric(axes[1, 0], df['Step'], df['Loss_Total'], 'Total Loss', 'tab:red', use_log=True)

#     # --- 子图 4: Adversarial Losses (线性坐标) ---
#     # 只绘制非 0 的部分（即 disc_start 之后）
#     df_adv = df[df['Loss_D'] != 0] # 简单的过滤
#     if len(df_adv) > 0:
#         # G_Adv
#         y_g_smooth = df_adv['Loss_G_Adv'].rolling(window=smooth_window, min_periods=1).mean()
#         axes[1, 1].plot(df_adv['Step'], y_g_smooth, label='G_Adv', color='tab:purple')
        
#         # D Loss
#         y_d_smooth = df_adv['Loss_D'].rolling(window=smooth_window, min_periods=1).mean()
#         axes[1, 1].plot(df_adv['Step'], y_d_smooth, label='Discriminator', color='tab:green')
        
#         axes[1, 1].set_title('Adversarial Losses (Post-Start)')
#         axes[1, 1].set_xlabel("Steps")
#         axes[1, 1].legend()
#         axes[1, 1].grid(True)
#     else:
#         axes[1, 1].text(0.5, 0.5, 'Discriminator not started yet', 
#                         horizontalalignment='center', verticalalignment='center', transform=axes[1, 1].transAxes)

#     # 3. 保存与显示
#     plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # 调整布局防止重叠
    
#     if output_path is None:
#         output_path = csv_path.replace('.csv', '_plot.png')
    
#     plt.savefig(output_path, dpi=300)
#     print(f"Plot saved to: {output_path}")

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Visualize Training Log CSV")
#     parser.add_argument('csv_path', type=str, help="Path to the train_log.csv file")
#     parser.add_argument('--window', type=int, default=50, help="Smoothing window size (default: 50)")
    
#     args = parser.parse_args()
    
#     plot_training_logs(args.csv_path, smooth_window=args.window)
# %%

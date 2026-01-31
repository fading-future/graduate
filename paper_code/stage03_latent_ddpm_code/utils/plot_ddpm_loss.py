#%%
import pandas as pd
import matplotlib
# matplotlib.use('Agg') # 关键：禁用 GUI 后端
import matplotlib.pyplot as plt
import os
import sys

#%%
# 设置路径 (根据实际情况修改，或者通过命令行传入)
# LOG_FILE = r"C:\Users\Administrator\Desktop\paper\stage2_latentddpm_code\exp_results\exp_02_latent_diffusion\logs\training_log.csv"
LOG_FILE = r"/chendou_space/chendou/paper_code/stage03_latent_ddpm_code/exp_results/exp2_final_stage2_graduation/logs/training_log.csv"

def plot_ddpm_curves(log_path):
    if not os.path.exists(log_path):
        print(f"Log file not found: {log_path}")
        return

    df = pd.read_csv(log_path)
    
    # 设置科研绘图风格
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
    plt.rcParams['font.size'] = 14
    plt.rcParams['axes.linewidth'] = 1.2
    
    fig, ax1 = plt.subplots(figsize=(10, 6), dpi=300)
    
    # 数据平滑
    window_size = 100 # Diffusion 的 Loss 震荡大，窗口可以设大点
    df['Loss_Smooth'] = df['Loss'].rolling(window=window_size, min_periods=1).mean()
    
    steps = df['Step']
    
    # --- 绘制 Loss ---
    color_loss = '#D62728' # 红色
    
    ax1.set_xlabel('Iteration Steps', fontsize=14, fontweight='bold')
    ax1.set_ylabel('MSE Loss (Log Scale)', color=color_loss, fontsize=14, fontweight='bold')
    
    # 原始数据背景 (Alpha 低)
    ax1.plot(steps, df['Loss'], color=color_loss, alpha=0.6, linewidth=0.5)
    # 平滑曲线
    lns1 = ax1.plot(steps, df['Loss_Smooth'], color=color_loss, linewidth=2, label='MSE Loss (Smoothed)')
    
    # 对数坐标
    ax1.set_yscale('log')
    ax1.tick_params(axis='y', labelcolor=color_loss)
    ax1.grid(True, which="both", ls="--", alpha=0.5)
    
    # 如果你想把 Learning Rate 也画上去 (虽然 Adam 通常是固定的，除非用了 Scheduler)
    ax2 = ax1.twinx()
    color_lr = '#1f77b4'
    ax2.set_ylabel('Learning Rate', color=color_lr, fontsize=14, fontweight='bold')
    lns2 = ax2.plot(steps, df['LR'], color=color_lr, linewidth=1.5, linestyle='--', label='Learning Rate')
    ax2.tick_params(axis='y', labelcolor=color_lr)
    lns = lns1 + lns2
    
    # 图例
    lns = lns1 
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc='upper right', frameon=True, fancybox=True, shadow=True)
    
    plt.title('Latent Diffusion Training Dynamics', fontsize=16, pad=15, fontweight='bold')
    plt.tight_layout()
    
    save_dir = os.path.dirname(log_path)
    # plt.savefig(os.path.join(save_dir, 'loss_curve_paper.png'))
    # plt.savefig(os.path.join(save_dir, 'loss_curve_paper.pdf'))
    print(f"Plot saved to {save_dir}")

if __name__ == "__main__":
    plot_ddpm_curves(LOG_FILE)
# %%

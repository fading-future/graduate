#%%
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import os


#%%
# ================= 配置区域 =================
# 请将此处修改为你实际的 CSV 文件路径
LOG_FILE = r"C:\Users\Administrator\Desktop\code\graduation_thesis_code\exp_results\exp_05\logs\training_log.csv"
# ===========================================

def plot_training_curves(log_path):
    if not os.path.exists(log_path):
        print(f"❌ Error: Log file not found at: {log_path}")
        return

    print(f"reading log from: {log_path} ...")
    try:
        df = pd.read_csv(log_path)
    except Exception as e:
        print(f"❌ Error reading CSV: {e}")
        return

    # 1. 自动生成 Global Step
    # 因为你的 CSV 是每个 step 记录一行，所以索引+1 就是 Global Step
    df['Global_Step'] = df.index + 1
    
    # 2. 设置科研绘图风格
    plt.rcParams['font.family'] = 'sans-serif'
    # 优先使用 Arial，如果没有则回退到系统默认 sans-serif
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans', 'sans-serif']
    plt.rcParams['font.size'] = 14
    plt.rcParams['axes.linewidth'] = 1.2
    
    fig, ax1 = plt.subplots(figsize=(12, 7), dpi=300)
    
    # 3. 数据平滑处理
    # Window size 可以根据你的总 Step 数调整。
    # 如果数据量很大（例如几万行），建议设为 100-500 以获得平滑的趋势线
    window_size = 250
    df['Loss_Smooth'] = df['Current_Loss'].rolling(window=window_size, min_periods=1).mean()
    
    steps = df['Global_Step']
    
    # --- 绘制 Loss ---
    color_loss = '#D62728' # 科研红
    
    ax1.set_xlabel('Iteration Steps', fontsize=16, fontweight='bold')
    ax1.set_ylabel('MSE Loss (Log Scale)', color=color_loss, fontsize=16, fontweight='bold')
    
    # 绘制原始数据背景 (透明度 Alpha 低，作为背景噪音显示)
    ax1.plot(steps, df['Current_Loss'], color=color_loss, alpha=0.15, linewidth=0.5, label='Raw Loss')
    
    # 绘制平滑曲线 (深色，作为主要趋势)
    lns1 = ax1.plot(steps, df['Loss_Smooth'], color=color_loss, linewidth=2.5, label=f'Smoothed Loss (MA={window_size})')
    
    # 关键：Diffusion 模型 Loss 变化幅度大，通常使用对数坐标
    ax1.set_yscale('log')
    ax1.tick_params(axis='y', labelcolor=color_loss, labelsize=12)
    ax1.tick_params(axis='x', labelsize=12)
    
    #以此添加网格
    ax1.grid(True, which="major", ls="-", alpha=0.4, color='gray')
    ax1.grid(True, which="minor", ls="--", alpha=0.1, color='gray')
    
    # --- (可选) 绘制 Learning Rate ---
    # 如果你想看学习率变化（例如使用了 Cosine Annealing），取消下面代码的注释
    """
    ax2 = ax1.twinx()
    color_lr = '#1f77b4' # 科研蓝
    ax2.set_ylabel('Learning Rate', color=color_lr, fontsize=16, fontweight='bold')
    lns2 = ax2.plot(steps, df['Learning_Rate'], color=color_lr, linewidth=2, linestyle='--', label='Learning Rate')
    ax2.tick_params(axis='y', labelcolor=color_lr)
    # 合并图例
    lns1 = lns1 + lns2
    """
    
    # 图例设置
    labs = [l.get_label() for l in lns1]
    ax1.legend(lns1, labs, loc='upper right', frameon=True, fancybox=True, shadow=True, fontsize=12)
    
    # 标题和布局
    plt.title('Training Dynamics', fontsize=18, pad=15, fontweight='bold')
    plt.tight_layout()
    
    # 保存图片
    save_dir = os.path.dirname(log_path)
    save_path_png = os.path.join(save_dir, 'loss_curve_paper.png')
    save_path_pdf = os.path.join(save_dir, 'loss_curve_paper.pdf')
    
    # plt.savefig(save_path_png)
    # plt.savefig(save_path_pdf)
    plt.show()
    print(f"✅ Plot saved to:\n   - {save_path_png}\n   - {save_path_pdf}")
    
    # 关闭图表释放内存
    plt.close()

if __name__ == "__main__":
    plot_training_curves(LOG_FILE)
# %%

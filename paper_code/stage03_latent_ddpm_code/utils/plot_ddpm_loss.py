#%%
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob

#%%
LOG_FILE = r"/chendou_space/chendou/paper_code/stage03_latent_ddpm_code/exp_results/exp6_final_stage2_graduation/logs/training_log.csv"
PLOT_EMA_ALPHA_REGION = 0.03  # 可调，0.01~0.05 常用

def _rolling_mean(s: pd.Series, window: int):
    return s.rolling(window=window, min_periods=1).mean()

def ema_smooth(series: pd.Series, alpha: float = 0.03):
    """
    Exponential moving average smoothing.
    alpha 越小越平滑（越慢），常用 0.01~0.05
    """
    return series.ewm(alpha=alpha, adjust=False).mean()

def plot_ddpm_and_region_curves(training_log_path: str,
                                window_loss: int = 100,
                                window_region: int = 100,
                                save_png: bool = True,
                                save_pdf: bool = True):
    if not os.path.exists(training_log_path):
        print(f"[ERR] training log not found: {training_log_path}")
        return

    log_dir = os.path.dirname(training_log_path)

    # ---------- load training_log.csv ----------
    df = pd.read_csv(training_log_path)
    if 'Step' not in df.columns or 'Loss' not in df.columns:
        raise ValueError(f"training_log.csv must contain columns: Step, Loss (and optionally LR). Got: {df.columns.tolist()}")

    df = df.sort_values('Step').reset_index(drop=True)
    df['Loss_Smooth'] = _rolling_mean(df['Loss'], window_loss)

    # ---------- locate region csv ----------
    # Prefer v2 (with BoundaryX0Cons)
    candidates = [
        os.path.join(log_dir, "region_loss_log_v2.csv"),
        os.path.join(log_dir, "region_loss_log.csv"),
    ]
    region_path = None
    for p in candidates:
        if os.path.exists(p):
            region_path = p
            break
    # fallback: try glob
    if region_path is None:
        matches = sorted(glob.glob(os.path.join(log_dir, "region_loss_log*.csv")))
        if matches:
            region_path = matches[-1]

    # ---------- style ----------
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
    plt.rcParams['font.size'] = 14
    plt.rcParams['axes.linewidth'] = 1.2

    # =========================================================
    # Figure 1: Loss + LR
    # =========================================================
    fig1, ax1 = plt.subplots(figsize=(10, 6), dpi=300)

    ax1.set_xlabel('Iteration Steps', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Loss (Log Scale)', fontsize=14, fontweight='bold')

    ax1.plot(df['Step'], df['Loss'], alpha=0.5, linewidth=0.5, label='Loss (Raw)')
    ax1.plot(df['Step'], df['Loss_Smooth'], linewidth=2.0, label=f'Loss (Smoothed, w={window_loss})')

    ax1.set_yscale('log')
    ax1.grid(True, which="both", ls="--", alpha=0.5)

    # LR on right axis (if exists)
    handles = []
    labels = []
    h1, l1 = ax1.get_legend_handles_labels()
    handles += h1
    labels += l1

    if 'LR' in df.columns:
        ax2 = ax1.twinx()
        ax2.set_ylabel('Learning Rate', fontsize=14, fontweight='bold')
        ax2.plot(df['Step'], df['LR'], linewidth=1.2, linestyle='--', label='Learning Rate')
        h2, l2 = ax2.get_legend_handles_labels()
        handles += h2
        labels += l2

    ax1.legend(handles, labels, loc='upper right', frameon=True, fancybox=True, shadow=True)
    plt.title('Latent Diffusion Training Dynamics', fontsize=16, pad=15, fontweight='bold')
    plt.tight_layout()

    # if save_png:
    #     plt.savefig(os.path.join(log_dir, 'loss_curve_paper.png'))
    # if save_pdf:
    #     plt.savefig(os.path.join(log_dir, 'loss_curve_paper.pdf'))
    print(f"[OK] Saved loss curves to: {log_dir}")

    # =========================================================
    # Figure 2: Region losses + UnknownRatio
    # =========================================================
    if region_path is None:
        print("[WARN] region csv not found (region_loss_log_v2.csv / region_loss_log.csv). Skip region plot.")
        plt.show()
        return

    # ---------- load region csv ----------
    dfr = pd.read_csv(region_path).sort_values('Step').reset_index(drop=True)

    # EMA 平滑强度：0.01~0.05
    alpha_region = float(PLOT_EMA_ALPHA_REGION, 0.03) if 'CONFIG' in globals() else 0.03

    # smooth with EMA
    for col in ['DiffLoss_unknown', 'DiffLoss_known', 'KnownConsistency']:
        dfr[col + '_EMA'] = ema_smooth(dfr[col], alpha=alpha_region)

    has_bx = 'BoundaryX0Cons' in dfr.columns
    if has_bx:
        dfr['BoundaryX0Cons_EMA'] = ema_smooth(dfr['BoundaryX0Cons'], alpha=alpha_region)

    dfr['UnknownRatio_EMA'] = ema_smooth(dfr['UnknownRatio'], alpha=alpha_region)

    # ---------- plot ----------
    fig2, axr1 = plt.subplots(figsize=(10, 6), dpi=300)
    axr1.set_xlabel('Iteration Steps', fontsize=14, fontweight='bold')
    axr1.set_ylabel('Region Loss (EMA Smoothed)', fontsize=14, fontweight='bold')

    axr1.plot(dfr['Step'], dfr['DiffLoss_unknown_EMA'], linewidth=2.0, label=f'DiffLoss_unknown (EMA α={alpha_region})')
    axr1.plot(dfr['Step'], dfr['DiffLoss_known_EMA'],   linewidth=2.0, label=f'DiffLoss_known (EMA α={alpha_region})')
    axr1.plot(dfr['Step'], dfr['KnownConsistency_EMA'], linewidth=2.0, label=f'KnownConsistency (EMA α={alpha_region})')

    if has_bx:
        axr1.plot(dfr['Step'], dfr['BoundaryX0Cons_EMA'], linewidth=2.0, label=f'BoundaryX0Cons (EMA α={alpha_region})')

    # 可选：Region y 轴用 log（注意：必须全 > 0）
    use_log_region = True  # 你想看“细微变化”，通常开
    if use_log_region:
        # 避免 0 值导致 log 崩溃
        axr1.set_yscale('log')

    axr1.grid(True, which="both", ls="--", alpha=0.5)

    # UnknownRatio on right axis
    axr2 = axr1.twinx()
    axr2.set_ylabel('UnknownRatio (EMA)', fontsize=14, fontweight='bold')
    axr2.plot(dfr['Step'], dfr['UnknownRatio_EMA'], linewidth=1.6, linestyle='--',
            label=f'UnknownRatio (EMA α={alpha_region})')
    axr2.set_ylim(0.0, 1.0)

    # combined legend
    h1, l1 = axr1.get_legend_handles_labels()
    h2, l2 = axr2.get_legend_handles_labels()
    axr1.legend(h1 + h2, l1 + l2, loc='upper right', frameon=True, fancybox=True, shadow=True)

    plt.title('Region-wise Training Dynamics (EMA)', fontsize=16, pad=15, fontweight='bold')
    plt.tight_layout()

    # plt.savefig(os.path.join(log_dir, 'region_curve_paper_ema.png'))
    # plt.savefig(os.path.join(log_dir, 'region_curve_paper_ema.pdf'))
    print(f"[OK] Saved: region_curve_paper_ema.(png/pdf) -> {log_dir}")

    plt.show()

if __name__ == "__main__":
    plot_ddpm_and_region_curves(LOG_FILE, window_loss=100, window_region=100)
#%%



# #%%
# import pandas as pd
# import matplotlib
# # matplotlib.use('Agg') # 关键：禁用 GUI 后端
# import matplotlib.pyplot as plt
# import os
# import sys

# #%%
# # 设置路径 (根据实际情况修改，或者通过命令行传入)
# # LOG_FILE = r"C:\Users\Administrator\Desktop\paper\stage2_latentddpm_code\exp_results\exp_02_latent_diffusion\logs\training_log.csv"
# LOG_FILE = r"/chendou_space/chendou/paper_code/stage03_latent_ddpm_code/exp_results/exp5_final_stage2_graduation/logs/training_log.csv"

# def plot_ddpm_curves(log_path):
#     if not os.path.exists(log_path):
#         print(f"Log file not found: {log_path}")
#         return

#     df = pd.read_csv(log_path)
    
#     # 设置科研绘图风格
#     plt.rcParams['font.family'] = 'sans-serif'
#     plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
#     plt.rcParams['font.size'] = 14
#     plt.rcParams['axes.linewidth'] = 1.2
    
#     fig, ax1 = plt.subplots(figsize=(10, 6), dpi=300)
    
#     # 数据平滑
#     window_size = 100 # Diffusion 的 Loss 震荡大，窗口可以设大点
#     df['Loss_Smooth'] = df['Loss'].rolling(window=window_size, min_periods=1).mean()
    
#     steps = df['Step']
    
#     # --- 绘制 Loss ---
#     color_loss = '#D62728' # 红色
    
#     ax1.set_xlabel('Iteration Steps', fontsize=14, fontweight='bold')
#     ax1.set_ylabel('MSE Loss (Log Scale)', color=color_loss, fontsize=14, fontweight='bold')
    
#     # 原始数据背景 (Alpha 低)
#     ax1.plot(steps, df['Loss'], color=color_loss, alpha=0.6, linewidth=0.5)
#     # 平滑曲线
#     lns1 = ax1.plot(steps, df['Loss_Smooth'], color=color_loss, linewidth=2, label='MSE Loss (Smoothed)')
    
#     # 对数坐标
#     ax1.set_yscale('log')
#     ax1.tick_params(axis='y', labelcolor=color_loss)
#     ax1.grid(True, which="both", ls="--", alpha=0.5)
    
#     # 如果你想把 Learning Rate 也画上去 (虽然 Adam 通常是固定的，除非用了 Scheduler)
#     ax2 = ax1.twinx()
#     color_lr = '#1f77b4'
#     ax2.set_ylabel('Learning Rate', color=color_lr, fontsize=14, fontweight='bold')
#     lns2 = ax2.plot(steps, df['LR'], color=color_lr, linewidth=1.5, linestyle='--', label='Learning Rate')
#     ax2.tick_params(axis='y', labelcolor=color_lr)
#     lns = lns1 + lns2
    
#     # 图例
#     lns = lns1 
#     labs = [l.get_label() for l in lns]
#     ax1.legend(lns, labs, loc='upper right', frameon=True, fancybox=True, shadow=True)
    
#     plt.title('Latent Diffusion Training Dynamics', fontsize=16, pad=15, fontweight='bold')
#     plt.tight_layout()
    
#     save_dir = os.path.dirname(log_path)
#     # plt.savefig(os.path.join(save_dir, 'loss_curve_paper.png'))
#     # plt.savefig(os.path.join(save_dir, 'loss_curve_paper.pdf'))
#     print(f"Plot saved to {save_dir}")

# if __name__ == "__main__":
#     plot_ddpm_curves(LOG_FILE)
# #%%

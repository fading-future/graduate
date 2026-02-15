#%%
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

#%%
# ================= 配置区域 =================
# 替换成你实际的 CSV 文件路径
csv_path = r"E:\chendou\paper_code\stage02_KLvae_single_code_v2\experiments\exp05_cube_structure_v2\train_log.csv"

# 平滑窗口大小
SMOOTH_WINDOW = 20

# 限制显示的最大 Step（如果想看前 2000 步，设为 2000；设为 None 显示所有）
MAX_STEPS_TO_SHOW = None 

# ================= 数据加载 =================
if not os.path.exists(csv_path):
    print(f"⚠️ 警告: 找不到文件 {csv_path}，无法绘图。")
    df = pd.DataFrame()
else:
    try:
        try:
            # 先按 UTF-8（含 BOM）读
            df = pd.read_csv(csv_path, encoding="utf-8-sig")
        except UnicodeDecodeError:
            # 再尝试 Windows 常见编码
            df = pd.read_csv(csv_path, encoding="gbk")  # 若仍失败，把 gbk 改成 "cp950" 或 "latin1"
        
        # 过滤掉可能的空行
        df.columns = [c.strip() for c in df.columns]  # 顺便去掉列名空格，避免 'Step ' 这种
        df['Step'] = pd.to_numeric(df['Step'], errors='coerce')  # 强制 Step 转数字
        df = df.dropna(subset=['Step'])

        print(f"✅ 成功加载日志，共 {len(df)} 条记录")
        # 过滤掉可能的空行
        df = df.dropna(subset=['Step'])
        
        # 强制将关键数值列转换为数字，防止混入字符串
        numeric_target_cols = ['Step', 'Loss_Total', 'Loss_Recon', 'Loss_KL', 
                               'KL_avg_per_latent', 'Loss_G_Adv', 'Loss_D', 
                               'GradNorm_VAE', 'GradNorm_Disc', 'KL_Weight']
        for col in numeric_target_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        print(f"✅ 成功加载日志，共 {len(df)} 条记录")
    except Exception as e:
        print(f"❌ 读取 CSV 失败: {e}")
        df = pd.DataFrame()

if not df.empty:
    if MAX_STEPS_TO_SHOW:
        df = df[df['Step'] <= MAX_STEPS_TO_SHOW]

    # ================= 数据处理 (平滑) - 修复部分 =================
    # 1. 仅选择数值类型的列进行平滑，避开 "Time" 字符串列
    cols_to_smooth = df.select_dtypes(include=[np.number]).columns
    
    # 2. 对数值列进行平滑
    df_smooth = df[cols_to_smooth].rolling(window=SMOOTH_WINDOW, min_periods=1).mean()
    
    # 3. 将不需要平滑的列（如 Step 和 线性变化的 KL_Weight）覆盖回去，保持原始值
    if 'Step' in df.columns:
        df_smooth['Step'] = df['Step']
    if 'KL_Weight' in df.columns:
        df_smooth['KL_Weight'] = df['KL_Weight']

    # ================= 绘图逻辑 =================
    # 创建 2x2 的图表布局
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharex=True)
    plt.subplots_adjust(hspace=0.25, wspace=0.2)

    # --- 图 1: 重建损失 (Reconstruction) ---
    ax1 = axes[0, 0]
    ax1.plot(df['Step'], df['Loss_Recon'], color='lightgray', alpha=0.5, label='Raw')
    ax1.plot(df_smooth['Step'], df_smooth['Loss_Recon'], color='#1f77b4', linewidth=2, label='Smoothed L1')
    
    ax1.set_title('1. Reconstruction Loss (L1) ↓', fontsize=12, fontweight='bold')
    ax1.set_ylabel('L1 Loss')
    ax1.set_yscale('log')
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend(loc='upper right')

    # --- 图 2: KL 散度与权重 (KL Logic) ---
    ax2 = axes[0, 1]
    if 'KL_avg_per_latent' in df.columns:
        # 左轴：显示每个 Latent 的平均 KL
        line1 = ax2.plot(df_smooth['Step'], df_smooth['KL_avg_per_latent'], color='#ff7f0e', linewidth=2, label='Avg KL per Latent')
        # ax2.set_ylabel('Avg KL (nats)', color='#ff7f0e')
        ax2.tick_params(axis='y', labelcolor='#ff7f0e')
        
        # 警戒线
        ax2.axhline(y=0.5, color='red', linestyle=':', alpha=0.5, label='Warning (0.5)')
        ax2.axhline(y=0.1, color='green', linestyle=':', alpha=0.5, label='Target (<0.1)')
        
        # 右轴：显示 KL 权重
        if 'KL_Weight' in df.columns:
            ax2b = ax2.twinx()
            line2 = ax2b.plot(df['Step'], df['KL_Weight'], color='gray', linestyle='--', alpha=0.6, label='KL Weight Schedule')
            ax2b.set_ylabel('KL Weight', color='gray')
            ax2b.tick_params(axis='y', labelcolor='gray')
            
            lines = line1 + line2
            # labels = [l.get_label() for l in lines] + ['Warning (0.5)', 'Target (0.1)']
            # ax2.legend(lines + [plt.Line2D([0], [0], color='red', linestyle=':'), plt.Line2D([0], [0], color='green', linestyle=':')], labels, loc='upper right')
        
        ax2.set_title('2. KL Divergence & Warmup Schedule', fontsize=12, fontweight='bold')
        ax2.grid(True, linestyle='--', alpha=0.5)
    else:
        ax2.text(0.5, 0.5, 'Missing KL_avg_per_latent data', ha='center', transform=ax2.transAxes)

    # --- 图 3: 对抗损失 (GAN) ---
    ax3 = axes[1, 0]
    if 'Loss_G_Adv' in df.columns and 'Loss_D' in df.columns:
        mask = df['Loss_D'] > 1e-6
        if mask.any():
            ax3.plot(df_smooth.loc[mask, 'Step'], df_smooth.loc[mask, 'Loss_G_Adv'], color='#2ca02c', label='G Adversarial Loss')
            ax3.plot(df_smooth.loc[mask, 'Step'], df_smooth.loc[mask, 'Loss_D'], color='#d62728', linestyle='--', alpha=0.7, label='Discriminator Loss')
            ax3.set_title('3. GAN Losses (Active after warmup)', fontsize=12, fontweight='bold')
            ax3.legend(loc='upper right')
        else:
            ax3.text(0.5, 0.5, 'Discriminator not started yet', ha='center', transform=ax3.transAxes)
            ax3.set_title('3. GAN Losses (Waiting...)', fontsize=12, fontweight='bold')
    
    ax3.set_ylabel('Loss')
    ax3.set_xlabel('Global Steps')
    ax3.grid(True, linestyle='--', alpha=0.5)

    # --- 图 4: 梯度范数 (Gradient Norms) ---
    ax4 = axes[1, 1]
    if 'GradNorm_VAE' in df.columns:
        ax4.plot(df['Step'], df['GradNorm_VAE'], color='purple', alpha=0.3, label='VAE Grad Raw')
        ax4.plot(df_smooth['Step'], df_smooth['GradNorm_VAE'], color='purple', linewidth=2, label='VAE Grad Smooth')
    
    if 'GradNorm_Disc' in df.columns:
        mask = df['GradNorm_Disc'] > 0
        if mask.any():
            ax4.plot(df_smooth.loc[mask, 'Step'], df_smooth.loc[mask, 'GradNorm_Disc'], color='brown', linewidth=1.5, linestyle='-', label='Disc Grad')

    ax4.set_title('4. Gradient Norms (Stability Check)', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Grad Norm')
    ax4.set_xlabel('Global Steps')
    ax4.set_yscale('log')
    ax4.grid(True, linestyle='--', alpha=0.5)
    ax4.legend(loc='upper right')

    plt.suptitle(f'Training Progress - {os.path.basename(csv_path)}', fontsize=16)
    plt.show()
# %%

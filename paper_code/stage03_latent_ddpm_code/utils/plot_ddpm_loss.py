#%%
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#%%
# 1) 稳定不常改：数据读取 / 平滑 / 通用工具
def read_csv_safe(csv_path: str) -> pd.DataFrame:
    """读取 CSV 并做基础校验，避免空文件/列缺失导致脚本崩。"""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV 不存在: {csv_path}")
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"CSV 为空: {csv_path}")
    return df

def rolling_smooth(series: pd.Series, window: int) -> pd.Series:
    """滚动均值平滑（window<=1 时直接返回原序列）"""
    window = int(window)
    if window <= 1:
        return series
    return series.rolling(window=window, min_periods=1).mean()

def ensure_dir(path: str):
    """确保目录存在"""
    os.makedirs(path, exist_ok=True)

def set_matplotlib_style():
    """统一画图风格：尽量让论文/报告可读（一般不需要频繁改）"""
    plt.style.use("default")
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]
    plt.rcParams["axes.titleweight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"

def place_legend_outside(ax, loc="upper left", anchor=(1.02, 1.0), fontsize=9):
    """
    把 legend 放到子图右侧外面，避免遮挡曲线。
    anchor=(1.02,1.0) 表示紧贴右边界，顶部对齐。
    """
    ax.legend(loc=loc, bbox_to_anchor=anchor, borderaxespad=0.0, fontsize=fontsize, frameon=True)

# ============================================================
# 2) 稳定但偶尔会改：各类图的绘制函数（你后续只动 main 配置即可）
# ============================================================
def plot_training_log(ax, df: pd.DataFrame, smooth_window: int = 100, y_log: bool = True):
    """
    绘制 training_log.csv 类：Step vs Loss +（可选）Learning Rate（右轴）
    需要列：Step, Loss, LR（LR 可选）
    """
    required = {"Step", "Loss"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"training_log.csv 缺少列: {missing}")

    steps = df["Step"]
    loss_raw = df["Loss"]
    loss_smooth = rolling_smooth(loss_raw, smooth_window)

    # 左轴：Loss
    ax.plot(steps, loss_raw, alpha=0.15, linewidth=0.7, label="Loss (Raw)")
    ax.plot(steps, loss_smooth, linewidth=2.2, label=f"Loss (Smoothed, w={smooth_window})")

    ax.set_xlabel("Iteration Steps")
    ax.set_ylabel("Loss")
    ax.grid(True, which="both", ls="--", alpha=0.3)

    if y_log:
        ax.set_yscale("log")

    # 右轴：Learning Rate（如果有）
    if "LR" in df.columns:
        ax2 = ax.twinx()
        ax2.plot(steps, df["LR"], linestyle="--", linewidth=1.6, label="Learning Rate")
        ax2.set_ylabel("Learning Rate")
        # 合并 legend：把两个轴的句柄取出来再统一放到外侧
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend().remove() if ax.get_legend() else None
        ax2.legend().remove() if ax2.get_legend() else None
        ax._combined_legend = (lines1 + lines2, labels1 + labels2)  # 临时挂在 ax 上供外部统一处理
    else:
        ax._combined_legend = ax.get_legend_handles_labels()

def plot_region_log(ax, df: pd.DataFrame, smooth_alpha: float = None, y_log: bool = True):
    """
    绘制 region_loss_log_v2.csv 类：Step vs 多条 region loss + UnknownRatio（右轴）
    你现在的列名大概率是：
      Step, DiffLoss_unknown, DiffLoss_known, KnownConsistency, BoundaryX0Cons, UnknownRatio
    如果你的列名略有不同，在 main 的映射里改即可。
    """
    required = {"Step"}
    if not required.issubset(df.columns):
        raise ValueError(f"region_loss_log 缺少列: {required - set(df.columns)}")

    steps = df["Step"]

    alpha = 0.03
    df["BoundaryX0Cons_ema"] = df["BoundaryX0Cons"].ewm(alpha=alpha, adjust=False).mean()


    # 需要画的曲线（存在才画）
    candidates = [
        ("DiffLoss_unknown", "DiffLoss_unknown"),
        ("DiffLoss_known", "DiffLoss_known"),
        ("KnownConsistency", "KnownConsistency"),
        # ("BoundaryX0Cons", "BoundaryX0Cons"),
    ]

    # 左轴：loss 曲线
    for col, label in candidates:
        if col in df.columns:
            y = df[col]
            # 你日志里如果已经是 EMA 平滑的，可以不再 smooth
            ax.plot(steps, y, linewidth=2.0, label=label)

    ax.plot(df["Step"], df["BoundaryX0Cons"], alpha=0.15, linewidth=1.0, label="BoundaryX0Cons (Raw)")
    ax.plot(df["Step"], df["BoundaryX0Cons_ema"], linewidth=2.0, label=f"BoundaryX0Cons (EMA α={alpha})")


    ax.set_xlabel("Iteration Steps")
    ax.set_ylabel("Region Loss")
    ax.grid(True, which="both", ls="--", alpha=0.3)
    if y_log:
        ax.set_yscale("log")

    # 右轴：UnknownRatio（可选）
    if "UnknownRatio" in df.columns:
        ax2 = ax.twinx()
        ax2.plot(steps, df["UnknownRatio"], linestyle="--", linewidth=1.6, label="UnknownRatio")
        ax2.set_ylabel("UnknownRatio")
        ax2.set_ylim(0.0, 1.0)

        # 合并 legend
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend().remove() if ax.get_legend() else None
        ax2.legend().remove() if ax2.get_legend() else None
        ax._combined_legend = (lines1 + lines2, labels1 + labels2)
    else:
        ax._combined_legend = ax.get_legend_handles_labels()

#%%
# 3) 主逻辑：你后续最可能改的都集中在这里
def main():
    # ------------------【你常改的配置区】------------------
    training_csv = "E:\\chendou\\paper_code\\stage03_latent_ddpm_code\\exp_results\\exp0_LDM_l1_v1\\logs\\training_log.csv"            # 你的第一个 CSV
    region_csv = "E:\\chendou\\paper_code\\stage03_latent_ddpm_code\\exp_results\\exp0_LDM_l1_v1\\logs\\region_loss_log_v2.csv"        # 你的第二个 CSV
    save_path = "combined_curves.png"            # 输出图

    # 平滑窗口（training 的 loss raw -> smooth）
    smooth_window = 100

    # 是否用 log y（一般 loss 用 log 更好看）
    training_ylog = True
    region_ylog = True

    # legend 放置方式：
    # 方案 A：每个子图 legend 放右侧外面（推荐，最不遮挡）
    legend_mode = "outside_top"   # 可选：outside_right / outside_top
    # ------------------------------------------------------

    set_matplotlib_style()

    # 读取数据
    df_train = read_csv_safe(training_csv)
    df_region = read_csv_safe(region_csv)

    # 1 行 2 列：两张大图同一行
    # constrained_layout=True：比 tight_layout 更适合 twin axis 的情况
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), dpi=150, constrained_layout=True)

    ax1, ax2 = axes

    # 左图：训练 loss + lr
    plot_training_log(ax1, df_train, smooth_window=smooth_window, y_log=training_ylog)
    ax1.set_title("Latent Diffusion Training Dynamics")

    # 右图：region-wise 曲线
    plot_region_log(ax2, df_region, y_log=region_ylog)
    ax2.set_title("Region-wise Training Dynamics (EMA)")

    # -------- legend：默认放图外，避免遮挡 --------
    if legend_mode == "outside_right":
        # 子图右侧留白，让 legend 不被裁切
        # 如果你觉得右侧空间不够，可把 rect 的右边界再缩小一些
        fig.subplots_adjust(right=0.52)

        # 左图 legend
        lines, labels = getattr(ax1, "_combined_legend", ax1.get_legend_handles_labels())
        ax1.legend(lines, labels, loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=9, frameon=True)

        # 右图 legend
        lines, labels = getattr(ax2, "_combined_legend", ax2.get_legend_handles_labels())
        ax2.legend(lines, labels, loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=9, frameon=True)

    elif legend_mode == "outside_top":
        # 把所有 legend 合并放到整张图顶部（适合特别拥挤时）
        # 合并两个子图的 legend
        lines1, labels1 = getattr(ax1, "_combined_legend", ax1.get_legend_handles_labels())
        lines2, labels2 = getattr(ax2, "_combined_legend", ax2.get_legend_handles_labels())

        # 去重（按 label）
        seen = set()
        lines_all, labels_all = [], []
        for ln, lb in list(zip(lines1, labels1)) + list(zip(lines2, labels2)):
            if lb not in seen:
                seen.add(lb)
                lines_all.append(ln)
                labels_all.append(lb)

        fig.legend(lines_all, labels_all, loc="upper center", ncol=3, fontsize=9, frameon=True)
        fig.subplots_adjust(top=1.8)

    else:
        # 退化：在图内显示（不推荐，会遮挡）
        ax1.legend(fontsize=9)
        ax2.legend(fontsize=9)

    # 保存
    # plt.savefig(save_path, dpi=150)
    plt.show()
    plt.close()
    print(f"✅ Saved: {save_path}")

if __name__ == "__main__":
    main()

# %%

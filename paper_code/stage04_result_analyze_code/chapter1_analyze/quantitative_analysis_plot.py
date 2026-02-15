import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['figure.figsize'] = (10, 3.2) 
plt.rcParams['figure.dpi'] = 300               
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['mathtext.fontset'] = 'stix'      
plt.rcParams['font.size'] = 10.5               
plt.rcParams['axes.grid'] = True 

thesis_colors = ["#0072B2", "#D55E00", "#E69F00", "#009E73"] 

def plot_v4():
    try:
        with open("quantitative_metrics_v4.json", "r") as f:
            data = json.load(f)
    except:
        print("❌ 未找到数据文件")
        return

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2))
    plt.subplots_adjust(wspace=0.35, bottom=0.2)

    # ---------------------------------------------------------
    # (a) REV 分析 (Key 现在包含了岩心名，直接用作 Label)
    # ---------------------------------------------------------
    ax1 = axes[0]
    
    # 遍历字典 (key 例如: "6-6-24 (w192_s64)")
    for key, curve_list in data["rev"].items():
        base_sizes = curve_list[0]["sizes"]
        all_phis = []
        for c in curve_list:
            if len(c["sizes"]) == len(base_sizes):
                all_phis.append(c["phis"])
        
        if all_phis:
            avg_phi = np.mean(all_phis, axis=0) * 100
            # 直接画线，Label 会自动区分不同文件夹
            ax1.plot(base_sizes, avg_phi, marker='o', markersize=3, label=key)

    ax1.set_title("(a) REV Stability Analysis")
    ax1.set_xlabel("Cube Size ($voxel^3$)")
    ax1.set_ylabel("Porosity (%)")
    # 字体调小一点，防止 Key 太长遮挡
    ax1.legend(fontsize=7, loc='upper right')

    # ---------------------------------------------------------
    # (b) S2(r) (修改：遍历字典画多条线)
    # ---------------------------------------------------------
    ax2 = axes[1]
    if "s2" in data:
        # 遍历每个岩心的数据
        for core_name, curve_data in data["s2"].items():
            x = curve_data["x"]
            y = curve_data["y"]
            # 画线，Label 为岩心名 (如 6-6-21)
            ax2.plot(x, y, linewidth=2, label=core_name)
            
    ax2.set_title("(b) Two-point Correlation")
    ax2.set_xlabel("Distance $r$ (pixels)")
    ax2.set_ylabel("$S_2(r)$")
    ax2.legend(fontsize=7, loc='upper right')

    # ---------------------------------------------------------
    # (c) 批次一致性
    # ---------------------------------------------------------
    ax3 = axes[2]
    raw_peaks = [x for x in data["consistency"]["raw"] if 5000 < x < 65000] # 绘图时再过滤一下极端值让图好看
    proc_peaks = data["consistency"]["processed"]
    
    import pandas as pd
    df_raw = pd.DataFrame({"Intensity": raw_peaks, "Stage": "Raw CT"})
    df_proc = pd.DataFrame({"Intensity": proc_peaks, "Stage": "Processed"})
    df = pd.concat([df_raw, df_proc])
    
    # sns.boxplot(
    #     x="Stage", 
    #     y="Intensity", 
    #     data=df, 
    #     ax=ax3, 
    #     width=0.5, 
    #     palette=[thesis_colors[2], thesis_colors[0]], 
    #     linewidth=1.2, 
    #     showfliers=False) # 不显示离群点，让箱体对比更明显

    # 修改后 (推荐写法)
    sns.boxplot(
        x="Stage", 
        y="Intensity", 
        data=df, 
        ax=ax3, 
        width=0.5, 
        palette="Set2", 
        hue="Stage",     # 显式指定 hue
        legend=False,     # 隐藏冗余的图例
        linewidth=1.2, 
        showfliers=True
    )

    # 标注靶值
    ax3.axhline(35000, color=thesis_colors[1], linestyle=':', label="Target: 35000")
    
    ax3.set_title("(c) Batch Consistency")
    ax3.set_ylabel("Matrix Peak Intensity")
    ax3.legend(loc='lower right', fontsize=7)
    
    plt.savefig("Figure_Quantitative_Analysis_V4.png", dpi=300, bbox_inches='tight')
    print("✅ 绘图完成")

if __name__ == "__main__":
    plot_v4()
#%%
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

# ==========================================
# 配置区域
# ==========================================
# 1. 找一张【失败】文件夹里的图 (找中间层，别找开头全黑的)
path_fail = r"/chendou_space/data/core_ctimg_data/6-6-20 全部/FdkRecon-ushort-1900x1900x79205694.tif" 

# 2. 找一张【成功】文件夹里的图
path_success = r"/chendou_space/data/core_ctimg_data/6-6-21/FdkRecon-ushort-1900x1900x14328.modif2742.tif"

# 3. 你当前使用的全局参数
GLOBAL_P1 = 272.0
GLOBAL_P99 = 55532.0
# ==========================================

#%%
def read_and_calc(path, label):
    if not os.path.exists(path):
        print(f"❌ 找不到文件: {path}")
        return None, None
    
    # 读取原始数据
    raw_data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(raw_data, cv2.IMREAD_UNCHANGED)
    
    if img is None:
        print(f"❌ 读取失败: {path}")
        return None, None

    # 模拟归一化
    norm = (img.astype(np.float32) - GLOBAL_P1) / (GLOBAL_P99 - GLOBAL_P1 + 1e-6)
    norm = np.clip(norm, 0, 1)
    
    return img, norm

def plot_analysis(img_f, norm_f, img_s, norm_s):
    plt.figure(figsize=(15, 10))

    # --- 1. 原始直方图对比 ---
    plt.subplot(2, 2, 1)
    plt.title("Raw Histogram Comparison (16-bit)")
    
    # 扁平化并过滤背景0值，为了看清岩石分布
    data_f = img_f.flatten()
    data_f = data_f[data_f > 100] 
    
    data_s = img_s.flatten()
    data_s = data_s[data_s > 100]

    plt.hist(data_f, bins=100, range=(0, 65535), color='red', alpha=0.5, label='Failed Folder (6-6-20)', density=True)
    plt.hist(data_s, bins=100, range=(0, 65535), color='green', alpha=0.5, label='Success Folder (6-6-21)', density=True)
    
    # 画出 P99 线
    plt.axvline(GLOBAL_P99, color='blue', linestyle='--', label=f'Your Global P99 ({GLOBAL_P99})')
    plt.legend()
    plt.xlabel("Pixel Value (0-65535)")
    plt.ylabel("Frequency")

    # --- 2. 归一化后的图像效果对比 ---
    # 成功组
    plt.subplot(2, 2, 2)
    plt.title(f"Success Case (Processed)\nMean Val: {norm_s.mean():.3f}")
    plt.imshow(norm_s, cmap='gray', vmin=0, vmax=1)
    plt.axis('off')

    # 失败组
    plt.subplot(2, 2, 4)
    plt.title(f"Failed Case (Processed)\nMean Val: {norm_f.mean():.3f}")
    plt.imshow(norm_f, cmap='gray', vmin=0, vmax=1)
    plt.axis('off')

    # --- 3. 解释 ---
    plt.subplot(2, 2, 3)
    plt.axis('off')
    text = (
        "Analysis Guide:\n\n"
        "1. Look at the Histogram (Top-Left).\n"
        "2. If the Red curve (Failed) is far to the LEFT of the Blue line,\n"
        "   it means the rock is too dark for your P99 setting.\n"
        "   -> Result: The image becomes dark gray (See Bottom-Right).\n\n"
        "3. If the Red curve is far to the RIGHT,\n"
        "   it means the rock is saturated (Too bright).\n\n"
        "4. If the curves overlap, the issue is NOT normalization,\n"
        "   but maybe the image is empty or broken."
    )
    plt.text(0.1, 0.5, text, fontsize=12, family='monospace')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # 请务必先修改上面的文件路径！
    print("正在分析...")
    img_fail, norm_fail = read_and_calc(path_fail, "Failed")
    img_succ, norm_succ = read_and_calc(path_success, "Success")
    
    if img_fail is not None and img_succ is not None:
        plot_analysis(img_fail, norm_fail, img_succ, norm_succ)
        print("分析完成，请看图。")
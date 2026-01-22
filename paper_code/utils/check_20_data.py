#%%
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from skimage.restoration import estimate_sigma, denoise_nl_means
# 设置中文字体
# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP', 'DejaVu Sans'] # 优先使用中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


# ================= 配置区域 =================
# 1. 替换为 6-6-20 文件夹里的一张真实图片路径 (找中间层，别找开头结尾)
file_path = r"/chendou_space/data/core_ctimg_data/6-6-20 全部/FdkRecon-ushort-1900x1900x79205694.tif" 

# 2. 填入刚才程序检测到的 ROI 参数 (从你的日志里抄下来的)
CX, CY = 905, 970
RADIUS = 428

# 3. 填入你 Config 里的参数
GLOBAL_P1 = 272.0       # 之前统计的值
GLOBAL_P99 = 55532.0    # 之前统计的值
DENOISE_H = 4
SEVER_SIZE = 5    # 这是你当前的配置
# ========================================

#%%
def test_sdr_survival():
    if not os.path.exists(file_path):
        print("❌ 文件路径不对")
        return

    # 1. 读取 & 预处理 (快速重现)
    raw = np.fromfile(file_path, dtype=np.uint8)
    img = cv2.imdecode(raw, cv2.IMREAD_UNCHANGED)
    
    r_pad = int(RADIUS * 1.05)
    y1, y2 = max(0, CY - r_pad), min(img.shape[0], CY + r_pad)
    x1, x2 = max(0, CX - r_pad), min(img.shape[1], CX + r_pad)
    crop = img[y1:y2, x1:x2]
    
    h_roi, w_roi = crop.shape
    mask_local = np.zeros((h_roi, w_roi), dtype=np.uint8)
    cv2.circle(mask_local, (CX-x1, CY-y1), RADIUS, 1, -1)

    norm_f = (crop.astype(np.float32) - GLOBAL_P1) / (GLOBAL_P99 - GLOBAL_P1 + 1e-6)
    norm_f = np.clip(norm_f, 0, 1)
    
    # 简单降噪
    sigma = np.mean(estimate_sigma(norm_f))
    denoised = denoise_nl_means(norm_f, h=4*sigma, fast_mode=True)
    denoised[mask_local == 0] = 0
    
    # 二值化
    norm_8u = (denoised * 255).astype(np.uint8)
    _, b_mask = cv2.threshold(norm_8u, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # === 关键步骤：SDR 压力测试 ===
    kernel_sever = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (SEVER_SIZE, SEVER_SIZE))
    eroded = cv2.erode(b_mask, kernel_sever)
    
    # 连通域分析
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(eroded)
    
    print(f"原始 Mask 像素数: {np.count_nonzero(b_mask)}")
    print(f"腐蚀后 Mask 像素数: {np.count_nonzero(eroded)}")
    print(f"连通域数量: {num_labels - 1}") # 减去背景

    # 还原 (模拟 Pipeline 逻辑)
    final_mask = np.zeros_like(b_mask)
    if num_labels > 1:
        largest_idx = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        temp = (labels == largest_idx).astype(np.uint8)
        # 膨胀还原 (这里简化，只看是否存活)
        final_mask = temp * 255
        status = "✅ 存活 (SURVIVED)"
    else:
        status = "❌ 死亡 (KILLED)"

    # === 画图 ===
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 3, 1)
    plt.title("1. Binary Mask (Otsu)")
    plt.imshow(b_mask, cmap='gray')
    
    plt.subplot(1, 3, 2)
    plt.title(f"2. After Erode (Size={SEVER_SIZE})")
    plt.imshow(eroded, cmap='gray')
    
    plt.subplot(1, 3, 3)
    plt.title(f"3. Result: {status}")
    plt.imshow(final_mask, cmap='gray')
    plt.text(10, 30, status, color='red' if 'KILLED' in status else 'green', fontsize=14, weight='bold')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    test_sdr_survival()

# %%

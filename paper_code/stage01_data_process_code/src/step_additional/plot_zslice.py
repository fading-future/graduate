import os
import glob
import re
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

# ================= 配置区域 =================
# 你的二值化图像文件夹路径
SRC_FOLDER = r"D:\多尺度岩心数据集\Lastest_Preprocess\Binary_Preprocessed_Slices\6-6-24"
# 输出诊断图保存位置
OUTPUT_IMAGE = "Diagnostic_24_SideView_XZ.jpg"
# 采样位置 (切中间)
X_SLICE_POS = 427  # 假设图像宽 854，取中间 427
IMG_SIZE = 854
# ===========================================

def strict_sort_key(filepath):
    filename = os.path.basename(filepath)
    match = re.search(r'modif(\d+)', filename)
    if match: return int(match.group(1))
    return 0

def diagnose_continuity():
    print(f"🔍 正在扫描文件夹: {SRC_FOLDER}")
    
    # 1. 获取并排序文件
    files = glob.glob(os.path.join(SRC_FOLDER, "*.tif"))
    files = sorted(files, key=strict_sort_key)
    
    if not files:
        print("❌ 未找到 .tif 文件！请检查路径。")
        return

    print(f"✅ 找到 {len(files)} 个文件。")
    print("⏳ 正在检查序号连续性...")
    
    # 2. 检查序号是否有缺失
    missing_indices = []
    prev_idx = strict_sort_key(files[0])
    for i in range(1, len(files)):
        curr_idx = strict_sort_key(files[i])
        if curr_idx != prev_idx + 1:
            missing_indices.append((prev_idx, curr_idx))
        prev_idx = curr_idx
    
    if missing_indices:
        print(f"⚠️ 警告: 发现 {len(missing_indices)} 处序号不连续！(可能是割裂的原因)")
        for start, end in missing_indices[:5]:
            print(f"   - 在 {start} 和 {end} 之间缺失")
        if len(missing_indices) > 5: print("   - ...")
    else:
        print("✅ 文件序号完全连续 (0, 1, 2...)，无缺失。")

    # 3. 生成 XZ 侧视图 (纵向切片)
    print(f"⏳ 正在生成侧视图 (X={X_SLICE_POS})... 这可能需要一点时间")
    
    xz_slice_pixels = []
    
    # 为了速度，每 500 张打印一次进度
    for i, p in enumerate(tqdm(files)):
        try:
            # 只读一行像素 (利用 np.fromfile 读取全部再 reshape 可能会慢，但对于 tif 只能这样)
            # 优化：我们不需要解码整张图，但 opencv 需要。
            # 这里我们只取中间那一列
            
            # 支持中文路径读取
            img_arr = np.fromfile(p, dtype=np.uint8)
            img = cv2.imdecode(img_arr, cv2.IMREAD_GRAYSCALE)
            
            if img is None:
                # 读失败补黑线
                col = np.zeros(IMG_SIZE, dtype=np.uint8) # 假设高度 1900
            else:
                h, w = img.shape
                # 确保切片位置在范围内
                target_x = min(X_SLICE_POS, w-1)
                # 取第 target_x 列
                col = img[:, target_x]
            
            xz_slice_pixels.append(col)
            
        except Exception as e:
            print(f"读图错误: {p} - {e}")
            xz_slice_pixels.append(np.zeros(IMG_SIZE, dtype=np.uint8))

    # 4. 堆叠并保存
    # xz_slice_pixels 是 list of columns (H,)
    # stack 后变成 (Z, H) -> 转置一下变成 (H, Z) 符合直觉
    side_view = np.array(xz_slice_pixels).T 
    
    # 二值化显示 (0, 255)
    side_view = (side_view > 127).astype(np.uint8) * 255

    print(f"💾 正在保存诊断图: {OUTPUT_IMAGE}")
    cv2.imwrite(OUTPUT_IMAGE, side_view)
    
    # 缩小显示在屏幕上 (如果支持)
    plt.figure(figsize=(12, 6))
    plt.title(f"XZ Side View (Raw Data Check) - X={X_SLICE_POS}")
    # 因为 Z 轴很长，我们旋转一下看，或者压缩 Z 轴
    plt.imshow(side_view, cmap='gray', aspect='auto')
    plt.xlabel("Z (Slice Index)")
    plt.ylabel("Y (Height)")
    plt.tight_layout()
    plt.show()
    
    print("✅ 完成。请打开生成的 Diagnostic_SideView_XZ.jpg 查看。")
    print("👉 如果这张图上有水平割裂线，说明是【原始数据】的问题，不是代码的问题。")

if __name__ == "__main__":
    diagnose_continuity()
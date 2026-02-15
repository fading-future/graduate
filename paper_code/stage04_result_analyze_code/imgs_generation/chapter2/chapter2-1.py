import os
import glob
import re
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ================= 论文统一绘图配置 =================
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.size'] = 10.5
plt.rcParams['axes.labelsize'] = 10.5
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['axes.titlesize'] = 11
plt.rcParams['axes.linewidth'] = 1.0
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['axes.grid'] = True
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

thesis_colors = [
    "#0072B2", "#D55E00", "#E69F00", "#009E73", 
    "#56B4E9", "#F0E442", "#CC79A7", "#333333"
]
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=thesis_colors)
# ================= 配置结束 =================

CONFIG = {
    'src_root': r"D:\多尺度岩心数据集",
    'target_folders': ["6-6-21", "6-6-22", "6-6-24"],
}

# ================= 新增：手动指定切片文件名 =================
# 请将这里的 .tif 名称替换为您在每个文件夹里挑出的最满意的切片文件名
MANUAL_FILES = {
    # "6-6-18": "FdkRecon-ushort-1900x1900x10780.modif2437.tif",  # 示例：假设您选了这张
    "6-6-21": "FdkRecon-ushort-1900x1900x14328.modif1254.tif",
    "6-6-22": "FdkRecon-ushort-1900x1900x9624.modif1745.tif",
    "6-6-24": "FdkRecon-ushort-1900x1900x9624.modif0969.tif"   # 给 6-6-24 挑一张完整没破碎的
}

def _read_img_raw(path):
    try:
        raw_data = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(raw_data, cv2.IMREAD_UNCHANGED)
        return img
    except:
        return None

def get_single_roi(img):
    """
    针对单张图片计算完美贴合的 ROI，解决因岩心倾斜导致的全局圈偏移问题
    """
    scale = 0.2
    h, w = img.shape
    small = cv2.resize(img, (int(w*scale), int(h*scale)))
    mi, ma = small.min(), small.max()
    
    # 归一化并二值化
    small_8bit = ((small - mi) / (ma - mi + 1e-6) * 255).astype(np.uint8)
    _, thresh = cv2.threshold(small_8bit, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 形态学去噪 (确保外接圆不会因为小的外部噪点而被过度撑大)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        c = max(contours, key=cv2.contourArea)
        (x, y), r = cv2.minEnclosingCircle(c)
        return (int(x / scale), int(y / scale)), int(r / scale)
        
    return (w//2, h//2), min(w, h)//3 # 保底返回中心区域

def get_manual_slice_with_mask(folder_name):
    folder_path = os.path.join(CONFIG['src_root'], folder_name)
    target_filename = MANUAL_FILES.get(folder_name)
    
    if not target_filename:
        print(f"警告：未在 MANUAL_FILES 中找到 {folder_name} 的配置。")
        return None, None, None
        
    target_path = os.path.join(folder_path, target_filename)
    img = _read_img_raw(target_path)
    if img is None:
        print(f"读取失败，请检查路径: {target_path}")
        return None, None, None
        
    # 1. 实时计算当前切片的专属中心和半径
    (cx, cy), r = get_single_roi(img)
    
    # 2. 统一画布机制 (彻底解决子图大小不一致问题)
    canvas_size = 1900
    canvas = np.zeros((canvas_size, canvas_size), dtype=np.uint16)
    
    h, w = img.shape
    y_off = (canvas_size - h) // 2
    x_off = (canvas_size - w) // 2
    canvas[y_off:y_off+h, x_off:x_off+w] = img
    
    # 3. 修正中心坐标以适应新画布
    cx_new = cx + x_off
    cy_new = cy + y_off
    
    # 4. 生成 Mask
    mask = np.zeros((canvas_size, canvas_size), dtype=np.uint8)
    # 收缩一点点半径 (r * 0.98)，避免把边缘的管壁高亮像素统计进去
    cv2.circle(mask, (int(cx_new), int(cy_new)), int(r * 0.98), 1, -1)
    
    return canvas, mask, ((cx_new, cy_new), r * 0.98)

def plot_figure_x1():
    print("正在绘制修复偏移后的增强版 图 x.1 ...")
    folders = [f for f in CONFIG['target_folders'] if os.path.exists(os.path.join(CONFIG['src_root'], f))]
    n_folders = len(folders)
    if n_folders == 0: return
        
    fig, axes = plt.subplots(2, n_folders, figsize=(2.2 * n_folders, 5.0))
    global_reference_peak = None
    
    for i, folder_name in enumerate(folders):
        canvas_img, mask, roi_info = get_manual_slice_with_mask(folder_name)
        label_name = folder_name.replace(" 全部", "")
        
        if canvas_img is not None and mask is not None:
            (cx, cy), r = roi_info
            
            # ================= 第一行：原图 + 自适应 ROI 圈 =================
            axes[0, i].imshow(canvas_img, cmap='gray', vmin=0, vmax=65535)
            
            circle = patches.Circle((cx, cy), r, edgecolor='#D55E00', facecolor='none', linewidth=1.5, linestyle='--')
            axes[0, i].add_patch(circle)
            
            axes[0, i].set_title(f'Core {label_name}', fontsize=11)
            axes[0, i].axis('off')
            axes[0, i].set_xlim(0, 1900)
            axes[0, i].set_ylim(1900, 0) 
            
            # ================= 第二行：有效像素直方图 =================
            valid_pixels = canvas_img[mask == 1]
            valid_pixels = valid_pixels[valid_pixels > 100].flatten()
            
            counts, bins = np.histogram(valid_pixels, bins=100, range=(100, 65535))
            current_peak = bins[np.argmax(counts)]
            
            if i == 0:
                global_reference_peak = current_peak
            
            axes[1, i].hist(valid_pixels, bins=100, range=(100, 65535), density=True, 
                            alpha=0.8, color=thesis_colors[0])
            
            if global_reference_peak is not None:
                axes[1, i].axvline(x=global_reference_peak, color='#D55E00', 
                                   linestyle='--', linewidth=1.5, zorder=5)
            
            axes[1, i].set_xlim(0, 65535) 
            axes[1, i].set_xlabel('Intensity (16-bit)')
            
            if i == 0:
                axes[1, i].set_ylabel('Density')
            else:
                axes[1, i].set_yticks([])

    plt.tight_layout()
    plt.savefig('Figure_x1_Histogram_Shift_PerfectFit.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("图件生成完毕！")

if __name__ == "__main__":
    plot_figure_x1()
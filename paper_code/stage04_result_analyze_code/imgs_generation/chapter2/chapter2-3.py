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

# ================= 严格复刻源文件的配置 =================
CONFIG = {
    'global_p1': 192.0,     
    'global_p99': 57256.0, 
    'src_root': r"D:\多尺度岩心数据集",  # 请确认您的真实路径
    'target_folders': ["6-6-22"]
}

MANUAL_FILES = {
    # "6-6-18": "FdkRecon-ushort-1900x1900x10780.modif2437.tif",  # 示例：假设您选了这张
    # "6-6-21": "FdkRecon-ushort-1900x1900x14328.modif0680.tif",
    "6-6-22": "FdkRecon-ushort-1900x1900x9624.modif1895.tif",  # 给 6-6-22 挑一张完整没破碎的
    # "6-6-24": "FdkRecon-ushort-1900x1900x9624.modif0873.tif"   # 给 6-6-24 挑一张完整没破碎的
}

def _read_img_raw(path):
    try:
        raw_data = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(raw_data, cv2.IMREAD_UNCHANGED)
        return img
    except:
        return None

def get_single_roi(img):
    scale = 0.2
    h, w = img.shape
    small = cv2.resize(img, (int(w*scale), int(h*scale)))
    mi, ma = small.min(), small.max()
    small_8bit = ((small - mi) / (ma - mi + 1e-6) * 255).astype(np.uint8)
    _, thresh = cv2.threshold(small_8bit, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        c = max(contours, key=cv2.contourArea)
        (x, y), r = cv2.minEnclosingCircle(c)
        return (int(x / scale), int(y / scale)), int(r / scale)
    return (w//2, h//2), min(w, h)//3

def plot_figure_x3_strict():
    print("正在按照 ctimg2npy_v5.py 严格复刻 2x3 矩阵 ...")
    
    folder_name = CONFIG['target_folders'][0]
    folder_path = os.path.join(CONFIG['src_root'], folder_name)
    target_filename = MANUAL_FILES.get(folder_name)
    
    if not target_filename: 
        print("❌ 错误: 未找到配置文件！")
        return
        
    img_path = os.path.join(folder_path, target_filename)
    img = _read_img_raw(img_path)
    if img is None: 
        print(f"❌ 读取失败: {img_path}")
        return

    (cx, cy), r = get_single_roi(img)
    h, w = img.shape
    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - cx)**2 + (Y - cy)**2)
    mask = dist_from_center <= r
    
    valid_mask = (mask) & (img > 100)
    
    # [A] Original Image (16-bit)
    orig_img = np.zeros_like(img, dtype=np.float32)
    orig_img[valid_mask] = img[valid_mask]
    
    # [B] HE Artifact (为了对比效果，依然保留 8-bit 的梳子状 HE)
    img_8u = np.zeros_like(img, dtype=np.uint8)
    local_p1 = np.percentile(img[valid_mask], 1)
    local_p99 = np.percentile(img[valid_mask], 99)
    img_8u[valid_mask] = np.clip((img[valid_mask] - local_p1) / (local_p99 - local_p1 + 1e-6) * 255, 0, 255).astype(np.uint8)
    he_pixels_8u = cv2.equalizeHist(img_8u[valid_mask])
    # 强行映射回 16-bit 的范围以便与原图对比
    he_pixels_16u = he_pixels_8u.flatten().astype(np.float32) / 255.0 * 65535.0
    he_img = np.zeros_like(img, dtype=np.float32)
    he_img[valid_mask] = he_pixels_16u
    
    # [C] 严格复刻 ctimg2npy_v5.py 中的 Soft-Tanh 逻辑
    slice_data = img.astype(np.float32)
    g_p1 = CONFIG['global_p1']
    g_p99 = CONFIG['global_p99']
    
    mid_val = (g_p1 + g_p99) / 2.0
    half_range = (g_p99 - g_p1) / 2.0 + 1e-6
    squeeze_factor = 2.0
    
    norm_temp = (slice_data - mid_val) / half_range
    norm_temp = np.tanh(norm_temp * squeeze_factor)
    norm_f = (norm_temp + 1) / 2.0
    norm_f = np.clip(norm_f, 0, 1)
    
    # 必须乘以 65535 映射回 16-bit 空间 (这是源文件最精妙的一步)
    st_img_full = (norm_f * 65535).astype(np.float32)
    st_img = np.zeros_like(img, dtype=np.float32)
    st_img[valid_mask] = st_img_full[valid_mask]

    # ================= 绘图开始 (2行3列) =================
    fig, axes = plt.subplots(2, 3, figsize=(10, 6.5)) 
    
    images = [orig_img, he_img, st_img]
    titles_img = ['Original CT Slice', 'Histogram Equalization (HE)', 'Soft-Tanh Normalization']
    
    # 全部统一在真实的 16-bit 物理范围内渲染！
    vmins = [0, 0, 0]
    vmaxs = [65535, 65535, 65535] 
    
    for i in range(3):
        crop_r = int(r * 1.05) 
        y1, y2 = max(0, cy - crop_r), min(h, cy + crop_r)
        x1, x2 = max(0, cx - crop_r), min(w, cx + crop_r)
        
        cropped_img = images[i][y1:y2, x1:x2]
        
        axes[0, i].imshow(cropped_img, cmap='gray', vmin=vmins[i], vmax=vmaxs[i])
        axes[0, i].set_title(titles_img[i], pad=10)
        axes[0, i].axis('off')
        
        circle = patches.Circle((cx - x1, cy - y1), r, edgecolor='#D55E00', facecolor='none', linewidth=1.5, linestyle='--')
        axes[0, i].add_patch(circle)
        axes[0, i].set_xlim(0, crop_r * 2)
        axes[0, i].set_ylim(crop_r * 2, 0)

    # 第二行：全部使用 16-bit X轴 (0 - 65535)
    axes[1, 0].hist(img[valid_mask], bins=120, range=(100, 65535), color=thesis_colors[2], alpha=0.85)
    axes[1, 0].set_title('Original Distribution')
    axes[1, 0].set_xlabel('Intensity (16-bit)')
    axes[1, 0].set_ylabel('Frequency')
    
    axes[1, 1].hist(he_pixels_16u, bins=120, range=(100, 65535), color=thesis_colors[1], alpha=0.85)
    axes[1, 1].set_title('Classic HE Artifact')
    axes[1, 1].set_xlabel('Intensity (16-bit)')
    max_y = axes[1, 1].get_ylim()[1]  # 获取 Y 轴最大值

    axes[1, 1].annotate(
        'Physical density\ncontrast destroyed', 
        xy=(30000, max_y * 0.8),      # 靶心：真实的 X 数据为 40000，Y 是最大值的一半
        xytext=(40000, max_y * 0.98),  # 文本：X 数据 10000 处，Y 是最大值的 80% 处
        # 不写 xycoords，默认就是 'data'
        arrowprops=dict(
            facecolor='black', 
            shrink=0.05,
            width=1.0, 
            headwidth=5
        )
    )
    
    axes[1, 2].hist(st_img[valid_mask], bins=120, range=(100, 65535), color=thesis_colors[0], alpha=0.85)
    axes[1, 2].set_title('Soft-Tanh Distribution')
    axes[1, 2].set_xlabel('Intensity (16-bit)')
    
    max_freq = max(axes[1, 1].get_ylim()[1], axes[1, 2].get_ylim()[1])
    axes[1, 1].set_ylim(0, max_freq * 1.1)
    axes[1, 2].set_ylim(0, max_freq * 1.1)
    
    plt.tight_layout()
    plt.savefig('Figure_x3_Full_Comparison_Matrix_Strict.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ 完美复刻版生成完毕！")

if __name__ == "__main__":
    plot_figure_x3_strict()
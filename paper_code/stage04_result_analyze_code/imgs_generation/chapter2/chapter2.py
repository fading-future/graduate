import os
import glob
import re
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage.restoration import denoise_nl_means, estimate_sigma

# ================= 论文统一绘图配置 =================
plt.rcParams['figure.figsize'] = (3.5, 2.625)
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.size'] = 10.5
plt.rcParams['axes.labelsize'] = 10.5
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['axes.titlesize'] = 11
plt.rcParams['axes.linewidth'] = 1.0
plt.rcParams['lines.linewidth'] = 1.5
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

# 全局参数与路径配置
CONFIG = {
    'global_p1': 192.0,     
    'global_p99': 57256.0, 
    'src_root': r"D:\多尺度岩心数据集",
    'target_folders': ["6-6-18", "6-6-21", "6-6-22", "6-6-24"],
    'denoise_h': 4
}

def natural_sort_key(filepath):
    filename = os.path.basename(filepath)
    numbers = re.findall(r'\d+', filename)
    if numbers:
        return int(numbers[-1]) 
    return 0

def _read_img_raw(path):
    try:
        raw_data = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(raw_data, cv2.IMREAD_UNCHANGED)
        return img
    except:
        return None

def _detect_global_roi(files, sample_count=100):
        """
        改进版 V3.2: 智能 ROI 探测
        1. 过滤纯黑/空气切片
        2. 增加采样鲁棒性
        3. 优先取中间切片
        """
        if not files: return None
        print(f"正在智能探测 ROI (文件数: {len(files)})...")
        
        # 策略 A: 优先扫描文件列表“中间 20%”的区域
        # 岩心通常肯定在中间，两头可能是空气
        mid_start = int(len(files) * 0.4)
        mid_end = int(len(files) * 0.6)
        # 确保至少有切片
        if mid_end <= mid_start: mid_start, mid_end = 0, len(files)
        
        # 在中间区域密集采样
        indices = np.linspace(mid_start, mid_end, sample_count, dtype=int)
        
        valid_rois = [] # 存储所有检测到的候选 (x, y, r)

        for idx in indices:
            img = _read_img_raw(files[idx])
            if img is None: continue
            
            # --- 核心改进 1: 亮度门控 ---
            # 计算图片均值。如果整张图太黑（空气），直接跳过
            # 16-bit图，岩石通常 > 20000。空气通常 < 5000。
            # 这里设个保守阈值 2000，防止把噪声当岩石
            if np.mean(img) < 2000: 
                continue

            # 缩放加速
            scale = 0.2
            h, w = img.shape
            small = cv2.resize(img, (int(w*scale), int(h*scale)))
            
            mi, ma = small.min(), small.max()
            # 再次防噪: 如果最大值和最小值差太小，说明是纯色图
            if ma - mi < 100: continue 
            
            # 归一化到 0-255
            small_8bit = ((small - mi) / (ma - mi + 1e-6) * 255).astype(np.uint8)
            
            # 二值化
            _, thresh = cv2.threshold(small_8bit, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # 形态学去噪 (开运算去掉小白点)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
            
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # 找面积最大的轮廓
                c = max(contours, key=cv2.contourArea)
                
                # --- 核心改进 2: 面积过滤 ---
                # 如果最大轮廓太小（比如只是个噪点），跳过
                if cv2.contourArea(c) < 500: continue

                (x, y), r = cv2.minEnclosingCircle(c)
                
                # 映射回原图
                orig_x, orig_y, orig_r = x / scale, y / scale, r / scale
                
                # --- 核心改进 3: 居中验证 ---
                # 真正的岩心应该大概在图像中心
                img_cx, img_cy = w / 2, h / 2
                dist_to_center = np.sqrt((orig_x - img_cx)**2 + (orig_y - img_cy)**2)
                
                # 如果圆心偏离图像中心太远（超过 1/3 图像宽度），认为是噪声
                if dist_to_center > w / 3:
                    continue
                    
                valid_rois.append((orig_x, orig_y, orig_r))

        if not valid_rois:
            print(f"❌ 警告: 在 {sample_count} 次采样中未找到任何有效岩心！")
            print("   原因可能是：1.文件命名排序混乱导致没采到中间；2.岩心对比度极低。")
            # 最后的保底：返回 None，让主程序跳过或报错
            return None

        # 统计中位数，过滤离群值
        valid_rois = np.array(valid_rois)
        avg_center = np.median(valid_rois[:, :2], axis=0).astype(int)
        max_radius = int(np.percentile(valid_rois[:, 2], 90)) # 取较大半径确保覆盖
        
        print(f"✅ 探测成功 (基于 {len(valid_rois)} 个有效切片): 中心{avg_center}, 半径{max_radius}")
        return avg_center, max_radius

def get_middle_slice_with_mask(folder_path):
    """获取中间切片并生成对应的岩心圆形 Mask"""
    files = sorted(glob.glob(os.path.join(folder_path, "*.tif")), key=natural_sort_key)
    if not files: return None, None, None
    
    # 提取 ROI 掩码参数
    roi_info = _detect_global_roi(files, sample_count=500)
    if not roi_info: return None, None, None
    (cx, cy), r = roi_info
    
    img = _read_img_raw(files[len(files)//2])
    if img is None: return None, None, None
    
    # 生成二值 Mask
    mask = np.zeros(img.shape, dtype=np.uint8)
    cv2.circle(mask, (cx, cy), r, 1, -1)
    
    return img, mask, (cx, cy)

def plot_figure_x1():
    """图 x.1：应用 ROI Mask 后不同批次的岩心像素分布直方图及原图对比"""
    print("正在绘制 图 x.1: 不同批次岩心亮度分布直方图 (包含原图与直方图矩阵)...")
    
    n_folders = len(CONFIG['target_folders'])
    # 采用通栏图尺寸 (宽 7 英寸)，适配 2 行 N 列的子图布局
    fig, axes = plt.subplots(2, n_folders, figsize=(7.5, 4.5))
    
    for i, folder_name in enumerate(CONFIG['target_folders']):
        folder_path = os.path.join(CONFIG['src_root'], folder_name)
        img, mask, _ = get_middle_slice_with_mask(folder_path)
        label_name = folder_name.replace(" 全部", "")
        
        if img is not None and mask is not None:
            # ================= 第一行：绘制岩心 CT 图像 =================
            # 拷贝原图并对 Mask 外的空气背景置 0，以便更清晰地展示岩心本身
            display_img = img.copy()
            display_img[mask == 0] = 0 
            
            axes[0, i].imshow(display_img, cmap='gray')
            axes[0, i].set_title(f'Core {label_name}', fontsize=11)
            axes[0, i].axis('off') # 关闭图像的坐标轴
            
            # ================= 第二行：绘制有效像素直方图 =================
            # 仅提取 Mask 范围内的像素，同时滤除空气噪点
            valid_pixels = img[mask == 1]
            valid_pixels = valid_pixels[valid_pixels > 100].flatten()
            
            axes[1, i].hist(valid_pixels, bins=100, range=(100, 65535), density=True, 
                            alpha=0.8, color=thesis_colors[i])
            
            # 强制固定横轴范围，凸显不同批次之间的亮度(灰度)漂移
            axes[1, i].set_xlim(0, 65535) 
            axes[1, i].set_xlabel('Intensity (16-bit)')
            
            # 调整 Y 轴显示以保持版面整洁
            if i == 0:
                axes[1, i].set_ylabel('Density')
            else:
                # 隐藏非第一列子图的 Y 轴刻度标签，避免拥挤
                axes[1, i].set_yticks([])

    # 调整子图间距
    plt.tight_layout()
    plt.savefig('Figure_x1_Histogram_Shift.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_figure_x3():
    """图 x.3：基于 ROI 区域的 Hard Truncation 与 Soft-Tanh 方法直方图对比"""
    print("正在绘制 图 x.3: Hard Truncation vs Soft-Tanh 直方图...")
    folder_path = os.path.join(CONFIG['src_root'], CONFIG['target_folders'][1])
    img, mask, _ = get_middle_slice_with_mask(folder_path)
    
    if img is None: return

    slice_data = img.astype(np.float32)
    
    # 严谨获取掩码内部的有效体素像素
    valid_pixels_raw = slice_data[mask == 1]
    valid_pixels_raw = valid_pixels_raw[valid_pixels_raw > 100]
    
    # 1. 传统硬截断
    hard_trunc = np.clip((valid_pixels_raw - CONFIG['global_p1']) / 
                         (CONFIG['global_p99'] - CONFIG['global_p1']), 0, 1)
    
    # 2. 软挤压 Soft-Tanh 逻辑
    mid_val = (CONFIG['global_p1'] + CONFIG['global_p99']) / 2.0
    half_range = (CONFIG['global_p99'] - CONFIG['global_p1']) / 2.0 + 1e-6
    squeeze_factor = 2.0
    
    norm_temp = (valid_pixels_raw - mid_val) / half_range
    norm_temp = np.tanh(norm_temp * squeeze_factor)
    soft_tanh_f = (norm_temp + 1) / 2.0
    soft_tanh_f = np.clip(soft_tanh_f, 0, 1)

    fig, axes = plt.subplots(1, 2, figsize=(7, 2.625)) 
    
    axes[0].hist(hard_trunc, bins=100, range=(0, 1), color=thesis_colors[1], alpha=0.8)
    axes[0].set_title('Hard Truncation (Linear Clip)')
    axes[0].set_xlabel('Normalized Value')
    axes[0].set_ylabel('Frequency')
    
    axes[1].hist(soft_tanh_f, bins=100, range=(0, 1), color=thesis_colors[0], alpha=0.8)
    axes[1].set_title('Soft-Tanh Normalization')
    axes[1].set_xlabel('Normalized Value')
    
    plt.tight_layout()
    plt.savefig('Figure_x3_SoftTanh_Comparison.png', dpi=300)
    plt.close()

def plot_figure_x4():
    """图 x.4：包含背景屏蔽逻辑的 NLM 算法局部效果对比"""
    print("正在绘制 图 x.4: NLM vs Median Filter 图像效果对比...")
    folder_path = os.path.join(CONFIG['src_root'], CONFIG['target_folders'][1])
    img, mask, center = get_middle_slice_with_mask(folder_path)
    
    if img is None: return
    cx, cy = center

    slice_data = img.astype(np.float32)
    mid_val = (CONFIG['global_p1'] + CONFIG['global_p99']) / 2.0
    half_range = (CONFIG['global_p99'] - CONFIG['global_p1']) / 2.0 + 1e-6
    squeeze_factor = 2.0
    
    norm_temp = np.tanh(((slice_data - mid_val) / half_range) * squeeze_factor)
    norm_f = np.clip((norm_temp + 1) / 2.0, 0, 1)
    
    # ================= 关键同步 =================
    # 在滤波前后严格把 Mask 以外的空气背景设置为 0，这与您的切片制作逻辑完全一致
    norm_f[mask == 0] = 0

    # 基于真实的岩心中心(cx, cy)进行裁剪，这样无论图片怎么偏移，裁剪区一定落在岩心内部
    crop_roi = norm_f[cy-100:cy+100, cx:cx+200]

    # 1. 传统中值滤波 
    median_filtered = ndimage.median_filter(crop_roi, size=5)

    # 2. NLM 滤波
    sigma_est = np.mean(estimate_sigma(crop_roi))
    if sigma_est > 0:
        nlm_filtered = denoise_nl_means(crop_roi, h=CONFIG['denoise_h'] * sigma_est, 
                                        fast_mode=True, patch_size=5, patch_distance=6)
    else:
        nlm_filtered = crop_roi

    fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.5))
    
    axes[0].imshow(crop_roi, cmap='gray')
    axes[0].set_title('Original (Noisy)')
    axes[0].axis('off')
    
    axes[1].imshow(median_filtered, cmap='gray')
    axes[1].set_title('Median Filter')
    axes[1].axis('off')
    
    axes[2].imshow(nlm_filtered, cmap='gray')
    axes[2].set_title('NLM Filter')
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig('Figure_x4_NLM_vs_Median.png', dpi=300)
    plt.close()

if __name__ == "__main__":
    print("开始生成论文配图...")
    plot_figure_x1()
    plot_figure_x3()
    plot_figure_x4()
    print("全部图件已生成并保存在当前目录下！")
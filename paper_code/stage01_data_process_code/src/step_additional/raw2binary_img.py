import os

# ================= 性能配置 =================
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["CV_NUM_THREADS"] = "1"

import glob
import re
import cv2
import numpy as np
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
from skimage.restoration import denoise_nl_means, estimate_sigma

# ================= 配置区域 =================
INPUT_DIR = r"D:\多尺度岩心数据集\6-6-21"
OUTPUT_DIR = r"D:\多尺度岩心数据集\6-6-21_Global_Consistency" # 输出目录改名
FILE_EXT = "*.tif"

CONFIG = {
    'roi_sample_count': 500,
    'global_otsu_sample': 1200, # 计算全局阈值时的采样数
    'squeeze_factor': 2.0,    
    'denoise_h': 4,           
    'min_fragment_size': 50, 
    'morph_open_size': 1,     
}

# ================= 工具函数 =================

def natural_sort_key(filepath):
    filename = os.path.basename(filepath)
    numbers = re.findall(r'\d+', filename)
    return int(numbers[-1]) if numbers else 0

def _read_img_raw(filepath):
    try:
        img_array = np.fromfile(filepath, dtype=np.uint8)
        return cv2.imdecode(img_array, cv2.IMREAD_UNCHANGED)
    except:
        return None

def _save_img_safe(filepath, img):
    try:
        cv2.imencode(".tif", img)[1].tofile(filepath)
    except:
        pass

# ================= 1. ROI & 统计 =================

def detect_global_roi_v3(files, sample_count=100):
    print(f"Step 1: 智能探测 ROI...")
    mid_start = int(len(files) * 0.4)
    mid_end = int(len(files) * 0.6)
    if mid_end <= mid_start: mid_start, mid_end = 0, len(files)
    indices = np.linspace(mid_start, mid_end, min(sample_count, len(files)), dtype=int)
    indices = np.unique(indices)
    
    valid_rois = [] 
    for idx in tqdm(indices, desc="ROI Sampling", leave=False):
        img = _read_img_raw(files[idx])
        if img is None or np.mean(img) < 2000: continue

        scale = 0.2
        h, w = img.shape
        small = cv2.resize(img, (int(w*scale), int(h*scale)))
        mi, ma = small.min(), small.max()
        if ma - mi < 100: continue 
        
        small_8bit = ((small - mi) / (ma - mi + 1e-6) * 255).astype(np.uint8)
        _, thresh = cv2.threshold(small_8bit, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            c = max(contours, key=cv2.contourArea)
            if cv2.contourArea(c) < 500: continue
            (x, y), r = cv2.minEnclosingCircle(c)
            orig_x, orig_y, orig_r = x / scale, y / scale, r / scale
            img_cx, img_cy = w / 2, h / 2
            if np.sqrt((orig_x - img_cx)**2 + (orig_y - img_cy)**2) > w / 3: continue
            valid_rois.append((orig_x, orig_y, orig_r))

    if not valid_rois:
        dummy = _read_img_raw(files[0])
        h, w = dummy.shape
        return (w//2, h//2), w//4

    valid_rois = np.array(valid_rois)
    avg_center = np.median(valid_rois[:, :2], axis=0).astype(int)
    max_radius = int(np.percentile(valid_rois[:, 2], 90)) 
    print(f"✅ ROI 锁定: Center={avg_center}, Radius={max_radius}")
    return avg_center, max_radius

def get_global_brightness_stats(files, center, radius):
    print("Step 2: 计算全局亮度 P1/P99...")
    cx, cy = center
    r = radius
    pixel_reservoir = []
    # 随机采样 50 张
    sample_files = random.sample(files, min(len(files), 50))
    for f in sample_files:
        img = _read_img_raw(f)
        if img is None: continue
        h, w = img.shape
        y1, y2 = max(0, cy-r), min(h, cy+r)
        x1, x2 = max(0, cx-r), min(w, cx+r)
        crop = img[y1:y2, x1:x2]
        vals = crop[crop > 500] 
        if len(vals) > 5000: vals = np.random.choice(vals, 5000)
        pixel_reservoir.append(vals)
        
    if not pixel_reservoir: return 0, 65535
    all_p = np.concatenate(pixel_reservoir)
    return np.percentile(all_p, 1), np.percentile(all_p, 99)

# ================= NEW: 计算全局统一 Otsu 基准值 =================

def calculate_global_otsu_base(files, center, radius, p1, p99):
    print("Step 2.5: 正在计算【全局统一】Otsu 基准阈值 (消除层间割裂)...")
    
    # 随机抽取 150 张图，把它们的像素混合在一起算直方图
    sample_indices = np.linspace(0, len(files)-1, CONFIG['global_otsu_sample'], dtype=int)
    pixel_reservoir = []
    
    cx, cy = center
    r = radius
    out_size = 2 * r
    
    # 预计算 Soft-Tanh 参数
    mid_val = (p1 + p99) / 2.0
    half_range = (p99 - p1) / 2.0 + 1e-6
    
    for idx in tqdm(sample_indices, desc="Global Sampling"):
        img = _read_img_raw(files[idx])
        if img is None or np.max(img) < 1000: continue
        
        # 裁剪
        x1, y1 = cx - r, cy - r
        x2, y2 = cx + r, cy + r
        h, w = img.shape
        crop = np.zeros((out_size, out_size), dtype=img.dtype)
        src_x1, src_y1 = max(0, x1), max(0, y1)
        src_x2, src_y2 = min(w, x2), min(h, y2)
        dst_x1, dst_y1 = src_x1 - x1, src_y1 - y1
        dst_x2, dst_y2 = dst_x1 + (src_x2 - src_x1), dst_y1 + (src_y2 - src_y1)
        if src_x2 > src_x1 and src_y2 > src_y1:
            crop[dst_y1:dst_y2, dst_x1:dst_x2] = img[src_y1:src_y2, src_x1:src_x2]
            
        # 增强 (必须与后续处理一致)
        crop_f = crop.astype(np.float32)
        norm = np.tanh((crop_f - mid_val) / half_range * CONFIG['squeeze_factor'])
        norm = (norm + 1) / 2.0
        norm = np.clip(norm, 0, 1)
        
        # 这里的采样不需要做 NLM (太慢)，直接统计灰度分布即可
        # 转为 8-bit
        norm_8u = (norm * 255).astype(np.uint8)
        
        # 仅取圆内有效区域
        circle_mask = np.zeros((out_size, out_size), dtype=np.uint8)
        cv2.circle(circle_mask, (r, r), r, 255, -1)
        
        valid_pixels = norm_8u[(circle_mask == 255) & (norm_8u > 5)]
        
        # 随机抽取部分像素加入蓄水池 (避免内存爆炸)
        if len(valid_pixels) > 10000:
            pixel_reservoir.append(np.random.choice(valid_pixels, 10000))
        else:
            pixel_reservoir.append(valid_pixels)
            
    if not pixel_reservoir:
        print("⚠️ 无法计算全局阈值，将使用默认值 128")
        return 128.0
        
    # 合并所有像素
    all_pixels = np.concatenate(pixel_reservoir)
    
    # 计算一次 Otsu
    otsu_val, _ = cv2.threshold(all_pixels, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    print(f"✅ 全局 Otsu 基准值: {otsu_val}")
    return otsu_val

# ================= 2. 交互式选择器 (基于全局固定值) =================

def get_random_valid_sample(files):
    search_indices = list(range(len(files)))
    random.shuffle(search_indices)
    for i in search_indices[:200]:
        img = _read_img_raw(files[i])
        if img is not None and np.max(img) > 2000:
            return img, os.path.basename(files[i])
    return None, None

def process_single_fixed_thresh(sample_img, center, radius, p1, p99, fixed_thresh_val):
    """使用固定的阈值数值进行分割"""
    cx, cy = center
    r = radius
    out_size = 2 * r
    
    # 1. 裁剪
    h, w = sample_img.shape
    crop = np.zeros((out_size, out_size), dtype=sample_img.dtype)
    x1, y1 = cx - r, cy - r
    x2, y2 = cx + r, cy + r
    src_x1, src_y1 = max(0, x1), max(0, y1)
    src_x2, src_y2 = min(w, x2), min(h, y2)
    dst_x1, dst_y1 = src_x1 - x1, src_y1 - y1
    dst_x2, dst_y2 = dst_x1 + (src_x2 - src_x1), dst_y1 + (src_y2 - src_y1)
    if src_x2 > src_x1 and src_y2 > src_y1:
        crop[dst_y1:dst_y2, dst_x1:dst_x2] = sample_img[src_y1:src_y2, src_x1:src_x2]

    # 2. 增强 (Soft-Tanh + NLM)
    crop_f = crop.astype(np.float32)
    mid_val = (p1 + p99) / 2.0
    half_range = (p99 - p1) / 2.0 + 1e-6
    norm = np.tanh((crop_f - mid_val) / half_range * CONFIG['squeeze_factor'])
    norm = (norm + 1) / 2.0
    norm = np.clip(norm, 0, 1)
    
    # 预览时也加上降噪，保证所见即所得
    sigma_est = np.mean(estimate_sigma(norm))
    if sigma_est > 0:
        norm = denoise_nl_means(norm, h=CONFIG['denoise_h'] * sigma_est, fast_mode=True, patch_size=5, patch_distance=6)
    
    norm_8u = (norm * 255).astype(np.uint8)
    
    # 3. 应用固定阈值
    circle_mask = np.zeros((out_size, out_size), dtype=np.uint8)
    cv2.circle(circle_mask, (r, r), r, 255, -1)
    
    # 直接使用传入的 fixed_thresh_val
    _, bin_img = cv2.threshold(norm_8u, fixed_thresh_val, 255, cv2.THRESH_BINARY)
    bin_img = cv2.bitwise_and(bin_img, bin_img, mask=circle_mask)
        
    return norm_8u, bin_img

def interactive_threshold_selector_global(files, center, radius, p1, p99, global_base_otsu):
    print("\n" + "="*60)
    print(" >>> 进入【全局一致性】阈值调节模式 <<<")
    print("="*60)
    print(f"全局 Otsu 基准值: {global_base_otsu}")
    print("现在调节的系数将作用于这个基准值，并应用到所有切片。")
    print("这能消除切片间的割裂感。")
    print("-" * 60)

    current_factor = 1.0
    sample_img, fname = get_random_valid_sample(files)
    
    while True:
        # 计算当前的绝对阈值
        current_abs_thresh = global_base_otsu * current_factor
        
        img_gray, img_bin = process_single_fixed_thresh(sample_img, center, radius, p1, p99, current_abs_thresh)

        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.title(f"Source: {fname}\nGlobal Base: {global_base_otsu:.1f}")
        plt.imshow(img_gray, cmap='gray')
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.title(f"Result (Factor={current_factor} -> Thresh={current_abs_thresh:.1f})")
        plt.imshow(img_bin, cmap='gray')
        plt.axis('off')
        
        plt.tight_layout()
        plt.show(block=False)
        plt.pause(0.1)
        
        user_input = input(f"当前系数 [{current_factor}] | 阈值 [{current_abs_thresh:.1f}] >> 指令: ").strip().lower()
        plt.close() 

        if user_input in ['run', 'ok', 'yes', 'y']:
            print(f"✅ 最终确认系数: {current_factor}")
            print(f"🔒 全局锁定阈值: {current_abs_thresh:.2f}")
            return current_abs_thresh # 返回绝对阈值
        
        elif user_input == 'check':
            new_img, new_fname = get_random_valid_sample(files)
            if new_img is not None:
                sample_img = new_img
                fname = new_fname
                print(f"已切换至: {fname}")
        else:
            try:
                val = float(user_input)
                if val <= 0: print("⚠️ 系数必须大于 0")
                else: current_factor = val
            except ValueError:
                print("⚠️ 输入无效")

# ================= 3. 最终处理函数 (应用固定阈值) =================

def process_slice_final_fixed(img, center, radius, p1, p99, fixed_thresh_val):
    cx, cy = center
    r = radius
    out_size = 2 * r
    
    circle_mask = np.zeros((out_size, out_size), dtype=np.uint8)
    cv2.circle(circle_mask, (r, r), r, 255, -1)
    
    # 裁剪
    h, w = img.shape
    crop = np.zeros((out_size, out_size), dtype=img.dtype)
    x1, y1 = cx - r, cy - r
    x2, y2 = cx + r, cy + r
    src_x1, src_y1 = max(0, x1), max(0, y1)
    src_x2, src_y2 = min(w, x2), min(h, y2)
    dst_x1, dst_y1 = src_x1 - x1, src_y1 - y1
    dst_x2, dst_y2 = dst_x1 + (src_x2 - src_x1), dst_y1 + (src_y2 - src_y1)
    if src_x2 > src_x1 and src_y2 > src_y1:
        crop[dst_y1:dst_y2, dst_x1:dst_x2] = img[src_y1:src_y2, src_x1:src_x2]
    
    if crop.max() == 0: return np.zeros((out_size, out_size), dtype=np.uint8)

    # 增强
    crop_f = crop.astype(np.float32)
    mid_val = (p1 + p99) / 2.0
    half_range = (p99 - p1) / 2.0 + 1e-6
    norm = np.tanh((crop_f - mid_val) / half_range * CONFIG['squeeze_factor'])
    norm = (norm + 1) / 2.0
    norm = np.clip(norm, 0, 1)
    
    # 降噪
    sigma_est = np.mean(estimate_sigma(norm))
    if sigma_est > 0:
        norm = denoise_nl_means(norm, h=CONFIG['denoise_h'] * sigma_est, fast_mode=True, patch_size=5, patch_distance=6)

    # 应用固定阈值 (核心改动: 不再计算 Otsu)
    norm_8u = (norm * 255).astype(np.uint8)
    
    _, bin_img = cv2.threshold(norm_8u, fixed_thresh_val, 255, cv2.THRESH_BINARY)
    bin_img = cv2.bitwise_and(bin_img, bin_img, mask=circle_mask)
    
    # 碎片过滤
    final_img = np.zeros((out_size, out_size), dtype=np.uint8)
    
    if CONFIG['morph_open_size'] > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (CONFIG['morph_open_size'], CONFIG['morph_open_size']))
        bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, kernel)
    
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bin_img, connectivity=8)
    if num_labels > 1:
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area > CONFIG['min_fragment_size']:
                final_img[labels == i] = 255
                    
    return final_img

# ================= 主程序 =================

def main():
    files = glob.glob(os.path.join(INPUT_DIR, FILE_EXT))
    files.sort(key=natural_sort_key)
    if not files: return

    # 1. 基础统计
    roi_center, roi_radius = detect_global_roi_v3(files, CONFIG['roi_sample_count'])
    p1, p99 = get_global_brightness_stats(files, roi_center, roi_radius)
    
    # 2. 计算全局 Otsu 基准
    global_base = calculate_global_otsu_base(files, roi_center, roi_radius, p1, p99)
    
    # 3. 交互式确定最终固定阈值
    final_fixed_thresh = interactive_threshold_selector_global(files, roi_center, roi_radius, p1, p99, global_base)
    
    # 4. 批量处理
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    
    print(f"\nStep 3: 开始批量处理... (使用固定阈值: {final_fixed_thresh:.2f})")
    for fpath in tqdm(files):
        try:
            fname = os.path.basename(fpath)
            save_path = os.path.join(OUTPUT_DIR, fname)
            img = _read_img_raw(fpath)
            if img is None:
                out_size = 2 * roi_radius
                dummy = np.zeros((out_size, out_size), dtype=np.uint8)
                _save_img_safe(save_path, dummy)
                continue
            
            res = process_slice_final_fixed(img, roi_center, roi_radius, p1, p99, final_fixed_thresh)
            _save_img_safe(save_path, res)
            
        except Exception as e:
            print(f"Error {fpath}: {e}")

    print("🎉 全部完成！")

if __name__ == "__main__":
    main()
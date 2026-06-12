import os

# ================= 性能配置 =================
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["CV_NUM_THREADS"] = "1"

import glob
import re
import argparse
import cv2
import numpy as np
import random
import matplotlib.pyplot as plt
from tqdm import tqdm

# ================= 配置区域 =================
# 【注意】这里的输入目录应该指向您跑完 "三合一预处理" 后的文件夹
INPUT_DIR = r"D:\浅层礁灰岩数据集\Preprocessed_Slices"
OUTPUT_DIR = r"D:\浅层礁灰岩数据集\binary_image" 
FILE_PATTERNS = ("*.tif", "*.tiff")

CONFIG = {
    'roi_sample_count': 500,      # 寻找 ROI 的采样数
    'global_otsu_sample': 1200,   # 计算全局阈值时的采样切片数
    'min_fragment_size': 50,      # 连通域过滤：剔除小于 50 像素的孤立噪点
    'morph_open_size': 1,         # 形态学开运算核大小 (0表示关闭)
}

# ================= 工具函数 =================
def natural_sort_key(filepath):
    filename = os.path.basename(filepath)
    numbers = re.findall(r'\d+', filename)
    return int(numbers[-1]) if numbers else 0

def list_tiff_files(folder_path):
    files = []
    for pattern in FILE_PATTERNS:
        files.extend(glob.glob(os.path.join(folder_path, pattern)))
    return sorted(files, key=natural_sort_key)

def discover_input_folders(input_dir):
    direct_files = list_tiff_files(input_dir)
    if direct_files:
        return [(input_dir, None)]

    folder_jobs = []
    for entry in sorted(os.listdir(input_dir)):
        folder_path = os.path.join(input_dir, entry)
        if not os.path.isdir(folder_path):
            continue
        tif_files = list_tiff_files(folder_path)
        if tif_files:
            folder_jobs.append((folder_path, entry))
    return folder_jobs

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

# ================= 1. ROI 探测 (由于预处理已经置零了背景，这一步会非常准) =================
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
        if img is None or np.max(img) < 1000: continue

        # 因为预处理已经把外部设为了纯黑(0)，我们直接用 > 0 找轮廓即可
        img_8u = (img / 256.0).astype(np.uint8)
        _, thresh = cv2.threshold(img_8u, 5, 255, cv2.THRESH_BINARY)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            c = max(contours, key=cv2.contourArea)
            if cv2.contourArea(c) < 500: continue
            (x, y), r = cv2.minEnclosingCircle(c)
            valid_rois.append((x, y, r))

    if not valid_rois:
        dummy = _read_img_raw(files[0])
        h, w = dummy.shape
        return (w//2, h//2), w//4

    valid_rois = np.array(valid_rois)
    avg_center = np.median(valid_rois[:, :2], axis=0).astype(int)
    max_radius = int(np.percentile(valid_rois[:, 2], 90)) 
    print(f"[OK] ROI locked: Center={avg_center}, Radius={max_radius}")
    return avg_center, max_radius

# ================= 2. 计算全局统一 Otsu 基准值 =================
def calculate_global_otsu_base(files, center, radius):
    print("Step 2: 正在计算【全局统一】Otsu 基准阈值 (消除层间割裂)...")
    
    sample_indices = np.linspace(0, len(files)-1, CONFIG['global_otsu_sample'], dtype=int)
    pixel_reservoir = []
    
    cx, cy = center
    r = radius
    
    for idx in tqdm(sample_indices, desc="Global Sampling"):
        img = _read_img_raw(files[idx])
        if img is None or np.max(img) < 1000: continue
        
        # 将 16-bit (0-65535) 线性映射到 8-bit (0-255) 以便计算 Otsu
        img_8u = (img / 256.0).astype(np.uint8)
        
        # 裁剪出 ROI 区域
        y1, y2 = max(0, cy-r), min(img.shape[0], cy+r)
        x1, x2 = max(0, cx-r), min(img.shape[1], cx+r)
        crop_8u = img_8u[y1:y2, x1:x2]
        
        # 提取有效像素 (排除背景 0)
        valid_pixels = crop_8u[crop_8u > 5]
        
        if len(valid_pixels) > 10000:
            pixel_reservoir.append(np.random.choice(valid_pixels, 10000))
        elif len(valid_pixels) > 0:
            pixel_reservoir.append(valid_pixels)
            
    if not pixel_reservoir:
        print("[WARN] Unable to compute global threshold, fallback to 128")
        return 128.0
        
    all_pixels = np.concatenate(pixel_reservoir)
    otsu_val, _ = cv2.threshold(all_pixels, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    print(f"[OK] Global Otsu baseline (8-bit): {otsu_val}")
    return otsu_val

# ================= 3. 交互式选择器 =================
def get_random_valid_sample(files):
    search_indices = list(range(len(files)))
    random.shuffle(search_indices)
    for i in search_indices[:200]:
        img = _read_img_raw(files[i])
        if img is not None and np.max(img) > 2000:
            return img, os.path.basename(files[i])
    return None, None

def process_single_fixed_thresh(sample_img, center, radius, fixed_thresh_val):
    cx, cy = center
    r = radius
    out_size = 2 * r
    
    # 裁剪
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

    # 映射为 8-bit
    crop_8u = (crop / 256.0).astype(np.uint8)
    
    # 掩膜限制
    circle_mask = np.zeros((out_size, out_size), dtype=np.uint8)
    cv2.circle(circle_mask, (r, r), r, 255, -1)
    
    # 二值化
    _, bin_img = cv2.threshold(crop_8u, fixed_thresh_val, 255, cv2.THRESH_BINARY)
    bin_img = cv2.bitwise_and(bin_img, bin_img, mask=circle_mask)
        
    return crop_8u, bin_img

def interactive_threshold_selector_global(files, center, radius, global_base_otsu,
                                         auto_mode=False, auto_factor=1.0):
    print("\n" + "="*60)
    print(" >>> 进入【全局一致性】阈值调节模式 <<<")
    print("="*60)

    current_factor = 1.0
    sample_img, fname = get_random_valid_sample(files)
    if sample_img is None:
        print("No valid sample found for preview. Falling back to global Otsu threshold.")
        return float(global_base_otsu)

    if auto_mode:
        current_abs_thresh = global_base_otsu * auto_factor
        print(f"Auto mode enabled: factor={auto_factor} -> thresh={current_abs_thresh:.2f}")
        return float(current_abs_thresh)
    
    while True:
        current_abs_thresh = global_base_otsu * current_factor
        img_8u, img_bin = process_single_fixed_thresh(sample_img, center, radius, current_abs_thresh)

        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.title(f"Preprocessed (8-bit View): {fname}")
        plt.imshow(img_8u, cmap='gray')
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.title(f"Binary Result (Factor={current_factor} -> Thresh={current_abs_thresh:.1f})")
        plt.imshow(img_bin, cmap='gray')
        plt.axis('off')
        
        plt.tight_layout()
        plt.show(block=False)
        plt.pause(0.1)
        
        user_input = input(f"当前系数 [{current_factor}] | 阈值 [{current_abs_thresh:.1f}] >> 指令 (数字/run/check): ").strip().lower()
        plt.close() 

        if user_input in ['run', 'ok', 'yes', 'y']:
            print(f"[OK] Final factor: {current_factor}")
            print(f"🔒 全局锁定阈值: {current_abs_thresh:.2f}")
            return current_abs_thresh 
        elif user_input == 'check':
            new_img, new_fname = get_random_valid_sample(files)
            if new_img is not None:
                sample_img, fname = new_img, new_fname
        else:
            try:
                val = float(user_input)
                if val > 0: current_factor = val
            except: pass

# ================= 4. 最终处理函数 =================
def process_slice_final_fixed(img, center, radius, fixed_thresh_val):
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

    # 映射到 8-bit (除以 256)
    crop_8u = (crop / 256.0).astype(np.uint8)
    
    # 二值化
    _, bin_img = cv2.threshold(crop_8u, fixed_thresh_val, 255, cv2.THRESH_BINARY)
    bin_img = cv2.bitwise_and(bin_img, bin_img, mask=circle_mask)
    
    # 形态学碎片过滤 (保留了您原本极其优秀的剔除孤立噪点逻辑)
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

def process_single_folder(input_dir, output_dir, auto_mode=False, auto_factor=1.0):
    files = list_tiff_files(input_dir)
    if not files:
        print(f"[ERROR] No files found, please check path: {input_dir}")
        return False

    os.makedirs(output_dir, exist_ok=True)
    print(f"\n{'=' * 70}")
    print(f"Input : {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Files : {len(files)}")
    print(f"{'=' * 70}")

    roi_center, roi_radius = detect_global_roi_v3(files, CONFIG['roi_sample_count'])
    global_base = calculate_global_otsu_base(files, roi_center, roi_radius)
    final_fixed_thresh = interactive_threshold_selector_global(
        files, roi_center, roi_radius, global_base,
        auto_mode=auto_mode, auto_factor=auto_factor
    )

    print(f"\nStep 3: Start batch processing... (fixed threshold {final_fixed_thresh:.2f})")
    for fpath in tqdm(files):
        try:
            fname = os.path.basename(fpath)
            save_path = os.path.join(output_dir, fname)

            img = _read_img_raw(fpath)
            if img is None:
                out_size = 2 * roi_radius
                dummy = np.zeros((out_size, out_size), dtype=np.uint8)
                _save_img_safe(save_path, dummy)
                continue

            res = process_slice_final_fixed(img, roi_center, roi_radius, final_fixed_thresh)
            _save_img_safe(save_path, res)

        except Exception as e:
            print(f"Error {fpath}: {e}")

    print(f"[OK] Finished: {input_dir}")
    return True

def parse_args():
    parser = argparse.ArgumentParser(description="Convert preprocessed CT slices to binary slices.")
    parser.add_argument("--input-dir", default=INPUT_DIR, help="Input directory or root directory.")
    parser.add_argument("--output-dir", default=OUTPUT_DIR, help="Output directory or root directory.")
    parser.add_argument("--auto", action="store_true", help="Run without interactive threshold tuning.")
    parser.add_argument("--factor", type=float, default=1.0, help="Threshold factor used in auto mode.")
    return parser.parse_args()

# ================= 主程序 =================
def main():
    args = parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir

    if not os.path.isdir(input_dir):
        print(f"[ERROR] Input directory does not exist: {input_dir}")
        return

    folder_jobs = discover_input_folders(input_dir)
    if not folder_jobs:
        print(f"[ERROR] No .tif/.tiff files found under {input_dir}")
        return

    total_ok = 0
    for src_dir, rel_name in folder_jobs:
        dst_dir = output_dir if rel_name is None else os.path.join(output_dir, rel_name)
        ok = process_single_folder(
            src_dir,
            dst_dir,
            auto_mode=args.auto,
            auto_factor=args.factor
        )
        total_ok += int(ok)

    print(f"\n[OK] All done. Processed {total_ok}/{len(folder_jobs)} input folders successfully.")
    return

    files = glob.glob(os.path.join(INPUT_DIR, FILE_EXT))
    files.sort(key=natural_sort_key)
    if not files: 
        print(f"❌ 未找到文件，请检查路径: {INPUT_DIR}")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. 基础统计
    roi_center, roi_radius = detect_global_roi_v3(files, CONFIG['roi_sample_count'])
    
    # 2. 计算全局 Otsu 基准
    global_base = calculate_global_otsu_base(files, roi_center, roi_radius)
    
    # 3. 交互式确定最终固定阈值
    final_fixed_thresh = interactive_threshold_selector_global(files, roi_center, roi_radius, global_base)
    
    # 4. 批量处理
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
            
            res = process_slice_final_fixed(img, roi_center, roi_radius, final_fixed_thresh)
            _save_img_safe(save_path, res)
            
        except Exception as e:
            print(f"Error {fpath}: {e}")

    print("🎉 全部完成！")

if __name__ == "__main__":
    main()

import os

# ==============================================================================
#  性能优化配置: 必须在 import numpy/cv2 之前设置
# ==============================================================================
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["CV_NUM_THREADS"] = "1"

import glob
import numpy as np
import cv2
import random
from tqdm import tqdm
from pathlib import Path
from skimage.restoration import denoise_nl_means, estimate_sigma
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

# ================= 全局配置 =================
CONFIG = {
    "src_root": r"D:\多尺度岩心数据集",         # 原始 CT 文件夹
    "dst_root": r"D:\多尺度岩心数据集\Preprocessed_Slices", # 一站式输出文件夹
    
    "target_folders": [
        "6-6-18",
        "6-6-21",
        "6-6-22",
        "6-6-24"
    ],
    
    "max_workers": 12, # 物理核心数
    
    # --- 1. 峰值锚定参数 ---
    "target_peak": 32000,      
    "knee_start": 60000,       
    "max_val": 65535,          
    "matrix_threshold": 10000, # 过滤 Raw 数据中的空气孔隙，寻找基质主峰
    
    # --- 2. Soft-Tanh 参数 ---
    "squeeze_factor": 2.0,
    
    # --- 3. NLM 降噪参数 ---
    "denoise_h": 4,            # 保持之前验证过的最佳强度
}

# ================= 核心算法模块 =================

def _read_img_raw(path):
    try:
        raw_data = np.fromfile(path, dtype=np.uint8)
        return cv2.imdecode(raw_data, cv2.IMREAD_UNCHANGED)
    except:
        return None

def _detect_global_roi(files, sample_count=50):
    """提取岩心圆形 ROI，屏蔽外部空气"""
    if not files: return None
    mid_start, mid_end = int(len(files) * 0.4), int(len(files) * 0.6)
    if mid_end <= mid_start: mid_start, mid_end = 0, len(files)
    
    indices = np.linspace(mid_start, mid_end, sample_count, dtype=int)
    valid_rois = []

    for idx in indices:
        img = _read_img_raw(files[idx])
        if img is None: continue

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

    if not valid_rois: return None
    valid_rois = np.array(valid_rois)
    avg_center = np.median(valid_rois[:, :2], axis=0).astype(int)
    max_radius = int(np.percentile(valid_rois[:, 2], 90))
    return avg_center, max_radius

def get_matrix_peak_with_roi(img, center, radius):
    """精准锁定原始岩石骨架主峰"""
    cx, cy = center
    h, w = img.shape
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (cx, cy), radius, 1, -1)
    
    valid_pixels = img[mask == 1]
    valid_pixels = valid_pixels[valid_pixels > CONFIG["matrix_threshold"]]
    
    if len(valid_pixels) < 1000: return None 
        
    counts = np.bincount(valid_pixels, minlength=65536)
    counts[60000:] = 0 # 屏蔽极端高光堆积
    return int(np.argmax(counts))

def soft_highlight_compression(data, knee_start=60000, max_val=65535):
    """高光柔性滚降算法 (解决截断堆积)"""
    over_knee = data > knee_start
    if not np.any(over_knee):
        return data.astype(np.uint16)
    
    x = data[over_knee].astype(np.float32)
    range_width = max_val - knee_start
    normalized_input = (x - knee_start) / range_width
    compressed_output = knee_start + range_width * np.tanh(normalized_input)
    
    data[over_knee] = compressed_output
    return np.clip(data, 0, max_val).astype(np.uint16)

def calculate_core_global_stats(folder_path, roi_info):
    """
    一站式计算模块：
    1. 计算 Raw 数据的基准峰值
    2. 虚拟应用锚定，并计算锚定后数据的 P1 和 P99 (完美衔接 Soft-Tanh)
    """
    files = glob.glob(os.path.join(folder_path, "*.tif"))
    if not files: return None
    
    sample_files = random.sample(files, min(len(files), 150))
    center, radius = roi_info
    
    # 1. 计算原始峰值
    peaks = []
    sampled_imgs = []
    for f in sample_files:
        img = _read_img_raw(f)
        if img is not None:
            sampled_imgs.append(img)
            peak = get_matrix_peak_with_roi(img, center, radius)
            if peak is not None: peaks.append(peak)
            
    if not peaks: return None
    core_ref_peak = int(np.median(peaks))
    
    # 2. 虚拟锚定并计算 P1/P99
    scale_factor = CONFIG["target_peak"] / float(core_ref_peak)
    anchored_pixels = []
    
    h_img, w_img = sampled_imgs[0].shape
    mask = np.zeros((h_img, w_img), dtype=np.uint8)
    cv2.circle(mask, (center[0], center[1]), radius, 1, -1)
    
    for img in sampled_imgs:
        # 应用线性缩放 + 高光压缩
        data_float = img.astype(np.float32) * scale_factor
        anchored_img = soft_highlight_compression(data_float, CONFIG["knee_start"], CONFIG["max_val"])
        
        # 获取有效像素
        valid_px = anchored_img[mask == 1]
        valid_px = valid_px[valid_px > 1000] # 过滤纯空气
        if len(valid_px) > 10000:
            valid_px = np.random.choice(valid_px, 10000)
        anchored_pixels.append(valid_px)
        
    all_px = np.concatenate(anchored_pixels)
    p1 = float(np.percentile(all_px, 1))
    p99 = float(np.percentile(all_px, 99))
    
    return core_ref_peak, p1, p99

# ================= 极速三合一 Worker =================
def worker_process_slice(args):
    fpath, core_ref_peak, p1, p99, roi_info, out_dir = args
    try:
        # 1. 读取原始图像
        img = _read_img_raw(fpath)
        if img is None: return False
        
        # 2. 【第一步】: 峰值锚定 (物理刻度对齐)
        scale_factor = CONFIG["target_peak"] / float(core_ref_peak)
        data_float = img.astype(np.float32) * scale_factor
        anchored_img = soft_highlight_compression(data_float, CONFIG["knee_start"], CONFIG["max_val"])
        
        # 3. 【第二步】: Soft-Tanh 动态范围压缩 (非线性增强)
        slice_data = anchored_img.astype(np.float32)
        mid_val = (p1 + p99) / 2.0
        half_range = (p99 - p1) / 2.0 + 1e-6
        norm_temp = np.tanh(((slice_data - mid_val) / half_range) * CONFIG["squeeze_factor"])
        norm_f = np.clip((norm_temp + 1) / 2.0, 0, 1)
        
        # 4. 【第三步】: NLM 降噪 (在 0-1 空间执行)
        (cx, cy), r = roi_info
        mask = np.zeros_like(anchored_img, dtype=np.uint8)
        cv2.circle(mask, (cx, cy), r, 1, -1)
        
        if np.mean(norm_f[mask == 1]) > 0.001:
            # 仅在有效区域计算噪声 sigma
            sigma_est = np.mean(estimate_sigma(norm_f))
            if sigma_est > 0:
                norm_f = denoise_nl_means(norm_f, h=CONFIG['denoise_h'] * sigma_est, 
                                          fast_mode=True, patch_size=5, patch_distance=6)
        
        # 背景置零
        norm_f[mask == 0] = 0
        
        # 5. 映射回 16-bit 物理空间并保存
        final_img = (norm_f * 65535).astype(np.uint16)
        
        save_path = os.path.join(out_dir, os.path.basename(fpath))
        cv2.imencode(".tif", final_img)[1].tofile(save_path)
        return True
        
    except Exception as e:
        print(f"Error processing {fpath}: {e}")
        return False

# ================= 主流程 =================
def run_unified_pipeline():
    print("🚀 启动终极预处理流水线 [锚定 -> Soft-Tanh -> NLM]")
    
    for folder_name in CONFIG["target_folders"]:
        src_folder = os.path.join(CONFIG["src_root"], folder_name)
        dst_folder = os.path.join(CONFIG["dst_root"], folder_name)
        
        files = glob.glob(os.path.join(src_folder, "*.tif"))
        if not files: continue
            
        Path(dst_folder).mkdir(parents=True, exist_ok=True)
        print(f"\n📂 正在处理岩心: {folder_name} (共 {len(files)} 张切片)")
        
        # 1. 自动探测 ROI
        roi_info = _detect_global_roi(files, sample_count=1000)
        if not roi_info: continue
            
        # 2. 一站式计算物理参数 (锚定峰值 & 锚定后的 P1/P99)
        stats = calculate_core_global_stats(src_folder, roi_info)
        if not stats: continue
        core_ref_peak, p1, p99 = stats
            
        print(f"   ⚓ 锚定前峰值: {core_ref_peak} -> 锚定目标: {CONFIG['target_peak']}")
        print(f"   📊 Soft-Tanh 参数锁定: P1={p1:.1f}, P99={p99:.1f}")
        
        # 3. 多进程批量“三合一”处理
        tasks = [(f, core_ref_peak, p1, p99, roi_info, dst_folder) for f in files]
        success_count = 0
        with ProcessPoolExecutor(max_workers=CONFIG["max_workers"]) as executor:
            for result in tqdm(executor.map(worker_process_slice, tasks, chunksize=50), 
                               total=len(tasks), desc="   ⚙️ 处理进度"):
                if result: success_count += 1
                    
        print(f"   🎉 {folder_name} 预处理彻底完成！成功: {success_count}/{len(files)}")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    run_unified_pipeline()
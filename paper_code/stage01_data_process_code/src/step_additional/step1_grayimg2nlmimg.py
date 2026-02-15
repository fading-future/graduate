import os

# ==============================================================================
#  性能优化配置: 必须在 import numpy/cv2 之前设置
# ==============================================================================
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["CV_NUM_THREADS"] = "1"

import glob
import re
import numpy as np
import cv2
import random
from tqdm import tqdm
from skimage.restoration import denoise_nl_means, estimate_sigma
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

# ==============================================================================
#  全局配置中心
# ==============================================================================
CONFIG = {
    # 路径配置
    'src_root': r"D:\多尺度岩心数据集", 
    'dst_root': r"D:\多尺度岩心数据集\NLM_Denoised_Slices",
    
    'target_folders': [
        "6-6-18",
        "6-6-21",
        "6-6-22", 
        "6-6-24"
    ],

    # 1. 统计与分位点 (建议先运行一次 'calculate_stats' 模式获取)
    'global_p1': 192.0,     
    'global_p99': 57256.0, 

    # 2. 算法参数
    'squeeze_factor': 2.0,   # Soft-Tanh 挤压因子
    'denoise_h': 4,          # NLM 噪声强度倍数
    'global_roi_sample': 100,
    
    # 3. 并行配置
    'max_workers': 12,       # 根据物理核心数调整
    'batch_size': 20,        # 每个子进程任务处理的切片数
}

# ==============================================================================
#  工具函数
# ==============================================================================
def natural_sort_key(filepath):
    filename = os.path.basename(filepath)
    numbers = re.findall(r'\d+', filename)
    return int(numbers[-1]) if numbers else 0

def _read_img_raw(path):
    try:
        raw_data = np.fromfile(path, dtype=np.uint8)
        return cv2.imdecode(raw_data, cv2.IMREAD_UNCHANGED)
    except:
        return None

def _save_img_16bit(path, img):
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # 使用 imencode 处理可能存在的中文路径
        cv2.imencode(".tif", img.astype(np.uint16))[1].tofile(path)
    except Exception as e:
        print(f"Error saving {path}: {e}")

# [cite_start]复用您之前的 ROI 探测逻辑 
def _detect_global_roi(files, sample_count=100):
    if not files: return None
    mid_start, mid_end = int(len(files) * 0.4), int(len(files) * 0.6)
    indices = np.linspace(mid_start, mid_end, min(sample_count, len(files)), dtype=int)
    valid_rois = []
    for idx in indices:
        img = _read_img_raw(files[idx])
        if img is None or np.mean(img) < 2000: continue
        scale = 0.2
        small = cv2.resize(img, (int(img.shape[1]*scale), int(img.shape[0]*scale)))
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
            valid_rois.append((x / scale, y / scale, r / scale))
    if not valid_rois: return None
    valid_rois = np.array(valid_rois)
    avg_center = np.median(valid_rois[:, :2], axis=0).astype(int)
    max_radius = int(np.percentile(valid_rois[:, 2], 90))
    return avg_center, max_radius

# ==============================================================================
#  核心处理 Worker
# ==============================================================================
def process_slice_batch_task(task_data):
    """
    子进程执行函数：处理一批切片并保存
    """
    slice_paths, folder_name, roi_info, local_config = task_data
    (cx, cy), r = roi_info
    
    # [cite_start]预计算 Soft-Tanh 参数 [cite: 13]
    g_p1 = local_config['global_p1']
    g_p99 = local_config['global_p99']
    mid_val = (g_p1 + g_p99) / 2.0
    half_range = (g_p99 - g_p1) / 2.0 + 1e-6
    squeeze_factor = local_config['squeeze_factor']

    processed_count = 0
    for p in slice_paths:
        img = _read_img_raw(p)
        if img is None: continue

        # [cite_start]1. Soft-Tanh 动态范围压缩逻辑 [cite: 13]
        slice_data = img.astype(np.float32)
        norm_temp = np.tanh(((slice_data - mid_val) / half_range) * squeeze_factor)
        norm_f = (norm_temp + 1) / 2.0
        norm_f = np.clip(norm_f, 0, 1)

        # [cite_start]2. NLM 降噪 [cite: 13]
        if np.mean(norm_f) > 0.001:
            sigma_est = np.mean(estimate_sigma(norm_f))
            if sigma_est > 0:
                norm_f = denoise_nl_means(
                    norm_f, 
                    h=local_config['denoise_h'] * sigma_est, 
                    fast_mode=True, 
                    patch_size=5, 
                    patch_distance=6
                )
        
        # 3. 背景置零（仅保留岩心区域）
        mask = np.zeros(img.shape, dtype=np.uint8)
        cv2.circle(mask, (cx, cy), int(r), 1, -1)
        norm_f[mask == 0] = 0
        
        # [cite_start]4. 映射回 16-bit 空间以便后续对齐 [cite: 13]
        final_img = (norm_f * 65535).astype(np.uint16)
        
        # 5. 保存结果
        rel_path = os.path.basename(p)
        save_path = os.path.join(local_config['dst_root'], folder_name, rel_path)
        _save_img_16bit(save_path, final_img)
        processed_count += 1

    return processed_count

def run_denoise_pipeline():
    print(f"🚀 启动降噪流水线 | 输出格式: 16-bit TIF")
    all_tasks = []
    
    for folder_name in CONFIG['target_folders']:
        src_dir = os.path.join(CONFIG['src_root'], folder_name)
        files = sorted(glob.glob(os.path.join(src_dir, "*.tif")), key=natural_sort_key)
        if not files: continue
        
        print(f"📂 正在分析文件夹: {folder_name}")
        roi_info = _detect_global_roi(files, CONFIG['global_roi_sample'])
        if not roi_info: continue

        # 按 Batch 分配任务
        batch_size = CONFIG['batch_size']
        for i in range(0, len(files), batch_size):
            batch_files = files[i : i + batch_size]
            all_tasks.append((batch_files, folder_name, roi_info, CONFIG))
            
    print(f"✅ 生成 {len(all_tasks)} 个批处理任务。开始运行...")

    with ProcessPoolExecutor(max_workers=CONFIG['max_workers']) as executor:
        futures = [executor.submit(process_slice_batch_task, task) for task in all_tasks]
        for future in tqdm(as_completed(futures), total=len(all_tasks), desc="降噪进度"):
            future.result()

    print(f"🎉 处理完成！结果保存在: {CONFIG['dst_root']}")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    run_denoise_pipeline()
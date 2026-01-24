import os

# ==============================================================================
#  关键修复: 必须在 import numpy/cv2 之前设置
#  限制底层库只使用单核，把 CPU 和内存留给多进程架构
# ==============================================================================
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["CV_NUM_THREADS"] = "1"

import glob
import re
import numpy as np
import cv2
import random
from scipy import ndimage
from tqdm import tqdm
from skimage.restoration import denoise_nl_means, estimate_sigma
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

# ==============================================================================
#  全局配置中心
# ==============================================================================
CONFIG = {
    # 'calculate_stats' 或 'pipeline'
    'run_mode': 'pipeline', 

    # 填入你计算好的值 (如果 P1 还是 0，可以在这里手动改为 2000 试试效果)
    'global_p1': 192.0,     
    'global_p99': 57256.0, 

    # 路径
    'src_root': r"/chendou_space/data/core_ctimg_data",
    'dst_root': r"/chendou_space/data/cleaned_npy_dataset",
    
    'target_folders': [
        "6-6-20 全部",
        "6-6-21",
        "6-6-22", 
    ],

    # 图像参数
    'crop_size': 256,
    'stride_z': 64,
    'stride_xy': 64,
    'denoise_type': 'nlm', 
    'denoise_h': 4,

    # 形态学参数
    'sever_size': 25,
    'restore_size': 24,
    'margin_size': 10,
    'global_roi_sample': 100,
    'validity_ratio_threshold': 0.90, 

    # 统计采样参数
    'stats_sample_rate': 0.3,    
    'stats_pixels_per_img': 10000,
    
    # --- 关键修改：并行参数 ---
    # 限制为 12 核，防止内存溢出 (OOM)
    'max_workers': 14, 
}

# ==============================================================================
#  工具函数
# ==============================================================================
def _read_img_raw(path):
    try:
        raw_data = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(raw_data, cv2.IMREAD_UNCHANGED)
        return img
    except:
        return None

# 定义一个稳健的排序函数
def natural_sort_key(filepath):
    filename = os.path.basename(filepath)
    # 寻找文件名里所有的数字串
    numbers = re.findall(r'\d+', filename)
    if numbers:
        # 通常取最后一个数字作为序号 (比如 img_slice_001.tif -> 1)
        # 如果你的文件名是 2023_01_01_slice_001.tif，可能需要根据实际情况调整
        return int(numbers[-1]) 
    return 0

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

# ==============================================================================
#  并行任务核心：以 Chunk (Z轴块) 为单位处理
# ==============================================================================
def process_single_chunk_task(task_data):
    """
    运行在子进程中的 Worker。
    一个任务负责处理 256 层图片，生成 NPY 块。
    """
    chunk_paths, folder_name, z_start, roi_info, local_config = task_data
    
    (cx, cy), r = roi_info
    crop_size = local_config['crop_size']
    
    # 在子进程内初始化核，避免跨进程传递
    kernel_sever = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (local_config['sever_size'], local_config['sever_size']))
    kernel_restore = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (local_config['restore_size'], local_config['restore_size']))
    margin = local_config['margin_size']
    kernel_erode = np.ones((crop_size + margin, crop_size + margin), np.uint8)
    
    # 确定裁剪 ROI
    r_pad = int(r * 1.05)
    y1, y2 = max(0, cy - r_pad), min(1900, cy + r_pad)
    x1, x2 = max(0, cx - r_pad), min(1900, cx + r_pad)
    h_roi, w_roi = y2-y1, x2-x1
    
    # 本地圆形 Mask
    mask_local = np.zeros((h_roi, w_roi), dtype=np.uint8)
    cv2.circle(mask_local, (cx-x1, cy-y1), r, 1, -1)
    
    # 1. 批量读取图片
    raw_stack = []
    for p in chunk_paths:
        full_img = _read_img_raw(p)
        if full_img is None: 
            raw_stack.append(np.zeros((h_roi, w_roi), dtype=np.uint16))
        else:
            # 容错裁剪
            crop = full_img[y1:y2, x1:x2]
            if crop.shape != (h_roi, w_roi):
                padded = np.zeros((h_roi, w_roi), dtype=np.uint16)
                padded[:crop.shape[0], :crop.shape[1]] = crop
                raw_stack.append(padded)
            else:
                raw_stack.append(crop)
    
    stack = np.array(raw_stack) 

    clean_stack = []
    binary_stack = []
    
    g_p1 = local_config['global_p1']
    g_p99 = local_config['global_p99']

    # --- 预计算 Soft-Tanh 参数 ---
    # 计算中心点和半宽
    # 这种映射策略不仅能去处尖峰，还能增强中间岩石纹理的对比度
    mid_val = (g_p1 + g_p99) / 2.0
    half_range = (g_p99 - g_p1) / 2.0 + 1e-6  # 加 epsilon 防止除零

    # 挤压因子 (Squeeze Factor)：
    # factor=2.0 意味着原本的 P99 位置会被映射到 tanh(2.0) ≈ 0.96
    # 这样给两头留出了约 4% 的空间来容纳超出的异常值 (Outliers)，实现"软着陆"
    squeeze_factor = 2.0

    # 2. 串行处理当前 Chunk 的每一层 (减少单个 Worker 的内存峰值)
    for i in range(stack.shape[0]):
        slice_data = stack[i].astype(np.float32)
        
        # 1. 中心化并缩放：将 [P1, P99] 映射到 [-1.0, 1.0] (暂定)
        norm_temp = (slice_data - mid_val) / half_range
        
        # 2. 应用 Tanh 软挤压
        # 此时：
        # - 位于中心 (mid_val) 的值 -> 0
        # - 位于 P99 的值 -> tanh(2.0) ≈ 0.96
        # - 远大于 P99 的极端值 -> 平滑逼近 1.0 (但不会突变为1.0)
        norm_temp = np.tanh(norm_temp * squeeze_factor)
        
        # 3. 映射回 [0, 1] 区间
        norm_f = (norm_temp + 1) / 2.0

        # 4. (可选) 最后的安全钳位，防止 float 精度误差溢出，但实际上极少触发
        # 注意：这里 clip 只是为了数值安全，大部分数据已经通过 tanh 落在 0.02~0.98 之间了
        norm_f = np.clip(norm_f, 0, 1)
        
        # 降噪
        if np.mean(norm_f) > 0.001: 
            if local_config['denoise_type'] == 'nlm':
                sigma_est = np.mean(estimate_sigma(norm_f))
                if sigma_est > 0:
                    norm_f = denoise_nl_means(norm_f, h=local_config['denoise_h'] * sigma_est, 
                                            fast_mode=True, patch_size=5, patch_distance=6)
        
        # 背景去除
        norm_f[mask_local == 0] = 0
        
        # 生成 Mask (用于结构判定)
        # 由于 Soft-Tanh 会把空气背景映射到接近 0 但非 0 的值 (例如 0~5 之间)
        # Otsu 通常能处理，但为了保险，可以在 Otsu 之前先做一个极小值的置零
        norm_8u = (norm_f * 255).astype(np.uint8)
        norm_8u[norm_8u < 5] = 0  # 显式清除空气底噪
        _, b_mask = cv2.threshold(norm_8u, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 保存结果
        clean_stack.append((norm_f * 65535).astype(np.uint16))
        binary_stack.append(b_mask)

    clean_stack = np.array(clean_stack) 

    # 3. 有效性判定 (Voting)
    valid_accumulator = np.zeros_like(mask_local, dtype=np.float32)
    for b_slice in binary_stack:
        eroded = cv2.erode(b_slice, kernel_sever)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(eroded)
        if num_labels > 1:
            largest_idx = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            temp_mask = (labels == largest_idx).astype(np.uint8)
            temp_mask = cv2.dilate(temp_mask, kernel_restore)
            temp_mask = cv2.bitwise_and(temp_mask, b_slice)
        else:
            temp_mask = np.zeros_like(b_slice)
        
        filled = ndimage.binary_fill_holes(temp_mask).astype(np.uint8)
        valid_accumulator += filled

    # 只要 90% 的层是好的，就保留
    validity_ratio = valid_accumulator / len(binary_stack)
    final_valid_map = (validity_ratio >= local_config['validity_ratio_threshold']).astype(np.uint8)
    
    # 4. 提取 NPY
    safe_zone = cv2.erode(final_valid_map, kernel_erode)
    ys, xs = np.where(safe_zone == 1)
    if len(ys) == 0: return 0
    
    count_saved = 0
    stride_xy = local_config['stride_xy']
    y_pts = range(np.min(ys), np.max(ys), stride_xy)
    x_pts = range(np.min(xs), np.max(xs), stride_xy)
    
    save_folder = local_config['dst_root']
    
    for py in y_pts:
        for px in x_pts:
            if safe_zone[py, px] == 1:
                half = crop_size // 2
                cube = clean_stack[:, py-half : py+half, px-half : px+half]
                if cube.shape == (crop_size, crop_size, crop_size):
                    fname = f"{folder_name}_z{z_start}_y{py}_x{px}.npy"
                    np.save(os.path.join(save_folder, fname), cube)
                    count_saved += 1
                    
    return count_saved

# ==============================================================================
#  主流程
# ==============================================================================
def run_global_stats_calculation():
    print("🚀 正在计算全局 P1/P99 (已启用 >100 像素过滤)...")
    global_pixel_reservoir = [] 

    for folder_name in CONFIG['target_folders']:
        folder_path = os.path.join(CONFIG['src_root'], folder_name)
        # files = sorted(glob.glob(os.path.join(folder_path, "*.tif")), key=lambda x: int(re.search(r'modif(\d+)', x).group(1)) if re.search(r'modif(\d+)', x) else 0)
        # 使用新排序
        files = sorted(glob.glob(os.path.join(folder_path, "*.tif")), key=natural_sort_key)
        # 打印一下，确保万无一失
        print(f"排序检查 (前3张): {[os.path.basename(f) for f in files[:3]]}")
        
        if not files: continue
        roi_info = _detect_global_roi(files, CONFIG['global_roi_sample'])
        if not roi_info: continue
        (cx, cy), r = roi_info

        sample_size = int(len(files) * CONFIG['stats_sample_rate'])
        sampled_files = random.sample(files, max(5, sample_size))

        for p in tqdm(sampled_files, leave=False):
            img = _read_img_raw(p)
            if img is None: continue
            
            mask = np.zeros(img.shape, dtype=np.uint8)
            cv2.circle(mask, (cx, cy), r, 1, -1)
            valid_pixels = img[mask == 1]
            
            # --- 优化：过滤掉空气背景 (值<=100) ---
            valid_pixels = valid_pixels[valid_pixels > 100]
            
            if len(valid_pixels) > CONFIG['stats_pixels_per_img']:
                valid_pixels = np.random.choice(valid_pixels, CONFIG['stats_pixels_per_img'])
            
            global_pixel_reservoir.append(valid_pixels)
    
    if not global_pixel_reservoir:
        print("❌ 无数据！")
        return

    all_data = np.concatenate(global_pixel_reservoir)
    p1 = np.percentile(all_data, 1)
    p99 = np.percentile(all_data, 99)
    print(f"\n✅ 新的统计值:\n'global_p1': {p1},\n'global_p99': {p99}")

def run_parallel_pipeline():
    print(f"🚀 启动并行流水线 | 核心数: {CONFIG['max_workers']}")
    
    if not os.path.exists(CONFIG['dst_root']):
        os.makedirs(CONFIG['dst_root'])

    all_tasks = []
    
    # 1. 准备任务
    for folder_name in CONFIG['target_folders']:
        folder_path = os.path.join(CONFIG['src_root'], folder_name)
        files = sorted(glob.glob(os.path.join(folder_path, "*.tif")), key=natural_sort_key)
        
        if len(files) < CONFIG['crop_size']: continue
        
        roi_info = _detect_global_roi(files, CONFIG['global_roi_sample'])
        if not roi_info: continue
            
        crop_size = CONFIG['crop_size']
        stride_z = CONFIG['stride_z']
        
        for z in range(0, len(files) - crop_size, stride_z):
            chunk_paths = files[z : z + crop_size]
            task = (chunk_paths, folder_name, z, roi_info, CONFIG)
            all_tasks.append(task)
            
    print(f"✅ 共 {len(all_tasks)} 个任务。开始处理...")

    # 2. 提交到进程池
    total_saved = 0
    with ProcessPoolExecutor(max_workers=CONFIG['max_workers']) as executor:
        futures = [executor.submit(process_single_chunk_task, task) for task in all_tasks]
        
        for future in tqdm(as_completed(futures), total=len(all_tasks)):
            try:
                total_saved += future.result()
            except Exception as e:
                import traceback
                traceback.print_exc()

    print(f"🎉 完成！共生成 {total_saved} 个数据块。")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    
    if CONFIG['run_mode'] == 'calculate_stats':
        run_global_stats_calculation()
    elif CONFIG['run_mode'] == 'pipeline':
        run_parallel_pipeline()
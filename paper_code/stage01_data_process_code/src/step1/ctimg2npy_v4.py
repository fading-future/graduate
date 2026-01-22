import os
import glob
import re
import numpy as np
import cv2
import random
from scipy import ndimage
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import multiprocessing


#  全局配置中心 (CONFIG) - 控制一切魔法参数
CONFIG = {
    # --- 两种运行模式 ---
    # 'calculate_stats': 仅运行统计，计算所有数据的全局 P1/P99 (第一步)
    # 'pipeline': 使用下方填入的 global_p1/p99 进行正式处理 (第二步)
    'run_mode': 'pipeline', 

    # --- 全局统计值 (从 calculate_stats 模式获得后填入此处) ---
    # 这一组参数将强制应用于所有岩心，确保物理密度的一致性
    'global_p1': 3500,     # 示例值，请替换
    'global_p99': 42000,   # 示例值，请替换

    # --- 文件路径配置 ---
    'src_root': r"D:\多尺度岩心数据集",
    'dst_root': r"D:\多尺度岩心数据集\Cleaned_NPY_Dataset_Final",
    
    # 需要处理的文件夹列表 (统计和处理都用这个列表)
    'target_folders': [
        "6-6-20 全部",
        "7-7-20 全部", 
        # "8-8-20 全部", 
    ],

    # --- 图像处理参数 ---
    'crop_size': 256,
    'stride_z': 32,
    'stride_xy': 32,
    
    # --- 降噪参数 ---
    'denoise_type': 'nlm',  # 'nlm', 'anisotropic', 'none'
    'denoise_h': 4,         # OpenCV NLM 强度

    # --- 形态学与ROI参数 ---
    'sever_size': 25,       # 粘连分离核大小
    'restore_size': 20,     # 还原核大小
    'margin_size': 10,      # 安全边距
    'global_roi_sample': 20,# ROI探测采样数

    # --- 统计计算采样参数 (仅用于 calculate_stats 模式) ---
    'stats_sample_rate': 0.1,    # 每个文件夹抽样 10% 的切片
    'stats_pixels_per_img': 10000, # 每张图抽样 1万个像素点 (防爆内存)
}

#  工具函数：读取 Raw 图片
def _read_img_raw(path):
    try:
        raw_data = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(raw_data, cv2.IMREAD_UNCHANGED)
        return img
    except:
        return None

#  工具函数：通用 ROI 探测 (适配不同岩心粗细)
def _detect_global_roi(files, sample_count=20):
    """
    针对给定的文件列表，探测其特有的岩心圆心和半径。
    """
    if not files: return None
    indices = np.linspace(0, len(files)-1, min(len(files), sample_count), dtype=int)
    all_centers = []
    all_radii = []

    for idx in indices:
        img = _read_img_raw(files[idx])
        if img is None: continue
        
        # 缩放加速
        scale = 0.2
        h, w = img.shape
        small = cv2.resize(img, (int(w*scale), int(h*scale)))
        
        mi, ma = small.min(), small.max()
        if ma - mi < 10: continue 
        
        small_8bit = ((small - mi) / (ma - mi + 1e-6) * 255).astype(np.uint8)
        _, thresh = cv2.threshold(small_8bit, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            c = max(contours, key=cv2.contourArea)
            (x, y), r = cv2.minEnclosingCircle(c)
            all_centers.append((x / scale, y / scale))
            all_radii.append(r / scale)
    
    if not all_centers: return None
    avg_center = np.median(all_centers, axis=0).astype(int)
    max_radius = int(np.percentile(all_radii, 90))
    return avg_center, max_radius

#  功能模块 1: 全局统计值计算器
def run_global_stats_calculation():
    print(f"\n{'='*60}")
    print(f"🚀 进入统计模式: 正在扫描 {len(CONFIG['target_folders'])} 个文件夹...")
    print(f"   目标: 解决岩心Mask大小不一问题，计算统一的 P1/P99")
    print(f"{'='*60}\n")

    global_pixel_reservoir = [] # 像素蓄水池

    for folder_name in CONFIG['target_folders']:
        folder_path = os.path.join(CONFIG['src_root'], folder_name)
        # 按数字排序
        files = sorted(glob.glob(os.path.join(folder_path, "*.tif")), 
                       key=lambda x: int(re.search(r'modif(\d+)', x).group(1)) if re.search(r'modif(\d+)', x) else 0)
        
        if not files: 
            print(f"⚠️ 跳过空文件夹: {folder_name}")
            continue

        # 1. 为当前文件夹探测它独有的 Mask (关键步骤)
        print(f"🔍 正在探测 {folder_name} 的 ROI...")
        roi_info = _detect_global_roi(files, CONFIG['global_roi_sample'])
        if not roi_info:
            print(f"❌ 无法识别 ROI: {folder_name}")
            continue
        (cx, cy), r = roi_info
        print(f"   -> 圆心: {cx, cy}, 半径: {r}")

        # 2. 随机抽样图片
        sample_size = int(len(files) * CONFIG['stats_sample_rate'])
        sampled_files = random.sample(files, max(5, sample_size)) # 至少采5张

        # 3. 提取有效像素
        print(f"   -> 正在抽样 {len(sampled_files)} 张切片数据...")
        for p in tqdm(sampled_files, leave=False):
            img = _read_img_raw(p)
            if img is None: continue
            
            # 创建当前图片的 Mask
            h, w = img.shape
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.circle(mask, (cx, cy), r, 1, -1)

            # 只取圆柱内的点
            valid_pixels = img[mask == 1]
            
            # 二次采样：每张图只取部分像素，防止内存溢出
            if len(valid_pixels) > CONFIG['stats_pixels_per_img']:
                valid_pixels = np.random.choice(valid_pixels, CONFIG['stats_pixels_per_img'])
            
            global_pixel_reservoir.append(valid_pixels)
    
    # 4. 汇总计算
    print("\n⚡ 正在合并所有样本并计算分位数 (这可能需要一点时间)...")
    if not global_pixel_reservoir:
        print("❌ 没有收集到任何有效数据！")
        return

    all_data = np.concatenate(global_pixel_reservoir)
    p1 = np.percentile(all_data, 1)
    p99 = np.percentile(all_data, 99)

    print(f"\n✅✅✅ 计算完成！请将以下数值填入 CONFIG 的 global_p1 和 global_p99：")
    print(f"global_p1  : {p1}")
    print(f"global_p99 : {p99}")
    print(f"{'='*60}\n")

#  功能模块 2: 并行处理流水线
# 并行工作函数 (Top-Level)
def process_slice_task(args):
    """
    Worker: 接收固定参数进行归一化，不再自适应计算
    """
    slice_data, mask_local, denoise_type, denoise_h, g_p1, g_p99 = args
    
    # 1. 快速检查是否为空白切片 (基于 Mask 内是否有内容)
    #    注意：这里不再计算 percentile，只看有没有非零值或均值，极大加速
    if np.sum(mask_local) == 0:
        return np.zeros_like(slice_data, dtype=np.uint16), np.zeros_like(slice_data, dtype=np.uint8)
    
    # 2. 这里的 slice_data 是 float32 吗？传入前最好确认，或者在这里转
    #    为了安全，先转 float
    img_f = slice_data.astype(np.float32)

    # 3. 全局一致性归一化 (核心修改点)
    #    使用 CONFIG 中传入的 g_p1, g_p99
    norm = (img_f - g_p1) / (g_p99 - g_p1 + 1e-6)
    norm = np.clip(norm, 0, 1)
    img_u8 = (norm * 255).astype(np.uint8)

    # 4. 降噪
    if denoise_type == 'nlm':
        denoised_u8 = cv2.fastNlMeansDenoising(img_u8, None, h=denoise_h, templateWindowSize=7, searchWindowSize=21)
    elif denoise_type == 'anisotropic':
        denoised_u8 = cv2.bilateralFilter(img_u8, 9, 75, 75)
    else:
        denoised_u8 = img_u8

    # 5. Masking
    denoised_u8[mask_local == 0] = 0

    # 6. 二值化
    _, b_mask = cv2.threshold(denoised_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 7. 转回 uint16
    cleaned_u16 = (denoised_u8.astype(np.float32) / 255.0 * 65535).astype(np.uint16)
    
    return cleaned_u16, (b_mask // 255).astype(np.uint8)

class EnhancedRockCorePipeline:
    def __init__(self):
        # 直接从 CONFIG 读取，不需要传参了
        self.max_workers = max(1, multiprocessing.cpu_count() - 4)
        print(f"初始化并行流水线: CPU核心 {self.max_workers}, 降噪: {CONFIG['denoise_type']}")
        
        self.kernel_sever = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (CONFIG['sever_size'], CONFIG['sever_size']))
        self.kernel_restore = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (CONFIG['restore_size'], CONFIG['restore_size']))
        self.kernel_erode = np.ones((CONFIG['crop_size'] + CONFIG['margin_size'], CONFIG['crop_size'] + CONFIG['margin_size']), np.uint8)

        if not os.path.exists(CONFIG['dst_root']):
            os.makedirs(CONFIG['dst_root'])

    def process_folder(self, folder_name):
        folder_path = os.path.join(CONFIG['src_root'], folder_name)
        files = sorted(glob.glob(os.path.join(folder_path, "*.tif")), 
                       key=lambda x: int(re.search(r'modif(\d+)', x).group(1)) if re.search(r'modif(\d+)', x) else 0)
        
        if len(files) < CONFIG['crop_size']: return

        # 1. 探测本文件夹的 Mask
        roi_info = _detect_global_roi(files, CONFIG['global_roi_sample'])
        if not roi_info: return
        (cx, cy), r = roi_info
        
        # 2. 准备裁剪
        r_pad = int(r * 1.05)
        y1, y2 = max(0, cy - r_pad), cy + r_pad
        x1, x2 = max(0, cx - r_pad), cx + r_pad
        roi_h, roi_w = y2 - y1, x2 - x1
        
        mask_local = np.zeros((roi_h, roi_w), dtype=np.uint8)
        cv2.circle(mask_local, (cx-x1, cy-y1), r, 1, -1)

        print(f"🚀 处理中: {folder_name} (R={r}) | 统一参数 P1={CONFIG['global_p1']}, P99={CONFIG['global_p99']}")

        # 3. Z轴滑动
        for z in tqdm(range(0, len(files) - CONFIG['crop_size'], CONFIG['stride_z']), desc=f"Folder {folder_name}"):
            chunk_paths = files[z : z + CONFIG['crop_size']]
            
            # 读取与裁剪 (IO密集，串行)
            raw_stack = []
            for p in chunk_paths:
                full_img = _read_img_raw(p)
                if full_img is None: 
                    raw_stack.append(np.zeros((roi_h, roi_w), dtype=np.uint8))
                else:
                    fh, fw = full_img.shape
                    crop = full_img[y1:min(y2, fh), x1:min(x2, fw)]
                    if crop.shape != (roi_h, roi_w):
                         padded = np.zeros((roi_h, roi_w), dtype=np.uint8)
                         padded[:crop.shape[0], :crop.shape[1]] = crop
                         raw_stack.append(padded)
                    else:
                        raw_stack.append(crop)
            
            stack = np.array(raw_stack)

            # 4. 并行计算 (传入 Global P1/P99)
            tasks = []
            for i in range(stack.shape[0]):
                tasks.append((
                    stack[i], 
                    mask_local, 
                    CONFIG['denoise_type'], 
                    CONFIG['denoise_h'],
                    CONFIG['global_p1'],  # <--- 注入魔法参数
                    CONFIG['global_p99']  # <--- 注入魔法参数
                ))

            with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                results = list(executor.map(process_slice_task, tasks))
            
            clean_stack = np.array([r[0] for r in results])
            binary_stack = [r[1] for r in results] # list 即可

            # 5. 有效性判定 (保持不变)
            valid_map = np.ones_like(mask_local)
            for b_slice in binary_stack:
                eroded = cv2.erode(b_slice, self.kernel_sever)
                num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(eroded)
                temp_mask = np.zeros_like(b_slice)
                if num_labels > 1:
                    largest_idx = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
                    temp_mask = (labels == largest_idx).astype(np.uint8)
                    temp_mask = cv2.dilate(temp_mask, self.kernel_restore)
                    temp_mask = cv2.bitwise_and(temp_mask, b_slice)
                filled = ndimage.binary_fill_holes(temp_mask).astype(np.uint8)
                valid_map = cv2.bitwise_and(valid_map, filled)

            # 6. 保存
            safe_zone = cv2.erode(valid_map, self.kernel_erode)
            ys, xs = np.where(safe_zone == 1)
            if len(ys) == 0: continue
            
            for py in range(np.min(ys), np.max(ys), CONFIG['stride_xy']):
                for px in range(np.min(xs), np.max(xs), CONFIG['stride_xy']):
                    if safe_zone[py, px] == 1:
                        half = CONFIG['crop_size'] // 2
                        rev = clean_stack[:, py-half : py+half, px-half : px+half]
                        if rev.shape == (CONFIG['crop_size'], CONFIG['crop_size'], CONFIG['crop_size']):
                            # 保存命名增加 P1/P99 信息，方便追溯
                            save_name = f"{folder_name}_z{z}_y{py}_x{px}.npy"
                            np.save(os.path.join(CONFIG['dst_root'], save_name), rev)

if __name__ == "__main__":
    multiprocessing.freeze_support()
    
    if CONFIG['run_mode'] == 'calculate_stats':
        # 模式一：算参数
        run_global_stats_calculation()
        
    elif CONFIG['run_mode'] == 'pipeline':
        # 模式二：跑任务
        if CONFIG['global_p1'] is None or CONFIG['global_p99'] is None:
            print("❌ 错误: 请先运行 'calculate_stats' 模式获取 global_p1 和 global_p99，并填入 CONFIG！")
        else:
            pipe = EnhancedRockCorePipeline()
            print(f"🎯 任务队列: 共 {len(CONFIG['target_folders'])} 个文件夹")
            for folder in CONFIG['target_folders']:
                pipe.process_folder(folder)
    else:
        print("未知的运行模式，请检查 CONFIG['run_mode']")
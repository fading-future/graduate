# import os
# import glob
# import re
# import numpy as np
# import cv2
# import matplotlib.pyplot as plt
# from tqdm import tqdm

# # ================= 配置区域 =================
# # 你的二值化图像文件夹路径
# SRC_FOLDER = r"D:\多尺度岩心数据集\6-6-24_Global_Consistency"
# # 输出诊断图保存位置
# OUTPUT_IMAGE = "Diagnostic_SideView_XZ.jpg"
# # 采样位置 (切中间)
# X_SLICE_POS = 422  # 假设图像宽 854，取中间 422
# IMG_SIZE = 854
# # ===========================================

# def strict_sort_key(filepath):
#     filename = os.path.basename(filepath)
#     match = re.search(r'modif(\d+)', filename)
#     if match: return int(match.group(1))
#     return 0

# def diagnose_continuity():
#     print(f"🔍 正在扫描文件夹: {SRC_FOLDER}")
    
#     # 1. 获取并排序文件
#     files = glob.glob(os.path.join(SRC_FOLDER, "*.tif"))
#     files = sorted(files, key=strict_sort_key)
    
#     if not files:
#         print("❌ 未找到 .tif 文件！请检查路径。")
#         return

#     print(f"✅ 找到 {len(files)} 个文件。")
#     print("⏳ 正在检查序号连续性...")
    
#     # 2. 检查序号是否有缺失
#     missing_indices = []
#     prev_idx = strict_sort_key(files[0])
#     for i in range(1, len(files)):
#         curr_idx = strict_sort_key(files[i])
#         if curr_idx != prev_idx + 1:
#             missing_indices.append((prev_idx, curr_idx))
#         prev_idx = curr_idx
    
#     if missing_indices:
#         print(f"⚠️ 警告: 发现 {len(missing_indices)} 处序号不连续！(可能是割裂的原因)")
#         for start, end in missing_indices[:5]:
#             print(f"   - 在 {start} 和 {end} 之间缺失")
#         if len(missing_indices) > 5: print("   - ...")
#     else:
#         print("✅ 文件序号完全连续 (0, 1, 2...)，无缺失。")

#     # 3. 生成 XZ 侧视图 (纵向切片)
#     print(f"⏳ 正在生成侧视图 (X={X_SLICE_POS})... 这可能需要一点时间")
    
#     xz_slice_pixels = []
    
#     # 为了速度，每 500 张打印一次进度
#     for i, p in enumerate(tqdm(files)):
#         try:
#             # 只读一行像素 (利用 np.fromfile 读取全部再 reshape 可能会慢，但对于 tif 只能这样)
#             # 优化：我们不需要解码整张图，但 opencv 需要。
#             # 这里我们只取中间那一列
            
#             # 支持中文路径读取
#             img_arr = np.fromfile(p, dtype=np.uint8)
#             img = cv2.imdecode(img_arr, cv2.IMREAD_GRAYSCALE)
            
#             if img is None:
#                 # 读失败补黑线
#                 col = np.zeros(IMG_SIZE, dtype=np.uint8) # 假设高度 1900
#             else:
#                 h, w = img.shape
#                 # 确保切片位置在范围内
#                 target_x = min(X_SLICE_POS, w-1)
#                 # 取第 target_x 列
#                 col = img[:, target_x]
            
#             xz_slice_pixels.append(col)
            
#         except Exception as e:
#             print(f"读图错误: {p} - {e}")
#             xz_slice_pixels.append(np.zeros(IMG_SIZE, dtype=np.uint8))

#     # 4. 堆叠并保存
#     # xz_slice_pixels 是 list of columns (H,)
#     # stack 后变成 (Z, H) -> 转置一下变成 (H, Z) 符合直觉
#     side_view = np.array(xz_slice_pixels).T 
    
#     # 二值化显示 (0, 255)
#     side_view = (side_view > 127).astype(np.uint8) * 255

#     print(f"💾 正在保存诊断图: {OUTPUT_IMAGE}")
#     cv2.imwrite(OUTPUT_IMAGE, side_view)
    
#     # 缩小显示在屏幕上 (如果支持)
#     plt.figure(figsize=(12, 6))
#     plt.title(f"XZ Side View (Raw Data Check) - X={X_SLICE_POS}")
#     # 因为 Z 轴很长，我们旋转一下看，或者压缩 Z 轴
#     plt.imshow(side_view, cmap='gray', aspect='auto')
#     plt.xlabel("Z (Slice Index)")
#     plt.ylabel("Y (Height)")
#     plt.tight_layout()
#     plt.show()
    
#     print("✅ 完成。请打开生成的 Diagnostic_SideView_XZ.jpg 查看。")
#     print("👉 如果这张图上有水平割裂线，说明是【原始数据】的问题，不是代码的问题。")

# if __name__ == "__main__":
#     diagnose_continuity()


import os

# ==============================================================================
# 限制底层库线程
# ==============================================================================
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["CV_NUM_THREADS"] = "1"

import glob
import re
import numpy as np
import cv2
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
from tqdm import tqdm
import matplotlib.pyplot as plt
import sys

# ==============================================================================
#  配置中心
# ==============================================================================
CONFIG = {
    'src_root': r"D:\多尺度岩心数据集",
    'dst_root': r"D:\多尺度岩心数据集\Final_Result_Sorted", 
    
    'target_folders': [
        "6-6-21_Global_Consistency"
    ],

    'crop_size': 256,   
    'stride_z': 64,    
    'stride_xy': 64,    

    'morph_close_size': 15,    
    'safety_margin': 5,        
    'validity_ratio_threshold': 0.95, 

    'max_workers': 4, 
    'debug_mode': True 
}

# ==============================================================================
#  核心修复：排序函数
# ==============================================================================
def strict_sort_key(filepath):
    """
    针对文件名：FdkRecon-ushort-1900x1900x9624.modif0139.tif
    必须提取 'modif' 后面的 '0139' 进行排序
    """
    filename = os.path.basename(filepath)
    match = re.search(r'modif(\d+)', filename)
    if match:
        return int(match.group(1))
    
    # 如果没找到 modif，尝试找最后的一串数字
    nums = re.findall(r'\d+', filename)
    if nums:
        return int(nums[-1])
    return 0

# ==============================================================================
#  工具函数
# ==============================================================================
def _read_binary_img(path):
    try:
        img_array = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
        if img is None: return None
        img = (img > 127).astype(np.uint8) 
        return img
    except:
        return None

def _fill_holes_and_close(binary_img, close_ksize):
    # 步骤 1：闭运算连接断裂处
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_ksize, close_ksize))
    closed = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, kernel)
    
    # 步骤 2：填充内部孔隙
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled_mask = np.zeros_like(binary_img)
    if contours:
        c = max(contours, key=cv2.contourArea)
        if cv2.contourArea(c) < 1000: return filled_mask
        cv2.drawContours(filled_mask, [c], -1, 1, -1)
        
    return filled_mask

def _save_debug_visualization(save_path, raw_img, filled_mask, safe_zone, crop_boxes):
    plt.figure(figsize=(18, 6))
    
    plt.subplot(1, 3, 1)
    plt.title("1. Raw Image (With Holes)")
    plt.imshow(raw_img, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title("2. Filled Mask (Used for Positioning)")
    plt.imshow(filled_mask, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title(f"3. Final Crop Areas")
    vis_img = cv2.cvtColor(raw_img * 255, cv2.COLOR_GRAY2RGB)
    
    # 红色区域表示不安全
    overlay = vis_img.copy()
    overlay[safe_zone == 0] = [255, 0, 0] 
    cv2.addWeighted(overlay, 0.3, vis_img, 0.7, 0, vis_img)

    for (x, y, w, h) in crop_boxes:
        cv2.rectangle(vis_img, (x, y), (x+w, y+h), (0, 255, 0), 5)

    plt.imshow(vis_img)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# ==============================================================================
#  处理核心
# ==============================================================================
def process_single_chunk_task(task_data):
    chunk_paths, folder_name, z_start, config, is_debug_target = task_data
    
    raw_stack_list = []
    filled_stack_list = []
    
    h_img, w_img = 1900, 1900 
    
    # 1. 读取 + 填充
    for p in chunk_paths:
        # 【关键】这里读入的是原始带孔隙的数据
        img = _read_binary_img(p)
        if img is None: img = np.zeros((h_img, w_img), dtype=np.uint8)
        else: h_img, w_img = img.shape
        
        raw_stack_list.append(img)
        
        # 【关键】这里生成用于计算位置的实心掩膜
        filled = _fill_holes_and_close(img, config['morph_close_size'])
        filled_stack_list.append(filled)
        
    raw_stack = np.array(raw_stack_list, dtype=np.uint8) 
    filled_stack = np.array(filled_stack_list, dtype=np.float32) 
    
    # 2. 计算安全区 (仅基于 Filled Stack)
    validity_map = np.mean(filled_stack, axis=0)
    safe_zone = (validity_map >= config['validity_ratio_threshold']).astype(np.uint8)
    
    kernel_erode = np.ones((config['safety_margin'], config['safety_margin']), np.uint8)
    safe_zone = cv2.erode(safe_zone, kernel_erode)
    
    # 3. 截取逻辑
    crop_size = config['crop_size']
    stride = config['stride_xy']
    count_saved = 0
    crop_boxes = [] 
    
    ys, xs = np.where(safe_zone == 1)
    
    if len(ys) > 0:
        y_min, y_max = np.min(ys), np.max(ys)
        x_min, x_max = np.min(xs), np.max(xs)
        save_folder = config['dst_root']
        
        # 严格限制搜索范围，加速
        y_range = range(max(0, y_min - crop_size), min(h_img, y_max), stride)
        x_range = range(max(0, x_min - crop_size), min(w_img, x_max), stride)

        for py in y_range:
            for px in x_range:
                if py + crop_size > h_img or px + crop_size > w_img: continue
                
                # 严格检查：截取框范围内必须全部是安全区 (Min == 1)
                mask_patch = safe_zone[py : py+crop_size, px : px+crop_size]
                if np.min(mask_patch) == 1:
                    
                    if not is_debug_target:
                        # 【关键】最后保存的是 raw_stack (带孔隙的)，而不是 filled_stack
                        cube = raw_stack[:, py : py+crop_size, px : px+crop_size]
                        fname = f"{folder_name}_z{z_start}_y{py}_x{px}.npy"
                        np.save(os.path.join(save_folder, fname), cube * 255)
                    
                    count_saved += 1
                    crop_boxes.append((px, py, crop_size, crop_size))

    # 4. 可视化
    if is_debug_target and config['debug_mode']:
        debug_dir = os.path.join(config['dst_root'], "debug_viz")
        os.makedirs(debug_dir, exist_ok=True)
        mid_idx = len(raw_stack) // 2
        viz_name = f"Check_{folder_name}_z{z_start}.jpg"
        _save_debug_visualization(
            os.path.join(debug_dir, viz_name),
            raw_stack[mid_idx], 
            filled_stack[mid_idx], 
            safe_zone, 
            crop_boxes
        )

    return count_saved

# ==============================================================================
#  主流程
# ==============================================================================
def run_pipeline():
    print(f"🚀 启动修复版流水线")
    
    if not os.path.exists(CONFIG['dst_root']):
        os.makedirs(CONFIG['dst_root'])

    all_tasks = []
    
    for folder_name in CONFIG['target_folders']:
        folder_path = os.path.join(CONFIG['src_root'], folder_name)
        
        # --- 核心修复：排序检查 ---
        files = glob.glob(os.path.join(folder_path, "*.tif"))
        files = sorted(files, key=strict_sort_key)
        
        print(f"\n📂 正在处理文件夹: {folder_name}")
        print(f"   共发现 {len(files)} 个文件")
        print(f"   [排序检查] 第1个文件:  {os.path.basename(files[0])}")
        print(f"   [排序检查] 第2个文件:  {os.path.basename(files[1])}")
        print(f"   [排序检查] 最后1个文件: {os.path.basename(files[-1])}")
        
        # 简单的人工确认机制
        # 如果你看到的数字不连续，程序会暂停让你看到
        # 实际上如果你批量跑，可以注释掉下面这行
        # input(">>> 请确认上述文件顺序是否正确 (数字连续递增)？按回车继续，Ctrl+C 退出...")

        if len(files) < CONFIG['crop_size']: continue
            
        crop_size = CONFIG['crop_size']
        stride_z = CONFIG['stride_z']
        
        chunks = range(0, len(files) - crop_size, stride_z)
        
        for i, z in enumerate(chunks):
            chunk_paths = files[z : z + crop_size]
            # 只可视化前 5 个 Chunk
            is_debug = (i < 5) 
            task = (chunk_paths, folder_name, z, CONFIG, is_debug)
            all_tasks.append(task)
            
    print(f"✅ 生成 {len(all_tasks)} 个任务。")

    total_saved = 0
    with ProcessPoolExecutor(max_workers=CONFIG['max_workers']) as executor:
        futures = [executor.submit(process_single_chunk_task, task) for task in all_tasks]
        
        for future in tqdm(as_completed(futures), total=len(all_tasks)):
            try:
                total_saved += future.result()
            except Exception as e:
                import traceback
                traceback.print_exc()

    print(f"🎉 完成！请检查 {os.path.join(CONFIG['dst_root'], 'debug_viz')}")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    run_pipeline()
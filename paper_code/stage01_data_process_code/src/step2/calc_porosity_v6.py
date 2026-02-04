import os
import matplotlib
# ==============================================================================
# 强制使用 'Agg' 后端
# ==============================================================================
matplotlib.use('Agg') 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import random
import warnings

warnings.filterwarnings('ignore')

# ================= 核心配置区域 =================
CONFIG = {
    # ---------------- 1. 路径配置 ----------------
    "src_root": r"D:\多尺度岩心数据集\cleaned_npy_dataset",           
    "dst_root": r"E:\aligned_Training_Data_Binary", 
    "preview_dir": r"./preview_cache", 
    
    # ---------------- 2. 部分处理控制 (新增功能) ----------------
    "PARTIAL_CONFIG": {
        "enabled": True,        # 【开关】 True=只处理一部分; False=处理全部
        "mode": "first_n",      # 模式: 'first_n'(前N个), 'random_n'(随机N个), 'keyword'(关键词)
        "value": 2000,           # 数值: 如果是n模式填数字(如100), 如果是keyword填字符串(如"sandstone")
        "seed": 42              # 随机种子(保证每次随机选的文件是一样的)
    },

    # ---------------- 3. 并行配置 ----------------
    "num_workers": 16,       # Windows下建议不要设太大，物理核心数即可
    "chunk_size": 10,        
    
    # ---------------- 4. 算法参数 ----------------
    "MANUAL_TARGET_PEAK": 35000,  
    "USE_MANUAL_TARGET": True,    
    
    "KNEE_START": 60000,          
    "DTYPE_MAX": 65535,           
    
    "noise_floor": 200,           
    "histogram_bins": 65536,
    "calib_samples": 3000,        
    "preview_count": 4            
}
# =======================================================

def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)

def get_peak_uint16(data):
    flat = data.ravel()
    counts = np.bincount(flat, minlength=CONFIG["histogram_bins"])
    counts[:CONFIG["noise_floor"]] = 0
    return np.argmax(counts)

def filter_file_list(all_files):
    """
    根据配置筛选文件列表
    """
    cfg = CONFIG["PARTIAL_CONFIG"]
    if not cfg["enabled"]:
        return all_files
    
    total = len(all_files)
    mode = cfg["mode"]
    val = cfg["value"]
    
    print(f"\n✂️  [筛选模式开启] Mode={mode}, Value={val}")
    
    selected_files = []
    
    if mode == "first_n":
        count = min(total, int(val))
        selected_files = all_files[:count]
        
    elif mode == "random_n":
        count = min(total, int(val))
        random.seed(cfg["seed"])
        selected_files = random.sample(all_files, count)
        
    elif mode == "keyword":
        # 筛选路径中包含特定字符串的文件
        keyword = str(val)
        selected_files = [f for f in all_files if keyword in str(f)]
        
    print(f"   筛选结果: 从 {total} 个文件中选中了 {len(selected_files)} 个")
    return selected_files

def soft_highlight_compression(data, knee_start=60000, max_val=65535):
    """
    预览时仍需要此函数来生成直观图像
    """
    over_knee = data > knee_start
    if not np.any(over_knee): return data
    x = data[over_knee].astype(np.float32)
    range_width = max_val - knee_start
    if range_width < 1: range_width = 1
    normalized_input = (x - knee_start) / range_width
    compressed_output = knee_start + range_width * np.tanh(normalized_input)
    data[over_knee] = compressed_output
    return data

def worker_calc_peak(file_path):
    try:
        d = np.load(file_path)
        return get_peak_uint16(d)
    except Exception:
        return None

def worker_process_file_optimized(args):
    """
    【极速版 Worker】
    不进行 float32/tanh 运算，直接逆推阈值生成 0/1 Mask
    """
    file_path, target_peak, offset, dst_root, src_root = args
    try:
        path_obj = Path(file_path)
        rel_path = path_obj.relative_to(src_root)
        save_path = Path(dst_root) / rel_path
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 1. 加载
        data = np.load(file_path)
        curr_peak = get_peak_uint16(data)
        if curr_peak < 10: curr_peak = 10 
        
        # 2. 计算缩放因子
        scale_factor = target_peak / float(curr_peak)
        
        # 3. 逆向计算阈值 (Inverse Thresholding)
        # 逻辑：aligned < (Target - Offset)  ==>  raw * scale < (Target - Offset)
        # ==> raw < (Target - Offset) / scale
        target_threshold = target_peak - offset
        raw_threshold = target_threshold / scale_factor
        
        # 4. 直接生成二值 Mask (利用 Numpy 广播机制，极快)
        # 假设小于阈值是孔隙(1)，大于等于阈值是岩石(0)
        mask = data < raw_threshold
        
        # 5. 转为 uint8 (0/1) 并保存
        np.save(save_path, mask.astype(np.uint8))
        
        # 6. 计算统计量
        porosity = np.sum(mask) / mask.size
        
        return {
            "file": path_obj.name, 
            "rel_path": str(rel_path), 
            "porosity": porosity, 
            "scale_factor": scale_factor,
            "orig_peak": curr_peak,
            "status": "ok"
        }
    except Exception as e:
        return {
            "file": str(file_path), 
            "rel_path": str(rel_path) if 'rel_path' in locals() else "unknown",
            "status": "error", 
            "msg": str(e)
        }

def get_target_peak(src_root):
    # 先获取全部文件列表
    all_files_raw = list(Path(src_root).rglob("*.npy"))
    if not all_files_raw: raise FileNotFoundError(f"未找到文件: {src_root}")

    # 【关键修改】在此处应用筛选逻辑
    selected_files = filter_file_list(all_files_raw)

    if CONFIG["USE_MANUAL_TARGET"]:
        print(f"\n[Step 1] 使用手动 Target Peak: {CONFIG['MANUAL_TARGET_PEAK']}")
        return CONFIG["MANUAL_TARGET_PEAK"], selected_files
    else:
        # 如果自动模式，只用筛选后的文件来计算平均峰值，速度更快
        print(f"\n[Step 1] 自动扫描 Target Peak (基于筛选后的 {len(selected_files)} 个文件)...")
        sample_count = min(len(selected_files), CONFIG["calib_samples"])
        sample_files = np.random.choice(selected_files, sample_count, replace=False)
        
        with ProcessPoolExecutor(max_workers=CONFIG["num_workers"]) as executor:
            results = list(tqdm(executor.map(worker_calc_peak, sample_files), total=len(sample_files)))
        
        valid = [p for p in results if p is not None]
        avg = int(np.mean(valid))
        print(f"✅ Auto Target Peak = {avg}")
        return avg, selected_files

def save_preview_image(files_to_show, target_peak, offset, filename="preview.png"):
    """
    预览图生成 (依然保留完整视觉逻辑，方便你调参)
    """
    plt.close('all')
    n = len(files_to_show)
    if n == 0: return None
    fig, axes = plt.subplots(n, 3, figsize=(15, 4 * n))
    if n == 1: axes = axes.reshape(1, -1)
    
    threshold = target_peak - offset

    for i, f_path in enumerate(files_to_show):
        raw = np.load(f_path) 
        curr_peak = get_peak_uint16(raw)
        if curr_peak < 10: curr_peak = 10
        scale_factor = target_peak / float(curr_peak)
        
        # 预览为了好看，还是模拟一下对齐效果
        d_float = raw.astype(np.float32) * scale_factor
        d_float = soft_highlight_compression(d_float, CONFIG["KNEE_START"], CONFIG["DTYPE_MAX"])
        aligned = np.clip(d_float, 0, CONFIG["DTYPE_MAX"]).astype(np.uint16)
        
        mask = aligned < threshold
        
        mid_idx = raw.shape[0] // 2
        sl_raw = raw[mid_idx]
        sl_align = aligned[mid_idx]
        sl_mask = mask[mid_idx]

        axes[i, 0].imshow(sl_raw, cmap='gray', vmin=0, vmax=65535)
        axes[i, 0].set_title(f"Raw\n{f_path.name}")
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(sl_align, cmap='gray', vmin=0, vmax=65535)
        axes[i, 1].set_title(f"Simulated View\n(For Visual Check)")
        axes[i, 1].axis('off')
        
        # 展示二值化结果 (白=1孔隙, 黑=0岩石)
        axes[i, 2].imshow(sl_mask.astype(np.uint8), cmap='gray', vmin=0, vmax=1) 
        axes[i, 2].set_title(f"Binary Result\nTh={threshold}", color='blue')
        axes[i, 2].axis('off')

    save_p = os.path.join(CONFIG["preview_dir"], filename)
    plt.tight_layout()
    plt.savefig(save_p)
    plt.close('all')
    return save_p

def interactive_loop(src_root, target_peak, all_files):
    print("\n" + "="*60)
    print("进入 [交互式预览模式]")
    print(f"请在 VSCode/文件夹 查看生成的图片: {CONFIG['preview_dir']}")
    print("="*60)
    
    current_offset = 2000 
    if not all_files: return current_offset

    while True:
        print(f"\n🔵 Target: {target_peak} | Offset: {current_offset} | Seg_Threshold: {target_peak - current_offset}")
        
        # 如果文件少于预览数，就全显示
        count = min(len(all_files), CONFIG['preview_count'])
        samples = random.sample(all_files, count)
        
        img_path = save_preview_image(samples, target_peak, current_offset, filename=f"preview_off_{current_offset}.png")
        
        print(f"🖼️  预览图: {img_path}")
        cmd = input(">>> 输入新Offset / 'run' 开始 / 'check' 换图: ").strip().lower()
        
        if cmd in ['run', 'go', 'yes']:
            plt.close('all')
            return current_offset
        elif cmd == 'check':
            continue
        else:
            try:
                current_offset = int(cmd)
            except ValueError:
                print("❌ 输入无效")

def main():
    ensure_dir(CONFIG["preview_dir"])

    # 1. 获取并筛选文件列表
    target_peak, selected_files = get_target_peak(CONFIG["src_root"])
    
    if not selected_files:
        print("❌ 筛选后没有文件被选中，请检查 PARTIAL_CONFIG 设置。")
        return

    # 2. 交互式调参 (只用筛选出来的这部分文件做预览)
    final_offset = interactive_loop(CONFIG["src_root"], target_peak, selected_files)
    
    # 3. 极速批量处理
    print(f"\n🚀 开始批量处理 {len(selected_files)} 个文件... (Target={target_peak}, Offset={final_offset})")
    
    tasks = [(str(f), target_peak, final_offset, CONFIG["dst_root"], CONFIG["src_root"]) for f in selected_files]
    
    results = []
    with ProcessPoolExecutor(max_workers=CONFIG["num_workers"]) as executor:
        for res in tqdm(executor.map(worker_process_file_optimized, tasks, chunksize=CONFIG["chunk_size"]), 
                        total=len(tasks), desc="Processing"):
            results.append(res)
            
    df = pd.DataFrame(results)
    csv_path = os.path.join(CONFIG["dst_root"], "processing_report.csv")
    if "porosity" in df.columns:
        df = df.sort_values(by="porosity")
    df.to_csv(csv_path, index=False)
    
    print(f"\n✅ 全部完成！已保存 {len(df)} 个二值化文件。")

if __name__ == "__main__":
    main()
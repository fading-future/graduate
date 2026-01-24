import os
import matplotlib
# ==============================================================================
# 关键修复: 在导入 pyplot 之前，强制使用 'Agg' 后端
# 这会禁用所有 GUI 窗口，只进行文件生成，彻底解决 Tkinter/多线程冲突报错
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

# 忽略除以零等非致命警告
warnings.filterwarnings('ignore')

# ================= 核心配置区域 =================
CONFIG = {
    # ---------------- 路径配置 (请修改这里) ----------------
    "src_root": r"D:\多尺度岩心数据集\cleaned_npy_dataset",          
    "dst_root": r"E:\aligned_Training_Data", 
    "preview_dir": r"./preview_cache",  # 预览图缓存路径
    
    # ---------------- 并行配置 ----------------
    "num_workers": 24,       # 核心数 (建议留少量余量)
    "chunk_size": 10,        # 任务分块
    
    # ---------------- 算法核心参数 (关键修改) ----------------
    # 1. 动态范围保留：强制设定目标峰值为 35000 (uint16的一半多一点)
    #    这给高亮矿物留出了 ~30000 的空间，防止乘法后溢出。
    "MANUAL_TARGET_PEAK": 35000,  
    "USE_MANUAL_TARGET": True,    # 开启手动模式
    
    # 2. 高光滚降参数
    "KNEE_START": 60000,          # 超过此值开始软压缩
    "DTYPE_MAX": 65535,           # uint16 极限
    
    # 3. 基础参数
    "noise_floor": 200,           # 忽略背景底噪
    "histogram_bins": 65536,
    "calib_samples": 3000,        # 自动扫描时的采样数(仅在USE_MANUAL_TARGET=False时用)
    "preview_count": 4            # 预览张数
}
# =======================================================

def ensure_dir(path):
    """确保目录存在"""
    Path(path).mkdir(parents=True, exist_ok=True)

def get_peak_uint16(data):
    """
    寻找 uint16 数据的骨架峰值
    使用 bincount 极速计算直方图
    """
    flat = data.ravel()
    counts = np.bincount(flat, minlength=CONFIG["histogram_bins"])
    counts[:CONFIG["noise_floor"]] = 0
    return np.argmax(counts)

def soft_highlight_compression(data, knee_start=60000, max_val=65535):
    """
    [核心算法] 高光柔性滚降
    对于 > knee_start 的值，使用 tanh 进行平滑压缩，防止硬截断。
    """
    # 找出需要压缩的高光区域
    over_knee = data > knee_start
    if not np.any(over_knee):
        return data

    # 提取高光数据
    x = data[over_knee].astype(np.float32)
    
    # 算法：非线性映射
    # y = knee + (range) * tanh( (x - knee) / range )
    range_width = max_val - knee_start
    # 防止除以零保护
    if range_width < 1: range_width = 1
    
    normalized_input = (x - knee_start) / range_width
    # tanh 将输入映射到 (0, 1)，再乘回 range_width
    compressed_output = knee_start + range_width * np.tanh(normalized_input)
    
    # 写回原数组
    data[over_knee] = compressed_output
    return data

def worker_calc_peak(file_path):
    """Step 1 并行工兵：仅计算峰值 (用于自动模式)"""
    try:
        d = np.load(file_path)
        return get_peak_uint16(d)
    except Exception:
        return None

def worker_process_file_optimized(args):
    """Step 3 并行工兵：处理单个文件 (乘性对齐 + 软压缩 + 分割 + 保存)"""
    file_path, target_peak, offset, dst_root, src_root = args
    try:
        path_obj = Path(file_path)
        rel_path = path_obj.relative_to(src_root)
        save_path = Path(dst_root) / rel_path
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 1. 加载数据
        data = np.load(file_path)
        curr_peak = get_peak_uint16(data)
        
        # 2. 计算乘性缩放因子
        if curr_peak < 10: curr_peak = 10 # 保护
        scale_factor = target_peak / float(curr_peak)
        
        # 3. 应用线性拉伸 (使用 float32 防止溢出)
        data_float = data.astype(np.float32) * scale_factor
        
        # 4. 应用高光柔性滚降
        data_float = soft_highlight_compression(
            data_float, 
            knee_start=CONFIG["KNEE_START"], 
            max_val=CONFIG["DTYPE_MAX"]
        )
        
        # 5. 安全转回 uint16
        data_aligned = np.clip(data_float, 0, CONFIG["DTYPE_MAX"]).astype(np.uint16)
        
        # 6. 分割逻辑
        threshold = target_peak - offset
        mask = data_aligned < threshold 
        
        # 7. 计算孔隙度
        porosity = np.sum(mask) / mask.size
        
        # 8. 保存
        np.save(save_path, data_aligned)
        
        # 统计截断率 (用于监控)
        clip_ratio = np.sum(data_aligned == CONFIG["DTYPE_MAX"]) / data_aligned.size
        
        return {
            "file": path_obj.name, 
            "rel_path": str(rel_path), 
            "porosity": porosity, 
            "scale_factor": scale_factor,
            "orig_peak": curr_peak,
            "clip_ratio": clip_ratio,
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
    """获取目标峰值 (手动或自动)"""
    if CONFIG["USE_MANUAL_TARGET"]:
        print(f"\n[Step 1] 使用手动设定 Target Peak: {CONFIG['MANUAL_TARGET_PEAK']}")
        print(f"   (目的：保留 Headroom，防止高亮像素溢出截断)")
        all_files = list(Path(src_root).rglob("*.npy"))
        return CONFIG["MANUAL_TARGET_PEAK"], all_files
    else:
        print(f"\n[Step 1] 自动扫描计算 Target Peak...")
        all_files = list(Path(src_root).rglob("*.npy"))
        if not all_files: raise FileNotFoundError(f"未找到文件: {src_root}")
        
        sample_count = min(len(all_files), CONFIG["calib_samples"])
        sample_files = np.random.choice(all_files, sample_count, replace=False)
        
        with ProcessPoolExecutor(max_workers=CONFIG["num_workers"]) as executor:
            results = list(tqdm(executor.map(worker_calc_peak, sample_files), total=len(sample_files)))
        
        valid = [p for p in results if p is not None]
        avg = int(np.mean(valid))
        print(f"✅ Auto Target Peak = {avg}")
        return avg, all_files

def save_preview_image(files_to_show, target_peak, offset, filename="preview.png"):
    """
    生成预览图 (必须与 Worker 逻辑完全一致)
    """
    # 显式关闭之前可能残留的图像，释放内存
    plt.close('all')
    
    n = len(files_to_show)
    fig, axes = plt.subplots(n, 3, figsize=(15, 4 * n))
    if n == 1: axes = axes.reshape(1, -1)
    
    threshold = target_peak - offset

    for i, f_path in enumerate(files_to_show):
        raw = np.load(f_path) 
        curr_peak = get_peak_uint16(raw)
        
        # --- 模拟 Worker 的完整逻辑 ---
        if curr_peak < 10: curr_peak = 10
        scale_factor = target_peak / float(curr_peak)
        
        # 1. 线性拉伸
        d_float = raw.astype(np.float32) * scale_factor
        # 2. 软压缩
        d_float = soft_highlight_compression(d_float, CONFIG["KNEE_START"], CONFIG["DTYPE_MAX"])
        # 3. 转回 uint16
        aligned = np.clip(d_float, 0, CONFIG["DTYPE_MAX"]).astype(np.uint16)
        # ---------------------------
        
        # 分割
        mask = aligned < threshold
        porosity = np.sum(mask) / mask.size
        
        # 取中间切片
        mid_idx = raw.shape[0] // 2
        sl_raw = raw[mid_idx]
        sl_align = aligned[mid_idx]
        sl_mask = mask[mid_idx]

        # 绘图
        # 1. Raw
        axes[i, 0].imshow(sl_raw, cmap='gray', vmin=0, vmax=65535)
        axes[i, 0].set_title(f"Raw (Peak={curr_peak})\n{f_path.name}")
        axes[i, 0].axis('off')
        
        # 2. Aligned
        axes[i, 1].imshow(sl_align, cmap='gray', vmin=0, vmax=65535)
        axes[i, 1].set_title(f"Aligned (Scale={scale_factor:.2f})\nTarget={target_peak}")
        axes[i, 1].axis('off')
        
        # 3. Seg
        axes[i, 2].imshow(~sl_mask, cmap='gray', vmin=0, vmax=1) 
        axes[i, 2].set_title(f"Seg (Th={threshold}) Phi={porosity:.2%}\n[Black=Pore]", color='red')
        axes[i, 2].axis('off')

    save_p = os.path.join(CONFIG["preview_dir"], filename)
    plt.tight_layout()
    plt.savefig(save_p)
    plt.close('all') # 生成完立即关闭
    return save_p

def interactive_loop(src_root, target_peak, all_files):
    """Step 2: 交互式预览"""
    print("\n" + "="*60)
    print("进入 [交互式预览模式]")
    print(f"请在 VSCode 左侧打开文件夹: {CONFIG['preview_dir']} 查看生成的图片")
    print("="*60)
    
    current_offset = 2000 # 初始经验值
    if not all_files: return current_offset

    while True:
        print(f"\n🔵 Target: {target_peak} | Offset: {current_offset} | Seg_Threshold: {target_peak - current_offset}")
        
        samples = random.sample(all_files, min(len(all_files), CONFIG['preview_count']))
        img_path = save_preview_image(samples, target_peak, current_offset, filename=f"preview_off_{current_offset}.png")
        
        print(f"🖼️  预览图: {img_path}")
        cmd = input(">>> 输入新Offset / 'run' 开始 / 'check' 换图: ").strip().lower()
        
        if cmd in ['run', 'go', 'yes']:
            # 退出前再次清理，防止带入多进程
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

    # Step 1: 确定 Target Peak (代码中强制使用了手动模式 35000)
    target_peak, all_files = get_target_peak(CONFIG["src_root"])
    
    # Step 2: 交互式确定 Threshold Offset
    final_offset = interactive_loop(CONFIG["src_root"], target_peak, all_files)
    
    # Step 3: 批量并行处理
    print(f"\n🚀 开始批量处理... (Target={target_peak}, Offset={final_offset})")
    print(f"   算法: 乘性对齐 + Tanh高光滚降 (Knee={CONFIG['KNEE_START']})")
    
    tasks = [(str(f), target_peak, final_offset, CONFIG["dst_root"], CONFIG["src_root"]) for f in all_files]
    
    results = []
    # 使用上下文管理器确保安全
    with ProcessPoolExecutor(max_workers=CONFIG["num_workers"]) as executor:
        for res in tqdm(executor.map(worker_process_file_optimized, tasks, chunksize=CONFIG["chunk_size"]), 
                        total=len(tasks), desc="Processing"):
            results.append(res)
            
    # Step 4: 报告
    df = pd.DataFrame(results)
    csv_path = os.path.join(CONFIG["dst_root"], "processing_report.csv")
    
    if "porosity" in df.columns:
        df = df.sort_values(by="porosity")
        
    df.to_csv(csv_path, index=False)
    
    # 打印截断警告
    if 'clip_ratio' in df.columns:
        clipped = df[df['clip_ratio'] > 0.01]
        print(f"\n✅ 全部完成！")
        print(f"📄 报告: {csv_path}")
        if not clipped.empty:
            print(f"⚠️ 警告: 有 {len(clipped)} 个文件高光截断率仍 > 1%，请检查是否需要降低 Target Peak。")
        else:
            print(f"✨ 完美: 所有文件高光截断率均 < 1%。")
    else:
        print("完成，但未检测到截断率数据。")

if __name__ == "__main__":
    main()
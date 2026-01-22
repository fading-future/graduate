import numpy as np
import os
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
    # 路径配置
    "src_root": r"/chendou_space/data/cleaned_npy_dataset",          
    "dst_root": r"/chendou_space/data/aligned_Training_Data_Interactive", 
    "preview_dir": r"./preview_cache",  # 专门存放预览图的文件夹
    
    # 并行配置
    "num_workers": 90,       # 96核机器建议留余量，防止系统卡顿
    "chunk_size": 10,        # 任务分块大小，减少进程通信开销
    
    # 算法参数
    "calib_samples": 3000,   # 计算基准峰值时的采样数
    "preview_count": 4,      # 每次预览抽取的图片数
    
    # 数据特性 (uint16)
    "dtype_range": (0, 65535),     # 保持uint16的物理极限，不要改成P99，防止对齐移位后被截断
    "noise_floor": 200,            # 忽略极低值背景噪点
    "histogram_bins": 65536        # 直方图分辨率
}
# ===========================================

def ensure_dir(path):
    """确保目录存在"""
    Path(path).mkdir(parents=True, exist_ok=True)

def get_peak_uint16(data):
    """
    寻找 uint16 数据的骨架峰值
    使用 bincount 极速计算直方图
    """
    flat = data.ravel()
    # minlength确保输出长度固定为65536
    counts = np.bincount(flat, minlength=CONFIG["histogram_bins"])
    # 屏蔽低值噪点
    counts[:CONFIG["noise_floor"]] = 0
    return np.argmax(counts)

def worker_calc_peak(file_path):
    """Step 1 并行工兵：仅计算峰值"""
    try:
        d = np.load(file_path)
        return get_peak_uint16(d)
    except Exception:
        return None

def worker_process_file(args):
    """Step 3 并行工兵：处理单个文件 (对齐 + 分割 + 保存)"""
    file_path, target_peak, offset, dst_root, src_root = args
    try:
        path_obj = Path(file_path)
        rel_path = path_obj.relative_to(src_root)
        save_path = Path(dst_root) / rel_path
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 1. 加载数据
        data = np.load(file_path)
        
        # 2. 对齐逻辑 (Alignment)
        curr_peak = get_peak_uint16(data)
        shift = int(target_peak) - int(curr_peak) # 转int防止溢出
        
        # 使用clip防止溢出uint16范围 (0-65535)
        # 注意：这里不用P99截断，因为移位后高亮像素超过P99是合理的
        data_aligned = np.clip(data.astype(np.int32) + shift, 0, 65535).astype(np.uint16)
        
        # 3. 分割逻辑 (Segmentation)
        # 这里的阈值是基于全局Target Peak确定的，保证了物理意义的一致性
        threshold = target_peak - offset
        mask = data_aligned < threshold 
        
        # 4. 计算孔隙度 (Porosity)
        porosity = np.sum(mask) / mask.size
        
        # 5. 保存处理后的数据
        np.save(save_path, data_aligned)
        
        # 返回详细信息用于CSV记录
        return {
            "file": path_obj.name, 
            "rel_path": str(rel_path), 
            "porosity": porosity, 
            "shift": shift,
            "target_peak": target_peak,
            "offset": offset,
            "status": "ok"
        }
    except Exception as e:
        return {
            "file": str(file_path), 
            "rel_path": str(rel_path) if 'rel_path' in locals() else "unknown",
            "status": "error", 
            "msg": str(e)
        }

def auto_calculate_target_peak(src_root):
    """Step 1: 自动计算全数据集 Target Peak"""
    print(f"\n[Step 1] 扫描文件计算 Target Peak (多核并行)...")
    all_files = list(Path(src_root).rglob("*.npy"))
    if not all_files:
        raise FileNotFoundError(f"未找到文件: {src_root}")

    sample_count = min(len(all_files), CONFIG["calib_samples"])
    sample_files = np.random.choice(all_files, sample_count, replace=False)
    
    with ProcessPoolExecutor(max_workers=CONFIG["num_workers"]) as executor:
        results = list(tqdm(executor.map(worker_calc_peak, sample_files, chunksize=CONFIG["chunk_size"]), 
                            total=len(sample_files), desc="Calc Peak"))
        
    valid_peaks = [p for p in results if p is not None]
    if not valid_peaks:
        raise ValueError("无法计算峰值，请检查数据")
        
    avg_peak = int(np.mean(valid_peaks))
    print(f"✅ Target Peak (Mean) = {avg_peak}")
    return avg_peak, all_files

def save_preview_image(files_to_show, target_peak, offset, filename="preview_current.png"):
    """
    生成预览图 (解决 Matplotlib 自动缩放导致的'全黑'误解)
    """
    n = len(files_to_show)
    fig, axes = plt.subplots(n, 3, figsize=(15, 4 * n))
    if n == 1: axes = axes.reshape(1, -1)
    
    threshold = target_peak - offset

    for i, f_path in enumerate(files_to_show):
        raw = np.load(f_path) 
        curr_peak = get_peak_uint16(raw)
        
        # 模拟对齐
        shift = int(target_peak) - int(curr_peak)
        aligned = np.clip(raw.astype(np.int32) + shift, 0, 65535).astype(np.uint16)
        
        # 模拟分割
        mask = aligned < threshold
        porosity = np.sum(mask) / mask.size
        
        # 取中间切片
        mid_idx = raw.shape[0] // 2
        sl_raw = raw[mid_idx]
        sl_align = aligned[mid_idx]
        sl_mask = mask[mid_idx]

        # 1. Raw Image
        axes[i, 0].imshow(sl_raw, cmap='gray', vmin=0, vmax=65535)
        axes[i, 0].set_title(f"Raw (Peak={curr_peak})\n{f_path.name}")
        axes[i, 0].axis('off')
        
        # 2. Aligned Image
        axes[i, 1].imshow(sl_align, cmap='gray', vmin=0, vmax=65535)
        axes[i, 1].set_title(f"Aligned (Shift={shift})\nTarget={target_peak}")
        axes[i, 1].axis('off')
        
        # 3. Seg Image (Critical Fix: vmin/vmax)
        # ~sl_mask: True(1)变为Matrix(白), False(0)变为Pore(黑)
        # 显式指定 vmin=0, vmax=1，防止全致密切片显示为全黑
        axes[i, 2].imshow(~sl_mask, cmap='gray', vmin=0, vmax=1) 
        axes[i, 2].set_title(f"Seg (Th={threshold}) Phi={porosity:.2%}\n[Black=Pore, White=Matrix]", color='red')
        axes[i, 2].axis('off')

    save_p = os.path.join(CONFIG["preview_dir"], filename)
    plt.tight_layout()
    plt.savefig(save_p)
    plt.close()
    return save_p

def interactive_loop(src_root, target_peak):
    """Step 2: 交互式参数确认"""
    print("\n" + "="*60)
    print("进入 [交互式预览模式]")
    print(f"请在 VSCode 左侧打开文件夹: {CONFIG['preview_dir']} 查看生成的图片")
    print("="*60)
    
    current_offset = 2000 # 初始经验值
    all_files = list(Path(src_root).rglob("*.npy"))
    
    if not all_files: return current_offset

    while True:
        print(f"\n🔵 Target Peak: {target_peak} | Offset: {current_offset} | Threshold: {target_peak - current_offset}")
        
        # 随机采样并绘图
        samples = random.sample(all_files, min(len(all_files), CONFIG['preview_count']))
        img_path = save_preview_image(samples, target_peak, current_offset, filename=f"preview_off_{current_offset}.png")
        
        print(f"🖼️  预览图已更新: {img_path}")
        cmd = input(">>> 请输入: 新Offset数值 / 'run' 开始处理 / 'check' 换一组图: ").strip().lower()
        
        if cmd in ['run', 'go', 'yes']:
            return current_offset
        elif cmd == 'check':
            continue
        else:
            try:
                current_offset = int(cmd)
            except ValueError:
                print("❌ 输入无效，请输入整数Offset")

def main():
    ensure_dir(CONFIG["preview_dir"])

    # Step 1: 自动计算基准
    target_peak, all_files = auto_calculate_target_peak(CONFIG["src_root"])
    
    # Step 2: 交互式确定阈值偏移量
    final_offset = interactive_loop(CONFIG["src_root"], target_peak)
    
    # Step 3: 批量并行处理
    print(f"\n🚀 开始批量处理... (Target={target_peak}, Offset={final_offset})")
    
    tasks = [(str(f), target_peak, final_offset, CONFIG["dst_root"], CONFIG["src_root"]) for f in all_files]
    
    results = []
    with ProcessPoolExecutor(max_workers=CONFIG["num_workers"]) as executor:
        # 使用 tqdm 显示进度
        for res in tqdm(executor.map(worker_process_file, tasks, chunksize=CONFIG["chunk_size"]), 
                        total=len(tasks), desc="Processing"):
            results.append(res)
            
    # Step 4: 保存完整报告
    df = pd.DataFrame(results)
    csv_path = os.path.join(CONFIG["dst_root"], "processing_report.csv")
    
    # 按照 status 和 porosity 排序，方便后续检查极端数据
    if "porosity" in df.columns:
        df = df.sort_values(by="porosity")
        
    df.to_csv(csv_path, index=False)
    
    print(f"\n✅ 处理全部完成！")
    print(f"📄 详细报告: {csv_path}")
    print(f"💡 提示: 建议检查 CSV 中 porosity < 0.1% 的文件，确认是否为无效扫描。")

if __name__ == "__main__":
    main()
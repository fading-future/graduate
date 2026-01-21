import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import random

# ================= 核心配置区域 =================
CONFIG = {
    "src_root": r"D:\多尺度岩心数据集\Raw_Data",          # 输入数据根目录
    "dst_root": r"D:\多尺度岩心数据集\Aligned_Training_Data_Interactive", # 输出路径
    "num_workers": 24,       # 并行核心数
    "calib_samples": 3000,   # 计算 Target Peak 时采样的样本数
    "preview_count": 5       # 每次预览时随机抽取的图片数量
}
# ===========================================

def normalize_fixed(data):
    """固定归一化: uint16 -> uint8 (0-255)"""
    return (data.astype(np.float32) / 65535.0 * 255.0).astype(np.uint8)

def get_peak(data_u8):
    """寻找骨架峰值 (忽略 < 40 的孔隙噪点)"""
    hist, _ = np.histogram(data_u8.ravel(), bins=256, range=(0, 255))
    hist[:40] = 0 
    return np.argmax(hist)

def auto_calculate_target_peak(src_root, sample_count=100):
    """第一步：自动计算全数据集的 Target Peak"""
    print(f"\n[Step 1] 正在扫描文件以计算基准峰值 (Target Peak)...")
    all_files = list(Path(src_root).rglob("*.npy"))
    if not all_files:
        raise FileNotFoundError(f"在 {src_root} 中未找到 .npy 文件")
    
    # 随机抽样
    real_sample_count = min(len(all_files), sample_count)
    sample_files = np.random.choice(all_files, real_sample_count, replace=False)
    
    peaks = []
    for f in tqdm(sample_files, desc="Calculating Peaks"):
        try:
            d = normalize_fixed(np.load(f))
            peaks.append(get_peak(d))
        except:
            pass
            
    avg_peak = int(np.mean(peaks))
    print(f"✅ 自动计算完成！全数据集平均峰值 (Target Peak) = {avg_peak}")
    return avg_peak, all_files

def visualize_preview(files_to_show, target_peak, offset):
    """可视化预览函数：Raw -> Aligned -> Segmented"""
    n = len(files_to_show)
    if n == 0:
        print("⚠️ 该文件夹下没有找到文件，无法预览。")
        return

    fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n))
    if n == 1: axes = axes.reshape(1, -1) # 处理单张图片的情况
    
    print(f"正在生成预览图... (Target={target_peak}, Offset={offset})")
    
    threshold = target_peak - offset

    for i, f_path in enumerate(files_to_show):
        # 1. 加载
        raw_u8 = normalize_fixed(np.load(f_path))
        curr_peak = get_peak(raw_u8)
        
        # 2. 对齐
        shift = target_peak - curr_peak
        aligned_u8 = np.clip(raw_u8.astype(np.int16) + shift, 0, 255).astype(np.uint8)
        
        # 3. 分割
        mask = aligned_u8 < threshold
        porosity = np.sum(mask) / mask.size
        
        # 取中间切片
        mid_idx = raw_u8.shape[0] // 2
        sl_raw = raw_u8[mid_idx]
        sl_align = aligned_u8[mid_idx]
        sl_mask = mask[mid_idx]

        # 绘图 - Col 1: Raw
        axes[i, 0].imshow(sl_raw, cmap='gray', vmin=0, vmax=255)
        axes[i, 0].set_title(f"Raw (Peak={curr_peak})\n{f_path.name}", fontsize=10)
        axes[i, 0].axis('off')

        # 绘图 - Col 2: Aligned
        axes[i, 1].imshow(sl_align, cmap='gray', vmin=0, vmax=255)
        sign = "+" if shift >= 0 else ""
        axes[i, 1].set_title(f"Aligned (Shift={sign}{shift})\nTarget={target_peak}", fontsize=10)
        axes[i, 1].axis('off')

        # 绘图 - Col 3: Segmented (注意反转mask显示: 白骨架黑孔隙)
        axes[i, 2].imshow(~sl_mask, cmap='gray', vmin=0, vmax=1)
        axes[i, 2].set_title(f"Segmented (Th={threshold})\nPhi={porosity:.2%}", fontsize=10, color='red')
        axes[i, 2].axis('off')

    plt.tight_layout()
    plt.show() # 阻塞式弹窗，关闭窗口后程序继续

def interactive_preview_loop(src_root, target_peak):
    """第二步 & 第三步：交互式确认 Offset"""
    print("\n" + "="*60)
    print("进入 [交互式预览模式]。请确认 Offset 参数。")
    print("输入 'run' 或 'go' 开始批量处理。")
    print("="*60)
    
    current_offset = 15 # 默认初始值
    
    while True:
        print(f"\n当前全局参数: Target Peak = {target_peak} | 当前 Offset = {current_offset}")
        
        # 1. 获取输入
        user_input = input(">>> 请输入要预览的数据子文件夹名 (直接回车随机抽样, 输入 'run' 开始处理): ").strip()
        
        if user_input.lower() in ['run', 'go', 'exit', 'quit']:
            confirm = input(f"确认使用 Offset = {current_offset} 处理所有数据吗? (y/n): ")
            if confirm.lower() == 'y':
                return current_offset
            else:
                continue

        # 2. 确定预览的文件夹
        if not user_input:
            # 如果回车，从根目录递归找
            search_path = Path(src_root)
            pattern = "**/*.npy"
        else:
            # 尝试拼接路径
            search_path = Path(src_root) / user_input
            if not search_path.exists():
                print(f"❌ 路径不存在: {search_path}")
                continue
            pattern = "*.npy"
        
        # 3. 获取 Offset
        offset_input = input(f"请输入测试 Offset (当前={current_offset}, 直接回车保持): ").strip()
        if offset_input:
            try:
                current_offset = int(offset_input)
            except ValueError:
                print("❌ Offset 必须是整数")
                continue

        # 4. 抽取文件并展示
        found_files = list(search_path.glob(pattern)) if "**" not in str(pattern) else list(search_path.rglob("*.npy"))
        
        if not found_files:
            print("❌ 该路径下未找到 .npy 文件")
            continue
            
        sample_size = min(len(found_files), CONFIG['preview_count'])
        files_to_show = random.sample(found_files, sample_size)
        
        # 弹窗展示
        visualize_preview(files_to_show, target_peak, current_offset)

def process_single_file(args):
    """单个文件处理逻辑 (已更新 CSV 记录字段)"""
    file_path, target_peak, offset, dst_root, src_root = args
    
    try:
        path_obj = Path(file_path)
        rel_path = path_obj.relative_to(src_root)
        save_path = Path(dst_root) / rel_path
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 1. 加载
        data_u8 = normalize_fixed(np.load(file_path))
        
        # 2. 对齐
        curr_peak = get_peak(data_u8)
        shift = target_peak - curr_peak
        data_aligned = np.clip(data_u8.astype(np.int16) + shift, 0, 255).astype(np.uint8)
        
        # 3. 分割
        threshold = target_peak - offset
        mask = data_aligned < threshold 
        
        # 4. 计算孔隙度
        porosity = np.sum(mask) / mask.size
        
        # 5. 保存
        np.save(save_path, data_aligned)
        
        # 返回信息 (增加了 target_peak 和 offset)
        return {
            "file": path_obj.name, 
            "rel_path": str(rel_path), 
            "porosity": porosity, 
            "shift": shift, 
            "target_peak": target_peak, # 新增记录
            "offset": offset,           # 新增记录
            "status": "ok"
        }

    except Exception as e:
        return {"file": str(file_path), "status": "error", "msg": str(e)}

def main():
    # --- 1. 自动计算基准 (Step 1) ---
    target_peak, all_files = auto_calculate_target_peak(CONFIG["src_root"], CONFIG["calib_samples"])
    
    # --- 2. 交互式确认参数 (Step 2 & 3) ---
    # 这里会卡住，直到用户满意并输入 run
    final_offset = interactive_preview_loop(CONFIG["src_root"], target_peak)
    
    final_thresh = target_peak - final_offset
    
    print("\n" + "="*60)
    print(f"🚀 开始批量处理 (Step 4)")
    print(f"   样本总数: {len(all_files)}")
    print(f"   Target Peak: {target_peak}")
    print(f"   Offset: {final_offset}")
    print(f"   Threshold: {final_thresh}")
    print(f"   Output: {CONFIG['dst_root']}")
    print("="*60 + "\n")
    
    # --- 3. 并行执行 (Step 4) ---
    tasks = [(str(f), target_peak, final_offset, CONFIG["dst_root"], CONFIG["src_root"]) for f in all_files]
    
    results = []
    with ProcessPoolExecutor(max_workers=CONFIG["num_workers"]) as executor:
        for res in tqdm(executor.map(process_single_file, tasks), total=len(tasks), desc="Processing"):
            results.append(res)
            
    # --- 4. 保存报告 (Step 4) ---
    df = pd.DataFrame(results)
    csv_path = os.path.join(CONFIG["dst_root"], "processing_report.csv")
    df.to_csv(csv_path, index=False)
    
    print(f"\n✅ 处理全部完成！")
    print(f"   报告路径: {csv_path}")
    print(f"   CSV已包含 target_peak 和 offset 列，方便后续追溯。")

if __name__ == "__main__":
    main()
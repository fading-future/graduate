import numpy as np
import os
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

"""
计算REV 数据的孔隙率的脚本
Results:
1. 输出对齐后的NPY 数据（用于后续的模型训练）
2. 输出处理报告 CSV 文件，包含每个样本的孔隙率等信息
"""

CONFIG = {
    "src_root": r"D:\多尺度岩心数据集\Raw_Data",          # 输入路径
    "dst_root": r"D:\多尺度岩心数据集\Aligned_Training_Data", # 输出路径
    "num_workers": 24,       # 并行核心数
    "calib_samples": 3000,   # 用多少个样本来自动计算对齐目标
    "fixed_offset": 15,     # 阈值偏移量 (Threshold = Peak - 15)
}

def normalize_fixed(data):
    """固定归一化"""
    return (data.astype(np.float32) / 65535.0 * 255.0).astype(np.uint8)

def get_peak(data_u8):
    """寻找骨架峰值 (忽略 < 40 的孔隙噪点)"""
    hist, _ = np.histogram(data_u8.ravel(), bins=256, range=(0, 255))
    hist[:40] = 0 
    return np.argmax(hist)

def auto_calculate_target_peak(src_root, sample_count=100):
    """自动计算目标峰值"""
    all_files = list(Path(src_root).rglob("*.npy"))
    if not all_files:
        raise FileNotFoundError("没有找到 .npy 文件")
    
    # 随机抽样
    sample_files = np.random.choice(all_files, min(len(all_files), sample_count), replace=False)
    
    print(f"正在基于 {len(sample_files)} 个样本自动计算 Target Peak...")
    peaks = []
    for f in tqdm(sample_files, desc="Auto-Calibration"):
        try:
            d = normalize_fixed(np.load(f))
            peaks.append(get_peak(d))
        except:
            pass
            
    avg_peak = int(np.mean(peaks))
    print(f"★ 自动计算完成！Target Peak = {avg_peak}")
    return avg_peak, all_files

def process_single_file(args):
    """单个文件处理逻辑"""
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
        mask = data_aligned < threshold # True=孔隙, False=骨架
        
        # 4. 计算孔隙度
        porosity = np.sum(mask) / mask.size
        
        # 5. 保存 (保存对齐后的灰度图，用于训练)
        np.save(save_path, data_aligned)
        
        # return {"file": path_obj.name, "porosity": porosity, "shift": shift, "status": "ok"}
        return {
            "file": path_obj.name, 
            "rel_path": str(rel_path), 
            "porosity": porosity, 
            "shift": shift, 
            "status": "ok"
        }

    except Exception as e:
        return {"file": str(file_path), "status": "error", "msg": str(e)}

def main():
    # 1. 自动计算 Target Peak
    target_peak, all_files = auto_calculate_target_peak(CONFIG["src_root"], CONFIG["calib_samples"])
    final_thresh = target_peak - CONFIG["fixed_offset"]
    
    print(f"准备处理 {len(all_files)} 个文件 | Target Peak: {target_peak} | Threshold: {final_thresh}")
    
    # 2. 并行处理
    tasks = [(str(f), target_peak, CONFIG["fixed_offset"], CONFIG["dst_root"], CONFIG["src_root"]) for f in all_files]
    
    results = []
    with ProcessPoolExecutor(max_workers=CONFIG["num_workers"]) as executor:
        for res in tqdm(executor.map(process_single_file, tasks), total=len(tasks), desc="Processing"):
            results.append(res)
            
    # 3. 保存报告
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(CONFIG["dst_root"], "processing_report.csv"), index=False)
    print(f"处理完成，报告已保存。成功率: {len(df[df['status']=='ok'])/len(df):.1%}")

if __name__ == "__main__":
    main()
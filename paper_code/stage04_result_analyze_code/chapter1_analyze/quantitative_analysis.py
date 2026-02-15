import os
import glob
import json
import numpy as np
import cv2
from pathlib import Path
import random
from tqdm import tqdm

# ================= 1. 数据集路径配置 (全量) =================
# 1.1 原始 CT 文件夹 (包含 6-6-21, 22, 24)
RAW_ROOT_DIRS = [
    r"D:\多尺度岩心数据集\6-6-21",
    r"D:\多尺度岩心数据集\6-6-22",
    r"D:\多尺度岩心数据集\6-6-24"
]

# 1.2 预处理后灰度图 (Gray_Preprocessed_Slices)
PROCESSED_ROOT_DIRS = [
    r"D:\多尺度岩心数据集\Lastest_Preprocess\Gray_Preprocessed_Slices\6-6-21",
    r"D:\多尺度岩心数据集\Lastest_Preprocess\Gray_Preprocessed_Slices\6-6-22",
    r"D:\多尺度岩心数据集\Lastest_Preprocess\Gray_Preprocessed_Slices\6-6-24"
]

# 1.3 二值化切片 (Binary_Preprocessed_Slices)
BINARY_ROOT_DIRS = [
    r"D:\多尺度岩心数据集\Lastest_Preprocess\Binary_Preprocessed_Slices\6-6-21",
    r"D:\多尺度岩心数据集\Lastest_Preprocess\Binary_Preprocessed_Slices\6-6-22",
    r"D:\多尺度岩心数据集\Lastest_Preprocess\Binary_Preprocessed_Slices\6-6-24"
]

# 1.4 REV NPY 数据根目录 (自动扫描下面的 w192_s32 等子文件夹)
REV_ROOT_DIR = r"D:\多尺度岩心数据集\window_slide_result"

# ================= 2. 核心算法工具 (增强鲁棒性) =================

def get_peak_robust(img_path, mode='raw'):
    """
    鲁棒的峰值检测，针对不同阶段的数据应用不同的屏蔽策略
    """
    try:
        img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        if img is None: return None
        
        flat = img.ravel()
        counts = np.bincount(flat, minlength=65536)
        
        if mode == 'raw':
            # Raw 数据痛点：空气底噪(0-3000) 和 饱和高光(65535)
            # 策略：只看 5000 - 65500 之间的峰值
            counts[:5000] = 0
            counts[65500:] = 0 
        elif mode == 'processed':
            # Processed 数据痛点：Soft-Tanh 后空气被拉伸到 2000-8000，高光被压缩到 60000+
            # 策略：只看 12000 - 60000 之间的峰值 (锁定 35000 锚定点)
            counts[:12000] = 0 
            counts[60000:] = 0
            
        peak = int(np.argmax(counts))
        # 如果过滤后找不到峰值（全是0），返回None
        if peak == 0: return None
        return peak
    except:
        return None

def compute_rev_curve(npy_path):
    """计算单个 NPY 的 REV 曲线"""
    data = np.load(npy_path)
    # 确保是 0/1 (1为孔隙)
    # if data.max() > 1: data = (data > 0).astype(np.uint8)

    data = (data == 0).astype(np.uint8)
    
    z, y, x = data.shape
    cz, cy, cx = z//2, y//2, x//2
    
    sizes = []
    porosities = []
    
    # 动态步长：小尺度细看，大尺度粗看
    max_dim = min(z, y, x)
    steps = list(range(16, 64, 4)) + list(range(64, max_dim + 1, 16))
    
    for s in steps:
        h = s // 2
        # 边界保护
        z1, z2 = max(0, cz-h), min(z, cz+h)
        y1, y2 = max(0, cy-h), min(y, cy+h)
        x1, x2 = max(0, cx-h), min(x, cx+h)
        
        vol = data[z1:z2, y1:y2, x1:x2]
        if vol.size == 0: continue
        
        phi = np.sum(vol) / vol.size
        sizes.append(s)
        porosities.append(phi)
        
    return sizes, porosities

def compute_s2_curve(img_path):
    """计算单张切片的 S2 曲线"""
    try:
        img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        if img is None: return None
        
        # 降采样加速 FFT (1900x1900 太慢，缩放到 512x512 不影响趋势)
        img_small = cv2.resize(img, (512, 512), interpolation=cv2.INTER_NEAREST)
        phi_img = (img_small > 127).astype(np.float32)
        
        # FFT 计算自相关
        f = np.fft.fft2(phi_img)
        fshift = np.fft.ifft2(f * np.conj(f)).real
        fshift = np.fft.fftshift(fshift) / phi_img.size
        
        cy, cx = fshift.shape[0]//2, fshift.shape[1]//2
        max_r = 80 # 像素距离 (对应原图约 300 像素)
        
        y, x = np.indices(fshift.shape)
        r_map = np.sqrt((x - cx)**2 + (y - cy)**2)
        
        r_val = np.arange(max_r)
        s2_val = np.zeros(max_r)
        
        # 向量化径向平均
        r_map_int = r_map.astype(int)
        mask_roi = r_map_int < max_r
        
        sums = np.bincount(r_map_int[mask_roi], weights=fshift[mask_roi], minlength=max_r)
        counts = np.bincount(r_map_int[mask_roi], minlength=max_r)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            s2_val = sums / counts
            
        return s2_val.tolist()
    except:
        return None

# ================= 3. 主执行流程 =================

def main():
    final_stats = {}
    
    # --- Part 1: 批次一致性分析 (遍历所有指定的文件夹) ---
    print("\n📊 [1/3] 分析批次一致性 (Batch Consistency)...")
    
    raw_peaks = []
    processed_peaks = []
    
    # 1.1 Raw
    for folder in RAW_ROOT_DIRS:
        files = glob.glob(os.path.join(folder, "*.tif"))
        if not files: continue
        # 每个文件夹随机抽 200 张
        samples = random.sample(files, min(len(files), 200))
        for f in samples:
            p = get_peak_robust(f, mode='raw')
            if p: raw_peaks.append(p)
            
    # 1.2 Processed
    for folder in PROCESSED_ROOT_DIRS:
        files = glob.glob(os.path.join(folder, "*.tif"))
        if not files: continue
        samples = random.sample(files, min(len(files), 200))
        for f in samples:
            p = get_peak_robust(f, mode='processed')
            if p: processed_peaks.append(p)
            
    final_stats["consistency"] = {"raw": raw_peaks, "processed": processed_peaks}
    print(f"   -> 采样统计: Raw={len(raw_peaks)}张, Processed={len(processed_peaks)}张")
    print(f"   -> Processed 均值: {np.mean(processed_peaks):.0f} (预期 30000-45000)")

    # --- Part 2: REV 稳定性分析 (修改：递归搜索 + 优化图例Key) ---
    print("\n🧊 [2/3] 分析 REV 稳定性 (递归搜索)...")
    rev_data = {}
    
    # 使用您改写的 rglob 逻辑
    root = Path(REV_ROOT_DIR)
    # 搜索所有符合特征的子文件夹
    subfolders = [p for p in root.rglob("Final_Result_Sorted_*") if p.is_dir()]
    
    for folder_path in subfolders:
        # 【关键修改】构造具备区分度的图例名称
        # folder_path.parent.name -> "6-6-24" (岩心名)
        # folder_path.name -> "Final_Result_Sorted_w192_s64" (配置名)
        core_name = folder_path.parent.name
        config_part = folder_path.name.split("Sorted_")[-1] # 提取 "w192_s64"
        
        # 组合成唯一的 Key: "6-6-24 (w192_s64)"
        legend_key = f"{core_name} ({config_part})"
        
        files = glob.glob(os.path.join(str(folder_path), "*.npy"))
        if not files: continue
        
        print(f"   -> 分析配置: {legend_key}")
        
        # 采样计算 (保持不变)
        samples = random.sample(files, min(len(files), 200))
        config_curves = []
        for f in tqdm(samples, leave=False):
            try:
                sizes, phis = compute_rev_curve(f)
                config_curves.append({"sizes": sizes, "phis": phis})
            except: pass
            
        if config_curves:
            rev_data[legend_key] = config_curves # 使用新 Key 存入
            
    final_stats["rev"] = rev_data

    # --- Part 3: S2(r) 空间相关性 (修改：按文件夹分别计算，不合并) ---
    print("\n📈 [3/3] 分析 S2(r) 空间相关性 (分岩心计算)...")
    s2_data = {} # 改为字典
    
    for folder in BINARY_ROOT_DIRS:
        folder_name = os.path.basename(folder) # 获取 "6-6-21" 等
        print(f"   -> 扫描: {folder_name}")
        
        files = glob.glob(os.path.join(folder, "*.tif"))
        if not files: continue
        
        # 计算当前岩心的平均曲线
        current_curves = []
        samples = random.sample(files, min(len(files), 200))
        for f in tqdm(samples, leave=False):
            curve = compute_s2_curve(f)
            if curve and not np.isnan(curve).any():
                current_curves.append(curve)
        
        if current_curves:
            # 对齐长度并取平均
            min_len = min([len(c) for c in current_curves])
            aligned = [c[:min_len] for c in current_curves]
            avg_curve = np.mean(aligned, axis=0).tolist()
            
            # 【关键】按岩心名存入字典
            s2_data[folder_name] = {
                "x": list(range(len(avg_curve))),
                "y": avg_curve
            }
            
    final_stats["s2"] = s2_data # 保存整个字典
    
    # --- 保存 ---
    with open("quantitative_metrics_v4.json", "w") as f:
        json.dump(final_stats, f)
    print("\n✅ 全量分析完成！数据已保存至 quantitative_metrics_v4.json")

if __name__ == "__main__":
    main()
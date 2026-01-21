import os
import glob
import re
import numpy as np
import tifffile as tiff
import cv2
from scipy import ndimage
from tqdm import tqdm

"""
CT 图像处理流水线 v1.0
包含：
1. 非对称 SDR 清洗
2. 安全区提取与滑动窗口保存为 NPY 格式
没有降噪和归一化步骤
"""


CONFIG = {
    'src_dir': r"D:\多尺度岩心数据集",            # 你的源数据路径
    'dst_dir': r"D:\多尺度岩心数据集\Final_Dataset_NPY_23" # 结果保存路径
}

class RockCorePipeline:
    def __init__(self, src_root, dst_root, 
                 crop_size=128, 
                 stride_z=64, 
                 stride_xy=64,
                 sever_size=25, 
                 restore_size=20):
        """
        :param src_root: 原始 .tif 文件夹路径
        :param dst_root: .npy 保存路径
        :param crop_size: REV 大小 (128)
        :param stride_z: Z轴滑动步长 (64 表示 Z方向 50% 重叠)
        :param stride_xy: XY平面滑动步长 (64 表示 XY方向 50% 重叠)
        :param sever_size: SDR算法-切断力度 (25)
        :param restore_size: SDR算法-恢复力度 (20)
        """
        self.src_root = src_root
        self.dst_root = dst_root
        self.crop_size = crop_size
        self.stride_z = stride_z
        self.stride_xy = stride_xy
        
        # SDR 参数
        self.sever_size = sever_size
        self.restore_size = restore_size
        
        # 预计算形态学核 (加速运算)
        self.kernel_sever = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (sever_size, sever_size))
        self.kernel_restore = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (restore_size, restore_size))
        
        # 安全区腐蚀核 (REV大小 + 20像素安全边距)
        margin = 10
        self.kernel_erode_size = crop_size + margin
        self.kernel_erode = np.ones((self.kernel_erode_size, self.kernel_erode_size), np.uint8)

        if not os.path.exists(dst_root):
            os.makedirs(dst_root)

    def _extract_sort_key(self, filepath):
        """文件名排序 key"""
        match = re.search(r'modif(\d+)', filepath)
        return int(match.group(1)) if match else 0

    def _calculate_mild_otsu(self, image_slice, scale_factor=1.05):
        """计算温和阈值"""
        small = cv2.resize(image_slice, (512, 512))
        img_min, img_max = small.min(), small.max()
        if img_max == img_min: return img_min
        
        norm = ((small - img_min) / (img_max - img_min) * 255).astype(np.uint8)
        otsu_val, _ = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        real_thresh = img_min + (otsu_val / 255.0) * (img_max - img_min)
        return real_thresh * scale_factor

    def _asymmetric_sdr_clean(self, binary_mask):
        """
        【不对称 SDR 清洗】
        针对单张切片进行去噪
        """
        binary_mask = binary_mask.astype(np.uint8)
        
        # 1. Sever (狠切)
        eroded = cv2.erode(binary_mask, self.kernel_sever, iterations=1)
        
        # 2. Delete (删孤岛)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(eroded, connectivity=8)
        if num_labels < 2: 
            return np.zeros_like(binary_mask) # 全黑
            
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        core_mask = np.zeros_like(binary_mask)
        core_mask[labels == largest_label] = 1
        
        # 3. Restore (少补)
        restored_mask = cv2.dilate(core_mask, self.kernel_restore, iterations=1)
        
        # 4. Limit (约束)
        final_mask = cv2.bitwise_and(restored_mask, binary_mask)
        return final_mask

    def _generate_sliding_window_points(self, safe_zone):
        """在安全区内生成滑动窗口坐标"""
        H, W = safe_zone.shape
        ys, xs = np.where(safe_zone == 1)
        
        if len(ys) == 0: return []
        
        # 确定扫描边界
        y_min, y_max = np.min(ys), np.max(ys)
        x_min, x_max = np.min(xs), np.max(xs)
        
        points = []
        # 按步长滑动
        for y in range(y_min, y_max + 1, self.stride_xy):
            for x in range(x_min, x_max + 1, self.stride_xy):
                # 检查点是否在安全区
                if safe_zone[y, x] == 1:
                    points.append((y, x))
        return points

    def process_folder(self, folder_name):
        folder_path = os.path.join(self.src_root, folder_name)
        if not os.path.isdir(folder_path): return

        # 1. 准备文件
        files = glob.glob(os.path.join(folder_path, "*.tif"))
        files.sort(key=self._extract_sort_key)
        total_files = len(files)
        
        if total_files < self.crop_size:
            print(f"[Warn] {folder_name} 图片太少 ({total_files}), 跳过.")
            return
            
        print(f"\n🚀 开始处理: {folder_name} | 总切片: {total_files}")
        
        folder_save_count = 0
        
        # 2. Z轴滑动循环 (处理 3D 块)
        # range(start, end, stride_z)
        for z in tqdm(range(0, total_files - self.crop_size, self.stride_z), desc=f"Processing {folder_name}"):
            
            # 2.1 加载 3D 块 (D, H, W)
            chunk_files = files[z : z + self.crop_size]
            try:
                stack = tiff.imread(chunk_files)
            except Exception as e:
                print(f"读取错误 z={z}: {e}")
                continue

            # 2.2 计算阈值 (使用中间层)
            mid_slice = stack[len(stack)//2]
            thresh = self._calculate_mild_otsu(mid_slice, scale_factor=1.05)

            # 2.3 【关键】Z轴投影有效性计算
            # 我们需要确保选出的(x,y)在整个128层深度内都是干净的
            
            # 为了节省内存和时间，我们逐层处理并累积
            # 初始化一个全1的 valid_map (H, W)
            valid_map_projected = np.ones_like(mid_slice, dtype=np.uint8)
            
            for i in range(stack.shape[0]):
                # A. 二值化
                raw_binary = (stack[i] > thresh).astype(np.uint8)
                
                # B. SDR 清洗
                clean_slice = self._asymmetric_sdr_clean(raw_binary)
                
                # C. 孔洞填充 (保护内部孔隙)
                filled_slice = ndimage.binary_fill_holes(clean_slice).astype(np.uint8)
                
                # D. Z轴求交集 (逻辑与: 只要有一层是0，结果就是0)
                # 这就是对 Z 轴的考虑！
                valid_map_projected = cv2.bitwise_and(valid_map_projected, filled_slice)
                
                # 优化: 如果 map 已经全黑了，就不用继续算后面层了
                if np.sum(valid_map_projected) == 0:
                    break
            
            if np.sum(valid_map_projected) == 0:
                continue

            # 2.4 生成安全区 (Safe Zone)
            # 对投影后的 map 进行大腐蚀
            safe_zone = cv2.erode(valid_map_projected, self.kernel_erode, iterations=1)
            
            # 2.5 滑动窗口采样
            centers = self._generate_sliding_window_points(safe_zone)
            
            if len(centers) == 0:
                continue
                
            # 2.6 提取并保存
            for (yc, xc) in centers:
                # 坐标转换: 中心 -> 左上角
                # 注意：这里减去的是腐蚀核半径，对应生成 safe_zone 时的逻辑
                y = yc - self.kernel_erode_size // 2
                x = xc - self.kernel_erode_size // 2
                
                # 截取 REV
                rev_block = stack[:, y:y+self.crop_size, x:x+self.crop_size]
                
                # 保存
                save_name = f"{folder_name}_z{z}_y{y}_x{x}.npy"
                np.save(os.path.join(self.dst_root, save_name), rev_block)
                folder_save_count += 1
        
        print(f"✅ {folder_name} 完成! 生成了 {folder_save_count} 个 REV.")

# ==========================================
# 运行入口
# ==========================================
if __name__ == "__main__":
    # --- 配置路径 ---
    SRC_DIR = CONFIG['src_dir']
    DST_DIR = CONFIG['dst_dir']
    
    # --- 初始化流水线 ---
    pipeline = RockCorePipeline(
        src_root=SRC_DIR, 
        dst_root=DST_DIR,
        crop_size=256,    # REV 尺寸
        stride_z=128,      # Z轴步长 (50%重叠)
        stride_xy=128,     # XY步长 (50%重叠) -> 如果想要无重叠改为128
        sever_size=25,    # SDR 切断力度
        restore_size=20   # SDR 恢复力度
    )
    
    # --- 指定要处理的文件夹 ---
    # folders = [
    #     "6-6-9", "6-6-12", "6-6-15", "6-6-18", 
    #     "6-6-20 全部", "6-6-21", "6-6-22", "6-6-23", "6-6-24"
    # ]

    folders = [
        "6-6-23"
    ]
    
    # --- 开始批量处理 ---
    print("开始执行全量数据提取...")
    for folder in folders:
        pipeline.process_folder(folder)
        
    print("\n🎉 全部处理完毕！数据已保存为 .npy 格式。")

import os
import glob
import re
import numpy as np
import tifffile as tiff
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy import ndimage

"""
论文插图绘制代码v1.0
"""

CONFIG = {
    'base_path': r"D:\多尺度岩心数据集",  # 数据集根目录
    'target_folder': "6-6-22",  # 目标文件夹名称
    "crop_size": 128,      # REV 尺寸
    "stride_ratio": 0.5,   # 步长比例 (相对于 crop_size)
    'start_idx': None,    # 起始位置，不指定则取中间位置
    'save_path': r".\img_data\Step1_Fig_Data_Construction.png"  # 保存路径
}

class ThesisRockPipeline:
    def __init__(self, source_dir, crop_size=128, stride_ratio=0.5):
        self.source_dir = source_dir
        self.crop_size = crop_size
        self.stride_ratio = stride_ratio
        self.stride = int(crop_size * stride_ratio)
        
    def _extract_sort_key(self, filepath):
        match = re.search(r'modif(\d+)', filepath)
        return int(match.group(1)) if match else 0

    def _calculate_otsu_threshold(self, image_slice):
        """自动计算16-bit图像的Otsu阈值"""
        small = cv2.resize(image_slice, (512, 512))
        img_min, img_max = small.min(), small.max()
        if img_max == img_min: return img_min
        norm_img = ((small - img_min) / (img_max - img_min) * 255).astype(np.uint8)
        otsu_val, _ = cv2.threshold(norm_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return img_min + (otsu_val / 255.0) * (img_max - img_min)

    def load_chunk_and_process(self, folder_name, start_idx=None):
        """加载数据并计算核心掩码"""
        folder_path = os.path.join(self.source_dir, folder_name)
        files = glob.glob(os.path.join(folder_path, "*.tif"))
        files.sort(key=self._extract_sort_key)
        
        # 如果不指定位置，默认取中间的一段用于展示
        if start_idx is None:
            start_idx = len(files) // 2 - 100
            
        print(f"正在读取 {folder_name} (Index: {start_idx} - {start_idx + self.crop_size})...")
        target_files = files[start_idx : start_idx + self.crop_size]
        stack = tiff.imread(target_files)
        
        # 1. 计算阈值
        mid_slice = stack[len(stack)//2]
        thresh = self._calculate_otsu_threshold(mid_slice)
        
        # 2. 3D 填充与投影 (Z-Projection)
        print("执行孔隙保护与投影算法...")
        filled_stack_list = []
        for i in range(stack.shape[0]):
            binary = stack[i] > thresh
            filled = ndimage.binary_fill_holes(binary) # 保护孔隙
            filled_stack_list.append(filled)
            
        valid_map = np.min(np.array(filled_stack_list), axis=0) # Z轴投影
        
        # 3. 腐蚀计算安全区 (Safe Zone)
        # 腐蚀核 = REV大小 + 安全边距
        margin = 5
        kernel_size = self.crop_size + margin
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        
        # 这里的 safe_centers 里的白点，代表以此为中心截取 REV 是绝对安全的
        safe_centers = cv2.erode(valid_map.astype(np.uint8), kernel, iterations=1)
        
        return mid_slice, valid_map, safe_centers, thresh

    def generate_grid_points(self, safe_centers_map):
        """生成网格采样点"""
        H, W = safe_centers_map.shape
        # 网格起点偏移
        offset = self.crop_size // 2
        
        y_grid = np.arange(offset, H - offset, self.stride)
        x_grid = np.arange(offset, W - offset, self.stride)
        
        valid_points = []
        for yc in y_grid:
            for xc in x_grid:
                if safe_centers_map[yc, xc] == 1:
                    valid_points.append((yc, xc))
        return valid_points

    def visualize_for_thesis(self, folder_name, save_path="thesis_figure.png", start_idx=None):
        """
        生成论文专用的三联图：
        1. 原始图像 (Raw)
        2. 算法处理逻辑 (ROI Detection)
        3. 最终采样方案 (Sampling Strategy)
        """
        # 1. 获取数据
        raw_img, valid_map, safe_centers, _ = self.load_chunk_and_process(folder_name, start_idx)
        grid_points = self.generate_grid_points(safe_centers)
        
        # 2. 准备绘图
        fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)
        
        # --- 子图 A: 原始切片 ---
        axes[0].imshow(raw_img, cmap='gray')
        axes[0].set_title("(a) Original CT Slice (Raw)", fontsize=16, pad=10)
        axes[0].axis('off')
        
        # --- 子图 B: 有效区域投影 ---
        # 展示这一摞切片投影后的样子，说明我们剔除了边缘残缺
        axes[1].imshow(raw_img, cmap='gray', alpha=0.6)
        # 叠加半透明的 valid_map (黄色表示有效岩石区)
        masked_valid = np.ma.masked_where(valid_map == 0, valid_map)
        axes[1].imshow(masked_valid, cmap='autumn', alpha=0.4) 
        axes[1].set_title("(b) Valid Region Projection", fontsize=16, pad=10)
        axes[1].axis('off')
        
        # --- 子图 C: 最终采样方案 (The Clever Part) ---
        axes[2].imshow(raw_img, cmap='gray')
        
        # 1. 画出“安全采样中心区” (Safe Centers Area) - 蓝色半透明
        # 这解释了为什么我们只在中间采，而不碰边缘
        masked_safe = np.ma.masked_where(safe_centers == 0, safe_centers)
        axes[2].imshow(masked_safe, cmap='winter', alpha=0.3)
        
        # 2. 画网格点 (不画乱七八糟的框，只画点)
        if len(grid_points) > 0:
            ys = [p[0] for p in grid_points]
            xs = [p[1] for p in grid_points]
            # s=5 表示点的大小，红色
            axes[2].scatter(xs, ys, c='red', s=4, marker='+', linewidth=0.8, alpha=0.8, label='Sampling Centers')
        
        # 3. 画 3 个示例框 (Representative REVs) - 证明尺寸
        # 只选中间、左边、右边各一个，清晰展示大小
        # if len(grid_points) > 0:
        #     sample_indices = [len(grid_points)//2, len(grid_points)//4, 3*len(grid_points)//4]
        #     for idx in sample_indices:
        #         yc, xc = grid_points[idx]
        #         # 转换回左上角
        #         y_tl = yc - self.crop_size // 2
        #         x_tl = xc - self.crop_size // 2
                
        #         rect = patches.Rectangle((x_tl, y_tl), self.crop_size, self.crop_size, 
        #                                  linewidth=2, edgecolor='yellow', facecolor='none')
        #         axes[2].add_patch(rect)

        # 3. 画示例框 (Representative REVs)
        # 修改版：均匀选取更多框进行展示
        num_boxes_to_show = 5000  # 【在这里修改】你想画多少个框，就填多少
        
        if len(grid_points) > 0:
            # 使用 linspace 均匀采样索引，确保框分布在图像的各个位置
            if len(grid_points) > num_boxes_to_show:
                sample_indices = np.linspace(0, len(grid_points)-1, num_boxes_to_show, dtype=int)
            else:
                sample_indices = range(len(grid_points))

            for idx in sample_indices:
                yc, xc = grid_points[idx]
                
                # 转换回左上角
                y_tl = yc - self.crop_size // 2
                x_tl = xc - self.crop_size // 2
                
                # 画框
                rect = patches.Rectangle((x_tl, y_tl), self.crop_size, self.crop_size, 
                                         linewidth=1.5, # 线宽适中
                                         edgecolor='yellow', 
                                         facecolor='none',
                                         alpha=0.9)     # 透明度高一点，更亮
                axes[2].add_patch(rect)
                
            # 图例保持不变
            dummy_rect = patches.Rectangle((0,0), 1, 1, linewidth=2, edgecolor='yellow', facecolor='none', label='REV Sample')
            axes[2].legend(handles=[dummy_rect], loc='upper right', fontsize=10)

            # 添加图例，解释那个黄框是什么
            # 创建一个虚拟的 handle 用于图例
            # dummy_rect = patches.Rectangle((0,0), 1, 1, linewidth=2, edgecolor='yellow', facecolor='none', label='REV Size (128$^3$)')
            # axes[2].legend(handles=[dummy_rect], loc='upper right', fontsize=12)

        axes[2].set_title("(c) Sampling Strategy (Grid)", fontsize=16, pad=10)
        axes[2].axis('off')
        
        # 保存高清大图
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n[成功] 论文插图已保存至: {save_path}")
        plt.show()

if __name__ == "__main__":
    # 1. 设置你的路径
    base_path = CONFIG['base_path']
    target_folder = CONFIG['target_folder']
    
    # 2. 初始化流水线
    # stride_ratio=0.5 表示 50% 重叠，这是标准配置
    pipeline = ThesisRockPipeline(base_path, crop_size=CONFIG['crop_size'], 
                                  stride_ratio=CONFIG['stride_ratio'])
    
    # 3. 生成论文图
    pipeline.visualize_for_thesis(target_folder, 
                                  save_path=CONFIG['save_path'], 
                                  start_idx=CONFIG['start_idx'])
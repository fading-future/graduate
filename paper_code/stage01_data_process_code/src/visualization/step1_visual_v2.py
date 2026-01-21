import os
import glob
import re
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
from scipy import ndimage
from skimage.restoration import denoise_nl_means, estimate_sigma

# 导入自定义模块
from utils.get_root_path import get_project_root, get_img_data_path

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

"""
科研论文插图生成脚本，与数据处理脚本配套使用。
展示了从原始CT图像到去噪、ROI裁剪、有效区域提取、滑动窗口采样的全过程。
最终生成一张包含多子图的综合图表，用于论文展示。
"""

ROOT_DIR = get_project_root()
IMG_DATA_DIR = get_img_data_path()

CONFIG = {
    # 基础配置
    'src_root': r"D:\多尺度岩心数据集",
    'target_folder': "6-6-18",
    'crop_size': 256,
    'stride': 64,
    'target_z_index': None,  # 如果为 None 则取中间切片
    'save_path': str(IMG_DATA_DIR / "Step1_Thesis_Figure_MethodNoise.png"),

    # 去噪算法参数
    'denoise_h': 4,  # NLM 去噪强度参数

    # 非对称SDR 参数（需与处理脚本保持一致）
    'sever_size': 25,
    'restore_size': 20,
    'margin_size': 10,  # 安全区腐蚀边缘余量
}


class ThesisIntegratedPipeline:
    def __init__(self, src_root, crop_size=256, stride=128, denoise_h=10):
        self.src_root = src_root
        self.crop_size = crop_size
        self.stride = stride
        self.denoise_h = denoise_h

    def _read_img(self, path):
        try:
            raw = np.fromfile(path, dtype=np.uint8)
            return cv2.imdecode(raw, cv2.IMREAD_UNCHANGED)
        except: return None

    def _extract_sort_key(self, filepath):
        match = re.search(r'modif(\d+)', filepath)
        return int(match.group(1)) if match else 0

    def _detect_global_roi(self, img):
        scale = 0.25
        small = cv2.resize(img, None, fx=scale, fy=scale)
        small_8bit = ((small - small.min()) / (small.max() - small.min()) * 255).astype(np.uint8)
        _, thresh = cv2.threshold(small_8bit, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours: return (img.shape[1]//2, img.shape[0]//2), img.shape[0]//2
        c = max(contours, key=cv2.contourArea)
        (x, y), r = cv2.minEnclosingCircle(c)
        return (int(x/scale), int(y/scale)), int(r/scale)

    def generate_figure(self, folder_name, target_z_index=None, save_path=CONFIG['save_path']):
        folder_path = os.path.join(self.src_root, folder_name)
        files = sorted(glob.glob(os.path.join(folder_path, "*.tif")), key=self._extract_sort_key)
        total_files = len(files)
        
        # =========================================================
        # 1. 定位目标切片
        # =========================================================
        # 如果未指定，默认取正中间
        if target_z_index is None:
            target_z_index = total_files // 2
        
        # 限制范围，防止越界
        target_z_index = np.clip(target_z_index, 0, total_files - 1)
        
        # 计算 3D 块的起止位置 (前后各取一半 crop_size，保证 target 在中间附近)
        # 这样 Z-Projection 才会利用到这一层附近的信息
        half_crop = self.crop_size // 2
        start_idx = target_z_index - half_crop
        end_idx = target_z_index + half_crop
        
        # 边界修正：如果太靠前或太靠后
        if start_idx < 0:
            start_idx = 0
            end_idx = min(total_files, self.crop_size)
        elif end_idx > total_files:
            end_idx = total_files
            start_idx = max(0, total_files - self.crop_size)
            
        chunk_files = files[start_idx : end_idx]
        print(f"1. 加载数据块: 索引 {start_idx} - {end_idx} (共 {len(chunk_files)} 张)...")
        print(f"   目标展示切片: Index {target_z_index} (在块中的相对位置: {target_z_index - start_idx})")
        
        stack_raw = np.array([self._read_img(f) for f in chunk_files if self._read_img(f) is not None])
        
        # 取出我们要展示的那一张 (即 target_z_index 对应的那张)
        relative_idx = target_z_index - start_idx
        # 防止意外的索引越界
        relative_idx = np.clip(relative_idx, 0, len(stack_raw)-1)
        slice_raw_full = stack_raw[relative_idx]

        # =========================================================
        # 2. 预处理与 ROI 裁剪
        # =========================================================
        print("2. 预处理与增强...")
        (cx, cy), r_global = self._detect_global_roi(slice_raw_full)
        r_pad = int(r_global * 1.05)
        y1, y2 = max(0, cy - r_pad), min(slice_raw_full.shape[0], cy + r_pad)
        x1, x2 = max(0, cx - r_pad), min(slice_raw_full.shape[1], cx + r_pad)
        
        stack_roi = stack_raw[:, y1:y2, x1:x2]
        slice_roi = slice_raw_full[y1:y2, x1:x2]
        mask_roi = np.zeros_like(slice_roi, dtype=np.uint8)
        cv2.circle(mask_roi, (slice_roi.shape[1]//2, slice_roi.shape[0]//2), int(r_global), 1, -1)

        # 图像增强
        pixels = slice_roi[mask_roi == 1]
        p1, p99 = np.percentile(pixels, [1, 99])
        norm_img = (slice_roi.astype(np.float32) - p1) / (p99 - p1 + 1e-6)
        norm_img = np.clip(norm_img, 0, 1)
        
        # 降噪
        sigma = np.mean(estimate_sigma(norm_img))
        denoised_img = denoise_nl_means(norm_img, h=self.denoise_h * sigma, fast_mode=True,
                                        patch_size=5, patch_distance=6)
        
        # 应用 Mask
        norm_img[mask_roi == 0] = 0
        denoised_img[mask_roi == 0] = 0

        # --- 计算方法噪声 (Method Noise) ---
        # 绝对值差
        residual_map = np.abs(norm_img - denoised_img)
        # 适当放大以便观察 (通常噪声很小，不放大是全黑的)
        # 这里的 5 倍放大在论文 Caption 里要注明 "contrast boosted for visualization"
        residual_viz = np.clip(residual_map * 5, 0, 1) 
        residual_viz[mask_roi == 0] = 0

        # 定义局部放大区域
        h, w = norm_img.shape
        box_size = 256 
        yA, xA = h // 2 - 60, w // 2 - 90 
        yB, xB = h // 2 + 30, w // 2 + 60
        
        zoom_noisy_A = norm_img[yA:yA+box_size, xA:xA+box_size]
        zoom_clean_A = denoised_img[yA:yA+box_size, xA:xA+box_size]
        zoom_noisy_B = norm_img[yB:yB+box_size, xB:xB+box_size]
        zoom_clean_B = denoised_img[yB:yB+box_size, xB:xB+box_size]

        # =========================================================
        # 3. 3D 投影与采样计算
        # =========================================================
        print("3. 计算有效性投影 (应用 SDR 算法)...")
        small = cv2.resize(slice_roi, (256, 256))
        t_val, _ = cv2.threshold(((small-small.min())/(small.max()-small.min())*255).astype(np.uint8), 
                                 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        real_thresh = slice_roi.min() + (t_val/255.0)*(slice_roi.max()-slice_roi.min())
        
        # --- 定义 SDR 结构元素 (必须与处理脚本一致) ---
        # 假设处理脚本中 sever_size=25, restore_size=20
        sever_size = CONFIG['sever_size']  # 25
        restore_size = CONFIG['restore_size']  # 20
        kernel_sever = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (sever_size, sever_size))
        kernel_restore = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (restore_size, restore_size))
        
        filled_stack = []
        for s in stack_roi:
            # A. 二值化
            binary = (s > real_thresh).astype(np.uint8) * 255
            
            # B. SDR 核心逻辑 (复刻处理脚本)
            # 1. 腐蚀断开连接
            eroded = cv2.erode(binary, kernel_sever)
            # 2. 连通域分析，只保留最大连通域 (岩心本体)
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(eroded)
            if num_labels > 1:
                largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA]) # 0是背景，所以从1开始找
                temp_mask = (labels == largest_label).astype(np.uint8) * 255
                # 3. 膨胀还原形状
                temp_mask = cv2.dilate(temp_mask, kernel_restore)
                # 4. 掩码约束
                clean_mask = cv2.bitwise_and(temp_mask, binary)
            else:
                clean_mask = np.zeros_like(binary)
            
            # C. 填充孔洞 (这一步还是需要的，为了保证内部完整)
            filled = ndimage.binary_fill_holes(clean_mask).astype(np.uint8)
            filled_stack.append(filled)
            
        valid_map = np.min(np.array(filled_stack), axis=0)
        
        # 安全区计算 (保持不变)
        margin = CONFIG['margin_size']
        kernel_safe = np.ones((self.crop_size + margin, self.crop_size + margin), np.uint8)
        safe_zone = cv2.erode(valid_map.astype(np.uint8), kernel_safe, iterations=1)
        
        offset = self.crop_size // 2
        H_roi, W_roi = valid_map.shape
        y_grid = np.arange(offset, H_roi - offset, self.stride)
        x_grid = np.arange(offset, W_roi - offset, self.stride)
        grid_points = [(yc, xc) for yc in y_grid for xc in x_grid if safe_zone[yc, xc] == 1]
        # =========================================================
        # 4. 绘图逻辑 (GridSpec 严格控制大小)
        # =========================================================
        fig = plt.figure(figsize=(20, 10)) # 调整长宽比
        # wspace=0.05 让列之间紧凑，width_ratios 保持一致确保大小相同
        gs = gridspec.GridSpec(2, 4, width_ratios=[1, 1, 1, 1], height_ratios=[1, 1], figure=fig)
        
        # --- Row 1 ---
        # (a) 全局
        ax_a = fig.add_subplot(gs[0, 0])
        viz_raw = ((slice_raw_full - slice_raw_full.min())/(slice_raw_full.max()-slice_raw_full.min())*255).astype(np.uint8)
        viz_raw = cv2.cvtColor(viz_raw, cv2.COLOR_GRAY2RGB)
        cv2.circle(viz_raw, (cx, cy), int(r_global), (255, 0, 0), 10)
        ax_a.imshow(viz_raw)
        ax_a.set_title("(a) Global ROI Detection", fontsize=13, fontweight='bold')
        ax_a.axis('off')

        # (b) 增强与框选
        ax_b = fig.add_subplot(gs[0, 1])
        ax_b.imshow(norm_img, cmap='gray')
        rect_A = patches.Rectangle((xA, yA), box_size, box_size, linewidth=2, edgecolor='yellow', facecolor='none')
        rect_B = patches.Rectangle((xB, yB), box_size, box_size, linewidth=2, edgecolor='cyan', facecolor='none')
        ax_b.add_patch(rect_A); ax_b.add_patch(rect_B)
        ax_b.text(xA, yA-5, 'R1', color='yellow', fontsize=12, fontweight='bold')
        ax_b.text(xB, yB-5, 'R2', color='cyan', fontsize=12, fontweight='bold')
        ax_b.set_title("(b) ROI Enhancement", fontsize=13, fontweight='bold')
        ax_b.axis('off')

        # (c) 局部对比 (2x2)
        # 为了保证排版整齐，这里虽然是 2x2，但外边框和其他图一样大
        gs_zoom = gs[0, 2].subgridspec(2, 2, wspace=0.05, hspace=0.05)
        
        ax_z1_n = fig.add_subplot(gs_zoom[0, 0])
        ax_z1_n.imshow(zoom_noisy_A, cmap='gray')
        for spine in ax_z1_n.spines.values(): spine.set_edgecolor('yellow'); spine.set_linewidth(2)
        ax_z1_n.set_xticks([]); ax_z1_n.set_yticks([])
        ax_z1_n.text(2, 15, 'R1: Raw', color='white', fontsize=10, fontweight='bold')

        ax_z1_c = fig.add_subplot(gs_zoom[0, 1])
        ax_z1_c.imshow(zoom_clean_A, cmap='gray')
        for spine in ax_z1_c.spines.values(): spine.set_edgecolor('yellow'); spine.set_linewidth(2)
        ax_z1_c.set_xticks([]); ax_z1_c.set_yticks([])
        ax_z1_c.text(2, 15, 'R1: NLM', color='yellow', fontsize=10, fontweight='bold')

        ax_z2_n = fig.add_subplot(gs_zoom[1, 0])
        ax_z2_n.imshow(zoom_noisy_B, cmap='gray')
        for spine in ax_z2_n.spines.values(): spine.set_edgecolor('cyan'); spine.set_linewidth(2)
        ax_z2_n.set_xticks([]); ax_z2_n.set_yticks([])
        ax_z2_n.text(2, 15, 'R2: Raw', color='white', fontsize=10, fontweight='bold')

        ax_z2_c = fig.add_subplot(gs_zoom[1, 1])
        ax_z2_c.imshow(zoom_clean_B, cmap='gray')
        for spine in ax_z2_c.spines.values(): spine.set_edgecolor('cyan'); spine.set_linewidth(2)
        ax_z2_c.set_xticks([]); ax_z2_c.set_yticks([])
        ax_z2_c.text(2, 15, 'R2: NLM', color='cyan', fontsize=10, fontweight='bold')
        
        # 这种设置title的方法能保证位置居中
        fig.text(0.615, 0.93, "(c) Detail Comparison", fontsize=13, fontweight='bold', ha='center')

        # (d) 方法噪声图 (Method Noise)
        ax_res = fig.add_subplot(gs[0, 3])
        im_res = ax_res.imshow(residual_viz, cmap='inferno')
        ax_res.set_title("(d) Method Noise", fontsize=13, fontweight='bold')
        ax_res.axis('off')
        # 色条放里面一点，或者放下面，保持对齐
        cbar = plt.colorbar(im_res, ax=ax_res, fraction=0.046, pad=0.04)

        # --- Row 2 ---
        ax_e = fig.add_subplot(gs[1, 0])
        ax_e.imshow(slice_roi, cmap='gray')
        ax_e.set_title("(e) Z-Projection Base", fontsize=13, fontweight='bold')
        ax_e.axis('off')

        ax_f = fig.add_subplot(gs[1, 1])
        ax_f.imshow(slice_roi, cmap='gray', alpha=0.6)
        masked_valid = np.ma.masked_where(valid_map == 0, valid_map)
        ax_f.imshow(masked_valid, cmap='autumn', alpha=0.4)
        ax_f.set_title("(f) Valid Rock Region", fontsize=13, fontweight='bold')
        ax_f.axis('off')

        ax_g = fig.add_subplot(gs[1, 2])
        ax_g.imshow(slice_roi, cmap='gray')
        masked_safe = np.ma.masked_where(safe_zone == 0, safe_zone)
        ax_g.imshow(masked_safe, cmap='winter', alpha=0.3)
        if len(grid_points) > 0:
            ys, xs = zip(*grid_points)
            ax_g.scatter(xs, ys, c='red', s=15, marker='+', linewidth=1.5)
        ax_g.set_title("(g) Safe Zone & Grid", fontsize=13, fontweight='bold')
        ax_g.axis('off')
        
        ax_h = fig.add_subplot(gs[1, 3])
        ax_h.imshow(slice_roi, cmap='gray')
        if len(grid_points) > 0:
            indices = np.linspace(0, len(grid_points)-1, min(40, len(grid_points)), dtype=int)
            for idx in indices:
                yc, xc = grid_points[idx]
                y_tl, x_tl = yc - self.crop_size//2, xc - self.crop_size//2
                rect = patches.Rectangle((x_tl, y_tl), self.crop_size, self.crop_size, 
                                         linewidth=1, edgecolor='yellow', facecolor='none', alpha=0.8)
                ax_h.add_patch(rect)
        ax_h.set_title("(h) Final REV Extraction", fontsize=13, fontweight='bold')
        ax_h.axis('off')

        plt.subplots_adjust(left=0.02, right=0.98, top=0.92, bottom=0.05, wspace=0.1, hspace=0.15)
        print(f"5. 保存图表至: {save_path}")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

if __name__ == "__main__":
    SRC = CONFIG['src_root']
    FOLDER = CONFIG['target_folder']
    
    pipeline = ThesisIntegratedPipeline(SRC, crop_size=CONFIG['crop_size'], 
                                        stride=CONFIG['stride'], 
                                        denoise_h=CONFIG['denoise_h'])
    
    # 【在这里修改 target_z_index】
    # 指定你想查看的切片索引。如果不确定，写 None 会自动取中间。
    # 比如你想看第 3000 张图：target_z_index=3000
    pipeline.generate_figure(FOLDER, target_z_index=CONFIG['target_z_index'], 
                             save_path=CONFIG['save_path'])
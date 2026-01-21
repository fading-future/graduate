import os
import glob
import re
import numpy as np
import cv2
import tifffile as tiff
from scipy import ndimage
from tqdm import tqdm
from skimage.restoration import denoise_nl_means, estimate_sigma

"""
CT 图像处理流水线 v2.0
原始的CT 图像数据预处理流水线，包含：
1. 全局 ROI 探测（统一坐标系）
2. 图像归一化 (P1-P99)
3. 图像降噪（可选 NLM 或各向异性扩散）
4. 非对称 SDR 清洗
5. 安全区提取与滑动窗口保存为 NPY 格式
"""

CONFIG = {
    # 基础配置
    'src_root': r"D:\多尺度岩心数据集",
    'dst_root': r"D:\多尺度岩心数据集\Cleaned_NPY_Dataset_24",
    'crop_size': 256,      
    'stride_z': 32,
    'stride_xy': 32,
    'denoise_type': 'nlm', # 'nlm', 'anisotropic' 或 'none'
    'denoise_h': 4,
    'process_folder': "6-6-24",

    # 非对称SDR 参数，不建议修改了，已经调试到比较合适的值
    'sever_size': 25,
    'restore_size': 20,
    'margin_size': 10, # 安全区腐蚀边缘余量，想要保留更多的npy块可以调小一些（更激进，但没必要）

    # 全局ROI探测参数，不建议修改了，已经调试到比较合适的值
    'global_roi_sample_count': 20,
} 

class EnhancedRockCorePipeline:
    def __init__(self, src_root, dst_root, 
                 crop_size=256, 
                 stride_z=128, 
                 stride_xy=128,
                 denoise_type='nlm',  # 'nlm', 'anisotropic' 或 'none'
                 denoise_h=10):
        
        self.src_root = src_root
        self.dst_root = dst_root
        self.crop_size = crop_size
        self.stride_z = stride_z
        self.stride_xy = stride_xy
        self.denoise_type = denoise_type
        self.denoise_h = denoise_h
        
        # 非对称SDR 算法参数
        self.sever_size = CONFIG['sever_size']
        self.restore_size = CONFIG['restore_size']
        self.kernel_sever = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.sever_size, self.sever_size))
        self.kernel_restore = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.restore_size, self.restore_size))
        
        # 安全区腐蚀核 (REV大小 + 边缘余量)
        self.kernel_erode = np.ones((crop_size + CONFIG['margin_size'], crop_size + CONFIG['margin_size']), np.uint8)

        if not os.path.exists(dst_root):
            os.makedirs(dst_root)

    def _read_img_raw(self, path):
        """兼容中文路径的读取方式"""
        try:
            raw_data = np.fromfile(path, dtype=np.uint8)
            img = cv2.imdecode(raw_data, cv2.IMREAD_UNCHANGED)
            return img
        except:
            return None

    def _detect_global_roi(self, files, sample_count=20):
        """
        第一步：全局探测。
        从文件夹中抽取切片，确定一个能包容该文件夹所有岩心的统一 ROI。
        """
        print("第一步：全局探测。正在探测该文件夹的全局 ROI (统一坐标系)...")
        indices = np.linspace(0, len(files)-1, sample_count, dtype=int)
        all_centers = []
        all_radii = []

        for idx in indices:
            img = self._read_img_raw(files[idx])
            if img is None: continue
            
            # 快速寻找圆
            small = cv2.resize(img, (380, 380))
            mi, ma = small.min(), small.max()
            if ma - mi < 10: continue # 跳过全黑/全白片
            
            small_8bit = ((small - mi) / (ma - mi + 1e-6) * 255).astype(np.uint8)
            _, thresh = cv2.threshold(small_8bit, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                c = max(contours, key=cv2.contourArea)
                (x, y), r = cv2.minEnclosingCircle(c)
                # 映射回原图坐标 (1900/380 = 5)
                all_centers.append((x * 5, y * 5))
                all_radii.append(r * 5)
        
        if not all_centers:
            return None # 探测失败

        # 统计中位数，防止破碎岩心的离群值干扰
        avg_center = np.median(all_centers, axis=0).astype(int)
        max_radius = int(np.percentile(all_radii, 90)) # 取90分位数，保证覆盖
        
        print(f"探测完成: 中心{avg_center}, 统一半径{max_radius}")
        return avg_center, max_radius

    def _apply_denoise(self, img_float):
        """第二步：应用降噪"""
        # print("第二步：应用降噪。正在处理图像降噪...")
        # --- 新增防错检查 ---
        # 如果图片最大值非常小（接近全黑），或者标准差为0（没有变化），直接返回原图
        if np.max(img_float) < 1e-5 or np.std(img_float) < 1e-6:
            return img_float 

        if self.denoise_type == 'nlm':
            try:
                sigma_est = estimate_sigma(img_float, channel_axis=None)
                # 检查 sigma_est 是否为空或全为0
                if sigma_est is None or np.size(sigma_est) == 0:
                    return img_float
                
                sigma_avg = np.mean(sigma_est)
                return denoise_nl_means(img_float, h=self.denoise_h * sigma_avg, 
                                        fast_mode=True, patch_size=5, patch_distance=6, channel_axis=None)
            except Exception as e:
                # 捕获可能的其他数学错误
                return img_float
                
        elif self.denoise_type == 'anisotropic':
            return self._anisotropic_diffusion_fast(img_float)
        
        return img_float

    def _anisotropic_diffusion_fast(self, img, n_iter=5, kappa=0.05, gamma=0.15):
        img = img.copy()
        for _ in range(n_iter):
            dN = np.roll(img, -1, axis=0) - img
            dS = np.roll(img, 1, axis=0) - img
            dE = np.roll(img, -1, axis=1) - img
            dW = np.roll(img, 1, axis=1) - img
            cN = np.exp(-(dN/kappa)**2); cS = np.exp(-(dS/kappa)**2)
            cE = np.exp(-(dE/kappa)**2); cW = np.exp(-(dW/kappa)**2)
            img += gamma * (cN*dN + cS*dS + cE*dE + cW*dW)
        return img

    def process_folder(self, folder_name):
        folder_path = os.path.join(self.src_root, folder_name)
        files = sorted(glob.glob(os.path.join(folder_path, "*.tif")), 
                       key=lambda x: int(re.search(r'modif(\d+)', x).group(1)) if re.search(r'modif(\d+)', x) else 0)
        
        if len(files) < self.crop_size: return

        # 1. 探测全局 ROI
        roi_info = self._detect_global_roi(files, sample_count=CONFIG['global_roi_sample_count'])
        if not roi_info: 
            print(f"文件夹 {folder_name} 无法找到有效岩心区域")
            return
        (cx, cy), r = roi_info
        
        # 确定统一的裁剪边界 (矩形)
        r_pad = int(r * 1.05)
        y1, y2 = max(0, cy - r_pad), min(1900, cy + r_pad)
        x1, x2 = max(0, cx - r_pad), min(1900, cx + r_pad)
        
        # 创建全局 Mask
        mask_local = np.zeros((y2-y1, x2-x1), dtype=np.uint8)
        cv2.circle(mask_local, (cx-x1, cy-y1), r, 1, -1)

        print(f"滑动窗口处理数据")
        
        # 2. Z轴滑动循环
        for z in tqdm(range(0, len(files) - self.crop_size, self.stride_z), desc=folder_name):
            chunk_paths = files[z : z + self.crop_size]
            
            # 加载并初步裁剪
            raw_stack = []
            for p in chunk_paths:
                full_img = self._read_img_raw(p)
                raw_stack.append(full_img[y1:y2, x1:x2])
            stack = np.array(raw_stack) # (D, H_roi, W_roi)

            # 3. 预处理流水线：归一化 + 降噪
            clean_stack = []
            binary_stack = []
            
            for i in range(stack.shape[0]):
                slice_data = stack[i].copy()
                # 3.1 P1-P99 归一化
                core_pixels = slice_data[mask_local == 1]
                if len(core_pixels) < 100: # 可能是极其破碎的片
                    p1, p99 = 0, 65535
                else:
                    p1, p99 = np.percentile(core_pixels, [1, 99])
                
                norm_f = (slice_data.astype(np.float32) - p1) / (p99 - p1 + 1e-6)
                norm_f = np.clip(norm_f, 0, 1)
                
                # 3.2 降噪
                denoised_f = self._apply_denoise(norm_f)
                
                # 3.3 背景遮罩
                denoised_f[mask_local == 0] = 0
                
                # 3.4 二值化 (用于后续 SDR 有效性判定)
                # 使用固定的 0.2 或 0.3 作为阈值（因为已经归一化到 0-1）
                _, b_mask = cv2.threshold((denoised_f*255).astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                
                clean_stack.append((denoised_f * 65535).astype(np.uint16))
                binary_stack.append(b_mask // 255)

            clean_stack = np.array(clean_stack)
            
            # 4. 有效性判定 (Z轴投影)
            # 只要有一层没岩心，该区域就不能要
            valid_map = np.ones_like(mask_local)
            for b_slice in binary_stack:
                # SDR 清洗
                eroded = cv2.erode(b_slice.astype(np.uint8), self.kernel_sever)
                num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(eroded)
                if num_labels > 1:
                    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
                    temp_mask = (labels == largest).astype(np.uint8)
                    temp_mask = cv2.dilate(temp_mask, self.kernel_restore)
                    temp_mask = cv2.bitwise_and(temp_mask, b_slice.astype(np.uint8))
                else:
                    temp_mask = np.zeros_like(b_slice)
                
                # 填充孔洞并累积到投影图
                filled = ndimage.binary_fill_holes(temp_mask).astype(np.uint8)
                valid_map = cv2.bitwise_and(valid_map, filled)

            # 5. 安全区提取与保存
            safe_zone = cv2.erode(valid_map, self.kernel_erode)
            ys, xs = np.where(safe_zone == 1)
            
            if len(ys) == 0: continue
            
            # 滑动采样坐标
            y_pts = range(np.min(ys), np.max(ys), self.stride_xy)
            x_pts = range(np.min(xs), np.max(xs), self.stride_xy)
            
            for py in y_pts:
                for px in x_pts:
                    if safe_zone[py, px] == 1:
                        # 提取 3D 块 (D, H, W)
                        # 注意：py, px 是安全区中心点坐标，需要还原回裁剪区域左上角
                        rev = clean_stack[:, py-self.crop_size//2 : py+self.crop_size//2, 
                                            px-self.crop_size//2 : px+self.crop_size//2]
                        
                        if rev.shape == (self.crop_size, self.crop_size, self.crop_size):
                            save_path = os.path.join(self.dst_root, f"{folder_name}_z{z}_y{py}_x{px}.npy")
                            np.save(save_path, rev)

if __name__ == "__main__":
    pipe = EnhancedRockCorePipeline(
        src_root=CONFIG['src_root'],
        dst_root=CONFIG['dst_root'],
        crop_size=CONFIG['crop_size'],      
        stride_z=CONFIG['stride_z'],
        stride_xy=CONFIG['stride_xy'],
        denoise_type=CONFIG['denoise_type'], 
        denoise_h=CONFIG['denoise_h']
    )
    
    pipe.process_folder(CONFIG['process_folder'])
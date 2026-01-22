import os
import glob
import re
import numpy as np
import cv2
import tifffile as tiff
from scipy import ndimage
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

"""
CT 图像处理流水线 v3.0 (并行加速 + OpenCV版)
"""

CONFIG = {
    # 基础配置
    'src_root': r"D:\多尺度岩心数据集",
    'dst_root': r"D:\多尺度岩心数据集\Cleaned_NPY_Dataset_20",
    'crop_size': 256,      
    'stride_z': 32,
    'stride_xy': 32,
    'denoise_type': 'nlm', # 'nlm', 'anisotropic' 或 'none'
    'denoise_h': 4,        # OpenCV NLM 强度，推荐 3-10
    'process_folder': "6-6-20 全部",

    'sever_size': 25,
    'restore_size': 20,
    'margin_size': 10,
    'global_roi_sample_count': 30,
} 

# ---------------------------------------------------------
#  并行工作函数 (必须定义在类外部，供多进程调用)
# ---------------------------------------------------------
def process_slice_task(args):
    """
    单个切片的处理核心逻辑
    Args:
        slice_data: 原始的单个切片 numpy array
        mask_local: 全局圆柱 Mask
        denoise_type: 降噪类型
        denoise_h: 降噪参数
    Returns:
        (cleaned_uint16, binary_uint8)
    """
    slice_data, mask_local, denoise_type, denoise_h = args
    
    # 1. 归一化 (P1-P99) -> 映射到 0-255 (uint8) 以加速处理
    # 提取岩心区域像素计算统计值
    core_pixels = slice_data[mask_local == 1]
    
    if len(core_pixels) < 100:
        # 异常切片（如全黑）
        return np.zeros_like(slice_data, dtype=np.uint16), np.zeros_like(slice_data, dtype=np.uint8)

    p1, p99 = np.percentile(core_pixels, [1, 99])
    
    # 线性拉伸并转 uint8
    norm_f = (slice_data.astype(np.float32) - p1) / (p99 - p1 + 1e-6)
    norm_f = np.clip(norm_f, 0, 1)
    img_u8 = (norm_f * 255).astype(np.uint8)

    # 2. 降噪 (在 uint8 空间进行，OpenCV 极快)
    if denoise_type == 'nlm':
        # templateWindowSize=7, searchWindowSize=21 是经典参数
        denoised_u8 = cv2.fastNlMeansDenoising(img_u8, None, h=denoise_h, templateWindowSize=7, searchWindowSize=21)
    
    elif denoise_type == 'anisotropic':
        # 简易版各向异性扩散模拟 (双边滤波是更快的近似替代)
        # d=9, sigmaColor=75, sigmaSpace=75
        denoised_u8 = cv2.bilateralFilter(img_u8, 9, 75, 75)
    else:
        denoised_u8 = img_u8

    # 3. 背景遮罩 (Masking)
    denoised_u8[mask_local == 0] = 0

    # 4. 二值化 (Otsu)
    # 因为已经是 0-255，直接做阈值分割
    _, b_mask = cv2.threshold(denoised_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 5. 格式转换回 uint16 (如需保存为 16bit 数据)
    # 将 0-255 映射回 0-65535
    cleaned_u16 = (denoised_u8.astype(np.float32) / 255.0 * 65535).astype(np.uint16)
    
    # 返回 mask 归一化为 0/1
    return cleaned_u16, (b_mask // 255).astype(np.uint8)


class EnhancedRockCorePipeline:
    def __init__(self, src_root, dst_root, 
                 crop_size=256, 
                 stride_z=128, 
                 stride_xy=128,
                 denoise_type='nlm',  
                 denoise_h=5):
        
        self.src_root = src_root
        self.dst_root = dst_root
        self.crop_size = crop_size
        self.stride_z = stride_z
        self.stride_xy = stride_xy
        self.denoise_type = denoise_type
        self.denoise_h = denoise_h
        
        # ------------------------------------------------
        # 并行参数配置
        # ------------------------------------------------
        # 保留4个核心给系统，避免电脑卡死
        self.max_workers = max(1, multiprocessing.cpu_count() - 4)
        print(f"初始化并行流水线: 使用 CPU 核心数 {self.max_workers}")

        self.sever_size = CONFIG['sever_size']
        self.restore_size = CONFIG['restore_size']
        self.kernel_sever = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.sever_size, self.sever_size))
        self.kernel_restore = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.restore_size, self.restore_size))
        
        # 安全区腐蚀核
        self.kernel_erode = np.ones((crop_size + CONFIG['margin_size'], crop_size + CONFIG['margin_size']), np.uint8)

        if not os.path.exists(dst_root):
            os.makedirs(dst_root)

    def _read_img_raw(self, path):
        """
        _read_img_raw - 读取原始图像 (使用 OpenCV 解码 tif)。可以避免中文路径问题
        Args:
        :param self: 说明
        :param path: 图像路径
        :return: 图像 numpy array 或 None
        """
        try:
            raw_data = np.fromfile(path, dtype=np.uint8)
            img = cv2.imdecode(raw_data, cv2.IMREAD_UNCHANGED)
            return img
        except:
            return None

    def _detect_global_roi(self, files, sample_count=20):
        """
        _detect_global_roi - 探测全局 ROI (岩心圆柱区域)
        通过随机采样若干切片，使用轮廓检测寻找岩心圆心和半径
        采用中值和百分位数统计，增强鲁棒性
        Args:
        :param self: 说明
        :param files: 要处理的文件列表
        :param sample_count: 采样数量
        :return: (center_x, center_y), radius 或 None
        """
        print("第一步：全局探测。正在探测该文件夹的全局 ROI (统一坐标系)...")
        indices = np.linspace(0, len(files)-1, sample_count, dtype=int)
        all_centers = []
        all_radii = []

        for idx in indices:
            img = self._read_img_raw(files[idx])
            if img is None: continue
            
            # 缩放加速
            scale = 0.2
            h, w = img.shape
            small = cv2.resize(img, (int(w*scale), int(h*scale)))
            
            mi, ma = small.min(), small.max()
            if ma - mi < 10: continue 
            
            small_8bit = ((small - mi) / (ma - mi + 1e-6) * 255).astype(np.uint8)
            _, thresh = cv2.threshold(small_8bit, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                c = max(contours, key=cv2.contourArea)
                (x, y), r = cv2.minEnclosingCircle(c)
                # 映射回原图坐标
                all_centers.append((x / scale, y / scale))
                all_radii.append(r / scale)
        
        if not all_centers:
            return None

        avg_center = np.median(all_centers, axis=0).astype(int)
        max_radius = int(np.percentile(all_radii, 90))
        
        print(f"探测完成: 中心{avg_center}, 统一半径{max_radius}")
        return avg_center, max_radius

    def process_folder(self, folder_name):
        # 1、根据文件名排序指定文件夹中的文件
        folder_path = os.path.join(self.src_root, folder_name)
        files = sorted(glob.glob(os.path.join(folder_path, "*.tif")), 
                       key=lambda x: int(re.search(r'modif(\d+)', x).group(1)) if re.search(r'modif(\d+)', x) else 0)
        
        if len(files) < self.crop_size: 
            print("文件数量不足，跳过。")
            return

        # 2、 探测全局 ROI（寻找岩心圆柱区域）。得到圆心和半径
        roi_info = self._detect_global_roi(files, sample_count=CONFIG['global_roi_sample_count'])
        if not roi_info: 
            print(f"文件夹 {folder_name} 无法找到有效岩心区域")
            return
        (cx, cy), r = roi_info
        
        # 3、确定统一的裁剪边界，裁剪出局部的ROI 区域，是一个正方形
        r_pad = int(r * 1.05) # 适当放大一点，避免边缘切掉
        y1, y2 = max(0, cy - r_pad), cy + r_pad
        x1, x2 = max(0, cx - r_pad), cx + r_pad
        
        # 4、确定 Mask 的大小
        roi_h, roi_w = y2 - y1, x2 - x1
        mask_local = np.zeros((roi_h, roi_w), dtype=np.uint8)

        # 5、绘制ROI 区域的圆形 Mask，圆形的值为1，背景为0
        cv2.circle(mask_local, (cx-x1, cy-y1), r, 1, -1)

        print(f"开始滑动窗口处理: {folder_name}")
        
        # 6、Z轴滑动循环
        for z in tqdm(range(0, len(files) - self.crop_size, self.stride_z), desc="Processing Chunks"):
            # 6.1 取出当前要处理的堆叠块的文件路径
            chunk_paths = files[z : z + self.crop_size]
            
            # 6.2 串行循环读取单个CT 图片，裁剪 ROI 区域，组成堆叠
            raw_stack = []
            for p in chunk_paths:
                # 1、读取单个图片
                full_img = self._read_img_raw(p)
                if full_img is None: 
                    # 容错：如果读取失败，补全黑帧
                    raw_stack.append(np.zeros((roi_h, roi_w), dtype=np.uint8))
                else:
                    # 2、裁剪 ROI 区域
                    fh, fw = full_img.shape
                    crop = full_img[y1:min(y2, fh), x1:min(x2, fw)]
                    if crop.shape != (roi_h, roi_w): # 如果 crop 尺寸不对（比如到了图像边缘），需要 padding
                         padded = np.zeros((roi_h, roi_w), dtype=np.uint8)
                         padded[:crop.shape[0], :crop.shape[1]] = crop
                         raw_stack.append(padded)
                    else:
                        raw_stack.append(crop)
            
            stack = np.array(raw_stack) # (D, H, W)

            # 6.3 并行计算 (归一化 + 降噪 + 二值化)
            clean_stack = []
            binary_stack = []
            
            # 组装任务参数
            tasks = []
            for i in range(stack.shape[0]):
                # 将 mask_local 传入每个任务
                tasks.append((stack[i], mask_local, self.denoise_type, self.denoise_h))

            # 启动多进程池
            with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                # map 保证结果顺序与输入顺序一致
                results = list(executor.map(process_slice_task, tasks))
            
            # 解包结果
            for res_clean, res_bin in results:
                clean_stack.append(res_clean)
                binary_stack.append(res_bin)

            clean_stack = np.array(clean_stack)
            # binary_stack 只是 list，后续投影不需要转成巨大 numpy array，节省内存
            
            # 2.3 有效性判定 (Z轴投影) - 计算量小，保持单线程
            valid_map = np.ones_like(mask_local)
            
            for b_slice in binary_stack:
                # 非对称 SDR 清洗
                eroded = cv2.erode(b_slice, self.kernel_sever)
                num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(eroded)
                
                temp_mask = np.zeros_like(b_slice)
                if num_labels > 1:
                    largest_idx = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
                    # 还原掩码
                    temp_mask = (labels == largest_idx).astype(np.uint8)
                    temp_mask = cv2.dilate(temp_mask, self.kernel_restore)
                    temp_mask = cv2.bitwise_and(temp_mask, b_slice)
                
                # 填充孔洞
                filled = ndimage.binary_fill_holes(temp_mask).astype(np.uint8)
                valid_map = cv2.bitwise_and(valid_map, filled)

            # 2.4 安全区提取与保存 NPY
            safe_zone = cv2.erode(valid_map, self.kernel_erode)
            ys, xs = np.where(safe_zone == 1)
            
            if len(ys) == 0: continue
            
            y_pts = range(np.min(ys), np.max(ys), self.stride_xy)
            x_pts = range(np.min(xs), np.max(xs), self.stride_xy)
            
            for py in y_pts:
                for px in x_pts:
                    if safe_zone[py, px] == 1:
                        # 提取
                        half = self.crop_size // 2
                        rev = clean_stack[:, py-half : py+half, px-half : px+half]
                        
                        if rev.shape == (self.crop_size, self.crop_size, self.crop_size):
                            save_path = os.path.join(self.dst_root, f"{folder_name}_z{z}_y{py}_x{px}.npy")
                            np.save(save_path, rev)

if __name__ == "__main__":
    # Windows 下多进程必须在 if __name__ == "__main__": 下运行
    multiprocessing.freeze_support() # 防止打包成 exe 时出错，脚本运行可忽略
    
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
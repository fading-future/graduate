#%%
import cv2
import numpy as np
import glob
import os
import matplotlib.pyplot as plt

# 配置你的坏文件夹路径
BAD_FOLDER = r"/chendou_space/data/core_ctimg_data/6-6-20 全部"

#%%
def debug_roi_detection(folder_path):
    files = sorted(glob.glob(os.path.join(folder_path, "*.tif")))
    if not files:
        print("文件夹为空")
        return

    print(f"文件夹共有 {len(files)} 张图片。正在采样检测...")
    
    # 采样 100 张
    indices = np.linspace(0, len(files)-1, 1000, dtype=int)
    
    valid_samples = []
    
    for idx in indices:
        # 读取
        raw = np.fromfile(files[idx], dtype=np.uint8)
        img = cv2.imdecode(raw, cv2.IMREAD_UNCHANGED)
        if img is None: continue

        # 缩小处理
        h, w = img.shape
        scale = 0.2
        small = cv2.resize(img, (int(w*scale), int(h*scale)))
        
        # 简单二值化找圆
        mi, ma = small.min(), small.max()
        if ma - mi < 50: continue # 跳过纯色/空气图
        
        norm = ((small - mi) / (ma - mi) * 255).astype(np.uint8)
        _, thresh = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            c = max(contours, key=cv2.contourArea)
            # 过滤太小的噪点
            if cv2.contourArea(c) < 100: continue 
            
            (x, y), r = cv2.minEnclosingCircle(c)
            valid_samples.append( (img, (x/scale, y/scale, r/scale)) )

    if not valid_samples:
        print("❌ 悲剧：采样了100张，没有一张检测到有效的圆！")
        print("原因可能是：1. 全是空气图；2. 岩心对比度太低；3. 图像全是噪点。")
        return

    print(f"✅ 成功在 {len(valid_samples)} 张图片中检测到圆。展示其中 3 张结果：")

    # 画图
    plt.figure(figsize=(15, 5))
    for i in range(min(3, len(valid_samples))):
        img, (x, y, r) = valid_samples[i]
        
        # 在原图上画圈
        debug_img = (img / 256).astype(np.uint8) # 转8bit显示
        debug_img = cv2.cvtColor(debug_img, cv2.COLOR_GRAY2BGR)
        
        cv2.circle(debug_img, (int(x), int(y)), int(r), (0, 0, 255), 10) # 红圈
        cv2.circle(debug_img, (int(x), int(y)), 20, (0, 255, 0), -1)   # 绿心
        
        plt.subplot(1, 3, i+1)
        plt.imshow(debug_img)
        plt.title(f"Sample {i+1}\nR={int(r)}")
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    debug_roi_detection(BAD_FOLDER)
# %%

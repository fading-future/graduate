import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt

# 导入你的项目模块
from src.config import CONFIG
from src.dataset_patch import PatchLatentDataset

def visualize_sample(dataset, index=None):
    """
    可视化 Dataset 返回的一个样本字典
    """
    if index is None:
        index = random.randint(0, len(dataset) - 1)
    
    print(f"🔍 正在检查数据索引: {index} ...")
    
    # 获取样本
    sample = dataset[index]
    
    # 解包数据 (转换为 Numpy 以便绘图)
    # GT: (C, D, H, W) -> (4, 24, 24, 24)
    gt = sample['GT'].numpy()
    # Condition: (C, D, H, W)
    cond = sample['Condition'].numpy()
    # Mask: (1, D, H, W)
    mask = sample['Mask'].numpy()
    # TargetMask: (1, D, H, W)
    target_mask = sample['TargetMask'].numpy()
    # Phi: (1, D, H, W) -> 注意这里是被 repeat 过的 phi
    phi = sample['Phi'].numpy()
    # Porosity
    por = sample['Porosity'].item()

    # 打印统计信息
    print(f"--- 样本统计 (Index {index}) ---")
    print(f"Latent Shape: {gt.shape} (C, D, H, W)")
    print(f"Porosity Value (Global/Local): {por:.4f}")
    print(f"GT Range: [{gt.min():.2f}, {gt.max():.2f}]")
    print(f"Condition Range: [{cond.min():.2f}, {cond.max():.2f}]")
    
    # --- 绘图逻辑 ---
    # 我们取 Z 轴的中间切片来观察 (D // 2)
    # 假设 window_size=3, patch_size=8, 那么 D=24,切片取索引 12
    D = gt.shape[1]
    slice_idx = D // 2
    
    # 创建画布
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'Debug View: Dataset Index {index} (Z-Slice: {slice_idx})\nPorosity: {por:.4f}', fontsize=16)

    # 1. GT (Channel 0)
    # 原始的 Latent
    im1 = axes[0, 0].imshow(gt[0, slice_idx, :, :], cmap='viridis', origin='lower')
    axes[0, 0].set_title(f'GT (Latent Ch0)\nShape: {gt.shape}')
    plt.colorbar(im1, ax=axes[0, 0], fraction=0.046)

    # 2. Condition (Channel 0)
    # 被 Mask 过的 Latent，你应该看到部分区域变黑（或者是0）
    im2 = axes[0, 1].imshow(cond[0, slice_idx, :, :], cmap='viridis', origin='lower')
    axes[0, 1].set_title('Condition (Input to Model)\n"Future" should be 0')
    plt.colorbar(im2, ax=axes[0, 1], fraction=0.046)

    # 3. Phi Map
    # 对应的孔隙率图
    im3 = axes[0, 2].imshow(phi[0, slice_idx, :, :], cmap='magma', origin='lower')
    axes[0, 2].set_title('Phi Map (Upsampled)')
    plt.colorbar(im3, ax=axes[0, 2], fraction=0.046)

    # 4. Mask
    # 哪些是"已知"的 (1.0)，哪些是"未知"的 (0.0)
    im4 = axes[1, 0].imshow(mask[0, slice_idx, :, :], cmap='gray', vmin=0, vmax=1, origin='lower')
    axes[1, 0].set_title('Causal Mask\n(White=Known, Black=Unknown)')
    
    # 5. Target Mask
    # Loss 计算区域（应该是中心的一块）
    im5 = axes[1, 1].imshow(target_mask[0, slice_idx, :, :], cmap='gray', vmin=0, vmax=1, origin='lower')
    axes[1, 1].set_title('Target Mask\n(Center Patch Only)')

    # 6. GT vs Condition Difference
    # 展示被抹除的部分 (GT - Condition)
    diff = gt[0, slice_idx, :, :] - cond[0, slice_idx, :, :]
    im6 = axes[1, 2].imshow(diff, cmap='bwr', origin='lower')
    axes[1, 2].set_title('Diff (GT - Condition)\nWhat the model needs to predict')
    plt.colorbar(im6, ax=axes[1, 2], fraction=0.046)

    plt.tight_layout()
    plt.show()

def main():
    # 检查配置路径是否存在，避免报错
    if not os.path.exists(CONFIG['latent_dir']):
        print(f"⚠️ 警告: Config 中的 latent_dir 不存在: {CONFIG['latent_dir']}")
        print("请修改 config.py 或确保路径正确。")
        return

    print("🚀 初始化数据集...")
    # 强制设为 False 以便观察原始分布，或者 True 观察翻转效果
    dataset = PatchLatentDataset(
        latent_dir=CONFIG['latent_dir'], 
        phi_map_dir=CONFIG['phi_map_dir'], 
        augment=False 
    )
    print(f"✅ 数据集加载完成，共 {len(dataset)} 个样本对。")

    # 随机采样 3 次进行观察
    for _ in range(3):
        visualize_sample(dataset)
        input("按 Enter 键查看下一个样本...")

if __name__ == "__main__":
    main()
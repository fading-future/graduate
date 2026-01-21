import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import math

# 导入自定义的模块
from stage02_vqvae_code.src.dataset_rev import VQVAEDataset
from stage02_vqvae_code.src.model_vqvae import VQVAE3D

"""
计算 VQ-VAE 潜在空间 (Latent) 的缩放因子 (Scale Factor)。

Note: 
1. 需要在paper_code 路径下执行python -m utils.calc_latentNPY_scale_factor 命令 
2. 原理：Scale Factor = 1.0 / std(Latent)
"""

CONFIG = {
    'device': "cuda" if torch.cuda.is_available() else "cpu",
    'embedding_dim': 64,    # 潜在特征维度
    'num_embeddings': 2048, # 码本大小
    'batch_size': 4,
    'processed_data_dir': r"./processed_data",
    'image_size': 256,
    'num_workers': 4,
    'model_path': r"./models/vqvae_epoch_200.pth",
    'num_samples': 400, # 用于计算统计量的样本数量
}

def calculate_scale_factor(checkpoint_path, num_samples=400):
    """
    计算 Latent 的缩放因子 (Scale Factor)。
    原理：Scale Factor = 1.0 / std(Latent)
    
    Args:
        checkpoint_path: VQ-VAE 模型权重文件的路径 (.pth)
        num_samples: 用于计算的样本数量。建议至少几百个，不需要全量数据。
    """
    
    # 1. 强制配置
    device = CONFIG['device']
    # 这里的 batch_size 可以比训练时大一点，因为不需要反向传播，为了求稳设为 4
    batch_size = CONFIG['batch_size']
    
    print(f"Loading model from: {checkpoint_path}")
    print(f"Device: {device}")

    # 2. 加载模型
    model = VQVAE3D(
        in_channels=1,
        embedding_dim=CONFIG['embedding_dim'],
        num_embeddings=CONFIG['num_embeddings']
    ).to(device)
    
    # 加载权重 (处理可能的 key 不匹配问题，比如有没有 'module.' 前缀)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # 3. 加载数据
    dataset = VQVAEDataset(
        data_dir=CONFIG['processed_data_dir'],
        volume_size=CONFIG['image_size'],
        augment=False # 计算统计量时不要做随机翻转/旋转，保持原始分布
    )
    
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, # 随机打乱以获得代表性样本
        num_workers=CONFIG['num_workers'],
        pin_memory=True
    )

    print(f"Start calculating statistics on {num_samples} samples...")

    # 4. 增量计算统计量 (防止 3D 数据内存溢出)
    # 变量初始化
    sum_x = 0.0
    sum_sq_x = 0.0
    count = 0
    
    pbar = tqdm(total=num_samples)
    
    with torch.no_grad():
        for batch in dataloader:
            # 获取数据
            img = batch["GT"].to(device)
            
            # 编码得到 Latent
            # model.encode 返回: quantized, loss, perplexity
            quantized, _, _ = model.encode(img)
            
            # quantized shape: [B, C, D', H', W'] (例如 [4, 64, 64, 64, 64])
            
            # 累加统计量
            # 注意：我们需要计算的是所有像素点、所有通道的全局 std
            sum_x += torch.sum(quantized).item()
            sum_sq_x += torch.sum(quantized ** 2).item()
            count += quantized.numel() # 累加元素总个数
            
            pbar.update(img.shape[0])
            if pbar.n >= num_samples:
                break
    
    pbar.close()

    # 5. 计算最终的 Mean 和 Std
    # Var(X) = E[X^2] - (E[X])^2
    mean_val = sum_x / count
    var_val = (sum_sq_x / count) - (mean_val ** 2)
    std_val = math.sqrt(var_val)

    # 6. 计算 Scale Factor
    scale_factor = 1.0 / std_val

    print("\n" + "="*40)
    print("Statistics Result:")
    print(f"Samples Processed : {count / (CONFIG['image_size']//4)**3 / CONFIG['embedding_dim']:.1f} (approx volumes)")
    print(f"Latent Mean       : {mean_val:.6f} (Should be close to 0)")
    print(f"Latent Std        : {std_val:.6f}")
    print("-" * 40)
    print(f"RECOMMENDED SCALE_FACTOR: {scale_factor:.6f}")
    print("="*40 + "\n")
    
    return scale_factor

if __name__ == "__main__":
    # --- 在这里修改你的模型路径 ---
    # 可以在 logs 文件夹或 models 文件夹找到训练好的 .pth
    
    # 检查路径是否存在
    if not os.path.exists(CONFIG['model_path']):
        print(f"Error: Model file not found at {CONFIG['model_path']}")
        print("Please edit the MODEL_PATH in the script.")
    else:
        factor = calculate_scale_factor(CONFIG['model_path'], num_samples=CONFIG['num_samples']) # 400个样本足够估算
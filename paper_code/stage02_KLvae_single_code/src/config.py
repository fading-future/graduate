# config.py
import torch

CONFIG = {
    "exp_name": "stage1_vae_f4_z4_a100",
    "data_path": "./data/train",  # 你的 npy 文件夹路径
    "save_dir": "./experiments",
    
    # 训练参数
    "batch_size": 2,          # 256^3 体素很大，A100 80G 建议试 2 或 4 (如果 OOM 就降到 1 并用 accumulate_grad_batches)
    "lr": 4.5e-6,             # 基础学习率
    "epochs": 100,
    "num_workers": 8,
    "seed": 42,
    "accumulate_grad_batches": 2, # 变相增大 Batch Size

    # 模型参数 (Factor = 2^len(ch_mult) = 2^2 = 4) -> 64^3 Latent
    "model": {
        "in_channels": 1,
        "base_channels": 64,  # 可以尝试 96 或 128 (A100 优势)
        "z_channels": 4,      # 压缩后的 Latent 通道数，保持为 4
        "ch_mult": [1, 2, 4], # [64, 128, 256] -> Downsample 2次 -> 64^3 分辨率
        "dropout": 0.0
    },

    # Loss 权重
    "loss_weights": {
        "kl_weight": 0.000001,
        "disc_weight": 0.5,
        "perceptual_weight": 1.0,
        "disc_start": 5001,   # 前 5000 步只训练重建，不训练 GAN，让模型先稳定
    }
}
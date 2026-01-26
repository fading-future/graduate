import torch
from utils.get_root_path import get_root_path

PROJECT_ROOT = get_root_path()

CONFIG = {
    "experiment_name": "exp_final_stage2_graduation",
    "note": "Stage 2: Latent Diffusion based on KL-VAE (4 channels). Task: 3D Inpainting with Porosity Condition.",

    # --- 核心训练参数 ---
    # 32^3 的 Latent 非常小，A100 上 Batch Size 可以开到起飞
    'batch_size': 64,      # 建议 64 或 128，跑得飞快
    'num_workers': 8,
    'pin_memory': True,
    'accumulation_steps': 1,
    'epochs': 300,         # LDM 收敛快，但多跑跑没坏处
    'lr': 1e-4,
    'device': "cuda" if torch.cuda.is_available() else "cpu",
    'save_model_every': 5,  # 没必要存太频繁

    # --- 模型参数 (针对 4通道 KL-VAE 调整) ---
    'image_size': 32,        # KL-VAE 压缩后的尺寸
    'latent_channels': 4,    # KL-VAE 的 Latent 通道数
    'in_channels': 9,        # 4(Noisy) + 4(Condition) + 1(Mask) = 9
    'out_channels': 4,       # 预测 4 通道的噪声
    'base_channels': 128,    # 保持 128 宽度，容量足够
    'channel_mults': (1, 2, 4), # 32 -> 16 -> 8
    'timesteps': 1000,
    
    # 【必须修改】运行 prepare_data.py 后得到的 scale_factor
    'scale_factor': 5.55,  # 举例，请填入你实际算出来的值 (1/std)
    'safe_threshold': 10.0, # 放宽一点阈值，避免误杀有效数据

    # --- 训练策略 ---
    'pred_x0_reg_weight': 0.1,    # 保留这个，对稳定性很有帮助
    'large_known_top_prob': 0.0,  # 既然是做切一半补一半，这个策略可以先关掉或设小一点

    # --- 路径 ---
    # 指向你刚才用 prepare_data.py 生成的那些 porosity_xxx.npy 的文件夹
    'processed_data_dir': "/chendou_space/data/stage2_latents_full_256", 
    'model_output_dir': './models',
    'log_output_dir': './logs',
    'inference_output_dir': './inference_outputs',
    
    # 这一行可以注释掉，Stage 2 训练不需要加载 Stage 1 的权重，只需要数据
    # 'stage1_model_path': ... 
}
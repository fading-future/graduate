import torch

CONFIG = {
    "experiment_name": "exp2_final_stage2_graduation",
    "note": "Stage 2: Latent Diffusion based on KL-VAE (4 channels). Task: 3D Inpainting with Porosity Condition.",

    # --- 核心训练参数 ---
    # 32^3 的 Latent 非常小，A100 上 Batch Size 可以开到起飞
    'batch_size': 64,                                           # 建议 64 或 128，跑得飞快
    'num_workers': 18,
    'pin_memory': True,
    'accumulation_steps': 1,
    'epochs': 300,                                              # LDM 收敛快，但多跑跑没坏处
    'lr': 5e-5,
    'device': "cuda" if torch.cuda.is_available() else "cpu",
    'save_model_every': 5,                                      # 没必要存太频繁
    'use_min_snr': True,                                        # 已打开。先关掉，mask 修好后再决定是否打开
    'loss_type': 'mse',                                         # 使用mse loss
    'min_snr_gamma': 5.0,

    # --- 模型参数 (针对 4通道 KL-VAE 调整) ---
    'image_size': 32,                                           # KL-VAE 压缩后的尺寸
    'latent_channels': 4,                                       # KL-VAE 的 Latent 通道数
    'in_channels': 9,                                           # 4(Noisy) + 4(Condition) + 1(Mask) = 9
    'out_channels': 4,                                          # 预测 4 通道的噪声
    'base_channels': 128,                                       # 保持 128 宽度，容量足够
    'channel_mults': (1, 2, 4),                                 # 32 -> 16 -> 8
    'use_attention': (False, True, True),
    'timesteps': 1000,
    
    # 【必须修改】运行 prepare_data.py 后得到的 scale_factor
    'scale_factor': 1.009,                                      # 举例，请填入你实际算出来的值 (1/std)
    'safe_threshold': 10.0,                                     # 放宽一点阈值，避免误杀有效数据

    # --- 训练策略 ---
    'pred_x0_reg_weight': 0.0,                                    # 保留这个，对稳定性很有帮助

    # --- 推理参数 ---
    'repaint_resample_start': 800,                             # RePaint 起始时间步
    'repaint_resample_end': 50,                                # RePaint 结束时间步

    # --- 路径 ---
    'processed_data_dir': "/chendou_space/data/stage2_latents_full_256", 
    'model_output_dir': './models',
    'log_output_dir': './logs',
    'inference_output_dir': './inference_outputs',

    'stage1_model_path': "/chendou_space/chendou/paper_code/stage02_KLvae_single_code_v2/experiments/exp01_cube_structure_v1/checkpoint_epoch_26.pt",
}
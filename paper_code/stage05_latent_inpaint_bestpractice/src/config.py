import torch

# ==========================================================
# ✅ 你主要修改这里：路径 + 少量关键超参
# ==========================================================
PATHS = {
    # 原始 256^3 数据
    # "raw_data_dir": r"E:\\aligned_Training_Data",
    "raw_data_dir": "/chendou_space/data/aligned_Training_Data", # Linux 路径

    # 仅用于解析 porosity（按文件名匹配 porosity_*.npy）
    # "latent_dir": r"E:\\stage2_latents_full_256",
    "latent_dir": "/chendou_space/data/stage2_latents_full_256", # Linux 路径

    # 孔隙率 CSV（以 file 列匹配原始文件名）
    # "porosity_csv": r"E:\\aligned_Training_Data\\processing_report.csv",
    "porosity_csv": "/chendou_space/data/aligned_Training_Data/processing_report.csv", # Linux 路径

    # Stage1 KLVAE
    # "vae_config_path": r"E:\\chendou\\paper_code\\stage02_KLvae_single_code_v2\\config\\train_config.yaml",
    # "vae_ckpt_path": r"E:\\chendou\\paper_code\\stage02_KLvae_single_code_v2\\experiments\\exp03_cube_structure_v1\\ckpt_epoch_36.pt",
    "vae_config_path": "/chendou_space/chendou/paper_code/stage02_KLvae_single_code_v2/config/train_config.yaml", # Linux 路径
    "vae_ckpt_path": "/chendou_space/chendou/paper_code/stage02_KLvae_single_code_v2/experiments/exp03_cube_structure_v1/ckpt_epoch_36.pt", # Linux 路径

    # 预处理输出（z_full / z_cond / mask / porosity）
    # "paired_data_dir": r"E:\\stage2_latents_inpaint_pairs",
    "paired_data_dir": "/chendou_space/data/stage2_latents_inpaint_pairs", # Linux 路径
}

# ==========================================================
# 任务设置：切面与 mask
# ==========================================================
MASK = {
    "axis": "D",           # D/H/W
    "ratio": 0.5,          # 0.5 表示切半
    "jitter_ratio": 0.0,   # 0.0 固定切点；例如 0.05 表示在±5%范围抖动
}

# ==========================================================
# 训练/推理主要超参
# ==========================================================
TRAIN = {
    "experiment_name": "ldm_inpaint_bestpractice_v1",
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "batch_size": 32,
    "num_workers": 12,
    "pin_memory": True,
    "epochs": 300,
    "lr": 5e-5,
    "save_every": 2,
    "resume": True,
}

MODEL = {
    "image_size": 32,
    "latent_channels": 4,
    "in_channels": 9,   # noisy(4) + cond(4) + mask(1)
    "out_channels": 4,  # pred noise
    "base_channels": 128,
    "channel_mults": (1, 2, 4),
    "use_attention": (False, True, True),
    "timesteps": 1000,
}

# ==========================================================
# 训练损失策略（Best Practice Defaults）
# ==========================================================
LOSS = {
    "loss_type": "l1",          # noise loss: l1 or mse
    "use_min_snr": True,
    "min_snr_gamma": 5.0,

    # known/unknown
    "known_diff_weight": 0.05,   # 给 known 少量扩散误差，稳定特征

    # x0 reconstruction (unknown)
    "x0_weight": 0.2,
    "x0_boundary_weight": 1.0,

    # 边界带
    "boundary_band_width": 4,
    "boundary_band_weight": 8.0,

    # 低频/大结构约束（avg_pool 后的 x0 L1）
    "lowfreq_weight": 0.15,
    "lowfreq_kernel": 4,
}

# ==========================================================
# Latent 归一化与安全阈值
# ==========================================================
NORMALIZE = {
    "scale_factor": 1.2375,  # 预处理脚本会统计并写入 stats.json；这里可手动覆写
    "safe_threshold": 12.0,
}

# ==========================================================
# 推理设置
# ==========================================================
INFER = {
    "ddim_steps": 200,
    "seed": 1234,
}

CONFIG = {
    **PATHS,
    **MASK,
    **TRAIN,
    **MODEL,
    **LOSS,
    **NORMALIZE,
    **INFER,
}

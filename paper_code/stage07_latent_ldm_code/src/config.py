import torch

# ==========================================================
# Stage07：基于 Patch 的潜空间扩散（自回归窗口）
# ==========================================================

CONFIG = {
    # ------------------ 实验配置 ------------------
    "experiment_name": "stage07_patch_ldm_v5",
    "note": "基于因果滑窗的 Patch 级潜空间扩散",
    "device": "cuda" if torch.cuda.is_available() else "cpu",

    # ------------------ 数据配置 ------------------
    # latent 数据目录（形状：C x D x H x W）
    "latent_dir": r"D:\多尺度岩心数据集\LDM_Data\Latent_NPY\w192_s64",
    # 孔隙率图目录（形状：gD x gH x gW）
    "phi_map_dir": r"D:\多尺度岩心数据集\LDM_Data\Phi_Maps_NPY\w192_s64",
    # 原始体素数据目录（preprocess_phi.py 会使用）
    "raw_data_dir": r"D:\多尺度岩心数据集\LDM_Data\Raw_NPY\w192_s64",
    # 全局孔隙率 CSV（可选；porosity_mode='global' 时优先使用）
    "porosity_csv": r"",

    # latent 统计参数
    "scale_factor": 1.410483,   # 通常设置为 1 / latent_std
    "safe_threshold": 8.0,

    # phi 图预处理
    "binarize_mode": "none",    # "fixed" | "otsu" | "none"
    "binarize_threshold": 0.5,   # 当 binarize_mode='fixed' 时生效
    "phi_input_target_size": 192, # 0 表示不裁剪

    # patch / window 参数
    "latent_channels": 4,
    "patch_size": 8,
    "window_size": 3,
    "downsample_factor": 8,

    # PorosityEmbedder 的标量孔隙率来源
    # 'local'：目标 patch 的局部 phi
    # 'global'：样本全局 phi（CSV 缺失时回退到 phi 均值）
    # 'mix'：局部 + 全局加权
    "porosity_mode": "mix",
    "porosity_mix_alpha": 0.9,
    "use_global_phi_channel": True,

    # 上下文与遍历策略
    "context_mode": "causal",  # causal | wavefront | full
    "order": "ijk",
    "train_random_order": True,
    "train_random_direction": True,
    "train_direction": "+++",

    # 目标 patch 采样策略
    "anchor_sampling_mode": "low_context_boost",  # uniform | low_context_boost
    "anchor_boost_power": 1.0,
    "anchor_boost_min_weight": 0.05,
    "pad_mode": "edge",  # edge | reflect | constant

    # ------------------ 模型配置 ------------------
    "in_channels": 2 * 4 + 3,  # 运行时会自动计算，此处为可读性保留
    "out_channels": 4,
    "base_channels": 128,
    "channel_mults": (1, 2, 4),
    "use_attention": (False, True, True),
    "timesteps": 1000,

    # ------------------ 训练配置 ------------------
    "batch_size": 32,
    "num_workers": 8,
    "pin_memory": True,
    "epochs": 200,
    "lr": 3e-5,
    "resume": True,
    "resume_load_optimizer": False,
    "resume_load_scheduler": True,
    "save_model_every": 1,
    "save_log_every": 1,
    "ema_decay": 0.999,

    # 损失与约束
    "loss_type": "l1",
    "use_min_snr": True,
    "min_snr_gamma": 5.0,
    "x0_weight": 0.35,
    "use_target_stats_loss": True,
    "target_stats_weight": 0.05,
    "use_phi_consistency_loss": True,
    "phi_consistency_weight": 0.08,
    # 解码式孔隙率一致性损失的轻量控制（仅在 use_phi_consistency_loss=True 时生效）
    "phi_loss_every_steps": 1,   # compute decode-based phi loss every step for stronger conditioning
    "phi_loss_max_batch": 4,     # use sub-batch to balance memory and effective supervision
    # 轻量 latent 代理孔隙率损失（不走 VAE 解码，显存开销很小）
    "use_phi_proxy_loss": True,
    "phi_proxy_weight": 0.05,
    "phi_proxy_ridge": 1e-4,
    "grad_clip_norm": 1.0,
    "boundary_band_width": 1,
    "boundary_band_weight": 4.0,

    # 孔隙率长尾重采样（强化高孔隙率样本）
    "use_porosity_weighted_sampler": True,
    "porosity_sampler_semantic": "pore",  # "pore" | "rock_rate"
    # For pore semantic, values roughly in [0, ~0.55] on current dataset
    "porosity_bin_edges": [0.0, 0.02, 0.05, 0.10, 0.18, 0.28, 0.60],
    "porosity_sampler_power": 1.8,
    "porosity_sampler_min_weight": 0.2,
    "porosity_sampler_max_weight": 8.0,

    # ------------------ 训练中评估 ------------------
    "eval_every_steps": 0,
    "eval_ddim_steps": 200,
    "eval_seed": 1234,
    "eval_index": 66,
    "eval_output_dir": "eval",
    "eval_save_png": True,
    "eval_decode_voxel": True,
    "eval_vae_config_path": r"E:\chendou\paper_code\stage02_KLvae_single_code_v2\config\train_config copy.yaml",
    "eval_vae_ckpt_path": r"E:\chendou\paper_code\stage02_KLvae_single_code_v2\experiments\exp04_cube_structure_v1\ckpt_epoch_11.pt",
    "eval_voxel_save_png": True,
    "eval_use_ema": False,

    # ------------------ 推理配置 ------------------
    "ddim_steps": 300,
    "seed": 6666,
    "infer_random_order": True,
    "infer_random_direction": True,
    "infer_direction": "+++",
    "infer_max_patch_batch": 16,
    "infer_use_ema": True,
    "ckpt_path": r"E:\chendou\paper_code\stage07_latent_ldm_code\exp_results\stage07_patch_ldm_v5\models\unet_epoch_25.pth",
    "phi_map_path": r"D:\多尺度岩心数据集\LDM_Data\Phi_Maps_NPY\w192_s64\6-6-22_Global_Consistency_z3072_y192_x576.npy",
    "output_latent_path": "generated_latent.npy",
    "output_unscaled": True,
}

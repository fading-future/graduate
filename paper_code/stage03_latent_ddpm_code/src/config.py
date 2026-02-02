import torch

# ==========================================================
# ✅ 你以后主要只改这里：主调参区（少量旋钮）
# ==========================================================
TUNE = {
    # 数据尺度（优先保证准确）
    "scale_factor": 1.243195,      # 推荐：来自统计 1/std
    "safe_threshold": 8.0,         # 推荐：6~8（10 也行，但先收紧）

    # 目标函数（结构优先）
    "loss_type": "l1",             # diffusion loss 建议 l1
    "known_consistency_weight": 0.1,
    "known_consistency_type": "l1",
    "boundary_x0_consistency_weight": 3.0,   # ⭐ 结构关键：建议 1→3→5 逐步加
    "boundary_x0_consistency_type": "l1",
    "boundary_band_width": 8,      # ⭐ 建议 6~10
    "boundary_band_weight": 10.0,  # 先不动

    # 训练 trick（先固定）
    "use_min_snr": True,
    "min_snr_gamma": 5.0,
    "pred_x0_reg_weight": 0.01,
    "known_diff_weight": 0.05,
}

# ==========================================================
# 基本训练参数（基本不动）
# ==========================================================
TRAIN = {
    "batch_size": 1,
    "num_workers": 0,
    "pin_memory": True,
    "accumulation_steps": 1,
    "epochs": 300,
    "lr": 5e-5,
    "save_model_every": 20,
    "resume": True,
}

# ==========================================================
# 模型结构参数（基本不动）
# ==========================================================
MODEL = {
    "image_size": 32,
    "latent_channels": 4,
    "in_channels": 9,      # 4(noisy)+4(cond)+1(mask)
    "out_channels": 4,     # pred noise/v
    "base_channels": 128,
    "channel_mults": (1, 2, 4),
    "use_attention": (False, True, True),
    "timesteps": 1000,
}

# ==========================================================
# Debug/Overfit（只在 sanity 时用）
# ==========================================================
DEBUG = {
    "overfit_num_samples": 16,     # 0 关闭
    "overfit_fixed_mask": True,
    "overfit_seed": 1234,
}

# ==========================================================
# 推理参数（基本不动，除非你在对比）
# ==========================================================
INFER = {
    "ddim_steps_infer": 200,       # ⭐ 建议先 200 看上限
    "repaint_resample_start": 800,
    "repaint_resample_end": 50,
}

# ==========================================================
# 路径（按机器改）
# ==========================================================
PATHS = {
    "processed_data_dir": r"E:\stage2_latents_full_256",
    "model_output_dir": "./models",
    "log_output_dir": "./logs",
    "inference_output_dir": "./inference_outputs",
    "stage1_model_path": r"E:\chendou\paper_code\stage02_KLvae_single_code_v2\experiments\exp03_cube_structure_v1\ckpt_epoch_36.pt",
}

# ==========================================================
# 最终导出：保持你现有代码不改（继续 CONFIG[...]）
# ==========================================================
CONFIG = {
    "experiment_name": "exp0_LDM_l1_v1",
    "note": "LDM stage2 inpainting sanity run",
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    **TRAIN,
    **MODEL,
    **DEBUG,
    **INFER,
    **PATHS,
    **TUNE,
}
import torch

# ==========================================================
# Stage07: Patch-wise Latent Diffusion (Autoregressive Window)
# ==========================================================

CONFIG = {
    # ------------------ Experiment ------------------
    "experiment_name": "stage07_patch_ldm_v2",
    "note": "Patch-wise latent diffusion with causal sliding window",
    "device": "cuda" if torch.cuda.is_available() else "cpu",

    # ------------------ 数据相关配置 ------------------
    # latent_dir: KLVAE/VQVAE latents (shape: C x D x H x W)
    # KLVAE latents (shape: 4 x 32 x 32 x 32 for 256^3 input)
    "latent_dir": r"D:\多尺度岩心数据集\LDM_Data\Latent_NPY\w192_s64",
    # phi_map_dir: precomputed local porosity maps (shape: gD x gH x gW)
    "phi_map_dir": r"D:\多尺度岩心数据集\LDM_Data\Phi_Maps_NPY\w192_s64",
    # raw data dir (used by preprocess_phi.py)
    "raw_data_dir": r"D:\多尺度岩心数据集\LDM_Data\Raw_NPY\w192_s64",
    # global porosity csv (optional, used when porosity_mode == 'global')
    "porosity_csv": r"",
    # latent stats
    "scale_factor": 1.410483,   # set to 1/std of latent
    "safe_threshold": 8.0,

    # phi map preprocessing
    "binarize_mode": "none",    # "fixed" | "otsu" | "none" (binary data -> use "none")
    "binarize_threshold": 0.5,  # used when binarize_mode == "fixed"
    # center-crop raw volume before phi map computation; set 0 to disable
    "phi_input_target_size": 192,

    # patch/window
    "latent_channels": 4,
    "patch_size": 8,       # in latent voxels (p)
    "window_size": 3,      # patches per axis (odd)
    # downsample factor from raw voxel space -> latent space
    "downsample_factor": 8,
    # porosity scalar used by PorosityEmbedder
    # "local": use phi_map value at target patch
    # "global": use per-sample porosity from CSV (fallback to phi mean)
    "porosity_mode": "local",
    # context_mode:
    # - "causal": lexicographic autoregressive order controlled by `order`
    # - "wavefront": dependency is previous patches along each axis (supports batched inference)
    # - "full": training-only dense context (in inference treated as causal fallback)
    "context_mode": "causal",
    # order: traversal order for causal mask
    "order": "ijk",        # i->j->k
    # training strategy:
    # randomize causal order in dataset sampling to reduce fixed-direction bias
    "train_random_order": True,
    # randomize causal direction (+/- on each axis) in dataset sampling.
    # if False, use fixed `train_direction`.
    "train_random_direction": True,
    # fixed direction for training when train_random_direction=False
    # examples: "+++", "--+", "+-+"
    "train_direction": "+++",
    # anchor sampling:
    # - "uniform": random target patch
    # - "low_context_boost": oversample hard patches with fewer known neighbors
    "anchor_sampling_mode": "low_context_boost",
    "anchor_boost_power": 1.0,
    "anchor_boost_min_weight": 0.05,
    # padding mode used when extracting patch windows: "edge" | "reflect" | "constant"
    "pad_mode": "edge",

    # ------------------ 模型相关配置 ------------------
    # input channels: x_t (C) + cond (C) + mask (1) + phi (1)
    "in_channels": 2 * 4 + 2,
    "out_channels": 4,
    "base_channels": 128,
    "channel_mults": (1, 2, 4),
    "use_attention": (False, True, True),
    "timesteps": 1000,

    # ------------------ 训练相关配置 ------------------
    "batch_size": 32,
    "num_workers": 8,
    "pin_memory": True,
    "epochs": 200,
    "lr": 3e-5,
    "resume": True,
    # when resuming from previous experiment, you can keep model weights but reset optimizer/scheduler
    "resume_load_optimizer": False,
    "resume_load_scheduler": True,
    "save_model_every": 2,
    "save_log_every": 1,
    # EMA
    # Note: with short training, too-large decay makes EMA lag behind.
    "ema_decay": 0.999,

    # loss
    "loss_type": "l1",
    "use_min_snr": True,
    "min_snr_gamma": 5.0,
    "x0_weight": 0.35,
    # additional target patch distribution constraint to reduce latent collapse
    "use_target_stats_loss": True,
    "target_stats_weight": 0.05,
    # set >0 to enable grad clipping (helps stabilize spikes after strategy changes)
    "grad_clip_norm": 1.0,
    "boundary_band_width": 1,
    "boundary_band_weight": 4.0,

    # ------------------ Eval During Training ------------------
    "eval_every_steps": 0,      # 0 disables
    "eval_ddim_steps": 200,
    "eval_seed": 1234,
    "eval_index": 66,            # which file index to sample from dataset
    "eval_output_dir": "eval",
    "eval_save_png": True,
    "eval_decode_voxel": True,
    "eval_vae_config_path": r"E:\chendou\paper_code\stage02_KLvae_single_code_v2\config\train_config copy.yaml",   # KLVAE config.yaml
    "eval_vae_ckpt_path": r"E:\chendou\paper_code\stage02_KLvae_single_code_v2\experiments\exp04_cube_structure_v1\ckpt_epoch_11.pt",     # KLVAE checkpoint
    "eval_voxel_save_png": True,
    # choose whether eval during training uses EMA weights or online model weights
    "eval_use_ema": False,


    # ------------------ 推理相关配置 ------------------
    "ddim_steps": 200,
    "seed": 1234,
    # inference traversal control:
    # - infer_random_order=True: sample one of 6 axis permutations per sample
    # - infer_random_direction=True: sample +/- direction for each axis per sample
    # if both False, use fixed `order` and `infer_direction`
    "infer_random_order": True,
    "infer_random_direction": True,
    # fixed direction for axes (i,j,k): "+" means low->high, "-" means high->low
    # examples: "+++", "--+", "+-+"
    "infer_direction": "+++",
    # max patch batch for parallel autoregressive inference (used by wavefront mode)
    "infer_max_patch_batch": 16,
    # with short training, prefer model_state_dict over ema_state_dict
    "infer_use_ema": True,
    "ckpt_path": r"E:\chendou\paper_code\stage07_latent_ldm_code\exp_results\stage07_patch_ldm_v2\models\unet_epoch_50.pth",
    "phi_map_path": r"D:\多尺度岩心数据集\LDM_Data\Phi_Maps_NPY\w192_s64\6-6-22_Global_Consistency_z1792_y192_x448.npy",
    "output_latent_path": "generated_latent.npy",
    "output_unscaled": True,    # divide by scale_factor before saving
}

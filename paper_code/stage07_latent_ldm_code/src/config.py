import torch

# ==========================================================
# Stage07: Patch-wise Latent Diffusion (Autoregressive Window)
# ==========================================================

CONFIG = {
    # ------------------ Experiment ------------------
    "experiment_name": "stage07_patch_ldm_v0",
    "note": "Patch-wise latent diffusion with causal sliding window",
    "device": "cuda" if torch.cuda.is_available() else "cpu",

    # ------------------ Data ------------------
    # latent_dir: KLVAE/VQVAE latents (shape: C x D x H x W)
    # KLVAE latents (shape: 4 x 32 x 32 x 32 for 256^3 input)
    "latent_dir": "/chendou_space/data/binary_klvae_latents_256",
    # phi_map_dir: precomputed local porosity maps (shape: gD x gH x gW)
    "phi_map_dir": "/chendou_space/data/binary_phi_maps_p32",
    # raw data dir (used by preprocess_phi.py)
    "raw_data_dir": "/chendou_space/data/binary_Training_Data",
    # global porosity csv (optional, used when porosity_mode == 'global')
    "porosity_csv": "/chendou_space/data/aligned_Training_Data/processing_report.csv",

    # latent stats
    "scale_factor": 1.0,   # set to 1/std of latent
    "safe_threshold": 8.0,

    # phi map preprocessing
    "binarize_mode": "none",    # "fixed" | "otsu" | "none" (binary data -> use "none")
    "binarize_threshold": 0.5,  # used when binarize_mode == "fixed"

    # patch/window
    "latent_channels": 4,
    "patch_size": 4,       # in latent voxels (p)
    "window_size": 3,      # patches per axis (odd)
    # downsample factor from raw voxel space -> latent space
    "downsample_factor": 8,
    # porosity scalar used by PorosityEmbedder
    # "local": use phi_map value at target patch
    # "global": use per-sample porosity from CSV (fallback to phi mean)
    "porosity_mode": "local",
    # context_mode: "causal" (only previous patches) or "full" (all except target)
    "context_mode": "causal",
    # order: traversal order for causal mask
    "order": "ijk",        # i->j->k

    # ------------------ Model ------------------
    # input channels: x_t (C) + cond (C) + mask (1) + phi (1)
    "in_channels": 2 * 4 + 2,
    "out_channels": 4,
    "base_channels": 128,
    "channel_mults": (1, 2, 4),
    "use_attention": (False, True, True),
    "timesteps": 1000,

    # ------------------ Train ------------------
    "batch_size": 1,
    "num_workers": 0,
    "pin_memory": True,
    "epochs": 200,
    "lr": 5e-5,
    "resume": True,
    "save_model_every": 20,

    # loss
    "loss_type": "l1",
    "use_min_snr": True,
    "min_snr_gamma": 5.0,
    "x0_weight": 0.2,
    "boundary_band_width": 1,
    "boundary_band_weight": 4.0,

    # ------------------ Inference ------------------
    "ddim_steps": 200,
    "seed": 1234,
    "ckpt_path": "",
    "phi_map_path": "",
    "output_latent_path": "generated_latent.npy",
    "output_unscaled": True,    # divide by scale_factor before saving
}

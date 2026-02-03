import torch

# ==========================================================
# Paths
# ==========================================================
PATHS = {
    "raw_data_dir": r"/chendou_space/data/aligned_Training_Data",
    "porosity_csv": r"/chendou_space/data/aligned_Training_Data/processing_report.csv",
    "exp_root": r"./exp_results",
}

# ==========================================================
# Task / Mask
# ==========================================================
TASK = {
    "axis": "D",          # D/H/W
    "ratio": 0.5,         # cut ratio
    "erosion_px": 2,      # shrink known region near boundary (pixel space)
}

# ==========================================================
# Coarse stage (global structure)
# ==========================================================
COARSE = {
    "enabled": True,
    "input_size": 256,     # raw size
    "coarse_size": 64,     # downsample size (64 or 128)

    "batch_size": 2,
    "num_workers": 8,
    "epochs": 200,
    "lr": 2e-4,
    "save_every": 5,
    "resume": True,

    "model_channels": 64,
    "channel_mults": (1, 2, 4),
    "use_attention": (False, True, True),
}

# ==========================================================
# Refine stage (detail)
# ==========================================================
REFINE = {
    "enabled": True,
    "patch_size": 128,
    "patch_overlap": 32,

    "batch_size": 2,
    "num_workers": 8,
    "epochs": 200,
    "lr": 2e-4,
    "save_every": 5,
    "resume": True,

    "model_channels": 64,
    "channel_mults": (1, 2, 4),
    "use_attention": (False, True, True),

    # optional cache to avoid recomputing coarse prediction during training
    "coarse_cache_dir": "",
}

# ==========================================================
# Loss weights
# ==========================================================
LOSS = {
    "known_weight": 0.1,      # small weight on known region
    "boundary_weight": 1.0,   # stronger on boundary band (unknown near cut)
    "boundary_band": 4,       # band width in pixels
    "coarse_guidance_weight": 0.1,  # refine: L1 to coarse guidance (unknown)
    "grad_weight": 0.0,       # optional gradient loss (keep 0 initially)
}

# ==========================================================
# Inference
# ==========================================================
INFER = {
    "ddp": False,
    "seed": 1234,
}

CONFIG = {
    "PATHS": PATHS,
    "TASK": TASK,
    "COARSE": COARSE,
    "REFINE": REFINE,
    "LOSS": LOSS,
    "INFER": INFER,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}

# config.py
CONFIG = {
    "exp_name": "stage1_vae_patch128_ch64",
    "data_path": "/chendou_space/data/aligned_Training_Data",
    "global_min": 878.0, 
    "global_max": 63366.0,
    "save_dir": "./experiments",
    
    # --- 训练策略调整 ---
    "patch_size": 128,        # <--- 新增：训练切块大小
    "batch_size": 4,          # <--- 128^3 下，A100 可以轻松跑 Batch=4 甚至 8
    "accumulate_grad_batches": 2, # 不需要累积了，或者设为 2
    "num_workers": 48,      # 充分利用多核 CPU
    "prefetch_factor": 6,     # 让 CPU 多预读一些数据给 GPU 备着
    
    # 模型参数 (可以恢复高性能配置)
    "model": {
        "in_channels": 1,
        "base_channels": 32,  # <--- 恢复到 64！保证纹理质量
        "z_channels": 4,
        "ch_mult": [1, 2, 4], 
        "dropout": 0.0,
        "use_checkpoint": False # <--- 关闭它！速度会快很多！
    },
    
    "lr": 4.5e-6,
    "epochs": 100,
    "loss_weights": {
        "kl_weight": 0.000001,
        "disc_weight": 0.5,
        "perceptual_weight": 1.0,
        "disc_start": 2001,
    }
}
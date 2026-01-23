import os
import csv
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from accelerate import Accelerator
from tqdm import tqdm
import numpy as np

# Import models
from model_vae_3d import AutoencoderKL, NLayerDiscriminator3D
from dataset_raw import RawCoreDataset

# --- Config ---
CONFIG = {
    "data_roots": [
        "/chendou_space/data/aligned_Training_Data_Interactive",
        # "/path/to/your/raw_data_folder_1", # 修改这里
        # "/path/to/your/raw_data_folder_2"
    ],
    "batch_size": 4, # A100 80G 可以开大一点，比如 4 或 8
    "lr": 1e-4,
    "kl_weight": 1e-6, # 初始 KL 权重，通常很小
    "disc_weight": 0.5,
    "disc_start": 2000, # Steps 之后才开启判别器训练，让 VAE 先收敛一会儿
    "epochs": 100,
    "save_every": 2,
    "output_dir": "./vae_results_exp01",
    "latent_dim": 4 # 对应 Stage 2 的 Channels
}

def train():
    # 1. Setup Accelerator
    accelerator = Accelerator(log_with="all", project_dir=CONFIG["output_dir"])
    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    
    # CSV Logger
    csv_path = os.path.join(CONFIG["output_dir"], "training_log.csv")
    if accelerator.is_main_process:
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Epoch', 'Step', 'Loss_G', 'Loss_Recon', 'Loss_KL', 'Loss_D'])

    # 2. Model & Optimizer
    vae = AutoencoderKL(z_channels=CONFIG["latent_dim"])
    discriminator = NLayerDiscriminator3D()
    
    # 使用 PatchGAN 损失
    opt_g = torch.optim.AdamW(vae.parameters(), lr=CONFIG["lr"], betas=(0.5, 0.9))
    opt_d = torch.optim.AdamW(discriminator.parameters(), lr=CONFIG["lr"], betas=(0.5, 0.9))

    # 3. Data
    dataset = RawCoreDataset(CONFIG["data_roots"])
    dataloader = DataLoader(dataset, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=8, pin_memory=True)

    # 4. Prepare
    vae, discriminator, opt_g, opt_d, dataloader = accelerator.prepare(
        vae, discriminator, opt_g, opt_d, dataloader
    )

    global_step = 0
    
    for epoch in range(CONFIG["epochs"]):
        vae.train()
        discriminator.train()
        
        for batch in tqdm(dataloader, disable=not accelerator.is_main_process):
            global_step += 1
            real_img = batch
            
            # ========================
            # Train Generator (VAE)
            # ========================
            opt_g.zero_grad()
            
            recon_img, mean, logvar = vae(real_img, sample_posterior=True)
            
            # 1. Reconstruction Loss (L1)
            loss_recon = F.l1_loss(recon_img, real_img)
            
            # 2. KL Divergence
            loss_kl = 0.5 * torch.sum(torch.exp(logvar) + mean**2 - 1. - logvar, dim=[1,2,3,4])
            loss_kl = torch.mean(loss_kl)
            
            # 3. GAN Generator Loss (让生成的图像骗过判别器)
            loss_g_gan = 0.0
            if global_step > CONFIG["disc_start"]:
                logits_fake = discriminator(recon_img)
                # Hinge Loss or BCE, here using LSGAN (MSE) style for stability
                loss_g_gan = torch.mean((logits_fake - 1.0) ** 2)
            
            loss_total_g = loss_recon + CONFIG["kl_weight"] * loss_kl + CONFIG["disc_weight"] * loss_g_gan
            
            accelerator.backward(loss_total_g)
            opt_g.step()
            
            # ========================
            # Train Discriminator
            # ========================
            loss_d = 0.0
            if global_step > CONFIG["disc_start"]:
                opt_d.zero_grad()
                
                logits_real = discriminator(real_img.detach())
                logits_fake = discriminator(recon_img.detach())
                
                # LSGAN Loss
                loss_d_real = torch.mean((logits_real - 1.0) ** 2)
                loss_d_fake = torch.mean(logits_fake ** 2)
                loss_d = 0.5 * (loss_d_real + loss_d_fake)
                
                accelerator.backward(loss_d)
                opt_d.step()

            # Logging
            if accelerator.is_main_process and global_step % 10 == 0:
                with open(csv_path, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        epoch, global_step, 
                        loss_total_g.item(), 
                        loss_recon.item(), 
                        loss_kl.item(), 
                        loss_d.item() if isinstance(loss_d, torch.Tensor) else 0.0
                    ])

        # Save Model
        if accelerator.is_main_process and (epoch + 1) % CONFIG["save_every"] == 0:
            save_path = os.path.join(CONFIG["output_dir"], f"vae_epoch_{epoch+1}.pth")
            torch.save(accelerator.get_state_dict(vae), save_path)
            print(f"Saved model to {save_path}")

if __name__ == "__main__":
    train()
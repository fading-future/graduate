import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import pandas as pd

# 导入你的模块
from src.dataset_rev import VQVAEDataset 
from src.model_vqvae import VQVAE3D         # 你的 Generator
from src.model_discriminator import NLayerDiscriminator3D # 新增的 Discriminator
from src.loss_gan import VQGANLoss          # 新增的 Loss
from src.config import CONFIG
from utils.get_root_path import get_project_root

# --- 配置 ---
DIR_ROOT = get_project_root()
EXP_ROOT = os.path.join(DIR_ROOT, CONFIG['experiment_name'] + "_GAN") # 新建一个实验文件夹
MODEL_DIR = os.path.join(EXP_ROOT, "models")
LOG_DIR = os.path.join(EXP_ROOT, "logs")

# 关键参数
DISC_START_STEP = 2000  # 前 2000 步只练 VQVAE，让它先学会基本形状，然后再开 GAN
LR = 1e-4

def train_vqgan():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    
    # 1. Dataset
    dataset = VQVAEDataset(
        data_dir=CONFIG['processed_data_dir'], 
        volume_size=CONFIG['image_size'],
        augment=True
    )
    dataloader = DataLoader(dataset, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=4, pin_memory=True)
    
    # 2. Models
    # Generator (VQVAE)
    vqvae = VQVAE3D(in_channels=1, embedding_dim=CONFIG['embedding_dim'], num_embeddings=CONFIG['num_embeddings']).to(CONFIG['device'])
    
    # Discriminator (PatchGAN)
    discriminator = NLayerDiscriminator3D(input_nc=1).to(CONFIG['device'])
    
    # Loss Module
    loss_module = VQGANLoss(disc_start=DISC_START_STEP, disc_weight=0.5).to(CONFIG['device'])

    # 3. Optimizers (分开优化)
    opt_vq = optim.Adam(vqvae.parameters(), lr=LR, betas=(0.5, 0.9))
    opt_disc = optim.Adam(discriminator.parameters(), lr=LR, betas=(0.5, 0.9))

    # 4. Logger CSV
    log_csv_path = os.path.join(LOG_DIR, "training_log_gan.csv")
    with open(log_csv_path, 'w') as f:
        f.write("Epoch,Step,Total_G_Loss,Recon_Loss,VQ_Loss,GAN_G_Loss,Disc_Loss,Perplexity\n")

    print(f"Start VQ-GAN Training... GAN starts at step {DISC_START_STEP}")
    
    global_step = 0
    
    for epoch in range(CONFIG['epochs']):
        vqvae.train()
        discriminator.train()
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
        for batch in pbar:
            global_step += 1
            images = batch["GT"].to(CONFIG['device'])
            
            # ====================================================
            #  Phase 1: Generator Update (VQVAE)
            # ====================================================
            opt_vq.zero_grad()
            
            # Forward
            reconstructions, vq_loss, perplexity = vqvae(images)
            
            # 计算 Discriminator 对假图的判决 (用于计算 G 的 Loss)
            if global_step > DISC_START_STEP:
                logits_fake = discriminator(reconstructions)
            else:
                logits_fake = None

            # Calculate Loss (Optimizer idx 0)
            loss_g, rec_loss, gan_g_loss = loss_module(
                codebook_loss=vq_loss,
                inputs=images,
                reconstructions=reconstructions,
                optimizer_idx=0,
                global_step=global_step,
                logits_fake=logits_fake
            )
            
            loss_g.backward()
            opt_vq.step()

            # ====================================================
            #  Phase 2: Discriminator Update
            # ====================================================
            d_loss_val = 0.0
            
            # 只有当预热结束后，才开始训练判别器
            if global_step > DISC_START_STEP:
                opt_disc.zero_grad()
                
                # 判别真图
                logits_real = discriminator(images.detach().requires_grad_())
                # 判别假图 (detach 这里的 reconstructions，不要传梯度回 VQVAE)
                logits_fake = discriminator(reconstructions.detach())
                
                # Calculate Loss (Optimizer idx 1)
                d_loss = loss_module(
                    codebook_loss=None,
                    inputs=None,
                    reconstructions=None,
                    optimizer_idx=1,
                    global_step=global_step,
                    logits_real=logits_real,
                    logits_fake=logits_fake
                )
                
                d_loss.backward()
                opt_disc.step()
                d_loss_val = d_loss.item()

            # ====================================================
            #  Logging
            # ====================================================
            # 记录到 CSV
            gan_loss_item = gan_g_loss.item() if torch.is_tensor(gan_g_loss) else gan_g_loss
            
            with open(log_csv_path, 'a') as f:
                f.write(f"{epoch+1},{global_step},{loss_g.item():.5f},{rec_loss.item():.5f},{vq_loss.item():.5f},{gan_loss_item:.5f},{d_loss_val:.5f},{perplexity.item():.2f}\n")

            pbar.set_postfix(
                Rec=f"{rec_loss.item():.3f}",
                G_GAN=f"{gan_loss_item:.3f}",
                D_Loss=f"{d_loss_val:.3f}",
                Perp=f"{perplexity.item():.1f}"
            )
        
        # Save Model
        if (epoch+1) % 5 == 0:
            torch.save(vqvae.state_dict(), os.path.join(MODEL_DIR, f"vqgan_epoch_{epoch+1}.pth"))
            torch.save(discriminator.state_dict(), os.path.join(MODEL_DIR, f"disc_epoch_{epoch+1}.pth"))

if __name__ == "__main__":
    train_vqgan()
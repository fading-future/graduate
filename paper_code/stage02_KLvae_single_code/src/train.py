# train.py
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from config import CONFIG
from dataset import Core3DDataset
from vae_model import VAE3D
from discriminator import NLayerDiscriminator3D
from losses import VAEGANLoss

# 开启 TF32 (A100 必开)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

def main():
    device = torch.device("cuda")
    os.makedirs(CONFIG['save_dir'], exist_ok=True)

    # 1. Data
    dataset = Core3DDataset(CONFIG['data_path'])
    loader = DataLoader(dataset, batch_size=CONFIG['batch_size'], shuffle=True, 
                        num_workers=CONFIG['num_workers'], pin_memory=True)
    print(f"Data loaded: {len(dataset)} volumes.")

    # 2. Models
    vae = VAE3D(CONFIG['model']).to(device)
    discriminator = NLayerDiscriminator3D().to(device)
    loss_module = VAEGANLoss(**CONFIG['loss_weights']).to(device)

    # 3. Optimizers
    opt_vae = optim.AdamW(vae.parameters(), lr=CONFIG['lr'], betas=(0.5, 0.9))
    opt_disc = optim.AdamW(discriminator.parameters(), lr=CONFIG['lr'], betas=(0.5, 0.9))
    
    scaler = torch.cuda.amp.GradScaler() # Mixed Precision

    global_step = 0

    # 4. Training Loop
    for epoch in range(CONFIG['epochs']):
        vae.train()
        discriminator.train()
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}")
        
        for batch in pbar:
            global_step += 1
            real_img = batch.to(device) # (B, 1, 256, 256, 256)
            
            # --- Train VAE ---
            with torch.cuda.amp.autocast():
                recon, mean, logvar = vae(real_img)
                loss_vae, log_vae = loss_module(
                    real_img, recon, (mean, logvar), 
                    optimizer_idx=0, 
                    global_step=global_step, 
                    discriminator=discriminator,
                    disc_start=CONFIG['loss_weights']['disc_start']
                )
            
            opt_vae.zero_grad()
            scaler.scale(loss_vae).backward()
            scaler.unscale_(opt_vae)
            torch.nn.utils.clip_grad_norm_(vae.parameters(), 1.0)
            scaler.step(opt_vae)
            
            # --- Train Discriminator ---
            # 只有在过了 warmup 阶段才训练 D
            if global_step > CONFIG['loss_weights']['disc_start']:
                with torch.cuda.amp.autocast():
                    loss_disc, log_disc = loss_module(
                        real_img, recon, (mean, logvar), 
                        optimizer_idx=1, 
                        global_step=global_step, 
                        discriminator=discriminator,
                        disc_start=CONFIG['loss_weights']['disc_start']
                    )
                
                opt_disc.zero_grad()
                scaler.scale(loss_disc).backward()
                scaler.step(opt_disc)
            else:
                log_disc = {"train/d_loss": 0.0}

            scaler.update()

            # Logging
            desc = f"VAE: {log_vae['train/total_loss']:.4f}"
            if 'train/d_loss' in log_disc:
                desc += f" | D: {log_disc['train/d_loss']:.4f}"
            pbar.set_postfix_str(desc)

        # Save Checkpoint
        if (epoch + 1) % 5 == 0:
            torch.save({
                'vae': vae.state_dict(),
                'disc': discriminator.state_dict(),
                'epoch': epoch
            }, os.path.join(CONFIG['save_dir'], f"checkpoint_ep{epoch+1}.pth"))
            print(f"Saved model at epoch {epoch+1}")

if __name__ == "__main__":
    main()
import os
import time
import csv
import datetime
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from config import CONFIG
from dataset import Core3DDataset
from vae_model import VAE3D
from discriminator import NLayerDiscriminator3D
from losses import VAEGANLoss

# ==========================================
# 🚀 A100 极速加速设置
# ==========================================
# 允许 TF32 (TensorFloat-32) 进行矩阵乘法和卷积，大幅提升 A100 速度
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
# 开启 CuDNN 自动寻优 (针对固定输入尺寸 128^3 非常有效)
torch.backends.cudnn.benchmark = True 

def main():
    device = torch.device("cuda")
    os.makedirs(CONFIG['save_dir'], exist_ok=True)

    # 1. 初始化 CSV 日志文件
    # =======================================================
    csv_path = os.path.join(CONFIG['save_dir'], f"training_logs_{CONFIG['exp_name']}.csv")
    print(f"📝 Logging full training metrics to: {csv_path}")
    
    # 定义 CSV 表头
    headers = [
        "Epoch", "Global_Step", "Time", "LR",
        "VAE_Total_Loss", "Rec_Loss(L1)", "P_Loss(LPIPS)", "KL_Loss", "G_Loss(Adv)", "D_Weight",
        "D_Loss(Disc)", "Logits_Real", "Logits_Fake"
    ]
    
    # 如果是第一次运行，创建文件并写入表头；如果是断点续训，则追加
    mode = 'a' if os.path.exists(csv_path) else 'w'
    log_file = open(csv_path, mode, newline='', buffering=1) # buffering=1 表示行缓冲，每写一行就存盘
    writer = csv.writer(log_file)
    if mode == 'w':
        writer.writerow(headers)
    # =======================================================

    # 2. Data
    dataset = Core3DDataset(
        data_dir=CONFIG['data_path'],
        global_min=CONFIG['global_min'],
        global_max=CONFIG['global_max'],
        patch_size=CONFIG['patch_size'], # 确保 Config 里有这个
        train=True
    )
    
    loader = DataLoader(
        dataset, 
        batch_size=CONFIG['batch_size'], 
        shuffle=True, 
        num_workers=CONFIG['num_workers'], 
        pin_memory=True,
        prefetch_factor=CONFIG.get('prefetch_factor', 2), # 默认给 2
        persistent_workers=True # 减少每个 Epoch 重新创建 Worker 的开销
    )
    print(f"✅ Data loaded: {len(dataset)} volumes. Batch Size: {CONFIG['batch_size']}")

    # 3. Models
    vae = VAE3D(CONFIG['model']).to(device)
    discriminator = NLayerDiscriminator3D().to(device)
    loss_module = VAEGANLoss(**CONFIG['loss_weights']).to(device)

    # 4. Optimizers
    opt_vae = optim.AdamW(vae.parameters(), lr=CONFIG['lr'], betas=(0.5, 0.9))
    opt_disc = optim.AdamW(discriminator.parameters(), lr=CONFIG['lr'], betas=(0.5, 0.9))
    
    # 注意：A100 使用 BFloat16 通常不需要 GradScaler，移除它可以进一步提速并减少 NaN 风险
    # scaler = torch.amp.GradScaler('cuda') 

    global_step = 0
    start_time = time.time()

    # 5. Training Loop
    for epoch in range(CONFIG['epochs']):
        vae.train()
        discriminator.train()
        
        # 进度条显示更多信息
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']}")
        
        for batch in pbar:
            global_step += 1
            real_img = batch.to(device) # (B, 1, 128, 128, 128)
            
            # =================================================
            # Phase 1: Train VAE (Generator)
            # =================================================
            # 使用 bfloat16 (A100 专用，比 float16 更稳更快)
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                recon, mean, logvar = vae(real_img)
                
                loss_vae, log_vae = loss_module(
                    inputs=real_img, 
                    reconstructions=recon, 
                    posteriors=(mean, logvar), 
                    optimizer_idx=0, 
                    global_step=global_step, 
                    discriminator=discriminator,
                    last_layer=vae.decoder.conv_out.weight,
                    split="train"
                )
            
            # Backward VAE
            opt_vae.zero_grad()
            loss_vae.backward()
            torch.nn.utils.clip_grad_norm_(vae.parameters(), 1.0) # 梯度裁剪防止爆炸
            opt_vae.step()
            
            # =================================================
            # Phase 2: Train Discriminator
            # =================================================
            loss_disc_val = 0.0
            logits_real_val = 0.0
            logits_fake_val = 0.0

            if global_step > CONFIG['loss_weights']['disc_start']:
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    # Detach recon 以防梯度传回 VAE
                    loss_disc, log_disc = loss_module(
                        inputs=real_img, 
                        reconstructions=recon.detach(), 
                        posteriors=(mean, logvar), 
                        optimizer_idx=1, 
                        global_step=global_step, 
                        discriminator=discriminator,
                        split="train"
                    )
                
                # Backward Disc
                opt_disc.zero_grad()
                loss_disc.backward()
                opt_disc.step()
                
                # 记录值
                loss_disc_val = log_disc.get('train/d_loss', 0.0)
                logits_real_val = log_disc.get('train/logits_real', 0.0)
                logits_fake_val = log_disc.get('train/logits_fake', 0.0)
            
            # =================================================
            # 📝 Logging to CSV & Tqdm
            # =================================================
            # 获取当前时间字符串
            current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            current_lr = opt_vae.param_groups[0]['lr']
            
            # 准备要写入 CSV 的一行数据
            # 对应 Headers: 
            # Epoch, Global_Step, Time, LR, 
            # VAE_Total, Rec, P, KL, G, D_Weight, 
            # D_Loss, Logits_Real, Logits_Fake
            
            row = [
                epoch + 1,
                global_step,
                current_time,
                f"{current_lr:.2e}",
                f"{log_vae['train/total_loss']:.6f}",
                f"{log_vae['train/rec_loss']:.6f}",
                f"{log_vae.get('train/p_loss', 0):.6f}",
                f"{log_vae['train/kl_loss']:.6f}",
                f"{log_vae.get('train/g_loss', 0):.6f}",
                f"{log_vae.get('train/d_weight', 0):.4f}",
                f"{loss_disc_val:.6f}",
                f"{logits_real_val:.4f}",
                f"{logits_fake_val:.4f}"
            ]
            writer.writerow(row)
            
            # 更新终端显示 (只挑几个重点显示，避免刷屏)
            desc_str = (f"L1: {log_vae['train/rec_loss']:.4f} | "
                        f"P: {log_vae.get('train/p_loss', 0):.4f} | "
                        f"KL: {log_vae['train/kl_loss']:.4f}")
            
            if global_step > CONFIG['loss_weights']['disc_start']:
                desc_str += f" | D: {loss_disc_val:.4f} | G: {log_vae.get('train/g_loss', 0):.4f}"
            
            pbar.set_postfix_str(desc_str)

        # =================================================
        # Save Checkpoint
        # =================================================
        if (epoch + 1) % 1 == 0:
            save_path = os.path.join(CONFIG['save_dir'], f"checkpoint_ep{epoch+1}.pth")
            print(f"💾 Saving checkpoint to {save_path}...")
            torch.save({
                'vae': vae.state_dict(),
                'disc': discriminator.state_dict(),
                'optimizer_vae': opt_vae.state_dict(),
                'optimizer_disc': opt_disc.state_dict(),
                'epoch': epoch,
                'global_step': global_step
            }, save_path)
    
    # 关闭文件
    log_file.close()
    print("✅ Training Finished!")

if __name__ == "__main__":
    main()
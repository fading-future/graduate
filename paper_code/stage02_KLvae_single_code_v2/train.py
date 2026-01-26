import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import yaml
import shutil
from tqdm import tqdm
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam

from models.vae import KLVAE3D
from models.discriminator import NLayerDiscriminator3D
from data.dataset import CubeDataset
from utils.logger import CSVLogger

# === 开启 A100 的 TF32 加速 ===
# 这行代码允许 PyTorch 在 A100 上使用 TF32 格式进行矩阵乘法
# 理论上可以带来数倍的加速，且精度损失几乎可以忽略不计
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def hinge_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.relu(1. - logits_real))
    loss_fake = torch.mean(F.relu(1. + logits_fake))
    return 0.5 * (loss_real + loss_fake)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/train_config.yaml')
    parser.add_argument('--resume', type=str, default=None, help='path to checkpoint to resume from')
    args = parser.parse_args()

    # 1. 配置与目录
    cfg = load_config(args.config)
    exp_dir = os.path.join(cfg['experiment']['save_dir'], cfg['experiment']['exp_name'])
    os.makedirs(exp_dir, exist_ok=True)
    
    # 备份配置 (如果是 resume，尽量不要覆盖原配置，或者确认配置一致)
    if args.resume is None:
        shutil.copy(args.config, os.path.join(exp_dir, 'config.yaml'))
    
    logger = CSVLogger(exp_dir)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on {device}")

    # 2. 数据
    dataset = CubeDataset(cfg['data']['data_root'], cfg['data']['file_extension'], crop_size=cfg['data']['croped_size'], is_train=True)
    dataloader = DataLoader(
        dataset, 
        batch_size=cfg['train']['batch_size'], 
        shuffle=True, 
        num_workers=cfg['data']['num_workers'],
        pin_memory=True
    )

    # 3. 模型
    vae = KLVAE3D(cfg).to(device)
    discriminator = NLayerDiscriminator3D().to(device)

    # === 新增：开启编译模式 ===
    # mode='max-autotune' 最快但启动慢，'reduce-overhead' 启动快
    # print("Compiling model... (First step will be slow)")
    # vae = torch.compile(vae, mode='reduce-overhead') 
    # discriminator = torch.compile(discriminator, mode='reduce-overhead')

    # 修改为 max-autotune 模式以获得最佳训练速度
    # print("Compiling model... (Max-Autotune: Slow start, fastest training)")
    # vae = torch.compile(vae, mode='max-autotune') 
    # discriminator = torch.compile(discriminator, mode='max-autotune')

    # 修改为默认模式，兼顾稳定性和速度
    print("Compiling model... (Default mode is stable and fast)")
    vae = torch.compile(vae) 
    discriminator = torch.compile(discriminator)

    # 4. 优化器
    opt_vae = Adam(vae.parameters(), lr=float(cfg['train']['lr']), betas=(0.5, 0.9))
    opt_disc = Adam(discriminator.parameters(), lr=float(cfg['train']['lr']), betas=(0.5, 0.9))

    # 5. A100 混合精度
    scaler = torch.amp.GradScaler('cuda')

    global_step = 0
    start_epoch = 0 

    # === 加载断点逻辑 (修复版) ===
    if args.resume is not None:
        if os.path.isfile(args.resume):
            print(f"Loading checkpoint from {args.resume}...")
            checkpoint = torch.load(args.resume, map_location=device)
            
            # --- 修复核心：处理 torch.compile 带来的 _orig_mod 前缀 ---
            def fix_compile_state_dict(state_dict):
                new_state_dict = {}
                for k, v in state_dict.items():
                    # 如果 checkpoint 里有 _orig_mod 前缀，去掉它
                    if k.startswith("_orig_mod."):
                        new_state_dict[k[10:]] = v
                    else:
                        new_state_dict[k] = v
                return new_state_dict

            # 1. 修复 VAE 参数
            vae_state = fix_compile_state_dict(checkpoint['vae_state_dict'])
            # 如果当前模型被编译了，但 checkpoint 是旧的（没前缀），
            # 或者当前模型没编译，但 checkpoint 是新的（有前缀），
            # 这种名字不匹配会导致报错。
            # 最稳妥的方法是：先尝试直接加载，如果不行，再尝试加上或去掉前缀
            
            # 这里的逻辑是：我们总是把 checkpoint 清洗成“干净的原始名字”
            # 然后利用 load_state_dict 的 strict=False (可选) 或者让模型自己去适配
            
            # 但由于你开启了 torch.compile，当前模型实例 vae 内部已经是 optimized 的了
            # 实际上，torch.compile 返回的 object 仍然代理了原始模型。
            # 官方推荐做法：在 compile 之前 load_state_dict，或者清洗 key。
            
            # 简单粗暴且最有效的方法：直接加载到 vae (如果是 compiled model, 它能识别原始 key)
            # 报错的原因通常是 checkpoint 里有 _orig_mod，而你现在的模型期待原始 key（或者反之）
            
            # 让我们使用清洗后的干净 key (没有 _orig_mod)
            # torch.compile 的模型通常能接受原始 key
            try:
                vae.load_state_dict(vae_state)
            except RuntimeError as e:
                # 如果失败，可能是因为当前是编译模式，非要带前缀？
                # 通常不会，报错显示的是 checkpoint 里没有 _orig_mod，但模型想要 _orig_mod (或者反之)
                # 你的报错显示：Missing key "_orig_mod..." -> 说明当前模型想要带前缀的
                # 而 checkpoint 里是 "encoder..." -> 说明 checkpoint 是旧的
                
                # 补救措施：如果当前模型需要前缀，我们给它加上
                print("Standard loading failed, trying adding '_orig_mod.' prefix for compiled model...")
                compiled_state = {"_orig_mod." + k: v for k, v in vae_state.items()}
                vae.load_state_dict(compiled_state)

            # 2. 修复判别器参数 (同理)
            disc_state = fix_compile_state_dict(checkpoint['disc_state_dict'])
            try:
                discriminator.load_state_dict(disc_state)
            except RuntimeError:
                compiled_disc = {"_orig_mod." + k: v for k, v in disc_state.items()}
                discriminator.load_state_dict(compiled_disc)

            # 3. 优化器参数通常不需要改 key，直接加载
            try:
                opt_vae.load_state_dict(checkpoint['optimizer_vae'])
            except:
                print("Warning: Failed to load VAE optimizer state. Starting optimizer from scratch.")
            
            # 4. 恢复步数
            start_epoch = checkpoint['epoch'] + 1
            if 'global_step' in checkpoint:
                global_step = checkpoint['global_step']
            else:
                csv_step = logger.get_last_step()
                if csv_step > 0:
                    global_step = csv_step + 1
                else:
                    global_step = start_epoch * len(dataloader)
            
            print(f"Resumed successfully. Start Epoch: {start_epoch}, Start Step: {global_step}")
        else:
            print(f"Warning: No checkpoint found at {args.resume}")

    # # === 加载断点逻辑 (优化版) ===
    # if args.resume is not None:
    #     if os.path.isfile(args.resume):
    #         print(f"Loading checkpoint from {args.resume}...")
    #         checkpoint = torch.load(args.resume, map_location=device)
            
    #         vae.load_state_dict(checkpoint['vae_state_dict'])
    #         discriminator.load_state_dict(checkpoint['disc_state_dict'])
    #         opt_vae.load_state_dict(checkpoint['optimizer_vae'])
            
    #         # 1. 尝试从 checkpoint 恢复 Epoch
    #         start_epoch = checkpoint['epoch'] + 1
            
    #         # 2. 尝试恢复 global_step
    #         # 优先级：Checkpoint > CSV文件 > 估算
    #         if 'global_step' in checkpoint:
    #             global_step = checkpoint['global_step']
    #             print(f"Global step loaded from checkpoint: {global_step}")
    #         else:
    #             # 如果是旧模型没存 step，尝试读 CSV
    #             csv_step = logger.get_last_step()
    #             if csv_step > 0:
    #                 global_step = csv_step + 1 # 下一步是记录的最后一步 + 1
    #                 print(f"Global step recovered from CSV: {global_step}")
    #             else:
    #                 # 最差情况：估算
    #                 global_step = start_epoch * len(dataloader)
    #                 print(f"Global step estimated from epoch: {global_step}")

    #         print(f"Resumed successfully. Start Epoch: {start_epoch}, Start Step: {global_step}")
    #     else:
    #         print(f"Warning: No checkpoint found at {args.resume}")

    kl_weight = float(cfg['loss']['kl_weight'])
    disc_start = int(cfg['loss']['disc_start'])
    disc_weight = float(cfg['loss']['disc_weight'])

    # 6. 训练循环
    # === 修改：range 必须从 start_epoch 开始，否则 resume 无效 ===
    for epoch in range(start_epoch, cfg['train']['epochs']):
        vae.train()
        discriminator.train()

        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{cfg['train']['epochs']}")
        
        for batch_idx, images in enumerate(progress_bar):
            images = images.to(device) # [-1, 1]
            
            # =========================================
            # part 1: 训练 VAE (Generator)
            # =========================================
            opt_vae.zero_grad()
            
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                reconstructions, posterior = vae(images)
                
                # 1. Reconstruction Loss
                rec_loss = torch.mean(torch.abs(images - reconstructions))

                # 2. KL Loss
                kl_loss = posterior.kl()
                kl_loss = torch.mean(kl_loss)

                # 3. Adversarial Loss
                if global_step > disc_start:
                    logits_fake = discriminator(reconstructions)
                    g_adv_loss = -torch.mean(logits_fake)
                else:
                    g_adv_loss = torch.tensor(0.0, device=device)

                total_loss = rec_loss + kl_weight * kl_loss + disc_weight * g_adv_loss

            # --- 关键修改开始 ---
            # 1. 先反向传播算出梯度
            scaler.scale(total_loss).backward()
            
            # 2. 解包梯度 (Unscale) - 必须在 clip 之前！
            scaler.unscale_(opt_vae)
            
            # 3. 梯度裁剪 (防止梯度爆炸) - max_norm 建议设为 1.0
            torch.nn.utils.clip_grad_norm_(vae.parameters(), max_norm=1.0)
            
            # 4. 更新权重 (Step)
            scaler.step(opt_vae)
            # --- 关键修改结束 ---
            
            # =========================================
            # part 2: 训练 Discriminator
            # =========================================
            loss_d = torch.tensor(0.0, device=device)
            if global_step > disc_start:
                opt_disc.zero_grad()
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    logits_real = discriminator(images)
                    logits_fake = discriminator(reconstructions.detach())
                    loss_d = hinge_d_loss(logits_real, logits_fake)

                # --- 判别器也要裁剪 ---
                scaler.scale(loss_d).backward()
                
                scaler.unscale_(opt_disc)
                torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)
                
                scaler.step(opt_disc)

            # 更新 Scaler
            scaler.update()

            progress_bar.set_postfix({
                "Rec": f"{rec_loss.item():.4f}",
                "KL": f"{kl_loss.item():.1f}", 
                "G_Adv": f"{g_adv_loss.item():.4f}"
            })

            # --- Logging ---
            if global_step % cfg['train']['log_interval'] == 0:
                logger.log({
                    "Epoch": epoch,
                    "Step": global_step,
                    "Loss_Total": total_loss.item(),
                    "Loss_Recon": rec_loss.item(),
                    "Loss_KL": kl_loss.item(),
                    "Loss_G_Adv": g_adv_loss.item(),
                    "Loss_D": loss_d.item() if isinstance(loss_d, float) else loss_d.item()
                })
            
            global_step += 1

        # --- Checkpointing ---
        if (epoch + 1) % cfg['train']['save_interval'] == 0:
            save_path = os.path.join(exp_dir, f"checkpoint_epoch_{epoch+1}.pt")
            
            # === 修改：将 global_step 存入 checkpoint ===
            torch.save({
                'epoch': epoch,
                'global_step': global_step, # 这样下次 resume 就非常精准了
                'vae_state_dict': vae.state_dict(),
                'disc_state_dict': discriminator.state_dict(),
                'optimizer_vae': opt_vae.state_dict(),
            }, save_path)
            
            print(f"Saving visualization for epoch {epoch+1}...")
            # 确保 images 和 reconstructions 有数据 (防止 batch 为空)
            if 'images' in locals() and 'reconstructions' in locals():
                logger.save_comparison_grid(images, reconstructions, epoch+1, global_step)
            
            print(f"Saved checkpoint and visualization to {exp_dir}")

if __name__ == "__main__":
    main()
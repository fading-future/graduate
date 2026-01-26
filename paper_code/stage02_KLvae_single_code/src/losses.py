import torch
import torch.nn as nn
import torch.nn.functional as F
import lpips  # pip install lpips

# ==========================================
# 1. 3D PatchGAN 判别器 (增强版)
# ==========================================
class NLayerDiscriminator3D(nn.Module):
    def __init__(self, input_nc=1, ndf=64, n_layers=3, use_actnorm=False):
        super(NLayerDiscriminator3D, self).__init__()
        kw = 4
        padw = 1
        sequence = [nn.Conv3d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), 
                    nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1

        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv3d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=False),
                # A100 Batch Size较小时，GroupNorm 比 BatchNorm 更稳定
                nn.GroupNorm(16, ndf * nf_mult), 
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv3d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=False),
            nn.GroupNorm(16, ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]
        
        # 输出层：输出 1 通道的 Logits map
        sequence += [nn.Conv3d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """
        Standard Forward.
        """
        return self.model(input)

# ==========================================
# 2. 生产级 VAE-GAN Loss 模块
# ==========================================
class VAEGANLoss(nn.Module):
    def __init__(self, 
                 kl_weight=1.0e-6, 
                 disc_weight=0.5, 
                 perceptual_weight=1.0, 
                 disc_start=5001,
                 disc_factor=1.0, # 用于控制 GAN Loss 的缩放
                 perceptual_type="vgg"
                 ):
        super().__init__()
        self.kl_weight = kl_weight
        self.disc_weight = disc_weight
        self.perceptual_weight = perceptual_weight
        self.disc_start = disc_start
        self.disc_factor = disc_factor
        
        print(f"Loading LPIPS model ({perceptual_type})...")
        self.perceptual_loss = lpips.LPIPS(net=perceptual_type).eval()
        # 冻结 LPIPS 权重，不参与训练
        for param in self.perceptual_loss.parameters():
            param.requires_grad = False
            
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()

    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer=None):
        """
        【核心代码】自适应权重计算
        计算 Reconstruction Loss 和 GAN Loss 针对最后一层参数的梯度模长之比。
        """
        if last_layer is not None:
            # 手动计算梯度 (autograd.grad)
            nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
            
            # 简单的梯度自适应逻辑
            d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
            
            # 限制权重范围，防止梯度爆炸或消失 (通常在 0.0 到 1e4 之间)
            d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
            
            # 乘以一个缩放因子 (disc_factor)，通常是 1.0 或 0.8
            d_weight = d_weight * self.disc_factor
        else:
            d_weight = torch.tensor(0.0)
            
        return d_weight

    def forward(self, inputs, reconstructions, posteriors, optimizer_idx, global_step, 
                discriminator=None, last_layer=None, split="train"):
        """
        inputs: GT (B, 1, D, H, W)
        reconstructions: Pred (B, 1, D, H, W)
        posteriors: (mean, logvar) from VAE
        optimizer_idx: 0 for VAE (Generator), 1 for Discriminator
        last_layer: VAE Decoder 最后一层卷积的权重 (用于计算自适应权重)
        """
        
        # 1. 基础重建 Loss (L1 + L2 混合通常效果最好，或者纯 L1)
        rec_loss = torch.abs(inputs - reconstructions) # L1
        # rec_loss = rec_loss + 0.1 * (inputs - reconstructions)**2 # Optional Mix
        
        # 2. Perceptual Loss (3D Multi-View Slicing)
        if self.perceptual_weight > 0:
            B, C, D, H, W = inputs.shape
            
            # 策略：从 D, H, W 三个方向各随机采 n 个切片
            n_slices = 2 # 每个方向采4张，共12张，既省显存又能覆盖3D结构
            
            # Z轴切片 (Depth)
            idx_d = torch.randperm(D, device=inputs.device)[:n_slices]
            slice_d_in = inputs[:, :, idx_d, :, :].permute(0, 2, 1, 3, 4).reshape(-1, C, H, W)
            slice_d_rec = reconstructions[:, :, idx_d, :, :].permute(0, 2, 1, 3, 4).reshape(-1, C, H, W)

            # Y轴切片 (Height)
            idx_h = torch.randperm(H, device=inputs.device)[:n_slices]
            slice_h_in = inputs[:, :, :, idx_h, :].permute(0, 3, 1, 2, 4).reshape(-1, C, D, W)
            slice_h_rec = reconstructions[:, :, :, idx_h, :].permute(0, 3, 1, 2, 4).reshape(-1, C, D, W)

            # X轴切片 (Width)
            idx_w = torch.randperm(W, device=inputs.device)[:n_slices]
            slice_w_in = inputs[:, :, :, :, idx_w].permute(0, 4, 1, 2, 3).reshape(-1, C, D, H)
            slice_w_rec = reconstructions[:, :, :, :, idx_w].permute(0, 4, 1, 2, 3).reshape(-1, C, D, H)

            # 合并 Batch
            slice_in = torch.cat([slice_d_in, slice_h_in, slice_w_in], dim=0)
            slice_rec = torch.cat([slice_d_rec, slice_h_rec, slice_w_rec], dim=0)
            
            # 转 RGB (LPIPS 需要 3 通道)
            if C == 1:
                slice_in = slice_in.repeat(1, 3, 1, 1)
                slice_rec = slice_rec.repeat(1, 3, 1, 1)

            p_loss = self.perceptual_loss(slice_in, slice_rec).mean()
            nll_loss = rec_loss.mean() + self.perceptual_weight * p_loss
        else:
            p_loss = torch.tensor(0.0)
            nll_loss = rec_loss.mean()

        # 3. KL Loss
        mean, logvar = posteriors
        kl_loss = 0.5 * torch.sum(torch.exp(logvar) + mean**2 - 1. - logvar, dim=[1, 2, 3, 4])
        kl_loss = torch.mean(kl_loss)

        # ================= UPDATE GENERATOR (VAE) =================
        if optimizer_idx == 0:
            # GAN Generator Loss
            if discriminator is not None:
                logits_fake = discriminator(reconstructions)
                g_loss = -torch.mean(logits_fake) # Hinge Loss (Gen part)
            else:
                g_loss = torch.tensor(0.0)

            # --- Adaptive Weight Calculation ---
            try:
                # 只有在 discriminator 启动后才计算自适应权重
                if self.disc_weight > 0 and global_step >= self.disc_start:
                    d_weight = self.calculate_adaptive_weight(nll_loss, g_loss, last_layer=last_layer)
                else:
                    d_weight = torch.tensor(0.0)
            except RuntimeError as e:
                # 容错：如果计算梯度出错（例如未启用 anomaly detection），回退到默认
                print(f"Warning: Adaptive weight calc failed: {e}")
                d_weight = torch.tensor(0.0)

            # 组合 Total Loss
            # 注意：在 disc_start 之前，d_weight 应该是 0
            disc_factor = 1.0 if global_step >= self.disc_start else 0.0
            
            total_loss = nll_loss + \
                         self.kl_weight * kl_loss + \
                         d_weight * disc_factor * g_loss

            log = {
                f"{split}/total_loss": total_loss.detach().item(),
                f"{split}/rec_loss": rec_loss.mean().detach().item(),
                f"{split}/p_loss": p_loss.detach().item(),
                f"{split}/kl_loss": kl_loss.detach().item(),
                f"{split}/g_loss": g_loss.detach().item(),
                f"{split}/d_weight": d_weight.detach().item(),
            }
            return total_loss, log

        # ================= UPDATE DISCRIMINATOR =================
        if optimizer_idx == 1:
            # 如果还没到 disc_start，Discriminator 不更新，返回 0 loss
            if global_step < self.disc_start:
                return torch.tensor(0.0, device=inputs.device, requires_grad=True), {}

            logits_real = discriminator(inputs.detach())
            logits_fake = discriminator(reconstructions.detach())

            # Hinge Loss (Disc part)
            # real -> 1, fake -> -1
            d_loss = 0.5 * (torch.mean(torch.nn.functional.relu(1.0 - logits_real)) + 
                            torch.mean(torch.nn.functional.relu(1.0 + logits_fake)))

            log = {
                f"{split}/d_loss": d_loss.detach().item(),
                f"{split}/logits_real": logits_real.mean().detach().item(),
                f"{split}/logits_fake": logits_fake.mean().detach().item(),
            }
            return d_loss, log
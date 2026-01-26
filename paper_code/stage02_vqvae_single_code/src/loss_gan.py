import torch
import torch.nn as nn
import torch.nn.functional as F

class VQGANLoss(nn.Module):
    def __init__(self, disc_start, codebook_weight=1.0, disc_weight=0.5):
        super().__init__()
        self.disc_start = disc_start         # 多少步之后开始训练 GAN
        self.codebook_weight = codebook_weight
        self.disc_weight = disc_weight       # GAN Loss 的权重 (0.5 - 1.0 比较合适)

    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer):
        """
        (可选高级技巧) 动态调整 GAN 权重。
        为了代码稳定性，你可以先跳过这个，直接用固定的 disc_weight。
        如果想追求极致，可以去参考 Taming Transformers 的源码。
        这里我们简化处理，直接返回固定权重。
        """
        return self.disc_weight

    def forward(self, codebook_loss, inputs, reconstructions, optimizer_idx, global_step, 
                logits_real=None, logits_fake=None):
        """
        optimizer_idx: 0 表示 Generator, 1 表示 Discriminator
        """
        
        # --- 1. 计算重建损失 (L2 + L1 混合通常对 CT 效果最好) ---
        # L2 负责大结构，L1 负责清晰度
        rec_loss = torch.abs(inputs - reconstructions) + (inputs - reconstructions)**2
        rec_loss = torch.mean(rec_loss)

        # --- 2. 训练 Generator (VQVAE) ---
        if optimizer_idx == 0:
            # 如果还没到预热步数，GAN Loss = 0
            if global_step < self.disc_start:
                d_weight = 0.0
            else:
                d_weight = self.disc_weight

            # Generator 想要骗过 Discriminator (希望 logits_fake 接近 1)
            # Hinge Loss for G: -mean(logits_fake)
            if logits_fake is not None:
                g_loss = -torch.mean(logits_fake)
            else:
                g_loss = torch.tensor(0.0, device=inputs.device)

            # 总 Loss = 重建 + 码本 + GAN
            loss = rec_loss + self.codebook_weight * codebook_loss + d_weight * g_loss
            
            return loss, rec_loss, g_loss

        # --- 3. 训练 Discriminator ---
        if optimizer_idx == 1:
            # Hinge Loss for D: 
            # Real 必须大于 1, Fake 必须小于 -1
            # d_loss = ReLU(1 - real) + ReLU(1 + fake)
            
            d_loss_real = torch.mean(F.relu(1.0 - logits_real))
            d_loss_fake = torch.mean(F.relu(1.0 + logits_fake))
            
            d_loss = 0.5 * (d_loss_real + d_loss_fake)
            
            return d_loss
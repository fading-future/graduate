import torch
import torch.nn as nn
import math

class DiffusionTrainer:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.device = config["device"]
        self.timesteps = config["timesteps"]
        
        # === 🔴 改动 1: 使用 Cosine Schedule 替代 Linear ===
        # Linear 对 64x64 这种小尺寸 Latent 破坏力太强，导致结构学不会
        # Cosine 能保留更多结构信息到后面的时间步
        self.betas = self.get_cosine_schedule(self.timesteps).to(self.device)
        
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        
    def get_cosine_schedule(self, timesteps, s=0.008):
        """
        Cosine schedule as proposed in https://arxiv.org/abs/2102.09672
        """
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)
        
    def add_noise(self, x_start, t):
        """
        前向扩散过程: q(x_t | x_0)
        """
        sqrt_alphas_cumprod_t = torch.sqrt(self.alphas_cumprod[t])
        sqrt_one_minus_alphas_cumprod_t = torch.sqrt(1.0 - self.alphas_cumprod[t])
        
        sqrt_alphas_cumprod_t = sqrt_alphas_cumprod_t.view(-1, 1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod_t.view(-1, 1, 1, 1, 1)
        
        noise = torch.randn_like(x_start)
        x_noisy = sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
        
        return x_noisy, noise
    
    def train_step(self, batch, optimizer, criterion):
        x_0 = batch["GT"].to(self.device)
        condition = batch["Condition"].to(self.device)
        mask = batch["Mask"].to(self.device)
        porosity = batch["Porosity"].to(self.device)
        
        batch_size = x_0.shape[0]
        t = torch.randint(0, self.timesteps, (batch_size,), device=self.device).long()
        
        # Forward Diffusion
        x_noisy, noise = self.add_noise(x_0, t)
        
        # Input Concat
        model_input = torch.cat([x_noisy, condition, mask], dim=1) 
        
        # Predict
        noise_pred = self.model(model_input, t, porosity)
        
        # === 🔴 改动 2: Min-SNR Loss Weighting (针对 L1 Loss) ===
        # 这是一个 "结构拯救者" 策略。
        # 它会降低高频噪声(t小)的权重，强迫模型关注低频结构(t大)。
        
        # 1. 计算信噪比 SNR(t)
        # SNR = alpha_bar / (1 - alpha_bar)
        alpha_bar_t = self.alphas_cumprod[t].view(-1, 1, 1, 1, 1)
        snr = alpha_bar_t / (1.0 - alpha_bar_t)
        
        # 2. 计算权重
        # Min-SNR-Gamma strategy: weight = min(SNR, gamma) / SNR
        # 对于预测噪声(epsilon)的目标，gamma 通常设为 5.0
        gamma = 5.0
        loss_weight = torch.minimum(snr, torch.tensor(gamma, device=self.device)) / snr
        
        # 3. 计算加权 L1 Loss
        # criterion 应该是 nn.L1Loss(reduction='none') 才能加权
        # 如果你外面传进来的是 reduction='mean'，我们需要手动算
        raw_loss = torch.abs(noise_pred - noise) # L1 误差
        weighted_loss = raw_loss * loss_weight # 加权
        loss = weighted_loss.mean()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        return loss.item()

# import torch
# import torch.nn as nn

# class DiffusionTrainer:
#     def __init__(self, model, config):
#         self.model = model
#         self.config = config
#         self.device = config["device"]
#         self.timesteps = config["timesteps"]
        
#         # 定义 Beta Schedule (线性)
#         # 也可以换成 Cosine Schedule，通常效果更好，但线性最简单
#         self.betas = torch.linspace(1e-4, 0.02, self.timesteps).to(self.device)
#         self.alphas = 1.0 - self.betas
#         self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        
#     def add_noise(self, x_start, t):
#         """
#         前向扩散过程: q(x_t | x_0)
#         根据时间步 t，向 x_start 添加噪声
#         """
#         # 获取当前 t 对应的 cumulative alpha
#         sqrt_alphas_cumprod_t = torch.sqrt(self.alphas_cumprod[t])
#         sqrt_one_minus_alphas_cumprod_t = torch.sqrt(1.0 - self.alphas_cumprod[t])
        
#         # 调整维度以支持广播: (Batch, ) -> (Batch, 1, 1, 1, 1)
#         sqrt_alphas_cumprod_t = sqrt_alphas_cumprod_t.view(-1, 1, 1, 1, 1)
#         sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod_t.view(-1, 1, 1, 1, 1)
        
#         # 生成随机噪声 epsilon
#         noise = torch.randn_like(x_start)
        
#         # 公式: x_t = sqrt(alpha_bar) * x_0 + sqrt(1 - alpha_bar) * epsilon
#         x_noisy = sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
        
#         return x_noisy, noise
    
#     def train_step(self, batch, optimizer, criterion):
#         x_0 = batch["GT"].to(self.device)          # (B, 64, 64, 64, 64)
#         condition = batch["Condition"].to(self.device)
#         mask = batch["Mask"].to(self.device)       # (B, 1, 64, 64, 64)
#         porosity = batch["Porosity"].to(self.device) # (B, 1) <--- 新增
        
#         batch_size = x_0.shape[0]
#         t = torch.randint(0, self.timesteps, (batch_size,), device=self.device).long()
        
#         # Forward Diffusion
#         x_noisy, noise = self.add_noise(x_0, t)
        
#         # Input Concat (Scheme B)
#         # 64 + 64 + 1 = 129 Channels
#         model_input = torch.cat([x_noisy, condition, mask], dim=1) 
        
#         # Predict (传入 porosity)
#         noise_pred = self.model(model_input, t, porosity)
        
#         loss = criterion(noise_pred, noise)
        
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
        
#         return loss.item()
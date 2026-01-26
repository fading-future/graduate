import torch
import torch.nn as nn

class DiffusionTrainer:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.device = config["device"]
        self.timesteps = config["timesteps"]
        
        # 定义 Beta Schedule (线性)
        # 也可以换成 Cosine Schedule，通常效果更好，但线性最简单
        self.betas = torch.linspace(1e-4, 0.02, self.timesteps).to(self.device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        
    def add_noise(self, x_start, t):
        """
        前向扩散过程: q(x_t | x_0)
        根据时间步 t，向 x_start 添加噪声
        """
        # 获取当前 t 对应的 cumulative alpha
        sqrt_alphas_cumprod_t = torch.sqrt(self.alphas_cumprod[t])
        sqrt_one_minus_alphas_cumprod_t = torch.sqrt(1.0 - self.alphas_cumprod[t])
        
        # 调整维度以支持广播: (Batch, ) -> (Batch, 1, 1, 1, 1)
        sqrt_alphas_cumprod_t = sqrt_alphas_cumprod_t.view(-1, 1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod_t.view(-1, 1, 1, 1, 1)
        
        # 生成随机噪声 epsilon
        noise = torch.randn_like(x_start)
        
        # 公式: x_t = sqrt(alpha_bar) * x_0 + sqrt(1 - alpha_bar) * epsilon
        x_noisy = sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
        
        return x_noisy, noise
    
    def train_step(self, batch, optimizer, criterion):
        x_0 = batch["GT"].to(self.device)          # (B, 64, 64, 64, 64)
        condition = batch["Condition"].to(self.device)
        mask = batch["Mask"].to(self.device)       # (B, 1, 64, 64, 64)
        porosity = batch["Porosity"].to(self.device) # (B, 1) <--- 新增
        
        batch_size = x_0.shape[0]
        t = torch.randint(0, self.timesteps, (batch_size,), device=self.device).long()
        
        # Forward Diffusion
        x_noisy, noise = self.add_noise(x_0, t)
        
        # Input Concat (Scheme B)
        # 64 + 64 + 1 = 129 Channels
        model_input = torch.cat([x_noisy, condition, mask], dim=1) 
        
        # Predict (传入 porosity)
        noise_pred = self.model(model_input, t, porosity)
        
        loss = criterion(noise_pred, noise)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        return loss.item()
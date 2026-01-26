import torch
import torch.nn as nn
import torch.nn.functional as F
import random

class VectorQuantizerEMA(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25, decay=0.9, epsilon=1e-5):
        super(VectorQuantizerEMA, self).__init__()
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        self._commitment_cost = commitment_cost
        
        self.decay = decay
        self.epsilon = epsilon
        
        # === 新增：将阈值变为类属性，默认为 0.1 (开启重启) ===
        self.restart_threshold = 0.1 
        
        embedding = torch.randn(self._num_embeddings, self._embedding_dim)
        self.register_buffer('_embedding', embedding)
        self.register_buffer('_ema_cluster_size', torch.zeros(num_embeddings))
        self.register_buffer('_ema_w', embedding.clone())

    def forward(self, inputs):
        inputs = inputs.permute(0, 2, 3, 4, 1).contiguous()
        input_shape = inputs.shape
        flat_input = inputs.view(-1, self._embedding_dim)
        
        # 1. 计算距离
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self._embedding**2, dim=1)
                    - 2 * torch.matmul(flat_input, self._embedding.t()))
            
        # 2. Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        # 3. Quantize
        quantized = torch.matmul(encodings, self._embedding).view(input_shape)
        
        # --- EMA 更新 + 暴力重启 ---
        if self.training:
            encodings_sum = encodings.sum(0)
            dw = torch.matmul(encodings.t(), flat_input)
            
            self._ema_cluster_size.data.mul_(self.decay).add_(encodings_sum, alpha=1 - self.decay)
            self._ema_w.data.mul_(self.decay).add_(dw, alpha=1 - self.decay)
            
            n = self._ema_cluster_size.sum()
            cluster_size = (self._ema_cluster_size + self.epsilon) / (n + self._num_embeddings * self.epsilon) * n
            self._embedding.data.copy_(self._ema_w / cluster_size.unsqueeze(1))
            
            # === 暴力重启逻辑 (受 self.restart_threshold 控制) ===
            # 如果外部把 threshold 设为 -1，这里就会全部为 False，也就是关闭重启
            if self.restart_threshold > 0:
                usage = (self._ema_cluster_size >= self.restart_threshold).float()
                num_dead = (usage == 0).sum().item()
                
                if num_dead > 0:
                    if num_dead > self._num_embeddings * 0.1 and random.random() < 0.05:
                        print(f"Warning: {num_dead}/{self._num_embeddings} codes are inactive. Resurrecting...")

                    dead_indices = torch.where(usage == 0)[0]
                    if flat_input.shape[0] > len(dead_indices):
                        rand_idx = torch.randperm(flat_input.shape[0])[:len(dead_indices)].to(flat_input.device)
                        new_codes = flat_input[rand_idx].detach()
                        new_codes = F.normalize(new_codes, p=2, dim=-1)
                        avg_size = self._ema_cluster_size.mean().item()
                        self._embedding.data[dead_indices] = new_codes
                        self._ema_w.data[dead_indices] = new_codes * avg_size
                        self._ema_cluster_size.data[dead_indices] = avg_size
        
        # 4. Loss & Perplexity
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        loss = self._commitment_cost * e_latent_loss
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        return loss, quantized.permute(0, 4, 1, 2, 3).contiguous(), perplexity, encoding_indices
    

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.GroupNorm(8, out_channels) # GroupNorm 对 BatchSize=1 更友好
        self.act = nn.SiLU()
        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.GroupNorm(8, out_channels)
        
        if in_channels != out_channels:
            self.shortcut = nn.Conv3d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        h = self.conv1(x)
        h = self.bn1(h)
        h = self.act(h)
        h = self.conv2(h)
        h = self.bn2(h)
        return self.act(h + self.shortcut(x))

class VQVAE3D(nn.Module):
    def __init__(self, in_channels=1, embedding_dim=64, num_embeddings=1024):
        super(VQVAE3D, self).__init__()
        
        # --- Encoder (压缩 4倍: 128 -> 32) ---
        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels, 32, 4, stride=2, padding=1), # 64
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            nn.Conv3d(32, 64, 4, stride=2, padding=1), # 32
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            ResBlock(64, 64),
            ResBlock(64, 64),
            nn.Conv3d(64, embedding_dim, 3, padding=1)
        )
        
        # --- Vector Quantizer ---
        self.quantizer = VectorQuantizerEMA(num_embeddings, embedding_dim)
        
        # --- Decoder (还原 4倍: 32 -> 128) ---
        self.decoder = nn.Sequential(
            nn.Conv3d(embedding_dim, 64, 3, padding=1),
            ResBlock(64, 64),
            ResBlock(64, 64),
            nn.ConvTranspose3d(64, 32, 4, stride=2, padding=1), # 64
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            nn.ConvTranspose3d(32, 16, 4, stride=2, padding=1), # 128
            nn.GroupNorm(8, 16),
            nn.SiLU(),
            nn.Conv3d(16, in_channels, 3, padding=1),
            nn.Tanh() # 输出归一化到 [-1, 1]
        )

    def encode(self, x):
        z = self.encoder(x)
        loss, quantized, perplexity, _ = self.quantizer(z)
        return quantized, loss, perplexity

    def decode(self, quantized):
        return self.decoder(quantized)

    def forward(self, x):
        z = self.encoder(x)
        loss, quantized, perplexity, _ = self.quantizer(z)
        x_recon = self.decoder(quantized)
        return x_recon, loss, perplexity

import torch
import torch.nn as nn
import torch.nn.functional as F

# 修改后的 EMA VectorQuantizer
class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25, decay=0.99, epsilon=1e-5):
        super(VectorQuantizer, self).__init__()
        
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        
        # 1. 创建 Embedding
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        # [关键修改] 关闭梯度！告诉 DDP 不要监控它
        self._embedding.weight.requires_grad = False 
        
        self._commitment_cost = commitment_cost
        
        # 2. EMA 影子变量
        self.register_buffer('_ema_cluster_size', torch.zeros(num_embeddings))
        
        self._ema_w = nn.Parameter(torch.Tensor(num_embeddings, self._embedding_dim))
        # [关键修改] 关闭梯度！
        self._ema_w.requires_grad = False 
        
        # 初始化
        self._embedding.weight.data.normal_()
        self._ema_w.data.normal_()
        
        self._decay = decay
        self._epsilon = epsilon

    def forward(self, inputs):
        # inputs: [B, C, D, H, W] -> [B, D, H, W, C]
        inputs = inputs.permute(0, 2, 3, 4, 1).contiguous()
        input_shape = inputs.shape
        flat_input = inputs.view(-1, self._embedding_dim)
        
        # 计算距离
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self._embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self._embedding.weight.t()))
            
        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        # --- EMA Update ---
        if self.training:
            # 统计
            self._ema_cluster_size = self._ema_cluster_size * self._decay + \
                                     (1 - self._decay) * torch.sum(encodings, 0)
            
            # 平滑
            n = torch.sum(self._ema_cluster_size.data)
            self._ema_cluster_size = (
                (self._ema_cluster_size + self._epsilon)
                / (n + self._num_embeddings * self._epsilon) * n)
            
            # 更新影子权重
            dw = torch.matmul(encodings.t(), flat_input)
            self._ema_w.data = self._ema_w.data * self._decay + (1 - self._decay) * dw
            
            # 赋值给真正的 Codebook
            self._embedding.weight.data.copy_(self._ema_w / self._ema_cluster_size.unsqueeze(1))
        
        # Quantize
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)
        
        # Loss (EMA 模式下只有 Commitment Loss)
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        loss = self._commitment_cost * e_latent_loss
        
        # Straight Through Estimator
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
        
        # --- Encoder (压缩 4倍: 256 -> 64) ---
        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels, 32, 4, stride=2, padding=1), # 128
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            nn.Conv3d(32, 64, 4, stride=2, padding=1), # 64
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            ResBlock(64, 64),
            ResBlock(64, 64),
            nn.Conv3d(64, embedding_dim, 3, padding=1),
            nn.GroupNorm(1, embedding_dim)
        )
        
        # --- Vector Quantizer ---
        self.quantizer = VectorQuantizer(num_embeddings, embedding_dim)
        
        # --- Decoder (还原 4倍: 64 -> 256) ---
        self.decoder = nn.Sequential(
            nn.Conv3d(embedding_dim, 64, 3, padding=1),
            ResBlock(64, 64),
            ResBlock(64, 64),
            nn.ConvTranspose3d(64, 32, 4, stride=2, padding=1), # 128
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            nn.ConvTranspose3d(32, 16, 4, stride=2, padding=1), # 256
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

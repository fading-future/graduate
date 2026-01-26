import torch
import torch.nn as nn
import torch.nn.functional as F

class VectorQuantizerEMA(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25, decay=0.99, epsilon=1e-5):
        super(VectorQuantizerEMA, self).__init__()
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        self._commitment_cost = commitment_cost
        
        # EMA 参数
        self.decay = decay
        self.epsilon = epsilon
        
        # 初始化 Embedding (不再作为 Parameter，因为我们手动更新)
        embedding = torch.randn(self._num_embeddings, self._embedding_dim)
        self.register_buffer('_embedding', embedding)
        
        # 记录每个 Code 被选中的次数 (Cluster Size)
        self.register_buffer('_ema_cluster_size', torch.zeros(num_embeddings))
        # 记录每个 Code 的 EMA 权重 (Cluster Sum)
        self.register_buffer('_ema_w', embedding.clone())

    def forward(self, inputs):
        # inputs: [B, C, D, H, W] -> [B, D, H, W, C]
        inputs = inputs.permute(0, 2, 3, 4, 1).contiguous()
        input_shape = inputs.shape
        flat_input = inputs.view(-1, self._embedding_dim)
        
        # 1. 计算距离
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self._embedding**2, dim=1)
                    - 2 * torch.matmul(flat_input, self._embedding.t()))
            
        # 2. Encoding (选择最近的 Code)
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        # 3. Quantize
        quantized = torch.matmul(encodings, self._embedding).view(input_shape)
        
        # --- EMA 更新逻辑 (训练时执行) ---
        if self.training:
            # 计算当前 batch 每个 code 被选中的次数
            encodings_sum = encodings.sum(0)
            # 计算当前 batch 分配给每个 code 的输入向量之和
            dw = torch.matmul(encodings.t(), flat_input)
            
            # EMA 更新 cluster size (加上平滑项 epsilon 防止除0)
            self._ema_cluster_size.data.mul_(self.decay).add_(encodings_sum, alpha=1 - self.decay)
            
            # EMA 更新 cluster sum
            self._ema_w.data.mul_(self.decay).add_(dw, alpha=1 - self.decay)
            
            # 归一化得到新的 embedding
            n = self._ema_cluster_size.sum()
            cluster_size = (self._ema_cluster_size + self.epsilon) / (n + self._num_embeddings * self.epsilon) * n
            
            self._embedding.data.copy_(self._ema_w / cluster_size.unsqueeze(1))
        
        # 4. Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        loss = self._commitment_cost * e_latent_loss
        
        # Straight Through Estimator
        quantized = inputs + (quantized - inputs).detach()
        
        # 计算 Perplexity
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

import torch
import torch.nn as nn
import torch.nn.functional as F

class VectorQuantizer(nn.Module):
    """
    [回归初心版] 
    1. 改回 Uniform 初始化：这是你之前单卡跑出漂亮曲线的关键！
    2. 保留 FP32 计算：防止 NaN。
    """
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25):
        super(VectorQuantizer, self).__init__()
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        self._commitment_cost = commitment_cost

        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        
        # --- [核心改动] 回归 Uniform 初始化 ---
        # 这种初始化让向量很小，容易被“激活”，解决 Perplexity=1 的问题
        limit = 1 / self._num_embeddings
        self._embedding.weight.data.uniform_(-limit, limit)

    def forward(self, inputs):
        # inputs: [B, C, D, H, W] -> [B, D, H, W, C]
        inputs = inputs.permute(0, 2, 3, 4, 1).contiguous()
        input_shape = inputs.shape
        
        flat_input = inputs.view(-1, self._embedding_dim)
        
        # --- [保留防爆] 强制 FP32 计算距离 ---
        flat_input_float = flat_input.float()
        embedding_weight_float = self._embedding.weight.float()
        
        distances = (torch.sum(flat_input_float**2, dim=1, keepdim=True) 
                    + torch.sum(embedding_weight_float**2, dim=1)
                    - 2 * torch.matmul(flat_input_float, embedding_weight_float.t()))
            
        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        # 类型对齐
        encodings = encodings.to(inputs.dtype) 
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)
        
        # Loss
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        loss = q_latent_loss + self._commitment_cost * e_latent_loss
        
        quantized = inputs + (quantized - inputs).detach()
        
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        return loss, quantized.permute(0, 4, 1, 2, 3).contiguous(), perplexity, encoding_indices

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.GroupNorm(8, out_channels) 
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
        
        # --- Encoder ---
        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels, 32, 4, stride=2, padding=1),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            nn.Conv3d(32, 64, 4, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            ResBlock(64, 64),
            ResBlock(64, 64),
            nn.Conv3d(64, embedding_dim, 3, padding=1),
            # [关键] 强制 Norm 到 std=1
            # nn.GroupNorm(32, embedding_dim, eps=1e-6, affine=False)
        )
        
        self.quantizer = VectorQuantizer(num_embeddings, embedding_dim)
        
        # --- Decoder ---
        self.decoder = nn.Sequential(
            nn.Conv3d(embedding_dim, 64, 3, padding=1),
            ResBlock(64, 64),
            ResBlock(64, 64),
            nn.ConvTranspose3d(64, 32, 4, stride=2, padding=1),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            nn.ConvTranspose3d(32, 16, 4, stride=2, padding=1),
            nn.GroupNorm(8, 16),
            nn.SiLU(),
            nn.Conv3d(16, in_channels, 3, padding=1),
            nn.Tanh()
        )

    def encode(self, x):
        z = self.encoder(x)
        loss, quantized, perplexity, _ = self.quantizer(z)
        return quantized, loss, perplexity

    def decode(self, quantized):
        return self.decoder(quantized)

    def forward(self, x):
        # [防爆核心 3] 输入数据清洗：防止数据集中有 NaN 导致的连锁反应
        if torch.isnan(x).any():
            print("Warning: Input data contains NaN! Replacing with 0.")
            x = torch.nan_to_num(x, nan=0.0)
            
        z = self.encoder(x)
        loss, quantized, perplexity, _ = self.quantizer(z)
        x_recon = self.decoder(quantized)
        return x_recon, loss, perplexity
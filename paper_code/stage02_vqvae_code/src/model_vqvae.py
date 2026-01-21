import torch
import torch.nn as nn
import torch.nn.functional as F

class VectorQuantizer(nn.Module):
    """
    它是一个**“翻译官”**。它把 Encoder 产生的模糊、连续的特征，强制“翻译”成 Codebook 里固定的几个单词（向量），
    并且利用 STE 技术骗过神经网络，让梯度能流过去。
    码本：码本本质上就是一个可学习的张量 (Tensor)。(1024, 64)
        1024: num_embeddings (词表大小，即有多少种不同的“特征单词”)
        64: embedding_dim (每个单词用一个 64 维的向量表示)
    """
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25):
        super(VectorQuantizer, self).__init__()
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        self._commitment_cost = commitment_cost

        # self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        # self._embedding.weight.data.uniform_(-1/self._num_embeddings, 1/self._num_embeddings)
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        # 使用正态分布初始化，std=1.0 或更小一点（如 0.5）
        self._embedding.weight.data.normal_(mean=0, std=1.0) 

    def forward(self, inputs):
        # inputs: [B, C, D, H, W] -> [B, D, H, W, C]
        inputs = inputs.permute(0, 2, 3, 4, 1).contiguous()
        input_shape = inputs.shape
        
        flat_input = inputs.view(-1, self._embedding_dim)
        
        # 计算距离
        # 这是一个数学技巧：(a-b)^2 = a^2 + b^2 - 2ab
        # 目的：计算每个输入向量和 Codebook 中 1024 个向量的欧氏距离    
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self._embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self._embedding.weight.t()))
            
        #Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        # Quantize and unflatten
        # 把原始的连续向量，直接替换成字典里查到的那个标准向量
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)
        
        # Loss
        # Commitment Loss (e_latent_loss): 让 Encoder 出来的输入数据不要乱跑，尽量靠近字典里的向量（约束 Encoder）。
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        # Codebook Loss (q_latent_loss): 让字典里的向量主动去靠近输入数据（更新字典）
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self._commitment_cost * e_latent_loss
        
        quantized = inputs + (quantized - inputs).detach() # Straight-through estimator
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        # convert quantized from BHWC -> BCHW
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

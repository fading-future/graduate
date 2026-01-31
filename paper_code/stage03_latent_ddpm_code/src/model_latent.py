import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.checkpoint import checkpoint

# ==========================================
# 1. 基础组件 (Embedding & Helpers)
# ==========================================

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class PorosityEmbedder(nn.Module):
    def __init__(self, frequency_embedding_size=256):
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.linear_proj = nn.Linear(frequency_embedding_size, frequency_embedding_size)

    def forward(self, porosity):
        device = porosity.device
        half_dim = self.frequency_embedding_size // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = porosity * emb[None, :] 
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return self.linear_proj(emb)

# ==========================================
# 2. 核心模块: ResBlock3D
# ==========================================

class ResBlock3D(nn.Module):
    """
    标准的 ResNet Block，支持 Time/Condition 注入
    结构: GN -> SiLU -> Conv -> GN -> SiLU -> Conv + (TimeEmb -> MLP)
    """
    def __init__(self, in_ch, out_ch, time_emb_dim, dropout=0.0):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, in_ch)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, padding=1)
        
        # 时间/条件 投影层
        self.time_emb_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_ch)
        )

        self.norm2 = nn.GroupNorm(32, out_ch)
        self.act2 = nn.SiLU()
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, padding=1)
        
        self.dropout = nn.Dropout(dropout)
        
        # 残差连接
        if in_ch != out_ch:
            self.shortcut = nn.Conv3d(in_ch, out_ch, 1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x, t_emb):
        h = self.norm1(x)
        h = self.act1(h)
        h = self.conv1(h)

        # 注入时间步信息 (Broadcasting)
        # t_emb: (B, time_dim) -> proj -> (B, out_ch) -> (B, out_ch, 1, 1, 1)
        time_hidden = self.time_emb_proj(t_emb)
        h = h + time_hidden[:, :, None, None, None]

        h = self.norm2(h)
        h = self.act2(h)
        h = self.dropout(h)
        h = self.conv2(h)

        return h + self.shortcut(x)

# ==========================================
# 3. 核心模块: Attention3D (关键升级)
# ==========================================

class Attention3D(nn.Module):
    def __init__(self, dim, heads=4, dim_head=64):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.norm = nn.GroupNorm(32, dim)
        self.to_qkv = nn.Conv3d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv3d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, d, h, w = x.shape
        x_in = x
        
        x = self.norm(x)
        
        # (B, 3*dim, D, H, W) -> (B, 3*dim, N)
        qkv = self.to_qkv(x).view(b, self.heads * 3, -1, d*h*w)
        
        # 分离 Q, K, V
        # 现在的形状: (B, 3*Heads, Dim_Head, N_pixels)
        # 我们需要调整为 Flash Attention 接受的形状: (B, Heads, N_pixels, Dim_Head)
        q, k, v = map(lambda t: t.permute(0, 1, 3, 2).contiguous(), qkv.chunk(3, dim=1))
        
        # --- 🚀【核心修改】使用 Flash Attention ---
        # 自动选择最省显存的算法，不再显式创建 (N, N) 矩阵
        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, 
            dropout_p=0.0, 
            is_causal=False
        )
        
        # 还原形状: (B, Heads, N, Dim_Head) -> (B, Heads*Dim_Head, N)
        out = out.permute(0, 1, 3, 2).reshape(b, -1, d, h, w)
        
        return self.to_out(out) + x_in

# ==========================================
# 4. 主模型: High-Fidelity 3D UNet
# ==========================================

class ConditionalLatentUNet(nn.Module):
    def __init__(
        self, 
        in_channels=129, 
        out_channels=64, 
        base_channels=128, # A100 推荐 128
        channel_mults=(1, 2, 4), # 通道倍率: 128 -> 256 -> 512
        use_attention=(False, True, True) # 仅在最底层使用 Attention
    ):
        super().__init__()
        
        self.base_channels = base_channels
        time_dim = base_channels * 4

        # 1. Embeddings
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(base_channels),
            nn.Linear(base_channels, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )
        
        self.porosity_mlp = nn.Sequential(
            PorosityEmbedder(base_channels),
            nn.Linear(base_channels, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        # 2. Encoder (Downsampling)
        self.inc = nn.Conv3d(in_channels, base_channels, 3, padding=1)
        
        self.downs = nn.ModuleList([])
        dims = [base_channels, *map(lambda m: base_channels * m, channel_mults)]
        # dims = [128, 128, 256, 512]
        
        in_out = list(zip(dims[:-1], dims[1:]))
        
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            use_attn = use_attention[ind]
            
            self.downs.append(nn.ModuleList([
                ResBlock3D(dim_in, dim_out, time_dim),
                ResBlock3D(dim_out, dim_out, time_dim),
                Attention3D(dim_out) if use_attn else nn.Identity(),
                nn.Conv3d(dim_out, dim_out, 4, stride=2, padding=1) if not is_last else nn.Identity()
            ]))

        # 3. Middle (Bottleneck with Attention)
        mid_dim = dims[-1]
        self.mid_block1 = ResBlock3D(mid_dim, mid_dim, time_dim)
        self.mid_attn = Attention3D(mid_dim) # 核心注意力层
        self.mid_block2 = ResBlock3D(mid_dim, mid_dim, time_dim)

        # 4. Decoder (Upsampling)
        self.ups = nn.ModuleList([])
        reversed_dims = list(reversed(in_out))
        # reversed_dims = [(256, 512), (128, 256), (128, 128)]
        
        for ind, (dim_out, dim_in) in enumerate(reversed_dims):
            is_last = ind >= (len(reversed_dims) - 1)
            use_attn = use_attention[len(use_attention) - 1 - ind]
            
            # 输入通道 = 原通道 + Skip Connection 通道
            actual_in_dim = dim_in + dim_in 
            
            self.ups.append(nn.ModuleList([
                ResBlock3D(actual_in_dim, dim_out, time_dim),
                ResBlock3D(dim_out, dim_out, time_dim),
                Attention3D(dim_out) if use_attn else nn.Identity(),
                nn.ConvTranspose3d(dim_out, dim_out, 2, stride=2) if not is_last else nn.Identity()
            ]))

        # 5. Output
        self.final_res_block = ResBlock3D(base_channels * channel_mults[0], base_channels, time_dim)
        self.outc = nn.Conv3d(base_channels, out_channels, 1)

    def forward(self, x, t, porosity):
        # 计算 Embeddings
        t_emb = self.time_mlp(t)
        p_emb = self.porosity_mlp(porosity)
        emb = t_emb + p_emb # 融合条件

        x = self.inc(x)
        
        # Encoder
        h = []
        for block1, block2, attn, downsample in self.downs:
            x = block1(x, emb)
            x = block2(x, emb)
            x = attn(x)
            h.append(x) # 存储 Skip Connection
            x = downsample(x)

        # Middle
        x = self.mid_block1(x, emb)
        x = self.mid_attn(x)
        x = self.mid_block2(x, emb)

        # Decoder
        for block1, block2, attn, upsample in self.ups:
            # Skip Connection Concatenation
            skip = h.pop()
            
            # 这里的 Cat 需要保证尺寸一致，虽然UNet一般尺寸是对齐的，但加上检查更安全
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode='nearest')
            
            x = torch.cat((x, skip), dim=1)
            
            x = block1(x, emb)
            x = block2(x, emb)
            x = attn(x)
            x = upsample(x)

        x = self.final_res_block(x, emb)
        return self.outc(x)
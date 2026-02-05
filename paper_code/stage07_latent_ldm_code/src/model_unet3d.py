import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ==========================================
# Embeddings
# ==========================================
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = time[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


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
# Core Blocks
# ==========================================
class ResBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, dropout=0.0):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, in_ch)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, padding=1)

        self.time_emb_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_ch)
        )

        self.norm2 = nn.GroupNorm(32, out_ch)
        self.act2 = nn.SiLU()
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, padding=1)
        self.dropout = nn.Dropout(dropout)

        self.shortcut = nn.Conv3d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        h = self.norm1(x)
        h = self.act1(h)
        h = self.conv1(h)

        time_hidden = self.time_emb_proj(t_emb)
        h = h + time_hidden[:, :, None, None, None]

        h = self.norm2(h)
        h = self.act2(h)
        h = self.dropout(h)
        h = self.conv2(h)

        return h + self.shortcut(x)


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

        qkv = self.to_qkv(x).view(b, self.heads * 3, -1, d * h * w)
        q, k, v = map(lambda t: t.permute(0, 1, 3, 2).contiguous(), qkv.chunk(3, dim=1))

        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, dropout_p=0.0, is_causal=False
        )

        out = out.permute(0, 1, 3, 2).reshape(b, -1, d, h, w)
        return self.to_out(out) + x_in


# ==========================================
# UNet
# ==========================================
class ConditionalLatentUNet(nn.Module):
    def __init__(
        self,
        in_channels=9,
        out_channels=4,
        base_channels=128,
        channel_mults=(1, 2, 4),
        use_attention=(False, True, True),
    ):
        super().__init__()

        time_dim = base_channels * 4

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

        self.inc = nn.Conv3d(in_channels, base_channels, 3, padding=1)

        self.downs = nn.ModuleList([])
        dims = [base_channels, *map(lambda m: base_channels * m, channel_mults)]
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

        mid_dim = dims[-1]
        self.mid_block1 = ResBlock3D(mid_dim, mid_dim, time_dim)
        self.mid_attn = Attention3D(mid_dim)
        self.mid_block2 = ResBlock3D(mid_dim, mid_dim, time_dim)

        self.ups = nn.ModuleList([])
        reversed_dims = list(reversed(in_out))

        for ind, (dim_out, dim_in) in enumerate(reversed_dims):
            is_last = ind >= (len(reversed_dims) - 1)
            use_attn = use_attention[len(use_attention) - 1 - ind]
            actual_in_dim = dim_in + dim_in
            self.ups.append(nn.ModuleList([
                ResBlock3D(actual_in_dim, dim_out, time_dim),
                ResBlock3D(dim_out, dim_out, time_dim),
                Attention3D(dim_out) if use_attn else nn.Identity(),
                nn.ConvTranspose3d(dim_out, dim_out, 2, stride=2) if not is_last else nn.Identity()
            ]))

        self.final_res_block = ResBlock3D(base_channels * channel_mults[0], base_channels, time_dim)
        self.outc = nn.Conv3d(base_channels, out_channels, 1)

    def forward(self, x, t, porosity):
        t_emb = self.time_mlp(t)
        p_emb = self.porosity_mlp(porosity)
        emb = t_emb + p_emb

        x = self.inc(x)

        h = []
        for block1, block2, attn, downsample in self.downs:
            x = block1(x, emb)
            x = block2(x, emb)
            x = attn(x)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, emb)
        x = self.mid_attn(x)
        x = self.mid_block2(x, emb)

        for block1, block2, attn, upsample in self.ups:
            skip = h.pop()
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode="nearest")
            x = torch.cat((x, skip), dim=1)
            x = block1(x, emb)
            x = block2(x, emb)
            x = attn(x)
            x = upsample(x)

        x = self.final_res_block(x, emb)
        return self.outc(x)

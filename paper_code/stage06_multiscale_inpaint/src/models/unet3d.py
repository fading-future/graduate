import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        # x: (B,1) or (B,)
        if x.dim() == 2:
            x = x.view(-1)
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class ResBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch, emb_dim, dropout=0.0):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, padding=1)

        self.emb_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, out_ch),
        )

        self.norm2 = nn.GroupNorm(8, out_ch)
        self.act2 = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, padding=1)

        self.shortcut = nn.Conv3d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, emb):
        h = self.conv1(self.act1(self.norm1(x)))
        e = self.emb_proj(emb).view(emb.shape[0], -1, 1, 1, 1)
        h = h + e
        h = self.conv2(self.dropout(self.act2(self.norm2(h))))
        return h + self.shortcut(x)


class Attention3D(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.heads = heads
        hidden = heads * dim_head
        self.norm = nn.GroupNorm(8, dim)
        self.to_qkv = nn.Conv3d(dim, hidden * 3, 1, bias=False)
        self.to_out = nn.Conv3d(hidden, dim, 1)

    def forward(self, x):
        b, c, d, h, w = x.shape
        x_in = x
        x = self.norm(x)
        qkv = self.to_qkv(x).view(b, self.heads * 3, -1, d * h * w)
        q, k, v = map(lambda t: t.permute(0, 1, 3, 2).contiguous(), qkv.chunk(3, dim=1))
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
        out = out.permute(0, 1, 3, 2).reshape(b, -1, d, h, w)
        return self.to_out(out) + x_in


class UNet3D(nn.Module):
    def __init__(
        self,
        in_channels=2,
        out_channels=1,
        base_channels=64,
        channel_mults=(1, 2, 4),
        use_attention=(False, True, True),
        emb_dim=256,
    ):
        super().__init__()
        self.time_emb = nn.Sequential(
            SinusoidalPositionEmbeddings(emb_dim),
            nn.Linear(emb_dim, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim),
        )

        self.inc = nn.Conv3d(in_channels, base_channels, 3, padding=1)

        dims = [base_channels, *[base_channels * m for m in channel_mults]]
        in_out = list(zip(dims[:-1], dims[1:]))

        self.downs = nn.ModuleList([])
        for i, (din, dout) in enumerate(in_out):
            is_last = i == len(in_out) - 1
            attn = use_attention[i]
            self.downs.append(nn.ModuleList([
                ResBlock3D(din, dout, emb_dim),
                ResBlock3D(dout, dout, emb_dim),
                Attention3D(dout) if attn else nn.Identity(),
                nn.Conv3d(dout, dout, 4, stride=2, padding=1) if not is_last else nn.Identity()
            ]))

        mid_dim = dims[-1]
        self.mid1 = ResBlock3D(mid_dim, mid_dim, emb_dim)
        self.mid_attn = Attention3D(mid_dim)
        self.mid2 = ResBlock3D(mid_dim, mid_dim, emb_dim)

        self.ups = nn.ModuleList([])
        rev = list(reversed(in_out))
        for i, (dout, din) in enumerate(rev):
            is_last = i == len(rev) - 1
            attn = use_attention[len(use_attention) - 1 - i]
            in_dim = din + din
            self.ups.append(nn.ModuleList([
                ResBlock3D(in_dim, dout, emb_dim),
                ResBlock3D(dout, dout, emb_dim),
                Attention3D(dout) if attn else nn.Identity(),
                nn.ConvTranspose3d(dout, dout, 2, stride=2) if not is_last else nn.Identity()
            ]))

        self.out = nn.Sequential(
            nn.GroupNorm(8, base_channels),
            nn.SiLU(),
            nn.Conv3d(base_channels, out_channels, 1)
        )

    def forward(self, x, emb_scalar):
        # emb_scalar: porosity (B,1) or (B,)
        emb = self.time_emb(emb_scalar)

        x = self.inc(x)
        skips = []
        for b1, b2, attn, down in self.downs:
            x = b1(x, emb)
            x = b2(x, emb)
            x = attn(x)
            skips.append(x)
            x = down(x)

        x = self.mid1(x, emb)
        x = self.mid_attn(x)
        x = self.mid2(x, emb)

        for b1, b2, attn, up in self.ups:
            skip = skips.pop()
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode="nearest")
            x = torch.cat([x, skip], dim=1)
            x = b1(x, emb)
            x = b2(x, emb)
            x = attn(x)
            x = up(x)

        return self.out(x)

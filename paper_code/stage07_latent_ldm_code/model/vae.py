import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def _gn(channels: int, max_groups: int = 32) -> nn.GroupNorm:
    """Choose a GroupNorm group count that divides channels."""
    g = min(max_groups, channels)
    while channels % g != 0 and g > 1:
        g -= 1
    return nn.GroupNorm(g, channels)


class ResBlock3D(nn.Module):
    """
    Standard pre-activation ResNet block used in (AutoencoderKL-style) VAEs.

    - supports channel change (in_ch -> out_ch)
    - optional dropout
    """
    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch

        self.norm1 = _gn(in_ch)
        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, padding=1)

        self.norm2 = _gn(out_ch)
        self.dropout = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, padding=1)

        if in_ch != out_ch:
            self.skip = nn.Conv3d(in_ch, out_ch, 1)
        else:
            self.skip = nn.Identity()

    def forward(self, x):
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return self.skip(x) + h


class AttnBlock3D(nn.Module):
    """
    Lightweight full self-attention on (D*H*W) tokens.
    Only use at low resolutions (e.g., 8^3 or 4^3) to keep it cheap.
    """
    def __init__(self, channels: int):
        super().__init__()
        self.norm = _gn(channels)
        self.q = nn.Conv3d(channels, channels, 1)
        self.k = nn.Conv3d(channels, channels, 1)
        self.v = nn.Conv3d(channels, channels, 1)
        self.proj = nn.Conv3d(channels, channels, 1)

    def forward(self, x):
        b, c, d, h, w = x.shape
        x_in = x
        x = self.norm(x)

        q = self.q(x).reshape(b, c, d * h * w).permute(0, 2, 1)  # (b, n, c)
        k = self.k(x).reshape(b, c, d * h * w)                   # (b, c, n)
        v = self.v(x).reshape(b, c, d * h * w).permute(0, 2, 1)  # (b, n, c)

        # (b, n, n)
        attn = torch.bmm(q, k) * (c ** -0.5)
        attn = torch.softmax(attn, dim=-1)

        out = torch.bmm(attn, v).permute(0, 2, 1).reshape(b, c, d, h, w)
        out = self.proj(out)
        return x_in + out


class Downsample3D(nn.Module):
    """Strided conv downsample (factor 2)."""
    def __init__(self, channels: int):
        super().__init__()
        self.op = nn.Conv3d(channels, channels, 4, stride=2, padding=1)

    def forward(self, x):
        return self.op(x)


class Upsample3D(nn.Module):
    """Transposed conv upsample (factor 2)."""
    def __init__(self, channels: int):
        super().__init__()
        self.op = nn.ConvTranspose3d(channels, channels, 4, stride=2, padding=1)

    def forward(self, x):
        return self.op(x)


class Encoder(nn.Module):
    """
    Stronger hierarchical encoder:
    - multiple ResBlocks per level
    - channel transitions inside ResBlocks (no extra "adapt conv" layers)
    - optional attention at the bottleneck (cheap and helpful for global structure)
    """
    def __init__(
        self,
        in_channels: int,
        base_channels: int,
        ch_mult: list,
        num_res_blocks: int,
        z_channels: int,
        dropout: float = 0.0,
        use_attn: bool = True,
    ):
        super().__init__()

        self.conv_in = nn.Conv3d(in_channels, base_channels, 3, padding=1)

        # build levels
        self.down = nn.ModuleList()
        in_ch = base_channels
        for i, mult in enumerate(ch_mult):
            out_ch = base_channels * mult
            level = nn.ModuleList()
            # first block can change channels
            level.append(ResBlock3D(in_ch, out_ch, dropout=dropout))
            for _ in range(num_res_blocks - 1):
                level.append(ResBlock3D(out_ch, out_ch, dropout=dropout))
            self.down.append(level)
            in_ch = out_ch
            if i != len(ch_mult) - 1:
                self.down.append(Downsample3D(in_ch))

        # middle (bottleneck)
        self.mid_block1 = ResBlock3D(in_ch, in_ch, dropout=dropout)
        self.mid_attn = AttnBlock3D(in_ch) if use_attn else nn.Identity()
        self.mid_block2 = ResBlock3D(in_ch, in_ch, dropout=dropout)

        self.norm_out = _gn(in_ch)
        self.conv_out = nn.Conv3d(in_ch, 2 * z_channels, 3, padding=1)  # mean & logvar

    def forward(self, x):
        h = self.conv_in(x)
        for m in self.down:
            if isinstance(m, nn.ModuleList):
                for blk in m:
                    h = blk(h)
            else:
                h = m(h)

        h = self.mid_block1(h)
        h = self.mid_attn(h)
        h = self.mid_block2(h)

        h = self.conv_out(F.silu(self.norm_out(h)))
        return h


class Decoder(nn.Module):
    """
    Stronger hierarchical decoder:
    - multiple ResBlocks per level
    - attention at bottleneck (mirrors encoder)
    """
    def __init__(
        self,
        out_channels: int,
        base_channels: int,
        ch_mult: list,
        num_res_blocks: int,
        z_channels: int,
        dropout: float = 0.0,
        use_attn: bool = True,
    ):
        super().__init__()
        # reverse channel multipliers for up path
        ch_mult_rev = list(ch_mult)[::-1]

        in_ch = base_channels * ch_mult_rev[0]
        self.conv_in = nn.Conv3d(z_channels, in_ch, 3, padding=1)

        # middle
        self.mid_block1 = ResBlock3D(in_ch, in_ch, dropout=dropout)
        self.mid_attn = AttnBlock3D(in_ch) if use_attn else nn.Identity()
        self.mid_block2 = ResBlock3D(in_ch, in_ch, dropout=dropout)

        # up levels
        self.up = nn.ModuleList()
        for i, mult in enumerate(ch_mult_rev):
            out_ch = base_channels * mult
            level = nn.ModuleList()
            # keep channels within level (already in_ch == out_ch at i=0)
            level.append(ResBlock3D(in_ch, out_ch, dropout=dropout))
            for _ in range(num_res_blocks - 1):
                level.append(ResBlock3D(out_ch, out_ch, dropout=dropout))
            self.up.append(level)
            in_ch = out_ch
            if i != len(ch_mult_rev) - 1:
                self.up.append(Upsample3D(in_ch))

        self.norm_out = _gn(in_ch)
        self.conv_out = nn.Conv3d(in_ch, out_channels, 3, padding=1)  # logits (no tanh)

    def forward(self, z):
        h = self.conv_in(z)

        h = self.mid_block1(h)
        h = self.mid_attn(h)
        h = self.mid_block2(h)

        for m in self.up:
            if isinstance(m, nn.ModuleList):
                for blk in m:
                    h = blk(h)
            else:
                h = m(h)

        h = self.conv_out(F.silu(self.norm_out(h)))
        return h


class DiagonalGaussianDistribution(object):
    def __init__(self, parameters, deterministic=False, logvar_max=5.0, logvar_min=-30.0):
        """
        parameters: tensor (B, 2*z_ch, D, H, W) -> chunk into mean, logvar
        logvar_max: upper clamp for logvar (exp(5) ~ 148 -> reasonable)
        """
        self.parameters = parameters
        self.mean, raw_logvar = torch.chunk(parameters, 2, dim=1)

        # clamp logvar to safe numerical range BEFORE exp
        self.logvar = torch.clamp(raw_logvar, min=logvar_min, max=logvar_max)

        self.deterministic = deterministic

        # compute var/std AFTER clamping -> avoid overflow
        self.var = torch.exp(self.logvar)
        self.std = torch.sqrt(self.var)

    def sample(self):
        if self.deterministic:
            return self.mean
        return self.mean + self.std * torch.randn_like(self.mean)

    def kl(self, other=None):
        # per-sample KL (shape: [B])
        if self.deterministic:
            return torch.zeros(self.mean.shape[0], device=self.mean.device, dtype=self.mean.dtype)
        return 0.5 * torch.sum(self.mean * self.mean + self.var - 1.0 - self.logvar, dim=[1, 2, 3, 4])


class KLVAE3D(nn.Module):
    def __init__(self, config):
        super().__init__()
        # keep backward compatibility: if dropout/use_attn not in config, use defaults
        dropout = float(config.get('model', {}).get('dropout', 0.0))
        use_attn = bool(config.get('model', {}).get('use_attn', True))

        self.encoder = Encoder(
            in_channels=config['data']['channels'],
            base_channels=config['model']['base_channels'],
            ch_mult=config['model']['ch_mult'],
            num_res_blocks=config['model']['num_res_blocks'],
            z_channels=config['model']['z_channels'],
            dropout=dropout,
            use_attn=use_attn,
        )
        self.decoder = Decoder(
            out_channels=config['data']['channels'],
            base_channels=config['model']['base_channels'],
            ch_mult=config['model']['ch_mult'],
            num_res_blocks=config['model']['num_res_blocks'],
            z_channels=config['model']['z_channels'],
            dropout=dropout,
            use_attn=use_attn,
        )

    def encode(self, x):
        h = self.encoder(x)
        posterior = DiagonalGaussianDistribution(h)
        return posterior

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x, sample_posterior=True):
        posterior = self.encode(x)
        z = posterior.sample() if sample_posterior else posterior.mean
        dec = self.decode(z)
        return dec, posterior



# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class ResBlock3D(nn.Module):
#     def __init__(self, channels):
#         super().__init__()
#         self.block = nn.Sequential(
#             nn.GroupNorm(8, channels),
#             nn.SiLU(),
#             nn.Conv3d(channels, channels, 3, padding=1),
#             nn.GroupNorm(8, channels),
#             nn.SiLU(),
#             nn.Conv3d(channels, channels, 3, padding=1),
#         )
#     def forward(self, x):
#         return x + self.block(x)

# class Encoder(nn.Module):
#     def __init__(self, 
#                  in_channels,       # 输入数据的通道数，1 
#                  base_channels,     # 基础通道数，64
#                  ch_mult,           # 每层通道倍增列表，[1,2,2,4]
#                  num_res_blocks,    # 每层的 ResBlock 数量，1
#                  z_channels):       # latent 的通道数，4
#         super().__init__()
#         self.layers = nn.ModuleList()
#         # Initial Conv
#         cur_channels = base_channels
#         self.layers.append(nn.Conv3d(in_channels, cur_channels, 3, padding=1))
        
#         # Downsampling
#         for i, mult in enumerate(ch_mult):
#             out_channels = base_channels * mult
#             for _ in range(num_res_blocks):
#                 self.layers.append(ResBlock3D(cur_channels))
#                 self.layers.append(nn.Conv3d(cur_channels, out_channels, 3, padding=1)) # Adapt channels
#                 cur_channels = out_channels
            
#             # Downsample (except last)
#             if i != len(ch_mult) - 1:
#                 self.layers.append(nn.Conv3d(cur_channels, cur_channels, 4, stride=2, padding=1))
        
#         # Middle
#         self.layers.append(ResBlock3D(cur_channels))
#         self.layers.append(nn.GroupNorm(8, cur_channels))
#         self.layers.append(nn.SiLU())
#         self.layers.append(nn.Conv3d(cur_channels, 2 * z_channels, 3, padding=1)) # Output mean and logvar

#     def forward(self, x):
#         for layer in self.layers:
#             x = layer(x)
#         return x

# class Decoder(nn.Module):
#     def __init__(self, out_channels, base_channels, ch_mult, num_res_blocks, z_channels):
#         super().__init__()
#         ch_mult = ch_mult[::-1] # Reverse for decoder
#         cur_channels = base_channels * ch_mult[0]
        
#         self.conv_in = nn.Conv3d(z_channels, cur_channels, 3, padding=1)
        
#         self.layers = nn.ModuleList()
        
#         # Upsampling
#         for i, mult in enumerate(ch_mult):
#             out_channels_layer = base_channels * mult
            
#             for _ in range(num_res_blocks):
#                 self.layers.append(ResBlock3D(cur_channels))
            
#             if i != len(ch_mult) - 1:
#                 # Upsample: ConvTranspose or Interpolate+Conv. ConvTranspose is standard for VAE
#                 self.layers.append(nn.ConvTranspose3d(cur_channels, base_channels * ch_mult[i+1], 4, stride=2, padding=1))
#                 cur_channels = base_channels * ch_mult[i+1]
#             else:
#                 cur_channels = out_channels_layer

#         self.final_block = nn.Sequential(
#             nn.GroupNorm(8, cur_channels),
#             nn.SiLU(),
#             nn.Conv3d(cur_channels, out_channels, 3, padding=1),
#             # nn.Tanh() # 强制输出到 [-1, 1]
#         )

#     def forward(self, x):
#         x = self.conv_in(x)
#         for layer in self.layers:
#             x = layer(x)
#         return self.final_block(x)

# class DiagonalGaussianDistribution(object):
#     def __init__(self, parameters, deterministic=False, logvar_max=5.0, logvar_min=-30.0):
#         """
#         parameters: tensor (B, 2*z_ch, D, H, W) -> chunk into mean, logvar
#         logvar_max: upper clamp for logvar (exp(5) ~ 148 -> 合理范围)
#         """
#         self.parameters = parameters
#         self.mean, raw_logvar = torch.chunk(parameters, 2, dim=1)

#         # clamp logvar to safe numerical range BEFORE exp
#         # 原来是 clamp(..., -30, 20) 会导致 exp(20) ~ 4.8e8 -> KL 爆炸
#         self.logvar = torch.clamp(raw_logvar, min=logvar_min, max=logvar_max)

#         self.deterministic = deterministic

#         # compute var/std AFTER clamping -> 避免 exp 导致溢出
#         self.var = torch.exp(self.logvar)
#         self.std = torch.sqrt(self.var)

#     def sample(self):
#         if self.deterministic:
#             return self.mean
#         # torch.randn_like already on same device & dtype
#         return self.mean + self.std * torch.randn_like(self.mean)

#     def kl(self, other=None):
#         # return per-sample KL (shape: [B])
#         if self.deterministic:
#             # 返回与 mean 在同 device/dtype 且按 batch 尺寸的零向量
#             return torch.zeros(self.mean.shape[0], device=self.mean.device, dtype=self.mean.dtype)
#         # 数值稳定的 KL 计算（使用 clamp 后的 logvar/var）
#         # KL = 0.5 * sum( mu^2 + var - 1 - logvar )
#         # sum over spatial+channel dims, keep batch dim
#         return 0.5 * torch.sum(self.mean * self.mean + self.var - 1.0 - self.logvar, dim=[1, 2, 3, 4])
    
    
# class KLVAE3D(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.encoder = Encoder(
#             in_channels=config['data']['channels'],
#             base_channels=config['model']['base_channels'],
#             ch_mult=config['model']['ch_mult'],
#             num_res_blocks=config['model']['num_res_blocks'],
#             z_channels=config['model']['z_channels']
#         )
#         self.decoder = Decoder(
#             out_channels=config['data']['channels'],
#             base_channels=config['model']['base_channels'],
#             ch_mult=config['model']['ch_mult'],
#             num_res_blocks=config['model']['num_res_blocks'],
#             z_channels=config['model']['z_channels']
#         )

#     def encode(self, x):
#         h = self.encoder(x)
#         posterior = DiagonalGaussianDistribution(h)
#         return posterior

#     def decode(self, z):
#         return self.decoder(z)

#     def forward(self, x, sample_posterior=True):
#         posterior = self.encode(x)
#         if sample_posterior:
#             z = posterior.sample()
#         else:
#             z = posterior.mean
#         dec = self.decode(z)
#         return dec, posterior
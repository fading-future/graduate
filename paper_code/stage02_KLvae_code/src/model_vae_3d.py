import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# 核心组件
# ==========================================

class ResBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels=None):
        super().__init__()
        out_channels = out_channels or in_channels
        self.norm1 = nn.GroupNorm(32, in_channels, eps=1e-6)
        self.conv1 = nn.Conv3d(in_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(32, out_channels, eps=1e-6)
        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, padding=1)
        
        self.shortcut = nn.Identity()
        if in_channels != out_channels:
            self.shortcut = nn.Conv3d(in_channels, out_channels, 1)

    def forward(self, x):
        h = x
        h = self.norm1(h)
        h = F.silu(h)
        h = self.conv1(h)
        h = self.norm2(h)
        h = F.silu(h)
        h = self.conv2(h)
        return h + self.shortcut(x)

class Encoder(nn.Module):
    def __init__(self, in_channels=1, z_channels=4, base_channels=32, ch_mult=(1, 2, 4)):
        super().__init__()
        self.conv_in = nn.Conv3d(in_channels, base_channels, 3, padding=1)
        
        self.blocks = nn.ModuleList([])
        ch = base_channels
        
        # Downsampling: 256 -> 128 -> 64 (Factor 4)
        for mult in ch_mult:
            ch_out = base_channels * mult
            self.blocks.append(nn.Sequential(
                ResBlock3D(ch, ch_out),
                ResBlock3D(ch_out, ch_out),
                nn.Conv3d(ch_out, ch_out, 3, stride=2, padding=1) # Downsample
            ))
            ch = ch_out
            
        self.mid_block = nn.Sequential(
            ResBlock3D(ch, ch),
            ResBlock3D(ch, ch)
        )
        
        self.norm_out = nn.GroupNorm(32, ch, eps=1e-6)
        self.conv_out = nn.Conv3d(ch, 2 * z_channels, 3, padding=1) # Mean + LogVar

    def forward(self, x):
        x = self.conv_in(x)
        for block in self.blocks:
            x = block(x)
        x = self.mid_block(x)
        x = self.norm_out(x)
        x = F.silu(x)
        x = self.conv_out(x)
        return x

class Decoder(nn.Module):
    def __init__(self, out_channels=1, z_channels=4, base_channels=32, ch_mult=(1, 2, 4)):
        super().__init__()
        ch_mult = list(reversed(ch_mult)) # (4, 2, 1)
        ch = base_channels * ch_mult[0]
        
        self.conv_in = nn.Conv3d(z_channels, ch, 3, padding=1)
        self.mid_block = nn.Sequential(
            ResBlock3D(ch, ch),
            ResBlock3D(ch, ch)
        )
        
        self.blocks = nn.ModuleList([])
        
        # Upsampling: 64 -> 128 -> 256
        for i, mult in enumerate(ch_mult):
            ch_out = base_channels * mult
            # 最后一个 Block 不再上采样，或者根据你的层数调整
            # 这里逻辑是: 4->2 (Up), 2->1 (Up), 1->1 (Up) -> 256
            
            # 修正 Upsample 逻辑: 只有前两层需要 Upsample (64->128, 128->256)
            # 如果 input 是 256, Encoder 下采样3次变 32? 
            # 你现在的需求是 latent 64, input 256 -> Factor=4. 
            # 所以 Encoder 只有两层 Downsample. 
            
            self.blocks.append(nn.Sequential(
                ResBlock3D(ch, ch_out),
                ResBlock3D(ch_out, ch_out),
                nn.Upsample(scale_factor=2.0, mode='nearest'),
                nn.Conv3d(ch_out, ch_out, 3, padding=1) 
            ))
            ch = ch_out

        self.norm_out = nn.GroupNorm(32, ch, eps=1e-6)
        self.conv_out = nn.Conv3d(ch, out_channels, 3, padding=1)

    def forward(self, z):
        h = self.conv_in(z)
        h = self.mid_block(h)
        for block in self.blocks:
            h = block(h)
        h = self.norm_out(h)
        h = F.silu(h)
        h = self.conv_out(h)
        return h

# ==========================================
# 主模型: AutoencoderKL
# ==========================================
class AutoencoderKL(nn.Module):
    def __init__(self, 
                 in_channels=1, 
                 out_channels=1, 
                 z_channels=4, 
                 base_channels=64):
        super().__init__()
        # Factor = 4 (256 -> 64) requires 2 downsampling steps
        # ch_mult=(1, 2) means: 64 -> 128 (Down) -> 256 (Down) -> Latent
        self.encoder = Encoder(in_channels, z_channels, base_channels, ch_mult=(1, 2))
        self.decoder = Decoder(out_channels, z_channels, base_channels, ch_mult=(1, 2))
        self.quant_conv = nn.Conv3d(2*z_channels, 2*z_channels, 1)
        self.post_quant_conv = nn.Conv3d(z_channels, z_channels, 1)

    def encode(self, x):
        moments = self.encoder(x)
        moments = self.quant_conv(moments)
        mean, logvar = torch.chunk(moments, 2, dim=1)
        return mean, logvar

    def decode(self, z):
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return dec

    def forward(self, x, sample_posterior=True):
        mean, logvar = self.encode(x)
        if sample_posterior:
            std = torch.exp(0.5 * logvar)
            z = mean + std * torch.randn_like(mean)
        else:
            z = mean
        recon = self.decode(z)
        return recon, mean, logvar

# ==========================================
# 判别器 (PatchGAN 3D)
# ==========================================
class NLayerDiscriminator3D(nn.Module):
    def __init__(self, input_nc=1, ndf=64, n_layers=3):
        super().__init__()
        kw = 4
        padw = 1
        sequence = [nn.Conv3d(input_nc, ndf, kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv3d(ndf * nf_mult_prev, ndf * nf_mult, kw, stride=2, padding=padw, bias=False),
                nn.BatchNorm3d(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv3d(ndf * nf_mult_prev, ndf * nf_mult, kw, stride=1, padding=padw, bias=False),
            nn.BatchNorm3d(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv3d(ndf * nf_mult, 1, kw, stride=1, padding=padw)]
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)
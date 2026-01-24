# vae_model.py
import torch
import torch.nn as nn
from modules import ResBlock3D, AttnBlock3D

class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        ch = config['base_channels']
        self.conv_in = nn.Conv3d(config['in_channels'], ch, 3, padding=1)
        
        self.downs = nn.ModuleList()
        ch_mult = config['ch_mult']
        current_res = 256 
        
        # Downsampling path
        for i, mult in enumerate(ch_mult):
            out_ch = config['base_channels'] * mult
            for _ in range(2): # 每个层级 2 个 ResBlock
                self.downs.append(ResBlock3D(ch, out_ch, config['dropout']))
                ch = out_ch
            
            if i != len(ch_mult) - 1:
                self.downs.append(nn.Conv3d(ch, ch, 3, stride=2, padding=1))
                current_res //= 2
        
        # Middle
        self.mid_block1 = ResBlock3D(ch, ch, config['dropout'])
        self.mid_attn = AttnBlock3D(ch) # Latent大小为64^3时，Attention还是能跑的
        self.mid_block2 = ResBlock3D(ch, ch, config['dropout'])
        
        self.norm_out = nn.GroupNorm(32, ch)
        self.act = nn.SiLU()
        self.conv_out = nn.Conv3d(ch, 2 * config['z_channels'], 3, padding=1)

    def forward(self, x):
        x = self.conv_in(x)
        for module in self.downs:
            x = module(x)
        x = self.mid_block1(x)
        x = self.mid_attn(x)
        x = self.mid_block2(x)
        x = self.act(self.norm_out(x))
        mean, logvar = torch.chunk(self.conv_out(x), 2, dim=1)
        return mean, logvar

class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        ch_mult = config['ch_mult']
        ch = config['base_channels'] * ch_mult[-1]
        z_channels = config['z_channels']
        
        self.conv_in = nn.Conv3d(z_channels, ch, 3, padding=1)
        
        # Middle
        self.mid_block1 = ResBlock3D(ch, ch, config['dropout'])
        self.mid_attn = AttnBlock3D(ch)
        self.mid_block2 = ResBlock3D(ch, ch, config['dropout'])
        
        self.ups = nn.ModuleList()
        reversed_mult = list(reversed(ch_mult))
        
        for i, mult in enumerate(reversed_mult):
            out_ch = config['base_channels'] * mult
            for _ in range(3): # 解码器层数多一点，恢复质量更好
                self.ups.append(ResBlock3D(ch, out_ch, config['dropout']))
                ch = out_ch
            
            if i != len(reversed_mult) - 1:
                self.ups.append(nn.Upsample(scale_factor=2.0, mode='nearest'))
                self.ups.append(nn.Conv3d(ch, ch, 3, padding=1))
                
        self.norm_out = nn.GroupNorm(32, ch)
        self.act = nn.SiLU()
        self.conv_out = nn.Conv3d(ch, config['in_channels'], 3, padding=1)

    def forward(self, z):
        h = self.conv_in(z)
        h = self.mid_block1(h)
        h = self.mid_attn(h)
        h = self.mid_block2(h)
        for module in self.ups:
            h = module(h)
        h = self.act(self.norm_out(h))
        return self.conv_out(h)

class VAE3D(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)
        # 1x1 conv to map to latent space properly
        self.quant_conv = nn.Conv3d(2*config['z_channels'], 2*config['z_channels'], 1)
        self.post_quant_conv = nn.Conv3d(config['z_channels'], config['z_channels'], 1)

    def encode(self, x):
        mean, logvar = self.encoder(x)
        # 这里实际上可以省略 quant_conv 如果直接在 encoder 输出对齐，但为了架构标准保留
        return mean, logvar

    def decode(self, z):
        return self.decoder(z)

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def forward(self, x, sample_posterior=True):
        mean, logvar = self.encode(x)
        if sample_posterior:
            z = self.reparameterize(mean, logvar)
        else:
            z = mean
        dec = self.decode(z)
        return dec, mean, logvar
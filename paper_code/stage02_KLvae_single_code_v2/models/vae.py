import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock3D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(8, channels),
            nn.SiLU(),
            nn.Conv3d(channels, channels, 3, padding=1),
            nn.GroupNorm(8, channels),
            nn.SiLU(),
            nn.Conv3d(channels, channels, 3, padding=1),
        )
    def forward(self, x):
        return x + self.block(x)

class Encoder(nn.Module):
    def __init__(self, 
                 in_channels,       # 输入数据的通道数，1 
                 base_channels,     # 基础通道数，64
                 ch_mult,           # 每层通道倍增列表，[1,2,2,4]
                 num_res_blocks,    # 每层的 ResBlock 数量，1
                 z_channels):       # latent 的通道数，4
        super().__init__()
        self.layers = nn.ModuleList()
        # Initial Conv
        cur_channels = base_channels
        self.layers.append(nn.Conv3d(in_channels, cur_channels, 3, padding=1))
        
        # Downsampling
        for i, mult in enumerate(ch_mult):
            out_channels = base_channels * mult
            for _ in range(num_res_blocks):
                self.layers.append(ResBlock3D(cur_channels))
                self.layers.append(nn.Conv3d(cur_channels, out_channels, 3, padding=1)) # Adapt channels
                cur_channels = out_channels
            
            # Downsample (except last)
            if i != len(ch_mult) - 1:
                self.layers.append(nn.Conv3d(cur_channels, cur_channels, 4, stride=2, padding=1))
        
        # Middle
        self.layers.append(ResBlock3D(cur_channels))
        self.layers.append(nn.GroupNorm(8, cur_channels))
        self.layers.append(nn.SiLU())
        self.layers.append(nn.Conv3d(cur_channels, 2 * z_channels, 3, padding=1)) # Output mean and logvar

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class Decoder(nn.Module):
    def __init__(self, out_channels, base_channels, ch_mult, num_res_blocks, z_channels):
        super().__init__()
        ch_mult = ch_mult[::-1] # Reverse for decoder
        cur_channels = base_channels * ch_mult[0]
        
        self.conv_in = nn.Conv3d(z_channels, cur_channels, 3, padding=1)
        
        self.layers = nn.ModuleList()
        
        # Upsampling
        for i, mult in enumerate(ch_mult):
            out_channels_layer = base_channels * mult
            
            for _ in range(num_res_blocks):
                self.layers.append(ResBlock3D(cur_channels))
            
            if i != len(ch_mult) - 1:
                # Upsample: ConvTranspose or Interpolate+Conv. ConvTranspose is standard for VAE
                self.layers.append(nn.ConvTranspose3d(cur_channels, base_channels * ch_mult[i+1], 4, stride=2, padding=1))
                cur_channels = base_channels * ch_mult[i+1]
            else:
                cur_channels = out_channels_layer

        self.final_block = nn.Sequential(
            nn.GroupNorm(8, cur_channels),
            nn.SiLU(),
            nn.Conv3d(cur_channels, out_channels, 3, padding=1),
            # nn.Tanh() # 强制输出到 [-1, 1]
        )

    def forward(self, x):
        x = self.conv_in(x)
        for layer in self.layers:
            x = layer(x)
        return self.final_block(x)

class DiagonalGaussianDistribution(object):
    def __init__(self, parameters, deterministic=False, logvar_max=5.0, logvar_min=-30.0):
        """
        parameters: tensor (B, 2*z_ch, D, H, W) -> chunk into mean, logvar
        logvar_max: upper clamp for logvar (exp(5) ~ 148 -> 合理范围)
        """
        self.parameters = parameters
        self.mean, raw_logvar = torch.chunk(parameters, 2, dim=1)

        # clamp logvar to safe numerical range BEFORE exp
        # 原来是 clamp(..., -30, 20) 会导致 exp(20) ~ 4.8e8 -> KL 爆炸
        self.logvar = torch.clamp(raw_logvar, min=logvar_min, max=logvar_max)

        self.deterministic = deterministic

        # compute var/std AFTER clamping -> 避免 exp 导致溢出
        self.var = torch.exp(self.logvar)
        self.std = torch.sqrt(self.var)

    def sample(self):
        if self.deterministic:
            return self.mean
        # torch.randn_like already on same device & dtype
        return self.mean + self.std * torch.randn_like(self.mean)

    def kl(self, other=None):
        # return per-sample KL (shape: [B])
        if self.deterministic:
            # 返回与 mean 在同 device/dtype 且按 batch 尺寸的零向量
            return torch.zeros(self.mean.shape[0], device=self.mean.device, dtype=self.mean.dtype)
        # 数值稳定的 KL 计算（使用 clamp 后的 logvar/var）
        # KL = 0.5 * sum( mu^2 + var - 1 - logvar )
        # sum over spatial+channel dims, keep batch dim
        return 0.5 * torch.sum(self.mean * self.mean + self.var - 1.0 - self.logvar, dim=[1, 2, 3, 4])
    
    
class KLVAE3D(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = Encoder(
            in_channels=config['data']['channels'],
            base_channels=config['model']['base_channels'],
            ch_mult=config['model']['ch_mult'],
            num_res_blocks=config['model']['num_res_blocks'],
            z_channels=config['model']['z_channels']
        )
        self.decoder = Decoder(
            out_channels=config['data']['channels'],
            base_channels=config['model']['base_channels'],
            ch_mult=config['model']['ch_mult'],
            num_res_blocks=config['model']['num_res_blocks'],
            z_channels=config['model']['z_channels']
        )

    def encode(self, x):
        h = self.encoder(x)
        posterior = DiagonalGaussianDistribution(h)
        return posterior

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x, sample_posterior=True):
        posterior = self.encode(x)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mean
        dec = self.decode(z)
        return dec, posterior
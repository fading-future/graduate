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
    def __init__(self, in_channels, base_channels, ch_mult, num_res_blocks, z_channels):
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
            nn.Tanh() # 强制输出到 [-1, 1]
        )

    def forward(self, x):
        x = self.conv_in(x)
        for layer in self.layers:
            x = layer(x)
        return self.final_block(x)

class DiagonalGaussianDistribution(object):
    def __init__(self, parameters, deterministic=False):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)

    def sample(self):
        if self.deterministic:
            return self.mean
        x = self.mean + self.std * torch.randn_like(self.mean).to(device=self.parameters.device)
        return x

    def kl(self, other=None):
        if self.deterministic:
            return torch.Tensor([0.])
        else:
            return 0.5 * torch.sum(torch.pow(self.mean, 2) + self.var - 1.0 - self.logvar, dim=[1, 2, 3, 4])

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
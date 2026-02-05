from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .quantizer import VectorQuantizer


class ResBlock3D(nn.Module):
    def __init__(self, channels: int, groups: int = 16):
        super().__init__()
        self.conv1 = nn.Conv3d(channels, channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(groups, channels)
        self.conv2 = nn.Conv3d(channels, channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(groups, channels)
        self.act = nn.SiLU()

    def forward(self, x):
        h = self.conv1(x)
        h = self.norm1(h)
        h = self.act(h)
        h = self.conv2(h)
        h = self.norm2(h)
        return self.act(x + h)


class Encoder3D(nn.Module):
    def __init__(self, in_channels: int, channels, latent_dim: int, num_res_blocks: int, groups: int = 16):
        super().__init__()
        layers = []
        # initial conv
        layers.append(nn.Conv3d(in_channels, channels[0], kernel_size=3, padding=1))
        layers.append(nn.GroupNorm(groups, channels[0]))
        layers.append(nn.SiLU())
        ch = channels[0]
        for next_ch in channels[1:]:
            for _ in range(num_res_blocks):
                layers.append(ResBlock3D(ch, groups))
            # downsample
            layers.append(nn.Conv3d(ch, next_ch, kernel_size=4, stride=2, padding=1))
            layers.append(nn.GroupNorm(groups, next_ch))
            layers.append(nn.SiLU())
            ch = next_ch
        for _ in range(num_res_blocks):
            layers.append(ResBlock3D(ch, groups))
        layers.append(nn.Conv3d(ch, latent_dim, kernel_size=1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class Decoder3D(nn.Module):
    def __init__(
        self,
        out_channels: int,
        channels,
        latent_dim: int,
        num_res_blocks: int,
        groups: int = 16,
        upsample_flags=None,
    ):
        super().__init__()
        layers = []
        layers.append(nn.Conv3d(latent_dim, channels[0], kernel_size=3, padding=1))
        layers.append(nn.GroupNorm(groups, channels[0]))
        layers.append(nn.SiLU())
        ch = channels[0]
        if upsample_flags is None:
            upsample_flags = [True] * (len(channels) - 1)
        if len(upsample_flags) != len(channels) - 1:
            raise ValueError("upsample_flags length must match transitions in channels")
        for i, next_ch in enumerate(channels[1:]):
            for _ in range(num_res_blocks):
                layers.append(ResBlock3D(ch, groups))
            # upsample or keep resolution depending on flag
            if upsample_flags[i]:
                layers.append(nn.ConvTranspose3d(ch, next_ch, kernel_size=4, stride=2, padding=1))
            else:
                layers.append(nn.Conv3d(ch, next_ch, kernel_size=3, padding=1))
            layers.append(nn.GroupNorm(groups, next_ch))
            layers.append(nn.SiLU())
            ch = next_ch
        for _ in range(num_res_blocks):
            layers.append(ResBlock3D(ch, groups))
        layers.append(nn.Conv3d(ch, out_channels, kernel_size=1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class VQVAE3D(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        channels=(16, 64, 128, 256, 512),
        decoder_channels=(512, 256, 256, 64, 16, 16),
        latent_dim: int = 256,
        num_res_blocks_enc: int = 2,
        num_res_blocks_dec: int = 3,
        groups: int = 16,
        codebook_size: int = 3000,
        beta: float = 1.0,
    ):
        super().__init__()
        self.encoder = Encoder3D(in_channels, channels, latent_dim, num_res_blocks_enc, groups)
        self.quantizer = VectorQuantizer(codebook_size, latent_dim, beta)
        # Decoder upsampling flags correspond to transitions in decoder_channels.
        # For [512,256,256,64,16,16], we keep one non-upsampling stage at 256.
        upsample_flags = [True, False, True, True, True]
        self.decoder = Decoder3D(
            out_channels,
            decoder_channels,
            latent_dim,
            num_res_blocks_dec,
            groups,
            upsample_flags=upsample_flags,
        )

    def forward(self, x):
        z = self.encoder(x)
        z_q, indices, codebook_loss, commit_loss = self.quantizer(z)
        x_hat = self.decoder(z_q)
        return x_hat, indices, codebook_loss, commit_loss

    @torch.no_grad()
    def encode_to_indices(self, x):
        z = self.encoder(x)
        _, indices, _, _ = self.quantizer(z)
        return indices

    @torch.no_grad()
    def decode_from_indices(self, indices):
        z_q = self.quantizer.indices_to_embedding(indices)
        return self.decoder(z_q)

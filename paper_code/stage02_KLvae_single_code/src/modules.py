import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint

def get_norm(channels):
    groups = 32
    for g in [32, 16, 8, 4, 2, 1]:
        if channels % g == 0:
            groups = g
            break
    return nn.GroupNorm(groups, channels, eps=1e-6, affine=True)

class ResBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch=None, dropout=0.0, use_checkpoint=False):
        super().__init__()
        out_ch = out_ch or in_ch
        self.use_checkpoint = use_checkpoint
        
        self.norm1 = get_norm(in_ch)
        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, padding=1)
        self.norm2 = get_norm(out_ch)
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, padding=1)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        
        if in_ch != out_ch:
            self.shortcut = nn.Conv3d(in_ch, out_ch, 1)
        else:
            self.shortcut = nn.Identity()

    def forward_impl(self, x):
        h = self.norm1(x)
        h = self.act(h)
        h = self.conv1(h)
        h = self.norm2(h)
        h = self.act(h)
        h = self.dropout(h)
        h = self.conv2(h)
        return h + self.shortcut(x)

    def forward(self, x):
        if self.use_checkpoint and self.training and x.requires_grad:
            return checkpoint.checkpoint(self.forward_impl, x, use_reentrant=False)
        else:
            return self.forward_impl(x)

class Downsample(nn.Module):
    def __init__(self, in_channels, out_channels, use_checkpoint=False):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, 3, stride=2, padding=1)
        self.use_checkpoint = use_checkpoint

    def forward(self, x):
        if self.use_checkpoint and self.training and x.requires_grad:
            return checkpoint.checkpoint(self.conv, x, use_reentrant=False)
        return self.conv(x)

# 【重点修改】替换了 Upsample 类
class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels, use_checkpoint=False):
        super().__init__()
        # 使用 ConvTranspose3d 替代 "插值+卷积"。
        # Kernel=4, Stride=2, Padding=1 是最经典的上采样配置，无棋盘效应。
        self.conv = nn.ConvTranspose3d(in_channels, out_channels, 4, stride=2, padding=1)
        self.use_checkpoint = use_checkpoint

    def forward(self, x):
        # 依然保留 Checkpoint 机制作为双重保险
        if self.use_checkpoint and self.training and x.requires_grad:
            return checkpoint.checkpoint(self.conv, x, use_reentrant=False)
        return self.conv(x)

class AttnBlock3D(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
    def forward(self, x):
        return x
# modules.py
import torch
import torch.nn as nn

class ResBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch=None, dropout=0.0):
        super().__init__()
        out_ch = out_ch or in_ch
        self.norm1 = nn.GroupNorm(32, in_ch, eps=1e-6, affine=True)
        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(32, out_ch, eps=1e-6, affine=True)
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, padding=1)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        
        if in_ch != out_ch:
            self.shortcut = nn.Conv3d(in_ch, out_ch, 1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        h = self.norm1(x)
        h = self.act(h)
        h = self.conv1(h)
        h = self.norm2(h)
        h = self.act(h)
        h = self.dropout(h)
        h = self.conv2(h)
        return h + self.shortcut(x)

class AttnBlock3D(nn.Module):
    # 简单的 Self-Attention，放在 Bottleneck 处提升全局一致性
    def __init__(self, in_channels):
        super().__init__()
        self.norm = nn.GroupNorm(32, in_channels, eps=1e-6, affine=True)
        self.q = nn.Conv3d(in_channels, in_channels, 1)
        self.k = nn.Conv3d(in_channels, in_channels, 1)
        self.v = nn.Conv3d(in_channels, in_channels, 1)
        self.proj_out = nn.Conv3d(in_channels, in_channels, 1)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        b, c, d, h, w = q.shape
        q = q.reshape(b, c, -1).permute(0, 2, 1) # B, N, C
        k = k.reshape(b, c, -1)                  # B, C, N
        w_ = torch.bmm(q, k) * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        v = v.reshape(b, c, -1).permute(0, 2, 1) # B, N, C
        h_ = torch.bmm(w_, v)                    # B, N, C
        h_ = h_.permute(0, 2, 1).reshape(b, c, d, h, w)

        return x + self.proj_out(h_)
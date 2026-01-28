import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ------------------------------
# 1. 基础组件 (保持不变)
# ------------------------------
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class Block3D(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, dropout=0.1):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)
        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, padding=1)
        self.transform = nn.Conv3d(out_ch, out_ch, 3, padding=1)
        self.bn1 = nn.GroupNorm(8, out_ch)
        self.bn2 = nn.GroupNorm(8, out_ch)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        if in_ch != out_ch:
            self.shortcut = nn.Conv3d(in_ch, out_ch, 1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x, t):
        h = self.conv1(x)
        h = self.bn1(h)
        h = self.act(h)
        time_emb = self.time_mlp(t)
        time_emb = time_emb[(..., ) + (None, ) * 3]
        h = h + time_emb
        h = self.transform(h)
        h = self.bn2(h)
        h = self.act(h)
        h = self.dropout(h)
        return h + self.shortcut(x)

# ------------------------------
# 2. 修正后的 Conditional 3D U-Net
# ------------------------------
class ConditionalUNet3D(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, base_channels=64):
        super().__init__()
        self.base_channels = base_channels
        time_dim = base_channels * 4

        # Time Encoding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(base_channels),
            nn.Linear(base_channels, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        # Encoder
        self.inc = nn.Conv3d(in_channels, base_channels, 3, padding=1) # x1: 64
        
        self.down1 = Block3D(base_channels, base_channels*2, time_dim) # x2: 128
        self.pool1 = nn.Conv3d(base_channels*2, base_channels*2, 4, stride=2, padding=1)
        
        self.down2 = Block3D(base_channels*2, base_channels*4, time_dim) # x3: 256
        self.pool2 = nn.Conv3d(base_channels*4, base_channels*4, 4, stride=2, padding=1)

        self.down3 = Block3D(base_channels*4, base_channels*8, time_dim) # x4: 512
        self.pool3 = nn.Conv3d(base_channels*8, base_channels*8, 4, stride=2, padding=1)

        # Bottleneck
        self.bot1 = Block3D(base_channels*8, base_channels*8, time_dim)
        self.bot2 = Block3D(base_channels*8, base_channels*8, time_dim)

        # Decoder
        # Level 3
        self.up1 = nn.ConvTranspose3d(base_channels*8, base_channels*4, 2, stride=2)
        # Concat: base*4 (up) + base*8 (skip x4) = base*12
        self.sa1 = Block3D(base_channels*4 + base_channels*8, base_channels*4, time_dim)

        # Level 2
        self.up2 = nn.ConvTranspose3d(base_channels*4, base_channels*2, 2, stride=2)
        # Concat: base*2 (up) + base*4 (skip x3) = base*6
        self.sa2 = Block3D(base_channels*2 + base_channels*4, base_channels*2, time_dim)

        # Level 1 (Fix Here!)
        self.up3 = nn.ConvTranspose3d(base_channels*2, base_channels, 2, stride=2)
        # Concat: base (up) + base (skip x1) = base*2
        # 原错误代码：Block3D(base_channels*2 + base_channels, ...) -> 3*base (48)
        # 修正代码：Block3D(base_channels * 2, ...) -> 2*base (32)
        self.sa3 = Block3D(base_channels * 2, base_channels, time_dim)

        # Output
        self.outc = nn.Conv3d(base_channels, out_channels, 1)

    def forward(self, x, t):
        t = self.time_mlp(t)
        
        # Encoding
        x1 = self.inc(x)            # Level 1
        
        x2 = self.down1(x1, t)      # Level 2
        p2 = self.pool1(x2)
        
        x3 = self.down2(p2, t)      # Level 3
        p3 = self.pool2(x3)
        
        x4 = self.down3(p3, t)      # Level 4
        p4 = self.pool3(x4)
        
        # Bottleneck
        x_bot = self.bot1(p4, t)
        x_bot = self.bot2(x_bot, t)
        
        # Decoding
        # ----------------- Decoding Level 3 -----------------
        x_up1 = self.up1(x_bot)
        
        # 【修改开始】解决 24 vs 25 不匹配问题
        # 计算 x4 (跳跃连接) 和 x_up1 (上采样) 在 D, H, W 上的差值
        diffD = x4.size(2) - x_up1.size(2)
        diffH = x4.size(3) - x_up1.size(3)
        diffW = x4.size(4) - x_up1.size(4)
        
        # 使用 F.pad 进行填充 (顺序: 前后, 上下, 左右)
        # 注意: PyTorch F.pad 的顺序是倒过来的 (W_left, W_right, H_top, H_bottom, D_front, D_back)
        x_up1 = F.pad(x_up1, (diffW // 2, diffW - diffW // 2,
                              diffH // 2, diffH - diffH // 2,
                              diffD // 2, diffD - diffD // 2))
        # 【修改结束】
        
        x_up1 = torch.cat([x_up1, x4], dim=1)
        x_dec1 = self.sa1(x_up1, t)
        
        # ----------------- Decoding Level 2 -----------------
        x_up2 = self.up2(x_dec1)
        
        # 【同样加上填充逻辑，防止 Level 2 也有奇数问题】
        diffD = x3.size(2) - x_up2.size(2)
        diffH = x3.size(3) - x_up2.size(3)
        diffW = x3.size(4) - x_up2.size(4)
        x_up2 = F.pad(x_up2, (diffW // 2, diffW - diffW // 2,
                              diffH // 2, diffH - diffH // 2,
                              diffD // 2, diffD - diffD // 2))
                              
        x_up2 = torch.cat([x_up2, x3], dim=1)
        x_dec2 = self.sa2(x_up2, t)
        
        # ----------------- Decoding Level 1 -----------------
        x_up3 = self.up3(x_dec2)
        
        # 【同样加上填充逻辑】
        diffD = x1.size(2) - x_up3.size(2)
        diffH = x1.size(3) - x_up3.size(3)
        diffW = x1.size(4) - x_up3.size(4)
        x_up3 = F.pad(x_up3, (diffW // 2, diffW - diffW // 2,
                              diffH // 2, diffH - diffH // 2,
                              diffD // 2, diffD - diffD // 2))
                              
        x_up3 = torch.cat([x_up3, x1], dim=1) 
        x_dec3 = self.sa3(x_up3, t)
        
        output = self.outc(x_dec3)
        return output
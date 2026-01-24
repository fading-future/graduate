import torch.nn as nn

class NLayerDiscriminator3D(nn.Module):
    """
    针对 3D 体素数据的 PatchGAN 判别器。
    
    特点：
    1. 使用 Conv3d 处理三维数据。
    2. 使用 GroupNorm 替代 BatchNorm (对小 Batch Size 更友好，A100 训练大模型必备)。
    3. 输出为 Patch Map，关注局部纹理细节。
    """
    def __init__(self, input_nc=1, ndf=64, n_layers=3):
        """
        Args:
            input_nc (int): 输入通道数 (岩心数据通常为 1)
            ndf (int): 基础 filter 通道数 (默认 64)
            n_layers (int): 下采样层数 (默认 3，意味着感受野适中)
        """
        super(NLayerDiscriminator3D, self).__init__()
        
        kw = 4 # Kernel Size
        padw = 1 # Padding
        
        # 1. 第一层: Conv -> LeakyReLU (第一层通常不加 Norm)
        sequence = [
            nn.Conv3d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), 
            nn.LeakyReLU(0.2, True)
        ]
        
        nf_mult = 1
        nf_mult_prev = 1

        # 2. 中间层: Conv -> GroupNorm -> LeakyReLU
        # 逐渐增加通道数，减小尺寸
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8) # 通道数倍率，最大限制为 8 倍 (即 512 通道)
            
            sequence += [
                nn.Conv3d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=False),
                # GroupNorm: 将通道分成 16 组进行归一化。
                # 相比 BatchNorm，它不依赖 Batch 维度的统计量，非常适合 Batch=1~4 的场景。
                nn.GroupNorm(16, ndf * nf_mult), 
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)

        # 3. 倒数第二层: Stride=1, 不改变尺寸，只增加深度
        sequence += [
            nn.Conv3d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=False),
            nn.GroupNorm(16, ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        # 4. 输出层: 映射到 1 通道 (Logits)
        sequence += [nn.Conv3d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]
        
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """
        Returns:
            tensor: (Batch, 1, D', H', W') 的 Patch Map
        """
        return self.model(input)
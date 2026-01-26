import torch
import torch.nn as nn

class NLayerDiscriminator3D(nn.Module):
    def __init__(self, input_nc=1, ndf=64, n_layers=3):
        """
        input_nc: 输入通道数 (灰度图=1)
        ndf: 基础滤波器数量 (64)
        n_layers: 下采样层数，决定了感受野大小
        """
        super(NLayerDiscriminator3D, self).__init__()
        kw = 4 # Kernel size
        padw = 1 # Padding
        
        sequence = [
            nn.Conv3d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]
        
        nf_mult = 1
        nf_mult_prev = 1
        
        # 逐步增加通道数，减小尺寸
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv3d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw),
                nn.InstanceNorm3d(ndf * nf_mult), # 3D 生成任务通常用 InstanceNorm 替代 BatchNorm
                nn.LeakyReLU(0.2, True)
            ]
        
        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        
        # 最后一层，Stride=1，不改变尺寸
        sequence += [
            nn.Conv3d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw),
            nn.InstanceNorm3d(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]
        
        # 输出层：输出 1 通道的 Logits map
        sequence += [nn.Conv3d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]
        
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)
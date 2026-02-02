import torch
import torch.nn as nn
from copy import deepcopy

class EMA(nn.Module):
    def __init__(self, model, decay=0.9999):
        """
        Args:
            model: 需要进行 EMA 的原模型
            decay: 衰减率，Diffusion 通常使用 0.9999
        """
        super().__init__()
        self.decay = decay
        
        # 1. 创建一个影子模型 (Shadow Model)
        # deepcopy 保证初始权重和原模型一致，但内存独立
        self.ema_model = deepcopy(model)
        
        # 2. 冻结 EMA 模型的参数，因为它只通过 update 更新，不通过梯度下降更新
        for param in self.ema_model.parameters():
            param.requires_grad = False
            
        # 3. 放到和原模型一样的设备上
        self.ema_model.eval()

    @torch.no_grad()
    def update(self, model):
        """
        在每个 step 训练后调用，更新 EMA 权重
        """
        # 更新参数 (Weights & Biases)
        # 公式: v_t = beta * v_{t-1} + (1 - beta) * theta_t
        for ema_param, param in zip(self.ema_model.parameters(), model.parameters()):
            ema_param.data.mul_(self.decay).add_(param.data, alpha=1 - self.decay)
            
        # 更新 Buffer (如 BatchNorm 的 running_mean/var)，直接复制即可，不需要加权
        for ema_buffer, buffer in zip(self.ema_model.buffers(), model.buffers()):
            ema_buffer.data.copy_(buffer.data)

    def state_dict(self):
        """重写 state_dict，只返回 ema_model 的参数"""
        return self.ema_model.state_dict()

    def load_state_dict(self, state_dict):
        """重写 load_state_dict，加载到 ema_model 中"""
        self.ema_model.load_state_dict(state_dict)
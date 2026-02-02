import torch
import math


def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


class DiffusionHelper:
    def __init__(self, timesteps: int, device: torch.device):
        self.timesteps = timesteps
        self.device = device
        self.betas = cosine_beta_schedule(timesteps).to(device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

    def q_sample(self, x0, t, noise):
        ab_t = self.alphas_cumprod[t].view(-1, 1, 1, 1, 1)
        sqrt_ab = torch.sqrt(ab_t)
        sqrt_om = torch.sqrt(1.0 - ab_t)
        return sqrt_ab * x0 + sqrt_om * noise

    def predict_x0_from_eps(self, x_t, eps, t):
        ab_t = self.alphas_cumprod[t].view(-1, 1, 1, 1, 1)
        sqrt_ab = torch.sqrt(ab_t)
        sqrt_om = torch.sqrt(1.0 - ab_t)
        return (x_t - sqrt_om * eps) / (sqrt_ab + 1e-8)

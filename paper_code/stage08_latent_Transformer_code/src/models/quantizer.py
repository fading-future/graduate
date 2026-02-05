from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, beta: float = 1.0):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.beta = beta
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        nn.init.uniform_(self.embedding.weight, -1.0 / num_embeddings, 1.0 / num_embeddings)

    def forward(self, z):
        # z: (B, C, D, H, W)
        z_perm = z.permute(0, 2, 3, 4, 1).contiguous()  # (B, D, H, W, C)
        z_flat = z_perm.view(-1, self.embedding_dim)  # (B*D*H*W, C)

        # compute distances
        dist = (
            z_flat.pow(2).sum(dim=1, keepdim=True)
            + self.embedding.weight.pow(2).sum(dim=1)
            - 2 * torch.matmul(z_flat, self.embedding.weight.t())
        )
        indices = torch.argmin(dist, dim=1)
        z_q = self.embedding(indices).view(z_perm.shape)

        # losses
        codebook_loss = F.mse_loss(z_q.detach(), z_perm)
        commit_loss = F.mse_loss(z_q, z_perm.detach())
        # straight-through
        z_q = z_perm + (z_q - z_perm).detach()
        z_q = z_q.permute(0, 4, 1, 2, 3).contiguous()

        return z_q, indices.view(z_perm.shape[:-1]), codebook_loss, commit_loss

    def indices_to_embedding(self, indices):
        # indices: (B, D, H, W)
        emb = self.embedding(indices)
        return emb.permute(0, 4, 1, 2, 3).contiguous()

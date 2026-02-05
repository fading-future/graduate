from __future__ import annotations

import argparse
import os
import numpy as np
import torch

from utils.config import load_yaml
from utils.checkpoint import load_checkpoint
from models.vqvae import VQVAE3D
from models.transformer import GPT, GPTConfig


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--vqvae_ckpt", type=str, required=True)
    p.add_argument("--transformer_ckpt", type=str, required=True)
    p.add_argument("--porosity_grid", type=str, required=True, help=".npy file of porosity grid (Z,Y,X)")
    p.add_argument("--out_path", type=str, required=True)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_k", type=int, default=None)
    p.add_argument("--context_patches", type=int, default=8)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_yaml(args.config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vqvae = VQVAE3D(
        in_channels=cfg["vqvae"]["in_channels"],
        out_channels=cfg["vqvae"]["out_channels"],
        channels=tuple(cfg["vqvae"]["encoder_channels"]),
        decoder_channels=tuple(cfg["vqvae"]["decoder_channels"]),
        latent_dim=cfg["vqvae"]["latent_dim"],
        num_res_blocks_enc=cfg["vqvae"]["num_res_blocks_enc"],
        num_res_blocks_dec=cfg["vqvae"]["num_res_blocks_dec"],
        groups=cfg["vqvae"]["groups"],
        codebook_size=cfg["vqvae"]["codebook_size"],
        beta=cfg["vqvae"]["beta"],
    ).to(device)
    load_checkpoint(args.vqvae_ckpt, vqvae, map_location=device)
    vqvae.eval()

    gpt_cfg = GPTConfig(
        vocab_size=cfg["model"]["vocab_size"],
        block_size=cfg["model"]["block_size"],
        n_layer=cfg["model"]["n_layer"],
        n_head=cfg["model"]["n_head"],
        n_embd=cfg["model"]["n_embd"],
        dropout=cfg["model"]["dropout"],
        bias=cfg["model"]["bias"],
        cond_dim=cfg["model"]["cond_dim"],
        cond_embd=cfg["model"]["cond_embd"],
    )
    model = GPT(gpt_cfg).to(device)
    load_checkpoint(args.transformer_ckpt, model, map_location=device)
    model.eval()

    porosity_grid = np.load(args.porosity_grid)
    if porosity_grid.ndim != 3:
        raise ValueError(f"porosity_grid must be 3D, got {porosity_grid.shape}")

    z_dim, y_dim, x_dim = porosity_grid.shape
    patch_size = cfg["data"]["patch_size"]
    tokens_per_patch = cfg["model"]["tokens_per_patch"]
    sos_token = cfg["model"]["sos_token"]
    latent_side = round(tokens_per_patch ** (1.0 / 3.0))
    if latent_side ** 3 != tokens_per_patch:
        raise ValueError("tokens_per_patch must be a perfect cube (e.g., 64 = 4^3)")

    # output volume
    out_vol = np.zeros((z_dim * patch_size, y_dim * patch_size, x_dim * patch_size), dtype=np.float32)

    generated_patches = []
    generated_conds = []

    def build_context():
        # use last context_patches patches
        if not generated_patches:
            return torch.empty(0, device=device, dtype=torch.long), torch.empty(0, device=device)
        ctx_tokens = torch.cat(generated_patches[-args.context_patches :], dim=0)
        ctx_conds = torch.cat(generated_conds[-args.context_patches :], dim=0)
        return ctx_tokens, ctx_conds

    for zi in range(z_dim):
        for yi in range(y_dim):
            for xi in range(x_dim):
                # build context
                ctx_tokens, ctx_conds = build_context()
                # generate 64 tokens for current patch
                cur_cond = float(porosity_grid[zi, yi, xi])
                gen_tokens = []
                for _ in range(tokens_per_patch):
                    # build input sequence with SOS + context tokens + generated tokens
                    if gen_tokens:
                        gen_flat = torch.cat(gen_tokens, dim=1).squeeze(0)
                    else:
                        gen_flat = torch.empty(0, device=device, dtype=torch.long)

                    idx = torch.cat(
                        [
                            torch.tensor([sos_token], device=device, dtype=torch.long),
                            ctx_tokens,
                            gen_flat,
                        ],
                        dim=0,
                    ).unsqueeze(0)

                    # conditional sequence aligns to target positions (context + current patch tokens)
                    cond_current = torch.full((gen_flat.numel() + 1,), cur_cond, device=device)
                    cond = torch.cat([ctx_conds, cond_current], dim=0).unsqueeze(0)

                    # keep last block_size
                    idx = idx[:, -gpt_cfg.block_size :]
                    cond = cond[:, -gpt_cfg.block_size :]

                    logits = model(idx.long(), cond)
                    logits = logits[:, -1, :] / max(args.temperature, 1e-8)
                    if args.top_k is not None:
                        v, _ = torch.topk(logits, args.top_k)
                        logits[logits < v[:, [-1]]] = -float("inf")
                    probs = torch.softmax(logits, dim=-1)
                    next_idx = torch.multinomial(probs, num_samples=1)
                    gen_tokens.append(next_idx)

                patch_tokens = torch.cat(gen_tokens, dim=1).squeeze(0)  # (64,)
                # decode to volume
                latent = patch_tokens.view(1, latent_side, latent_side, latent_side)
                with torch.no_grad():
                    patch_vol = vqvae.decode_from_indices(latent).squeeze(0).squeeze(0).cpu().numpy()

                z0, y0, x0 = zi * patch_size, yi * patch_size, xi * patch_size
                out_vol[z0:z0+patch_size, y0:y0+patch_size, x0:x0+patch_size] = patch_vol

                generated_patches.append(patch_tokens.to(device))
                generated_conds.append(torch.full((tokens_per_patch,), cur_cond, device=device))

    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
    np.save(args.out_path, out_vol)
    print(f"Saved generated volume to {args.out_path}")


if __name__ == "__main__":
    main()

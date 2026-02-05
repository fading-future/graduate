from __future__ import annotations

import argparse
import glob
import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.config import load_yaml
from utils.seed import set_seed
from utils.checkpoint import save_checkpoint, load_checkpoint
from data.dataset import TransformerPatchDataset
from models.vqvae import VQVAE3D
from models.transformer import GPT, GPTConfig


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--vqvae_ckpt", type=str, required=True)
    p.add_argument("--out_dir", type=str, default="outputs/transformer")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_yaml(args.config)
    set_seed(cfg.get("seed", 42))

    os.makedirs(args.out_dir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(cfg["data"]["raw_data_dir"], cfg["data"]["file_glob"])))
    if not files:
        raise RuntimeError(f"No NPY files found in {cfg['data']['raw_data_dir']}")

    split = cfg["data"].get("train_split", 0.9)
    split_idx = int(len(files) * split)
    train_files = files[:split_idx]

    train_ds = TransformerPatchDataset(
        train_files,
        patch_size=cfg["data"]["transformer_patch_size"],
        stride=cfg["data"]["transformer_patch_stride"],
        pore_value=cfg["data"].get("pore_value", 1),
        porosity_source=cfg["data"].get("porosity_source", "compute"),
        porosity_csv=cfg["data"].get("porosity_csv"),
        max_samples=cfg["data"].get("max_train_samples"),
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=cfg["data"].get("num_workers", 4),
        pin_memory=True,
    )

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
    for p in vqvae.parameters():
        p.requires_grad = False

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

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["train"]["lr"],
        betas=(cfg["train"].get("beta1", 0.9), cfg["train"].get("beta2", 0.999)),
    )

    sos_token = cfg["model"]["sos_token"]
    t_per_patch = cfg["model"]["tokens_per_patch"]

    for epoch in range(cfg["train"]["epochs"]):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg['train']['epochs']}")
        for vol_128, cond8 in pbar:
            # vol_128: (B,1,128,128,128), cond8: (B,8)
            vol_128 = vol_128.to(device)
            cond8 = cond8.to(device)

            # split into 8 subvolumes (2x2x2) and encode to token indices
            # vol_128 shape: (B,1,128,128,128)
            subvols = vol_128.view(-1, 1, 2, 64, 2, 64, 2, 64)
            subvols = subvols.permute(0, 2, 4, 6, 1, 3, 5, 7).contiguous()
            subvols = subvols.view(-1, 1, 64, 64, 64)

            with torch.no_grad():
                indices = vqvae.encode_to_indices(subvols)  # (B*8, d, h, w)
            indices = indices.view(vol_128.size(0), 8, -1)  # (B,8,64)
            if indices.size(-1) != t_per_patch:
                raise ValueError(
                    f"tokens_per_patch mismatch: expected {t_per_patch}, got {indices.size(-1)}"
                )
            seq = indices.reshape(vol_128.size(0), -1)  # (B,512)

            # build conditional sequence (repeat each cond for 64 tokens)
            cond_seq = cond8.unsqueeze(-1).repeat(1, 1, t_per_patch).reshape(vol_128.size(0), -1)

            # input tokens are shifted by one with SOS, conditional sequence stays aligned to targets
            sos = torch.full((seq.size(0), 1), sos_token, device=device, dtype=seq.dtype)
            idx_in = torch.cat([sos, seq[:, :-1]], dim=1)
            cond_in = cond_seq

            logits = model(idx_in.long(), cond_in)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), seq.reshape(-1).long())

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            pbar.set_postfix({"loss": float(loss.item())})

        save_checkpoint(
            os.path.join(args.out_dir, f"transformer_epoch_{epoch+1}.pt"),
            model,
            optimizer,
            step=epoch + 1,
        )

    print("Training complete.")


if __name__ == "__main__":
    main()

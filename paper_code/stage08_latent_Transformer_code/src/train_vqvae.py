from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.config import load_yaml
from utils.seed import set_seed
from utils.checkpoint import save_checkpoint
from data.dataset import VQVaePatchDataset
from models.vqvae import VQVAE3D


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--out_dir", type=str, default="outputs/vqvae")
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
    val_files = files[split_idx:] if split_idx < len(files) else []

    train_ds = VQVaePatchDataset(
        train_files,
        patch_size=cfg["data"]["patch_size"],
        stride=cfg["data"]["patch_stride"],
        pore_value=cfg["data"].get("pore_value", 1),
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

    model = VQVAE3D(
        in_channels=cfg["model"]["in_channels"],
        out_channels=cfg["model"]["out_channels"],
        channels=tuple(cfg["model"]["encoder_channels"]),
        decoder_channels=tuple(cfg["model"]["decoder_channels"]),
        latent_dim=cfg["model"]["latent_dim"],
        num_res_blocks_enc=cfg["model"]["num_res_blocks_enc"],
        num_res_blocks_dec=cfg["model"]["num_res_blocks_dec"],
        groups=cfg["model"]["groups"],
        codebook_size=cfg["model"]["codebook_size"],
        beta=cfg["model"]["beta"],
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["train"]["lr"],
        betas=(cfg["train"]["beta1"], cfg["train"]["beta2"]),
    )

    codebook_weight = cfg["train"]["codebook_weight_init"]
    for epoch in range(cfg["train"]["epochs"]):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg['train']['epochs']}")
        for batch in pbar:
            x = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            x_hat, _, codebook_loss, commit_loss = model(x)
            recon_loss = F.mse_loss(x_hat, x)
            loss = recon_loss + codebook_weight * (codebook_loss + cfg["model"]["beta"] * commit_loss)
            loss.backward()
            optimizer.step()
            pbar.set_postfix({"loss": float(loss.item()), "recon": float(recon_loss.item())})

        # update codebook weight per epoch
        codebook_weight = min(
            codebook_weight + cfg["train"]["codebook_weight_increase"],
            cfg["train"]["codebook_weight_max"],
        )

        save_checkpoint(
            os.path.join(args.out_dir, f"vqvae_epoch_{epoch+1}.pt"),
            model,
            optimizer,
            step=epoch + 1,
            extra={"codebook_weight": codebook_weight},
        )

    print("Training complete.")


if __name__ == "__main__":
    main()

import os
import csv
import json
import re
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import autocast, GradScaler
from tqdm import tqdm

from src.config import CONFIG
from src.dataset_refine import RefineDataset
from src.models.unet3d import UNet3D
from src.utils import get_root


def setup_experiment():
    root = get_root()
    exp_dir = os.path.join(root, "exp_results", "refine")
    os.makedirs(exp_dir, exist_ok=True)
    model_dir = os.path.join(exp_dir, "models")
    log_dir = os.path.join(exp_dir, "logs")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    with open(os.path.join(exp_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(CONFIG, f, indent=2, ensure_ascii=False)

    csv_path = os.path.join(log_dir, "training_log.csv")
    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Epoch", "Step", "Loss", "LR", "LossUnknown", "LossKnown", "LossBoundary", "LossCoarse", "LossGrad"])

    return exp_dir, model_dir, log_dir, csv_path


def load_latest(model_dir):
    latest = os.path.join(model_dir, "unet_latest.pth")
    if os.path.exists(latest):
        return latest
    ckpts = [f for f in os.listdir(model_dir) if f.startswith("unet_epoch_")]
    if not ckpts:
        return None
    ckpts.sort(key=lambda x: int(re.findall(r"\d+", x)[0]))
    return os.path.join(model_dir, ckpts[-1])


def boundary_band(mask, band):
    if band <= 0:
        return torch.zeros_like(mask)
    k = band * 2 + 1
    known = mask
    unknown = 1.0 - mask
    known_dil = F.max_pool3d(known, kernel_size=k, stride=1, padding=band)
    unk_dil = F.max_pool3d(unknown, kernel_size=k, stride=1, padding=band)
    return (known_dil * unk_dil).clamp(0.0, 1.0)


def grad_loss(pred, gt):
    # simple gradient loss using finite difference
    dz = torch.abs(pred[:, :, 1:, :, :] - pred[:, :, :-1, :, :])
    dy = torch.abs(pred[:, :, :, 1:, :] - pred[:, :, :, :-1, :])
    dx = torch.abs(pred[:, :, :, :, 1:] - pred[:, :, :, :, :-1])
    dz_gt = torch.abs(gt[:, :, 1:, :, :] - gt[:, :, :-1, :, :])
    dy_gt = torch.abs(gt[:, :, :, 1:, :] - gt[:, :, :, :-1, :])
    dx_gt = torch.abs(gt[:, :, :, :, 1:] - gt[:, :, :, :, :-1])
    return (torch.abs(dz - dz_gt).mean() + torch.abs(dy - dy_gt).mean() + torch.abs(dx - dx_gt).mean()) / 3.0


def main():
    device = torch.device(CONFIG["device"])
    exp_dir, model_dir, log_dir, csv_path = setup_experiment()

    r_cfg = CONFIG["REFINE"]
    dataset = RefineDataset(CONFIG["PATHS"]["raw_data_dir"], r_cfg["patch_size"], r_cfg.get("coarse_cache_dir", ""), augment=True)
    loader = DataLoader(dataset, batch_size=r_cfg["batch_size"], shuffle=True, num_workers=r_cfg["num_workers"], pin_memory=True)

    model = UNet3D(
        in_channels=3,  # cond + mask + coarse
        out_channels=1,
        base_channels=r_cfg["model_channels"],
        channel_mults=r_cfg["channel_mults"],
        use_attention=r_cfg["use_attention"],
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=r_cfg["lr"])
    scheduler = CosineAnnealingLR(optimizer, T_max=r_cfg["epochs"], eta_min=1e-6)
    scaler = GradScaler("cuda" if device.type == "cuda" else "cpu")

    start_epoch = 0
    global_step = 0
    if r_cfg.get("resume", True):
        latest = load_latest(model_dir)
        if latest:
            ckpt = torch.load(latest, map_location=device)
            model.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            start_epoch = int(ckpt["epoch"])
            global_step = int(ckpt.get("global_step", 0))

    known_w = CONFIG["LOSS"].get("known_weight", 0.1)
    boundary_w = CONFIG["LOSS"].get("boundary_weight", 1.0)
    band = int(CONFIG["LOSS"].get("boundary_band", 4))
    coarse_w = CONFIG["LOSS"].get("coarse_guidance_weight", 0.1)
    grad_w = CONFIG["LOSS"].get("grad_weight", 0.0)

    for epoch in range(start_epoch, r_cfg["epochs"]):
        model.train()
        pbar = tqdm(loader, desc=f"Refine Epoch {epoch+1}/{r_cfg['epochs']}")
        for batch in pbar:
            global_step += 1
            gt = batch["GT"].to(device)
            cond = batch["Condition"].to(device)
            mask = batch["Mask"].to(device)
            coarse = batch["Coarse"].to(device)
            por = batch["Porosity"].to(device)

            with autocast("cuda" if device.type == "cuda" else "cpu"):
                inp = torch.cat([cond, mask, coarse], dim=1)
                pred = model(inp, por)

                unknown = 1.0 - mask
                known = mask

                l1 = torch.abs(pred - gt)
                loss_unknown = (l1 * unknown).sum() / unknown.sum().clamp_min(1.0)
                loss_known = (l1 * known).sum() / known.sum().clamp_min(1.0)

                bband = boundary_band(mask, band)
                loss_boundary = (l1 * bband).sum() / bband.sum().clamp_min(1.0)

                # coarse guidance on unknown region
                loss_coarse = (torch.abs(pred - coarse) * unknown).sum() / unknown.sum().clamp_min(1.0)

                loss = loss_unknown + known_w * loss_known + boundary_w * loss_boundary + coarse_w * loss_coarse

                loss_grad = torch.tensor(0.0, device=device)
                if grad_w > 0:
                    loss_grad = grad_loss(pred, gt)
                    loss = loss + grad_w * loss_grad

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            lr = optimizer.param_groups[0]["lr"]
            pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{lr:.2e}")

            with open(csv_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([epoch + 1, global_step, f"{loss.item():.6f}", f"{lr:.8f}",
                                 f"{loss_unknown.item():.6f}", f"{loss_known.item():.6f}", f"{loss_boundary.item():.6f}",
                                 f"{loss_coarse.item():.6f}", f"{loss_grad.item():.6f}"])

            if global_step % 500 == 0:
                latest_path = os.path.join(model_dir, "unet_latest.pth")
                torch.save({
                    "epoch": epoch + 1,
                    "global_step": global_step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                }, latest_path)

        scheduler.step()

        if (epoch + 1) % r_cfg["save_every"] == 0:
            save_path = os.path.join(model_dir, f"unet_epoch_{epoch+1}.pth")
            torch.save({
                "epoch": epoch + 1,
                "global_step": global_step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
            }, save_path)


if __name__ == "__main__":
    main()

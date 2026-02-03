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
from src.dataset_coarse import CoarseDataset
from src.models.unet3d import UNet3D
from src.utils import get_root


def setup_experiment():
    root = get_root()
    exp_dir = os.path.join(root, "exp_results", "coarse")
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
            writer.writerow(["Epoch", "Step", "Loss", "LR", "LossUnknown", "LossKnown", "LossBoundary"])

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


def main():
    device = torch.device(CONFIG["device"])
    exp_dir, model_dir, log_dir, csv_path = setup_experiment()

    c_cfg = CONFIG["COARSE"]
    dataset = CoarseDataset(CONFIG["PATHS"]["raw_data_dir"], c_cfg["coarse_size"], augment=True)
    loader = DataLoader(dataset, batch_size=c_cfg["batch_size"], shuffle=True, num_workers=c_cfg["num_workers"], pin_memory=True)

    model = UNet3D(
        in_channels=2,  # cond + mask
        out_channels=1,
        base_channels=c_cfg["model_channels"],
        channel_mults=c_cfg["channel_mults"],
        use_attention=c_cfg["use_attention"],
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=c_cfg["lr"])
    scheduler = CosineAnnealingLR(optimizer, T_max=c_cfg["epochs"], eta_min=1e-6)
    scaler = GradScaler("cuda" if device.type == "cuda" else "cpu")

    start_epoch = 0
    global_step = 0
    if c_cfg.get("resume", True):
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

    for epoch in range(start_epoch, c_cfg["epochs"]):
        model.train()
        pbar = tqdm(loader, desc=f"Coarse Epoch {epoch+1}/{c_cfg['epochs']}")
        for batch in pbar:
            global_step += 1
            gt = batch["GT"].to(device)
            cond = batch["Condition"].to(device)
            mask = batch["Mask"].to(device)
            por = batch["Porosity"].to(device)

            with autocast("cuda" if device.type == "cuda" else "cpu"):
                inp = torch.cat([cond, mask], dim=1)
                pred = model(inp, por)

                unknown = 1.0 - mask
                known = mask

                l1 = torch.abs(pred - gt)
                loss_unknown = (l1 * unknown).sum() / unknown.sum().clamp_min(1.0)
                loss_known = (l1 * known).sum() / known.sum().clamp_min(1.0)

                bband = boundary_band(mask, band)
                loss_boundary = (l1 * bband).sum() / bband.sum().clamp_min(1.0)

                loss = loss_unknown + known_w * loss_known + boundary_w * loss_boundary

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
                                 f"{loss_unknown.item():.6f}", f"{loss_known.item():.6f}", f"{loss_boundary.item():.6f}"])

            # save latest checkpoint occasionally
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

        if (epoch + 1) % c_cfg["save_every"] == 0:
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

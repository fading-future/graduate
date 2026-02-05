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
from src.dataset_patch import PatchLatentDataset
from src.model_unet3d import ConditionalLatentUNet
from src.diffusion import DiffusionHelper
from src.ema import EMA
from src.utils_path import get_root


def setup_experiment():
    root = get_root()
    exp_dir = os.path.join(root, "exp_results", CONFIG["experiment_name"])
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
            writer.writerow(["Epoch", "Step", "Loss", "LR", "DiffTarget", "X0Target"])

    return exp_dir, model_dir, log_dir, csv_path


def load_latest_checkpoint(model_dir, device):
    latest_path = os.path.join(model_dir, "unet_latest.pth")
    if os.path.exists(latest_path):
        return latest_path
    ckpts = [f for f in os.listdir(model_dir) if f.startswith("unet_epoch_") and f.endswith(".pth")]
    if len(ckpts) == 0:
        return None
    ckpts.sort(key=lambda x: int(re.findall(r"\d+", x)[0]))
    return os.path.join(model_dir, ckpts[-1])


def main():
    device = torch.device(CONFIG["device"])
    exp_dir, model_dir, log_dir, csv_path = setup_experiment()

    dataset = PatchLatentDataset(CONFIG["latent_dir"], CONFIG["phi_map_dir"], augment=True)
    dataloader = DataLoader(
        dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        num_workers=CONFIG["num_workers"],
        pin_memory=CONFIG.get("pin_memory", True),
    )

    model = ConditionalLatentUNet(
        in_channels=CONFIG["in_channels"],
        out_channels=CONFIG["out_channels"],
        base_channels=CONFIG["base_channels"],
        channel_mults=CONFIG["channel_mults"],
        use_attention=CONFIG["use_attention"],
    ).to(device)

    ema = EMA(model, decay=0.9999).to(device)
    optimizer = AdamW(model.parameters(), lr=CONFIG["lr"])
    scheduler = CosineAnnealingLR(optimizer, T_max=CONFIG["epochs"], eta_min=1e-6)
    diffusion = DiffusionHelper(CONFIG["timesteps"], device)
    scaler = GradScaler("cuda" if device.type == "cuda" else "cpu")

    start_epoch = 0
    global_step = 0
    if CONFIG.get("resume", True):
        latest = load_latest_checkpoint(model_dir, device)
        if latest:
            print(f"🔄 Loading checkpoint: {latest}")
            ckpt = torch.load(latest, map_location=device)
            model.load_state_dict(ckpt["model_state_dict"])
            if "ema_state_dict" in ckpt:
                ema.load_state_dict(ckpt["ema_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            start_epoch = int(ckpt["epoch"])
            global_step = int(ckpt.get("global_step", 0))

    print(f"🚀 Start training at epoch {start_epoch}, step {global_step}")

    loss_type = CONFIG.get("loss_type", "l1").lower()
    use_min_snr = bool(CONFIG.get("use_min_snr", True))
    gamma = float(CONFIG.get("min_snr_gamma", 5.0))
    x0_w = float(CONFIG.get("x0_weight", 0.2))
    band_w = int(CONFIG.get("boundary_band_width", 0))
    band_weight = float(CONFIG.get("boundary_band_weight", 0.0))
    safe_thresh = float(CONFIG.get("safe_threshold", 8.0))

    for epoch in range(start_epoch, CONFIG["epochs"]):
        model.train()
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']}")

        for batch in pbar:
            global_step += 1
            x0 = batch["GT"].to(device, non_blocking=True)           # (B,C,D,H,W)
            cond = batch["Condition"].to(device, non_blocking=True)  # (B,C,D,H,W)
            mask = batch["Mask"].to(device, non_blocking=True)       # (B,1,D,H,W)
            tmask = batch["TargetMask"].to(device, non_blocking=True) # (B,1,D,H,W)
            phi = batch["Phi"].to(device, non_blocking=True)         # (B,1,D,H,W)
            por = batch["Porosity"].to(device, non_blocking=True)    # (B,1)

            B = x0.shape[0]
            t = torch.randint(0, CONFIG["timesteps"], (B,), device=device).long()

            with autocast("cuda" if device.type == "cuda" else "cpu"):
                ab_t = diffusion.alphas_cumprod[t].view(-1, 1, 1, 1, 1)
                sqrt_ab = torch.sqrt(ab_t)
                sqrt_om = torch.sqrt(1.0 - ab_t)

                noise = torch.randn_like(x0)
                x_t = sqrt_ab * x0 + sqrt_om * noise

                # keep known region consistent (same noise)
                known_xt = sqrt_ab * cond + sqrt_om * noise
                x_t = x_t * (1.0 - mask) + known_xt * mask

                model_in = torch.cat([x_t, cond, mask, phi], dim=1)
                eps_pred = model(model_in, t, por)

                # diffusion loss on target patch only
                if loss_type == "mse":
                    raw = (eps_pred - noise) ** 2
                else:
                    raw = torch.abs(eps_pred - noise)

                if use_min_snr:
                    snr = ab_t / (1.0 - ab_t)
                    w = torch.minimum(snr, torch.tensor(gamma, device=device)) / snr
                    raw = raw * w

                C = eps_pred.shape[1]
                tmask_b = tmask.expand(-1, C, -1, -1, -1)
                # boundary-weighted target loss (emphasize seams)
                if band_w > 0 and band_weight > 0:
                    # erosion: inner = 1 - maxpool(1 - tmask)
                    inner = 1.0 - F.max_pool3d(
                        1.0 - tmask, kernel_size=2 * band_w + 1, stride=1, padding=band_w
                    )
                    inner = inner.clamp(0.0, 1.0)
                    boundary = (tmask - inner).clamp(0.0, 1.0)
                    weight = tmask + boundary * band_weight
                else:
                    weight = tmask

                weight_b = weight.expand(-1, C, -1, -1, -1)
                loss_diff = (raw * weight_b).sum() / weight_b.sum().clamp_min(1.0)

                # x0 loss on target patch (optional)
                pred_x0 = (x_t - sqrt_om * eps_pred) / (sqrt_ab + 1e-8)
                pred_x0 = torch.clamp(pred_x0, -safe_thresh, safe_thresh)
                x0_raw = torch.abs(pred_x0 - x0)
                loss_x0 = (x0_raw * weight_b).sum() / weight_b.sum().clamp_min(1.0)

                loss = loss_diff + x0_w * loss_x0

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            ema.update()

            pbar.set_postfix({"loss": f"{loss.item():.4f}", "diff": f"{loss_diff.item():.4f}"})

            if global_step % 20 == 0:
                with open(csv_path, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        epoch, global_step, f"{loss.item():.6f}",
                        optimizer.param_groups[0]["lr"],
                        f"{loss_diff.item():.6f}",
                        f"{loss_x0.item():.6f}",
                    ])

        scheduler.step()

        if (epoch + 1) % CONFIG["save_model_every"] == 0:
            ckpt = {
                "epoch": epoch + 1,
                "global_step": global_step,
                "model_state_dict": model.state_dict(),
                "ema_state_dict": ema.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
            }
            torch.save(ckpt, os.path.join(model_dir, f"unet_epoch_{epoch+1}.pth"))
            torch.save(ckpt, os.path.join(model_dir, "unet_latest.pth"))


if __name__ == "__main__":
    main()

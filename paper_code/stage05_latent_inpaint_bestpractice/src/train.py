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
from src.dataset_pairs import PairedLatentDataset
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

    # save config
    with open(os.path.join(exp_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(CONFIG, f, indent=2, ensure_ascii=False)

    csv_path = os.path.join(log_dir, "training_log.csv")
    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "Epoch", "Step", "Loss", "LR",
                "DiffUnknown", "DiffKnown", "X0Unknown", "X0Boundary", "LowFreq"
            ])

    return exp_dir, model_dir, log_dir, csv_path


def load_latest_checkpoint(model_dir, device):
    ckpts = [f for f in os.listdir(model_dir) if f.startswith("unet_epoch_") and f.endswith(".pth")]
    if len(ckpts) == 0:
        return None
    ckpts.sort(key=lambda x: int(re.findall(r"\d+", x)[0]))
    return os.path.join(model_dir, ckpts[-1])


def main():
    device = torch.device(CONFIG["device"])
    exp_dir, model_dir, log_dir, csv_path = setup_experiment()

    dataset = PairedLatentDataset(CONFIG["paired_data_dir"], augment=True)
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
            start_epoch = int(ckpt["epoch"])  # resume from next epoch
            global_step = int(ckpt.get("global_step", 0))
            # reset lr to config
            for pg in optimizer.param_groups:
                pg["lr"] = CONFIG["lr"]

    print(f"🚀 Start training at epoch {start_epoch}, step {global_step}")

    # loss config
    loss_type = CONFIG.get("loss_type", "l1").lower()
    use_min_snr = bool(CONFIG.get("use_min_snr", True))
    gamma = float(CONFIG.get("min_snr_gamma", 5.0))

    known_diff_w = float(CONFIG.get("known_diff_weight", 0.05))
    x0_w = float(CONFIG.get("x0_weight", 0.2))
    x0_b_w = float(CONFIG.get("x0_boundary_weight", 1.0))

    band_w = int(CONFIG.get("boundary_band_width", 4))
    band_weight = float(CONFIG.get("boundary_band_weight", 8.0))

    lowfreq_w = float(CONFIG.get("lowfreq_weight", 0.15))
    lowfreq_k = int(CONFIG.get("lowfreq_kernel", 4))

    safe_thresh = float(CONFIG.get("safe_threshold", 8.0))

    for epoch in range(start_epoch, CONFIG["epochs"]):
        model.train()
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']}")

        for batch in pbar:
            global_step += 1
            x0 = batch["GT"].to(device, non_blocking=True)           # (B,4,D,H,W)
            cond = batch["Condition"].to(device, non_blocking=True)  # (B,4,D,H,W)
            mask = batch["Mask"].to(device, non_blocking=True)       # (B,1,D,H,W)
            por = batch["Porosity"].to(device, non_blocking=True)    # (B,1)

            B = x0.shape[0]
            t = torch.randint(0, CONFIG["timesteps"], (B,), device=device).long()

            with autocast("cuda" if device.type == "cuda" else "cpu"):
                ab_t = diffusion.alphas_cumprod[t].view(-1, 1, 1, 1, 1)
                sqrt_ab = torch.sqrt(ab_t)
                sqrt_om = torch.sqrt(1.0 - ab_t)

                noise = torch.randn_like(x0)

                # full noisy
                x_t = sqrt_ab * x0 + sqrt_om * noise
                # known-hard (use same noise)
                known_xt = sqrt_ab * cond + sqrt_om * noise
                x_t = x_t * (1.0 - mask) + known_xt * mask

                model_in = torch.cat([x_t, cond, mask], dim=1)
                eps_pred = model(model_in, t, por)

                # diffusion loss
                if loss_type == "mse":
                    raw = (eps_pred - noise) ** 2
                else:
                    raw = torch.abs(eps_pred - noise)

                # min-SNR weighting
                if use_min_snr:
                    snr = ab_t / (1.0 - ab_t)
                    w = torch.minimum(snr, torch.tensor(gamma, device=device)) / snr
                    raw = raw * w

                unknown = (1.0 - mask)
                known = mask

                C = eps_pred.shape[1]
                unknown_b = unknown.expand(-1, C, -1, -1, -1)
                known_b = known.expand(-1, C, -1, -1, -1)

                # boundary band (unknown near known)
                boundary_band = torch.zeros_like(unknown)
                if band_w > 0 and band_weight > 0:
                    k = 2 * band_w + 1
                    known_dil = F.max_pool3d(known, kernel_size=k, stride=1, padding=band_w)
                    unk_dil = F.max_pool3d(unknown, kernel_size=k, stride=1, padding=band_w)
                    boundary_band = (known_dil * unk_dil).clamp(0.0, 1.0)
                    weight_map = (1.0 + band_weight * boundary_band) * unknown
                    weight_map_b = weight_map.expand(-1, C, -1, -1, -1)
                else:
                    weight_map_b = unknown_b

                den_u = weight_map_b.sum().clamp_min(1.0)
                loss_diff_unknown = (raw * weight_map_b).sum() / den_u

                den_k = known_b.sum().clamp_min(1.0)
                loss_diff_known = (raw * known_b).sum() / den_k

                # pred x0
                pred_x0 = (x_t - sqrt_om * eps_pred) / (sqrt_ab + 1e-8)
                pred_x0 = torch.clamp(pred_x0, -safe_thresh, safe_thresh)

                # x0 L1 on unknown
                x0_raw = torch.abs(pred_x0 - x0)
                loss_x0_unknown = (x0_raw * unknown_b).sum() / unknown_b.sum().clamp_min(1.0)

                # boundary x0 loss
                boundary_unknown = boundary_band * unknown
                boundary_unknown_b = boundary_unknown.expand(-1, C, -1, -1, -1)
                loss_x0_boundary = (x0_raw * boundary_unknown_b).sum() / boundary_unknown_b.sum().clamp_min(1.0)

                # low-frequency x0 loss (avg pool with SAME output size)
                if lowfreq_w > 0 and lowfreq_k > 1:
                    if lowfreq_k % 2 == 1:
                        pad = lowfreq_k // 2
                        x0_lp = F.avg_pool3d(x0, kernel_size=lowfreq_k, stride=1, padding=pad)
                        pred_lp = F.avg_pool3d(pred_x0, kernel_size=lowfreq_k, stride=1, padding=pad)
                    else:
                        # for even kernel, use asymmetric padding to keep size
                        total_pad = lowfreq_k - 1
                        p0 = total_pad // 2
                        p1 = total_pad - p0
                        pad_tuple = (p0, p1, p0, p1, p0, p1)  # W, H, D
                        x0_pad = F.pad(x0, pad_tuple, mode="replicate")
                        pred_pad = F.pad(pred_x0, pad_tuple, mode="replicate")
                        x0_lp = F.avg_pool3d(x0_pad, kernel_size=lowfreq_k, stride=1, padding=0)
                        pred_lp = F.avg_pool3d(pred_pad, kernel_size=lowfreq_k, stride=1, padding=0)
                    lp_raw = torch.abs(pred_lp - x0_lp)
                    loss_lowfreq = (lp_raw * unknown_b).sum() / unknown_b.sum().clamp_min(1.0)
                else:
                    loss_lowfreq = torch.zeros((), device=device)

                loss = (
                    loss_diff_unknown
                    + known_diff_w * loss_diff_known
                    + x0_w * loss_x0_unknown
                    + x0_b_w * loss_x0_boundary
                    + lowfreq_w * loss_lowfreq
                )

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            ema.update(model)

            lr = optimizer.param_groups[0]["lr"]
            pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{lr:.2e}")

            # log
            with open(csv_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    epoch + 1,
                    global_step,
                    f"{loss.item():.6f}",
                    f"{lr:.8f}",
                    f"{loss_diff_unknown.item():.6f}",
                    f"{loss_diff_known.item():.6f}",
                    f"{loss_x0_unknown.item():.6f}",
                    f"{loss_x0_boundary.item():.6f}",
                    f"{loss_lowfreq.item():.6f}",
                ])

        scheduler.step(epoch + 1)

        if (epoch + 1) % CONFIG["save_every"] == 0:
            save_path = os.path.join(model_dir, f"unet_epoch_{epoch+1}.pth")
            torch.save({
                "epoch": epoch + 1,
                "global_step": global_step,
                "model_state_dict": model.state_dict(),
                "ema_state_dict": ema.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
            }, save_path)
            print(f"💾 Saved checkpoint: {save_path}")


if __name__ == "__main__":
    main()

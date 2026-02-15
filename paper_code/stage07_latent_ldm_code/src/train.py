import os
import csv
import json
import re
from contextlib import contextmanager
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
from utils.utils_path import get_root
from src.infer import ddim_sample
from model.vae import KLVAE3D

import numpy as np
import random
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml


def _safe_torch_load(path: str, map_location):
    """Prefer restricted checkpoint loading when supported by local PyTorch."""
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)
    except Exception:
        # Compatibility fallback for checkpoints that include unsupported objects.
        return torch.load(path, map_location=map_location)


@contextmanager
def _preserve_rng_state(device: torch.device):
    py_state = random.getstate()
    np_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    cuda_state = None
    if device.type == "cuda" and torch.cuda.is_available():
        cuda_state = torch.cuda.get_rng_state_all()
    try:
        yield
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
        torch.random.set_rng_state(torch_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)


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


def _save_latent_slices(latent: np.ndarray, out_path: str, title: str = ""):
    # latent: (C, D, H, W)
    c = 0
    vol = latent[c]
    D, H, W = vol.shape
    cz, cy, cx = D // 2, H // 2, W // 2
    slices = [vol[cz], vol[:, cy, :], vol[:, :, cx]]
    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    for i, ax in enumerate(axes):
        ax.imshow(slices[i], cmap="gray")
        ax.axis("off")
    fig.suptitle(title, fontsize=10)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close(fig)


def _save_voxel_slices(vol: np.ndarray, out_path: str, title: str = ""):
    # vol: (D,H,W)
    D, H, W = vol.shape
    cz, cy, cx = D // 2, H // 2, W // 2
    slices = [vol[cz], vol[:, cy, :], vol[:, :, cx]]
    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    for i, ax in enumerate(axes):
        ax.imshow(slices[i], cmap="gray")
        ax.axis("off")
    fig.suptitle(title, fontsize=10)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close(fig)


def _load_klvae(cfg_path: str, ckpt_path: str, device: torch.device):
    if not cfg_path or not ckpt_path:
        return None
    if not os.path.exists(cfg_path) or not os.path.exists(ckpt_path):
        print("⚠️ eval_decode_voxel enabled but VAE config/ckpt not found.")
        return None
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    vae = KLVAE3D(cfg).to(device)
    ckpt = _safe_torch_load(ckpt_path, map_location=device)
    if isinstance(ckpt, dict) and "vae_state_dict" in ckpt:
        state = ckpt["vae_state_dict"]
    elif isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
    else:
        state = ckpt
    new_state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
    vae.load_state_dict(new_state)
    vae.eval()
    return vae


def run_eval_step(model, diffusion, device, step, exp_dir):
    eval_every = int(CONFIG.get("eval_every_steps", 0))
    if eval_every <= 0:
        return

    eval_dir = os.path.join(exp_dir, CONFIG.get("eval_output_dir", "eval"))
    os.makedirs(eval_dir, exist_ok=True)

    with _preserve_rng_state(device):
        # deterministic sample
        seed = int(CONFIG.get("eval_seed", 1234))
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)

        dataset_eval = PatchLatentDataset(CONFIG["latent_dir"], CONFIG["phi_map_dir"], augment=False)
        idx = int(CONFIG.get("eval_index", 0)) % len(dataset_eval)
        sample = dataset_eval[idx]

        x0 = sample["GT"].unsqueeze(0).to(device)
        cond = sample["Condition"].unsqueeze(0).to(device)
        mask = sample["Mask"].unsqueeze(0).to(device)
        phi = sample["Phi"].unsqueeze(0).to(device)
        por = sample["Porosity"].unsqueeze(0).to(device)

        model_was_train = model.training
        model.eval()
        with torch.no_grad():
            x_pred = ddim_sample(
                model, cond, mask, phi, por,
                diffusion,
                steps=int(CONFIG.get("eval_ddim_steps", 50)),
                seed=seed,
                safe_thresh=float(CONFIG.get("safe_threshold", 8.0)),
            )
        if model_was_train:
            model.train()

    # save latents
    step_tag = f"step{step:07d}"
    np.save(os.path.join(eval_dir, f"{step_tag}_gt.npy"), x0.cpu().float().numpy()[0])
    np.save(os.path.join(eval_dir, f"{step_tag}_pred.npy"), x_pred.cpu().float().numpy()[0])
    np.save(os.path.join(eval_dir, f"{step_tag}_cond.npy"), cond.cpu().float().numpy()[0])

    # quick slice png
    if bool(CONFIG.get("eval_save_png", True)):
        _save_latent_slices(x_pred.cpu().float().numpy()[0], os.path.join(eval_dir, f"{step_tag}_pred.png"), "pred")
        _save_latent_slices(x0.cpu().float().numpy()[0], os.path.join(eval_dir, f"{step_tag}_gt.png"), "gt")

    # optional voxel decode + visualization
    if bool(CONFIG.get("eval_decode_voxel", False)):
        vae_cfg = CONFIG.get("eval_vae_config_path", "")
        vae_ckpt = CONFIG.get("eval_vae_ckpt_path", "")
        vae = getattr(run_eval_step, "_vae_cache", None)
        if vae is None:
            vae = _load_klvae(vae_cfg, vae_ckpt, device)
            run_eval_step._vae_cache = vae

        if vae is not None:
            scale = float(CONFIG.get("scale_factor", 1.0))
            if scale == 0.0:
                scale = 1.0
            # unscale latents before decode
            z_pred = x_pred / scale
            z_gt = x0 / scale

            with torch.no_grad():
                if device.type == "cuda":
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        vox_pred = vae.decode(z_pred)
                        vox_gt = vae.decode(z_gt)
                else:
                    vox_pred = vae.decode(z_pred)
                    vox_gt = vae.decode(z_gt)

            vox_pred = vox_pred.cpu().float().numpy()[0, 0]
            vox_gt = vox_gt.cpu().float().numpy()[0, 0]

            np.save(os.path.join(eval_dir, f"{step_tag}_pred_voxel.npy"), vox_pred)
            np.save(os.path.join(eval_dir, f"{step_tag}_gt_voxel.npy"), vox_gt)

            if bool(CONFIG.get("eval_voxel_save_png", True)):
                _save_voxel_slices(vox_pred, os.path.join(eval_dir, f"{step_tag}_pred_voxel.png"), "pred_voxel")
                _save_voxel_slices(vox_gt, os.path.join(eval_dir, f"{step_tag}_gt_voxel.png"), "gt_voxel")


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

    ema = EMA(model, decay=float(CONFIG.get("ema_decay", 0.9999))).to(device)
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
            ckpt = _safe_torch_load(latest, map_location=device)
            model.load_state_dict(ckpt["model_state_dict"])
            if "ema_state_dict" in ckpt:
                ema.load_state_dict(ckpt["ema_state_dict"])
            if bool(CONFIG.get("resume_load_optimizer", True)) and "optimizer_state_dict" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            else:
                print("ℹ️ Resume without optimizer state (fresh optimizer).")
            if bool(CONFIG.get("resume_load_scheduler", True)) and "scheduler_state_dict" in ckpt:
                scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            else:
                print("ℹ️ Resume without scheduler state (fresh scheduler).")
            start_epoch = int(ckpt["epoch"])
            global_step = int(ckpt.get("global_step", 0))

    print(f"🚀 Start training at epoch {start_epoch}, step {global_step}")

    loss_type = CONFIG.get("loss_type", "l1").lower()
    use_min_snr = bool(CONFIG.get("use_min_snr", True))
    gamma = float(CONFIG.get("min_snr_gamma", 5.0))
    x0_w = float(CONFIG.get("x0_weight", 0.2))
    use_target_stats_loss = bool(CONFIG.get("use_target_stats_loss", False))
    target_stats_weight = float(CONFIG.get("target_stats_weight", 0.0))
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

                if use_target_stats_loss and target_stats_weight > 0.0:
                    # Match mean/std inside target region to discourage collapsed predictions.
                    stat_w = tmask_b
                    denom = stat_w.sum(dim=(2, 3, 4), keepdim=True).clamp_min(1.0)
                    pred_mean = (pred_x0 * stat_w).sum(dim=(2, 3, 4), keepdim=True) / denom
                    gt_mean = (x0 * stat_w).sum(dim=(2, 3, 4), keepdim=True) / denom
                    pred_var = ((pred_x0 - pred_mean) ** 2 * stat_w).sum(dim=(2, 3, 4), keepdim=True) / denom
                    gt_var = ((x0 - gt_mean) ** 2 * stat_w).sum(dim=(2, 3, 4), keepdim=True) / denom
                    pred_std = torch.sqrt(pred_var + 1e-8)
                    gt_std = torch.sqrt(gt_var + 1e-8)
                    loss_stats = torch.abs(pred_mean - gt_mean).mean() + torch.abs(pred_std - gt_std).mean()
                else:
                    loss_stats = torch.zeros((), device=device, dtype=loss_x0.dtype)

                loss = loss_diff + x0_w * loss_x0 + target_stats_weight * loss_stats

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            grad_clip_norm = float(CONFIG.get("grad_clip_norm", 0.0))
            if grad_clip_norm > 0.0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()

            ema.update(model)

            postfix = {"loss": f"{loss.item():.4f}", "diff": f"{loss_diff.item():.4f}"}
            if use_target_stats_loss and target_stats_weight > 0.0:
                postfix["stats"] = f"{loss_stats.item():.4f}"
            pbar.set_postfix(postfix)

            if global_step % CONFIG.get("save_log_every", 1) == 0:
                with open(csv_path, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        epoch, global_step, f"{loss.item():.6f}",
                        optimizer.param_groups[0]["lr"],
                        f"{loss_diff.item():.6f}",
                        f"{loss_x0.item():.6f}",
                    ])

            # eval during training
            if int(CONFIG.get("eval_every_steps", 0)) > 0 and (global_step % int(CONFIG.get("eval_every_steps", 0)) == 0):
                use_ema_eval = bool(CONFIG.get("eval_use_ema", False))
                if use_ema_eval and hasattr(ema, "ema_model"):
                    eval_model = ema.ema_model
                else:
                    eval_model = model
                run_eval_step(eval_model, diffusion, device, global_step, exp_dir)

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

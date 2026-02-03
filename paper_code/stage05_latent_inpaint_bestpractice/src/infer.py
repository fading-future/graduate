import os
import glob
import re
import argparse
import json
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
import yaml

from src.config import CONFIG
from src.model_unet3d import ConditionalLatentUNet
from src.diffusion import DiffusionHelper
from src.models.vae import KLVAE3D
from src.utils_path import get_root


def build_porosity_map_from_csv(csv_path: str):
    por_map = {}
    if not csv_path or not os.path.exists(csv_path):
        return por_map
    with open(csv_path, "r", encoding="utf-8") as f:
        _ = f.readline()  # header
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 3:
                continue
            fname = parts[0].strip()
            por = parts[2].strip()
            try:
                por_map[fname] = float(por)
            except ValueError:
                continue
    return por_map


def load_scale_factor(paired_dir: str):
    if CONFIG.get("scale_factor") is not None:
        return float(CONFIG["scale_factor"])
    stats_path = os.path.join(paired_dir, "stats.json")
    if os.path.exists(stats_path):
        with open(stats_path, "r", encoding="utf-8") as f:
            stats = json.load(f)
        return float(stats.get("suggested_scale_factor", 1.0))
    return 1.0


def load_vae(cfg_path: str, ckpt_path: str, device: torch.device):
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    vae = KLVAE3D(cfg).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    if isinstance(ckpt, dict) and "vae_state_dict" in ckpt:
        state = ckpt["vae_state_dict"]
    elif isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
    else:
        state = ckpt
    new_state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
    vae.load_state_dict(new_state)
    vae.eval()
    return vae, cfg


def make_mask_pixel(shape, axis: str, ratio: float):
    d, h, w = shape
    if axis.upper() == "D":
        size = d
    elif axis.upper() == "H":
        size = h
    else:
        size = w
    cut = max(1, min(int(size * ratio), size - 1))
    mask = np.zeros((1, d, h, w), dtype=np.float32)
    if axis.upper() == "D":
        mask[:, :cut, :, :] = 1.0
    elif axis.upper() == "H":
        mask[:, :, :cut, :] = 1.0
    else:
        mask[:, :, :, :cut] = 1.0
    return mask, cut


def ddim_sample(model, cond, mask, porosity, diffusion: DiffusionHelper, steps=200, seed=1234, safe_thresh=8.0):
    model.eval()
    total_timesteps = diffusion.timesteps
    alphas_cumprod = diffusion.alphas_cumprod

    times = torch.linspace(0, total_timesteps - 1, steps=steps, device=cond.device)
    times = torch.unique(torch.round(times).long(), sorted=True)
    times = list(reversed(times.tolist()))

    if seed is not None:
        # torch.randn_like(generator=...) is not available in some torch builds
        # Use global RNG seeding for compatibility
        torch.manual_seed(int(seed))
        if cond.device.type == "cuda":
            torch.cuda.manual_seed_all(int(seed))
        fixed_noise = torch.randn_like(cond)
        x = fixed_noise.clone()
    else:
        fixed_noise = torch.randn_like(cond)
        x = fixed_noise.clone()

    t_start = times[0]
    ab_start = alphas_cumprod[t_start]
    known_xt = torch.sqrt(ab_start) * cond + torch.sqrt(1.0 - ab_start) * fixed_noise
    x = x * (1.0 - mask) + known_xt * mask
    x = torch.clamp(x, -safe_thresh, safe_thresh)

    with torch.no_grad():
        for i, t in enumerate(tqdm(times, desc="DDIM")):
            t_tensor = torch.full((cond.shape[0],), t, device=cond.device, dtype=torch.long)
            t_prev = times[i + 1] if i < len(times) - 1 else -1

            model_in = torch.cat([x, cond, mask], dim=1)
            eps = model(model_in, t_tensor, porosity)

            ab_t = alphas_cumprod[t]
            ab_prev = alphas_cumprod[t_prev] if t_prev >= 0 else torch.tensor(1.0, device=cond.device)

            pred_x0 = (x - torch.sqrt(1.0 - ab_t) * eps) / (torch.sqrt(ab_t) + 1e-8)
            pred_x0 = torch.clamp(pred_x0, -safe_thresh, safe_thresh)

            x_prev = torch.sqrt(ab_prev) * pred_x0 + torch.sqrt(1.0 - ab_prev) * eps

            if t_prev >= 0:
                known_prev = torch.sqrt(ab_prev) * cond + torch.sqrt(1.0 - ab_prev) * fixed_noise
                x = x_prev * (1.0 - mask) + known_prev * mask
            else:
                x = pred_x0 * (1.0 - mask) + cond * mask

            x = torch.clamp(x, -safe_thresh, safe_thresh)

    return x


def decode_tiled_klvae(vae_model, z, latent_tile=16, latent_overlap=4, up_factor=8):
    assert z.dim() == 5 and z.size(0) == 1
    B, C, D, H, W = z.shape
    device = z.device

    step = latent_tile - latent_overlap
    out_D, out_H, out_W = D * up_factor, H * up_factor, W * up_factor
    out = torch.zeros((1, 1, out_D, out_H, out_W), device=device, dtype=torch.float32)
    wgt = torch.zeros_like(out)

    def tri(n):
        x = torch.linspace(0, 1, n, device=device)
        w = 1.0 - (2.0 * (x - 0.5)).abs()
        return w.clamp_min(0.0)

    for dz in range(0, D, step):
        for dy in range(0, H, step):
            for dx in range(0, W, step):
                z0, y0, x0 = dz, dy, dx
                z1 = min(z0 + latent_tile, D)
                y1 = min(y0 + latent_tile, H)
                x1 = min(x0 + latent_tile, W)

                z0 = max(0, z1 - latent_tile)
                y0 = max(0, y1 - latent_tile)
                x0 = max(0, x1 - latent_tile)

                patch = z[:, :, z0:z1, y0:y1, x0:x1]
                with torch.no_grad():
                    ctx = torch.amp.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else torch.no_grad()
                    with ctx:
                        dec = vae_model.decode(patch)
                dec = dec.float()

                oz0, oy0, ox0 = z0 * up_factor, y0 * up_factor, x0 * up_factor
                oz1, oy1, ox1 = z1 * up_factor, y1 * up_factor, x1 * up_factor

                pD, pH, pW = dec.shape[-3:]
                wz = tri(pD).view(1, 1, pD, 1, 1)
                wy = tri(pH).view(1, 1, 1, pH, 1)
                wx = tri(pW).view(1, 1, 1, 1, pW)
                ww = (wz * wy * wx).float()

                out[:, :, oz0:oz1, oy0:oy1, ox0:ox1] += dec * ww
                wgt[:, :, oz0:oz1, oy0:oy1, ox0:ox1] += ww

    out = out / (wgt + 1e-8)
    return out


def visualize_slices(vol_gt, vol_cond, vol_gen, mask_pixel, save_path, title):
    flat_gt = vol_gt.flatten()
    vmin, vmax = np.percentile(flat_gt, 1), np.percentile(flat_gt, 99)

    D, H, W = vol_gt.shape
    cz, cy, cx = D // 2, H // 2, W // 2

    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    plt.suptitle(title, fontsize=16, y=0.98)

    cols = ["GT", "Condition", "Generated"]
    vols = [vol_gt, vol_cond, vol_gen]

    def draw_line(ax):
        z_profile = mask_pixel.mean(axis=(1, 2))
        split = np.where(np.diff(z_profile) != 0)[0]
        if len(split) > 0:
            ax.axhline(y=split[0], color='red', linestyle='--', linewidth=2, alpha=0.8)

    for i, (name, vol) in enumerate(zip(cols, vols)):
        ax = axes[0, i]
        ax.imshow(vol[cz], cmap='gray', vmin=vmin, vmax=vmax)
        ax.set_title(f"{name} XY")
        ax.axis('off')

        ax = axes[1, i]
        ax.imshow(vol[:, cy, :], cmap='gray', vmin=vmin, vmax=vmax, origin='lower')
        draw_line(ax)
        ax.set_title(f"{name} XZ")
        ax.axis('off')

        ax = axes[2, i]
        ax.imshow(vol[:, :, cx], cmap='gray', vmin=vmin, vmax=vmax, origin='lower')
        draw_line(ax)
        ax.set_title(f"{name} ZY")
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--paired_npz", type=str, default=None, help="path to paired .npz (optional)")
    parser.add_argument("--raw_file", type=str, default=None, help="path to raw 256^3 .npy (optional)")
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--decode", action="store_true", help="decode to pixel and save PNG")
    args = parser.parse_args()

    device = torch.device(CONFIG["device"])
    scale_factor = load_scale_factor(CONFIG["paired_data_dir"])
    safe_thresh = float(CONFIG.get("safe_threshold", 8.0))

    # load model
    root = get_root()
    exp_dir = os.path.join(root, "exp_results", CONFIG["experiment_name"])
    model_dir = os.path.join(exp_dir, "models")
    ckpts = sorted(glob.glob(os.path.join(model_dir, "unet_epoch_*.pth")), key=os.path.getmtime)
    if len(ckpts) == 0:
        raise FileNotFoundError("No checkpoints found.")
    ckpt_path = ckpts[-1]

    model = ConditionalLatentUNet(
        in_channels=CONFIG["in_channels"],
        out_channels=CONFIG["out_channels"],
        base_channels=CONFIG["base_channels"],
        channel_mults=CONFIG["channel_mults"],
        use_attention=CONFIG["use_attention"],
    ).to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    if "ema_state_dict" in ckpt:
        model.load_state_dict(ckpt["ema_state_dict"])
    else:
        model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    diffusion = DiffusionHelper(CONFIG["timesteps"], device)

    # prepare condition
    gt_latent = None
    if args.paired_npz is not None:
        data = np.load(args.paired_npz)
        z_cond = data["z_cond"].astype(np.float32)
        mask = data["mask"].astype(np.float32)
        por = data["porosity"].astype(np.float32)
        z_cond = torch.from_numpy(z_cond).unsqueeze(0).to(device)
        mask = torch.from_numpy(mask).unsqueeze(0).to(device)
        por = torch.from_numpy(por).to(device)
        if "z_full" in data:
            gt_latent = torch.from_numpy(data["z_full"].astype(np.float32)).unsqueeze(0).to(device)
    elif args.raw_file is not None:
        vae, _ = load_vae(CONFIG["vae_config_path"], CONFIG["vae_ckpt_path"], device)
        por_map = build_porosity_map_from_csv(CONFIG.get("porosity_csv", ""))
        base = os.path.basename(args.raw_file)
        por = por_map.get(base, 0.15)
        por = torch.tensor([por], dtype=torch.float32, device=device)

        raw = np.load(args.raw_file, mmap_mode="r").astype(np.float32)
        raw = (raw / 65535.0) * 2.0 - 1.0
        mask_pixel, _ = make_mask_pixel(raw.shape, CONFIG["axis"], float(CONFIG["ratio"]))
        cond_pixel = raw[None, ...] * mask_pixel

        x_cond = torch.from_numpy(cond_pixel).unsqueeze(0).to(device)
        with torch.no_grad():
            if device.type == "cuda":
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    z_cond = vae.encode(x_cond).mean
            else:
                z_cond = vae.encode(x_cond).mean
        # latent mask from cond
        zc, zd, zh, zw = z_cond.shape[1:]
        mask = torch.zeros((1, 1, zd, zh, zw), device=device)
        cut = int(zd * float(CONFIG["ratio"]))
        mask[:, :, :cut, :, :] = 1.0
    else:
        raise ValueError("Provide --paired_npz or --raw_file")

    # scale + clamp
    z_cond = z_cond * scale_factor
    z_cond = torch.clamp(z_cond, -safe_thresh, safe_thresh)

    # sample
    gen = ddim_sample(
        model,
        z_cond,
        mask,
        por,
        diffusion,
        steps=int(CONFIG.get("ddim_steps", 200)),
        seed=int(CONFIG.get("seed", 1234)),
        safe_thresh=safe_thresh,
    )

    # unscale to VAE latent space
    gen_unscaled = gen / scale_factor

    # save latent
    out_dir = args.out_dir or os.path.join(exp_dir, "inference_outputs")
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "gen_latent.npy"), gen_unscaled.detach().cpu().numpy()[0])

    if args.decode:
        vae, _ = load_vae(CONFIG["vae_config_path"], CONFIG["vae_ckpt_path"], device)
        with torch.no_grad():
            recon_gen = decode_tiled_klvae(vae, gen_unscaled)
        vol_gen = recon_gen[0, 0].cpu().float().numpy()

        if gt_latent is not None:
            gt_latent = gt_latent / scale_factor
            with torch.no_grad():
                recon_gt = decode_tiled_klvae(vae, gt_latent)
            vol_gt = recon_gt[0, 0].cpu().float().numpy()
        else:
            vol_gt = vol_gen.copy()

        # visualize condition in pixel by masking GT if available
        mask_pixel = None
        if args.raw_file is not None:
            raw = np.load(args.raw_file, mmap_mode="r").astype(np.float32)
            raw = (raw / 65535.0) * 2.0 - 1.0
            mask_pixel, _ = make_mask_pixel(raw.shape, CONFIG["axis"], float(CONFIG["ratio"]))
            mask_pixel = mask_pixel[0]  # (D,H,W)
        else:
            # upsample latent mask for visualization
            up = vol_gen.shape[0] // mask.shape[2]
            mask_pixel = F.interpolate(mask, scale_factor=up, mode="nearest")[0, 0].cpu().numpy()

        vol_cond = vol_gt.copy()
        vol_cond[mask_pixel == 0] = vol_gt.min()

        viz_path = os.path.join(out_dir, "inpaint_viz.png")
        visualize_slices(vol_gt, vol_cond, vol_gen, mask_pixel, viz_path, "Inpaint Result")
        print(f"Saved visualization: {viz_path}")


if __name__ == "__main__":
    main()

import os
import re
import json
import numpy as np
import torch
from tqdm import tqdm

from src.config import CONFIG
from src.model_unet3d import ConditionalLatentUNet
from src.diffusion import DiffusionHelper


def load_model(ckpt_path: str, device: torch.device):
    model = ConditionalLatentUNet(
        in_channels=CONFIG["in_channels"],
        out_channels=CONFIG["out_channels"],
        base_channels=CONFIG["base_channels"],
        channel_mults=CONFIG["channel_mults"],
        use_attention=CONFIG["use_attention"],
    ).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt.get("ema_state_dict", ckpt.get("model_state_dict", ckpt))
    model.load_state_dict(state)
    model.eval()
    return model


def ddim_sample(model, cond, mask, phi, porosity, diffusion: DiffusionHelper, steps=200, seed=1234, safe_thresh=8.0):
    model.eval()
    total_timesteps = diffusion.timesteps
    alphas_cumprod = diffusion.alphas_cumprod

    times = torch.linspace(0, total_timesteps - 1, steps=steps, device=cond.device)
    times = torch.unique(torch.round(times).long(), sorted=True)
    times = list(reversed(times.tolist()))

    if seed is not None:
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
        for i, t in enumerate(times):
            t_tensor = torch.full((cond.shape[0],), t, device=cond.device, dtype=torch.long)
            t_prev = times[i + 1] if i < len(times) - 1 else -1

            model_in = torch.cat([x, cond, mask, phi], dim=1)
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


def generate_volume(phi_map: np.ndarray, model, diffusion, patch_size: int, window_size: int, steps: int, seed: int):
    device = next(model.parameters()).device
    C = CONFIG["out_channels"]

    gD, gH, gW = phi_map.shape
    D, H, W = gD * patch_size, gH * patch_size, gW * patch_size
    z_full = np.zeros((C, D, H, W), dtype=np.float32)
    known_patch = np.zeros((gD, gH, gW), dtype=bool)

    w = window_size
    r = w // 2

    pad_p = r * patch_size
    phi_pad = np.pad(phi_map, ((r, r), (r, r), (r, r)), mode="constant")

    for i in range(gD):
        for j in range(gH):
            for k in range(gW):
                # build window
                z_pad = np.pad(z_full, ((0, 0), (pad_p, pad_p), (pad_p, pad_p), (pad_p, pad_p)), mode="constant")
                ci, cj, ck = i + r, j + r, k + r
                wi0, wi1 = ci - r, ci + r + 1
                wj0, wj1 = cj - r, cj + r + 1
                wk0, wk1 = ck - r, ck + r + 1

                phi_win = phi_pad[wi0:wi1, wj0:wj1, wk0:wk1]  # (w,w,w)
                zi0, zi1 = wi0 * patch_size, wi1 * patch_size
                zj0, zj1 = wj0 * patch_size, wj1 * patch_size
                zk0, zk1 = wk0 * patch_size, wk1 * patch_size
                z_win = z_pad[:, zi0:zi1, zj0:zj1, zk0:zk1]

                # known mask from generated patches
                mask_patch = np.zeros((w, w, w), dtype=np.float32)
                for di in range(w):
                    for dj in range(w):
                        for dk in range(w):
                            gi = i - r + di
                            gj = j - r + dj
                            gk = k - r + dk
                            if gi < 0 or gj < 0 or gk < 0 or gi >= gD or gj >= gH or gk >= gW:
                                continue
                            if known_patch[gi, gj, gk]:
                                mask_patch[di, dj, dk] = 1.0

                mask = np.repeat(mask_patch, patch_size, axis=0)
                mask = np.repeat(mask, patch_size, axis=1)
                mask = np.repeat(mask, patch_size, axis=2)[None, ...]
                cond = z_win * mask
                phi_vol = np.repeat(phi_win, patch_size, axis=0)
                phi_vol = np.repeat(phi_vol, patch_size, axis=1)
                phi_vol = np.repeat(phi_vol, patch_size, axis=2)[None, ...]

                if str(CONFIG.get("porosity_mode", "local")).lower() == "global":
                    porosity = np.array([float(phi_map.mean())], dtype=np.float32)
                else:
                    porosity = np.array([phi_map[i, j, k]], dtype=np.float32)

                # to torch
                cond_t = torch.from_numpy(cond).unsqueeze(0).to(device)
                mask_t = torch.from_numpy(mask).unsqueeze(0).to(device)
                phi_t = torch.from_numpy(phi_vol).unsqueeze(0).to(device)
                por_t = torch.from_numpy(porosity).unsqueeze(0).to(device)

                # sample
                x = ddim_sample(
                    model, cond_t, mask_t, phi_t, por_t,
                    diffusion, steps=steps, seed=seed, safe_thresh=CONFIG["safe_threshold"]
                )

                x_np = x.detach().cpu().float().numpy()[0]

                # write target patch back
                ti0 = (i * patch_size)
                tj0 = (j * patch_size)
                tk0 = (k * patch_size)
                ti1, tj1, tk1 = ti0 + patch_size, tj0 + patch_size, tk0 + patch_size

                # center patch in window
                c0 = r * patch_size
                c1 = c0 + patch_size
                z_full[:, ti0:ti1, tj0:tj1, tk0:tk1] = x_np[:, c0:c1, c0:c1, c0:c1]
                known_patch[i, j, k] = True

    return z_full


def main():
    device = torch.device(CONFIG["device"])
    ckpt = CONFIG.get("ckpt_path", "")
    if not ckpt or not os.path.exists(ckpt):
        raise FileNotFoundError("Please set CONFIG['ckpt_path'] to a valid checkpoint.")

    phi_path = CONFIG.get("phi_map_path", "")
    if not phi_path or not os.path.exists(phi_path):
        raise FileNotFoundError("Please set CONFIG['phi_map_path'] to a valid phi_map .npy")

    phi_map = np.load(phi_path).astype(np.float32)

    model = load_model(ckpt, device)
    diffusion = DiffusionHelper(CONFIG["timesteps"], device)

    z_full = generate_volume(
        phi_map, model, diffusion,
        patch_size=CONFIG["patch_size"],
        window_size=CONFIG["window_size"],
        steps=CONFIG["ddim_steps"],
        seed=CONFIG["seed"],
    )

    # unscale before saving (so VAE decode uses correct range)
    if bool(CONFIG.get("output_unscaled", True)):
        scale = float(CONFIG.get("scale_factor", 1.0))
        if scale != 0.0 and scale != 1.0:
            z_full = z_full / scale

    out_path = CONFIG.get("output_latent_path", "generated_latent.npy")
    np.save(out_path, z_full)
    print(f"Saved latent volume to {out_path}")


if __name__ == "__main__":
    main()

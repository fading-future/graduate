import os
import glob
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.config import CONFIG
from src.models.unet3d import UNet3D
from src.utils import get_root, build_porosity_map


def normalize_volume(vol: np.ndarray) -> np.ndarray:
    vol = vol.astype(np.float32)
    vol = vol / 65535.0
    vol = (vol * 2.0) - 1.0
    return vol


def make_mask(shape, axis: str, ratio: float, erosion: int):
    D, H, W = shape
    axis = axis.upper()
    if axis == "D":
        size = D
    elif axis == "H":
        size = H
    else:
        size = W
    cut = int(size * ratio)
    cut = max(1, min(cut, size - 1))
    cut = max(0, cut - erosion)

    mask = np.zeros((1, D, H, W), dtype=np.float32)
    if axis == "D":
        mask[:, :cut, :, :] = 1.0
    elif axis == "H":
        mask[:, :, :cut, :] = 1.0
    else:
        mask[:, :, :, :cut] = 1.0
    return mask


def load_model(model_dir: str, in_channels: int, out_channels: int, base_channels: int, channel_mults, use_attention, device):
    ckpts = sorted(glob.glob(os.path.join(model_dir, "unet_epoch_*.pth")), key=os.path.getmtime)
    if not ckpts:
        latest = os.path.join(model_dir, "unet_latest.pth")
        if os.path.exists(latest):
            ckpts = [latest]
        else:
            raise FileNotFoundError(f"No checkpoints found in {model_dir}")
    ckpt_path = ckpts[-1]

    model = UNet3D(
        in_channels=in_channels,
        out_channels=out_channels,
        base_channels=base_channels,
        channel_mults=channel_mults,
        use_attention=use_attention,
    ).to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def inference_coarse(model, vol, mask, por, coarse_size):
    # vol: (1,1,256,256,256)
    # mask: (1,1,256,256,256)
    cond = vol * mask
    # downsample
    cond_c = F.interpolate(cond, size=(coarse_size,)*3, mode="trilinear", align_corners=False)
    mask_c = F.interpolate(mask, size=(coarse_size,)*3, mode="nearest")

    inp = torch.cat([cond_c, mask_c], dim=1)
    with torch.no_grad():
        pred_c = model(inp, por)

    # upsample to full
    pred_full = F.interpolate(pred_c, size=vol.shape[-3:], mode="trilinear", align_corners=False)
    return pred_full


def inference_refine(model, vol, mask, coarse_full, por, patch_size, overlap):
    # sliding window with weighted blending
    device = vol.device
    _, _, D, H, W = vol.shape
    step = patch_size - overlap
    out = torch.zeros_like(vol)
    wgt = torch.zeros_like(vol)

    # triangular window
    def tri(n):
        x = torch.linspace(0, 1, n, device=device)
        w = 1.0 - (2.0 * (x - 0.5)).abs()
        return w.clamp_min(0.0)

    wz = tri(patch_size).view(1,1,patch_size,1,1)
    wy = tri(patch_size).view(1,1,1,patch_size,1)
    wx = tri(patch_size).view(1,1,1,1,patch_size)
    ww = wz * wy * wx

    for z in range(0, D, step):
        for y in range(0, H, step):
            for x in range(0, W, step):
                z1 = min(z + patch_size, D)
                y1 = min(y + patch_size, H)
                x1 = min(x + patch_size, W)
                z0 = max(0, z1 - patch_size)
                y0 = max(0, y1 - patch_size)
                x0 = max(0, x1 - patch_size)

                gt_patch = vol[:, :, z0:z1, y0:y1, x0:x1]
                mask_patch = mask[:, :, z0:z1, y0:y1, x0:x1]
                cond_patch = gt_patch * mask_patch
                coarse_patch = coarse_full[:, :, z0:z1, y0:y1, x0:x1]

                inp = torch.cat([cond_patch, mask_patch, coarse_patch], dim=1)
                with torch.no_grad():
                    pred = model(inp, por)

                out[:, :, z0:z1, y0:y1, x0:x1] += pred * ww
                wgt[:, :, z0:z1, y0:y1, x0:x1] += ww

    out = out / (wgt + 1e-8)
    return out


def visualize(vol_gt, vol_cond, vol_gen, mask, save_path):
    flat = vol_gt.flatten()
    vmin, vmax = np.percentile(flat, 1), np.percentile(flat, 99)
    D, H, W = vol_gt.shape
    cz, cy, cx = D//2, H//2, W//2

    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    cols = ["GT", "Condition", "Generated"]
    vols = [vol_gt, vol_cond, vol_gen]

    def draw_line(ax):
        z_profile = mask.mean(axis=(1,2))
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
    parser.add_argument("--raw_file", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default=None)
    args = parser.parse_args()

    device = torch.device(CONFIG["device"])
    root = get_root()

    # load models
    coarse_model_dir = os.path.join(root, "exp_results", "coarse", "models")
    refine_model_dir = os.path.join(root, "exp_results", "refine", "models")

    coarse_model = load_model(
        coarse_model_dir,
        in_channels=2,
        out_channels=1,
        base_channels=CONFIG["COARSE"]["model_channels"],
        channel_mults=CONFIG["COARSE"]["channel_mults"],
        use_attention=CONFIG["COARSE"]["use_attention"],
        device=device,
    )

    refine_model = load_model(
        refine_model_dir,
        in_channels=3,
        out_channels=1,
        base_channels=CONFIG["REFINE"]["model_channels"],
        channel_mults=CONFIG["REFINE"]["channel_mults"],
        use_attention=CONFIG["REFINE"]["use_attention"],
        device=device,
    )

    # load data
    raw = np.load(args.raw_file, mmap_mode="r")
    vol = normalize_volume(raw)

    mask = make_mask(vol.shape, CONFIG["TASK"]["axis"], CONFIG["TASK"]["ratio"], CONFIG["TASK"]["erosion_px"])

    por_map = build_porosity_map(CONFIG["PATHS"]["porosity_csv"])
    base = os.path.basename(args.raw_file)
    por = por_map.get(base, 0.15)

    vol_t = torch.from_numpy(vol).unsqueeze(0).unsqueeze(0).float().to(device)
    mask_t = torch.from_numpy(mask).unsqueeze(0).float().to(device)
    por_t = torch.tensor([por], dtype=torch.float32, device=device)

    # coarse
    coarse_full = inference_coarse(coarse_model, vol_t, mask_t, por_t, CONFIG["COARSE"]["coarse_size"])

    # refine
    refine_full = inference_refine(
        refine_model,
        vol_t,
        mask_t,
        coarse_full,
        por_t,
        CONFIG["REFINE"]["patch_size"],
        CONFIG["REFINE"]["patch_overlap"],
    )

    # residual mode: refine predicts delta on top of coarse
    if CONFIG["REFINE"].get("residual_pred", False):
        refine_full = refine_full + coarse_full

    # output
    out_dir = args.out_dir or os.path.join(root, "exp_results", "inference_outputs")
    os.makedirs(out_dir, exist_ok=True)

    coarse_np = coarse_full[0,0].detach().cpu().numpy()
    refine_np = refine_full[0,0].detach().cpu().numpy()
    np.save(os.path.join(out_dir, "coarse_pred.npy"), coarse_np)
    np.save(os.path.join(out_dir, "refine_pred.npy"), refine_np)

    # visualization
    vol_gt = vol
    cond = vol * mask[0]
    vol_cond = cond.copy()
    vol_cond[mask[0] == 0] = vol_gt.min()

    viz_path = os.path.join(out_dir, "multiscale_inpaint_viz.png")
    visualize(vol_gt, vol_cond, refine_np, mask[0], viz_path)
    print(f"Saved: {viz_path}")

    # log inference
    log_path = os.path.join(out_dir, "inference_log.csv")
    is_new = not os.path.exists(log_path)
    with open(log_path, "a", encoding="utf-8") as f:
        if is_new:
            f.write("file,porosity,coarse_pred,refine_pred,vis,mae_unknown,mae_known\n")
        # compute MAE for quick sanity
        mask_np = mask[0]
        unknown = (mask_np == 0)
        known = (mask_np == 1)
        mae_unknown = float(np.mean(np.abs(refine_np[unknown] - coarse_np[unknown]))) if unknown.any() else 0.0
        mae_known = float(np.mean(np.abs(refine_np[known] - coarse_np[known]))) if known.any() else 0.0
        f.write(
            f"{os.path.basename(args.raw_file)},{por},"
            f"{os.path.join(out_dir,'coarse_pred.npy')},"
            f"{os.path.join(out_dir,'refine_pred.npy')},"
            f"{viz_path},{mae_unknown:.6f},{mae_known:.6f}\n"
        )


if __name__ == "__main__":
    main()

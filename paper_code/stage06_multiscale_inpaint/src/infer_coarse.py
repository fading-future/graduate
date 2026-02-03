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


def load_model(model_dir: str, device):
    ckpts = sorted(glob.glob(os.path.join(model_dir, "unet_epoch_*.pth")), key=os.path.getmtime)
    if not ckpts:
        latest = os.path.join(model_dir, "unet_latest.pth")
        if os.path.exists(latest):
            ckpts = [latest]
        else:
            raise FileNotFoundError(f"No checkpoints found in {model_dir}")
    ckpt_path = ckpts[-1]

    c_cfg = CONFIG["COARSE"]
    model = UNet3D(
        in_channels=2,
        out_channels=1,
        base_channels=c_cfg["model_channels"],
        channel_mults=c_cfg["channel_mults"],
        use_attention=c_cfg["use_attention"],
    ).to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def inference_coarse(model, vol, mask, por, coarse_size):
    cond = vol * mask
    cond_c = F.interpolate(cond, size=(coarse_size,)*3, mode="trilinear", align_corners=False)
    mask_c = F.interpolate(mask, size=(coarse_size,)*3, mode="nearest")

    inp = torch.cat([cond_c, mask_c], dim=1)
    with torch.no_grad():
        pred_c = model(inp, por)

    pred_full = F.interpolate(pred_c, size=vol.shape[-3:], mode="trilinear", align_corners=False)
    return pred_full


def visualize(vol_gt, vol_cond, vol_gen, mask, save_path):
    flat = vol_gt.flatten()
    vmin, vmax = np.percentile(flat, 1), np.percentile(flat, 99)
    D, H, W = vol_gt.shape
    cz, cy, cx = D//2, H//2, W//2

    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    cols = ["GT", "Condition", "Coarse"]
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

    coarse_model_dir = os.path.join(root, "exp_results", "coarse", "models")
    model = load_model(coarse_model_dir, device)

    raw = np.load(args.raw_file, mmap_mode="r")
    vol = normalize_volume(raw)

    mask = make_mask(vol.shape, CONFIG["TASK"]["axis"], CONFIG["TASK"]["ratio"], CONFIG["TASK"]["erosion_px"])

    por_map = build_porosity_map(CONFIG["PATHS"]["porosity_csv"])
    base = os.path.basename(args.raw_file)
    por = por_map.get(base, 0.15)

    vol_t = torch.from_numpy(vol).unsqueeze(0).unsqueeze(0).float().to(device)
    mask_t = torch.from_numpy(mask).unsqueeze(0).float().to(device)
    por_t = torch.tensor([por], dtype=torch.float32, device=device)

    coarse_full = inference_coarse(model, vol_t, mask_t, por_t, CONFIG["COARSE"]["coarse_size"])

    out_dir = args.out_dir or os.path.join(root, "exp_results", "inference_outputs")
    os.makedirs(out_dir, exist_ok=True)

    np.save(os.path.join(out_dir, "coarse_pred.npy"), coarse_full[0,0].detach().cpu().numpy())

    vol_gt = vol
    cond = vol * mask[0]
    vol_cond = cond.copy()
    vol_cond[mask[0] == 0] = vol_gt.min()

    viz_path = os.path.join(out_dir, "coarse_inpaint_viz.png")
    visualize(vol_gt, vol_cond, coarse_full[0,0].detach().cpu().numpy(), mask[0], viz_path)
    print(f"Saved: {viz_path}")

    # log inference
    log_path = os.path.join(out_dir, "inference_log.csv")
    is_new = not os.path.exists(log_path)
    with open(log_path, "a", encoding="utf-8") as f:
        if is_new:
            f.write("file,porosity,coarse_pred,vis\n")
        f.write(f"{os.path.basename(args.raw_file)},{por},{os.path.join(out_dir,'coarse_pred.npy')},{viz_path}\n")


if __name__ == "__main__":
    main()

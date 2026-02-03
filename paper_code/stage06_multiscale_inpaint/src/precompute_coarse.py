import os
import glob
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.config import CONFIG
from src.models.unet3d import UNet3D
from src.utils import build_porosity_map, get_root


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, required=True)
    args = parser.parse_args()

    device = torch.device(CONFIG["device"])
    model_dir = os.path.join(get_root(), "exp_results", "coarse", "models")
    model = load_model(model_dir, device)

    por_map = build_porosity_map(CONFIG["PATHS"]["porosity_csv"])
    c_cfg = CONFIG["COARSE"]
    files = sorted(glob.glob(os.path.join(CONFIG["PATHS"]["raw_data_dir"], "*.npy")))

    os.makedirs(args.out_dir, exist_ok=True)

    for fp in tqdm(files, desc="Coarse Cache"):
        base = os.path.basename(fp)
        out_path = os.path.join(args.out_dir, base.replace('.npy', '_coarse.npy'))
        if os.path.exists(out_path):
            continue

        raw = np.load(fp, mmap_mode="r")
        vol = normalize_volume(raw)
        mask = make_mask(vol.shape, CONFIG["TASK"]["axis"], CONFIG["TASK"]["ratio"], CONFIG["TASK"]["erosion_px"])

        por = por_map.get(base, 0.15)

        vol_t = torch.from_numpy(vol).unsqueeze(0).unsqueeze(0).float().to(device)
        mask_t = torch.from_numpy(mask).unsqueeze(0).float().to(device)
        por_t = torch.tensor([por], dtype=torch.float32, device=device)

        # downsample cond
        cond = vol_t * mask_t
        cond_c = F.interpolate(cond, size=(c_cfg["coarse_size"],)*3, mode="trilinear", align_corners=False)
        mask_c = F.interpolate(mask_t, size=(c_cfg["coarse_size"],)*3, mode="nearest")
        inp = torch.cat([cond_c, mask_c], dim=1)

        with torch.no_grad():
            pred_c = model(inp, por_t)

        pred_full = F.interpolate(pred_c, size=vol.shape, mode="trilinear", align_corners=False)
        np.save(out_path, pred_full[0,0].cpu().numpy())


if __name__ == "__main__":
    main()

import argparse
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from src.config import CONFIG
from model.vae import KLVAE3D


def safe_torch_load(path: str, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)
    except Exception:
        return torch.load(path, map_location=map_location)


def save_three_slices(volume: np.ndarray, out_path: str, title: str):
    d, h, w = volume.shape
    cz, cy, cx = d // 2, h // 2, w // 2
    slices = [volume[cz], volume[:, cy, :], volume[:, :, cx]]

    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    for i, ax in enumerate(axes):
        ax.imshow(slices[i], cmap="gray")
        ax.axis("off")
    fig.suptitle(title, fontsize=10)
    plt.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def load_latent(path: str) -> np.ndarray:
    z = np.load(path).astype(np.float32)
    if z.ndim == 5 and z.shape[0] == 1:
        z = z[0]
    if z.ndim != 4:
        raise ValueError(f"Expected latent shape (C,D,H,W) or (1,C,D,H,W), got {z.shape}.")
    return z


def load_vae(vae_cfg_path: str, vae_ckpt_path: str, device: torch.device) -> KLVAE3D:
    with open(vae_cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    vae = KLVAE3D(cfg).to(device).eval()
    ckpt = safe_torch_load(vae_ckpt_path, map_location=device)
    if isinstance(ckpt, dict) and "vae_state_dict" in ckpt:
        state = ckpt["vae_state_dict"]
    elif isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
    else:
        state = ckpt
    state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
    vae.load_state_dict(state)
    return vae


def main():
    default_latent_scaled = not bool(CONFIG.get("output_unscaled", True))

    parser = argparse.ArgumentParser(
        description="Visualize generated latent: decode with KLVAE and save prob/bin previews."
    )
    parser.add_argument(
        "--latent-path",
        default=str(Path(__file__).resolve().parents[1] / "generated_latent.npy"),
        help="Path to latent .npy, shape (C,D,H,W).",
    )
    parser.add_argument(
        "--vae-config",
        default=CONFIG.get("eval_vae_config_path", ""),
        help="Path to stage02 VAE yaml config.",
    )
    parser.add_argument(
        "--vae-ckpt",
        default=CONFIG.get("eval_vae_ckpt_path", ""),
        help="Path to stage02 VAE checkpoint.",
    )
    parser.add_argument(
        "--out-dir",
        default="",
        help="Output directory. Default: same folder as latent file.",
    )
    parser.add_argument(
        "--scale-factor",
        type=float,
        default=float(CONFIG.get("scale_factor", 1.0)),
        help="Stage07 latent scale_factor.",
    )
    parser.add_argument(
        "--latent-is-scaled",
        action="store_true",
        default=default_latent_scaled,
        help="Set if latent is still scaled by scale_factor.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Binarization threshold applied on sigmoid(prob).",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="cuda or cpu.",
    )
    args = parser.parse_args()

    latent_path = os.path.abspath(args.latent_path)
    out_dir = os.path.abspath(args.out_dir) if args.out_dir else os.path.dirname(latent_path)
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.exists(latent_path):
        raise FileNotFoundError(f"latent file not found: {latent_path}")
    if not os.path.exists(args.vae_config):
        raise FileNotFoundError(f"vae config not found: {args.vae_config}")
    if not os.path.exists(args.vae_ckpt):
        raise FileNotFoundError(f"vae checkpoint not found: {args.vae_ckpt}")

    device = torch.device(args.device)

    z = load_latent(latent_path)
    print(
        f"[latent] shape={z.shape}, dtype={z.dtype}, "
        f"min={z.min():.6f}, max={z.max():.6f}, mean={z.mean():.6f}, std={z.std():.6f}"
    )

    latent_ch0_png = os.path.join(out_dir, "latent_ch0_slices.png")
    save_three_slices(z[0], latent_ch0_png, "latent channel 0")

    z_t = torch.from_numpy(z).unsqueeze(0).to(device)
    if args.latent_is_scaled:
        sf = args.scale_factor if args.scale_factor != 0.0 else 1.0
        z_t = z_t / sf
        print(f"[info] latent treated as scaled, divided by scale_factor={sf:.6f} before decode")
    else:
        print("[info] latent treated as unscaled, decode directly")

    vae = load_vae(args.vae_config, args.vae_ckpt, device)
    with torch.no_grad():
        if device.type == "cuda":
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits = vae.decode(z_t)
        else:
            logits = vae.decode(z_t)

    if logits.ndim != 5:
        raise ValueError(f"Unexpected decoded tensor shape: {tuple(logits.shape)}")

    prob = torch.sigmoid(logits)[0, 0].detach().cpu().float().numpy()
    binv = (prob >= float(args.threshold)).astype(np.uint8)

    prob_npy = os.path.join(out_dir, "generated_voxel_prob.npy")
    bin_npy = os.path.join(out_dir, "generated_voxel_bin.npy")
    prob_png = os.path.join(out_dir, "generated_voxel_prob_slices.png")
    bin_png = os.path.join(out_dir, "generated_voxel_bin_slices.png")

    np.save(prob_npy, prob)
    np.save(bin_npy, binv)
    save_three_slices(prob, prob_png, "decoded probability")
    save_three_slices(binv, bin_png, "decoded binary")

    print(
        f"[decoded] shape={prob.shape}, prob_min={prob.min():.6f}, prob_max={prob.max():.6f}, "
        f"prob_mean={prob.mean():.6f}, bin_mean={binv.mean():.6f}"
    )
    print("[saved]")
    print(f"  {latent_ch0_png}")
    print(f"  {prob_npy}")
    print(f"  {bin_npy}")
    print(f"  {prob_png}")
    print(f"  {bin_png}")


if __name__ == "__main__":
    main()

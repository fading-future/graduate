import argparse
import json
import os
from pathlib import Path
from typing import Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from model.vae import KLVAE3D
from src.config import CONFIG


def safe_torch_load(path: str, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)
    except Exception:
        return torch.load(path, map_location=map_location)


def resolve_path(path_str: str, repo_root: Path) -> str:
    if not path_str:
        return ""
    p = Path(path_str)
    if p.is_absolute():
        return str(p)
    p_cwd = Path.cwd() / p
    if p_cwd.exists():
        return str(p_cwd.resolve())
    p_repo = repo_root / p
    if p_repo.exists():
        return str(p_repo.resolve())
    return str(p_cwd.resolve())


def strip_porosity_prefix(name: str) -> str:
    # support: porosity_0.123456_xxx.npy -> xxx.npy
    import re
    return re.sub(r"^porosity_[0-9]*\.?[0-9]+_", "", name)


def find_matching_file(base_name: str, search_dir: str) -> str:
    if not search_dir:
        return ""
    if not os.path.isdir(search_dir):
        return ""
    exact = os.path.join(search_dir, base_name)
    if os.path.exists(exact):
        return exact
    # fallback: allow porosity prefix
    matches = [
        os.path.join(search_dir, f)
        for f in os.listdir(search_dir)
        if f.endswith(".npy") and strip_porosity_prefix(f) == base_name
    ]
    if len(matches) == 0:
        return ""
    matches.sort()
    return matches[0]


def load_latent(path: str) -> np.ndarray:
    z = np.load(path).astype(np.float32)
    if z.ndim == 5 and z.shape[0] == 1:
        z = z[0]
    if z.ndim != 4:
        raise ValueError(f"Expected latent shape (C,D,H,W) or (1,C,D,H,W), got {z.shape}")
    return z


def to_binary_volume(vol: np.ndarray) -> np.ndarray:
    arr = vol.astype(np.float32)
    mn, mx = float(arr.min()), float(arr.max())
    if mn >= -1.01 and mx <= 1.01:
        if mn < 0.0:
            arr = (arr + 1.0) * 0.5
        return (arr >= 0.5).astype(np.uint8)
    if mx <= 1.5:
        return (arr >= 0.5).astype(np.uint8)
    if mx <= 255.5:
        return (arr >= 127.5).astype(np.uint8)
    return (arr >= 32767.5).astype(np.uint8)


def center_crop_to_shape(arr: np.ndarray, shape: Tuple[int, ...]) -> np.ndarray:
    if arr.shape == shape:
        return arr
    if len(arr.shape) != len(shape):
        raise ValueError(f"rank mismatch: arr {arr.shape} vs target {shape}")
    slices = []
    for dim, target in zip(arr.shape, shape):
        if target > dim:
            raise ValueError(f"cannot crop dim {dim} to larger target {target}")
        start = (dim - target) // 2
        slices.append(slice(start, start + target))
    return arr[tuple(slices)]


def align_pair(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    target = tuple(min(sa, sb) for sa, sb in zip(a.shape, b.shape))
    return center_crop_to_shape(a, target), center_crop_to_shape(b, target)


def latent_metrics(pred: np.ndarray, gt: np.ndarray) -> dict:
    diff = pred - gt
    return {
        "latent_mae": float(np.mean(np.abs(diff))),
        "latent_mse": float(np.mean(diff ** 2)),
        "latent_rmse": float(np.sqrt(np.mean(diff ** 2))),
    }


def voxel_metrics(pred_bin: np.ndarray, gt_bin: np.ndarray) -> dict:
    pred = pred_bin.astype(bool)
    gt = gt_bin.astype(bool)
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    pred_sum = pred.sum()
    gt_sum = gt.sum()
    eps = 1e-8
    return {
        "voxel_dice": float((2.0 * inter) / (pred_sum + gt_sum + eps)),
        "voxel_iou": float(inter / (union + eps)),
        "voxel_precision": float(inter / (pred_sum + eps)),
        "voxel_recall": float(inter / (gt_sum + eps)),
        "porosity_pred": float(pred.mean()),
        "porosity_gt": float(gt.mean()),
        "porosity_abs_err": float(abs(pred.mean() - gt.mean())),
    }


def corrcoef_safe(a: np.ndarray, b: np.ndarray) -> float:
    aa = a.reshape(-1).astype(np.float64)
    bb = b.reshape(-1).astype(np.float64)
    if aa.std() < 1e-12 or bb.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(aa, bb)[0, 1])


def phi_metrics(pred_phi: np.ndarray, gt_phi: np.ndarray, prefix: str) -> dict:
    diff = pred_phi - gt_phi
    out = {
        f"{prefix}_phi_mae": float(np.mean(np.abs(diff))),
        f"{prefix}_phi_mse": float(np.mean(diff ** 2)),
        f"{prefix}_phi_rmse": float(np.sqrt(np.mean(diff ** 2))),
        f"{prefix}_phi_corr": corrcoef_safe(pred_phi, gt_phi),
    }
    return out


def compute_phi_map(vol: np.ndarray, patch_voxel: int) -> np.ndarray:
    # vol: (D,H,W) binary/probability in [0,1]
    d, h, w = vol.shape
    g_d, g_h, g_w = d // patch_voxel, h // patch_voxel, w // patch_voxel
    vol = vol[: g_d * patch_voxel, : g_h * patch_voxel, : g_w * patch_voxel]
    vol = vol.reshape(g_d, patch_voxel, g_h, patch_voxel, g_w, patch_voxel)
    phi = vol.mean(axis=(1, 3, 5)).astype(np.float32)
    return phi


def save_slices_single(vol3d: np.ndarray, out_path: str, title: str, cmap: str = "gray", vmin=None, vmax=None):
    d, h, w = vol3d.shape
    cz, cy, cx = d // 2, h // 2, w // 2
    slices = [vol3d[cz], vol3d[:, cy, :], vol3d[:, :, cx]]
    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    for i, ax in enumerate(axes):
        ax.imshow(slices[i], cmap=cmap, vmin=vmin, vmax=vmax)
        ax.axis("off")
    fig.suptitle(title, fontsize=10)
    plt.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def save_slices_pair(pred3d: np.ndarray, gt3d: np.ndarray, out_path: str, title: str, cmap: str = "gray", vmin=None, vmax=None):
    d, h, w = pred3d.shape
    cz, cy, cx = d // 2, h // 2, w // 2
    pred_slices = [pred3d[cz], pred3d[:, cy, :], pred3d[:, :, cx]]
    gt_slices = [gt3d[cz], gt3d[:, cy, :], gt3d[:, :, cx]]

    fig, axes = plt.subplots(2, 3, figsize=(9, 6))
    for i in range(3):
        axes[0, i].imshow(pred_slices[i], cmap=cmap, vmin=vmin, vmax=vmax)
        axes[0, i].axis("off")
        axes[1, i].imshow(gt_slices[i], cmap=cmap, vmin=vmin, vmax=vmax)
        axes[1, i].axis("off")
    axes[0, 0].set_title("Pred", fontsize=10)
    axes[1, 0].set_title("GT", fontsize=10)
    fig.suptitle(title, fontsize=10)
    plt.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def load_vae(vae_cfg: str, vae_ckpt: str, device: torch.device) -> KLVAE3D:
    with open(vae_cfg, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    vae = KLVAE3D(cfg).to(device).eval()
    ckpt = safe_torch_load(vae_ckpt, map_location=device)
    if isinstance(ckpt, dict) and "vae_state_dict" in ckpt:
        state = ckpt["vae_state_dict"]
    elif isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
    else:
        state = ckpt
    state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
    vae.load_state_dict(state)
    return vae


def decode_to_prob_and_bin(z_unscaled: np.ndarray, vae: KLVAE3D, device: torch.device, threshold: float) -> Tuple[np.ndarray, np.ndarray]:
    z_t = torch.from_numpy(z_unscaled).unsqueeze(0).to(device)
    with torch.no_grad():
        if device.type == "cuda":
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits = vae.decode(z_t)
        else:
            logits = vae.decode(z_t)
    prob = torch.sigmoid(logits)[0, 0].detach().cpu().float().numpy()
    binv = (prob >= threshold).astype(np.uint8)
    return prob, binv


def main():
    repo_root = Path(__file__).resolve().parents[1]
    default_pred_scaled = not bool(CONFIG.get("output_unscaled", True))

    NPY_NAME = CONFIG.get("phi_map_path", "").split("\\")[-1].split(".npy")[0]
    EPOCH = CONFIG.get("ckpt_path", "unknown").split("_epoch_")[-1].split(".pth")[0]
    print(f"Evaluating sample {NPY_NAME} from epoch {EPOCH}...")

    parser = argparse.ArgumentParser(description="Evaluate one generated latent sample against GT (if available).")
    parser.add_argument("--pred-latent", default=CONFIG.get("output_latent_path", "generated_latent.npy"))
    parser.add_argument("--phi-map", default=CONFIG.get("phi_map_path", ""))
    parser.add_argument("--gt-latent", default=f"D:\多尺度岩心数据集\LDM_Data\Latent_NPY\w192_s64\{NPY_NAME}.npy")
    parser.add_argument("--gt-voxel", default=f"D:\多尺度岩心数据集\LDM_Data\Raw_NPY\w192_s64\{NPY_NAME}.npy")
    parser.add_argument("--gt-phi", default=f"D:\多尺度岩心数据集\LDM_Data\Phi_Maps_NPY\w192_s64\{NPY_NAME}.npy")
    parser.add_argument("--latent-dir", default=CONFIG.get("latent_dir", ""))
    parser.add_argument("--raw-dir", default=CONFIG.get("raw_data_dir", ""))
    parser.add_argument("--vae-config", default=CONFIG.get("eval_vae_config_path", ""))
    parser.add_argument("--vae-ckpt", default=CONFIG.get("eval_vae_ckpt_path", ""))
    parser.add_argument("--scale-factor", type=float, default=float(CONFIG.get("scale_factor", 1.0)))
    parser.add_argument("--pred-latent-scaled", action="store_true", default=default_pred_scaled)
    parser.add_argument("--gt-latent-scaled", action="store_true", default=False)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out-dir", default=str(repo_root / f"eval_generated_{EPOCH}_{NPY_NAME}"))
    args = parser.parse_args()

    out_dir = resolve_path(args.out_dir, repo_root)
    os.makedirs(out_dir, exist_ok=True)

    pred_latent_path = resolve_path(args.pred_latent, repo_root)
    phi_path = resolve_path(args.phi_map, repo_root)
    latent_dir = resolve_path(args.latent_dir, repo_root)
    raw_dir = resolve_path(args.raw_dir, repo_root)
    gt_latent_path = resolve_path(args.gt_latent, repo_root) if args.gt_latent else ""
    gt_voxel_path = resolve_path(args.gt_voxel, repo_root) if args.gt_voxel else ""
    gt_phi_path = resolve_path(args.gt_phi, repo_root) if args.gt_phi else ""
    vae_cfg = resolve_path(args.vae_config, repo_root)
    vae_ckpt = resolve_path(args.vae_ckpt, repo_root)

    if not os.path.exists(pred_latent_path):
        raise FileNotFoundError(f"pred latent not found: {pred_latent_path}")
    if not os.path.exists(vae_cfg):
        raise FileNotFoundError(f"vae config not found: {vae_cfg}")
    if not os.path.exists(vae_ckpt):
        raise FileNotFoundError(f"vae checkpoint not found: {vae_ckpt}")

    base_name = ""
    if phi_path and os.path.exists(phi_path):
        base_name = os.path.basename(phi_path)
    else:
        base_name = strip_porosity_prefix(os.path.basename(pred_latent_path))

    if not gt_latent_path:
        gt_latent_path = find_matching_file(base_name, latent_dir)
    if not gt_voxel_path:
        candidate = os.path.join(raw_dir, base_name) if raw_dir else ""
        gt_voxel_path = candidate if candidate and os.path.exists(candidate) else ""
    if not gt_phi_path and phi_path and os.path.exists(phi_path):
        gt_phi_path = phi_path

    device = torch.device(args.device)
    sf = args.scale_factor if args.scale_factor != 0.0 else 1.0

    pred_lat = load_latent(pred_latent_path)
    pred_lat_unscaled = pred_lat / sf if args.pred_latent_scaled else pred_lat

    metrics = {
        "pred_latent_path": pred_latent_path,
        "gt_latent_path": gt_latent_path,
        "gt_voxel_path": gt_voxel_path,
        "gt_phi_path": gt_phi_path,
        "pred_shape": list(pred_lat.shape),
        "pred_min": float(pred_lat.min()),
        "pred_max": float(pred_lat.max()),
        "pred_mean": float(pred_lat.mean()),
        "pred_std": float(pred_lat.std()),
    }

    # latent comparison
    gt_lat = None
    if gt_latent_path and os.path.exists(gt_latent_path):
        gt_lat = load_latent(gt_latent_path)
        gt_lat_unscaled = gt_lat / sf if args.gt_latent_scaled else gt_lat
        pred_lat_cmp, gt_lat_cmp = align_pair(pred_lat_unscaled, gt_lat_unscaled)
        metrics.update(latent_metrics(pred_lat_cmp, gt_lat_cmp))
        metrics["latent_eval_shape"] = list(pred_lat_cmp.shape)
        save_slices_pair(
            pred_lat_cmp[0],
            gt_lat_cmp[0],
            os.path.join(out_dir, "latent_ch0_pred_vs_gt.png"),
            "Latent channel 0 (pred vs gt)",
            cmap="gray",
        )
    else:
        save_slices_single(
            pred_lat_unscaled[0],
            os.path.join(out_dir, "latent_ch0_pred.png"),
            "Latent channel 0 (pred)",
            cmap="gray",
        )

    # decode predicted latent
    vae = load_vae(vae_cfg, vae_ckpt, device)
    pred_prob, pred_bin = decode_to_prob_and_bin(pred_lat_unscaled, vae, device, args.threshold)
    np.save(os.path.join(out_dir, "pred_voxel_prob.npy"), pred_prob)
    np.save(os.path.join(out_dir, "pred_voxel_bin.npy"), pred_bin)
    save_slices_single(pred_prob, os.path.join(out_dir, "pred_voxel_prob.png"), "Pred voxel probability", cmap="gray", vmin=0.0, vmax=1.0)
    save_slices_single(pred_bin, os.path.join(out_dir, "pred_voxel_bin.png"), "Pred voxel binary", cmap="gray", vmin=0.0, vmax=1.0)

    # voxel GT metrics
    gt_bin = None
    if gt_voxel_path and os.path.exists(gt_voxel_path):
        gt_vol = np.load(gt_voxel_path)
        gt_bin = to_binary_volume(gt_vol)
        pred_bin_al, gt_bin_al = align_pair(pred_bin, gt_bin)
        pred_prob_al, _ = align_pair(pred_prob, gt_bin.astype(np.float32))
        metrics.update(voxel_metrics(pred_bin_al, gt_bin_al))
        save_slices_pair(
            pred_bin_al,
            gt_bin_al,
            os.path.join(out_dir, "voxel_bin_pred_vs_gt.png"),
            "Voxel binary (pred vs gt)",
            cmap="gray",
            vmin=0.0,
            vmax=1.0,
        )
        save_slices_pair(
            pred_prob_al,
            gt_bin_al.astype(np.float32),
            os.path.join(out_dir, "voxel_prob_pred_vs_gt.png"),
            "Voxel prob (pred) vs binary gt",
            cmap="gray",
            vmin=0.0,
            vmax=1.0,
        )

    # phi consistency
    patch_voxel = int(CONFIG.get("patch_size", 8)) * int(CONFIG.get("downsample_factor", 8))
    pred_phi_bin = compute_phi_map(pred_bin.astype(np.float32), patch_voxel)
    pred_phi_prob = compute_phi_map(pred_prob.astype(np.float32), patch_voxel)
    np.save(os.path.join(out_dir, "pred_phi_from_bin.npy"), pred_phi_bin)
    np.save(os.path.join(out_dir, "pred_phi_from_prob.npy"), pred_phi_prob)

    gt_phi = None
    if gt_phi_path and os.path.exists(gt_phi_path):
        gt_phi = np.load(gt_phi_path).astype(np.float32)
    elif gt_bin is not None:
        gt_phi = compute_phi_map(gt_bin.astype(np.float32), patch_voxel)

    if gt_phi is not None:
        pred_phi_bin_al, gt_phi_al = align_pair(pred_phi_bin, gt_phi)
        pred_phi_prob_al, _ = align_pair(pred_phi_prob, gt_phi)
        metrics.update(phi_metrics(pred_phi_bin_al, gt_phi_al, prefix="bin"))
        metrics.update(phi_metrics(pred_phi_prob_al, gt_phi_al, prefix="prob"))
        save_slices_pair(
            pred_phi_bin_al,
            gt_phi_al,
            os.path.join(out_dir, "phi_from_bin_pred_vs_gt.png"),
            "Phi map from binary (pred vs gt)",
            cmap="viridis",
        )
        save_slices_pair(
            pred_phi_prob_al,
            gt_phi_al,
            os.path.join(out_dir, "phi_from_prob_pred_vs_gt.png"),
            "Phi map from probability (pred vs gt)",
            cmap="viridis",
        )
    else:
        save_slices_single(
            pred_phi_bin,
            os.path.join(out_dir, "phi_from_bin_pred.png"),
            "Pred phi from binary",
            cmap="viridis",
        )
        save_slices_single(
            pred_phi_prob,
            os.path.join(out_dir, "phi_from_prob_pred.png"),
            "Pred phi from probability",
            cmap="viridis",
        )

    metrics_path = os.path.join(out_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print("[done] evaluation finished")
    print(f"  metrics: {metrics_path}")
    for k in sorted(metrics.keys()):
        v = metrics[k]
        if isinstance(v, float):
            print(f"  {k}: {v:.6f}")
        else:
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()

import argparse
import os
from pathlib import Path
import sys
from typing import List, Optional, Tuple

import numpy as np
import torch
import yaml

_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from model.vae import KLVAE3D
from src.config import CONFIG
from src.diffusion import DiffusionHelper
from src.infer import generate_volume, load_model
from src.preprocess_phi import center_crop_cube, compute_phi_map, normalize_to_unit, otsu_threshold


def _safe_torch_load(path: str, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)
    except Exception:
        return torch.load(path, map_location=map_location)


def _resolve_path(path: str, repo_root: Path) -> str:
    if not path:
        return ""
    if os.path.isabs(path):
        return path
    return str((repo_root / path).resolve())


def load_vae(vae_cfg: str, vae_ckpt: str, device: torch.device) -> KLVAE3D:
    with open(vae_cfg, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    vae = KLVAE3D(cfg).to(device).eval()
    ckpt = _safe_torch_load(vae_ckpt, map_location=device)
    if isinstance(ckpt, dict) and "vae_state_dict" in ckpt:
        state = ckpt["vae_state_dict"]
    elif isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
    else:
        state = ckpt
    state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
    vae.load_state_dict(state)
    return vae


def _make_starts(length: int, tile: int, overlap: int) -> List[int]:
    if tile <= 0:
        raise ValueError("tile size must be > 0")
    if length <= tile:
        return [0]
    if overlap < 0 or overlap >= tile:
        raise ValueError(f"overlap must satisfy 0 <= overlap < tile, got overlap={overlap}, tile={tile}")
    step = tile - overlap
    starts = list(range(0, length - tile + 1, step))
    if starts[-1] != length - tile:
        starts.append(length - tile)
    return starts


def _axis_weight(length: int, overlap: int, at_start: bool, at_end: bool) -> np.ndarray:
    w = np.ones(length, dtype=np.float32)
    if overlap <= 0 or length <= 1:
        return w
    ov = min(overlap, length // 2)
    if ov <= 0:
        return w
    ramp = (np.arange(ov, dtype=np.float32) + 1.0) / float(ov + 1.0)
    if not at_start:
        w[:ov] = ramp
    if not at_end:
        w[-ov:] = ramp[::-1]
    return w


def _tile_weight(
    shape_dhw: Tuple[int, int, int],
    overlap_dhw: Tuple[int, int, int],
    at_start_dhw: Tuple[bool, bool, bool],
    at_end_dhw: Tuple[bool, bool, bool],
) -> np.ndarray:
    d, h, w = shape_dhw
    od, oh, ow = overlap_dhw
    s_d, s_h, s_w = at_start_dhw
    e_d, e_h, e_w = at_end_dhw
    wz = _axis_weight(d, od, s_d, e_d)
    wy = _axis_weight(h, oh, s_h, e_h)
    wx = _axis_weight(w, ow, s_w, e_w)
    return (wz[:, None, None] * wy[None, :, None] * wx[None, None, :]).astype(np.float32, copy=False)


def _decode_one_tile(
    z_tile: np.ndarray,
    vae: KLVAE3D,
    device: torch.device,
    use_cuda_bf16: bool,
) -> np.ndarray:
    z_t = torch.from_numpy(z_tile[None, ...]).to(device=device, dtype=torch.float32)
    with torch.no_grad():
        if device.type == "cuda" and use_cuda_bf16:
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits = vae.decode(z_t)
        else:
            logits = vae.decode(z_t)
    prob_tile = torch.sigmoid(logits)[0, 0].detach().cpu().float().numpy()
    return prob_tile.astype(np.float32, copy=False)


def decode_latent_tiled(
    z_unscaled: np.ndarray,
    vae: KLVAE3D,
    device: torch.device,
    threshold: float,
    tile_latent: int,
    overlap_latent: int,
    use_cuda_bf16: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    if z_unscaled.ndim != 4:
        raise ValueError(f"Expected latent shape [C,D,H,W], got {z_unscaled.shape}")

    c, ld, lh, lw = z_unscaled.shape
    if c <= 0 or ld <= 0 or lh <= 0 or lw <= 0:
        raise ValueError(f"Invalid latent shape: {z_unscaled.shape}")

    td = min(int(tile_latent), ld)
    th = min(int(tile_latent), lh)
    tw = min(int(tile_latent), lw)
    od = min(int(overlap_latent), max(0, td - 1))
    oh = min(int(overlap_latent), max(0, th - 1))
    ow = min(int(overlap_latent), max(0, tw - 1))

    d_starts = _make_starts(ld, td, od)
    h_starts = _make_starts(lh, th, oh)
    w_starts = _make_starts(lw, tw, ow)
    total_tiles = len(d_starts) * len(h_starts) * len(w_starts)
    print(
        f"[decode] latent={z_unscaled.shape}, tile={td}^3, overlap={od} "
        f"-> {len(d_starts)}x{len(h_starts)}x{len(w_starts)} = {total_tiles} tiles"
    )

    prob_sum: Optional[np.ndarray] = None
    w_sum: Optional[np.ndarray] = None
    sd = sh = sw = 0

    tile_idx = 0
    for z0 in d_starts:
        z1 = min(z0 + td, ld)
        for y0 in h_starts:
            y1 = min(y0 + th, lh)
            for x0 in w_starts:
                x1 = min(x0 + tw, lw)
                tile_idx += 1

                z_tile = z_unscaled[:, z0:z1, y0:y1, x0:x1].astype(np.float32, copy=False)
                prob_tile = _decode_one_tile(
                    z_tile=z_tile,
                    vae=vae,
                    device=device,
                    use_cuda_bf16=use_cuda_bf16,
                )

                if prob_sum is None or w_sum is None:
                    sd = prob_tile.shape[0] // max(1, (z1 - z0))
                    sh = prob_tile.shape[1] // max(1, (y1 - y0))
                    sw = prob_tile.shape[2] // max(1, (x1 - x0))
                    if sd <= 0 or sh <= 0 or sw <= 0:
                        raise RuntimeError("Failed to infer decoder upsample factor from first tile.")
                    if not (sd == sh == sw):
                        raise RuntimeError(
                            f"Non-isotropic decode scale detected: sd={sd}, sh={sh}, sw={sw}."
                        )

                    out_shape = (ld * sd, lh * sh, lw * sw)
                    prob_sum = np.zeros(out_shape, dtype=np.float32)
                    w_sum = np.zeros(out_shape, dtype=np.float32)
                    print(f"[decode] inferred decode scale x{sd}, voxel output shape={out_shape}")

                oz0, oz1 = z0 * sd, z1 * sd
                oy0, oy1 = y0 * sh, y1 * sh
                ox0, ox1 = x0 * sw, x1 * sw

                tile_w = _tile_weight(
                    shape_dhw=prob_tile.shape,
                    overlap_dhw=(od * sd, oh * sh, ow * sw),
                    at_start_dhw=(z0 == 0, y0 == 0, x0 == 0),
                    at_end_dhw=(z1 == ld, y1 == lh, x1 == lw),
                )

                prob_sum[oz0:oz1, oy0:oy1, ox0:ox1] += prob_tile * tile_w
                w_sum[oz0:oz1, oy0:oy1, ox0:ox1] += tile_w

                if tile_idx == 1 or tile_idx % 8 == 0 or tile_idx == total_tiles:
                    print(f"[decode] processed {tile_idx}/{total_tiles} tiles")

    if prob_sum is None or w_sum is None:
        raise RuntimeError("No tiles were decoded.")

    prob = prob_sum / np.clip(w_sum, 1e-6, None)
    binv = (prob >= float(threshold)).astype(np.uint8)
    return prob.astype(np.float32, copy=False), binv


def _ensure_parent(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def _build_phi_from_raw(
    raw_volume_path: str,
    patch_size: int,
    downsample_factor: int,
    binarize_mode: str,
    binarize_threshold: float,
    phi_input_target_size: int,
) -> np.ndarray:
    vol = np.load(raw_volume_path).astype(np.float32)
    vol = center_crop_cube(vol, int(phi_input_target_size))
    vol = normalize_to_unit(vol)

    mode = str(binarize_mode).lower().strip()
    if mode == "otsu":
        thr = otsu_threshold(vol)
        bin_vol = (vol >= thr).astype(np.float32)
    elif mode == "none":
        bin_vol = vol
    else:
        bin_vol = (vol >= float(binarize_threshold)).astype(np.float32)

    patch_voxel = int(patch_size) * int(downsample_factor)
    return compute_phi_map(bin_vol, patch_voxel)


def main():
    repo_root = Path(__file__).resolve().parents[1]

    parser = argparse.ArgumentParser(
        description="Large-volume inference wrapper (reuse infer.py) with optional tiled VAE decode."
    )
    parser.add_argument("--phi-map", default=CONFIG.get("phi_map_path", ""))
    parser.add_argument(
        "--raw-volume",
        default="",
        help="Optional raw voxel .npy. If set, phi_map is built on-the-fly and --phi-map is ignored.",
    )
    parser.add_argument("--save-phi-map", default="", help="Optional path to save generated phi_map .npy.")
    parser.add_argument("--ckpt", default=CONFIG.get("ckpt_path", ""))
    parser.add_argument("--output-latent", default=CONFIG.get("output_latent_path", "generated_latent.npy"))
    parser.add_argument(
        "--input-latent",
        default="",
        help="Optional existing latent .npy. If set, skip generation and decode from this file.",
    )
    parser.add_argument(
        "--input-latent-scaled",
        action="store_true",
        help="Set when --input-latent is still in diffusion scaled space; script will divide by scale_factor before decode.",
    )
    parser.add_argument("--device", default=CONFIG.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--patch-size", type=int, default=int(CONFIG.get("patch_size", 8)))
    parser.add_argument("--window-size", type=int, default=int(CONFIG.get("window_size", 3)))
    parser.add_argument("--downsample-factor", type=int, default=int(CONFIG.get("downsample_factor", 8)))
    parser.add_argument("--binarize-mode", default=CONFIG.get("binarize_mode", "none"))
    parser.add_argument("--binarize-threshold", type=float, default=float(CONFIG.get("binarize_threshold", 0.5)))
    parser.add_argument("--phi-input-target-size", type=int, default=int(CONFIG.get("phi_input_target_size", 0)))
    parser.add_argument("--ddim-steps", type=int, default=int(CONFIG.get("ddim_steps", 200)))
    parser.add_argument("--seed", type=int, default=int(CONFIG.get("seed", 1234)))
    parser.add_argument(
        "--infer-max-patch-batch",
        type=int,
        default=int(CONFIG.get("infer_max_patch_batch", 16)),
        help="Per-iteration patch count in autoregressive generation.",
    )
    parser.add_argument("--safe-threshold", type=float, default=float(CONFIG.get("safe_threshold", 8.0)))

    parser.set_defaults(output_unscaled=bool(CONFIG.get("output_unscaled", True)))
    parser.add_argument(
        "--output-unscaled",
        dest="output_unscaled",
        action="store_true",
        help="Divide latent by scale_factor before saving (recommended for VAE decode).",
    )
    parser.add_argument(
        "--keep-scaled-latent",
        dest="output_unscaled",
        action="store_false",
        help="Keep latent in diffusion scaled space.",
    )

    parser.set_defaults(decode_voxel=True)
    parser.add_argument("--decode-voxel", dest="decode_voxel", action="store_true")
    parser.add_argument("--no-decode-voxel", dest="decode_voxel", action="store_false")
    parser.add_argument("--vae-config", default=CONFIG.get("eval_vae_config_path", ""))
    parser.add_argument("--vae-ckpt", default=CONFIG.get("eval_vae_ckpt_path", ""))
    parser.add_argument("--output-prob", default="")
    parser.add_argument("--output-bin", default="")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--decode-latent-tile", type=int, default=24, help="Tile size in latent voxels.")
    parser.add_argument(
        "--decode-latent-overlap",
        type=int,
        default=4,
        help="Overlap in latent voxels for seam-free blending.",
    )
    parser.set_defaults(decode_bf16=True)
    parser.add_argument("--decode-bf16", dest="decode_bf16", action="store_true")
    parser.add_argument("--decode-fp32", dest="decode_bf16", action="store_false")

    args = parser.parse_args()

    ckpt_path = _resolve_path(args.ckpt, repo_root)
    latent_path = _resolve_path(args.output_latent, repo_root)
    input_latent_path = _resolve_path(args.input_latent, repo_root)
    if not input_latent_path and (not ckpt_path or not os.path.exists(ckpt_path)):
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")

    device = torch.device(args.device)
    print(f"[run] device={device}")
    if input_latent_path:
        print(f"[run] input_latent={input_latent_path}")
    else:
        print(f"[run] ckpt={ckpt_path}")

    # Runtime-only overrides for infer.py internals.
    CONFIG["infer_max_patch_batch"] = max(1, int(args.infer_max_patch_batch))
    CONFIG["safe_threshold"] = float(args.safe_threshold)

    if input_latent_path:
        if not os.path.exists(input_latent_path):
            raise FileNotFoundError(f"input latent not found: {input_latent_path}")
        z_full = np.load(input_latent_path).astype(np.float32, copy=False)
        if z_full.ndim != 4:
            raise ValueError(f"Expected latent [C,D,H,W], got {z_full.shape}")
        if bool(args.input_latent_scaled):
            scale = float(CONFIG.get("scale_factor", 1.0))
            if scale not in (0.0, 1.0):
                z_full = z_full / scale
                print(f"[run] input latent unscaled by scale_factor={scale}")
        print(f"[run] loaded latent: {input_latent_path}, shape={z_full.shape}")
    else:
        raw_volume_path = _resolve_path(args.raw_volume, repo_root)
        phi_path = _resolve_path(args.phi_map, repo_root)
        if raw_volume_path:
            if not os.path.exists(raw_volume_path):
                raise FileNotFoundError(f"raw volume not found: {raw_volume_path}")
            print(f"[run] raw_volume={raw_volume_path}")
            phi_map = _build_phi_from_raw(
                raw_volume_path=raw_volume_path,
                patch_size=int(args.patch_size),
                downsample_factor=int(args.downsample_factor),
                binarize_mode=str(args.binarize_mode),
                binarize_threshold=float(args.binarize_threshold),
                phi_input_target_size=int(args.phi_input_target_size),
            )
            print(f"[run] phi_map built from raw, shape={phi_map.shape}")
            if args.save_phi_map:
                phi_save_path = _resolve_path(args.save_phi_map, repo_root)
                _ensure_parent(phi_save_path)
                np.save(phi_save_path, phi_map.astype(np.float32, copy=False))
                print(f"[run] saved phi_map: {phi_save_path}")
        else:
            if not phi_path or not os.path.exists(phi_path):
                raise FileNotFoundError(f"phi map not found: {phi_path}")
            print(f"[run] phi_map={phi_path}")
            phi_map = np.load(phi_path).astype(np.float32)
        print(f"[run] phi_map shape={phi_map.shape}")

        model = load_model(ckpt_path, device)
        diffusion = DiffusionHelper(int(CONFIG["timesteps"]), device)
        seed = None if int(args.seed) < 0 else int(args.seed)

        z_full = generate_volume(
            phi_map=phi_map,
            model=model,
            diffusion=diffusion,
            patch_size=int(args.patch_size),
            window_size=int(args.window_size),
            steps=int(args.ddim_steps),
            seed=seed,
        )

        if bool(args.output_unscaled):
            scale = float(CONFIG.get("scale_factor", 1.0))
            if scale not in (0.0, 1.0):
                z_full = z_full / scale
                print(f"[run] latent unscaled by scale_factor={scale}")

        _ensure_parent(latent_path)
        np.save(latent_path, z_full.astype(np.float32, copy=False))
        print(f"[run] saved latent: {latent_path}, shape={z_full.shape}")

    if not bool(args.decode_voxel):
        return

    vae_cfg = _resolve_path(args.vae_config, repo_root)
    vae_ckpt = _resolve_path(args.vae_ckpt, repo_root)
    if not vae_cfg or not os.path.exists(vae_cfg):
        raise FileNotFoundError(f"vae config not found: {vae_cfg}")
    if not vae_ckpt or not os.path.exists(vae_ckpt):
        raise FileNotFoundError(f"vae checkpoint not found: {vae_ckpt}")

    vae = load_vae(vae_cfg, vae_ckpt, device)
    prob, binv = decode_latent_tiled(
        z_unscaled=z_full.astype(np.float32, copy=False),
        vae=vae,
        device=device,
        threshold=float(args.threshold),
        tile_latent=int(args.decode_latent_tile),
        overlap_latent=int(args.decode_latent_overlap),
        use_cuda_bf16=bool(args.decode_bf16),
    )

    if args.output_prob:
        prob_path = _resolve_path(args.output_prob, repo_root)
    else:
        prob_path = os.path.splitext(latent_path)[0] + "_voxel_prob.npy"
    if args.output_bin:
        bin_path = _resolve_path(args.output_bin, repo_root)
    else:
        bin_path = os.path.splitext(latent_path)[0] + "_voxel_bin.npy"

    _ensure_parent(prob_path)
    _ensure_parent(bin_path)
    np.save(prob_path, prob.astype(np.float32, copy=False))
    np.save(bin_path, binv.astype(np.uint8, copy=False))
    print(f"[run] saved voxel prob: {prob_path}, shape={prob.shape}")
    print(f"[run] saved voxel bin : {bin_path}, shape={binv.shape}")


if __name__ == "__main__":
    main()

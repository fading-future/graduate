import os
import glob
import yaml
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

from models.vae import KLVAE3D

# ================= Configuration =================
CONFIG_PATH = r"E:\chendou\paper_code\stage02_KLvae_single_code_v2\config\train_config copy.yaml"
CHECKPOINT_PATH = r"E:\chendou\paper_code\stage02_KLvae_single_code_v2\experiments\exp04_cube_structure_v1\ckpt_epoch_11.pt"
CSV_PATH = r"D:\多尺度岩心数据集\binary_klvae_latents_256_p64\processing_report.csv"
SAVE_DIR = r"D:\多尺度岩心数据集\LDM_Data\Latent_NPY\w192_s64"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# direct: encode full volume once (recommended)
# tiled: encode by tiles and stitch in latent space
ENCODE_MODE = "direct"
ALLOW_TILED_FALLBACK = True
ALLOW_CPU_FALLBACK = False

# Optional spatial crop before encoding:
# set to an int like 192 to center-crop each input volume to target^3;
# set to None to keep original size.
TARGET_SIZE = None

TILE_SIZE = 64
STRIDE = 64
TILE_BATCH = 4


def _safe_torch_load(path: str, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)
    except Exception:
        return torch.load(path, map_location=map_location)


def load_porosity_map(csv_path):
    if not csv_path or not os.path.exists(csv_path):
        print("No porosity CSV found; latents will be saved without porosity prefix.")
        return {}
    print(f"Loading porosity labels from {csv_path}...")
    df = pd.read_csv(csv_path)
    if "file" not in df.columns or "porosity" not in df.columns:
        print("CSV does not contain required columns: file, porosity.")
        return {}
    return dict(zip(df["file"], df["porosity"]))


def normalize_volume(data: np.ndarray) -> np.ndarray:
    """Normalize raw volume to [-1, 1] (same semantic as training input)."""
    data = data.astype(np.float32)
    mn, mx = float(data.min()), float(data.max())

    if mn >= -1.01 and mx <= 1.01:
        if mn >= 0.0:
            return data * 2.0 - 1.0
        return data
    if mx <= 1.5:
        return data * 2.0 - 1.0
    if mx <= 255.5:
        return (data / 255.0) * 2.0 - 1.0
    return (data / 65535.0) * 2.0 - 1.0


def center_crop_cube(vol: np.ndarray, target_size: int) -> np.ndarray:
    if target_size is None:
        return vol
    if vol.ndim != 3:
        raise ValueError(f"Expected 3D volume, got shape={vol.shape}.")
    D, H, W = vol.shape
    if target_size > D or target_size > H or target_size > W:
        raise ValueError(
            f"target_size={target_size} is larger than input shape {vol.shape}."
        )
    if D == target_size and H == target_size and W == target_size:
        return vol
    d0 = (D - target_size) // 2
    h0 = (H - target_size) // 2
    w0 = (W - target_size) // 2
    return vol[d0:d0 + target_size, h0:h0 + target_size, w0:w0 + target_size]


def downsample_factor_from_cfg(cfg: dict) -> int:
    ch_mult = cfg["model"]["ch_mult"]
    return 2 ** max(0, (len(ch_mult) - 1))


def tile_positions(size: int, tile: int, stride: int):
    if size < tile:
        return [0]
    pos = list(range(0, size - tile + 1, stride))
    last = size - tile
    if pos[-1] != last:
        pos.append(last)
    return pos


def _encode_once(vae, x: torch.Tensor):
    with torch.inference_mode():
        if x.device.type == "cuda":
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                z = vae.encode(x).mean
        else:
            z = vae.encode(x).mean
    return z


def encode_direct(vae, vol: np.ndarray, device: str):
    x = torch.from_numpy(vol).unsqueeze(0).unsqueeze(0).to(device)
    z = _encode_once(vae, x)
    return z.detach().cpu().float().numpy()[0]


def _encode_batch(vae, tiles, coords, z_full, wgt, lat_tile, downsample, device: str):
    x = np.stack(tiles, axis=0)  # (B, D, H, W)
    x = torch.from_numpy(x).unsqueeze(1).to(device)  # (B,1,D,H,W)
    z = _encode_once(vae, x).detach().cpu().float().numpy()

    for i, (z0, y0, x0) in enumerate(coords):
        lz0 = z0 // downsample
        ly0 = y0 // downsample
        lx0 = x0 // downsample
        lz1 = lz0 + lat_tile
        ly1 = ly0 + lat_tile
        lx1 = lx0 + lat_tile
        z_full[:, lz0:lz1, ly0:ly1, lx0:lx1] += z[i]
        wgt[:, lz0:lz1, ly0:ly1, lx0:lx1] += 1.0


def encode_tiled(
    vae,
    vol: np.ndarray,
    downsample: int,
    tile: int,
    stride: int,
    device: str,
    z_channels: int,
    tile_batch: int,
):
    D, H, W = vol.shape
    if not (D == H == W):
        raise ValueError(f"Only cubic volumes are supported, got {vol.shape}.")
    if tile % downsample != 0:
        raise ValueError("tile_size must be divisible by downsample.")
    if stride % downsample != 0:
        raise ValueError("stride must be divisible by downsample.")

    lat_tile = tile // downsample
    lat_size = D // downsample

    z_full = np.zeros((z_channels, lat_size, lat_size, lat_size), dtype=np.float32)
    wgt = np.zeros((1, lat_size, lat_size, lat_size), dtype=np.float32)

    zs = tile_positions(D, tile, stride)
    ys = tile_positions(H, tile, stride)
    xs = tile_positions(W, tile, stride)

    tiles = []
    coords = []
    for z0 in zs:
        for y0 in ys:
            for x0 in xs:
                tiles.append(vol[z0:z0 + tile, y0:y0 + tile, x0:x0 + tile])
                coords.append((z0, y0, x0))
                if len(tiles) == tile_batch:
                    _encode_batch(vae, tiles, coords, z_full, wgt, lat_tile, downsample, device)
                    tiles, coords = [], []

    if tiles:
        _encode_batch(vae, tiles, coords, z_full, wgt, lat_tile, downsample, device)

    return z_full / (wgt + 1e-8)


def _is_oom_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "out of memory" in msg


def _try_encode_tiled_with_adaptive_batch(
    vae,
    vol: np.ndarray,
    downsample: int,
    z_channels: int,
    device: str,
):
    tried = []
    for tb in [int(TILE_BATCH), 2, 1]:
        if tb <= 0 or tb in tried:
            continue
        tried.append(tb)
        try:
            return encode_tiled(
                vae,
                vol,
                downsample=downsample,
                tile=TILE_SIZE,
                stride=STRIDE,
                device=device,
                z_channels=z_channels,
                tile_batch=tb,
            )
        except torch.cuda.OutOfMemoryError:
            if device == "cuda":
                torch.cuda.empty_cache()
            print(f"Tiled encoding OOM with tile_batch={tb}, retrying with smaller batch...")
            continue
        except RuntimeError as exc:
            if not _is_oom_error(exc):
                raise
            if device == "cuda":
                torch.cuda.empty_cache()
            print(f"Tiled encoding OOM with tile_batch={tb}, retrying with smaller batch...")
            continue
    raise RuntimeError("Tiled encoding still OOM after trying tile_batch=[4,2,1].")


def encode_volume(
    vae,
    vol: np.ndarray,
    downsample: int,
    z_channels: int,
    device: str,
    mode: str = "direct",
    allow_tiled_fallback: bool = True,
):
    mode = str(mode).lower().strip()
    if mode == "tiled":
        return _try_encode_tiled_with_adaptive_batch(
            vae,
            vol,
            downsample=downsample,
            z_channels=z_channels,
            device=device,
        )

    try:
        return encode_direct(vae, vol, device=device)
    except torch.cuda.OutOfMemoryError:
        if not allow_tiled_fallback:
            raise
        print("Direct encoding OOM; fallback to tiled encoding.")
        if device == "cuda":
            torch.cuda.empty_cache()
        return _try_encode_tiled_with_adaptive_batch(
            vae,
            vol,
            downsample=downsample,
            z_channels=z_channels,
            device=device,
        )
    except RuntimeError as exc:
        if not (_is_oom_error(exc) and allow_tiled_fallback):
            raise
        print("Direct encoding OOM; fallback to tiled encoding.")
        if device == "cuda":
            torch.cuda.empty_cache()
        return _try_encode_tiled_with_adaptive_batch(
            vae,
            vol,
            downsample=downsample,
            z_channels=z_channels,
            device=device,
        )


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    if DEVICE == "cuda":
        torch.backends.cudnn.benchmark = True

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    porosity_map = load_porosity_map(CSV_PATH)

    vae = KLVAE3D(cfg).to(DEVICE)
    print(f"Loading model from {CHECKPOINT_PATH}...")
    checkpoint = _safe_torch_load(CHECKPOINT_PATH, map_location=DEVICE)
    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get("vae_state_dict", checkpoint.get("model_state_dict", checkpoint))
    else:
        state_dict = checkpoint
    state_dict = {k[10:] if k.startswith("_orig_mod.") else k: v for k, v in state_dict.items()}
    vae.load_state_dict(state_dict)
    vae.eval()

    data_root = cfg["data"]["data_root"]
    ext = cfg["data"]["file_extension"]
    files = sorted(glob.glob(os.path.join(data_root, f"*{ext}")))
    print(f"Found {len(files)} files for inference.")
    print(f"Encoding mode: {ENCODE_MODE} (fallback to tiled: {ALLOW_TILED_FALLBACK})")
    print(f"TARGET_SIZE: {TARGET_SIZE}")

    all_latents_stats = []
    save_count = 0
    downsample = downsample_factor_from_cfg(cfg)
    z_channels = int(cfg["model"]["z_channels"])

    for fp in tqdm(files, desc="Encode"):
        orig_name = os.path.basename(fp)
        data = np.load(fp, mmap_mode="r")
        data = normalize_volume(data)
        data = center_crop_cube(data, TARGET_SIZE)

        if data.ndim != 3:
            raise ValueError(f"Expected 3D volume, got shape={data.shape} for {orig_name}.")
        D, H, W = data.shape
        if not (D == H == W):
            raise ValueError(f"Only cubic volume supported, got shape={data.shape} for {orig_name}.")
        if D % downsample != 0:
            raise ValueError(
                f"Input size {D} is not divisible by downsample factor {downsample} for {orig_name}."
            )

        try:
            z_full = encode_volume(
                vae,
                data,
                downsample=downsample,
                z_channels=z_channels,
                device=DEVICE,
                mode=ENCODE_MODE,
                allow_tiled_fallback=ALLOW_TILED_FALLBACK,
            )
        except RuntimeError as exc:
            can_try_cpu = ALLOW_CPU_FALLBACK and DEVICE == "cuda" and _is_oom_error(exc)
            if not can_try_cpu:
                raise
            print("GPU encoding still OOM, retrying on CPU for this sample...")
            z_full = encode_volume(
                vae.to("cpu"),
                data,
                downsample=downsample,
                z_channels=z_channels,
                device="cpu",
                mode=ENCODE_MODE,
                allow_tiled_fallback=True,
            )
            vae.to(DEVICE)
            vae.eval()

        if orig_name in porosity_map:
            por = float(porosity_map[orig_name])
            save_name = f"porosity_{por:.6f}_{orig_name}"
        else:
            save_name = orig_name

        np.save(os.path.join(SAVE_DIR, save_name), z_full)
        save_count += 1

        if len(all_latents_stats) < 200:
            all_latents_stats.append(z_full.reshape(-1))

    print("Calculating latent statistics...")
    if len(all_latents_stats) == 0:
        print("No data saved.")
        return

    all_data = np.concatenate(all_latents_stats)
    std = float(np.std(all_data))
    mean = float(np.mean(all_data))
    print("\n" + "=" * 40)
    print("Data Preparation Finished")
    print(f"Total files saved: {save_count}")
    print(f"Saved to: {SAVE_DIR}")
    print("-" * 40)
    print("Stage07 Configuration Suggestion")
    print(f"Latent Mean: {mean:.6f}")
    print(f"Latent Std:  {std:.6f}")
    if std > 0:
        print(f"Set stage07 scale_factor to: {1.0 / std:.6f}")
    else:
        print("Latent std is zero; cannot compute reciprocal scale_factor.")
    print("=" * 40)


if __name__ == "__main__":
    main()

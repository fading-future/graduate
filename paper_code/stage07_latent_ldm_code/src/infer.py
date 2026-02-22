import os
import random
from collections import defaultdict
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from tqdm import tqdm

from src.config import CONFIG
from src.model_unet3d import ConditionalLatentUNet
from src.diffusion import DiffusionHelper


Coord3D = Tuple[int, int, int]
Direction3D = Tuple[int, int, int]
_ORDER_CANDIDATES = ("ijk", "ikj", "jik", "jki", "kij", "kji")
_AXIS_TO_IDX = {"i": 0, "j": 1, "k": 2}


def _safe_torch_load(path: str, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)
    except Exception:
        return torch.load(path, map_location=map_location)


def _normalize_order(order: str) -> str:
    order = str(order).lower().strip()
    if len(order) != 3 or set(order) != {"i", "j", "k"}:
        return "ijk"
    return order


def _phi_channels() -> int:
    return 2 if bool(CONFIG.get("use_global_phi_channel", False)) else 1


def _resolve_in_channels() -> int:
    # model input = x_t(C) + cond(C) + mask(1) + phi(phi_channels)
    c = int(CONFIG.get("out_channels", CONFIG.get("latent_channels", 4)))
    expected = 2 * c + 1 + _phi_channels()
    cfg_in = int(CONFIG.get("in_channels", expected))
    if cfg_in != expected:
        print(f"[warn] CONFIG['in_channels']={cfg_in} != expected={expected}; using expected.")
    return expected


def _value_to_sign(v) -> int:
    if isinstance(v, (int, np.integer, float, np.floating)):
        return 1 if float(v) >= 0 else -1
    txt = str(v).strip().lower()
    if txt in ("+", "1", "+1", "pos", "forward", "fwd"):
        return 1
    if txt in ("-", "-1", "neg", "reverse", "rev", "backward", "bwd"):
        return -1
    return 1


def _normalize_direction(direction) -> Direction3D:
    # Accept forms like "+++", "+-+", "1,-1,1", [1,-1,1], ("+","-","+")
    if isinstance(direction, str):
        txt = direction.strip().lower().replace(" ", "")
        if len(txt) == 3 and set(txt).issubset({"+", "-"}):
            return tuple(1 if ch == "+" else -1 for ch in txt)  # type: ignore[return-value]
        if "," in txt:
            parts = txt.split(",")
            if len(parts) == 3:
                return tuple(_value_to_sign(p) for p in parts)  # type: ignore[return-value]
        return (1, 1, 1)
    if isinstance(direction, Sequence) and not isinstance(direction, (bytes, bytearray)):
        parts = list(direction)
        if len(parts) == 3:
            return tuple(_value_to_sign(p) for p in parts)  # type: ignore[return-value]
    return (1, 1, 1)


def _direction_str(direction: Direction3D) -> str:
    return "".join("+" if x >= 0 else "-" for x in direction)


def _select_infer_order_and_direction(seed) -> Tuple[str, Direction3D]:
    order = _normalize_order(CONFIG.get("order", "ijk"))
    direction = _normalize_direction(CONFIG.get("infer_direction", "+++"))

    random_order = bool(CONFIG.get("infer_random_order", False))
    random_direction = bool(CONFIG.get("infer_random_direction", False))

    if not random_order and not random_direction:
        return order, direction

    rng = random.Random(None if seed is None else int(seed))
    if random_order:
        order = rng.choice(_ORDER_CANDIDATES)
    if random_direction:
        direction = tuple(rng.choice((1, -1)) for _ in range(3))  # type: ignore[assignment]
    return order, direction


def _normalize_context_mode(mode: str) -> str:
    mode = str(mode).lower().strip()
    if mode not in ("causal", "full", "wavefront"):
        return "causal"
    return mode


def _normalize_pad_mode(mode: str) -> str:
    mode = str(mode).lower().strip()
    if mode not in ("constant", "edge", "reflect"):
        return "edge"
    return mode


def _is_prev_lexicographic(a: Coord3D, b: Coord3D, order: str, direction: Direction3D) -> bool:
    ai, aj, ak = a
    bi, bj, bk = b
    axis_a = {"i": ai, "j": aj, "k": ak}
    axis_b = {"i": bi, "j": bj, "k": bk}
    axis_s = {"i": direction[0], "j": direction[1], "k": direction[2]}
    for axis in _normalize_order(order):
        va, vb = axis_a[axis] * axis_s[axis], axis_b[axis] * axis_s[axis]
        if va < vb:
            return True
        if va > vb:
            return False
    return False


def _is_prev_wavefront(a: Coord3D, b: Coord3D, direction: Direction3D) -> bool:
    ai, aj, ak = a
    bi, bj, bk = b
    si, sj, sk = direction
    cond_i = ai <= bi if si > 0 else ai >= bi
    cond_j = aj <= bj if sj > 0 else aj >= bj
    cond_k = ak <= bk if sk > 0 else ak >= bk
    return (cond_i and cond_j and cond_k) and (a != b)


def _is_prev_by_mode(
    a: Coord3D,
    b: Coord3D,
    context_mode: str,
    order: str,
    direction: Direction3D,
) -> bool:
    if context_mode == "wavefront":
        return _is_prev_wavefront(a, b, direction)
    return _is_prev_lexicographic(a, b, order, direction)


def _coord_key(coord: Coord3D, context_mode: str, order: str, direction: Direction3D):
    sign_i, sign_j, sign_k = direction
    if context_mode == "wavefront":
        i, j, k = coord
        ti, tj, tk = sign_i * i, sign_j * j, sign_k * k
        return (ti + tj + tk, ti, tj, tk)
    norm_order = _normalize_order(order)
    return tuple(coord[_AXIS_TO_IDX[a]] * direction[_AXIS_TO_IDX[a]] for a in norm_order)


def _repeat_phi(phi_patch: np.ndarray, patch_size: int) -> np.ndarray:
    out = np.repeat(phi_patch, patch_size, axis=0)
    out = np.repeat(out, patch_size, axis=1)
    out = np.repeat(out, patch_size, axis=2)
    return out


def _pad_with_mode(x: np.ndarray, pad_width, mode: str) -> np.ndarray:
    # numpy reflect mode requires pad < axis length; fallback to edge if invalid
    if mode == "reflect":
        for axis, (pad_before, pad_after) in enumerate(pad_width):
            axis_len = x.shape[axis]
            if axis_len <= 1 and (pad_before > 0 or pad_after > 0):
                return np.pad(x, pad_width, mode="edge")
            if pad_before >= axis_len or pad_after >= axis_len:
                return np.pad(x, pad_width, mode="edge")
    return np.pad(x, pad_width, mode=mode)


def _seeded_randn_like(x: torch.Tensor, seed):
    if seed is None:
        return torch.randn_like(x)

    if isinstance(seed, (int, np.integer)):
        gen = torch.Generator(device=x.device)
        gen.manual_seed(int(seed))
        return torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=gen)

    if isinstance(seed, Sequence):
        if len(seed) != x.shape[0]:
            raise ValueError(f"Seed list length {len(seed)} != batch size {x.shape[0]}")
        out = torch.empty_like(x)
        for bi, one_seed in enumerate(seed):
            if one_seed is None:
                out[bi] = torch.randn(x[bi].shape, device=x.device, dtype=x.dtype)
            else:
                gen = torch.Generator(device=x.device)
                gen.manual_seed(int(one_seed))
                out[bi] = torch.randn(x[bi].shape, device=x.device, dtype=x.dtype, generator=gen)
        return out

    gen = torch.Generator(device=x.device)
    gen.manual_seed(int(seed))
    return torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=gen)


def load_model(ckpt_path: str, device: torch.device):
    model = ConditionalLatentUNet(
        in_channels=_resolve_in_channels(),
        out_channels=CONFIG["out_channels"],
        base_channels=CONFIG["base_channels"],
        channel_mults=CONFIG["channel_mults"],
        use_attention=CONFIG["use_attention"],
    ).to(device)
    ckpt = _safe_torch_load(ckpt_path, map_location=device)
    state_name = "raw"
    if isinstance(ckpt, dict):
        use_ema = bool(CONFIG.get("infer_use_ema", False))
        if use_ema and "ema_state_dict" in ckpt:
            state = ckpt["ema_state_dict"]
            state_name = "ema_state_dict"
        elif "model_state_dict" in ckpt:
            state = ckpt["model_state_dict"]
            state_name = "model_state_dict"
        elif "ema_state_dict" in ckpt:
            state = ckpt["ema_state_dict"]
            state_name = "ema_state_dict"
        else:
            state = ckpt
    else:
        state = ckpt
    if isinstance(state, dict):
        state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
    model.load_state_dict(state)
    model.eval()
    print(f"[infer] loaded weights from {state_name}")
    return model


def ddim_sample(model, cond, mask, phi, porosity, diffusion: DiffusionHelper, steps=200, seed=1234, safe_thresh=8.0):
    model.eval()
    total_timesteps = diffusion.timesteps
    alphas_cumprod = diffusion.alphas_cumprod

    times = torch.linspace(0, total_timesteps - 1, steps=steps, device=cond.device)
    times = torch.unique(torch.round(times).long(), sorted=True)
    times = list(reversed(times.tolist()))

    fixed_noise = _seeded_randn_like(cond, seed)
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


def _build_dependency_graph(
    gD: int,
    gH: int,
    gW: int,
    radius: int,
    context_mode: str,
    order: str,
    direction: Direction3D,
) -> Tuple[Dict[Coord3D, List[Coord3D]], Dict[Coord3D, List[Coord3D]]]:
    deps: Dict[Coord3D, List[Coord3D]] = {}
    dependents: Dict[Coord3D, List[Coord3D]] = defaultdict(list)

    for i in range(gD):
        for j in range(gH):
            for k in range(gW):
                center = (i, j, k)
                one_deps: List[Coord3D] = []
                for gi in range(max(0, i - radius), min(gD, i + radius + 1)):
                    for gj in range(max(0, j - radius), min(gH, j + radius + 1)):
                        for gk in range(max(0, k - radius), min(gW, k + radius + 1)):
                            neigh = (gi, gj, gk)
                            if neigh == center:
                                continue
                            if _is_prev_by_mode(neigh, center, context_mode, order, direction):
                                one_deps.append(neigh)
                deps[center] = one_deps
                for d in one_deps:
                    dependents[d].append(center)
    return deps, dependents


def _build_generation_batches(
    gD: int,
    gH: int,
    gW: int,
    window_size: int,
    context_mode: str,
    order: str,
    direction: Direction3D,
    max_patch_batch: int,
) -> List[List[Coord3D]]:
    radius = window_size // 2
    deps, dependents = _build_dependency_graph(gD, gH, gW, radius, context_mode, order, direction)
    pending = {coord: len(one_deps) for coord, one_deps in deps.items()}
    ready = [coord for coord, cnt in pending.items() if cnt == 0]
    max_patch_batch = max(1, int(max_patch_batch))

    total = gD * gH * gW
    done = 0
    batches: List[List[Coord3D]] = []

    while ready:
        ready.sort(key=lambda c: _coord_key(c, context_mode, order, direction))
        batch = ready[:max_patch_batch]
        ready = ready[max_patch_batch:]
        batches.append(batch)

        for coord in batch:
            done += 1
            for dep in dependents.get(coord, []):
                pending[dep] -= 1
                if pending[dep] == 0:
                    ready.append(dep)

    if done != total:
        unresolved = [coord for coord, cnt in pending.items() if cnt > 0]
        unresolved_preview = ", ".join(str(x) for x in unresolved[:5])
        raise RuntimeError(
            f"Generation dependency graph has unresolved nodes ({len(unresolved)}). "
            f"Examples: {unresolved_preview}"
        )
    return batches


def generate_volume(phi_map: np.ndarray, model, diffusion, patch_size: int, window_size: int, steps: int, seed: int):
    device = next(model.parameters()).device
    C = CONFIG["out_channels"]

    gD, gH, gW = phi_map.shape
    D, H, W = gD * patch_size, gH * patch_size, gW * patch_size
    z_full = np.zeros((C, D, H, W), dtype=np.float32)
    known_patch = np.zeros((gD, gH, gW), dtype=bool)

    w = window_size
    r = w // 2
    order, direction = _select_infer_order_and_direction(seed)
    context_mode = _normalize_context_mode(CONFIG.get("context_mode", "causal"))
    pad_mode = _normalize_pad_mode(CONFIG.get("pad_mode", "edge"))
    max_patch_batch = int(CONFIG.get("infer_max_patch_batch", 16))
    if max_patch_batch < 1:
        max_patch_batch = 1

    # full context is only meaningful during training; inference must rely on generated history.
    if context_mode == "full":
        print("[warn] context_mode='full' is not autoregressive for inference; fallback to causal.")
        context_mode = "causal"
    print(f"[infer] traversal order={order}, direction={_direction_str(direction)}, context_mode={context_mode}")

    phi_pad = _pad_with_mode(phi_map, ((r, r), (r, r), (r, r)), pad_mode)
    pad_p = r * patch_size
    global_phi_val = float(phi_map.mean())

    batches = _build_generation_batches(
        gD=gD,
        gH=gH,
        gW=gW,
        window_size=w,
        context_mode=context_mode,
        order=order,
        direction=direction,
        max_patch_batch=max_patch_batch,
    )

    total_patches = gD * gH * gW
    pbar = tqdm(total=total_patches, desc="GeneratePatches")
    patch_counter = 0

    for coord_batch in batches:
        z_pad = _pad_with_mode(
            z_full,
            ((0, 0), (pad_p, pad_p), (pad_p, pad_p), (pad_p, pad_p)),
            pad_mode,
        )

        cond_batch: List[np.ndarray] = []
        mask_batch: List[np.ndarray] = []
        phi_batch: List[np.ndarray] = []
        por_batch: List[float] = []

        for (i, j, k) in coord_batch:
            ci, cj, ck = i + r, j + r, k + r
            wi0, wi1 = ci - r, ci + r + 1
            wj0, wj1 = cj - r, cj + r + 1
            wk0, wk1 = ck - r, ck + r + 1

            phi_win = phi_pad[wi0:wi1, wj0:wj1, wk0:wk1]

            zi0, zi1 = wi0 * patch_size, wi1 * patch_size
            zj0, zj1 = wj0 * patch_size, wj1 * patch_size
            zk0, zk1 = wk0 * patch_size, wk1 * patch_size
            z_win = z_pad[:, zi0:zi1, zj0:zj1, zk0:zk1]

            mask_patch = np.zeros((w, w, w), dtype=np.float32)
            for di in range(w):
                for dj in range(w):
                    for dk in range(w):
                        gi = i - r + di
                        gj = j - r + dj
                        gk = k - r + dk
                        if gi < 0 or gj < 0 or gk < 0 or gi >= gD or gj >= gH or gk >= gW:
                            continue
                        known = known_patch[gi, gj, gk] and _is_prev_by_mode(
                            (gi, gj, gk),
                            (i, j, k),
                            context_mode,
                            order,
                            direction,
                        )
                        if known:
                            mask_patch[di, dj, dk] = 1.0

            mask = _repeat_phi(mask_patch, patch_size)[None, ...]
            cond = z_win * mask
            local_phi_vol = _repeat_phi(phi_win, patch_size)
            if _phi_channels() >= 2:
                global_phi_patch = np.full_like(phi_win, global_phi_val, dtype=np.float32)
                global_phi_vol = _repeat_phi(global_phi_patch, patch_size)
                phi_vol = np.stack([local_phi_vol, global_phi_vol], axis=0)
            else:
                phi_vol = local_phi_vol[None, ...]

            por_mode = str(CONFIG.get("porosity_mode", "local")).lower()
            local_phi_val = float(phi_map[i, j, k])
            if por_mode == "global":
                porosity = global_phi_val
            elif por_mode in ("mix", "local_global_mix"):
                a = float(np.clip(CONFIG.get("porosity_mix_alpha", 0.7), 0.0, 1.0))
                porosity = a * local_phi_val + (1.0 - a) * global_phi_val
            else:
                porosity = local_phi_val

            cond_batch.append(cond)
            mask_batch.append(mask)
            phi_batch.append(phi_vol)
            por_batch.append(porosity)

        cond_t = torch.from_numpy(np.stack(cond_batch, axis=0)).to(device)
        mask_t = torch.from_numpy(np.stack(mask_batch, axis=0)).to(device)
        phi_t = torch.from_numpy(np.stack(phi_batch, axis=0)).to(device)
        por_t = torch.from_numpy(np.array(por_batch, dtype=np.float32)[:, None]).to(device)

        if seed is None:
            batch_seed = None
        else:
            batch_seed = [int(seed) + patch_counter + bi for bi in range(len(coord_batch))]

        x = ddim_sample(
            model=model,
            cond=cond_t,
            mask=mask_t,
            phi=phi_t,
            porosity=por_t,
            diffusion=diffusion,
            steps=steps,
            seed=batch_seed,
            safe_thresh=CONFIG["safe_threshold"],
        )
        x_np = x.detach().cpu().float().numpy()

        c0 = r * patch_size
        c1 = c0 + patch_size
        for bi, (i, j, k) in enumerate(coord_batch):
            ti0, tj0, tk0 = i * patch_size, j * patch_size, k * patch_size
            ti1, tj1, tk1 = ti0 + patch_size, tj0 + patch_size, tk0 + patch_size
            z_full[:, ti0:ti1, tj0:tj1, tk0:tk1] = x_np[bi, :, c0:c1, c0:c1, c0:c1]
            known_patch[i, j, k] = True

        patch_counter += len(coord_batch)
        pbar.update(len(coord_batch))

    pbar.close()
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

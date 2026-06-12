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


def _masked_mean_with_fallback(values: np.ndarray, mask: np.ndarray, fallback: float) -> float:
    sel = mask > 0.5
    if not np.any(sel):
        return float(fallback)
    return float(values[sel].mean())


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


# ─── Latent Variance Rescaling ───────────────────────────────────────
# 数据集统计：pore_rate → 期望 latent_std（scaled 前）的线性映射参数
# 由 `latent_std = _RESCALE_STD_BASE + _RESCALE_STD_SLOPE * pore_rate` 拟合
# pore_rate=0.0 → std≈0.58,  pore_rate=0.5 → std≈0.95
_RESCALE_STD_BASE = 0.58       # 纯岩石 latent std (intercept)
_RESCALE_STD_SLOPE = 0.72      # 每增加 1.0 pore_rate, std 增长量


def _expected_latent_std(rock_rate: float) -> float:
    """根据 phi_map 值（rock_rate）估算该 patch 的期望 latent std。

    数据集统计（unscaled latent）：
      pore [0.00, 0.05): latent_std ≈ 0.64
      pore [0.05, 0.10): latent_std ≈ 0.77
      pore [0.10, 0.18): latent_std ≈ 0.80
      pore [0.18, 0.28): latent_std ≈ 0.87
      pore [0.28, 0.60): latent_std ≈ 0.90

    使用线性插值 std = base + slope * pore_rate 近似。
    """
    pore_rate = max(0.0, min(1.0, 1.0 - rock_rate))
    return _RESCALE_STD_BASE + _RESCALE_STD_SLOPE * pore_rate


def _rescale_patch_variance(
    patch: np.ndarray,
    target_std: float,
    strength: float = 1.0,
    eps: float = 1e-6,
) -> np.ndarray:
    """将 patch latent 的标准差校正到 target_std。

    strength ∈ [0, 1]：0=不修改，1=完全校正到 target_std。
    中间值做线性插值（soft rescale），避免突变。
    """
    if strength <= 0.0 or target_std <= 0.0:
        return patch
    cur_std = float(patch.std())
    if cur_std < eps:
        return patch
    # 目标缩放因子
    ratio = target_std / cur_std
    # soft: 实际 ratio = lerp(1.0, ratio, strength)
    effective_ratio = 1.0 + strength * (ratio - 1.0)
    # 保持均值不变，仅缩放方差
    mean = patch.mean()
    return (patch - mean) * effective_ratio + mean


def load_model(ckpt_path: str, device: torch.device):
    model = ConditionalLatentUNet(
        in_channels=_resolve_in_channels(),
        out_channels=CONFIG["out_channels"],
        base_channels=CONFIG["base_channels"],
        channel_mults=CONFIG["channel_mults"],
        use_attention=CONFIG["use_attention"],
        use_adagn=bool(CONFIG.get("use_adagn", False)),
        cfg_drop_prob=0.0,  # no dropout at inference
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


def ddim_sample(model, cond, mask, phi, porosity, diffusion: DiffusionHelper,
                steps=200, seed=1234, safe_thresh=8.0, cfg_scale=1.0,
                context_cfg_scale=1.0):
    """DDIM sampling with optional Porosity-CFG and Context-CFG.

    Supports two orthogonal guidance axes:

    1. **Porosity-CFG** (``cfg_scale > 1``): amplifies porosity conditioning
       by comparing conditional vs null-porosity predictions.
    2. **Context-CFG** (``context_cfg_scale > 1``): amplifies spatial context
       influence by comparing full-context vs zero-context predictions.
       This requires the model to have been trained with context dropout
       (``context_drop_prob > 0``) so that zero-context is in-distribution.

    Guidance combination (when both active)::

        eps_base = model(cond=0, por=null)          # fully unconditional
        eps_ctx  = model(cond=ctx, por=null)         # + context only
        eps_por  = model(cond=0, por=real)            # + porosity only
        eps_full = model(cond=ctx, por=real)          # fully conditional

        eps = eps_base
              + context_cfg * (eps_ctx - eps_base)    # context guidance
              + cfg_scale   * (eps_por - eps_base)    # porosity guidance

    When only one guidance is active, it reduces to the standard two-call CFG.
    """
    model.eval()
    total_timesteps = diffusion.timesteps
    alphas_cumprod = diffusion.alphas_cumprod
    use_por_cfg = cfg_scale > 1.0 + 1e-6
    use_ctx_cfg = context_cfg_scale > 1.0 + 1e-6

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

    # Pre-build zero-context input for context-CFG (cond & mask zeroed)
    if use_ctx_cfg:
        cond_zero = torch.zeros_like(cond)
        mask_zero = torch.zeros_like(mask)

    with torch.no_grad():
        for i, t in enumerate(times):
            t_tensor = torch.full((cond.shape[0],), t, device=cond.device, dtype=torch.long)
            t_prev = times[i + 1] if i < len(times) - 1 else -1

            model_in = torch.cat([x, cond, mask, phi], dim=1)

            if use_por_cfg and use_ctx_cfg:
                # ─── Dual-axis CFG: 3 forward passes ───
                # 1) fully conditional
                eps_full = model(model_in, t_tensor, porosity)
                # 2) no-porosity (null embedding), with context
                eps_ctx = model(model_in, t_tensor, porosity,
                                force_null_porosity=True)
                # 3) no-context, with porosity
                model_in_noctx = torch.cat([x, cond_zero, mask_zero, phi], dim=1)
                eps_por = model(model_in_noctx, t_tensor, porosity)
                # 4) fully unconditional (no context, no porosity)
                eps_base = model(model_in_noctx, t_tensor, porosity,
                                 force_null_porosity=True)
                # Compose guidance
                eps = (eps_base
                       + context_cfg_scale * (eps_ctx - eps_base)
                       + cfg_scale * (eps_por - eps_base))

            elif use_por_cfg:
                # ─── Porosity-only CFG: 2 forward passes ───
                eps_cond = model(model_in, t_tensor, porosity)
                eps_uncond = model(model_in, t_tensor, porosity,
                                   force_null_porosity=True)
                eps = eps_uncond + cfg_scale * (eps_cond - eps_uncond)

            elif use_ctx_cfg:
                # ─── Context-only CFG: 2 forward passes ───
                eps_with_ctx = model(model_in, t_tensor, porosity)
                model_in_noctx = torch.cat([x, cond_zero, mask_zero, phi], dim=1)
                eps_no_ctx = model(model_in_noctx, t_tensor, porosity)
                eps = eps_no_ctx + context_cfg_scale * (eps_with_ctx - eps_no_ctx)

            else:
                eps = model(model_in, t_tensor, porosity)

            ab_t = alphas_cumprod[t]
            ab_prev = alphas_cumprod[t_prev] if t_prev >= 0 else torch.tensor(1.0, device=cond.device)

            pred_x0 = (x - torch.sqrt(1.0 - ab_t) * eps) / (torch.sqrt(ab_t) + 1e-8)
            pred_x0 = torch.clamp(pred_x0, -safe_thresh, safe_thresh)

            # Recompute eps from clamped pred_x0 to keep DDIM update consistent
            # (follows HuggingFace Diffusers' use_clipped_model_output approach)
            eps = (x - torch.sqrt(ab_t) * pred_x0) / (torch.sqrt(1.0 - ab_t) + 1e-8)

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


def _run_one_pass(
    phi_map: np.ndarray,
    z_full: np.ndarray,
    known_patch: np.ndarray,
    model,
    diffusion,
    patch_size: int,
    window_size: int,
    steps: int,
    seed: int,
    batches: List[List[Coord3D]],
    order: str,
    direction: Direction3D,
    context_mode: str,
    pad_mode: str,
    cfg_scale: float,
    context_cfg_scale: float,
    desc: str = "GeneratePatches",
):
    """Execute a single autoregressive pass over all patches.

    Modifies ``z_full`` and ``known_patch`` **in place**.
    """
    device = next(model.parameters()).device
    C = int(CONFIG["out_channels"])
    w = window_size
    r = w // 2
    pad_p = r * patch_size
    global_phi_val = float(phi_map.mean())
    use_dynamic_porosity_condition = bool(CONFIG.get("use_dynamic_porosity_condition", False))
    dynamic_phi_include_target = bool(CONFIG.get("dynamic_phi_include_target", True))
    dynamic_global_phi_channel = bool(CONFIG.get("dynamic_global_phi_channel", True))
    gD, gH, gW = phi_map.shape
    phi_pad = _pad_with_mode(phi_map, ((r, r), (r, r), (r, r)), pad_mode)
    total_patches = gD * gH * gW
    pbar = tqdm(total=total_patches, desc=desc)
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
            dynamic_mask_patch = mask_patch.copy()
            if dynamic_phi_include_target:
                dynamic_mask_patch[r, r, r] = 1.0
            dynamic_phi_val = _masked_mean_with_fallback(phi_win, dynamic_mask_patch, fallback=global_phi_val)
            global_like_phi_val = dynamic_phi_val if use_dynamic_porosity_condition else global_phi_val
            local_phi_vol = _repeat_phi(phi_win, patch_size)
            if _phi_channels() >= 2:
                global_phi_ch_val = global_like_phi_val if dynamic_global_phi_channel else global_phi_val
                global_phi_patch = np.full_like(phi_win, global_phi_ch_val, dtype=np.float32)
                global_phi_vol = _repeat_phi(global_phi_patch, patch_size)
                phi_vol = np.stack([local_phi_vol, global_phi_vol], axis=0)
            else:
                phi_vol = local_phi_vol[None, ...]

            por_mode = str(CONFIG.get("porosity_mode", "local")).lower()
            local_phi_val = float(phi_map[i, j, k])
            if por_mode == "global":
                porosity = global_like_phi_val
            elif por_mode in ("mix", "local_global_mix"):
                a = float(np.clip(CONFIG.get("porosity_mix_alpha", 0.7), 0.0, 1.0))
                porosity = a * local_phi_val + (1.0 - a) * global_like_phi_val
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
            cfg_scale=cfg_scale,
            context_cfg_scale=context_cfg_scale,
        )
        x_np = x.detach().cpu().float().numpy()

        # Latent variance rescaling 参数
        rescale_strength = float(CONFIG.get("infer_latent_rescale_strength", 0.0))
        scale_factor = float(CONFIG.get("scale_factor", 1.0))

        c0 = r * patch_size
        c1 = c0 + patch_size
        for bi, (i, j, k) in enumerate(coord_batch):
            patch_out = x_np[bi, :, c0:c1, c0:c1, c0:c1]  # (C, p, p, p) scaled

            # ── Per-patch latent variance rescaling ──
            # 模型倾向生成低方差 latent（均值回归），此处根据目标 phi 补偿方差。
            if rescale_strength > 0.0:
                rock_rate_ijk = float(phi_map[i, j, k])
                target_std_unscaled = _expected_latent_std(rock_rate_ijk)
                # 生成的 patch 是 scaled 空间，target 也要转换
                target_std_scaled = target_std_unscaled * scale_factor
                patch_out = _rescale_patch_variance(
                    patch_out, target_std_scaled, strength=rescale_strength
                )

            ti0, tj0, tk0 = i * patch_size, j * patch_size, k * patch_size
            ti1, tj1, tk1 = ti0 + patch_size, tj0 + patch_size, tk0 + patch_size
            z_full[:, ti0:ti1, tj0:tj1, tk0:tk1] = patch_out
            known_patch[i, j, k] = True

        patch_counter += len(coord_batch)
        pbar.update(len(coord_batch))

    pbar.close()


def generate_volume(phi_map: np.ndarray, model, diffusion, patch_size: int, window_size: int, steps: int, seed: int):
    """Generate a full latent volume with optional Draft → Refine passes.

    Pass 0 (Draft): autoregressive generation, early patches have little/no
    context.  Uses fewer DDIM steps (controlled by ``infer_draft_steps_ratio``)
    when refine passes are enabled, for speed.

    Pass 1..N (Refine): re-generate every patch using the previous pass output
    as context.  Every patch now has full-window neighbours, closely matching
    the training distribution.  Uses full ``ddim_steps``.
    """
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

    if context_mode == "full":
        print("[warn] context_mode='full' is not autoregressive for inference; fallback to causal.")
        context_mode = "causal"
    print(f"[infer] traversal order={order}, direction={_direction_str(direction)}, context_mode={context_mode}")

    cfg_scale = float(CONFIG.get("cfg_scale", 1.0))
    context_cfg_scale = float(CONFIG.get("context_cfg_scale", 1.0))
    refine_passes = max(0, int(CONFIG.get("infer_refine_passes", 0)))
    draft_ratio = float(CONFIG.get("infer_draft_steps_ratio", 0.5))

    batches = _build_generation_batches(
        gD=gD, gH=gH, gW=gW,
        window_size=w,
        context_mode=context_mode,
        order=order,
        direction=direction,
        max_patch_batch=max_patch_batch,
    )

    # ─── Draft pass (pass 0) ───
    draft_steps = max(10, int(round(steps * draft_ratio))) if refine_passes > 0 else steps
    print(f"[infer] Draft pass: {draft_steps} DDIM steps (refine_passes={refine_passes})")

    _run_one_pass(
        phi_map=phi_map, z_full=z_full, known_patch=known_patch,
        model=model, diffusion=diffusion,
        patch_size=patch_size, window_size=window_size,
        steps=draft_steps, seed=seed,
        batches=batches, order=order, direction=direction,
        context_mode=context_mode, pad_mode=pad_mode,
        cfg_scale=cfg_scale, context_cfg_scale=context_cfg_scale,
        desc="Draft",
    )

    # ─── Refine passes ───
    # In refine, every patch is re-generated with ALL neighbours already filled
    # from the previous pass → known_patch is all-True, mask is fully populated.
    # This brings inference in line with the training distribution.
    for rp in range(refine_passes):
        refine_seed = seed + 10000 * (rp + 1) if seed is not None else None
        # known_patch stays all-True from the draft pass
        # But we need to re-run generation for every patch
        # Reset known_patch so _run_one_pass fills them in order again
        # BUT keep z_full (previous pass output) as context source
        known_patch[:] = True  # all patches have content from previous pass

        # For refine, we use full steps and full batches
        # Build "refine batches" where every patch can be done in any order
        # since all neighbours are known. We can batch more aggressively.
        refine_batches = _build_generation_batches(
            gD=gD, gH=gH, gW=gW,
            window_size=w,
            context_mode=context_mode,
            order=order,
            direction=direction,
            max_patch_batch=max_patch_batch,
        )

        print(f"[infer] Refine pass {rp+1}/{refine_passes}: {steps} DDIM steps")
        _run_one_pass(
            phi_map=phi_map, z_full=z_full, known_patch=known_patch,
            model=model, diffusion=diffusion,
            patch_size=patch_size, window_size=window_size,
            steps=steps, seed=refine_seed,
            batches=refine_batches, order=order, direction=direction,
            context_mode=context_mode, pad_mode=pad_mode,
            cfg_scale=cfg_scale, context_cfg_scale=context_cfg_scale,
            desc=f"Refine-{rp+1}",
        )

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

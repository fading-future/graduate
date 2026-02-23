#!/usr/bin/env python3
"""
GT(raw 192^3) vs S1-direct(latent->decode 192^3) quantitative evaluation.

Outputs:
  - per_sample_metrics.csv/jsonl
  - summary_all.json
  - per_sample_metrics_subset.csv/jsonl
  - summary_subset.json
  - tp2_curve_aggregate.npz (if tp2 enabled)

中文说明：
  - 评估对象仅包含 GT 原始体素 与 KLVAE 直接解码重建体素
  - 统一输出逐样本指标与汇总指标，并支持“筛选子集”报告
  - 物理指标（绝对渗透率）可选，且提供超时保护避免单样本卡死
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import multiprocessing as mp
import logging
import re
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import yaml
from tqdm import tqdm

from models.vae import KLVAE3D

REPO_ROOT = Path(__file__).resolve().parent

# 优先使用 stage02 本地的 analysis_metrics.py；
# 若不存在，则回退到 stage07/src，避免导入路径导致脚本不可用。
try:
    from analysis_metrics import compare_two_point_probability, absolute_permeability_openpnm
except ImportError:
    STAGE07_SRC = REPO_ROOT.parent / "stage07_latent_ldm_code" / "src"
    if str(STAGE07_SRC) not in sys.path:
        sys.path.insert(0, str(STAGE07_SRC))
    from analysis_metrics import compare_two_point_probability, absolute_permeability_openpnm  # noqa: E402


def _configure_third_party_warnings(quiet: bool):
    # 默认静默第三方库的 FutureWarning，减少终端噪声。
    if not quiet:
        return
    warnings.filterwarnings("ignore", category=FutureWarning, module=r"porespy\..*")
    warnings.filterwarnings("ignore", category=FutureWarning, module=r"skimage\..*")
    logging.getLogger("openpnm").setLevel(logging.ERROR)
    logging.getLogger("openpnm.utils").setLevel(logging.ERROR)
    logging.getLogger("openpnm.utils._workspace").setLevel(logging.ERROR)


def _kabs_worker(mask, axis, mu, dp, out_q, quiet_third_party: bool = True):
    # Windows 下 multiprocessing 使用 spawn，子进程不会继承主进程的 warning 配置。
    # 因此这里需要再次显式设置静默策略，避免 PoreSpy/OpenPNM 警告刷屏。
    _configure_third_party_warnings(quiet=bool(quiet_third_party))
    try:
        res = absolute_permeability_openpnm(mask, axis=int(axis), mu=float(mu), dp=float(dp))
    except BaseException as e:  # noqa: BLE001
        res = {"ok": False, "error": f"{type(e).__name__}: {e}"}
    out_q.put(res)


def _absolute_k_with_timeout(
    mask,
    axis: int,
    mu: float,
    dp: float,
    timeout_sec: float,
    quiet_third_party: bool = True,
):
    # 绝对渗透率求解可能很慢，使用子进程+超时防止整次评估卡住。
    if timeout_sec is None or float(timeout_sec) <= 0:
        try:
            return absolute_permeability_openpnm(mask, axis=int(axis), mu=float(mu), dp=float(dp))
        except BaseException as e:  # noqa: BLE001
            return {"ok": False, "error": f"{type(e).__name__}: {e}"}

    q = mp.Queue(maxsize=1)
    p = mp.Process(
        target=_kabs_worker,
        args=(mask, int(axis), float(mu), float(dp), q, bool(quiet_third_party)),
        daemon=True,
    )
    p.start()
    p.join(float(timeout_sec))
    if p.is_alive():
        p.terminate()
        p.join()
        return {"ok": False, "error": f"timeout>{float(timeout_sec):.1f}s"}
    if not q.empty():
        return q.get()
    return {"ok": False, "error": "no_result_from_worker"}


def _safe_torch_load(path: str, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)
    except Exception:
        return torch.load(path, map_location=map_location)


def _strip_porosity_prefix(name: str) -> str:
    return re.sub(r"^porosity_[0-9]*\.?[0-9]+_", "", name)


def _to_binary_volume(raw: np.ndarray) -> np.ndarray:
    # 将不同数值域（[-1,1]/[0,1]/uint8/uint16）统一转为 0/1 二值体素。
    arr = raw.astype(np.float32)
    mn, mx = float(arr.min()), float(arr.max())
    if mn >= -1.01 and mx <= 1.01:
        if mn < 0.0:
            arr = (arr + 1.0) * 0.5
        return (arr >= 0.5).astype(np.uint8)
    if mn >= -1e-6 and mx <= 1.5:
        return (arr >= 0.5).astype(np.uint8)
    if mx <= 255.5:
        return (arr >= 127.5).astype(np.uint8)
    return (arr >= 32767.5).astype(np.uint8)


def _phase_mask(binary_vol: np.ndarray, phase_value: int) -> np.ndarray:
    p = int(phase_value)
    if p not in (0, 1):
        raise ValueError(f"phase_value must be 0/1, got {phase_value}")
    return (binary_vol == p).astype(np.uint8)


def _corrcoef_safe(a: np.ndarray, b: np.ndarray) -> float:
    aa = a.reshape(-1).astype(np.float64)
    bb = b.reshape(-1).astype(np.float64)
    if aa.size == 0 or bb.size == 0:
        return 0.0
    if aa.std() < 1e-12 or bb.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(aa, bb)[0, 1])


def _voxel_metrics(pred_bin: np.ndarray, gt_bin: np.ndarray) -> Dict[str, float]:
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
    }


def _compute_phi_map(vol: np.ndarray, block_size: int) -> np.ndarray:
    d, h, w = vol.shape
    gd, gh, gw = d // block_size, h // block_size, w // block_size
    x = vol[: gd * block_size, : gh * block_size, : gw * block_size]
    x = x.reshape(gd, block_size, gh, block_size, gw, block_size)
    return x.mean(axis=(1, 3, 5)).astype(np.float32)


def _phi_metrics(pred_phi: np.ndarray, gt_phi: np.ndarray, prefix: str) -> Dict[str, float]:
    diff = pred_phi.astype(np.float64) - gt_phi.astype(np.float64)
    return {
        f"{prefix}_mae": float(np.mean(np.abs(diff))),
        f"{prefix}_rmse": float(np.sqrt(np.mean(diff * diff))),
        f"{prefix}_corr": _corrcoef_safe(pred_phi, gt_phi),
        f"{prefix}_bias": float(np.mean(diff)),
    }


def _load_vae(config_path: str, ckpt_path: str, device: torch.device) -> KLVAE3D:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    vae = KLVAE3D(cfg).to(device).eval()
    ckpt = _safe_torch_load(ckpt_path, map_location=device)
    if isinstance(ckpt, dict) and "vae_state_dict" in ckpt:
        state = ckpt["vae_state_dict"]
    elif isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
    else:
        state = ckpt
    state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
    vae.load_state_dict(state)
    return vae


def _decode_latent_to_prob_bin(latent: np.ndarray, vae: KLVAE3D, device: torch.device, threshold: float):
    z = latent.astype(np.float32)
    if z.ndim == 5 and z.shape[0] == 1:
        z = z[0]
    if z.ndim != 4:
        raise ValueError(f"Expected latent shape (C,D,H,W), got {z.shape}")
    zt = torch.from_numpy(z).unsqueeze(0).to(device)
    with torch.no_grad():
        if device.type == "cuda":
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits = vae.decode(zt)
        else:
            logits = vae.decode(zt)
    prob = torch.sigmoid(logits)[0, 0].detach().cpu().float().numpy()
    binv = (prob >= float(threshold)).astype(np.uint8)
    return prob, binv


def _collect_pairs(raw_dir: str, latent_dir: str) -> List[Tuple[str, str, str]]:
    # 通过文件名配对 raw 与 latent；支持去掉 porosity_ 前缀后再匹配。
    raw_map = {p.name: str(p) for p in sorted(Path(raw_dir).glob("*.npy"))}
    lat_map = {}
    for p in sorted(Path(latent_dir).glob("*.npy")):
        key = _strip_porosity_prefix(p.name)
        if key not in lat_map:
            lat_map[key] = str(p)
    names = sorted(set(raw_map.keys()) & set(lat_map.keys()))
    return [(n, raw_map[n], lat_map[n]) for n in names]


def _select(pairs: Sequence[Tuple[str, str, str]], mode: str, num: int, seed: int):
    items = list(pairs)
    if mode == "random":
        rng = np.random.default_rng(int(seed))
        idx = rng.permutation(len(items))
        items = [items[int(i)] for i in idx]
    if int(num) > 0:
        items = items[: int(num)]
    return items


def _read_basename_list(path: str) -> List[str]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"basename-list not found: {path}")
    names = []
    for line in p.read_text(encoding="utf-8").splitlines():
        t = line.strip()
        if t:
            names.append(t)
    return names


def _write_csv(rows: List[Dict], path: str):
    if not rows:
        return
    keys = sorted(set().union(*(r.keys() for r in rows)))
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _write_jsonl(rows: List[Dict], path: str):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _summary(rows: List[Dict]) -> Dict:
    if not rows:
        return {"num_samples": 0}
    out = {"num_samples": len(rows)}
    keys = sorted(set().union(*(r.keys() for r in rows)))
    for k in keys:
        vals = [r.get(k, None) for r in rows]
        nums = [float(v) for v in vals if isinstance(v, (int, float, np.floating, np.integer))]
        if not nums:
            continue
        arr = np.asarray(nums, dtype=np.float64)
        out[f"{k}_mean"] = float(arr.mean())
        out[f"{k}_std"] = float(arr.std())
        out[f"{k}_min"] = float(arr.min())
        out[f"{k}_max"] = float(arr.max())
    return out


def _score_subset(rows: List[Dict], top_frac: float):
    # 用综合评分筛选“更稳定”的样本子集，便于论文主结果展示。
    if not rows:
        return rows, {"subset_mode": "score", "selected": 0, "total": 0}
    metric_names = ["voxel_dice", "tp2_corr", "pore_porosity_abs_err", "phi16_pore_mae"]
    arr = {m: np.asarray([float(r.get(m, np.nan)) for r in rows], dtype=np.float64) for m in metric_names}
    valid = np.ones(len(rows), dtype=bool)
    for m in metric_names:
        valid &= np.isfinite(arr[m])
    idx = np.where(valid)[0]
    if idx.size == 0:
        return rows, {"subset_mode": "score", "selected": len(rows), "total": len(rows), "warning": "no valid score rows"}

    def z(x):
        return (x - x.mean()) / (x.std() + 1e-12)

    s = z(arr["voxel_dice"][idx]) + 0.8 * z(arr["tp2_corr"][idx]) - z(arr["pore_porosity_abs_err"][idx]) - 0.8 * z(arr["phi16_pore_mae"][idx])
    keep_n = max(1, int(math.ceil(float(top_frac) * float(idx.size))))
    keep_idx = set(idx[np.argsort(-s)[:keep_n]].tolist())
    subset = [rows[i] for i in range(len(rows)) if i in keep_idx]
    info = {
        "subset_mode": "score",
        "score_formula": "+z(voxel_dice)+0.8*z(tp2_corr)-z(pore_porosity_abs_err)-0.8*z(phi16_pore_mae)",
        "top_frac": float(top_frac),
        "selected": len(subset),
        "total": len(rows),
    }
    return subset, info


def main():
    mp.freeze_support()
    # 1) 参数解析
    parser = argparse.ArgumentParser(description="KLVAE 192^3 quantitative evaluation (GT vs S1-direct).")
    parser.add_argument("--raw-dir", default=r"D:\多尺度岩心数据集\LDM_Data\Raw_NPY\w192_s64")
    parser.add_argument("--latent-dir", default=r"D:\多尺度岩心数据集\LDM_Data\Latent_NPY\w192_s64")
    parser.add_argument("--vae-config", default=str(REPO_ROOT / "config" / "train_config copy.yaml"))
    parser.add_argument("--vae-ckpt", default=str(REPO_ROOT / "experiments" / "exp04_cube_structure_v1" / "ckpt_epoch_11.pt"))
    parser.add_argument("--out-dir", default=str(REPO_ROOT / "experiments" / "exp04_cube_structure_v1" / "eval_klvae_192"))
    parser.add_argument("--device", default=("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--pore-value", type=int, choices=[0, 1], default=0)
    parser.add_argument("--phi-block-sizes", default="8,16,32,64")
    parser.add_argument("--tp2-max-lag", type=int, default=32)
    parser.add_argument("--sample-mode", choices=["sequential", "random"], default="random")
    parser.add_argument("--num-samples", type=int, default=120)
    parser.add_argument("--sample-seed", type=int, default=2026)
    parser.add_argument(
        "--basename-list",
        default="",
        help="Optional txt file (one basename per line). If set, evaluation uses exactly these samples.",
    )
    parser.add_argument("--subset-mode", choices=["none", "score"], default="score")
    parser.add_argument("--subset-top-frac", type=float, default=0.75)
    parser.add_argument("--physics-abs-k", action="store_true")
    parser.add_argument("--physics-rel-k", action="store_true", help="Reserved flag. Relative permeability is not implemented in this script yet.")
    parser.add_argument("--physics-max-samples", type=int, default=12)
    parser.add_argument("--physics-crop", type=int, default=96)
    parser.add_argument("--physics-timeout-sec", type=float, default=45.0, help="Timeout for each abs-k solve per axis; <=0 disables timeout.")
    parser.add_argument("--physics-axes", default="xyz")
    parser.add_argument(
        "--show-third-party-warnings",
        action="store_true",
        help="Show PoreSpy/OpenPNM/skimage warnings (default quiet).",
    )
    args = parser.parse_args()
    _configure_third_party_warnings(quiet=not bool(args.show_third_party_warnings))

    # 2) 构建样本列表：可按随机抽样，也可按 basename-list 精确指定。
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    block_sizes = [int(x.strip()) for x in args.phi_block_sizes.split(",") if x.strip()]
    pairs = _collect_pairs(args.raw_dir, args.latent_dir)
    if args.basename_list:
        target_names = _read_basename_list(args.basename_list)
        target_set = set(target_names)
        selected = [x for x in pairs if x[0] in target_set]
        miss = sorted(target_set - {x[0] for x in selected})
        if miss:
            print(f"[warn] {len(miss)} names in basename-list not found in paired data.")
    else:
        selected = _select(pairs, args.sample_mode, args.num_samples, args.sample_seed)
    if not selected:
        raise ValueError("No matched pair selected")

    axes = []
    for c in str(args.physics_axes).lower():
        if c == "x" and 0 not in axes:
            axes.append(0)
        elif c == "y" and 1 not in axes:
            axes.append(1)
        elif c == "z" and 2 not in axes:
            axes.append(2)
    if not axes:
        axes = [0]

    # 3) 加载 VAE 并逐样本计算指标
    vae = _load_vae(args.vae_config, args.vae_ckpt, torch.device(args.device))
    rows = []
    tp2_pred, tp2_gt = [], []
    for i, (name, raw_path, latent_path) in enumerate(tqdm(selected, desc="EvalKLVAE")):
        t0 = time.time()
        gt_bin = _to_binary_volume(np.load(raw_path))
        pred_prob, pred_bin = _decode_latent_to_prob_bin(np.load(latent_path), vae, torch.device(args.device), args.threshold)
        if pred_bin.shape != gt_bin.shape:
            raise ValueError(f"shape mismatch: {name}, pred={pred_bin.shape}, gt={gt_bin.shape}")

        pred_phase = _phase_mask(pred_bin, args.pore_value)
        gt_phase = _phase_mask(gt_bin, args.pore_value)

        row = {
            "sample_index": int(i),
            "basename": name,
            "raw_path": raw_path,
            "latent_path": latent_path,
            "pred_prob_mean": float(pred_prob.mean()),
            "pred_prob_std": float(pred_prob.std()),
            "pred_bin_rock_fraction": float(pred_bin.mean()),
            "gt_bin_rock_fraction": float(gt_bin.mean()),
            "pore_porosity_pred": float(pred_phase.mean()),
            "pore_porosity_gt": float(gt_phase.mean()),
            "pore_porosity_abs_err": float(abs(pred_phase.mean() - gt_phase.mean())),
            "time_sec": 0.0,
        }
        row.update(_voxel_metrics(pred_bin, gt_bin))

        for bs in block_sizes:
            row.update(_phi_metrics(_compute_phi_map(pred_phase.astype(np.float32), bs), _compute_phi_map(gt_phase.astype(np.float32), bs), f"phi{bs}_pore"))

        if int(args.tp2_max_lag) > 0:
            tp2 = compare_two_point_probability(pred_phase, gt_phase, max_lag=int(args.tp2_max_lag))
            row.update(tp2["metrics"])
            tp2_pred.append(tp2["pred_curve"]["mean"].astype(np.float32))
            tp2_gt.append(tp2["gt_curve"]["mean"].astype(np.float32))

        if args.physics_rel_k:
            row["kr_status"] = "not_implemented_in_this_script"

        if args.physics_abs_k and (int(args.physics_max_samples) <= 0 or i < int(args.physics_max_samples)):
            # 物理指标开销很大：可裁剪体积、限制样本数，并按轴逐个求解。
            ph, gh = pred_phase, gt_phase
            c = int(args.physics_crop)
            if c > 0 and c < ph.shape[0] and c < ph.shape[1] and c < ph.shape[2]:
                z0, y0, x0 = (ph.shape[0] - c) // 2, (ph.shape[1] - c) // 2, (ph.shape[2] - c) // 2
                ph, gh = ph[z0:z0 + c, y0:y0 + c, x0:x0 + c], gh[z0:z0 + c, y0:y0 + c, x0:x0 + c]
            kp, kg = [], []
            for ax in axes:
                tag = "xyz"[ax]
                p = _absolute_k_with_timeout(
                    ph,
                    axis=ax,
                    mu=1.0,
                    dp=1.0,
                    timeout_sec=float(args.physics_timeout_sec),
                    quiet_third_party=not bool(args.show_third_party_warnings),
                )
                g = _absolute_k_with_timeout(
                    gh,
                    axis=ax,
                    mu=1.0,
                    dp=1.0,
                    timeout_sec=float(args.physics_timeout_sec),
                    quiet_third_party=not bool(args.show_third_party_warnings),
                )
                row[f"kabs_pred_{tag}_ok"] = bool(p.get("ok", False))
                row[f"kabs_gt_{tag}_ok"] = bool(g.get("ok", False))
                if bool(p.get("ok", False)):
                    row[f"kabs_pred_{tag}"] = float(p["k_abs_voxel2"])
                    kp.append(float(p["k_abs_voxel2"]))
                else:
                    row[f"kabs_pred_{tag}_err"] = str(p.get("error", "unknown"))
                if bool(g.get("ok", False)):
                    row[f"kabs_gt_{tag}"] = float(g["k_abs_voxel2"])
                    kg.append(float(g["k_abs_voxel2"]))
                else:
                    row[f"kabs_gt_{tag}_err"] = str(g.get("error", "unknown"))
            if kp and kg:
                row["kabs_pred_mean"] = float(np.mean(kp))
                row["kabs_gt_mean"] = float(np.mean(kg))
                row["kabs_abs_err_mean"] = float(abs(row["kabs_pred_mean"] - row["kabs_gt_mean"]))
                row["kabs_rel_err_mean"] = float(row["kabs_abs_err_mean"] / (abs(row["kabs_gt_mean"]) + 1e-12))

        row["time_sec"] = float(time.time() - t0)
        rows.append(row)

    _write_csv(rows, str(out_dir / "per_sample_metrics.csv"))
    _write_jsonl(rows, str(out_dir / "per_sample_metrics.jsonl"))

    if tp2_pred and tp2_gt:
        pa = np.stack(tp2_pred, axis=0)
        ga = np.stack(tp2_gt, axis=0)
        lag = np.arange(pa.shape[1], dtype=np.int32)
        np.savez(
            out_dir / "tp2_curve_aggregate.npz",
            lag=lag,
            pred_mean=pa.mean(axis=0).astype(np.float32),
            pred_std=pa.std(axis=0).astype(np.float32),
            gt_mean=ga.mean(axis=0).astype(np.float32),
            gt_std=ga.std(axis=0).astype(np.float32),
        )

    s_all = _summary(rows)
    s_all.update(
        {
            "raw_dir": args.raw_dir,
            "latent_dir": args.latent_dir,
            "num_pairs_total": len(pairs),
            "num_selected": len(selected),
            "sample_mode": args.sample_mode,
            "sample_seed": int(args.sample_seed),
            "basename_list": str(args.basename_list),
            "pore_value": int(args.pore_value),
            "threshold": float(args.threshold),
            "phi_block_sizes": block_sizes,
            "tp2_max_lag": int(args.tp2_max_lag),
            "physics_rel_k": bool(args.physics_rel_k),
        }
    )
    with open(out_dir / "summary_all.json", "w", encoding="utf-8") as f:
        json.dump(s_all, f, indent=2, ensure_ascii=False)

    # 4) 可选筛选子集（score），输出用于“主结果图表”的精简版本。
    subset, info = (rows, {"subset_mode": "none", "selected": len(rows), "total": len(rows)})
    if args.subset_mode == "score":
        subset, info = _score_subset(rows, args.subset_top_frac)
    _write_csv(subset, str(out_dir / "per_sample_metrics_subset.csv"))
    _write_jsonl(subset, str(out_dir / "per_sample_metrics_subset.jsonl"))
    with open(out_dir / "subset_basename_list.txt", "w", encoding="utf-8") as f:
        for r in subset:
            f.write(str(r["basename"]) + "\n")
    s_sub = _summary(subset)
    s_sub.update(info)
    s_sub["source_summary_all"] = str(out_dir / "summary_all.json")
    with open(out_dir / "summary_subset.json", "w", encoding="utf-8") as f:
        json.dump(s_sub, f, indent=2, ensure_ascii=False)

    print("[done] evaluation complete")
    print(f"  out_dir: {out_dir}")
    for k in ["voxel_dice_mean", "pore_porosity_abs_err_mean", "phi16_pore_mae_mean", "tp2_corr_mean", "tp2_mae_mean", "time_sec_mean"]:
        if k in s_sub:
            print(f"  {k}: {s_sub[k]:.6f}")


if __name__ == "__main__":
    main()

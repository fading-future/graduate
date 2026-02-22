import argparse
import csv
import itertools
import json
import math
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch


DEFAULT_SCORE_TERMS = [
    "2.0*voxel_dice_mean",
    "1.2*pore_parity_corr|phase_bin_phi_corr_mean|bin_phi_corr_mean",
    "-0.8*abs(pore_parity_slope_gap)",
    "-0.8*abs(pore_parity_bias)",
    "-1.2*target_phase_fraction_abs_err_mean|pore_porosity_abs_err_mean|porosity_abs_err_mean",
    "-0.9*phase_bin_phi_mae_mean|bin_phi_mae_mean",
    "-0.3*abs(z_head_phase_gap_mean|z_head_porosity_gap_mean)",
    "-0.3*abs(z_tail_phase_gap_mean|z_tail_porosity_gap_mean)",
    "-0.15*time_sec_mean",
]


@dataclass
class ScoreTerm:
    raw: str
    weight: float
    use_abs: bool
    keys: List[str]


def _parse_csv_tokens(text: str) -> List[str]:
    out = []
    for tok in str(text).split(","):
        t = tok.strip()
        if t:
            out.append(t)
    return out


def _parse_bool_token(tok: str) -> bool:
    t = str(tok).strip().lower()
    if t in ("1", "true", "t", "yes", "y", "on"):
        return True
    if t in ("0", "false", "f", "no", "n", "off"):
        return False
    raise ValueError(f"invalid bool token: {tok}")


def _parse_bool_list(text: str) -> List[bool]:
    vals = [_parse_bool_token(t) for t in _parse_csv_tokens(text)]
    if not vals:
        raise ValueError("bool option list cannot be empty")
    return vals


def _parse_int_list(text: str) -> List[int]:
    vals = [int(t) for t in _parse_csv_tokens(text)]
    if not vals:
        raise ValueError("int option list cannot be empty")
    return vals


def _parse_optional_int_list(text: str) -> List[Optional[int]]:
    out: List[Optional[int]] = []
    for tok in _parse_csv_tokens(text):
        t = tok.lower()
        if t in ("none", "null", "na"):
            out.append(None)
        else:
            out.append(int(tok))
    if not out:
        raise ValueError("seed option list cannot be empty")
    return out


def _safe_float(v) -> Optional[float]:
    if v is None:
        return None
    try:
        x = float(v)
    except Exception:
        return None
    if math.isnan(x) or math.isinf(x):
        return None
    return x


def _dir_to_code(direction: str) -> str:
    direction = str(direction).strip()
    if len(direction) != 3 or not set(direction).issubset({"+", "-"}):
        raise ValueError(f"invalid direction token: {direction}")
    return "".join("p" if c == "+" else "m" for c in direction)


def _sanitize_name(text: str, max_len: int = 120) -> str:
    s = re.sub(r"[^0-9a-zA-Z_.-]+", "_", text.strip())
    if not s:
        s = "run"
    if len(s) > max_len:
        s = s[:max_len]
    return s


def _find_repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _resolve_path(path_str: str, repo_root: Path) -> str:
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


def _collect_ckpts(models_dir: Path, ckpt_glob: str, epochs: List[int], ckpt_list: List[str], repo_root: Path) -> List[str]:
    out: List[Path] = []

    for ckpt in ckpt_list:
        p = Path(_resolve_path(ckpt, repo_root))
        if not p.exists():
            raise FileNotFoundError(f"ckpt not found: {p}")
        out.append(p)

    if not ckpt_list:
        for p in sorted(models_dir.glob(ckpt_glob)):
            if not p.is_file():
                continue
            out.append(p.resolve())

    if epochs:
        ep_set = set(int(e) for e in epochs)
        filtered = []
        for p in out:
            m = re.search(r"epoch_(\d+)", p.name)
            if m is None:
                continue
            if int(m.group(1)) in ep_set:
                filtered.append(p)
        out = filtered

    uniq = []
    seen = set()
    for p in out:
        s = str(p.resolve())
        if s in seen:
            continue
        seen.add(s)
        uniq.append(s)

    if not uniq:
        raise ValueError("no checkpoint selected after filtering")
    return uniq


def _parse_score_terms(text: str) -> List[ScoreTerm]:
    parts = _parse_csv_tokens(text)
    if not parts:
        raise ValueError("score terms cannot be empty")

    out: List[ScoreTerm] = []
    for raw in parts:
        m = re.match(r"^\s*([+-]?\d*\.?\d+)\s*\*\s*(.+?)\s*$", raw)
        if m:
            weight = float(m.group(1))
            expr = m.group(2).strip()
        else:
            weight = 1.0
            expr = raw.strip()

        use_abs = False
        abs_match = re.match(r"^abs\((.+)\)$", expr, flags=re.IGNORECASE)
        if abs_match:
            use_abs = True
            expr = abs_match.group(1).strip()

        keys = [k.strip() for k in expr.split("|") if k.strip()]
        if not keys:
            raise ValueError(f"invalid score term: {raw}")
        out.append(ScoreTerm(raw=raw, weight=weight, use_abs=use_abs, keys=keys))
    return out


def _pick_metric(summary: Dict, keys: Sequence[str]) -> Optional[float]:
    for k in keys:
        if k in summary:
            v = _safe_float(summary.get(k))
            if v is not None:
                return v
    return None


def _score_summary(summary: Dict, terms: Sequence[ScoreTerm]) -> Tuple[float, List[Dict]]:
    total = 0.0
    details: List[Dict] = []
    for t in terms:
        v = _pick_metric(summary, t.keys)
        if v is None:
            details.append(
                {
                    "term": t.raw,
                    "used_key": None,
                    "value": None,
                    "contribution": 0.0,
                    "missing": True,
                }
            )
            continue

        used_key = next(k for k in t.keys if _safe_float(summary.get(k)) is not None)
        vv = abs(v) if t.use_abs else v
        c = t.weight * vv
        total += c
        details.append(
            {
                "term": t.raw,
                "used_key": used_key,
                "value": vv,
                "contribution": c,
                "missing": False,
            }
        )
    return total, details


def _maybe_add_pore_parity_from_per_sample(summary: Dict, summary_path: Path) -> Dict:
    # Backfill parity metrics from per-sample table for legacy summaries.
    if (
        _safe_float(summary.get("pore_parity_corr")) is not None
        and _safe_float(summary.get("pore_parity_slope")) is not None
        and _safe_float(summary.get("pore_parity_bias")) is not None
    ):
        return summary

    per_sample_csv = summary_path.parent / "per_sample_metrics.csv"
    if not per_sample_csv.exists():
        return summary

    xs: List[float] = []
    ys: List[float] = []
    try:
        with open(per_sample_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                gx = _safe_float(r.get("pore_porosity_gt_aligned"))
                gy = _safe_float(r.get("pore_porosity_pred_aligned"))
                if gx is None or gy is None:
                    continue
                xs.append(float(gx))
                ys.append(float(gy))
    except Exception:
        return summary

    if len(xs) == 0:
        return summary

    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    xm = float(x.mean())
    ym = float(y.mean())
    dx = x - xm
    dy = y - ym
    vx = float(np.sum(dx * dx))
    vy = float(np.sum(dy * dy))
    cov = float(np.sum(dx * dy))

    corr = cov / (math.sqrt(vx * vy) + 1e-12) if vx > 0.0 and vy > 0.0 else 0.0
    slope = cov / (vx + 1e-12) if vx > 0.0 else 0.0
    bias = ym - xm
    mae = float(np.mean(np.abs(y - x)))
    rmse = float(np.sqrt(np.mean((y - x) ** 2)))

    summary = dict(summary)
    summary.setdefault("pore_parity_corr", float(corr))
    summary.setdefault("pore_parity_slope", float(slope))
    summary.setdefault("pore_parity_bias", float(bias))
    summary.setdefault("pore_parity_mae", float(mae))
    summary.setdefault("pore_parity_rmse", float(rmse))
    summary.setdefault("pore_parity_slope_gap", float(slope - 1.0))
    summary.setdefault("pore_parity_n", int(x.size))
    return summary


def _build_eval_cmd(args, run_cfg: Dict, out_dir: Path, repo_root: Path) -> List[str]:
    cmd = [
        sys.executable,
        "-m",
        "src.evaluate_batch_generated",
        "--ckpt",
        run_cfg["ckpt"],
        "--out-dir",
        str(out_dir),
        "--device",
        args.device,
        "--threshold",
        str(args.threshold),
        "--pore-value",
        str(args.pore_value),
        "--gt-phi-semantic",
        args.gt_phi_semantic,
        "--ddim-steps",
        str(run_cfg["ddim_steps"]),
        "--offset",
        str(args.offset),
        "--num-samples",
        str(args.num_samples),
        "--sample-mode",
        args.sample_mode,
        "--sample-seed",
        str(run_cfg["sample_seed"]),
        "--seed-mode",
        args.seed_mode,
        "--infer-weight-source",
        run_cfg["infer_weight_source"],
    ]

    if args.phi_dir:
        cmd.extend(["--phi-dir", args.phi_dir])
    if args.latent_dir:
        cmd.extend(["--latent-dir", args.latent_dir])
    if args.raw_dir:
        cmd.extend(["--raw-dir", args.raw_dir])
    if args.vae_config:
        cmd.extend(["--vae-config", args.vae_config])
    if args.vae_ckpt:
        cmd.extend(["--vae-ckpt", args.vae_ckpt])

    if run_cfg["infer_seed"] is not None:
        cmd.extend(["--seed", str(run_cfg["infer_seed"])])

    if run_cfg["infer_random_order"]:
        cmd.append("--infer-random-order")
    else:
        cmd.extend(["--infer-order", run_cfg["infer_order"]])

    if run_cfg["infer_random_direction"]:
        cmd.append("--infer-random-direction")
    else:
        cmd.extend(["--infer-direction-code", _dir_to_code(run_cfg["infer_direction"])])

    if args.tp2_max_lag > 0:
        cmd.extend(["--tp2-max-lag", str(args.tp2_max_lag), "--tp2-phase", args.tp2_phase])
    if args.export_phi_cells:
        cmd.append("--export-phi-cells")
    if args.save_each:
        cmd.append("--save-each")
    if args.physics_abs_k:
        cmd.extend(
            [
                "--physics-abs-k",
                "--physics-max-samples",
                str(args.physics_max_samples),
                "--physics-crop",
                str(args.physics_crop),
                "--physics-axes",
                args.physics_axes,
                "--physics-mu",
                str(args.physics_mu),
                "--physics-dp",
                str(args.physics_dp),
            ]
        )
    if args.show_third_party_warnings:
        cmd.append("--show-third-party-warnings")

    return cmd


def _run_one(cmd: List[str], log_path: Path, cwd: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("COMMAND:\n")
        f.write(subprocess.list2cmdline(cmd))
        f.write("\n\n")
        f.flush()
        p = subprocess.run(cmd, cwd=str(cwd), stdout=f, stderr=subprocess.STDOUT)
    return int(p.returncode)


def _make_run_name(idx: int, cfg: Dict) -> str:
    ckpt_name = Path(cfg["ckpt"]).stem
    parts = [
        f"{idx:04d}",
        ckpt_name,
        f"w{cfg['infer_weight_source']}",
        f"ddim{cfg['ddim_steps']}",
        f"sseed{cfg['sample_seed']}",
        f"iseed{cfg['infer_seed'] if cfg['infer_seed'] is not None else 'none'}",
        "ro1" if cfg["infer_random_order"] else f"ro0_{cfg['infer_order']}",
        "rd1" if cfg["infer_random_direction"] else f"rd0_{cfg['infer_direction']}",
    ]
    return _sanitize_name("__".join(parts), max_len=150)


def _collect_metric_columns(rows: Sequence[Dict]) -> List[str]:
    cols = set()
    for r in rows:
        for k in r.keys():
            cols.add(k)
    return sorted(cols)


def _write_csv(rows: Sequence[Dict], out_csv: Path):
    if not rows:
        return
    keys = _collect_metric_columns(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def _expand_run_space(args, ckpts: Sequence[str]) -> List[Dict]:
    run_cfgs: List[Dict] = []

    infer_weight_sources = _parse_csv_tokens(args.infer_weight_sources)
    if not infer_weight_sources:
        raise ValueError("infer-weight-sources cannot be empty")

    random_order_options = _parse_bool_list(args.infer_random_order_options)
    random_direction_options = _parse_bool_list(args.infer_random_direction_options)
    infer_orders = _parse_csv_tokens(args.infer_orders)
    infer_directions = _parse_csv_tokens(args.infer_directions)
    sample_seeds = _parse_int_list(args.sample_seed_options)
    infer_seeds = _parse_optional_int_list(args.infer_seed_options)
    ddim_steps_list = _parse_int_list(args.ddim_steps_options)

    for (
        ckpt,
        infer_weight_source,
        infer_random_order,
        infer_random_direction,
        sample_seed,
        infer_seed,
        ddim_steps,
    ) in itertools.product(
        ckpts,
        infer_weight_sources,
        random_order_options,
        random_direction_options,
        sample_seeds,
        infer_seeds,
        ddim_steps_list,
    ):
        order_vals = [None] if infer_random_order else infer_orders
        direction_vals = [None] if infer_random_direction else infer_directions

        if not order_vals:
            raise ValueError("infer-orders cannot be empty when infer-random-order includes false")
        if not direction_vals:
            raise ValueError("infer-directions cannot be empty when infer-random-direction includes false")

        for infer_order, infer_direction in itertools.product(order_vals, direction_vals):
            cfg = {
                "ckpt": ckpt,
                "infer_weight_source": infer_weight_source,
                "infer_random_order": bool(infer_random_order),
                "infer_random_direction": bool(infer_random_direction),
                "infer_order": infer_order if infer_order is not None else "",
                "infer_direction": infer_direction if infer_direction is not None else "",
                "sample_seed": int(sample_seed),
                "infer_seed": None if infer_seed is None else int(infer_seed),
                "ddim_steps": int(ddim_steps),
            }
            run_cfgs.append(cfg)

    if args.max_runs > 0:
        run_cfgs = run_cfgs[: int(args.max_runs)]
    if not run_cfgs:
        raise ValueError("no run configs generated")
    return run_cfgs


def _args_parser() -> argparse.ArgumentParser:
    repo_root = _find_repo_root()
    default_models_dir = repo_root / "exp_results" / "stage07_patch_ldm_v4" / "models"
    default_out_dir = repo_root / "exp_results" / "stage07_patch_ldm_v4" / "search_runs"
    p = argparse.ArgumentParser(description="Search best inference config by running batch evaluation grid.")
    p.add_argument("--models-dir", default=str(default_models_dir))
    p.add_argument("--ckpt-glob", default="unet_epoch_*.pth")
    p.add_argument("--epochs", default="", help="Optional epochs, e.g. 20,30,40")
    p.add_argument("--ckpt-list", default="", help="Optional explicit ckpt paths, comma-separated")
    p.add_argument("--out-dir", default=str(default_out_dir))

    p.add_argument("--phi-dir", default="")
    p.add_argument("--latent-dir", default="")
    p.add_argument("--raw-dir", default="")
    p.add_argument("--vae-config", default="")
    p.add_argument("--vae-ckpt", default="")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--pore-value", type=int, choices=[0, 1], default=0)
    p.add_argument("--gt-phi-semantic", choices=["rock_rate", "porosity"], default="rock_rate")

    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--num-samples", type=int, default=20)
    p.add_argument("--sample-mode", choices=["sequential", "random"], default="random")
    p.add_argument("--sample-seed-options", default="1234,2026")
    p.add_argument("--seed-mode", choices=["fixed", "offset", "name_hash"], default="name_hash")
    p.add_argument("--infer-seed-options", default="6666")
    p.add_argument("--ddim-steps-options", default="200,300")

    p.add_argument("--infer-weight-sources", default="ema,model")
    p.add_argument("--infer-random-order-options", default="true,false")
    p.add_argument("--infer-random-direction-options", default="true,false")
    p.add_argument("--infer-orders", default="ijk")
    p.add_argument("--infer-directions", default="+++,---,+-+,-++")

    p.add_argument("--tp2-max-lag", type=int, default=0)
    p.add_argument("--tp2-phase", choices=["raw", "phase"], default="phase")
    p.add_argument("--export-phi-cells", action="store_true")
    p.add_argument("--save-each", action="store_true")
    p.add_argument("--show-third-party-warnings", action="store_true")

    p.add_argument("--physics-abs-k", action="store_true")
    p.add_argument("--physics-max-samples", type=int, default=0)
    p.add_argument("--physics-crop", type=int, default=0)
    p.add_argument("--physics-axes", default="xyz")
    p.add_argument("--physics-mu", type=float, default=1.0)
    p.add_argument("--physics-dp", type=float, default=1.0)

    p.add_argument("--max-runs", type=int, default=0, help="0 means all generated combinations")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--resume", action="store_true", help="Skip run if summary.json already exists")
    p.add_argument("--score-terms", default=",".join(DEFAULT_SCORE_TERMS))

    return p


def main():
    parser = _args_parser()
    args = parser.parse_args()
    repo_root = _find_repo_root()

    models_dir = Path(_resolve_path(args.models_dir, repo_root))
    out_dir = Path(_resolve_path(args.out_dir, repo_root))
    out_dir.mkdir(parents=True, exist_ok=True)

    if not models_dir.exists():
        raise FileNotFoundError(f"models-dir not found: {models_dir}")

    epochs = [int(x) for x in _parse_csv_tokens(args.epochs)] if args.epochs else []
    ckpt_list = _parse_csv_tokens(args.ckpt_list) if args.ckpt_list else []
    ckpts = _collect_ckpts(models_dir, args.ckpt_glob, epochs, ckpt_list, repo_root)

    score_terms = _parse_score_terms(args.score_terms)
    run_cfgs = _expand_run_space(args, ckpts)
    print(f"[search] planned runs: {len(run_cfgs)}")

    if args.dry_run:
        preview = run_cfgs[: min(5, len(run_cfgs))]
        print("[search] dry run preview:")
        for i, cfg in enumerate(preview):
            print(f"  {i}: {json.dumps(cfg, ensure_ascii=False)}")
        print("[search] dry run complete")
        return

    leaderboard_rows: List[Dict] = []
    records_jsonl = out_dir / "run_records.jsonl"
    records_jsonl.write_text("", encoding="utf-8")

    for idx, cfg in enumerate(run_cfgs):
        run_name = _make_run_name(idx, cfg)
        run_dir = out_dir / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        summary_path = run_dir / "summary.json"
        log_path = run_dir / "run.log"
        cmd = _build_eval_cmd(args, cfg, run_dir, repo_root)
        cmd_str = subprocess.list2cmdline(cmd)

        if args.resume and summary_path.exists():
            status = "skipped_existing"
            rc = 0
            print(f"[{idx+1}/{len(run_cfgs)}] skip existing: {run_name}")
        else:
            print(f"[{idx+1}/{len(run_cfgs)}] running: {run_name}")
            rc = _run_one(cmd, log_path, repo_root)
            status = "ok" if rc == 0 else "failed"

        summary = {}
        if summary_path.exists():
            try:
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                summary = _maybe_add_pore_parity_from_per_sample(summary, summary_path)
            except Exception:
                summary = {}

        score = float("-inf")
        score_details = []
        if summary:
            score, score_details = _score_summary(summary, score_terms)
            if not np.isfinite(score):
                score = float("-inf")

        row = {
            "run_index": idx,
            "run_name": run_name,
            "status": status,
            "return_code": rc,
            "score": score,
            "summary_path": str(summary_path),
            "log_path": str(log_path),
            "cmd": cmd_str,
            **cfg,
        }

        for k in [
            "voxel_dice_mean",
            "voxel_iou_mean",
            "pore_parity_corr",
            "pore_parity_slope",
            "pore_parity_slope_gap",
            "pore_parity_bias",
            "phase_bin_phi_corr_mean",
            "bin_phi_corr_mean",
            "target_phase_fraction_abs_err_mean",
            "pore_porosity_abs_err_mean",
            "porosity_abs_err_mean",
            "phase_bin_phi_mae_mean",
            "bin_phi_mae_mean",
            "z_head_phase_gap_mean",
            "z_tail_phase_gap_mean",
            "z_head_porosity_gap_mean",
            "z_tail_porosity_gap_mean",
            "time_sec_mean",
        ]:
            row[k] = summary.get(k, "")

        leaderboard_rows.append(row)

        record = {
            "row": row,
            "score_details": score_details,
            "summary": summary,
        }
        with open(records_jsonl, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    valid_rows = [
        r
        for r in leaderboard_rows
        if r.get("status") in ("ok", "skipped_existing") and _safe_float(r.get("score")) is not None
    ]
    valid_rows.sort(key=lambda x: float(x["score"]), reverse=True)

    rank = 1
    for r in valid_rows:
        r["rank"] = rank
        rank += 1

    leaderboard_csv = out_dir / "leaderboard.csv"
    _write_csv(valid_rows, leaderboard_csv)

    failed_rows = [r for r in leaderboard_rows if r.get("status") == "failed"]
    failed_csv = out_dir / "failed_runs.csv"
    if failed_rows:
        _write_csv(failed_rows, failed_csv)

    if not valid_rows:
        raise RuntimeError("No successful runs found. Check failed runs and logs.")

    best = valid_rows[0]
    best_payload = {
        "best_row": best,
        "score_terms": args.score_terms,
        "total_runs": len(run_cfgs),
        "successful_runs": len(valid_rows),
        "failed_runs": len(failed_rows),
    }
    best_json = out_dir / "best_config.json"
    best_json.write_text(json.dumps(best_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print("[search] done")
    print(f"  out_dir: {out_dir}")
    print(f"  leaderboard: {leaderboard_csv}")
    print(f"  best_config: {best_json}")
    if failed_rows:
        print(f"  failed_runs: {failed_csv} ({len(failed_rows)})")
    print(f"  best_score: {best['score']}")
    print(f"  best_run: {best['run_name']}")
    print(f"  best_summary: {best['summary_path']}")
    print("  reproduce_command:")
    print(f"    {best['cmd']}")


if __name__ == "__main__":
    main()

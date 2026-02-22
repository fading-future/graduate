import numpy as np


def _corrcoef_safe(a: np.ndarray, b: np.ndarray) -> float:
    aa = a.reshape(-1).astype(np.float64)
    bb = b.reshape(-1).astype(np.float64)
    if aa.size == 0 or bb.size == 0:
        return 0.0
    if aa.std() < 1e-12 or bb.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(aa, bb)[0, 1])


def to_phase_mask(binary_vol: np.ndarray, phase_value: int) -> np.ndarray:
    """
    Convert a binary volume to target-phase mask where 1 means target phase.
    phase_value=0 means treat zeros as target phase; phase_value=1 means ones.
    """
    phase_value = int(phase_value)
    if phase_value not in (0, 1):
        raise ValueError(f"phase_value must be 0 or 1, got {phase_value}")
    v = np.asarray(binary_vol)
    return (v == phase_value).astype(np.uint8)


def porosity_from_phase_mask(phase_mask: np.ndarray) -> float:
    return float(np.asarray(phase_mask).mean())


def maybe_center_crop_3d(vol: np.ndarray, target_size: int) -> np.ndarray:
    if target_size is None:
        return vol
    t = int(target_size)
    if t <= 0:
        return vol
    d, h, w = vol.shape
    if t >= d or t >= h or t >= w:
        return vol
    z0 = (d - t) // 2
    y0 = (h - t) // 2
    x0 = (w - t) // 2
    return vol[z0 : z0 + t, y0 : y0 + t, x0 : x0 + t]


def two_point_probability_curve(binary_phase_mask: np.ndarray, max_lag: int = 16) -> dict:
    """
    Compute directional 2-point probability curve S2(r) along x/y/z.
    Input must be a phase mask where 1 indicates the phase of interest.
    """
    v = (binary_phase_mask > 0).astype(np.float32)
    if v.ndim != 3:
        raise ValueError(f"Expected 3D volume, got shape {v.shape}")

    d, h, w = v.shape
    max_lag = int(max_lag)
    max_lag = max(0, min(max_lag, d - 1, h - 1, w - 1))

    lags = np.arange(max_lag + 1, dtype=np.int32)
    sx = np.zeros(max_lag + 1, dtype=np.float64)
    sy = np.zeros(max_lag + 1, dtype=np.float64)
    sz = np.zeros(max_lag + 1, dtype=np.float64)

    for lag in lags:
        if lag == 0:
            p = float(v.mean())
            sx[lag] = p
            sy[lag] = p
            sz[lag] = p
            continue
        sx[lag] = float((v[:, :, :-lag] * v[:, :, lag:]).mean())
        sy[lag] = float((v[:, :-lag, :] * v[:, lag:, :]).mean())
        sz[lag] = float((v[:-lag, :, :] * v[lag:, :, :]).mean())

    smean = (sx + sy + sz) / 3.0
    return {
        "lag": lags.astype(np.int32),
        "x": sx.astype(np.float32),
        "y": sy.astype(np.float32),
        "z": sz.astype(np.float32),
        "mean": smean.astype(np.float32),
    }


def compare_two_point_probability(
    pred_phase_mask: np.ndarray,
    gt_phase_mask: np.ndarray,
    max_lag: int = 16,
) -> dict:
    pred_curve = two_point_probability_curve(pred_phase_mask, max_lag=max_lag)
    gt_curve = two_point_probability_curve(gt_phase_mask, max_lag=max_lag)

    pred_mean = pred_curve["mean"].astype(np.float64)
    gt_mean = gt_curve["mean"].astype(np.float64)
    diff = pred_mean - gt_mean

    metrics = {
        "tp2_mae": float(np.mean(np.abs(diff))),
        "tp2_mse": float(np.mean(diff ** 2)),
        "tp2_rmse": float(np.sqrt(np.mean(diff ** 2))),
        "tp2_corr": _corrcoef_safe(pred_mean, gt_mean),
        "tp2_bias_mean": float(np.mean(diff)),
    }

    for axis in ("x", "y", "z"):
        pa = pred_curve[axis].astype(np.float64)
        ga = gt_curve[axis].astype(np.float64)
        d = pa - ga
        metrics[f"tp2_{axis}_mae"] = float(np.mean(np.abs(d)))
        metrics[f"tp2_{axis}_rmse"] = float(np.sqrt(np.mean(d ** 2)))
        metrics[f"tp2_{axis}_corr"] = _corrcoef_safe(pa, ga)

    return {
        "pred_curve": pred_curve,
        "gt_curve": gt_curve,
        "metrics": metrics,
    }


def _pick_boundary_pores_from_coords(coords: np.ndarray, axis: int):
    vals = coords[:, axis]
    vmin = float(vals.min())
    vmax = float(vals.max())
    span = max(vmax - vmin, 1e-12)
    tol = span * 1e-3 + 1e-12
    inlet = np.where(vals <= vmin + tol)[0]
    outlet = np.where(vals >= vmax - tol)[0]
    return inlet, outlet


def _network_from_porespy_dict(net_dict):
    import openpnm as op  # type: ignore

    if hasattr(op.io, "network_from_porespy"):
        return op.io.network_from_porespy(net_dict)

    # Compatibility fallback placeholder for older/newer APIs.
    raise RuntimeError("openpnm.io.network_from_porespy is not available in this OpenPNM version")


def _run_stokes_flow_absolute_k(pn, shape, axis=0, mu=1.0, dp=1.0):
    import openpnm as op  # type: ignore

    # Build phase object with viscosity.
    phase = None
    for cls_name in ("Air", "Water", "Phase", "GenericPhase"):
        cls = getattr(op.phase, cls_name, None)
        if cls is not None:
            try:
                phase = cls(network=pn)
                break
            except Exception:
                phase = None
    if phase is None:
        raise RuntimeError("Failed to create OpenPNM phase object")

    phase["pore.viscosity"] = float(mu)
    phase["throat.viscosity"] = float(mu)

    # Add geometry models to ensure sizes are available for hydraulic conductance.
    if hasattr(pn, "add_model_collection") and hasattr(op, "models"):
        try:
            geom_coll = getattr(op.models.collections.geometry, "spheres_and_cylinders", None)
            if geom_coll is not None:
                pn.add_model_collection(geom_coll)
        except Exception:
            pass
    if hasattr(pn, "regenerate_models"):
        try:
            pn.regenerate_models()
        except Exception:
            pass

    # Add hydraulic conductance model on phase if absent.
    if "throat.hydraulic_conductance" not in phase.keys():
        model_fn = None
        try:
            model_fn = op.models.physics.hydraulic_conductance.hagen_poiseuille
        except Exception:
            model_fn = None
        if model_fn is not None and hasattr(phase, "add_model"):
            phase.add_model(propname="throat.hydraulic_conductance", model=model_fn)
            if hasattr(phase, "regenerate_models"):
                phase.regenerate_models()

    alg_cls = getattr(op.algorithms, "StokesFlow", None)
    if alg_cls is None:
        raise RuntimeError("OpenPNM StokesFlow algorithm is unavailable")

    alg = alg_cls(network=pn, phase=phase)

    coords = pn["pore.coords"]
    inlet, outlet = _pick_boundary_pores_from_coords(coords, axis=int(axis))
    if inlet.size == 0 or outlet.size == 0:
        raise RuntimeError("Could not detect inlet/outlet pores from pore coordinates")

    alg.set_value_BC(pores=inlet, values=float(dp))
    alg.set_value_BC(pores=outlet, values=0.0)
    alg.run()

    q_in = alg.rate(pores=inlet, mode="group")
    if isinstance(q_in, (tuple, list)):
        q_in = np.array(q_in).sum()
    if isinstance(q_in, np.ndarray):
        q_in = float(np.array(q_in).sum())
    q_in = float(q_in)

    # Darcy-like conversion to voxel^2 (relative unit under voxel size = 1)
    dims = np.array(shape, dtype=np.float64)
    axis = int(axis)
    L = float(dims[axis])
    A = float(np.prod(np.delete(dims, axis)))
    k = abs(q_in) * float(mu) * L / (A * float(dp) + 1e-12)
    return {
        "k_abs_voxel2": float(k),
        "flow_rate": float(q_in),
        "num_pores": int(getattr(pn, "Np", -1)),
        "num_throats": int(getattr(pn, "Nt", -1)),
        "axis": axis,
    }


def absolute_permeability_openpnm(
    pore_phase_mask: np.ndarray,
    axis: int = 0,
    mu: float = 1.0,
    dp: float = 1.0,
) -> dict:
    """
    Estimate absolute permeability from a pore-phase mask using PoreSpy + OpenPNM.
    Returns {"ok": bool, ...} with detailed fields or error text.
    """
    try:
        import porespy as ps  # type: ignore
        import openpnm as op  # noqa: F401  # type: ignore
    except Exception as e:
        return {"ok": False, "error": f"Missing dependency: {e}"}

    im = (np.asarray(pore_phase_mask) > 0).astype(bool)
    if im.ndim != 3:
        return {"ok": False, "error": f"Expected 3D mask, got {im.shape}"}
    if im.mean() <= 0.0:
        return {"ok": False, "error": "No pore phase voxels in mask"}

    # Extract pore network from voxel image
    net_dict = None
    try:
        if hasattr(ps, "networks") and hasattr(ps.networks, "snow2"):
            try:
                snow = ps.networks.snow2(im=im)
            except TypeError:
                snow = ps.networks.snow2(im)
            if isinstance(snow, dict) and "network" in snow:
                net_dict = snow["network"]
            elif hasattr(snow, "network"):
                net_dict = snow.network
            elif isinstance(snow, dict):
                net_dict = snow
    except Exception as e:
        return {"ok": False, "error": f"PoreSpy snow2 failed: {e}"}

    if net_dict is None:
        return {"ok": False, "error": "Could not obtain network dictionary from PoreSpy"}

    try:
        pn = _network_from_porespy_dict(net_dict)
    except Exception as e:
        return {"ok": False, "error": f"OpenPNM network import failed: {e}"}

    try:
        out = _run_stokes_flow_absolute_k(pn, shape=im.shape, axis=int(axis), mu=float(mu), dp=float(dp))
        out["ok"] = True
        return out
    except Exception as e:
        return {"ok": False, "error": f"OpenPNM StokesFlow failed: {e}"}


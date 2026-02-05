from __future__ import annotations

import torch


def save_checkpoint(path: str, model, optimizer=None, step: int | None = None, extra: dict | None = None) -> None:
    payload = {
        "model": model.state_dict(),
        "step": step,
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_checkpoint(path: str, model, optimizer=None, map_location="cpu") -> dict:
    payload = torch.load(path, map_location=map_location)
    model.load_state_dict(payload["model"], strict=True)
    if optimizer is not None and "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])
    return payload

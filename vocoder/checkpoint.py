from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from .config import NHNVocoderConfig
from .model import NHNVocoder


def _upgrade_model_state(payload: Dict[str, Any], model: NHNVocoder) -> Dict[str, Any]:
    state = dict(payload["model"])
    if int(payload.get("format_version", 1)) < 3:
        taps = model.config.hop_length
        for head in range(model.config.spectrum_heads):
            for suffix in ("weight", "bias"):
                key = f"spectrum_gen.heads.{head}.4.{suffix}"
                if key in state and state[key].shape[0] == taps * 3:
                    state[key] = state[key][:taps].contiguous()
    return state


def save_checkpoint(
    path: str | Path,
    model: NHNVocoder,
    optimizer: Optional[torch.optim.Optimizer] = None,
    step: int = 0,
    epoch: int = 0,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
    scaler: Optional[Any] = None,
    metrics: Optional[Dict[str, float]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    payload: Dict[str, Any] = {
        "model": model.state_dict(),
        "config": model.config.to_dict(),
        "step": step,
        "epoch": epoch,
        "format_version": 3,
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler"] = scheduler.state_dict()
    if scaler is not None:
        payload["scaler"] = scaler.state_dict()
    if metrics is not None:
        payload["metrics"] = metrics
    if extra is not None:
        payload.update(extra)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_training_checkpoint(
    path: str | Path,
    model: NHNVocoder,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
    scaler: Optional[Any] = None,
    device: str | torch.device = "cpu",
) -> Dict[str, Any]:
    """Restore model and all available training state from a checkpoint."""

    payload = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(payload, dict) or "model" not in payload:
        raise ValueError("checkpoint must contain a model state")
    version = int(payload.get("format_version", 1))
    if version < 3:
        warnings.warn(
            f"upgrading a v{version} checkpoint to the dynamic-FIR/PQMF signal path; "
            "the first third of each legacy output head is retained and fine-tuning is required",
            RuntimeWarning,
            stacklevel=2,
        )
    missing, unexpected = model.load_state_dict(_upgrade_model_state(payload, model), strict=False)
    allowed_missing = {
        "noise_source.filters", "filterbank.analysis_filter",
        "filterbank.synthesis_filter", "filterbank.updown_filter",
    }
    if unexpected or set(missing) - allowed_missing:
        raise RuntimeError(f"incompatible checkpoint: missing={missing}, unexpected={unexpected}")
    if optimizer is not None and "optimizer" in payload and version >= 3:
        optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None and "scheduler" in payload and version >= 3:
        scheduler.load_state_dict(payload["scheduler"])
    if scaler is not None and "scaler" in payload and version >= 3:
        scaler.load_state_dict(payload["scaler"])
    return payload


def load_checkpoint(path: str | Path, device: str | torch.device = "cpu") -> NHNVocoder:
    payload = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(payload, dict) or "model" not in payload:
        raise ValueError("checkpoint must contain 'model' and 'config' keys")
    version = int(payload.get("format_version", 1))
    if version < 3:
        warnings.warn(
            f"upgrading a v{version} checkpoint to the dynamic-FIR/PQMF signal path; "
            "the first third of each legacy output head is retained and fine-tuning is required",
            RuntimeWarning,
            stacklevel=2,
        )
    model = NHNVocoder(NHNVocoderConfig.from_dict(payload.get("config", {})))
    missing, unexpected = model.load_state_dict(_upgrade_model_state(payload, model), strict=False)
    allowed_missing = {
        "noise_source.filters", "filterbank.analysis_filter",
        "filterbank.synthesis_filter", "filterbank.updown_filter",
    }
    if unexpected or set(missing) - allowed_missing:
        raise RuntimeError(f"incompatible checkpoint: missing={missing}, unexpected={unexpected}")
    return model.to(device).eval()

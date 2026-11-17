"""Mode-balanced MSE and raw-count metrics for baseline-relative flow changes.

Works with daily/selected-hour targets [N,M] or [B,N,M], and optional
hourly targets [N,T,M] or [B,N,T,M]. Train-only scales are essential.
"""
from __future__ import annotations

import torch


def estimate_mode_scales(train_dataset, *, min_scale: float = 1e-3) -> torch.Tensor:
    """Per-mode RMS of TRAIN target changes, in the target's count units.

    The small floor prevents division by zero; it is NOT a one-vehicle floor.
    Save these scales with the trained model, and do not use val/test targets.
    """
    if min_scale <= 0:
        raise ValueError("min_scale must be positive")
    totals = None
    count = 0
    for item in train_dataset:
        y = item["y_delta"].to(dtype=torch.float64)
        if y.ndim not in (2, 3) or not torch.isfinite(y).all():
            raise ValueError("Expected finite y_delta [N,M] or [N,T,M]")
        m = y.shape[-1]
        squares = y.reshape(-1, m).square().sum(dim=0)
        if totals is not None and totals.shape != squares.shape:
            raise ValueError("Inconsistent modal dimensions")
        totals = squares if totals is None else totals + squares
        count += y.numel() // m
    if totals is None or count == 0:
        raise ValueError("Cannot estimate mode scales from an empty training dataset")
    return torch.sqrt(totals / count).clamp_min(min_scale).to(torch.float32)


def mode_balanced_mse_loss(
    predicted_delta: torch.Tensor,
    target_delta: torch.Tensor,
    mode_scales: torch.Tensor,
) -> torch.Tensor:
    """Mean squared error of changes normalized by each mode's train-only RMS.

    Last dimension must be modes; other axes are averaged equally.
    """
    if predicted_delta.shape != target_delta.shape or predicted_delta.ndim not in (2, 3, 4):
        raise ValueError("Prediction and target must match [N,M], [B,N,M], [N,T,M], or [B,N,T,M]")
    if mode_scales.ndim != 1 or mode_scales.numel() != target_delta.shape[-1]:
        raise ValueError("mode_scales must have shape [M]")
    if not torch.isfinite(mode_scales).all() or (mode_scales <= 0).any():
        raise ValueError("mode_scales must be positive and finite")
    scales = mode_scales.to(device=predicted_delta.device, dtype=predicted_delta.dtype)
    return ((predicted_delta - target_delta) / scales).square().mean()


def modal_mae_rmse(
    predicted_delta: torch.Tensor,
    target_delta: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-mode MAE/RMSE in raw flow-count units (not normalized)."""
    if predicted_delta.shape != target_delta.shape or predicted_delta.ndim not in (2, 3, 4):
        raise ValueError("Prediction and target shapes must match with last dimension modes")
    error = (predicted_delta - target_delta).to(torch.float64)
    dims = tuple(range(error.ndim - 1))
    return error.abs().mean(dim=dims), error.square().mean(dim=dims).sqrt()

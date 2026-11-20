"""MSE of baseline-relative travel-time changes on eligible road/bus links."""
from __future__ import annotations

import torch


def masked_mse_loss(prediction: torch.Tensor, target: torch.Tensor,
                    mask: torch.Tensor) -> torch.Tensor:
    if prediction.shape != target.shape or mask.shape != target.shape:
        raise ValueError("Prediction, target and mask shapes must match")
    if mask.dtype != torch.bool or not mask.any():
        raise ValueError("Expected a nonempty Boolean eligibility mask")
    return (prediction[mask] - target[mask]).square().mean()

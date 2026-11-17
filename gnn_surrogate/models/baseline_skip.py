"""Reconstruct absolute modal flows by adding raw baseline to predicted changes.

Works for daily [N,M]/[B,N,M], and optional hourly [N,T,M]/[B,N,T,M].
The baseline must be RAW counts in the same temporal units as the prediction.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn


def _compatible_baseline(predicted: Tensor, baseline: Tensor) -> bool:
    return (
        predicted.ndim in (2, 3, 4)
        and (baseline.shape == predicted.shape or
             (predicted.ndim >= 3 and baseline.shape == predicted.shape[1:]))
    )


class BaselineSkip(nn.Module):
    """Add fixed raw baseline counts to signed predicted deltas; no clamping."""

    def forward(self, predicted_change: Tensor, baseline_raw: Tensor) -> Tensor:
        if not _compatible_baseline(predicted_change, baseline_raw):
            raise ValueError(
                f"Incompatible prediction/baseline shapes: "
                f"{tuple(predicted_change.shape)}, {tuple(baseline_raw.shape)}"
            )
        if predicted_change.device != baseline_raw.device:
            raise ValueError("Move baseline and prediction to the same device")
        return predicted_change + baseline_raw


def baseline_relative_target(scenario_raw: Tensor, baseline_raw: Tensor) -> Tensor:
    """Signed changes: scenario flows minus baseline flows, in identical units."""
    if not _compatible_baseline(scenario_raw, baseline_raw):
        raise ValueError(
            f"Incompatible scenario/baseline shapes: "
            f"{tuple(scenario_raw.shape)}, {tuple(baseline_raw.shape)}"
        )
    return scenario_raw - baseline_raw

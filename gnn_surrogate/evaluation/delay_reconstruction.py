"""Reconstruct general-road travel times from predicted congestion-delay changes.

The congestion-delay surrogate predicts
    delta_delay = (T_scenario - FF_scenario) - (T_baseline - FF_baseline).

For an eligible car link, the dynamic car free-speed change fraction is
    (speed_scenario - speed_baseline) / speed_baseline.
The intervention generator preserves length of existing car links, therefore
    FF_scenario = FF_baseline / (1 + car_freespeed_change_fraction).

This module uses only the fixed preprocessing tensors and scenario features:
no MATSim targets or network files are required during inference.
"""
from __future__ import annotations

import numpy as np


def reconstruct_road_travel_times(
    baseline_tt_s: np.ndarray,
    baseline_freeflow_s: np.ndarray,
    x_dynamic: np.ndarray,
    dynamic_feature_names: list[str] | tuple[str, ...],
    predicted_delay_delta_s: np.ndarray,
    valid_mask: np.ndarray,
    actual_delay_delta_s: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Reconstruct link times in seconds. Shapes are [N,1], except dynamics [N,F].

    Only valid outputs are meaningful. Invalid positions are returned as NaN
    to prevent accidental inclusion in downstream evaluation or accessibility.
    """
    base = np.asarray(baseline_tt_s, dtype=np.float64)
    ff = np.asarray(baseline_freeflow_s, dtype=np.float64)
    x = np.asarray(x_dynamic, dtype=np.float64)
    pred = np.asarray(predicted_delay_delta_s, dtype=np.float64)
    mask = np.asarray(valid_mask)
    if base.ndim != 2 or base.shape[1] != 1:
        raise ValueError("baseline_tt_s must have shape [N,1]")
    if ff.shape != base.shape or pred.shape != base.shape or mask.shape != base.shape:
        raise ValueError("Travel time, free flow, prediction, and mask must be [N,1]")
    if x.shape != (len(base), len(dynamic_feature_names)):
        raise ValueError("Dynamic feature shape does not match the node/feature count")
    if mask.dtype != np.bool_:
        raise ValueError("valid_mask must be boolean")
    if "car_freespeed_change_fraction" not in dynamic_feature_names:
        raise ValueError("Missing car_freespeed_change_fraction dynamic feature")
    if (not np.isfinite(base[mask]).all() or not np.isfinite(ff[mask]).all()
            or not np.isfinite(pred[mask]).all()
            or (base[mask] <= 0).any() or (ff[mask] <= 0).any()):
        raise ValueError("Invalid travel time or free-flow reference on eligible links")
    fraction = x[:, list(dynamic_feature_names).index("car_freespeed_change_fraction")]
    scale = 1.0 + fraction
    eligible = mask[:, 0]
    if not np.isfinite(scale[eligible]).all() or (scale[eligible] <= 0).any():
        raise ValueError("An eligible car link has an invalid scenario free-speed factor")

    freeflow_change = np.full_like(base, np.nan)
    scenario_ff = np.full_like(base, np.nan)
    scenario_ff[eligible, 0] = ff[eligible, 0] / scale[eligible]
    freeflow_change[eligible, 0] = scenario_ff[eligible, 0] - ff[eligible, 0]

    predicted_total_delta = np.full_like(base, np.nan)
    predicted_tt = np.full_like(base, np.nan)
    predicted_total_delta[mask] = pred[mask] + freeflow_change[mask]
    predicted_tt[mask] = base[mask] + predicted_total_delta[mask]

    result = {
        "scenario_freeflow_s": scenario_ff,
        "freeflow_change_s": freeflow_change,
        "predicted_total_delta_s": predicted_total_delta,
        "predicted_tt_s": predicted_tt,
    }
    if actual_delay_delta_s is not None:
        truth = np.asarray(actual_delay_delta_s, dtype=np.float64)
        if truth.shape != base.shape or not np.isfinite(truth[mask]).all():
            raise ValueError("Invalid actual congestion-delay-change targets")
        actual_total_delta = np.full_like(base, np.nan)
        actual_tt = np.full_like(base, np.nan)
        actual_total_delta[mask] = truth[mask] + freeflow_change[mask]
        actual_tt[mask] = base[mask] + actual_total_delta[mask]
        result["actual_total_delta_s"] = actual_total_delta
        result["actual_tt_s"] = actual_tt
    return result

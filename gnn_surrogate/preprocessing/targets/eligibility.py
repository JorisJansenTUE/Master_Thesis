"""Road/bus travel-time target eligibility derived solely from model inputs.

The existing x_static baseline_allows_car indicator is binary (not scaled).
The existing x_dynamic car_access_change is -1/0/+1.
The existing x_dynamic dedicated_bus_lane indicates scenario bus *_spec
presence, including pre-existing baseline *_spec links (see scenario_features).
"""
from __future__ import annotations

import numpy as np


def mask_from_inputs(x_static, x_dynamic, static_names, dynamic_names):
    """Return [N,2] or [B,N,2] Boolean mask: (road, bus_spec).

    Can accept NumPy arrays or torch.Tensors. Feature order comes from the
    existing input_metadata.json / dynamic_metadata.json, not fixed indices.
    No MATSim files, target arrays, or network access are needed.
    """
    if x_static.ndim != 2 or x_dynamic.ndim not in (2, 3):
        raise ValueError("Expected static [N,F], dynamic [N,D] or [B,N,D]")
    if x_static.shape[0] != x_dynamic.shape[-2]:
        raise ValueError("x_static and x_dynamic have different node counts")
    idx_car = list(static_names).index("baseline_allows_car")
    idx_change = list(dynamic_names).index("car_access_change")
    idx_spec = list(dynamic_names).index("dedicated_bus_lane")

    road = (x_static[:, idx_car] + x_dynamic[..., idx_change]) > 0.5
    bus_spec = x_dynamic[..., idx_spec] > 0.5

    # Preserve the incoming array/tensor type and device.
    if isinstance(x_static, np.ndarray) and isinstance(x_dynamic, np.ndarray):
        return np.stack((road, bus_spec), axis=-1)
    import torch
    if torch.is_tensor(x_static) and torch.is_tensor(x_dynamic):
        return torch.stack((road, bus_spec), dim=-1)
    raise TypeError("x_static and x_dynamic must both be numpy or torch tensors")

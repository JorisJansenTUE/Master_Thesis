"""General-road travel-time eligibility derived entirely from model features."""
from __future__ import annotations


def mask_from_inputs(x_static, x_dynamic, static_names, dynamic_names):
    """Return [N,1] or [B,N,1] Boolean mask for scenario car access.

    Supports matching NumPy arrays or PyTorch tensors, without reading target
    data, linkstats, or the scenario network. The full graph remains unchanged.
    """
    if x_static.ndim != 2 or x_dynamic.ndim not in (2, 3):
        raise ValueError("Expected static [N,F], dynamic [N,D] or [B,N,D]")
    if x_static.shape[0] != x_dynamic.shape[-2]:
        raise ValueError("x_static and x_dynamic have different node counts")
    idx_car = list(static_names).index("baseline_allows_car")
    idx_change = list(dynamic_names).index("car_access_change")
    if type(x_static) is not type(x_dynamic):
        raise TypeError("Static and dynamic inputs must have matching array types")
    road = (x_static[:, idx_car] + x_dynamic[..., idx_change]) > 0.5
    return road[..., None]

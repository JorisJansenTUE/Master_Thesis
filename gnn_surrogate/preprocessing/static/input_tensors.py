"""Select fixed GNN inputs from the saved dual graph and *baseline* MATSim flows.

No scenario-dependent features or training targets are constructed here.
Graph topology is already stored in dual_edge_index.npy by build_graph.py.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

FLOW_MODES = ("cars", "trucks", "bikes", "buses")

# Small initial, Natterer-style static set plus modal accessibility. Expand only
# after evaluating the basic model; IDs/highway names are not numeric features.
DEFAULT_STATIC_COLUMNS = (
    "length_m",
    "baseline_car_capacity",
    "baseline_car_freespeed_m_s",
    "mid_x",
    "mid_y",
    "baseline_active",
    "baseline_allows_car",
    "baseline_allows_bike",
    "baseline_allows_bus",
)

BINARY_COLUMNS = {"baseline_active", "baseline_allows_car", "baseline_allows_bike", "baseline_allows_bus"}


def _check_node_order(nodes: pd.DataFrame) -> None:
    if "node_id" not in nodes or not np.array_equal(
        nodes["node_id"].to_numpy(), np.arange(len(nodes))
    ):
        raise ValueError("dual_nodes rows must be ordered by consecutive node_id 0..N-1")


def select_static_features(
    nodes: pd.DataFrame,
    columns: tuple[str, ...] = DEFAULT_STATIC_COLUMNS,
) -> tuple[np.ndarray, list[str], dict]:
    """Return numeric [N, F] float32 data, feature names, and scaling metadata.

    Binary indicators stay 0/1. Continuous columns are z-scored over finite
    baseline graph values. Missing/nonfinite values are filled to 0 *after*
    standardization
    """
    _check_node_order(nodes)
    if not columns or len(set(columns)) != len(columns):
        raise ValueError("Choose a nonempty list of unique static feature columns")
    missing = set(columns) - set(nodes.columns)
    if missing:
        raise ValueError(f"Unknown static feature columns: {sorted(missing)}")

    arrays: list[np.ndarray] = []
    names: list[str] = []
    scales: dict = {}
    for col in columns:
        values = pd.to_numeric(nodes[col], errors="raise").to_numpy(dtype=np.float64)
        if col in BINARY_COLUMNS:
            if not np.isfinite(values).all() or not np.isin(values, (0, 1)).all():
                raise ValueError(f"Expected a complete 0/1 feature: {col}")
            arrays.append(values.astype(np.float32))
            names.append(col)
            continue

        finite = np.isfinite(values)
        if not finite.any():
            raise ValueError(f"No finite values in {col}; cannot select it")
        if np.isneginf(values).any():
            raise ValueError(f"Negative infinity found in {col}")
        mean = float(values[finite].mean())
        std = float(values[finite].std())
        if std == 0:
            std = 1.0
        normalized = np.zeros(len(nodes), dtype=np.float64)
        normalized[finite] = (values[finite] - mean) / std
        arrays.append(normalized.astype(np.float32))
        names.append(col)
        scales[col] = {"mean": mean, "std": std}


    x = np.column_stack(arrays).astype(np.float32, copy=False)
    if not np.isfinite(x).all():
        raise AssertionError("Selected static features must be finite")
    return x, names, scales


def load_baseline_hourly(
    nodes: pd.DataFrame,
    flow_path: Path,
    hours: int = 30,
) -> np.ndarray:
    """Aggregate sparse MATSim baseline link-hours into dense [N, T, 4].

    Each original MATSim link is assigned to its stable directed street slot
    via dual_nodes.baseline_link_ids. Missing link-hour rows are zero traffic.
    A nonzero flow from a link outside the fixed representation raises an
    error, rather than silently discarding flow.
    """
    _check_node_order(nodes)
    if hours < 1:
        raise ValueError("hours must be positive")
    if "baseline_link_ids" not in nodes.columns:
        raise ValueError("Missing baseline_link_ids in dual_nodes")

    link_to_node: dict[str, int] = {}
    for node_id, raw in zip(nodes.node_id, nodes.baseline_link_ids):
        ids = json.loads(raw)
        if not isinstance(ids, list):
            raise ValueError(f"Invalid baseline_link_ids at node {node_id}")
        for link_id in ids:
            if link_id in link_to_node:
                raise ValueError(f"Duplicate original link ID {link_id}")
            link_to_node[link_id] = int(node_id)

    required = ["link_id", "hour", *FLOW_MODES]
    flows = pd.read_csv(flow_path, dtype={"link_id": str}, usecols=required)
    if flows.empty:
        raise ValueError(f"Baseline hourly flow file is empty: {flow_path}")
    if flows[required].isna().any().any():
        raise ValueError("Null identifiers, hours, or modal volumes in baseline flow file")
    if flows.duplicated(["link_id", "hour"]).any():
        raise ValueError("Duplicate link-hour rows in baseline flow file")
    if not np.all(np.equal(flows.hour.to_numpy(), flows.hour.to_numpy().astype(np.int64))):
        raise ValueError("Noninteger hour values in baseline flows")
    if not flows.hour.between(0, hours - 1).all():
        raise ValueError(f"Flow file contains hours outside 0..{hours - 1}")

    values = flows[list(FLOW_MODES)].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Baseline flow volumes must be finite and nonnegative")

    mapped_node_id = flows.link_id.map(link_to_node)
    if mapped_node_id.isna().any():
        unknown = flows.loc[mapped_node_id.isna()]
        nonzero = unknown.loc[unknown[list(FLOW_MODES)].sum(axis=1) > 0]
        if not nonzero.empty:
            examples = nonzero.link_id.unique()[:5].tolist()
            raise ValueError(
                f"{len(nonzero)} nonzero baseline flow records are outside the "
                f"fixed physical graph (example link IDs: {examples}). "
                "Do not drop these without deciding how to represent them."
            )
        flows = flows.loc[mapped_node_id.notna()].copy()
        mapped_node_id = mapped_node_id.loc[mapped_node_id.notna()]

    # np.add.at correctly sums parallel original MATSim links per street/hour.
    dense = np.zeros((len(nodes), hours, len(FLOW_MODES)), dtype=np.float32)
    np.add.at(
        dense,
        (mapped_node_id.to_numpy(dtype=np.int64), flows.hour.to_numpy(dtype=np.int64)),
        flows[list(FLOW_MODES)].to_numpy(dtype=np.float32),
    )
    expected = flows[list(FLOW_MODES)].sum().to_numpy(dtype=np.float64)
    if not np.allclose(dense.sum(axis=(0, 1), dtype=np.float64), expected, atol=0.1, rtol=1e-6):
        raise AssertionError("Baseline link-hour aggregation lost traffic")
    if "baseline_active" in nodes and (dense[nodes.baseline_active.to_numpy() == 0] != 0).any():
        raise AssertionError("Baseline-inactive direction has nonzero baseline traffic")
    return dense


def select_input_tensors(
    nodes: pd.DataFrame,
    baseline_flows_path: Path,
    *,
    columns: tuple[str, ...] = DEFAULT_STATIC_COLUMNS,
    hours: int = 30,
) -> tuple[dict[str, np.ndarray], dict]:
    """Select reusable scenario-independent inputs (no scenario targets).

    x_static: [N,F] standardized static features.
    x_baseline_hourly: [N,T,4] log1p baseline flow inputs.
    baseline_hourly_raw: [N,T,4] original counts for later delta targets.
    """
    static, feature_names, scaling = select_static_features(nodes, columns)
    raw = load_baseline_hourly(nodes, baseline_flows_path, hours)
    arrays = {
        "x_static": static,
        "x_baseline_hourly": np.log1p(raw),
        "baseline_hourly_raw": raw,
    }
    metadata = {
        "static_columns": list(columns),
        "static_feature_names": feature_names,
        "static_standardization": scaling,
        "baseline_transform": "log1p",
        "flow_modes": list(FLOW_MODES),
        "hours": hours,
        "node_count": len(nodes),
    }
    return arrays, metadata

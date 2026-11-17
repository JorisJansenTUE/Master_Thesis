"""Map an intervention scenario onto the fixed directed-street GNN nodes.

Inputs: baseline dual_nodes.csv.gz (including cached car reference columns),
and one scenario's network.xml.gz, link_mapping.csv.gz, interventions.json.
No flow targets are read.
The physical graph and node IDs are never changed.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from network_sampling.network.matsim_network import load_matsim_network

# All features are zero for an unmodified baseline scenario. Keep the input
# selector separate so alternative feature sets are easy to test later.
DEFAULT_DYNAMIC_COLUMNS = (
    "bicycle_street", "protected_bike_lane", "dedicated_bus_lane", "destination_only",
    "car_capacity_change_fraction", "car_freespeed_change_fraction",
    "car_access_change", "bike_access_change", "bus_access_change",
)
MODES = ("car", "bike", "bus")
TYPES = set(DEFAULT_DYNAMIC_COLUMNS[:4])


def _node_lookup(nodes: pd.DataFrame) -> dict[tuple[str, str, str], int]:
    if not np.array_equal(nodes.node_id.to_numpy(), np.arange(len(nodes))):
        raise ValueError("dual_nodes must be ordered by node_id 0..N-1")
    key = list(zip(nodes.segment_id.astype(str), nodes.from_node.astype(str),
                   nodes.to_node.astype(str)))
    if len(key) != len(set(key)):
        raise ValueError("Duplicate directional slots in dual_nodes")
    return dict(zip(key, range(len(nodes))))


def _aggregate_links(node_links: dict[int, list], n: int):
    """Aggregate car-only road capacity/speed and any-link mode access by slot."""
    capacity = np.zeros(n, dtype=np.float64)
    speed_numerator = np.zeros(n, dtype=np.float64)
    access = {m: np.zeros(n, dtype=np.float32) for m in MODES}
    for node_id, links in node_links.items():
        for link in links:
            for mode in MODES:
                if mode in link.allowed_modes:
                    access[mode][node_id] = 1
            if "car" in link.allowed_modes:
                if not (np.isfinite(link.capacity) and link.capacity > 0
                        and np.isfinite(link.freespeed)):
                    raise ValueError(f"Invalid car capacity or speed: {link.id}")
                capacity[node_id] += link.capacity
                speed_numerator[node_id] += link.capacity * link.freespeed
    speed = np.divide(speed_numerator, capacity, out=np.zeros_like(capacity),
                      where=capacity > 0)
    return capacity, speed, access


def build_scenario_feature_table(
    nodes: pd.DataFrame,
    *,
    scenario_dir: Path,
) -> pd.DataFrame:
    """Return [N, 9] raw scenario-change features in fixed node order.

    Bike streets / PBLs are marked on both directions. Dedicated bus lanes are
    marked only on their target direction. Destination-only is marked only on
    baseline car-accessible directions. Road-space effects come from the ACTUAL
    scenario network, not inferred from the intervention variant.
    """
    n = len(nodes)
    keys = _node_lookup(nodes)
    scenario_dir = Path(scenario_dir)
    modified = load_matsim_network(scenario_dir / "network.xml.gz")
    mapping = pd.read_csv(
        scenario_dir / "link_mapping.csv.gz",
        dtype={"link_id": str, "segment_id": str, "from_node": str, "to_node": str},
    )
    interventions = json.loads((scenario_dir / "interventions.json").read_text())
    if not isinstance(interventions, list):
        raise ValueError("interventions.json must be a list")
    if mapping.link_id.duplicated().any():
        raise ValueError("Duplicate link IDs in scenario mapping")

    # Read the baseline link IDs for mapping checks, but not the baseline XML.
    # These values were aggregated by build_graph.py using the same car-only
    # capacity-weighted speed definition as _aggregate_links().
    baseline_fields = ("baseline_car_capacity", "baseline_car_freespeed_m_s")
    absent = set(baseline_fields) - set(nodes.columns)
    if absent:
        raise ValueError(
            f"Missing {sorted(absent)} in dual_nodes.csv.gz. "
            "Rebuild the static graph using the updated build_graph.py."
        )
    base_cap = nodes["baseline_car_capacity"].to_numpy(dtype=np.float64)
    base_speed = nodes["baseline_car_freespeed_m_s"].to_numpy(dtype=np.float64)
    if not (np.isfinite(base_cap).all() and np.isfinite(base_speed).all()
            and (base_cap >= 0).all() and (base_speed >= 0).all()):
        raise ValueError("Invalid cached baseline car capacity or free speed")
    base_access = {
        m: nodes[f"baseline_allows_{m}"].to_numpy(dtype=np.float32)
        for m in MODES
    }
    if np.any((base_cap > 0) != (base_access["car"] > 0)):
        raise ValueError("Cached car capacity disagrees with baseline car accessibility")

    original_id_to_node: dict[str, int] = {}
    for row in nodes.itertuples(index=False):
        for lid in json.loads(row.baseline_link_ids):
            if lid in original_id_to_node:
                raise ValueError(f"Duplicate baseline link {lid}")
            original_id_to_node[lid] = int(row.node_id)

    scenario_links = {i: [] for i in range(n)}
    scenario_original_ids = set()
    for row in mapping.itertuples(index=False):
        key = (row.segment_id, row.from_node, row.to_node)
        if key not in keys:
            raise ValueError(f"Unknown fixed directional slot: {key}")
        if row.link_id not in modified.links:
            raise ValueError(f"Mapped MATSim link missing: {row.link_id}")
        link = modified.links[row.link_id]
        if (str(link.from_node), str(link.to_node)) != (row.from_node, row.to_node):
            raise ValueError(f"Direction mismatch for link {row.link_id}")
        node_id = keys[key]
        if row.link_id in original_id_to_node:
            if original_id_to_node[row.link_id] != node_id:
                raise ValueError(f"Original link moved slots: {row.link_id}")
            scenario_original_ids.add(row.link_id)
        scenario_links[node_id].append(link)
    if scenario_original_ids != set(original_id_to_node):
        raise ValueError("Scenario mapping does not cover every original physical link")

    scen_cap, scen_speed, scen_access = _aggregate_links(scenario_links, n)

    data = {name: np.zeros(n, dtype=np.float32) for name in DEFAULT_DYNAMIC_COLUMNS}
    segment_to_nodes = {}
    for row in nodes.itertuples(index=False):
        segment_to_nodes.setdefault(row.segment_id, []).append(int(row.node_id))

    used_segments = set()
    for entry in interventions:
        sid, kind = str(entry["segment_id"]), entry["intervention_type"]
        if sid in used_segments:
            raise ValueError(f"Multiple interventions on segment {sid}")
        used_segments.add(sid)
        if kind not in TYPES or sid not in segment_to_nodes:
            raise ValueError(f"Unrecognized intervention {kind} on {sid}")
        if kind in ("bicycle_street", "protected_bike_lane"):
            data[kind][segment_to_nodes[sid]] = 1.0
        elif kind == "destination_only":
            for i in segment_to_nodes[sid]:
                if base_access["car"][i]:
                    data[kind][i] = 1.0
        else:  # Dedicated bus lane is directional.
            target = entry.get("target_link_ids", [])
            if len(target) != 1 or target[0] not in original_id_to_node:
                raise ValueError(f"Invalid bus-lane target on {sid}: {target}")
            i = original_id_to_node[target[0]]
            if nodes.iloc[i].segment_id != sid:
                raise ValueError(f"Bus-lane target {target[0]} is outside {sid}")
            data[kind][i] = 1.0

    np.divide(scen_cap - base_cap, base_cap,
              out=data["car_capacity_change_fraction"], where=base_cap > 0)
    np.divide(scen_speed - base_speed, base_speed,
              out=data["car_freespeed_change_fraction"],
              where=(base_cap > 0) & (scen_cap > 0) & (base_speed > 0))
    # A cached weighted mean can differ from re-aggregation by roundoff only.
    # Exact zeros on unchanged links are important for a baseline-relative input.
    for column in ("car_capacity_change_fraction", "car_freespeed_change_fraction"):
        data[column][np.abs(data[column]) < 1e-12] = 0.0
    for mode in MODES:
        data[f"{mode}_access_change"] = scen_access[mode] - base_access[mode]

    features = pd.DataFrame(data, columns=list(DEFAULT_DYNAMIC_COLUMNS))
    features.insert(0, "node_id", nodes.node_id.to_numpy())
    if not np.isfinite(features[list(DEFAULT_DYNAMIC_COLUMNS)].to_numpy()).all():
        raise ValueError("Scenario features contain nonfinite values")
    return features


def select_dynamic_features(
    features: pd.DataFrame,
    columns: tuple[str, ...] = DEFAULT_DYNAMIC_COLUMNS,
) -> tuple[np.ndarray, list[str]]:
    """Simple selector; no implicit extra features or per-scenario normalization."""
    if not columns or len(columns) != len(set(columns)):
        raise ValueError("Choose a nonempty set of unique dynamic columns")
    unknown = set(columns) - set(features.columns)
    if unknown:
        raise ValueError(f"Unknown dynamic columns: {sorted(unknown)}")
    x = features[list(columns)].to_numpy(dtype=np.float32)
    if not np.isfinite(x).all():
        raise ValueError("Selected scenario features must be finite")
    return x, list(columns)

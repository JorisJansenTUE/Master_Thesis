"""Construct per-scenario hourly flow targets on the fixed directed-street graph.

This module reads only a scenario's saved link_mapping.csv.gz and sparse
MATSim output_detailed_link_statistics.csv.gz, plus the fixed graph nodes and
already-built *raw* baseline hourly tensor. It never reads scenario/network XML
or intervention labels, and adds no features.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .input_tensors import FLOW_MODES, _check_node_order


SLOT_COLUMNS = ("segment_id", "from_node", "to_node")


def load_scenario_hourly(
    nodes: pd.DataFrame,
    scenario_dir: Path,
    *,
    hours: int = 30,
) -> tuple[np.ndarray, dict]:
    """Return scenario vehicle traversals [N,T,4] and conservation diagnostics.

    Several MATSim links can map to one directed GNN node. Newly generated
    links (bike, PBL, bus) are included via the per-scenario link mapping.
    Missing link-hour records represent zero flow. Unknown links, duplicate
    link-hours, invalid counts and unmapped directions cause a hard failure.
    """
    _check_node_order(nodes)
    if hours < 1:
        raise ValueError("hours must be positive")
    for col in SLOT_COLUMNS:
        if col not in nodes:
            raise ValueError(f"Missing fixed-node column {col}")
    if nodes.duplicated(list(SLOT_COLUMNS)).any():
        raise ValueError("Duplicate physical-street directional nodes")

    scenario_dir = Path(scenario_dir)
    mapping_path = scenario_dir / "link_mapping.csv.gz"
    flow_path = scenario_dir / "targets" / "output_detailed_link_statistics.csv.gz"
    mapping = pd.read_csv(
        mapping_path,
        dtype={"link_id": str, "segment_id": str, "from_node": str, "to_node": str},
    )
    required_map = {"link_id", *SLOT_COLUMNS}
    if missing := required_map.difference(mapping.columns):
        raise ValueError(f"Link mapping missing columns: {sorted(missing)}")
    if mapping.empty or mapping[list(required_map)].isna().any().any():
        raise ValueError("Empty or incomplete scenario link mapping")
    if mapping.link_id.duplicated().any():
        raise ValueError("Duplicate link IDs in scenario mapping")

    slots = nodes[[*SLOT_COLUMNS, "node_id"]]
    mapping = mapping.merge(slots, on=list(SLOT_COLUMNS), how="left", validate="many_to_one")
    if mapping.node_id.isna().any():
        examples = mapping.loc[mapping.node_id.isna(), "link_id"].head(5).tolist()
        raise ValueError(f"Scenario links have no fixed graph node: {examples}")

    # Confirm all existing street links still map to the same node. This also
    # protects against accidentally pairing a scenario with the wrong graph.
    if "baseline_link_ids" in nodes.columns:
        baseline_lookup = {
            str(lid): int(node.node_id)
            for node in nodes.itertuples(index=False)
            for lid in json.loads(node.baseline_link_ids)
        }
        present = mapping.set_index("link_id")["node_id"]
        missing_baseline = baseline_lookup.keys() - present.index
        if missing_baseline:
            raise ValueError(
                f"Scenario mapping is missing {len(missing_baseline)} original links "
                f"(examples: {sorted(missing_baseline)[:5]})"
            )
        mismatched = [lid for lid, node_id in baseline_lookup.items()
                      if int(present[lid]) != node_id]
        if mismatched:
            raise ValueError(f"Original links assigned to wrong graph nodes: {mismatched[:5]}")

    cols = ["link_id", "hour", *FLOW_MODES]
    flows = pd.read_csv(flow_path, dtype={"link_id": str}, usecols=cols)
    if flows.empty or flows[cols].isna().any().any():
        raise ValueError("Scenario flow file is empty or contains missing values")
    if flows.duplicated(["link_id", "hour"]).any():
        raise ValueError("Duplicate link-hour rows in scenario flow file")

    hour_values = pd.to_numeric(flows.hour, errors="raise").to_numpy(dtype=np.float64)
    if (not np.isfinite(hour_values).all()
            or (hour_values != np.floor(hour_values)).any()
            or (hour_values < 0).any() or (hour_values >= hours).any()):
        raise ValueError(f"Flow hours must be integers within 0..{hours - 1}")
    hour_idx = hour_values.astype(np.int64)

    modal = flows[list(FLOW_MODES)].to_numpy(dtype=np.float64)
    if (not np.isfinite(modal).all() or (modal < 0).any()
            or (modal != np.floor(modal)).any()):
        raise ValueError("MATSim modal flows must be finite nonnegative integer counts")
    modal = modal.astype(np.int64)

    joined = flows[["link_id"]].merge(
        mapping[["link_id", "node_id"]], on="link_id", how="left", validate="many_to_one"
    )
    if joined.node_id.isna().any():
        unknown = joined.loc[joined.node_id.isna(), "link_id"].unique()[:5].tolist()
        raise ValueError(f"Flow records have no fixed graph mapping: {unknown}")
    node_idx = joined.node_id.to_numpy(dtype=np.int64)

    dense_int = np.zeros((len(nodes), hours, len(FLOW_MODES)), dtype=np.int64)
    np.add.at(dense_int, (node_idx, hour_idx), modal)

    # Verify conservation separately for every hour and mode (not only daily).
    original_hourly = np.zeros((hours, len(FLOW_MODES)), dtype=np.int64)
    np.add.at(original_hourly, hour_idx, modal)
    if not np.array_equal(dense_int.sum(axis=0), original_hourly):
        raise AssertionError("Scenario hourly/modal aggregation lost traffic")
    if dense_int.max() > 2 ** 24:
        raise ValueError("Flows exceed exact integer precision of float32")

    report = {
        "link_hour_rows": int(len(flows)),
        "mapped_link_count": int(len(mapping)),
        "mapped_node_count": int(mapping.node_id.nunique()),
        "nonzero_node_hours": int(np.count_nonzero(dense_int.sum(axis=2))),
        "scenario_totals": {mode: int(original_hourly[:, k].sum())
                            for k, mode in enumerate(FLOW_MODES)},
        "per_hour_conservation": True,
    }
    return dense_int.astype(np.float32), report


def build_delta_targets(
    nodes: pd.DataFrame,
    scenario_dir: Path,
    baseline_hourly_raw: np.ndarray,
) -> tuple[np.ndarray, dict]:
    """Return signed baseline-relative targets [N,T,4] in vehicles/hour.

    The model can later reconstruct absolute predictions by adding the same
    baseline_hourly_raw with models/baseline_skip.py.
    """
    _check_node_order(nodes)
    baseline = np.asarray(baseline_hourly_raw)
    if (baseline.ndim != 3 or baseline.shape[0] != len(nodes)
            or baseline.shape[2] != len(FLOW_MODES)):
        raise ValueError("Baseline must have shape [number of graph nodes, hours, 4]")
    if (not np.isfinite(baseline).all() or (baseline < 0).any()
            or (baseline != np.floor(baseline)).any()):
        raise ValueError("Baseline raw flows must be finite nonnegative vehicle counts")
    baseline = baseline.astype(np.float32, copy=False)
    scenario, report = load_scenario_hourly(nodes, scenario_dir, hours=baseline.shape[1])
    delta = scenario - baseline
    if not np.isfinite(delta).all() or not np.array_equal(delta + baseline, scenario):
        raise AssertionError("Baseline-relative target reconstruction failed")
    report.update({
        "baseline_totals": {mode: int(baseline[:, :, k].sum(dtype=np.float64))
                            for k, mode in enumerate(FLOW_MODES)},
        "delta_totals": {mode: int(delta[:, :, k].sum(dtype=np.float64))
                         for k, mode in enumerate(FLOW_MODES)},
        "changed_node_hours": int(np.count_nonzero(np.any(delta != 0, axis=2))),
        "delta_positive_values": int(np.count_nonzero(delta > 0)),
        "delta_negative_values": int(np.count_nonzero(delta < 0)),
    })
    return delta, report

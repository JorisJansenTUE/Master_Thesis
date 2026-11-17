"""Inspect fixed dual-graph topology and alignment with one MATSim scenario.

Does not train a model, generate features, or interpret interventions.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

# ------------------------------------------------------------------
# Local inspection settings (edit these paths; no CLI arguments needed)
# ------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
GRAPH_DIR = PROJECT_ROOT / "data" / "GNN" / "gnn_static"
SCENARIO_DIR: Path | None = Path(r"C:\Users\20201733\Downloads\cluster_00000")
# Set SCENARIO_DIR = None to check only the graph's static topology.

FLOW_MODES = ("cars", "trucks", "bikes", "buses")


def inspect(graph_dir: Path, scenario_dir: Path | None = None) -> None:
    nodes = pd.read_csv(
        graph_dir / "dual_nodes.csv.gz",
        dtype={"segment_id": str, "from_node": str, "to_node": str,
               "direction": str, "osm_way_id": str, "baseline_link_ids": str},
    )
    edge_index = np.load(graph_dir / "dual_edge_index.npy", allow_pickle=False)
    n = len(nodes)
    assert nodes.node_id.tolist() == list(range(n)), "Node IDs not consecutive"
    assert not nodes.duplicated(["segment_id", "from_node", "to_node"]).any(), (
        "Duplicate directed street slots")
    assert edge_index.shape[0] == 2 and edge_index.dtype == np.int64
    assert not edge_index.size or (edge_index.min() >= 0 and edge_index.max() < n)
    assert not np.any(edge_index[0] == edge_index[1]), "Unexpected self-edge"
    assert len({tuple(pair) for pair in edge_index.T}) == edge_index.shape[1]
    if edge_index.size:
        assert (nodes.to_node.to_numpy()[edge_index[0]] ==
                nodes.from_node.to_numpy()[edge_index[1]]).all(), (
            "A dual edge violates head-to-tail street connectivity")

    degree = np.bincount(edge_index.ravel(), minlength=n)
    print(f"GNN nodes:                  {n:,}")
    print(f"Physical streets:            {nodes.segment_id.nunique():,}")
    print(f"Special PT self-loops:       {int(nodes.is_pt_self_loop.sum()):,}")
    print(f"Directed dual edges:         {edge_index.shape[1]:,}")
    print(f"Isolated graph nodes:        {int((degree == 0).sum()):,}")
    print(f"Baseline-active slots:       {int(nodes.baseline_active.sum()):,}"
          if nodes.baseline_active.notna().all() else "Baseline-active slots:       not available")
    print(f"Missing physical widths:     {int(nodes.total_width_m.isna().sum()):,}")
    for feature in ["mid_x", "baseline_capacity", "baseline_freespeed_m_s",
                    "baseline_daily_cars"]:
        print(f"{feature:27s} available {int(nodes[feature].notna().sum()):,}/{n:,}")

    if scenario_dir is None:
        print("No scenario passed; topology checks complete.")
        return

    mapping = pd.read_csv(
        scenario_dir / "link_mapping.csv.gz",
        dtype={"link_id": str, "segment_id": str, "from_node": str, "to_node": str},
    )
    assert not mapping.link_id.duplicated().any(), "Duplicate mapped link ID"
    slots = nodes[["segment_id", "from_node", "to_node", "node_id"]]
    joined = mapping.merge(slots, on=["segment_id", "from_node", "to_node"],
                           how="left", validate="many_to_one")
    missing = joined.loc[joined.node_id.isna()]
    assert missing.empty, f"{len(missing)} MATSim links have no fixed dual-graph node"
    mapped_ids = set(mapping.link_id)
    baseline_ids = {lid for raw in nodes.baseline_link_ids.dropna()
                    for lid in json.loads(raw)}
    assert baseline_ids.issubset(mapped_ids), "Some original links not mapped in scenario"
    print(f"Scenario mapped links:       {len(mapping):,}")
    print(f"Mapped directional slots:    {joined.node_id.nunique():,}")
    print(f"Generated scenario links:    {int((mapping.link_kind != 'baseline').sum()):,}")
    inactive_ids = set(nodes.loc[nodes.baseline_active == 0, "node_id"])
    newly_enabled = joined.loc[joined.node_id.isin(inactive_ids), "node_id"].nunique()
    print(f"Newly enabled directions:    {newly_enabled:,}")

    flows_path = scenario_dir / "targets" / "output_detailed_link_statistics.csv.gz"
    if flows_path.exists():
        flows = pd.read_csv(flows_path, dtype={"link_id": str})
        assert not flows.duplicated(["link_id", "hour"]).any(), (
            "Duplicate observed MATSim link-hours")
        unmatched = set(flows.link_id) - mapped_ids
        print(f"Flow rows / hours:           {len(flows):,} / {flows.hour.nunique()}")
        print(f"Unmapped links with flows:   {len(unmatched):,}")
        if unmatched:
            print("  Example unmapped links:", sorted(unmatched)[:8])
        else:
            print("All flow records map to a fixed graph node.")
        # Special PT loops carry bus volume: keep them, do not drop them.
        loop_ids = set(joined.loc[joined.node_id.isin(
            nodes.loc[nodes.is_pt_self_loop, "node_id"]), "link_id"])
        pt_loop_bus = flows.loc[flows.link_id.isin(loop_ids), "buses"].sum()
        print(f"Bus traversals on PT loops: {pt_loop_bus:,.0f}")
        assert not unmatched, "Flow links without dual-graph mapping"
        # Check that many MATSim links -> one directional street slot is
        # lossless for hourly vehicle flows (including dedicated bus links).
        aggregated = flows.merge(joined[["link_id", "node_id"]], on="link_id",
                                 how="left", validate="many_to_one")
        assert not aggregated.node_id.isna().any()
        by_slot_hour = aggregated.groupby(["node_id", "hour"], sort=False)[
            list(FLOW_MODES)].sum()
        original = flows[list(FLOW_MODES)].sum()
        recovered = by_slot_hour.sum()
        assert original.equals(recovered), (
            f"Hourly dual-node aggregation lost traffic: {original} vs {recovered}")
        print("Hourly modal totals:         " + ", ".join(
            f"{mode}={int(original[mode]):,}" for mode in FLOW_MODES))
        print("Flow conservation:           PASS (all four modes)")
    print("PASS: fixed-node and scenario-mapping consistency checks")


def main() -> None:
    inspect(GRAPH_DIR, SCENARIO_DIR)


if __name__ == "__main__":
    main()

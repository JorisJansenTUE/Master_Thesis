"""CLI: select and save fixed, model-ready GNN inputs once per baseline."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .input_tensors import DEFAULT_STATIC_COLUMNS, select_input_tensors
from gnn_surrogate.preprocessing.targets.target_tensors import build_baseline_travel_times


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", required=True, type=Path,
                        help="Directory containing dual_nodes.csv.gz and dual_edge_index.npy")
    parser.add_argument("--baseline-scenario", required=True, type=Path,
                        help="Unmodified baseline_00000 directory with targets/")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--features", nargs="+", default=list(DEFAULT_STATIC_COLUMNS),
                        help="Numeric columns from dual_nodes.csv.gz to select")
    parser.add_argument("--hours", type=int, default=30)
    parser.add_argument("--hour", type=int, default=8,
                        help="Morning peak hour for baseline travel-time features")
    args = parser.parse_args()

    # Prevent accidentally using an intervention scenario's flows as baseline inputs.
    interventions = args.baseline_scenario / "interventions.json"
    if not interventions.is_file() or json.loads(interventions.read_text(encoding="utf-8")) != []:
        raise ValueError("--baseline-scenario must contain an empty interventions.json")
    scenario_json = args.baseline_scenario / "scenario.json"
    if scenario_json.is_file():
        manifest = json.loads(scenario_json.read_text(encoding="utf-8"))
        if manifest.get("strategy") != "baseline":
            raise ValueError("--baseline-scenario must have strategy='baseline'")

    graph_nodes = args.graph / "dual_nodes.csv.gz"
    graph_edges = args.graph / "dual_edge_index.npy"
    if not graph_edges.is_file():
        raise FileNotFoundError(graph_edges)
    nodes = pd.read_csv(graph_nodes, dtype={"baseline_link_ids": str})
    baseline_path = args.baseline_scenario / "targets" / "output_detailed_link_statistics.csv.gz"
    # Build baseline travel-time references once, alongside the fixed graph.
    # Only general-road travel time is predicted; existing bus _spec links are not targets.
    tt, present, freeflow = build_baseline_travel_times(
        nodes, args.baseline_scenario, args.hour)
    np.savez_compressed(args.graph / "baseline_tt.npz",
                        baseline_tt_s=tt, present_mask=present,
                        baseline_freeflow_s=freeflow,
                        hour=np.asarray(args.hour, dtype=np.int16))
    # Feed only the road travel time as one additional static feature.
    nodes["baseline_road_tt_s"] = tt[:, 0]
    columns = tuple(args.features) + ("baseline_road_tt_s",)
    arrays, metadata = select_input_tensors(
        nodes, baseline_path, columns=columns, hours=args.hours
    )
    metadata["travel_time_hour"] = args.hour
    args.out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out / "fixed_inputs.npz", **arrays)
    (args.out / "input_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Saved fixed inputs: {args.out / 'fixed_inputs.npz'}")
    for key, value in arrays.items():
        print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
    print(f"  Static feature names: {', '.join(metadata['static_feature_names'])}")
    print("  Graph connectivity remains in dual_edge_index.npy; it is not duplicated.")


if __name__ == "__main__":
    main()

"""CLI: build a scenario's signed hourly targets from saved MATSim outputs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from gnn_surrogate.preprocessing.static.input_tensors import FLOW_MODES
from .target_tensors import build_delta_targets


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", required=True, type=Path,
                        help="Folder with fixed dual_nodes.csv.gz and dual_edge_index.npy")
    parser.add_argument("--inputs", required=True, type=Path,
                        help="Folder containing fixed_inputs.npz and input_metadata.json")
    parser.add_argument("--scenario", required=True, type=Path,
                        help="Scenario directory with link_mapping.csv.gz and targets/")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    if not (args.graph / "dual_edge_index.npy").is_file():
        raise FileNotFoundError(args.graph / "dual_edge_index.npy")
    nodes = pd.read_csv(
        args.graph / "dual_nodes.csv.gz",
        dtype={"segment_id": str, "from_node": str,
               "to_node": str, "baseline_link_ids": str},
    )
    metadata = json.loads((args.inputs / "input_metadata.json").read_text(encoding="utf-8"))
    with np.load(args.inputs / "fixed_inputs.npz", allow_pickle=False) as fixed:
        baseline = fixed["baseline_hourly_raw"]
    if (metadata["flow_modes"] != list(FLOW_MODES)
            or metadata["node_count"] != len(nodes)
            or metadata["hours"] != baseline.shape[1]):
        raise ValueError("Fixed input metadata does not match graph or flow mode order")

    delta, report = build_delta_targets(nodes, args.scenario, baseline)
    args.out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out / "target_delta.npz", y_delta=delta)
    details = {
        "scenario": args.scenario.name,
        "shape": list(delta.shape),
        "dtype": str(delta.dtype),
        "flow_modes": list(FLOW_MODES),
        "target": "scenario_hourly_raw - baseline_hourly_raw",
        "units": "signed vehicle traversals/hour",
        **report,
    }
    (args.out / "target_metadata.json").write_text(
        json.dumps(details, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Saved {args.out / 'target_delta.npz'}: {delta.shape} (N,T,M)")
    print("Scenario modal totals: " + ", ".join(
        f"{mode}={report['scenario_totals'][mode]:,}" for mode in FLOW_MODES))
    print("Delta modal totals: " + ", ".join(
        f"{mode}={report['delta_totals'][mode]:+,}" for mode in FLOW_MODES))
    print("PASS: fixed-node mapping, per-hour conservation, baseline-relative reconstruction")


if __name__ == "__main__":
    main()

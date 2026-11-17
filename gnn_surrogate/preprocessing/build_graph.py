"""CLI: construct and save the fixed dual graph once for all scenarios."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .dual_graph import build_dual_graph


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--segments", type=Path, required=True,
                        help="Dataset-root street_segments.csv.gz")
    parser.add_argument("--baseline-network", type=Path, required=True,
                        help="Original network_with_widths.xml.gz or baseline_00000/network.xml.gz")
    parser.add_argument("--baseline-flows", type=Path,
                        help="Optional baseline scenario's hourly link statistics")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    nodes, edge_index = build_dual_graph(
        args.segments,
        baseline_network_path=args.baseline_network,
        baseline_flows_path=args.baseline_flows,
    )
    args.out.mkdir(parents=True, exist_ok=True)
    nodes.to_csv(args.out / "dual_nodes.csv.gz", index=False, compression="gzip")
    np.save(args.out / "dual_edge_index.npy", edge_index)
    print(f"Saved {len(nodes):,} directed nodes and {edge_index.shape[1]:,} "
          f"potential dual edges to {args.out}")
    if args.baseline_flows is None:
        print("Optional baseline daily traffic fields are empty (no --baseline-flows).")


if __name__ == "__main__":
    main()

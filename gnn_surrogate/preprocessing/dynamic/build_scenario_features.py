"""Build one scenario's infrastructure-change node tensor (no target flows)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .scenario_features import (DEFAULT_DYNAMIC_COLUMNS,
                                build_scenario_feature_table, select_dynamic_features)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, required=True,
                        help="Folder containing dual_nodes.csv.gz")
    parser.add_argument("--scenario", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--features", nargs="+", default=DEFAULT_DYNAMIC_COLUMNS)
    args = parser.parse_args()

    nodes = pd.read_csv(args.graph / "dual_nodes.csv.gz",
                        dtype={"segment_id": str, "from_node": str,
                               "to_node": str, "baseline_link_ids": str})
    table = build_scenario_feature_table(
        nodes, scenario_dir=args.scenario)
    x, names = select_dynamic_features(table, tuple(args.features))
    args.out.mkdir(parents=True, exist_ok=True)
    np.save(args.out / "x_dynamic.npy", x)
    (args.out / "dynamic_metadata.json").write_text(
        json.dumps({"feature_names": names, "node_count": len(nodes),
                    "scenario": args.scenario.name}, indent=2) + "\n")
    print(f"Saved {args.out / 'x_dynamic.npy'}: {x.shape} (N, F_dynamic)")
    print("Features: " + ", ".join(names))


if __name__ == "__main__":
    main()

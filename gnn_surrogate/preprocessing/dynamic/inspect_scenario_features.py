"""Quick inspection and independent intervention audit; edit paths below."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .intervention_audit import audit_scenario_interventions
from .scenario_features import build_scenario_feature_table, select_dynamic_features
from preprocessing.utils import PROJECT_ROOT
# Settings — no command-line arguments.
# Settings: adjust these paths for your own repository.

GRAPH_DIR = PROJECT_ROOT / "data" / "GNN" / "gnn_static"
BASELINE_NETWORK = PROJECT_ROOT / "data" / "NDA_Eth_Model" / "Final_Baseline" / "network_with_widths.xml.gz"
SCENARIO_DIR = Path("C:/Users/20201733/Downloads/cluster_00000")


def main() -> None:
    nodes = pd.read_csv(
        GRAPH_DIR / "dual_nodes.csv.gz",
        dtype={"segment_id": str, "from_node": str,
               "to_node": str, "baseline_link_ids": str},
    )
    table = build_scenario_feature_table(nodes, scenario_dir=SCENARIO_DIR)
    x, names = select_dynamic_features(table)
    print(f"Dynamic input:  {x.shape} (N, F_dynamic)")
    print(f"Scenario:       {SCENARIO_DIR.name}")
    for name in names:
        v = table[name].to_numpy()
        print(f"  {name:<35s} nonzero={np.count_nonzero(v):>5,}, "
              f"min={v.min():>8.3f}, max={v.max():>8.3f}")
    print(f"PASS: {len(nodes):,} fixed nodes, finite scenario features")

    print("\nChecking interventions against independent baseline + sampler rules...")
    report = audit_scenario_interventions(
        nodes, table,
        baseline_network_path=BASELINE_NETWORK,
        scenario_dir=SCENARIO_DIR,
    )
    print(f"Interventions checked: {report.interventions:,}")
    for (kind, variant), count in sorted(report.by_type_variant.items()):
        label = kind if variant is None else f"{kind} ({variant})"
        print(f"  {label:<43s} {count:>4}")
    print(f"Original MATSim links checked: {report.original_links_checked:,}")
    print(f"New MATSim links checked:      {report.new_links_checked:,}")
    print(f"Dedicated bus-route replacements checked: {report.bus_routes_checked:,}")
    print("Dynamic feature cross-check:")
    for name, count in report.expected_feature_counts.items():
        print(f"  {name:<35s} {count:>5,} expected nonzero")
    print("PASS: intervention rules, affected directions, lane capacities, speeds, "
          "permissions, generated links, mapping, PT schedule and all 9 dynamic features")


if __name__ == "__main__":
    main()

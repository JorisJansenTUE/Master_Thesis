"""Quick target inspection: edit paths below, then run as a Python module."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from gnn_surrogate.preprocessing.static.input_tensors import FLOW_MODES
from .target_tensors import load_scenario_hourly
from preprocessing.utils import PROJECT_ROOT
# ------------------------------------------------------------------
# Local inspection settings (no CLI arguments needed)
# ------------------------------------------------------------------

GRAPH_DIR = PROJECT_ROOT / "data" / "GNN" / "gnn_static"
INPUTS_DIR = PROJECT_ROOT / "data" / "GNN" / "gnn_inputs"
SCENARIO_DIR = Path("C:/Users/20201733/Downloads/cluster_00000")
TARGET_DIR = PROJECT_ROOT / "data" / "GNN" / "targets" / "cluster_00000"


def main() -> None:
    nodes = pd.read_csv(GRAPH_DIR / "dual_nodes.csv.gz", dtype={
        "segment_id": str, "from_node": str,
        "to_node": str, "baseline_link_ids": str,
    })
    fixed_metadata = json.loads((INPUTS_DIR / "input_metadata.json").read_text(encoding="utf-8"))
    target_metadata = json.loads((TARGET_DIR / "target_metadata.json").read_text(encoding="utf-8"))
    with np.load(INPUTS_DIR / "fixed_inputs.npz", allow_pickle=False) as fixed:
        baseline = fixed["baseline_hourly_raw"]
    with np.load(TARGET_DIR / "target_delta.npz", allow_pickle=False) as targets:
        delta = targets["y_delta"]

    n, t, m = baseline.shape
    assert fixed_metadata["flow_modes"] == target_metadata["flow_modes"] == list(FLOW_MODES)
    assert n == len(nodes) == fixed_metadata["node_count"]
    assert delta.shape == baseline.shape == tuple(target_metadata["shape"])
    assert np.isfinite(baseline).all() and np.isfinite(delta).all()
    assert target_metadata["scenario"] == SCENARIO_DIR.name

    # Independent re-aggregation from the original sparse MATSim CSV.
    actual, report = load_scenario_hourly(nodes, SCENARIO_DIR, hours=t)
    reconstructed = baseline + delta
    assert np.array_equal(actual, reconstructed), (
        "Saved targets + baseline do not reproduce scenario MATSim volumes"
    )
    assert np.array_equal(delta, actual - baseline)
    assert (reconstructed >= 0).all()

    print(f"Scenario:             {SCENARIO_DIR.name}")
    print(f"Target delta:         {delta.shape} (N,T,M), signed vehicles/hour")
    print(f"Mapped MATSim links:  {report['mapped_link_count']:,}")
    print(f"Flow link-hour rows:  {report['link_hour_rows']:,}")
    print(f"Changed node-hours:   {np.count_nonzero(np.any(delta != 0, axis=2)):,}")
    print(f"Positive/negative:    {np.count_nonzero(delta > 0):,} / {np.count_nonzero(delta < 0):,}")
    print("Mode               baseline     scenario        delta")
    for k, mode in enumerate(FLOW_MODES):
        a = int(baseline[:, :, k].sum(dtype=np.float64))
        b = int(actual[:, :, k].sum(dtype=np.float64))
        print(f"{mode:<12s} {a:>12,} {b:>12,} {b-a:>+12,}")
    print("PASS: saved targets match original MATSim flows, hourly conservation and baseline reconstruction")


if __name__ == "__main__":
    main()

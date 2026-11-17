"""CLI: inspect previously selected fixed GNN input arrays without rebuilding."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from preprocessing.utils import PROJECT_ROOT   
# ------------------------------------------------------------------
# Local inspection settings (edit these paths; no CLI arguments needed)
# ------------------------------------------------------------------

GRAPH_DIR = PROJECT_ROOT / "data" / "GNN" / "gnn_static"
INPUTS_DIR = PROJECT_ROOT / "data" / "GNN" / "gnn_inputs"


def main() -> None:
    nodes = pd.read_csv(GRAPH_DIR / "dual_nodes.csv.gz")
    edge_index = np.load(GRAPH_DIR / "dual_edge_index.npy", allow_pickle=False)
    metadata = json.loads((INPUTS_DIR / "input_metadata.json").read_text(encoding="utf-8"))
    with np.load(INPUTS_DIR / "fixed_inputs.npz", allow_pickle=False) as data:
        static = data["x_static"]
        input_hourly = data["x_baseline_hourly"]
        raw = data["baseline_hourly_raw"]
    n, hours, modes = raw.shape
    assert n == len(nodes) == metadata["node_count"]
    assert static.shape == (n, len(metadata["static_feature_names"]))
    assert input_hourly.shape == raw.shape == (n, metadata["hours"], len(metadata["flow_modes"]))
    assert edge_index.shape[0] == 2 and edge_index.max() < n
    assert all(np.isfinite(array).all() for array in (static, input_hourly, raw))
    assert (raw >= 0).all()
    assert np.allclose(input_hourly, np.log1p(raw), atol=1e-5)
    assert (raw[nodes.baseline_active.to_numpy() == 0] == 0).all()
    print(f"Static input:       {static.shape} (N,F)")
    print(f"Baseline input:     {input_hourly.shape} (N,T,M), log1p counts")
    print(f"Baseline raw flows: {raw.shape} (N,T,M), vehicles/hour")
    print(f"Fixed graph edges:  {edge_index.shape[1]:,}")
    print("Static features:    " + ", ".join(metadata["static_feature_names"]))
    print("Baseline totals:    " + ", ".join(
        f"{mode}={int(raw[:, :, i].sum()):,}"
        for i, mode in enumerate(metadata["flow_modes"])
    ))
    print("PASS: tensor shape, finiteness, baseline alignment and log transform")


if __name__ == "__main__":
    main()

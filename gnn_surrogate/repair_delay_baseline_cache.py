"""Repair an older congestion-delay fixed cache without rebuilding its model inputs.

Run from the delay experiment's project root. Verifies the existing baseline
cache and one scenario's trained targets before writing baseline_freeflow_s.
"""
from __future__ import annotations

import argparse
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from gnn_surrogate.preprocessing.targets.target_tensors import (
    build_baseline_travel_times,
    build_delta_targets,
)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gnn-dir", type=Path, required=True)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--scenario", default="random_00768")
    args = p.parse_args()

    static = args.gnn_dir / "gnn_static"
    cache_path = static / "baseline_tt.npz"
    nodes_path = static / "dual_nodes.csv.gz"
    baseline_path = args.dataset / "baseline_00000"
    scenario_path = args.dataset / args.scenario
    target_path = args.gnn_dir / "targets" / args.scenario / "target_delta.npz"

    with np.load(cache_path, allow_pickle=False) as file:
        cache = {key: file[key] for key in file.files}
    hour = int(cache["hour"])
    nodes = pd.read_csv(nodes_path, dtype={
        "segment_id": str, "from_node": str, "to_node": str,
        "baseline_link_ids": str,
    })
    baseline_tt, baseline_present, baseline_ff = build_baseline_travel_times(
        nodes, baseline_path, hour
    )

    if not np.array_equal(cache["present_mask"], baseline_present):
        raise RuntimeError("Baseline eligibility differs. Cache not modified.")
    if not np.allclose(cache["baseline_tt_s"], baseline_tt, atol=1e-3, rtol=1e-6):
        raise RuntimeError("Baseline travel times differ. Cache not modified.")
    if "baseline_freeflow_s" in cache and not np.allclose(
        cache["baseline_freeflow_s"], baseline_ff, atol=1e-3, rtol=1e-6
    ):
        raise RuntimeError("Cached baseline free-flow values disagree. Cache not modified.")

    expected_y, expected_mask, _ = build_delta_targets(
        nodes, scenario_path, cache["baseline_tt_s"],
        cache["present_mask"], baseline_ff, hour=hour
    )
    with np.load(target_path, allow_pickle=False) as file:
        stored_y = file["y_delta"]
        stored_mask = file["valid_mask"]
    if not np.array_equal(stored_mask, expected_mask):
        raise RuntimeError("Saved scenario eligibility differs. Cache not modified.")
    max_difference = float(np.max(np.abs(stored_y[stored_mask] - expected_y[stored_mask])))
    print(f"Verified baseline and scenario {args.scenario}; hour={hour}")
    print(f"Maximum difference between saved and recomputed delay targets: {max_difference:.6f} s")
    if max_difference > 1e-3:
        raise RuntimeError(
            "Saved targets do not match congestion-delay changes. "
            "Do not evaluate or relabel the dataset until this is understood."
        )

    if "baseline_freeflow_s" in cache:
        print("baseline_freeflow_s already exists and matches; no changes needed.")
        return

    cache["baseline_freeflow_s"] = baseline_ff
    backup = cache_path.with_suffix(".npz.bak")
    shutil.copy2(cache_path, backup)
    fd, temp_name = tempfile.mkstemp(prefix="baseline_tt_", suffix=".npz", dir=static)
    os.close(fd)
    try:
        np.savez_compressed(temp_name, **cache)
        os.replace(temp_name, cache_path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)
    print(f"Added baseline_freeflow_s, shape={baseline_ff.shape}")
    print(f"Preserved backup: {backup}")
    print("No targets, training tensors, split files, or checkpoints were changed.")


if __name__ == "__main__":
    main()

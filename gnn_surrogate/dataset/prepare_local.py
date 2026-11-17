"""Preprocess all intervention scenarios in data/Simulated_Runs, then split.

Edit SETTINGS below and run from the project root:
    python -m gnn_surrogate.data.prepare_local

Requires gnn_static and gnn_inputs to have been built from the SAME baseline
as the runs in data/Simulated_Runs. Only intervention scenarios are examples.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .build_splits import build_split_manifest
from .dataset import discover_scenarios, load_splits
from preprocessing.utils import PROJECT_ROOT

# SETTINGS:
RUNS_DIR = PROJECT_ROOT / "data" / "Simulated_Runs"
GNN_DIR = PROJECT_ROOT / "data" / "GNN"
GRAPH_DIR = GNN_DIR / "gnn_static"
INPUTS_DIR = GNN_DIR / "gnn_inputs"
DYNAMIC_DIR = GNN_DIR / "dynamic"
TARGETS_DIR = GNN_DIR / "targets"
LOCAL_SPLITS_FILE = GNN_DIR / "splits_local.json"  # NEVER overwrite full-dataset splits.json
SEED = 42
REBUILD_EXISTING = False  # True if you need to regenerate scenario tensors


def _build(module: str, *args: Path) -> None:
    command = [sys.executable, "-m", module, *map(str, args)]
    print("  " + " ".join(command), flush=True)
    subprocess.run(command, check=True, cwd=PROJECT_ROOT)


def main() -> None:
    for path in (GRAPH_DIR / "dual_nodes.csv.gz", GRAPH_DIR / "dual_edge_index.npy",
                 INPUTS_DIR / "fixed_inputs.npz", INPUTS_DIR / "input_metadata.json"):
        if not path.is_file():
            raise FileNotFoundError(f"Build fixed graph and baseline inputs first: {path}")
    if not RUNS_DIR.is_dir():
        raise FileNotFoundError(RUNS_DIR)

    baseline_dirs = [p for p in RUNS_DIR.iterdir() if p.is_dir() and p.name.lower().startswith("baseline")]
    if len(baseline_dirs) != 1:
        raise ValueError("Expected exactly one baseline_* folder in Simulated_Runs")
    baseline = baseline_dirs[0]
    if not (baseline / "targets" / "output_detailed_link_statistics.csv.gz").is_file():
        raise FileNotFoundError("Baseline detailed link statistics missing in " + str(baseline))

    # Catch a common mistake: using GNN inputs from a different baseline run.
    # Check exact hourly/mode totals; node-level alignment is checked by
    # the existing baseline and target preprocessing modules.
    with np.load(INPUTS_DIR / "fixed_inputs.npz", allow_pickle=False) as data:
        reference = data["baseline_hourly_raw"]
    modes = ["cars", "trucks", "bikes", "buses"]
    baseline_flow_file = baseline / "targets" / "output_detailed_link_statistics.csv.gz"
    source = pd.read_csv(baseline_flow_file, usecols=["hour", *modes])
    source["hour"] = pd.to_numeric(source["hour"], errors="raise")
    if (source["hour"].isna().any() or not source["hour"].between(0, reference.shape[1] - 1).all()
            or (source["hour"] % 1 != 0).any()):
        raise ValueError("Baseline flow file contains invalid hours")
    per_hour = (source.groupby("hour")[modes].sum()
                .reindex(range(reference.shape[1]), fill_value=0)
                .to_numpy(dtype=np.float64))
    if not np.array_equal(per_hour, reference.sum(axis=0, dtype=np.float64)):
        raise ValueError("Baseline hourly totals disagree with gnn_inputs/fixed_inputs.npz. "
                         "Rebuild fixed inputs from this Simulated_Runs baseline before continuing.")
    print("Baseline hourly modal totals agree with fixed input tensors")

    scenarios = sorted(p for p in RUNS_DIR.iterdir() if p.is_dir() and p != baseline)
    if len(scenarios) < 3:
        raise ValueError("Need at least 3 intervention scenarios for nonempty train/val/test splits")

    requirements = ("network.xml.gz", "interventions.json", "link_mapping.csv.gz",
                    "targets/output_detailed_link_statistics.csv.gz")
    for scenario in scenarios:
        missing = [rel for rel in requirements if not (scenario / rel).is_file()]
        if missing:
            raise FileNotFoundError(f"Incomplete scenario {scenario.name}: {missing}")
        interventions = json.loads((scenario / "interventions.json").read_text(encoding="utf-8"))
        if not isinstance(interventions, list) or not interventions:
            raise ValueError(f"Expected a nonempty intervention list: {scenario}")

    print(f"Baseline (excluded from training): {baseline.name}")
    print(f"Found {len(scenarios)} intervention scenarios")
    for scenario in scenarios:
        dyn = DYNAMIC_DIR / scenario.name
        target = TARGETS_DIR / scenario.name
        dyn_complete = (dyn / "x_dynamic.npy").is_file() and (dyn / "dynamic_metadata.json").is_file()
        target_complete = (target / "target_delta.npz").is_file() and (target / "target_metadata.json").is_file()
        print(f"\n{scenario.name}:")
        if REBUILD_EXISTING or not dyn_complete:
            _build("gnn_surrogate.preprocessing.dynamic.build_scenario_features",
                   "--graph", GRAPH_DIR, "--scenario", scenario, "--out", dyn)
        else:
            print("  Dynamic features already exist; skipping")
        if REBUILD_EXISTING or not target_complete:
            _build("gnn_surrogate.preprocessing.targets.build_targets",
                   "--graph", GRAPH_DIR, "--inputs", INPUTS_DIR,
                   "--scenario", scenario, "--out", target)
        else:
            print("  Targets already exist; skipping")

    complete = discover_scenarios(DYNAMIC_DIR, TARGETS_DIR)
    selected = [p.name for p in scenarios]
    if not set(selected).issubset(complete):
        raise ValueError("Missing processed feature/target pairs after building")
    # Use only scenarios from Simulated_Runs, even if other processed scenarios exist.
    if LOCAL_SPLITS_FILE.exists():
        splits = load_splits(LOCAL_SPLITS_FILE)
        observed = set(splits["train"] + splits["val"] + splits["test"])
        if observed != set(selected):
            raise ValueError(f"Existing {LOCAL_SPLITS_FILE} has different scenarios. "
                             "Review and rename it before regenerating.")
        print("\nReused existing local split (no reshuffle)")
    else:
        manifest = build_split_manifest(selected, seed=SEED)
        LOCAL_SPLITS_FILE.parent.mkdir(parents=True, exist_ok=True)
        LOCAL_SPLITS_FILE.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        splits = manifest
        print(f"\nSaved {LOCAL_SPLITS_FILE}")
    print("Split counts: " + ", ".join(f"{name}={len(splits[name])}" for name in ("train", "val", "test")))
    print("PASS: baseline excluded; matching dynamic/target pairs and disjoint local splits")


if __name__ == "__main__":
    main()

"""Build masked 08:00-09:00 (configurable) travel-time change targets."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .target_tensors import CHANNELS, build_delta_targets


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--scenario", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--hour", type=int, default=8)
    args = parser.parse_args()

    nodes = pd.read_csv(args.graph / "dual_nodes.csv.gz", dtype={
        "segment_id": str, "from_node": str, "to_node": str, "baseline_link_ids": str})
    with np.load(args.graph / "baseline_tt.npz", allow_pickle=False) as fixed:
        if int(fixed["hour"]) != args.hour:
            raise ValueError("Target hour differs from gnn_static baseline hour")
        baseline_tt, baseline_present = fixed["baseline_tt_s"], fixed["present_mask"]
    delta, valid, report = build_delta_targets(
        nodes, args.scenario, baseline_tt, baseline_present, hour=args.hour)
    args.out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out / "target_delta.npz", y_delta=delta, valid_mask=valid)
    metadata = {"scenario": args.scenario.name, "shape": list(delta.shape),
                "channels": list(CHANNELS), "hour": args.hour, "units": "seconds", **report}
    (args.out / "target_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"{args.scenario.name}: y_delta={delta.shape}, valid={int(valid.sum())}, "
          f"new bus links={report['new_bus_spec_targets']}")


if __name__ == "__main__":
    main()

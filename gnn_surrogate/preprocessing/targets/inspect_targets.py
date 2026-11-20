"""Inspect an already generated travel-time target; edit paths below."""
from pathlib import Path
import numpy as np
from preprocessing.utils import PROJECT_ROOT

GNN_DIR = PROJECT_ROOT / "data" / "GNN"
SCENARIO_ID = "cluster_00051"


def main():
    target = GNN_DIR / "targets" / SCENARIO_ID / "target_delta.npz"
    with np.load(target, allow_pickle=False) as file:
        y, mask = file["y_delta"], file["valid_mask"]
    assert y.shape == mask.shape and y.shape[1] == 2 and mask.dtype == np.bool_
    print("Scenario:", SCENARIO_ID, "target:", y.shape)
    for channel, name in enumerate(("road", "bus_spec")):
        values = y[:, channel][mask[:, channel]]
        print(name, "eligible:", len(values), "mean delta [s]:",
              float(values.mean()) if len(values) else None)
    print("PASS: target delta and eligibility mask")


if __name__ == "__main__":
    main()

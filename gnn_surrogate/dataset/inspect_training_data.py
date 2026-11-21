"""Inspect fixed inputs, masked travel-time target shapes, and no-change MSE."""
from pathlib import Path
import torch
from .dataset import ScenarioDataset, load_splits
from gnn_surrogate.training.losses import masked_mse_loss
from preprocessing.utils import PROJECT_ROOT

GNN_DIR = PROJECT_ROOT / "data" / "GNN"
SPLITS_FILE = GNN_DIR / "splits.json"


def main():
    splits = load_splits(SPLITS_FILE)
    data = ScenarioDataset(GNN_DIR / "gnn_static", GNN_DIR / "gnn_inputs",
                           GNN_DIR / "dynamic", GNN_DIR / "targets",
                           scenario_ids=[splits["train"][0]])
    item = data[0]
    print("Splits:", {k: len(v) for k, v in splits.items()})
    print("Fixed:", {k: tuple(v.shape) for k, v in data.fixed.items()})
    print("Dynamic:", tuple(item["x_dynamic"].shape),
          "target:", tuple(item["y_delta"].shape))
    print("Eligible road:", item["valid_mask"].sum(dim=0).tolist())
    loss = masked_mse_loss(torch.zeros_like(item["y_delta"]),
                           item["y_delta"], item["valid_mask"])
    print("No-change MSE [seconds^2]:", float(loss))


if __name__ == "__main__":
    main()

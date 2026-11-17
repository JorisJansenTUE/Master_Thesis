"""Inspect dataset, mode-balanced MSE gradients and no-change benchmark.

Edit SETTINGS, then run:
    python -m gnn_surrogate.dataset.inspect_training_data

Does not train a model. No-change benchmark is not learned accuracy.
"""
from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .dataset import ScenarioDataset, load_splits
from gnn_surrogate.training.losses import (
    estimate_mode_scales, mode_balanced_mse_loss, modal_mae_rmse,
)

# SETTINGS: replace PROJECT_ROOT with your existing preprocessing.utils helper.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
GNN_DIR = PROJECT_ROOT / "data" / "GNN"
GRAPH_DIR = GNN_DIR / "gnn_static"
INPUTS_DIR = GNN_DIR / "gnn_inputs"
DYNAMIC_DIR = GNN_DIR / "dynamic"
TARGETS_DIR = GNN_DIR / "targets"
SPLITS_FILE = GNN_DIR / "splits_local.json"

TEMPORAL_MODE = "daily"       # "daily", "peak_hour", or "hourly"
PEAK_HOUR = None              # e.g. 8 for 08:00-08:59; only for "peak_hour"


def _dataset(ids):
    return ScenarioDataset(
        GRAPH_DIR, INPUTS_DIR, DYNAMIC_DIR, TARGETS_DIR,
        scenario_ids=ids, temporal_mode=TEMPORAL_MODE, hour=PEAK_HOUR,
    )


def _format_metrics(mae, rmse, modes):
    return ", ".join(f"{m}: MAE {mae[i]:.4g}, RMSE {rmse[i]:.4g}" for i, m in enumerate(modes))


def main() -> None:
    splits = load_splits(SPLITS_FILE)
    print("Splits:", ", ".join(f"{k}={len(v)}" for k, v in splits.items()))
    datasets = {name: _dataset(ids) for name, ids in splits.items()}
    train = datasets["train"]
    fixed = train.fixed
    modes = train.target_modes
    print(f"Baseline input modes: {', '.join(train.input_modes)}")
    print(f"Prediction target modes: {', '.join(modes)}")
    units = (f"vehicles/{train.hours}-hour simulation" if TEMPORAL_MODE == "daily"
             else "vehicles/hour")
    print(f"Temporal selection: {TEMPORAL_MODE}" +
          (f" (hour {PEAK_HOUR})" if TEMPORAL_MODE == "peak_hour" else ""))
    scales = estimate_mode_scales(train)
    print(f"Train-only RMS delta scales ({units}): " +
          ", ".join(f"{m}={v:.4g}" for m, v in zip(modes, scales.tolist())))
    print("Shared:", {k: tuple(v.shape) for k, v in fixed.items()})

    batch = next(iter(DataLoader(train, batch_size=min(2, len(train)), shuffle=False, num_workers=0)))
    assert batch["x_dynamic"].ndim == 3
    assert batch["y_delta"].ndim == (4 if TEMPORAL_MODE == "hourly" else 3)
    assert batch["y_delta"].shape[0] == batch["x_dynamic"].shape[0]
    assert batch["y_delta"].shape[-1] == len(modes)
    print("Batch:", tuple(batch["x_dynamic"].shape), tuple(batch["y_delta"].shape))

    sample = train[0]
    pred = torch.zeros_like(sample["y_delta"], requires_grad=True)
    actual = sample["y_delta"]
    loss = mode_balanced_mse_loss(pred, actual, scales)
    exact = mode_balanced_mse_loss(actual, actual, scales)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(exact) and exact.item() == 0.0
    assert pred.grad is not None and torch.isfinite(pred.grad).all()
    print(f"MSE loss: zero-prediction={loss.item():.6g}, perfect-prediction={exact.item():.6g}; gradients PASS")

    # Report raw-flow-count errors and distinguish affected nodes and neighbours.
    for split, dataset in datasets.items():
        for item in dataset:
            y = item["y_delta"]
            predicted_zero = torch.zeros_like(y)
            reconstructed = fixed["baseline_raw"] + y
            assert torch.isfinite(reconstructed).all() and (reconstructed >= -1e-5).all()

            modified = item["x_dynamic"].abs().sum(dim=1) > 0
            neighborhood = modified.clone()
            src, dst = fixed["edge_index"]
            neighborhood[src[modified[dst]]] = True
            neighborhood[dst[modified[src]]] = True

            mae, rmse = modal_mae_rmse(predicted_zero, y)
            print(f"\n[{split}] {item['scenario_id']}")
            print(f"  modified nodes={int(modified.sum())}; modified + 1-hop={int(neighborhood.sum())}")
            print("  No-change benchmark, all nodes: " + _format_metrics(mae, rmse, modes))
            if modified.any():
                affected_mae, affected_rmse = modal_mae_rmse(predicted_zero[modified], y[modified])
                print("  No-change benchmark, modified nodes: " + _format_metrics(affected_mae, affected_rmse, modes))
            if neighborhood.any():
                near_mae, near_rmse = modal_mae_rmse(predicted_zero[neighborhood], y[neighborhood])
                print("  No-change benchmark, modified + 1-hop: " + _format_metrics(near_mae, near_rmse, modes))
            print("  Modal total changes: " + ", ".join(
                f"{m}={y[..., j].sum().item():+,.0f}" for j, m in enumerate(modes)))
    print("\nPASS: loading, splits, batching, MSE gradients, reconstruction and baseline benchmark")
    print("Note: four scenarios are for smoke testing, not trustworthy model generalization estimates.")


if __name__ == "__main__":
    main()

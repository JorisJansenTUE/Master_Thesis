"""Train the daily Locarno GNN-Transformer on baseline-relative modal flows.

Run from the project root:
    python -m gnn_surrogate.training.train

Edit SETTINGS below for local or SLURM runs. Optional CLI overrides are listed
by --help. A limited-node run is ONLY for debugging, not model evaluation.

The training split alone determines the loss scales and fitted parameters.
Validation selects the best checkpoint; test is evaluated ONCE at the end.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from gnn_surrogate.dataset.dataset import ScenarioDataset, load_splits
from gnn_surrogate.models.gnn_transformer import GNNTransformer
from gnn_surrogate.training.losses import estimate_mode_scales, mode_balanced_mse_loss
from preprocessing.utils import PROJECT_ROOT

# ------------------------------ SETTINGS ----------------------------
GNN_DIR = PROJECT_ROOT / "data" / "GNN"
SPLITS_FILE = GNN_DIR / "splits_local.json"   # use splits.json on the cluster
RUN_DIR = GNN_DIR / "runs" / "trial_001"

DEVICE = "auto"          # "auto", "cuda", or "cpu"
MAX_NODES = 128          # local smoke: 128; REAL cluster training: None
EPOCHS = 5              # local smoke: 5; cluster: e.g. 50
BATCH_SIZE = 1          # full-graph global attention is memory intensive
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0
PATIENCE = 10
MIN_DELTA = 0.0
SEED = 42
NUM_WORKERS = 0       # portable on Windows and Linux; tune later if needed

# Same core architecture as the existing GNNTransformer defaults.
MODEL_KWARGS = {
    "hidden_dim": 192,
    "heads": 4,
    "graph_layers": 2,
    "transformer_layers": 3,
    "feedforward_dim": 768,
    "dropout": 0.1,
}
# -------------------------------------------------------------------------


@dataclass
class TrainSettings:
    gnn_dir: Path
    split_file: Path
    run_dir: Path
    device: str = DEVICE
    max_nodes: int | None = MAX_NODES
    epochs: int = EPOCHS
    batch_size: int = BATCH_SIZE
    lr: float = LEARNING_RATE
    weight_decay: float = WEIGHT_DECAY
    grad_clip: float = GRAD_CLIP
    patience: int = PATIENCE
    min_delta: float = MIN_DELTA
    seed: int = SEED
    num_workers: int = NUM_WORKERS


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # A fixed seed aids reproducibility, but does not guarantee fully
    # deterministic CUDA/PyG operations on all GPU architectures.
    torch.backends.cudnn.benchmark = False


def _device(request: str) -> torch.device:
    if request == "auto":
        request = "cuda" if torch.cuda.is_available() else "cpu"
    if request == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
    return torch.device(request)


def _build_dataset(cfg: TrainSettings, scenario_ids: list[str]) -> ScenarioDataset:
    return ScenarioDataset(
        cfg.gnn_dir / "gnn_static",
        cfg.gnn_dir / "gnn_inputs",
        cfg.gnn_dir / "dynamic",
        cfg.gnn_dir / "targets",
        scenario_ids=scenario_ids,
        temporal_mode="daily",
    )


def _trim_graph(fixed: dict[str, torch.Tensor], max_nodes: int | None) -> dict[str, torch.Tensor]:
    n = fixed["x_static"].shape[0]
    if max_nodes is None or max_nodes >= n:
        return fixed
    if max_nodes < 2:
        raise ValueError("max_nodes must be >=2 or None")
    edges = fixed["edge_index"]
    edges = edges[:, (edges < max_nodes).all(dim=0)]
    if edges.shape[1] == 0:
        raise ValueError("No edges in selected node prefix; increase --max-nodes")
    return {
        "edge_index": edges,
        "x_static": fixed["x_static"][:max_nodes],
        "x_baseline": fixed["x_baseline"][:max_nodes],
        "baseline_raw": fixed["baseline_raw"][:max_nodes],
    }


def _loader(dataset: ScenarioDataset, cfg: TrainSettings, *, shuffle: bool) -> DataLoader:
    # The fixed graph stays outside the DataLoader; only scenario inputs and
    # targets are batched. Workers=0 avoids Windows multiprocessing surprises.
    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        num_workers=cfg.num_workers,
        pin_memory=False,
    )


def _prepare_batch(batch: dict, device: torch.device, n: int) -> tuple[torch.Tensor, torch.Tensor]:
    x_dynamic = batch["x_dynamic"][:, :n].to(device, non_blocking=True)
    target = batch["y_delta"][:, :n].to(device, non_blocking=True)
    if not torch.isfinite(x_dynamic).all() or not torch.isfinite(target).all():
        raise ValueError("Nonfinite scenario inputs/targets")
    return x_dynamic, target


def _neighborhood(modified: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
    """Modified positions plus their direct dual-graph predecessors/successors."""
    nearby = modified.clone()
    src, dst = edges
    for b in range(modified.shape[0]):
        nearby[b, src[modified[b, dst]]] = True
        nearby[b, dst[modified[b, src]]] = True
    return nearby


class ErrorSummary:
    """Accumulate raw-count MAE/RMSE across scenarios, not batch averages."""

    def __init__(self, modes: list[str]) -> None:
        self.modes = modes
        self.sums = {
            label: {"abs": torch.zeros(len(modes), dtype=torch.float64),
                    "sq": torch.zeros(len(modes), dtype=torch.float64), "count": 0}
            for label in ("all", "modified", "nearby","unchanged")
        }

    def add(self, prediction: torch.Tensor, target: torch.Tensor,
            x_dynamic: torch.Tensor, edges: torch.Tensor) -> None:
        errors = (prediction.detach() - target).to(torch.float64)
        modified = (x_dynamic != 0).any(dim=-1)
        unchanged = ~modified
        masks = {"all": torch.ones_like(modified, dtype=torch.bool),
                 "modified": modified,
                 "nearby": _neighborhood(modified, edges),
                 "unchanged": unchanged}
        for label, mask in masks.items():
            if not mask.any():
                continue
            sub = errors[mask].cpu()
            stats = self.sums[label]
            stats["abs"] += sub.abs().sum(dim=0)
            stats["sq"] += sub.square().sum(dim=0)
            stats["count"] += sub.shape[0]

    def result(self) -> dict:
        output = {}
        for label, stats in self.sums.items():
            n = stats["count"]
            output[label] = {
                "count": n,
                "mae": {mode: float(stats["abs"][i] / n) if n else None
                        for i, mode in enumerate(self.modes)},
                "rmse": {mode: float((stats["sq"][i] / n).sqrt()) if n else None
                         for i, mode in enumerate(self.modes)},
            }
        return output


def _evaluate(
    model: GNNTransformer,
    loader: DataLoader,
    fixed: dict[str, torch.Tensor],
    scales: torch.Tensor,
    modes: list[str],
    device: torch.device,
    *,
    zero_prediction: bool = False,
) -> dict:
    model.eval()
    metrics = ErrorSummary(modes)
    total_loss = 0.0
    total_scenarios = 0
    n = fixed["x_static"].shape[0]
    with torch.inference_mode():
        for batch in loader:
            x_dynamic, target = _prepare_batch(batch, device, n)
            prediction = (torch.zeros_like(target) if zero_prediction else model(
                fixed["x_static"], x_dynamic, fixed["x_baseline"], fixed["edge_index"]
            ))
            if not torch.isfinite(prediction).all():
                raise FloatingPointError("Nonfinite predictions during evaluation")
            loss = mode_balanced_mse_loss(prediction, target, scales)
            b = target.shape[0]
            total_loss += float(loss) * b
            total_scenarios += b
            metrics.add(prediction, target, x_dynamic, fixed["edge_index"])
    if total_scenarios == 0:
        raise ValueError("Cannot evaluate an empty data split")
    return {"loss": total_loss / total_scenarios, "metrics": metrics.result()}


def _train_one_epoch(
    model: GNNTransformer,
    loader: DataLoader,
    fixed: dict[str, torch.Tensor],
    scales: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    cfg: TrainSettings,
    device: torch.device,
) -> float:
    model.train()
    running_loss = 0.0
    seen = 0
    n = fixed["x_static"].shape[0]
    for batch in loader:
        x_dynamic, target = _prepare_batch(batch, device, n)
        optimizer.zero_grad(set_to_none=True)
        prediction = model(
            fixed["x_static"], x_dynamic, fixed["x_baseline"], fixed["edge_index"]
        )
        loss = mode_balanced_mse_loss(prediction, target, scales)
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite training loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip,
                                       error_if_nonfinite=True)
        optimizer.step()
        b = target.shape[0]
        running_loss += float(loss.detach()) * b
        seen += b
    return running_loss / seen


def _save_checkpoint(path: Path, model: GNNTransformer,
                     optimizer: torch.optim.Optimizer, epoch: int,
                     validation: dict, scales: torch.Tensor,
                     model_kwargs: dict, cfg: TrainSettings,
                     train_dataset: ScenarioDataset, splits: dict) -> None:
    checkpoint = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "validation": validation,
        "mode_scales": scales.detach().cpu(),
        "model_kwargs": model_kwargs,
        "config": {k: str(v) if isinstance(v, Path) else v
                   for k, v in asdict(cfg).items()},
        "splits": splits,
        "input_modes": train_dataset.input_modes,
        "target_modes": train_dataset.target_modes,
        "static_feature_names": train_dataset.input_metadata["static_feature_names"],
        "dynamic_feature_names": train_dataset.dynamic_feature_names,
        "torch_version": torch.__version__,
    }
    torch.save(checkpoint, path)


def _json_save(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def train(cfg: TrainSettings, *, model_overrides: dict | None = None) -> dict:
    if cfg.epochs < 1 or cfg.batch_size < 1 or cfg.patience < 1 or cfg.grad_clip <= 0:
        raise ValueError("epochs, batch_size, patience and grad_clip must be positive")
    if cfg.run_dir.exists() and any(cfg.run_dir.iterdir()):
        raise FileExistsError(
            f"Run directory is not empty: {cfg.run_dir}. Choose a NEW run directory."
        )
    _seed_everything(cfg.seed)
    device = _device(cfg.device)
    splits = load_splits(cfg.split_file)
    train_dataset = _build_dataset(cfg, splits["train"])
    val_dataset = _build_dataset(cfg, splits["val"])
    test_dataset = _build_dataset(cfg, splits["test"])
    fixed = _trim_graph(train_dataset.fixed, cfg.max_nodes)
    fixed = {k: v.to(device) for k, v in fixed.items()}

    # Calculate normalization only from TRAINING targets (all nodes, even
    # during limited-node smoke tests, which are not for model comparison).
    scales = estimate_mode_scales(train_dataset).to(device)
    example = train_dataset[0]
    model_kwargs = dict(MODEL_KWARGS)
    if model_overrides:
        model_kwargs.update(model_overrides)
    model_kwargs.update({
        "static_dim": fixed["x_static"].shape[1],
        "dynamic_dim": example["x_dynamic"].shape[-1],
        "baseline_dim": fixed["x_baseline"].shape[-1],
        "output_dim": example["y_delta"].shape[-1],
    })
    model = GNNTransformer(**model_kwargs).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                                  weight_decay=cfg.weight_decay)
    train_loader = _loader(train_dataset, cfg, shuffle=True)
    val_loader = _loader(val_dataset, cfg, shuffle=False)
    test_loader = _loader(test_dataset, cfg, shuffle=False)

    cfg.run_dir.mkdir(parents=True, exist_ok=True)
    _json_save(cfg.run_dir / "settings.json", {
        "config": {k: str(v) if isinstance(v, Path) else v
                   for k, v in asdict(cfg).items()},
        "model_kwargs": model_kwargs,
        "scales": dict(zip(train_dataset.target_modes, scales.cpu().tolist())),
        "split_sizes": {k: len(v) for k, v in splits.items()},
        "node_count": fixed["x_static"].shape[0],
        "edge_count": fixed["edge_index"].shape[1],
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "device": str(device),
        "note": "max_nodes < full graph is a debugging run only",
    })

    print(f"Device={device}; graph={fixed['x_static'].shape[0]:,} nodes / "
          f"{fixed['edge_index'].shape[1]:,} edges; "
          f"parameters={sum(p.numel() for p in model.parameters()):,}", flush=True)
    print(f"Scenarios train/val/test: {len(train_dataset)}/{len(val_dataset)}/{len(test_dataset)}",
          flush=True)
    print("Train-only scales: " + ", ".join(
        f"{m}={v:.4g}" for m, v in zip(train_dataset.target_modes, scales.cpu().tolist())
    ), flush=True)
    if fixed["x_static"].shape[0] < train_dataset.n_nodes:
        print("WARNING: SUBGRAPH DEBUG MODE — losses are NOT representative of full graph.",
              flush=True)

    # Baseline benchmark calculated once, on validation only.
    zero_val = _evaluate(model, val_loader, fixed, scales,
                         train_dataset.target_modes, device, zero_prediction=True)
    _json_save(cfg.run_dir / "no_change_validation.json", zero_val)
    print(f"No-change validation loss: {zero_val['loss']:.6g}", flush=True)

    best_loss = float("inf")
    best_epoch = 0
    stale = 0
    history_path = cfg.run_dir / "history.csv"
    with history_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["epoch", "train_mse", "val_mse", "val_modified_car_rmse",
                         "val_modified_truck_rmse", "val_modified_bike_rmse",
                         "val_nearby_car_rmse", "val_unchanged_car_rmse", "val_unchanged_truck_rmse", "val_unchanged_bike_rmse", "epoch_seconds"])
        for epoch in range(1, cfg.epochs + 1):
            start = time.monotonic()
            train_loss = _train_one_epoch(model, train_loader, fixed,
                                          scales, optimizer, cfg, device)
            validation = _evaluate(model, val_loader, fixed, scales,
                                   train_dataset.target_modes, device)
            val_loss = validation["loss"]
            if not np.isfinite(val_loss):
                raise FloatingPointError("Nonfinite validation loss")
            elapsed = time.monotonic() - start
            rmse_modified = validation["metrics"]["modified"]["rmse"]
            rmse_nearby = validation["metrics"]["nearby"]["rmse"]
            rmse_unchanged = validation["metrics"]["unchanged"]["rmse"]
            writer.writerow([epoch, train_loss, val_loss,
                             rmse_modified.get("cars"), rmse_modified.get("trucks"),
                             rmse_modified.get("bikes"), rmse_nearby.get("cars"), 
                             rmse_unchanged.get("cars"), rmse_unchanged.get("trucks"), rmse_unchanged.get("bikes"), elapsed])
            file.flush()
            print(f"Epoch {epoch:03d} | train={train_loss:.5g} | val={val_loss:.5g} "
                  f"| modified car RMSE={rmse_modified.get('cars')} "
                  f"| {elapsed:.1f}s", flush=True)

            _save_checkpoint(cfg.run_dir / "last.pt", model, optimizer, epoch,
                             validation, scales, model_kwargs, cfg, train_dataset, splits)
            if val_loss < best_loss - cfg.min_delta:
                best_loss, best_epoch, stale = val_loss, epoch, 0
                _save_checkpoint(cfg.run_dir / "best.pt", model, optimizer, epoch,
                                 validation, scales, model_kwargs, cfg, train_dataset, splits)
                _json_save(cfg.run_dir / "best_validation.json", validation)
            else:
                stale += 1
            if stale >= cfg.patience:
                print(f"Early stopping: no val improvement for {cfg.patience} epochs",
                      flush=True)
                break

    # Test set is evaluated only AFTER checkpoint selection, never for tuning.
    best_ckpt = torch.load(cfg.run_dir / "best.pt", map_location=device,
                           weights_only=False)  # our own trusted checkpoint
    model.load_state_dict(best_ckpt["model_state"])
    test_result = _evaluate(model, test_loader, fixed, scales,
                            train_dataset.target_modes, device)
    zero_test = _evaluate(model, test_loader, fixed, scales,
                          train_dataset.target_modes, device, zero_prediction=True)
    _json_save(cfg.run_dir / "test_metrics.json", {
        "model": test_result, "no_change": zero_test,
        "best_epoch": best_epoch,
        "warning": ("Limited-node debug results must not be used for model evaluation"
                    if fixed["x_static"].shape[0] < train_dataset.n_nodes else None),
    })
    summary = {"best_epoch": best_epoch, "best_val_loss": best_loss,
               "no_change_val_loss": zero_val["loss"],
               "test_loss": test_result["loss"],
               "no_change_test_loss": zero_test["loss"],
               "run_dir": str(cfg.run_dir)}
    _json_save(cfg.run_dir / "summary.json", summary)
    print("Best epoch:", best_epoch, "| test loss:", f"{test_result['loss']:.6g}",
          "| no-change test:", f"{zero_test['loss']:.6g}", flush=True)
    for region in ("all", "modified", "nearby"):
        ours = test_result["metrics"][region]
        reference = zero_test["metrics"][region]
        if ours["count"] == 0:
            continue
        print(f"Test {region} ({ours['count']} node-scenario positions):", flush=True)
        for mode in train_dataset.target_modes:
            print(f"  {mode:7s} RMSE={ours['rmse'][mode]:.3f}, "
                  f"no-change={reference['rmse'][mode]:.3f}", flush=True)
    print("Saved:", cfg.run_dir, flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gnn-dir", type=Path, default=GNN_DIR)
    parser.add_argument("--splits", type=Path, default=SPLITS_FILE)
    parser.add_argument("--out", type=Path, default=RUN_DIR)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default=DEVICE)
    parser.add_argument("--max-nodes", type=int, default=MAX_NODES,
                        help="Debug-only prefix of nodes; 0 selects full graph")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--patience", type=int, default=PATIENCE)
    args = parser.parse_args()
    cfg = TrainSettings(
        gnn_dir=args.gnn_dir, split_file=args.splits, run_dir=args.out,
        device=args.device, max_nodes=(None if args.max_nodes == 0 else args.max_nodes),
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        patience=args.patience,
    )
    train(cfg)


if __name__ == "__main__":
    main()

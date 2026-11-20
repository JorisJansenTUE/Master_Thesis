"""Train the unchanged GNN-Transformer backbone on masked link travel-time changes.

The fixed graph and baseline features are loaded once. Each scenario supplies
its existing dynamic [N,9] features, signed travel-time deltas [N,2] in seconds,
and a Boolean eligibility mask [N,2]. Validation selects the best checkpoint.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from gnn_surrogate.dataset.dataset import ScenarioDataset, load_splits
from gnn_surrogate.models.gnn_transformer import GNNTransformer
from gnn_surrogate.training.losses import masked_mse_loss
from preprocessing.utils import PROJECT_ROOT

GNN_DIR = PROJECT_ROOT / "data" / "GNN"
SPLITS_FILE = GNN_DIR / "splits.json"
RUN_DIR = GNN_DIR / "runs" / "tt_001"
DEVICE = "auto"
MAX_NODES = None
EPOCHS = 50
BATCH_SIZE = 1
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0
PATIENCE = 10
MIN_DELTA = 0.0
SEED = 42
NUM_WORKERS = 0

MODEL_KWARGS = dict(hidden_dim=192, heads=4, graph_layers=2,
                    transformer_layers=3, feedforward_dim=768, dropout=0.1)
CHANNELS = ("road", "bus_spec")
REGIONS = ("all", "modified", "nearby", "unchanged")
HISTORY_COLUMNS = ("epoch", "train_mse", "val_mse", "val_road_rmse",
                   "val_bus_spec_rmse", "val_modified_road_rmse", "epoch_seconds")


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
    resume_from: Path | None = None
    skip_test: bool = False


def _save_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


def _device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU requested but unavailable")
    return torch.device(name)


def _dataset(cfg: TrainSettings, ids: list[str]) -> ScenarioDataset:
    return ScenarioDataset(cfg.gnn_dir / "gnn_static", cfg.gnn_dir / "gnn_inputs",
                           cfg.gnn_dir / "dynamic", cfg.gnn_dir / "targets", ids)


def _trim_graph(fixed: dict, max_nodes: int | None) -> dict:
    n = fixed["x_static"].shape[0]
    if max_nodes is None or max_nodes >= n:
        return fixed
    if max_nodes < 2:
        raise ValueError("--max-nodes must be >=2 or 0 for full graph")
    edges = fixed["edge_index"]
    edges = edges[:, (edges < max_nodes).all(dim=0)]
    if edges.shape[1] == 0:
        raise ValueError("Selected debug subgraph has no edges")
    return {"edge_index": edges, **{k: v[:max_nodes] for k, v in fixed.items()
                                     if k != "edge_index"}}


def _loader(data: ScenarioDataset, cfg: TrainSettings, shuffle: bool) -> DataLoader:
    return DataLoader(data, batch_size=cfg.batch_size, shuffle=shuffle,
                      num_workers=cfg.num_workers)


def _batch(batch: dict, device: torch.device, n: int):
    dynamic = batch["x_dynamic"][:, :n].to(device)
    target = batch["y_delta"][:, :n].to(device)
    valid = batch["valid_mask"][:, :n].to(device)
    if not (torch.isfinite(dynamic).all() and torch.isfinite(target).all()
            and valid.dtype == torch.bool and valid.any()):
        raise ValueError("Invalid batch inputs or travel-time eligibility mask")
    return dynamic, target, valid


def _neighbor_mask(modified: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
    nearby = modified.clone()
    src, dst = edges
    for b in range(modified.shape[0]):
        nearby[b, src[modified[b, dst]]] = True
        nearby[b, dst[modified[b, src]]] = True
    return nearby


class ErrorSummary:
    """Masked error in seconds, separately for road and bus-specific channels."""
    def __init__(self):
        self.data = {region: {mode: [0, 0.0, 0.0] for mode in CHANNELS}
                     for region in REGIONS}

    def add(self, pred: torch.Tensor, truth: torch.Tensor, valid: torch.Tensor,
            dynamic: torch.Tensor, edges: torch.Tensor) -> None:
        diff = (pred.detach() - truth).to(torch.float64)
        changed = (dynamic != 0).any(dim=-1)
        regions = {"all": torch.ones_like(changed), "modified": changed,
                   "nearby": _neighbor_mask(changed, edges), "unchanged": ~changed}
        for region, node_mask in regions.items():
            for i, channel in enumerate(CHANNELS):
                masked = diff[..., i][node_mask & valid[..., i]]
                if masked.numel():
                    stats = self.data[region][channel]
                    stats[0] += int(masked.numel())
                    stats[1] += float(masked.abs().sum())
                    stats[2] += float(masked.square().sum())

    def result(self) -> dict:
        result = {}
        for region, by_channel in self.data.items():
            count = sum(v[0] for v in by_channel.values())
            result[region] = {
                "count": count,
                "count_by_channel": {c: v[0] for c, v in by_channel.items()},
                "mae": {c: v[1] / v[0] if v[0] else None for c, v in by_channel.items()},
                "rmse": {c: (v[2] / v[0]) ** 0.5 if v[0] else None
                         for c, v in by_channel.items()},
            }
        return result


def _evaluate(model: GNNTransformer, loader: DataLoader,
              fixed: dict, device: torch.device, zero_prediction: bool = False) -> dict:
    model.eval()
    metrics = ErrorSummary()
    mse_total, scenarios = 0.0, 0
    n = fixed["x_static"].shape[0]
    with torch.inference_mode():
        for batch in loader:
            x, y, mask = _batch(batch, device, n)
            pred = (torch.zeros_like(y) if zero_prediction else model(
                fixed["x_static"], x, fixed["x_baseline"], fixed["edge_index"]))
            mse = masked_mse_loss(pred, y, mask)
            mse_total += float(mse) * y.shape[0]
            scenarios += y.shape[0]
            metrics.add(pred, y, mask, x, fixed["edge_index"])
    if not scenarios:
        raise ValueError("Cannot evaluate empty data split")
    return {"loss": mse_total / scenarios, "metrics": metrics.result()}


def _train_epoch(model: GNNTransformer, loader: DataLoader,
                 fixed: dict, optimizer, cfg: TrainSettings,
                 device: torch.device) -> float:
    model.train()
    loss_total, count = 0.0, 0
    for batch in loader:
        x, y, mask = _batch(batch, device, fixed["x_static"].shape[0])
        optimizer.zero_grad(set_to_none=True)
        prediction = model(fixed["x_static"], x, fixed["x_baseline"], fixed["edge_index"])
        loss = masked_mse_loss(prediction, y, mask)
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite masked travel-time loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip,
                                       error_if_nonfinite=True)
        optimizer.step()
        loss_total += float(loss.detach()) * y.shape[0]
        count += y.shape[0]
    return loss_total / count


def _checkpoint(path: Path, model, optimizer, epoch: int, validation: dict,
                model_kwargs: dict, cfg: TrainSettings, data: ScenarioDataset,
                splits: dict) -> None:
    torch.save({
        "epoch": epoch, "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(), "validation": validation,
        "model_kwargs": model_kwargs, "splits": splits,
        "target_channels": data.target_channels,
        "static_feature_names": data.input_metadata["static_feature_names"],
        "dynamic_feature_names": data.dynamic_feature_names,
        "config": {k: str(v) if isinstance(v, Path) else v
                   for k, v in asdict(cfg).items()},
        "rng_state": {"python": random.getstate(), "numpy": np.random.get_state(),
                      "torch": torch.get_rng_state(),
                      "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None},
    }, path)


def _resume(cfg: TrainSettings, model, optimizer, model_kwargs: dict,
            data: ScenarioDataset, splits: dict, device: torch.device):
    path = cfg.resume_from.resolve()
    if path.name != "last.pt" or not path.is_file():
        raise FileNotFoundError("Resume requires a previous run/last.pt")
    old = path.parent
    if cfg.run_dir.resolve() == old:
        raise ValueError("Resume into a NEW run directory")
    for name in ("best.pt", "history.csv"):
        if not (old / name).is_file():
            raise FileNotFoundError(old / name)
    last = torch.load(path, map_location=device, weights_only=False)
    best = torch.load(old / "best.pt", map_location="cpu", weights_only=False)
    if (last["model_kwargs"] != model_kwargs or best["model_kwargs"] != model_kwargs
            or last["splits"] != splits or best["splits"] != splits
            or last["target_channels"] != data.target_channels
            or last["static_feature_names"] != data.input_metadata["static_feature_names"]
            or last["dynamic_feature_names"] != data.dynamic_feature_names):
        raise ValueError("Resume checkpoint architecture/data differ")
    for key in ("lr", "weight_decay", "batch_size", "grad_clip", "min_delta",
                "seed", "max_nodes"):
        if last["config"].get(key) != getattr(cfg, key):
            raise ValueError(f"Resume setting changed: {key}")
    with (old / "history.csv").open(newline="") as stream:
        records = list(csv.DictReader(stream))
    if not records or tuple(records[0]) != HISTORY_COLUMNS or int(records[-1]["epoch"]) != last["epoch"]:
        raise ValueError("Resume history differs")
    model.load_state_dict(last["model_state"])
    optimizer.load_state_dict(last["optimizer_state"])
    rng = last["rng_state"]
    random.setstate(rng["python"])
    np.random.set_state(rng["numpy"])
    torch.set_rng_state(rng["torch"].cpu())
    if torch.cuda.is_available() and rng["cuda"] is not None:
        torch.cuda.set_rng_state_all(rng["cuda"])
    shutil.copy2(old / "best.pt", cfg.run_dir / "best.pt")
    shutil.copy2(old / "history.csv", cfg.run_dir / "history.csv")
    if (old / "best_validation.json").exists():
        shutil.copy2(old / "best_validation.json", cfg.run_dir / "best_validation.json")
    return int(last["epoch"]) + 1, float(best["validation"]["loss"]), int(best["epoch"])


def train(cfg: TrainSettings, *, model_overrides: dict | None = None) -> dict:
    if cfg.epochs < 1 or cfg.batch_size < 1 or cfg.patience < 1 or cfg.grad_clip <= 0:
        raise ValueError("Invalid training configuration")
    if cfg.run_dir.exists() and any(cfg.run_dir.iterdir()):
        raise FileExistsError(f"Run directory not empty: {cfg.run_dir}")
    _seed_everything(cfg.seed)
    device = _device(cfg.device)
    splits = load_splits(cfg.split_file)
    tr = _dataset(cfg, splits["train"])
    val = _dataset(cfg, splits["val"])
    test = None if cfg.skip_test else _dataset(cfg, splits["test"])
    fixed = {k: v.to(device) for k, v in _trim_graph(tr.fixed, cfg.max_nodes).items()}
    example = tr[0]
    model_kwargs = dict(MODEL_KWARGS)
    if model_overrides:
        model_kwargs.update(model_overrides)
    model_kwargs.update(static_dim=fixed["x_static"].shape[-1],
                        dynamic_dim=example["x_dynamic"].shape[-1],
                        baseline_dim=fixed["x_baseline"].shape[-1],
                        output_dim=example["y_delta"].shape[-1])
    model = GNNTransformer(**model_kwargs).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    cfg.run_dir.mkdir(parents=True, exist_ok=True)
    _save_json(cfg.run_dir / "settings.json", {
        "config": {k: str(v) if isinstance(v, Path) else v for k, v in asdict(cfg).items()},
        "model_kwargs": model_kwargs,
        "split_sizes": {k: len(v) for k, v in splits.items()},
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "node_count": fixed["x_static"].shape[0], "device": str(device),
        "loss": "masked_mse_seconds_squared",
    })
    print(f"Device={device}, nodes={fixed['x_static'].shape[0]}, "
          f"train/val/test={len(tr)}/{len(val)}/{len(splits['test'])}, "
          f"parameters={sum(p.numel() for p in model.parameters()):,}", flush=True)
    train_loader = _loader(tr, cfg, True)
    val_loader = _loader(val, cfg, False)
    zero_val = _evaluate(model, val_loader, fixed, device, zero_prediction=True)
    _save_json(cfg.run_dir / "no_change_validation.json", zero_val)
    best_loss, best_epoch = float("inf"), 0
    start = 1
    if cfg.resume_from is not None:
        start, best_loss, best_epoch = _resume(
            cfg, model, optimizer, model_kwargs, tr, splits, device)
    stale = start - 1 - best_epoch
    with (cfg.run_dir / "history.csv").open("a" if cfg.resume_from else "w", newline="") as stream:
        writer = csv.writer(stream)
        if cfg.resume_from is None:
            writer.writerow(HISTORY_COLUMNS)
        for epoch in range(start, cfg.epochs + 1):
            started = time.monotonic()
            train_loss = _train_epoch(model, train_loader, fixed, optimizer, cfg, device)
            validation = _evaluate(model, val_loader, fixed, device)
            val_loss = validation["loss"]
            if not np.isfinite(val_loss):
                raise FloatingPointError("Nonfinite validation MSE")
            rmse = validation["metrics"]
            writer.writerow((epoch, train_loss, val_loss,
                             rmse["all"]["rmse"]["road"], rmse["all"]["rmse"]["bus_spec"],
                             rmse["modified"]["rmse"]["road"], time.monotonic() - started))
            stream.flush()
            print(f"Epoch {epoch:03d} train={train_loss:.5g} val={val_loss:.5g} "
                  f"road RMSE={rmse['all']['rmse']['road']:.4g} s", flush=True)
            _checkpoint(cfg.run_dir / "last.pt", model, optimizer, epoch,
                        validation, model_kwargs, cfg, tr, splits)
            if val_loss < best_loss - cfg.min_delta:
                best_loss, best_epoch, stale = val_loss, epoch, 0
                _checkpoint(cfg.run_dir / "best.pt", model, optimizer, epoch,
                            validation, model_kwargs, cfg, tr, splits)
                _save_json(cfg.run_dir / "best_validation.json", validation)
            else:
                stale += 1
            if stale >= cfg.patience:
                print(f"Early stopping at epoch {epoch}", flush=True)
                break
    summary = {"best_epoch": best_epoch, "best_val_loss": best_loss,
               "no_change_val_loss": zero_val["loss"],
               "test_evaluated": not cfg.skip_test,
               "run_dir": str(cfg.run_dir)}
    if test is not None:
        ckpt = torch.load(cfg.run_dir / "best.pt", map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        test_loader = _loader(test, cfg, False)
        measured = _evaluate(model, test_loader, fixed, device)
        zero = _evaluate(model, test_loader, fixed, device, zero_prediction=True)
        _save_json(cfg.run_dir / "test_metrics.json", {"model": measured,
                    "no_change": zero, "best_epoch": best_epoch})
        summary.update(test_loss=measured["loss"], no_change_test_loss=zero["loss"])
        print(f"Test MSE={measured['loss']:.5g}, no-change MSE={zero['loss']:.5g}", flush=True)
        for region in REGIONS:
            print(f"  {region}: RMSE {measured['metrics'][region]['rmse']}", flush=True)
    _save_json(cfg.run_dir / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gnn-dir", type=Path, default=GNN_DIR)
    parser.add_argument("--splits", type=Path, default=SPLITS_FILE)
    parser.add_argument("--out", type=Path, default=RUN_DIR)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default=DEVICE)
    parser.add_argument("--max-nodes", type=int, default=0, help="0 selects full graph")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--patience", type=int, default=PATIENCE)
    parser.add_argument("--skip-test", action="store_true")
    parser.add_argument("--resume-from", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    cfg = TrainSettings(gnn_dir=args.gnn_dir, split_file=args.splits, run_dir=args.out,
                        device=args.device, max_nodes=(None if args.max_nodes == 0 else args.max_nodes),
                        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
                        patience=args.patience, seed=args.seed, resume_from=args.resume_from,
                        skip_test=args.skip_test)
    train(cfg)


if __name__ == "__main__":
    main()

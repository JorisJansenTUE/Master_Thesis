"""Diagnose daily link-flow DELTA predictions from a trained GNN–Transformer.

Runs on the existing full Locarno graph and existing train/val/test split.
Does not fit or update the model. Each prediction is baseline-relative and
measured in link traversals per entire simulation horizon (typically 30 h).

Example, from repository root:

    python -m gnn_surrogate.evaluation.evaluate \
      --gnn-dir /path/to/gnn_training_v3 \
      --checkpoint /path/to/runs/fullgraph_100Ep_resume_001/best.pt \
      --split val --device cpu

Use ``--split test`` for held-out reporting. Add ``--save-predictions`` to
retain per-scenario numpy arrays for further analysis. Output path defaults to
<checkpoint directory>/evaluation_<split>.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from gnn_surrogate.dataset.dataset import ScenarioDataset, load_splits
from gnn_surrogate.models.gnn_transformer import GNNTransformer


# Edit these only if you prefer direct configuration instead of CLI flags.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GNN_DIR = PROJECT_ROOT / "data" / "GNN"
# A delta magnitude must reach this value to evaluate the change's SIGN.
DEFAULT_DIRECTION_THRESHOLDS = {"cars": 5.0, "trucks": 1.0, "bikes": 1.0}
INTERVENTION_TYPES = (
    "bicycle_street", "protected_bike_lane", "dedicated_bus_lane", "destination_only"
)
# Volumes are link traversals over the entire simulated period (usually 30 h).
BASELINE_BINS = ((0.0, 0.0, "zero"),
                 (0.0, 10.0, "(0,10]"),
                 (10.0, 50.0, "(10,50]"),
                 (50.0, 200.0, "(50,200]"),
                 (200.0, 1000.0, "(200,1000]"),
                 (1000.0, float("inf"), ">1000"))
CHANGE_BINS = ((0.0, 0.0, "zero"),
               (0.0, 1.0, "(0,1]"),
               (1.0, 5.0, "(1,5]"),
               (5.0, 20.0, "(5,20]"),
               (20.0, 100.0, "(20,100]"),
               (100.0, float("inf"), ">100"))


def _json_write(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _csv_write(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"No rows for {path.name}")
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _mask_neighborhood(modified: np.ndarray, edge_index: np.ndarray) -> np.ndarray:
    """One hop in either direction, including the directly modified nodes."""
    src, dst = edge_index
    result = modified.copy()
    result[src[modified[dst]]] = True
    result[dst[modified[src]]] = True
    return result


def _bin_mask(values: np.ndarray, lower: float, upper: float) -> np.ndarray:
    if lower == upper == 0:
        return values == 0
    return (values > lower) & (values <= upper)


@dataclass
class RunningError:
    """Metrics over node–scenario positions; never average scenario RMSEs."""
    direction_threshold: float
    count: int = 0
    sum_abs_error: float = 0.0
    sum_sq_error: float = 0.0
    sum_error: float = 0.0
    sum_abs_truth: float = 0.0
    sum_sq_truth: float = 0.0
    sum_truth: float = 0.0
    n_sign: int = 0
    n_sign_correct: int = 0
    n_negative_absolute: int = 0

    def add(self, truth: np.ndarray, prediction: np.ndarray,
            baseline: np.ndarray, mask: np.ndarray) -> None:
        if mask.dtype != np.bool_:
            raise TypeError("Evaluation masks must be Boolean")
        if not np.any(mask):
            return
        y = np.asarray(truth[mask], dtype=np.float64)
        p = np.asarray(prediction[mask], dtype=np.float64)
        b = np.asarray(baseline[mask], dtype=np.float64)
        error = p - y
        self.count += y.size
        self.sum_abs_error += float(np.abs(error).sum())
        self.sum_sq_error += float(np.square(error).sum())
        self.sum_error += float(error.sum())
        self.sum_abs_truth += float(np.abs(y).sum())
        self.sum_sq_truth += float(np.square(y).sum())
        self.sum_truth += float(y.sum())
        self.n_negative_absolute += int(np.count_nonzero((b + p) < -1e-6))
        significant = np.abs(y) >= self.direction_threshold
        self.n_sign += int(significant.sum())
        self.n_sign_correct += int(np.count_nonzero(
            np.sign(y[significant]) == np.sign(p[significant])
        ))

    def result(self) -> dict:
        n = self.count
        if n == 0:
            return dict(count=0, mae=None, rmse=None, no_change_mae=None,
                        no_change_rmse=None, rmse_reduction_pct=None,
                        mae_reduction_pct=None, wape_change_pct=None,
                        direction_accuracy_pct=None, direction_count=0,
                        r2=None, mean_abs_true_delta=None, bias=None,
                        negative_absolute_predictions=0)
        mae = self.sum_abs_error / n
        rmse = math.sqrt(self.sum_sq_error / n)
        no_change_mae = self.sum_abs_truth / n
        no_change_rmse = math.sqrt(self.sum_sq_truth / n)
        sst = self.sum_sq_truth - (self.sum_truth ** 2) / n
        return dict(
            count=n, mae=mae, rmse=rmse,
            no_change_mae=no_change_mae, no_change_rmse=no_change_rmse,
            mae_reduction_pct=100 * (1 - mae / no_change_mae) if no_change_mae > 0 else None,
            rmse_reduction_pct=100 * (1 - rmse / no_change_rmse) if no_change_rmse > 0 else None,
            wape_change_pct=100 * self.sum_abs_error / self.sum_abs_truth if self.sum_abs_truth > 0 else None,
            direction_accuracy_pct=(100 * self.n_sign_correct / self.n_sign if self.n_sign else None),
            direction_count=self.n_sign,
            r2=(1 - self.sum_sq_error / sst) if sst > 1e-10 else None,
            mean_abs_true_delta=no_change_mae,
            bias=self.sum_error / n,
            negative_absolute_predictions=self.n_negative_absolute,
        )


def _strategy(scenario_id: str) -> str:
    return scenario_id.split("_", 1)[0]


def _plot_scatter(path: Path, truth: np.ndarray, pred: np.ndarray,
                  mode: str, region: str) -> None:
    import matplotlib.pyplot as plt
    if truth.size == 0:
        return
    # The visible range is truncated to avoid a small number of extreme deltas
    # obscuring thousands of typical links. Raw predictions remain untruncated.
    pooled = np.concatenate([truth, pred])
    lo, hi = np.percentile(pooled, [0.5, 99.5])
    lo, hi = min(float(lo), 0), max(float(hi), 0)
    if hi <= lo:
        lo, hi = lo - 1, hi + 1
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(truth, pred, s=6, alpha=0.17, rasterized=True)
    ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.3, label="Perfect prediction")
    ax.axhline(0, linewidth=0.6, alpha=0.35)
    ax.axvline(0, linewidth=0.6, alpha=0.35)
    ax.set(xlim=(lo, hi), ylim=(lo, hi), xlabel="MATSim change (vehicles / simulation)",
           ylabel="Predicted change (vehicles / simulation)",
           title=f"{mode.capitalize()} — {region} (n={truth.size:,})")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_spatial(path: Path, nodes: pd.DataFrame, error_rmse: np.ndarray,
                  mode: str) -> None:
    import matplotlib.pyplot as plt
    x = pd.to_numeric(nodes["mid_x"], errors="coerce").to_numpy()
    y = pd.to_numeric(nodes["mid_y"], errors="coerce").to_numpy()
    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(error_rmse)
    if not valid.any():
        return
    fig, ax = plt.subplots(figsize=(9, 7))
    # Color stretch at the 99th percentile; numerical CSV is NOT truncated.
    vmax = max(float(np.percentile(error_rmse[valid], 99)), 1e-6)
    sc = ax.scatter(x[valid], y[valid], c=error_rmse[valid],
                    s=6, alpha=0.85, vmin=0, vmax=vmax, rasterized=True)
    ax.set_title(f"{mode.capitalize()} — per-node test/validation RMSE")
    ax.set_xlabel("Projected x coordinate (m)")
    ax.set_ylabel("Projected y coordinate (m)")
    ax.set_aspect("equal", adjustable="box")
    fig.colorbar(sc, ax=ax, label="RMSE (vehicles / simulation)")
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def _plot_bins(path: Path, rows: list[dict], mode: str, title: str) -> None:
    import matplotlib.pyplot as plt
    selected = [row for row in rows if row["mode"] == mode and row["count"]]
    if not selected:
        return
    fig, ax = plt.subplots(figsize=(8, 4.5))
    xx = np.arange(len(selected))
    ax.plot(xx, [r["rmse"] for r in selected], marker="o", label="Model")
    ax.plot(xx, [r["no_change_rmse"] for r in selected], marker="o", label="No change")
    ax.set_xticks(xx, [r["bin"] for r in selected])
    ax.set_ylabel("RMSE (vehicles / simulation)")
    ax.set_xlabel(title)
    ax.set_title(f"{mode.capitalize()}: prediction error by {title.lower()}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def evaluate(*, gnn_dir: Path, split_file: Path, checkpoint_file: Path,
             output_dir: Path, split: str, device_name: str,
             make_plots: bool, save_predictions: bool,
             direction_thresholds: dict[str, float]) -> dict:
    gnn_dir, split_file, checkpoint_file = map(Path, (gnn_dir, split_file, checkpoint_file))
    if not checkpoint_file.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_file}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Evaluation output directory is not empty: {output_dir}")
    splits = load_splits(split_file)
    dataset = ScenarioDataset(
        graph_dir=gnn_dir / "gnn_static", inputs_dir=gnn_dir / "gnn_inputs",
        dynamic_root=gnn_dir / "dynamic", target_root=gnn_dir / "targets",
        scenario_ids=splits[split], temporal_mode="daily",
    )
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but no GPU is available")
    device = torch.device(device_name)

    # Checkpoints from the user's own training jobs contain numpy/Python RNG
    # objects, so weights_only=False is necessary with PyTorch 2.5.1.
    # Only load checkpoints from a trusted source.
    checkpoint = torch.load(checkpoint_file, map_location="cpu", weights_only=False)
    if checkpoint.get("splits") != splits:
        raise ValueError("Split manifest differs from the one stored in checkpoint")
    if checkpoint.get("target_modes") != dataset.target_modes:
        raise ValueError("Target modes/order differ from the checkpoint")
    if checkpoint.get("input_modes") != dataset.input_modes:
        raise ValueError("Baseline input modes/order differ from checkpoint")
    if checkpoint.get("static_feature_names") != dataset.input_metadata["static_feature_names"]:
        raise ValueError("Static features/order differ from checkpoint")
    if checkpoint.get("dynamic_feature_names") != dataset.dynamic_feature_names:
        raise ValueError("Dynamic features/order differ from checkpoint")
    if checkpoint.get("config", {}).get("max_nodes") not in (None, 0):
        raise ValueError("Checkpoint was trained on a debugging subgraph, not the full graph")

    n = dataset.n_nodes
    kwargs = checkpoint["model_kwargs"]
    if (kwargs["static_dim"] != dataset.fixed["x_static"].shape[-1]
            or kwargs["dynamic_dim"] != len(dataset.dynamic_feature_names)
            or kwargs["baseline_dim"] != dataset.fixed["x_baseline"].shape[-1]
            or kwargs["output_dim"] != len(dataset.target_modes)):
        raise ValueError("Checkpoint feature dimensions do not match dataset")
    model = GNNTransformer(**kwargs)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.to(device).eval()
    fixed = {name: tensor.to(device) for name, tensor in dataset.fixed.items()}
    baseline = dataset.fixed["baseline_raw"].numpy().astype(np.float64)
    edges = dataset.fixed["edge_index"].numpy()
    columns = dataset.dynamic_feature_names
    mode_names = dataset.target_modes
    scales = torch.as_tensor(checkpoint["mode_scales"], dtype=torch.float64).cpu().numpy()
    if scales.shape != (len(mode_names),) or not np.isfinite(scales).all() or (scales <= 0).any():
        raise ValueError("Invalid train-only mode scales in checkpoint")

    nodes = pd.read_csv(gnn_dir / "gnn_static" / "dual_nodes.csv.gz",
                        low_memory=False, dtype={"segment_id": str, "from_node": str, "to_node": str})
    if "node_id" not in nodes or not np.array_equal(nodes["node_id"].to_numpy(), np.arange(n)):
        raise ValueError("dual_nodes.csv.gz must be ordered by fixed node_id 0..N-1")
    if make_plots:
        try:
            import matplotlib  # noqa: F401
            matplotlib.use("Agg")
        except ImportError as exc:
            raise ImportError("Plots need matplotlib: pip install matplotlib==3.9.4; "
                              "or pass --no-plots") from exc
    output_dir.mkdir(parents=True, exist_ok=True)

    accum: dict[tuple[str, str, str], RunningError] = {}
    # Dedicated per-key factory keeps mode-specific significance thresholds.
    def get(group: str, name: str, mode: str) -> RunningError:
        key = (group, name, mode)
        if key not in accum:
            accum[key] = RunningError(direction_thresholds[mode])
        return accum[key]

    scenario_rows: list[dict] = []
    samples: dict[tuple[str, str], list[tuple[np.ndarray, np.ndarray]]] = defaultdict(list)
    per_node_sq = np.zeros((n, len(mode_names)), dtype=np.float64)
    per_node_abs = np.zeros((n, len(mode_names)), dtype=np.float64)
    per_node_truth_abs = np.zeros((n, len(mode_names)), dtype=np.float64)
    for idx in range(len(dataset)):
        item = dataset[idx]
        sid = item["scenario_id"]
        dynamic = item["x_dynamic"].numpy()
        truth = item["y_delta"].numpy().astype(np.float64)
        with torch.inference_mode():
            predicted = model(fixed["x_static"], item["x_dynamic"].to(device),
                              fixed["x_baseline"], fixed["edge_index"])
        predicted = predicted.detach().cpu().numpy().astype(np.float64)
        if predicted.shape != truth.shape or not np.isfinite(predicted).all():
            raise ValueError(f"Invalid predictions for {sid}")
        error = predicted - truth
        per_node_sq += error ** 2
        per_node_abs += np.abs(error)
        per_node_truth_abs += np.abs(truth)

        modified = np.any(dynamic != 0, axis=1)
        nearby = _mask_neighborhood(modified, edges)
        masks = {
            "all": np.ones(n, dtype=bool),
            "modified": modified,
            "nearby": nearby,
            "unchanged": ~modified,
            "unchanged_nearby": (~modified) & nearby,
            "far_unchanged": ~nearby,
        }
        group = _strategy(sid)
        for mode_i, mode in enumerate(mode_names):
            true = truth[:, mode_i]
            pred = predicted[:, mode_i]
            base = baseline[:, mode_i]
            for region, mask in masks.items():
                get("region", region, mode).add(true, pred, base, mask)
                if region in ("all", "modified", "nearby", "far_unchanged"):
                    val = RunningError(direction_thresholds[mode])
                    val.add(true, pred, base, mask)
                    scenario_rows.append(dict(scenario_id=sid, strategy=group,
                                              region=region, mode=mode, **val.result(),
                                              actual_total_delta=float(true[mask].sum()),
                                              predicted_total_delta=float(pred[mask].sum()),
                                              actual_absolute_delta=float(np.abs(true[mask]).sum()),
                                              predicted_absolute_delta=float(np.abs(pred[mask]).sum())))
                if region in ("modified", "nearby") and make_plots and mask.any():
                    samples[(mode, region)].append((true[mask].astype(np.float32),
                                                     pred[mask].astype(np.float32)))
            # Compare strategies on every street and on their direct interventions.
            for region in ("all", "modified"):
                get("strategy", f"{group}|{region}", mode).add(
                    true, pred, base, masks[region]
                )
            for low, high, name in BASELINE_BINS:
                get("baseline_bin", name, mode).add(
                    true, pred, base, _bin_mask(base, low, high)
                )
            for low, high, name in CHANGE_BINS:
                get("change_bin", name, mode).add(
                    true, pred, base, _bin_mask(np.abs(true), low, high)
                )
            for int_type in INTERVENTION_TYPES:
                if int_type in columns:
                    selected = dynamic[:, columns.index(int_type)] > 0.5
                    get("intervention", int_type, mode).add(true, pred, base, selected)

        if save_predictions:
            folder = output_dir / "predictions"
            folder.mkdir(exist_ok=True)
            np.savez_compressed(folder / f"{sid}.npz",
                                node_id=np.arange(n, dtype=np.int32),
                                actual_delta=truth.astype(np.float32),
                                predicted_delta=predicted.astype(np.float32),
                                modified_mask=modified, nearby_mask=nearby)
        if (idx + 1) % 10 == 0 or idx + 1 == len(dataset):
            print(f"Evaluated {idx + 1}/{len(dataset)} scenarios", flush=True)

    buckets: dict[str, list[dict]] = defaultdict(list)
    for (kind, label, mode), stats in sorted(accum.items()):
        if kind == "strategy":
            strategy, region = label.split("|", 1)
            buckets[kind].append(dict(strategy=strategy, region=region,
                                      mode=mode, **stats.result()))
        else:
            buckets[kind].append({("bin" if kind.endswith("bin") else
                                   "intervention" if kind == "intervention" else
                                   "region"): label, "mode": mode, **stats.result()})

    for key, filename in (
        ("region", "region_metrics.csv"),
        ("baseline_bin", "by_baseline_volume.csv"),
        ("change_bin", "by_change_magnitude.csv"),
        ("intervention", "by_intervention_type.csv"),
        ("strategy", "by_strategy.csv"),
    ):
        if buckets.get(key):
            _csv_write(output_dir / filename, buckets[key])
    _csv_write(output_dir / "scenario_metrics.csv", scenario_rows)

    node_rows = nodes[["node_id", "segment_id", "from_node", "to_node"]].copy()
    for coord in ("mid_x", "mid_y"):
        if coord in nodes:
            node_rows[coord] = nodes[coord]
    for j, mode in enumerate(mode_names):
        node_rows[f"baseline_{mode}"] = baseline[:, j]
        node_rows[f"rmse_{mode}"] = np.sqrt(per_node_sq[:, j] / len(dataset))
        node_rows[f"mae_{mode}"] = per_node_abs[:, j] / len(dataset)
        node_rows[f"mean_abs_true_delta_{mode}"] = per_node_truth_abs[:, j] / len(dataset)
    node_rows.to_csv(output_dir / "node_metrics.csv.gz", compression="gzip", index=False)

    if make_plots:
        plotdir = output_dir / "plots"
        plotdir.mkdir(exist_ok=True)
        for mode in mode_names:
            for region in ("modified", "nearby"):
                pair_list = samples[(mode, region)]
                if pair_list:
                    actual = np.concatenate([pair[0] for pair in pair_list])
                    pred = np.concatenate([pair[1] for pair in pair_list])
                    _plot_scatter(plotdir / f"scatter_{mode}_{region}.png",
                                  actual, pred, mode, region)
            if {"mid_x", "mid_y"}.issubset(nodes.columns):
                _plot_spatial(plotdir / f"spatial_rmse_{mode}.png", nodes,
                              np.sqrt(per_node_sq[:, mode_names.index(mode)] / len(dataset)), mode)
            _plot_bins(plotdir / f"baseline_error_{mode}.png",
                       buckets["baseline_bin"], mode, "Baseline traffic volume")
            _plot_bins(plotdir / f"delta_error_{mode}.png",
                       buckets["change_bin"], mode, "Actual flow-change magnitude")

    all_stats = [get("region", "all", mode) for mode in mode_names]
    model_balanced_mse = float(np.mean([
        s.sum_sq_error / s.count / (float(scale) ** 2)
        for s, scale in zip(all_stats, scales)
    ]))
    baseline_balanced_mse = float(np.mean([
        s.sum_sq_truth / s.count / (float(scale) ** 2)
        for s, scale in zip(all_stats, scales)
    ]))
    summary = {
        "model_balanced_mse": model_balanced_mse,
        "no_change_balanced_mse": baseline_balanced_mse,
        "balanced_mse_reduction_pct": (
            100 * (1 - model_balanced_mse / baseline_balanced_mse)
            if baseline_balanced_mse > 0 else None
        ),
        "mode_scales": dict(zip(mode_names, [float(x) for x in scales])),
        "checkpoint": str(checkpoint_file.resolve()),
        "best_epoch": int(checkpoint["epoch"]),
        "split": split,
        "scenario_count": len(dataset),
        "node_count": n,
        "graph_edges": int(edges.shape[1]),
        "target_modes": mode_names,
        "simulation_hours": dataset.hours,
        "device": str(device),
        "direction_thresholds": {mode: direction_thresholds[mode] for mode in mode_names},
        "region_metrics": {region: {
            mode: get("region", region, mode).result() for mode in mode_names
        } for region in ("all", "modified", "nearby", "unchanged", "unchanged_nearby", "far_unchanged")},
        "notes": [
            "Predictions and errors are changes in link traversals across all simulation hours (30h if configured so).",
            "NearBy includes directly modified nodes; unchanged includes adjacent nodes.",
            "Mode metrics pool node-scenario observations, not unweighted scenario RMSEs.",
            "WAPE is sum(abs(predicted_delta - actual_delta)) / sum(abs(actual_delta)); no-change WAPE=100% when denominator>0.",
            "Direction accuracy is restricted to |actual_delta| >= the configured mode-specific threshold.",
            "Negative absolute predictions count cases with baseline_raw + predicted_delta < 0; values are NOT clipped.",
        ],
    }
    _json_write(output_dir / "summary.json", summary)
    print("Evaluation complete:", output_dir, flush=True)
    for mode in mode_names:
        main = summary["region_metrics"]["modified"][mode]
        print(f"Modified {mode}: MAE={main['mae']:.3f}, RMSE={main['rmse']:.3f}, "
              f"no-change RMSE={main['no_change_rmse']:.3f}, "
              f"sign acc={main['direction_accuracy_pct']}", flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--gnn-dir", type=Path, default=DEFAULT_GNN_DIR,
                        help="GNN dataset root containing gnn_static/gnn_inputs/dynamic/targets")
    parser.add_argument("--splits", type=Path, default=None,
                        help="Default: <gnn-dir>/splits.json")
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="Trained best.pt from a trusted training run")
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument("--cars-sign-threshold", type=float, default=5.0)
    parser.add_argument("--trucks-sign-threshold", type=float, default=1.0)
    parser.add_argument("--bikes-sign-threshold", type=float, default=1.0)
    args = parser.parse_args()
    thresholds = {"cars": args.cars_sign_threshold,
                  "trucks": args.trucks_sign_threshold,
                  "bikes": args.bikes_sign_threshold}
    if any(not math.isfinite(v) or v <= 0 for v in thresholds.values()):
        parser.error("Direction-significance thresholds must be finite and >0")
    output_dir = args.out or args.checkpoint.parent / f"evaluation_{args.split}"
    evaluate(gnn_dir=args.gnn_dir, split_file=args.splits or args.gnn_dir / "splits.json",
             checkpoint_file=args.checkpoint, output_dir=output_dir,
             split=args.split, device_name=args.device,
             make_plots=not args.no_plots,
             save_predictions=args.save_predictions,
             direction_thresholds=thresholds)


if __name__ == "__main__":
    main()

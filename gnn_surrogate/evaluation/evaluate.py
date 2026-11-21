"""Evaluate the general-road congestion-delay surrogate in travel-time seconds.

Targets contain congestion-delay CHANGES, but the main evaluation compares
reconstructed TOTAL travel-time changes with MATSim. All statistics and plots
exclude ineligible links. Two benchmarks are reported:
  - no_change: total travel time remains at baseline (delta TT = 0)
  - unchanged_delay: free-flow changes are known, delay stays at baseline
The latter is the natural zero-output baseline for the delay surrogate.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from gnn_surrogate.dataset.dataset import ScenarioDataset, load_splits
from gnn_surrogate.evaluation.delay_reconstruction import reconstruct_road_travel_times
from gnn_surrogate.models.gnn_transformer import GNNTransformer
from preprocessing.utils import PROJECT_ROOT

CHANNEL = "road"
MAGNITUDE_EDGES_S = np.array([0.0, 1.0, 5.0, 20.0, 100.0])
MAGNITUDE_LABELS = ("0", "0–1", "1–5", "5–20", "20–100", ">100")
BASELINE_DELAY_EDGES_S = np.array([0.0, 5.0, 15.0, 60.0])
BASELINE_DELAY_LABELS = ("≤0", "0–5", "5–15", "15–60", ">60")
SCATTER_MAX_POINTS = 50_000
TARGET_TYPE = "road_congestion_delay_change"


def _empty_stats() -> dict:
    return {"n": 0, "abs": 0.0, "sq": 0.0, "sq_no_change": 0.0,
            "sq_unchanged_delay": 0.0, "sum_total": 0.0, "sum_total2": 0.0,
            "sum_delay": 0.0, "sum_delay2": 0.0, "sum_absolute": 0.0,
            "sum_absolute2": 0.0, "bias": 0.0, "below_ff": 0}


def _metrics(s: dict) -> dict:
    n = s["n"]
    if not n:
        return {"count": 0, "mae_s": None, "rmse_s": None, "r2_delta": None,
                "r2_delay_delta": None, "r2_absolute_tt": None,
                "no_change_rmse_s": None, "unchanged_delay_rmse_s": None,
                "mean_error_s": None, "predicted_below_freeflow": 0}

    def r2(sum_y: float, sum_y2: float):
        sst = sum_y2 - sum_y ** 2 / n
        return 1.0 - s["sq"] / sst if sst > 1e-12 else None

    return {
        "count": n,
        "mae_s": s["abs"] / n,
        "rmse_s": np.sqrt(s["sq"] / n).item(),
        "r2_delta": r2(s["sum_total"], s["sum_total2"]),
        "r2_delay_delta": r2(s["sum_delay"], s["sum_delay2"]),
        "r2_absolute_tt": r2(s["sum_absolute"], s["sum_absolute2"]),
        "no_change_rmse_s": np.sqrt(s["sq_no_change"] / n).item(),
        "unchanged_delay_rmse_s": np.sqrt(s["sq_unchanged_delay"] / n).item(),
        "mean_error_s": s["bias"] / n,
        "predicted_below_freeflow": s["below_ff"],
    }


def _plot_history(path: Path, out: Path, best_epoch: int, plt) -> None:
    if not path.is_file():
        print(f"No loss curve: {path} not found", flush=True)
        return
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows or not {"epoch", "train_mse", "val_mse"}.issubset(rows[0]):
        print(f"No loss curve: {path} has no epoch/train_mse/val_mse columns", flush=True)
        return
    epoch = np.array([int(r["epoch"]) for r in rows])
    train = np.array([float(r["train_mse"]) for r in rows])
    val = np.array([float(r["val_mse"]) for r in rows])
    fig, ax = plt.subplots(figsize=(9, 5.1))
    ax.plot(epoch, train, lw=2, label="Training")
    ax.plot(epoch, val, lw=2, label="Validation")
    if best_epoch in epoch:
        ax.axvline(best_epoch, ls="--", color="0.4", label=f"Best epoch {best_epoch}")
    ax.set(xlabel="Epoch", ylabel="Masked MSE of Δ congestion delay (s²)",
           title="Training and validation loss")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "loss_train_validation.png", dpi=180)
    plt.close(fig)


def _plot_scatter(scatter: list, metrics: dict, out: Path, plt) -> None:
    if not scatter:
        return
    actual = np.concatenate([a for a, _ in scatter])
    pred = np.concatenate([b for _, b in scatter])
    fig, ax = plt.subplots(figsize=(7.2, 6.4))
    ax.scatter(actual, pred, s=5, alpha=0.12, linewidths=0, rasterized=True)
    lo, hi = min(actual.min(), pred.min(), 0), max(actual.max(), pred.max(), 0)
    if hi <= lo:
        hi = lo + 1
    ax.plot([lo, hi], [lo, hi], ls="--", color="0.2", label="Perfect prediction")
    ax.axhline(0, color="0.6", lw=0.7)
    ax.axvline(0, color="0.6", lw=0.7)
    ax.set(xlabel="Actual total travel-time change (s)",
           ylabel="Predicted total travel-time change (s)",
           title="Road links — reconstructed total travel-time changes",
           xlim=(lo, hi), ylim=(lo, hi))
    ax.set_aspect("equal", adjustable="box")
    details = f"RMSE: {metrics['rmse_s']:.2f} s"
    if metrics["r2_delta"] is not None:
        details += f"\nR²: {metrics['r2_delta']:.3f}"
    ax.text(.03, .97, details, va="top", transform=ax.transAxes,
            bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"})
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "scatter_road.png", dpi=180)
    plt.close(fig)


def _plot_bins(bin_data: dict, labels: tuple[str, ...], name: str,
               title: str, xlabel: str, out: Path, plt) -> None:
    n = bin_data["n"]
    keep = n > 0
    if not keep.any():
        return
    x = np.arange(len(labels))
    series = [
        ("GNN–Transformer", "sq"),
        ("Unchanged congestion delay", "sq_unchanged_delay"),
        ("No change in total travel time", "sq_no_change"),
    ]
    fig, ax = plt.subplots(figsize=(11.0, 5.4))
    width = 0.25
    for j, (label, key) in enumerate(series):
        rmse = np.zeros(n.shape, dtype=float)
        rmse[keep] = np.sqrt(bin_data[key][keep] / n[keep])
        ax.bar(x[keep] + (j - 1) * width, rmse[keep], width, label=label)
    ax.set_xticks(x, labels)
    ax.set(xlabel=xlabel, ylabel="RMSE of total travel-time change (s)", title=title)
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)
    ax.legend(fontsize=9)
    for i in x[keep]:
        height = max(np.sqrt(bin_data[key][i] / n[i]) for _, key in series)
        ax.annotate(f"n={n[i]:,}", (i, height), xytext=(0, 5),
                    textcoords="offset points", ha="center", fontsize=8)
    ax.margins(y=.13)
    fig.tight_layout()
    fig.savefig(out / name, dpi=180)
    plt.close(fig)


def _plot_scenarios(rows: list[dict], out: Path, plt) -> None:
    valid = [r for r in rows if r["road_rmse_s"] is not None]
    if not valid:
        return
    data = np.array([[r["road_rmse_s"], r["road_no_change_rmse_s"],
                      r["road_unchanged_delay_rmse_s"]] for r in valid], dtype=float)
    data = data[np.argsort(data[:, 1])]
    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    x = np.arange(1, len(data) + 1)
    ax.plot(x, data[:, 0], lw=1.1, label="GNN–Transformer")
    ax.plot(x, data[:, 1], lw=1.1, label="No total TT change")
    ax.plot(x, data[:, 2], lw=1.1, label="Unchanged congestion delay")
    ax.set(xlabel="Scenario, sorted by no-total-change RMSE",
           ylabel="RMSE of total travel-time change (s)",
           title="Travel-time prediction error by scenario")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "rmse_by_scenario_road.png", dpi=180)
    plt.close(fig)


def _save_plots(out: Path, checkpoint: Path, best_epoch: int, scatter: list,
                magnitude: dict, congestion: dict, rows: list[dict], metrics: dict,
                history: Path | None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _plot_history(history or checkpoint.parent / "history.csv", out, best_epoch, plt)
    if metrics["count"]:
        _plot_scatter(scatter, metrics, out, plt)
        _plot_bins(magnitude, MAGNITUDE_LABELS, "rmse_by_change_road.png",
                   "Error by actual total travel-time change",
                   "Absolute actual total travel-time change (s)", out, plt)
        _plot_bins(congestion, BASELINE_DELAY_LABELS,
                   "rmse_by_baseline_congestion_road.png",
                   "Error by baseline congestion delay",
                   "Baseline congestion delay (s)", out, plt)
        _plot_scenarios(rows, out, plt)


def evaluate(gnn_dir: Path, splits_file: Path, checkpoint: Path, output: Path,
             split: str = "val", device_name: str = "cpu", *, plots: bool = True,
             save_predictions: bool = False, history: Path | None = None) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    ids = load_splits(splits_file)[split]
    data = ScenarioDataset(gnn_dir / "gnn_static", gnn_dir / "gnn_inputs",
                           gnn_dir / "dynamic", gnn_dir / "targets", ids)
    # A travel-time-change dataset with the same shape would otherwise give
    # misleading metrics; refuse to mix experiments.
    for sid in ids:
        meta = json.loads((gnn_dir / "targets" / sid / "target_metadata.json").read_text())
        if meta.get("target_type") != TARGET_TYPE:
            raise ValueError(f"{sid}: expected target_type={TARGET_TYPE}, found {meta.get('target_type')!r}")
    with np.load(gnn_dir / "gnn_static" / "baseline_tt.npz", allow_pickle=False) as cache:
        baseline_ff = cache["baseline_freeflow_s"]
        if int(cache["hour"]) != data.hour:
            raise ValueError("Baseline free-flow reference hour does not match dataset")
    fixed = data.fixed
    baseline_tt = fixed["baseline_tt_s"].numpy()
    device = torch.device("cuda" if device_name == "auto" and torch.cuda.is_available()
                          else "cpu" if device_name == "auto" else device_name)
    fixed_device = {k: v.to(device) for k, v in fixed.items()}
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    if (ckpt["static_feature_names"] != data.static_feature_names
            or ckpt["dynamic_feature_names"] != data.dynamic_feature_names
            or ckpt["target_channels"] != data.target_channels):
        raise ValueError("Checkpoint feature/output names do not match dataset")
    model = GNNTransformer(**ckpt["model_kwargs"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    stats = _empty_stats()
    magnitudes = {k: np.zeros(6, dtype=np.float64 if k != "n" else np.int64)
                  for k in ("n", "sq", "sq_no_change", "sq_unchanged_delay")}
    congestion = {k: np.zeros(5, dtype=np.float64 if k != "n" else np.int64)
                  for k in ("n", "sq", "sq_no_change", "sq_unchanged_delay")}
    scatter = []
    rows = []
    rng = np.random.default_rng(42)
    max_scatter_scenario = max(1, SCATTER_MAX_POINTS // len(data))
    prediction_dir = output / "predictions"
    if save_predictions:
        prediction_dir.mkdir(exist_ok=True)

    with torch.inference_mode():
        for item in data:
            dynamic = item["x_dynamic"]
            truth = item["y_delta"]
            valid = item["valid_mask"].numpy()
            pred = model(fixed_device["x_static"], dynamic.to(device),
                         fixed_device["x_baseline"], fixed_device["edge_index"])
            if pred.shape != truth.shape or truth.shape != valid.shape:
                raise ValueError(f"Shape mismatch in {item['scenario_id']}")
            reconstruction = reconstruct_road_travel_times(
                baseline_tt, baseline_ff, dynamic.numpy(), data.dynamic_feature_names,
                pred.cpu().numpy(), valid, actual_delay_delta_s=truth.numpy())
            predicted = reconstruction["predicted_total_delta_s"][valid]
            actual = reconstruction["actual_total_delta_s"][valid]
            predicted_delay = pred.cpu().numpy()[valid].astype(np.float64)
            actual_delay = truth.numpy()[valid].astype(np.float64)
            ff_change = reconstruction["freeflow_change_s"][valid]
            abs_tt = reconstruction["actual_tt_s"][valid]
            predicted_tt = reconstruction["predicted_tt_s"][valid]
            scenario_ff = reconstruction["scenario_freeflow_s"][valid]
            n = actual.size
            row = {"scenario": item["scenario_id"], "road_n": n,
                   "road_rmse_s": None, "road_no_change_rmse_s": None,
                   "road_unchanged_delay_rmse_s": None}
            if n:
                error = predicted - actual
                sq = error ** 2
                stats["n"] += n
                stats["abs"] += float(np.abs(error).sum())
                stats["sq"] += float(sq.sum())
                stats["sq_no_change"] += float(np.square(actual).sum())
                stats["sq_unchanged_delay"] += float(np.square(actual_delay).sum())
                stats["sum_total"] += float(actual.sum())
                stats["sum_total2"] += float(np.square(actual).sum())
                stats["sum_delay"] += float(actual_delay.sum())
                stats["sum_delay2"] += float(np.square(actual_delay).sum())
                stats["sum_absolute"] += float(abs_tt.sum())
                stats["sum_absolute2"] += float(np.square(abs_tt).sum())
                stats["bias"] += float(error.sum())
                below_ff = int(np.count_nonzero(predicted_tt < scenario_ff - 0.01))
                stats["below_ff"] += below_ff
                row.update({
                    "road_rmse_s": float(np.sqrt(sq.mean())),
                    "road_no_change_rmse_s": float(np.sqrt(np.square(actual).mean())),
                    "road_unchanged_delay_rmse_s": float(np.sqrt(np.square(actual_delay).mean())),
                    "road_mae_s": float(np.abs(error).mean()),
                    "road_mean_error_s": float(error.mean()),
                    "road_predicted_below_freeflow": below_ff,
                })
                if plots:
                    b = np.searchsorted(MAGNITUDE_EDGES_S, np.abs(actual), side="left")
                    c = np.searchsorted(BASELINE_DELAY_EDGES_S,
                                        np.maximum(0.0, baseline_tt[valid] - baseline_ff[valid]), side="left")
                    for bucket, indices in ((magnitudes, b), (congestion, c)):
                        bucket["n"] += np.bincount(indices, minlength=len(bucket["n"]))
                        bucket["sq"] += np.bincount(indices, weights=sq, minlength=len(bucket["n"]))
                        bucket["sq_no_change"] += np.bincount(indices, weights=np.square(actual),
                                                               minlength=len(bucket["n"]))
                        bucket["sq_unchanged_delay"] += np.bincount(indices, weights=np.square(actual_delay),
                                                                     minlength=len(bucket["n"]))
                    chosen = rng.choice(n, size=min(n, max_scatter_scenario), replace=False)
                    scatter.append((actual[chosen], predicted[chosen]))
            if save_predictions:
                np.savez_compressed(prediction_dir / f"{item['scenario_id']}.npz",
                                    predicted_delay_change_s=pred.cpu().numpy().astype(np.float32),
                                    actual_delay_change_s=truth.numpy().astype(np.float32),
                                    freeflow_change_s=reconstruction["freeflow_change_s"].astype(np.float32),
                                    predicted_total_tt_change_s=reconstruction["predicted_total_delta_s"].astype(np.float32),
                                    actual_total_tt_change_s=reconstruction["actual_total_delta_s"].astype(np.float32),
                                    predicted_tt_s=reconstruction["predicted_tt_s"].astype(np.float32),
                                    actual_tt_s=reconstruction["actual_tt_s"].astype(np.float32),
                                    valid_mask=valid)
            rows.append(row)
    result = _metrics(stats)
    summary = {
        "split": split,
        "scenarios": len(data),
        "hour": data.hour,
        "best_epoch": int(ckpt["epoch"]),
        "target_type": TARGET_TYPE,
        "metrics": {CHANNEL: result},
        "units": "seconds; main metrics use reconstructed TOTAL travel-time changes",
        "benchmarks": {
            "no_change": "T_scenario_pred = T_baseline",
            "unchanged_delay": "T_scenario_pred = T_baseline + (FF_scenario - FF_baseline)",
        },
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    with (output / "scenario_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    if plots:
        _save_plots(output, checkpoint, int(ckpt["epoch"]), scatter,
                    magnitudes, congestion, rows, result, history)
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gnn-dir", type=Path, default=PROJECT_ROOT / "data" / "GNN")
    parser.add_argument("--splits", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cpu")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument("--history", type=Path, default=None)
    args = parser.parse_args()
    evaluate(args.gnn_dir, args.splits or args.gnn_dir / "splits.json",
             args.checkpoint, args.out, args.split, args.device,
             plots=not args.no_plots, save_predictions=args.save_predictions,
             history=args.history)


if __name__ == "__main__":
    main()

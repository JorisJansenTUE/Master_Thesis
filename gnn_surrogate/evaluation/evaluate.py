"""Evaluate one-hour general-road link travel-time DELTAS with eligibility masks.

Produces summary.json and scenario_metrics.csv as before. By default, also
creates diagnostic PNG plots including train/validation MSE from history.csv.
Only links with valid_mask=True contribute to every reported error/plot.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from gnn_surrogate.dataset.dataset import ScenarioDataset, load_splits
from gnn_surrogate.models.gnn_transformer import GNNTransformer
from preprocessing.utils import PROJECT_ROOT

CHANNELS = ("road",)
# Zero, (0,1], (1,5], (5,20], (20,100], >100 seconds.
MAGNITUDE_EDGES_S = np.array([0.0, 1.0, 5.0, 20.0, 100.0])
MAGNITUDE_LABELS = ("0", "0–1", "1–5", "5–20", "20–100", ">100")
SCATTER_MAX_POINTS = 50_000


def _empty_stats() -> dict:
    return {"n": 0, "abs": 0.0, "sq": 0.0, "sq_zero": 0.0,
            "sum": 0.0, "sum2": 0.0, "bias": 0.0}


def _plot_training_history(path: Path, out: Path, best_epoch: int, plt) -> bool:
    if not path.is_file():
        print(f"Training curve not generated: missing {path}", flush=True)
        return False
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows or any(k not in rows[0] for k in ("epoch", "train_mse", "val_mse")):
        print(f"Training curve not generated: invalid or empty {path}", flush=True)
        return False
    x = np.array([int(r["epoch"]) for r in rows], dtype=int)
    train = np.array([float(r["train_mse"]) for r in rows], dtype=float)
    val = np.array([float(r["val_mse"]) for r in rows], dtype=float)
    fig, ax = plt.subplots(figsize=(9, 5.1))
    ax.plot(x, train, lw=2.0, label="Training masked MSE")
    ax.plot(x, val, lw=2.0, label="Validation masked MSE")
    if best_epoch in x:
        ax.axvline(best_epoch, ls="--", lw=1.2, color="0.4",
                   label=f"Best checkpoint (epoch {best_epoch})")
    ax.set(xlabel="Training epoch", ylabel="Masked MSE (seconds²)",
           title="Training and validation loss")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "loss_train_validation.png", dpi=180)
    plt.close(fig)
    return True


def _plot_scatter(channel: str, data: list[tuple[np.ndarray, np.ndarray]],
                  out: Path, metrics: dict, plt) -> None:
    if not data:
        return
    true = np.concatenate([p[0] for p in data])
    pred = np.concatenate([p[1] for p in data])
    if not true.size:
        return
    fig, ax = plt.subplots(figsize=(7.0, 6.3))
    ax.scatter(true, pred, s=5, alpha=0.11, rasterized=True, linewidths=0)
    lo = min(float(true.min()), float(pred.min()), 0.0)
    hi = max(float(true.max()), float(pred.max()), 0.0)
    if hi <= lo:
        lo, hi = lo - 1, hi + 1
    ax.plot([lo, hi], [lo, hi], color="0.2", ls="--", lw=1.3,
            label="Perfect prediction")
    ax.axhline(0.0, color="0.6", lw=0.7)
    ax.axvline(0.0, color="0.6", lw=0.7)
    ax.set(xlabel="Actual travel-time change (s)",
           ylabel="Predicted travel-time change (s)",
           title=f"Predicted vs actual Δ travel time — {channel.replace('_', ' ')}")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.text(0.03, 0.97,
            f"Masked RMSE: {metrics['rmse_s']:.2f} s\nR²: {metrics['r2_delta']:.3f}" if metrics['r2_delta'] is not None
            else f"Masked RMSE: {metrics['rmse_s']:.2f} s",
            transform=ax.transAxes, va="top", ha="left",
            bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"})
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out / f"scatter_{channel}.png", dpi=180)
    plt.close(fig)


def _plot_magnitude(channel: str, bins: dict, out: Path, plt) -> None:
    count, sq, zero_sq = bins["n"], bins["sq"], bins["zero_sq"]
    keep = count > 0
    if not keep.any():
        return
    model = np.full(count.shape, np.nan, dtype=float)
    no_change = np.full(count.shape, np.nan, dtype=float)
    model[keep] = np.sqrt(sq[keep] / count[keep])
    no_change[keep] = np.sqrt(zero_sq[keep] / count[keep])
    x = np.arange(len(count))
    fig, ax = plt.subplots(figsize=(10, 5.3))
    width = 0.37
    ax.bar(x[keep] - width / 2, model[keep], width, label="GNN–Transformer")
    ax.bar(x[keep] + width / 2, no_change[keep], width, label="No-change baseline")
    ax.set_xticks(x, MAGNITUDE_LABELS)
    ax.set(xlabel="Absolute actual travel-time change |ΔTT| (s)",
           ylabel="RMSE of predicted ΔTT (s)",
           title=f"Error by actual change magnitude — {channel.replace('_', ' ')}")
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)
    ax.legend()
    # Counts are useful because bus-specific bins can be sparsely populated.
    for i in x[keep]:
        height = max(model[i], no_change[i])
        ax.annotate(f"n={count[i]:,}", (i, height), xytext=(0, 5),
                    textcoords="offset points", ha="center", va="bottom", fontsize=8)
    ax.margins(y=0.13)
    fig.tight_layout()
    fig.savefig(out / f"rmse_by_change_{channel}.png", dpi=180)
    plt.close(fig)


def _plot_scenarios(channel: str, scenario_rows: list[dict], out: Path, plt) -> None:
    pairs = [(float(r[f"{channel}_rmse_s"]), float(r[f"{channel}_no_change_rmse_s"]))
             for r in scenario_rows if r[f"{channel}_rmse_s"] is not None]
    if not pairs:
        return
    arr = np.array(pairs, dtype=float)
    # Sort by baseline RMSE to reveal whether the model improves on high-impact scenarios.
    arr = arr[np.argsort(arr[:, 1])]
    fig, ax = plt.subplots(figsize=(9, 5.1))
    x = np.arange(1, len(arr) + 1)
    ax.plot(x, arr[:, 0], lw=1.2, label="GNN–Transformer")
    ax.plot(x, arr[:, 1], lw=1.2, label="No-change baseline")
    ax.set(xlabel="Scenario (sorted by no-change RMSE)",
           ylabel="RMSE of predicted ΔTT (s)",
           title=f"Per-scenario RMSE — {channel.replace('_', ' ')}")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / f"rmse_by_scenario_{channel}.png", dpi=180)
    plt.close(fig)


def _save_plots(out: Path, checkpoint: Path, best_epoch: int, scatter: dict,
                bins: dict, rows: list[dict], results: dict, history: Path | None) -> None:
    import matplotlib
    matplotlib.use("Agg")  # Safe in noninteractive SLURM jobs.
    import matplotlib.pyplot as plt

    history_file = history or checkpoint.parent / "history.csv"
    _plot_training_history(history_file, out, best_epoch, plt)
    for ch in CHANNELS:
        if results[ch]["count"]:
            _plot_scatter(ch, scatter[ch], out, results[ch], plt)
            _plot_magnitude(ch, bins[ch], out, plt)
            _plot_scenarios(ch, rows, out, plt)


def evaluate(gnn_dir: Path, splits_file: Path, checkpoint: Path,
             output: Path, split: str = "val", device_name: str = "cpu",
             *, plots: bool = True, save_predictions: bool = False,
             history: Path | None = None) -> dict:
    # Safe rerun: overwrite our own summary/plot files, leave other files untouched.
    output.mkdir(parents=True, exist_ok=True)
    ids = load_splits(splits_file)[split]
    data = ScenarioDataset(gnn_dir / "gnn_static", gnn_dir / "gnn_inputs",
                           gnn_dir / "dynamic", gnn_dir / "targets", ids)
    device = torch.device("cuda" if device_name == "auto" and torch.cuda.is_available()
                          else "cpu" if device_name == "auto" else device_name)
    fixed = {k: v.to(device) for k, v in data.fixed.items()}
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    if ckpt["static_feature_names"] != data.input_metadata["static_feature_names"]:
        raise ValueError("Checkpoint static feature names differ")
    if ckpt["target_channels"] != list(CHANNELS):
        raise ValueError("Checkpoint travel-time targets differ")
    model = GNNTransformer(**ckpt["model_kwargs"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    stats = {c: _empty_stats() for c in CHANNELS}
    bins = {c: {"n": np.zeros(6, dtype=np.int64),
                "sq": np.zeros(6, dtype=np.float64),
                "zero_sq": np.zeros(6, dtype=np.float64)} for c in CHANNELS}
    scatter = {c: [] for c in CHANNELS}
    rng = np.random.default_rng(42)  # Deterministic scatter-only subsampling.
    scatter_per_scenario = max(1, SCATTER_MAX_POINTS // len(data))
    rows = []
    prediction_dir = output / "predictions"
    if save_predictions:
        prediction_dir.mkdir(exist_ok=True)

    with torch.inference_mode():
        for item in data:
            x = item["x_dynamic"].to(device)
            y = item["y_delta"].to(device)
            valid = item["valid_mask"].to(device)
            pred = model(fixed["x_static"], x, fixed["x_baseline"], fixed["edge_index"])
            if pred.shape != y.shape or valid.shape != y.shape:
                raise ValueError(f"Shape mismatch for {item['scenario_id']}")
            row = {"scenario": item["scenario_id"]}
            for idx, channel in enumerate(CHANNELS):
                # Crucial: ALL metrics and plot data use ONLY valid_mask=True.
                mask = valid[:, idx]
                n = int(mask.sum())
                row[f"{channel}_n"] = n
                row[f"{channel}_rmse_s"] = None
                row[f"{channel}_no_change_rmse_s"] = None
                if not n:
                    continue
                truth = y[:, idx][mask].double().cpu().numpy()
                predicted = pred[:, idx][mask].double().cpu().numpy()
                error = predicted - truth
                stat = stats[channel]
                stat["n"] += n
                stat["abs"] += float(np.abs(error).sum())
                stat["sq"] += float(np.square(error).sum())
                stat["sq_zero"] += float(np.square(truth).sum())
                stat["sum"] += float(truth.sum())
                stat["sum2"] += float(np.square(truth).sum())
                stat["bias"] += float(error.sum())
                row[f"{channel}_rmse_s"] = float(np.sqrt(np.mean(np.square(error))))
                row[f"{channel}_no_change_rmse_s"] = float(np.sqrt(np.mean(np.square(truth))))
                if plots:
                    mag_idx = np.searchsorted(MAGNITUDE_EDGES_S, np.abs(truth), side="left")
                    b = bins[channel]
                    b["n"] += np.bincount(mag_idx, minlength=6)
                    b["sq"] += np.bincount(mag_idx, weights=np.square(error), minlength=6)
                    b["zero_sq"] += np.bincount(mag_idx, weights=np.square(truth), minlength=6)
                    # Scatter is a subset for visual legibility; metrics above use ALL valid links.
                    choice = rng.choice(n, size=min(n, scatter_per_scenario), replace=False)
                    scatter[channel].append((truth[choice], predicted[choice]))
            if save_predictions:
                np.savez_compressed(prediction_dir / f"{item['scenario_id']}.npz",
                                    predicted_delta_s=pred.cpu().numpy().astype(np.float32),
                                    actual_delta_s=y.cpu().numpy().astype(np.float32),
                                    valid_mask=valid.cpu().numpy())
            rows.append(row)

    results = {}
    for channel, s in stats.items():
        n = s["n"]
        sst = s["sum2"] - s["sum"] ** 2 / n if n else 0
        results[channel] = {
            "count": n,
            "mae_s": s["abs"] / n if n else None,
            "rmse_s": (s["sq"] / n) ** 0.5 if n else None,
            "no_change_rmse_s": (s["sq_zero"] / n) ** 0.5 if n else None,
            "r2_delta": 1 - s["sq"] / sst if sst > 0 else None,
            "mean_error_s": s["bias"] / n if n else None,
        }
    summary = {"split": split, "scenarios": len(data),
               "hour": data.hour, "best_epoch": int(ckpt["epoch"]),
               "metrics": results,
               "units": "seconds (signed link travel-time changes)"}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (output / "scenario_metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    if plots:
        _save_plots(output, checkpoint, int(ckpt["epoch"]), scatter,
                    bins, rows, results, history)
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
    parser.add_argument("--history", type=Path, default=None,
                        help="Training history.csv (default: beside best.pt)")
    args = parser.parse_args()
    evaluate(args.gnn_dir, args.splits or args.gnn_dir / "splits.json",
             args.checkpoint, args.out, args.split, args.device,
             plots=not args.no_plots, save_predictions=args.save_predictions,
             history=args.history)


if __name__ == "__main__":
    main()

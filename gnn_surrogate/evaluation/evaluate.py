"""Evaluate masked 08:00-09:00 travel-time changes against no-change baseline."""
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

CHANNELS = ("road", "bus_spec")


def evaluate(gnn_dir: Path, splits_file: Path, checkpoint: Path,
             output: Path, split: str = "val", device_name: str = "cpu") -> dict:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory not empty: {output}")
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
    stats = {c: {"n": 0, "abs": 0.0, "sq": 0.0, "sq_zero": 0.0,
                 "sum": 0.0, "sum2": 0.0} for c in CHANNELS}
    rows = []
    with torch.inference_mode():
        for item in data:
            x = item["x_dynamic"].to(device)
            y = item["y_delta"].to(device)
            valid = item["valid_mask"].to(device)
            pred = model(fixed["x_static"], x, fixed["x_baseline"], fixed["edge_index"])
            row = {"scenario": item["scenario_id"]}
            for idx, channel in enumerate(CHANNELS):
                mask = valid[:, idx]
                n = int(mask.sum())
                row[f"{channel}_n"] = n
                row[f"{channel}_rmse_s"] = None
                if not n:
                    continue
                error = (pred[:, idx][mask] - y[:, idx][mask]).double()
                truth = y[:, idx][mask].double()
                stat = stats[channel]
                stat["n"] += n
                stat["abs"] += float(error.abs().sum())
                stat["sq"] += float(error.square().sum())
                stat["sq_zero"] += float(truth.square().sum())
                stat["sum"] += float(truth.sum())
                stat["sum2"] += float(truth.square().sum())
                row[f"{channel}_rmse_s"] = float(error.square().mean().sqrt())
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
        }
    summary = {"split": split, "scenarios": len(data),
               "hour": data.hour, "best_epoch": int(ckpt["epoch"]),
               "metrics": results,
               "units": "seconds (signed link travel-time changes)"}
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (output / "scenario_metrics.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
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
    args = parser.parse_args()
    evaluate(args.gnn_dir, args.splits or args.gnn_dir / "splits.json",
             args.checkpoint, args.out, args.split, args.device)


if __name__ == "__main__":
    main()

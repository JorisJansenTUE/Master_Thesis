"""Run a nested-data-size GNN learning-curve experiment with one GPU allocation.

Only the training scenario count changes. All runs:
  * use identical, held-out validation and test manifests;
  * use the same initialization seed, graph, architecture and hyperparameters;
  * use mode-balanced loss scales from the existing full-data checkpoint;
  * train from scratch and select the best checkpoint using validation ONLY;
  * NEVER evaluate or inspect test predictions during this experiment.

The reference full-data model is read but never retrained. Results are saved
incrementally, so completed sizes are skipped if an HPC job is resubmitted.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import subprocess
import sys
from collections import Counter, defaultdict, deque
from pathlib import Path

import torch


DEFAULT_SIZES = (150, 350, 750, 1100)
MODES = ("cars", "trucks", "bikes")
REGIONS = ("all", "modified", "nearby", "unchanged")


def _load(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, item: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(item, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_if_absent_or_equal(path: Path, item: dict) -> None:
    if path.exists():
        if _load(path) != item:
            raise ValueError(f"Existing experiment metadata differs: {path}")
    else:
        _write_json(path, item)


def _prefix(sid: str) -> str:
    return sid.split("_", maxsplit=1)[0]


def stratified_nested_order(train: list[str], seed: int) -> list[str]:
    """Deterministic progressive stratification by sampler family.

    Each prefix (cluster/corridor/random/...) is shuffled independently,
    then groups are interleaved by proportional deficit. Every prefix of the
    resulting order approximately matches the full pool's strategy ratios.
    Prefixes of the order are nested training subsets by construction.
    """
    rng = random.Random(seed)
    grouped = defaultdict(list)
    for sid in train:
        grouped[_prefix(sid)].append(sid)
    groups = sorted(grouped)
    for group in groups:
        rng.shuffle(grouped[group])
    queues = {group: deque(grouped[group]) for group in groups}
    counts = {group: len(grouped[group]) for group in groups}
    selected = Counter()
    n = len(train)
    output = []
    for position in range(1, n + 1):
        # Largest positive deficit from proportional target for this prefix.
        group = max(
            (g for g in groups if queues[g]),
            key=lambda g: (position * counts[g] / n - selected[g], -groups.index(g)),
        )
        output.append(queues[group].popleft())
        selected[group] += 1
    assert len(output) == len(train) and set(output) == set(train)
    return output


def _reference(checkpoint: Path, full_splits: dict) -> dict:
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Full-data checkpoint missing: {checkpoint}")
    # The user's own trusted training checkpoint, not an external model.
    best = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if any(best["splits"].get(name) != full_splits[name]
           for name in ("train", "val", "test")):
        raise ValueError("Reference checkpoint uses a DIFFERENT train/val/test split")
    if tuple(best["target_modes"]) != MODES:
        raise ValueError(f"Unexpected reference modes: {best['target_modes']}")
    if best.get("config", {}).get("max_nodes") not in (None, 0):
        raise ValueError("Reference model was trained on a limited debug subgraph")
    scales = torch.as_tensor(best["mode_scales"])
    if scales.shape != (len(MODES),) or not torch.isfinite(scales).all() or (scales <= 0).any():
        raise ValueError("Invalid reference training loss scales")
    if "validation" not in best or "metrics" not in best["validation"]:
        raise ValueError("Missing validation metrics in full-data checkpoint")
    return best


def _make_row(n: int, best_epoch: int, val: dict, label: str) -> dict:
    row = {"train_scenarios": n, "model": label, "best_epoch": best_epoch,
           "val_mode_balanced_mse": val["loss"]}
    for region in REGIONS:
        summary = val["metrics"][region]
        row[f"{region}_positions"] = summary["count"]
        for mode in MODES:
            row[f"{region}_{mode}_mae"] = summary["mae"][mode]
            row[f"{region}_{mode}_rmse"] = summary["rmse"][mode]
    return row


def summarize(out_root: Path, sizes: list[int], reference: dict,
              full_n: int, *, make_plots: bool = True) -> list[dict]:
    """Write a partial or complete comparison without touching test data."""
    rows = []
    for size in sizes:
        run = out_root / f"train_{size:04d}"
        if not (run / "summary.json").is_file():
            continue
        summary = _load(run / "summary.json")
        if summary.get("test_evaluated") is not False:
            raise ValueError(f"Unexpected test evaluation in learning curve run: {run}")
        validation = _load(run / "best_validation.json")
        rows.append(_make_row(size, int(summary["best_epoch"]), validation, "subset"))
    rows.append(_make_row(full_n, int(reference["epoch"]), reference["validation"], "reference"))
    rows.sort(key=lambda r: r["train_scenarios"])
    if rows:
        path = out_root / "learning_curve.csv"
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Saved comparison of {len(rows)} sizes: {path}", flush=True)
    if make_plots and len(rows) > 1:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("Matplotlib not available; CSV saved but plots skipped", flush=True)
            return rows
        xs = [r["train_scenarios"] for r in rows]
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(xs, [r["val_mode_balanced_mse"] for r in rows], "o-")
        ax.set(xlabel="Training scenarios", ylabel="Best validation mode-balanced MSE",
               title="GNN learning curve (fixed loss scales)")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(out_root / "learning_curve_val_loss.png", dpi=160)
        plt.close(fig)
        for region in ("all", "modified", "unchanged"):
            fig, ax = plt.subplots(figsize=(7, 4.5))
            for mode in MODES:
                ax.plot(xs, [r[f"{region}_{mode}_rmse"] for r in rows], "o-", label=mode)
            ax.set(xlabel="Training scenarios", ylabel="Validation RMSE (30-hour link traversals)",
                   title=f"GNN learning curve: {region} streets")
            ax.grid(alpha=0.25)
            ax.legend()
            fig.tight_layout()
            fig.savefig(out_root / f"learning_curve_{region}_rmse.png", dpi=160)
            plt.close(fig)
    return rows


def run(args: argparse.Namespace) -> None:
    original_splits = _load(args.splits)
    for name in ("train", "val", "test"):
        if name not in original_splits or not isinstance(original_splits[name], list):
            raise ValueError(f"Invalid split manifest: {args.splits}")
    all_ids = sum((original_splits[k] for k in ("train", "val", "test")), [])
    if len(set(all_ids)) != len(all_ids) or original_splits["scenario_count"] != len(all_ids):
        raise ValueError("Original splits contain duplicates or wrong scenario_count")
    train_ids = original_splits["train"]
    sizes = list(args.sizes)
    if (not sizes or sizes != sorted(set(sizes)) or sizes[0] <= 0
            or sizes[-1] >= len(train_ids)):
        raise ValueError(f"Use increasing subset sizes between 1 and {len(train_ids)-1}")
    reference = _reference(args.reference, original_splits)
    full_n = len(train_ids)
    checksum = hashlib.sha256(args.splits.read_bytes()).hexdigest()
    ordered = stratified_nested_order(train_ids, args.subset_seed)
    args.out.mkdir(parents=True, exist_ok=True)
    config = {
        "experiment": "nested_data_scaling", "sizes": sizes,
        "subset_seed": args.subset_seed, "model_seed": args.model_seed,
        "epochs_cap": args.epochs, "patience": args.patience,
        "learning_rate": args.lr,
        "full_splits": str(args.splits.resolve()),
        "full_splits_sha256": checksum,
        "reference_checkpoint": str(args.reference.resolve()),
        "reference_best_epoch": int(reference["epoch"]),
        "reference_mode_scales": torch.as_tensor(reference["mode_scales"]).tolist(),
        "full_train_size": full_n,
        "validation_count": len(original_splits["val"]),
        "test_count_unused": len(original_splits["test"]),
        "reference_sampler_counts": dict(Counter(map(_prefix, train_ids))),
        "note": "All subsets from scratch. Validation-only comparisons; test is never evaluated."
    }
    _write_if_absent_or_equal(args.out / "experiment.json", config)
    split_dir = args.out / "subsets"
    for size in sizes:
        manifest = {
            "train": ordered[:size],
            "val": original_splits["val"],
            "test": original_splits["test"],
            "scenario_count": size + len(original_splits["val"]) + len(original_splits["test"]),
        }
        _write_if_absent_or_equal(split_dir / f"splits_{size:04d}.json", manifest)
        print(f"Subset {size}: {dict(Counter(map(_prefix, manifest['train'])))}", flush=True)

    if args.prepare_only:
        print("Split manifests generated; no training was launched.", flush=True)
        summarize(args.out, sizes, reference, full_n, make_plots=False)
        return

    for size in sizes:
        subset_path = split_dir / f"splits_{size:04d}.json"
        run_dir = args.out / f"train_{size:04d}"
        if (run_dir / "summary.json").is_file() and (run_dir / "best_validation.json").is_file():
            finished = _load(run_dir / "summary.json")
            if finished.get("test_evaluated") is not False:
                raise ValueError(f"Found unexpected test results in {run_dir}")
            print(f"Skip completed experiment: n={size}", flush=True)
        elif run_dir.exists() and any(run_dir.iterdir()):
            raise RuntimeError(
                f"Incomplete experiment found: {run_dir}. Do NOT overwrite it. "
                "Rename/delete only this incomplete run directory before resubmitting."
            )
        else:
            cmd = [
                sys.executable, "-u", "-m", "gnn_surrogate.training.train",
                "--gnn-dir", str(args.gnn_dir),
                "--splits", str(subset_path),
                "--out", str(run_dir),
                "--device", "cuda",
                "--max-nodes", "0",
                "--epochs", str(args.epochs),
                "--batch-size", "1",
                "--lr", str(args.lr),
                "--patience", str(args.patience),
                "--seed", str(args.model_seed),
                "--scales-from", str(args.reference),
                "--skip-test",
            ]
            print(f"\n=== Training n={size}/{full_n} from scratch ===", flush=True)
            subprocess.run(cmd, check=True)
        summarize(args.out, sizes, reference, full_n)
    print("\nLearning-curve experiment finished successfully.", flush=True)
    print("Summary:", args.out / "learning_curve.csv", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gnn-dir", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True,
                        help="Trusted existing full-data best.pt checkpoint")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--sizes", type=int, nargs="+", default=list(DEFAULT_SIZES))
    parser.add_argument("--subset-seed", type=int, default=2026)
    parser.add_argument("--model-seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    for key in ("gnn_dir", "splits", "reference", "out"):
        setattr(args, key, getattr(args, key).expanduser().resolve())
    run(args)


if __name__ == "__main__":
    main()

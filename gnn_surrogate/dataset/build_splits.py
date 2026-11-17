"""Create a fixed random train/val/test split of complete processed scenarios.

Only pairs with BOTH x_dynamic.npy and target_delta.npz (+ metadata) qualify.
No original MATSim files are read. Never overwrite an existing split silently.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .dataset import discover_scenarios


def build_split_manifest(
    scenario_ids: list[str],
    *,
    seed: int = 42,
    train_fraction: float = 0.8,
    val_fraction: float = 0.1,
) -> dict:
    if (len(scenario_ids) != len(set(scenario_ids)) or len(scenario_ids) < 3):
        raise ValueError("At least 3 unique, complete scenarios are required")
    if not (0 < train_fraction < 1 and 0 < val_fraction < 1
            and train_fraction + val_fraction < 1):
        raise ValueError("Train/val fractions must be positive and sum to less than 1")

    # Sorting before shuffling makes the split independent of filesystem order.
    randomized = np.random.default_rng(seed).permutation(sorted(scenario_ids)).tolist()
    total = len(randomized)
    # Keep every split nonempty, even for a small inspection dataset.
    n_train = min(max(1, round(total * train_fraction)), total - 2)
    n_val = min(max(1, round(total * val_fraction)), total - n_train - 1)
    return {
        "seed": seed,
        "fractions": {"train": train_fraction,
                      "val": val_fraction,
                      "test": 1.0 - train_fraction - val_fraction},
        "scenario_count": total,
        "train": randomized[:n_train],
        "val": randomized[n_train:n_train + n_val],
        "test": randomized[n_train + n_val:],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dynamic", required=True, type=Path,
                        help="Root directory with dynamic/cluster_XXXXX/x_dynamic.npy")
    parser.add_argument("--targets", required=True, type=Path,
                        help="Root directory with targets/cluster_XXXXX/target_delta.npz")
    parser.add_argument("--out", required=True, type=Path,
                        help="Output JSON file, e.g. data/GNN/splits.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.out.exists() and not args.overwrite:
        raise FileExistsError(f"Split file already exists (use --overwrite explicitly): {args.out}")
    scenario_ids = discover_scenarios(args.dynamic, args.targets)
    manifest = build_split_manifest(scenario_ids, seed=args.seed,
                                    train_fraction=args.train_fraction,
                                    val_fraction=args.val_fraction)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Saved scenario split: {args.out}")
    print(f"Total complete scenarios: {len(scenario_ids):,}")
    print(" / ".join(f"{part}={len(manifest[part]):,}" for part in ("train", "val", "test")))
    print("Keep this split file fixed for reproducible experiments.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Extend a previously VERIFIED source-aware GNN split without reshuffling it.

Each original train/val/test ID retains its existing split. Additional GA roots
are used as TRAIN by default. An optional *independent* test GA root is TEST.

No network/targets are copied: the staging directory contains only symlinks.
The existing intervention fingerprinting and completion validation are reused
from stage_expanded_gnn_v3.py.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path

from stage_expanded_gnn_v3 import direct_ga_scenarios, eligible, fingerprint


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _extra(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise argparse.ArgumentTypeError("Expected GROUP=/absolute/path/to/GA_run")
    label, path = spec.split("=", 1)
    if not label or not label.replace("_", "").isalnum():
        raise argparse.ArgumentTypeError(f"Invalid group label {label!r}")
    if not path or not Path(path).is_absolute():
        raise argparse.ArgumentTypeError("GA directory must be an absolute path")
    return label, Path(path)


def _splits(d: dict) -> dict[str, list[str]]:
    out = {k: list(d[k]) for k in ("train", "val", "test")}
    for k, ids in out.items():
        if not isinstance(ids, list) or not ids or not all(isinstance(s, str) for s in ids):
            raise ValueError(f"Invalid {k} split in parent dataset")
        if len(set(ids)) != len(ids):
            raise ValueError(f"Duplicate IDs within parent {k} split")
    ids = sum(out.values(), [])
    if len(set(ids)) != len(ids):
        raise ValueError("Overlap across existing train/val/test splits")
    return out


def run(args: argparse.Namespace) -> dict:
    parent = args.parent_gnn.resolve(strict=True)
    historical = args.historical.resolve(strict=True)
    baseline = historical / "baseline_00000"
    segments = historical / "street_segments.csv.gz"
    for p in (baseline / "scenario.json", baseline / "network.xml.gz", segments,
              parent / "splits.json", parent / "fixed_provenance.json",
              parent / "preparation_metadata" / "source_manifest.csv"):
        if not p.is_file():
            raise FileNotFoundError(p)
    baseline_meta = _json(baseline / "scenario.json")
    baseline_hashes = baseline_meta.get("baseline_hashes")
    if not baseline_hashes:
        raise ValueError("Historical baseline has no baseline_hashes")
    reason = eligible(baseline, baseline_hashes, is_baseline=True)
    if reason:
        raise ValueError(f"Invalid historical baseline: {reason}")
    provenance = _json(parent / "fixed_provenance.json")
    expected_sha = provenance.get("baseline_network_sha256")
    if not expected_sha or _sha(baseline / "network.xml.gz") != expected_sha:
        raise ValueError("Parent GNN and supplied historical baseline have different networks")

    parent_roles = _splits(_json(parent / "splits.json"))
    original_id_to_role = {sid: split for split, ids in parent_roles.items() for sid in ids}
    manifest_path = parent / "preparation_metadata" / "source_manifest.csv"
    with manifest_path.open(newline="", encoding="utf-8") as f:
        prior = list(csv.DictReader(f))
    if set(original_id_to_role) != {row["scenario"] for row in prior} or len(prior) != len(original_id_to_role):
        raise ValueError("Parent source manifest and split membership disagree")

    selected: dict[str, dict] = {}
    by_fp: dict[str, str] = {}
    counts = Counter()
    for row in sorted(prior, key=lambda x: x["scenario"]):
        sid, role = row["scenario"], original_id_to_role[row["scenario"]]
        if row.get("split") != role:
            raise ValueError(f"Existing split changed for {sid}")
        source = Path(row["source"]).resolve(strict=True)
        if source.name != sid or not source.is_dir():
            raise ValueError(f"Scenario ID/source mismatch: {sid} -> {source}")
        reason = eligible(source, baseline_hashes)
        if reason:
            raise ValueError(f"Previously included scenario became invalid: {sid}: {reason}")
        fp = fingerprint(source)
        if fp != row.get("fingerprint"):
            raise ValueError(f"Interventions changed since prior preparation: {sid}")
        if fp in by_fp:
            raise ValueError(f"Existing manifest has duplicated intervention fingerprints: {sid}")
        selected[sid] = dict(row, split=role, source=str(source))
        by_fp[fp] = sid
        counts[f"parent_{role}"] += 1

    train = list(args.train_ga or [])
    test = list(args.test_ga or [])
    if not train:
        raise ValueError("Supply at least one --train-ga GROUP=/absolute/path")
    groups = train + test
    labels = [label for label, _ in groups]
    if len(set(labels)) != len(labels):
        raise ValueError("Repeated group labels")
    if set(labels) & {r["group"] for r in prior}:
        raise ValueError("New GA group label collides with existing manifest")
    canon_roots = [str(p.resolve(strict=True)) for _, p in groups]
    if len(set(canon_roots)) != len(canon_roots):
        raise ValueError("The same GA directory was provided more than once (training/test leakage risk)")

    # Add test FIRST: if a new training run duplicates the independent test,
    # the training duplicate is dropped. Old parent assignments are immutable.
    for split, new_groups in (("test", test), ("train", train)):
        for label, path in new_groups:
            root = path.resolve(strict=True)
            if not root.is_dir():
                raise NotADirectoryError(root)
            candidates = direct_ga_scenarios(root)
            counts[f"{label}_discovered"] = len(candidates)
            if not candidates:
                raise RuntimeError(f"No GA scenario folders containing scenario.json under {root}")
            for source in candidates:
                reason = eligible(source, baseline_hashes)
                if reason:
                    counts[f"{label}_excluded_incomplete_or_invalid"] += 1
                    continue
                sid, fp = source.name, fingerprint(source)
                if sid in selected:
                    if selected[sid]["fingerprint"] != fp:
                        raise ValueError(f"ID collision with different intervention: {sid}")
                    counts[f"{label}_duplicate_id"] += 1
                    continue
                if fp in by_fp:
                    existing = selected[by_fp[fp]]
                    counts[f"{label}_duplicate_configuration"] += 1
                    if split == "test" and existing["split"] != "test":
                        counts[f"{label}_heldout_overlap_with_existing_trainval"] += 1
                    continue
                selected[sid] = {
                    "scenario": sid, "group": label, "split": split,
                    "source": str(source.resolve()), "fingerprint": fp,
                    "baseline_network_hash": baseline_hashes.get("network", ""),
                }
                by_fp[fp] = sid
                counts[f"{label}_included"] += 1
            if counts[f"{label}_included"] < args.min_per_run:
                raise RuntimeError(
                    f"{label}: only {counts[f'{label}_included']} distinct complete scenarios "
                    f"(min={args.min_per_run}). Check completion, baseline identity, and duplicates."
                )

    roles = {role: sorted(sid for sid, item in selected.items() if item["split"] == role)
             for role in ("train", "val", "test")}
    if not all(roles.values()):
        raise ValueError("Empty split")
    for split in ("val", "test"):
        if set(roles[split]) & set(roles["train"]):
            raise ValueError(f"Training leakage into {split}")
    for role, ids in parent_roles.items():
        if not set(ids).issubset(roles[role]):
            raise AssertionError(f"Parent {role} assignments not preserved")

    # Finish all validation before creating output.
    out = args.out.absolute()
    if out.exists():
        raise FileExistsError(f"Use a fresh --out stage directory: {out}")
    out.mkdir(parents=True)
    (out / "baseline_00000").symlink_to(baseline, target_is_directory=True)
    (out / "street_segments.csv.gz").symlink_to(segments)
    for sid, item in sorted(selected.items()):
        (out / sid).symlink_to(item["source"], target_is_directory=True)
    columns = ["scenario", "group", "split", "source", "fingerprint", "baseline_network_hash"]
    with (out / "source_manifest.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for sid in sorted(selected):
            writer.writerow({c: selected[sid].get(c, "") for c in columns})
    plan = {
        "policy": ("Immutable parent train/val/test; new same-seed GA runs TRAIN only; "
                   "optional independent seed GA TEST; intervention fingerprints deduplicated"),
        "seed": args.seed,
        "parent_gnn": str(parent),
        "parent_split_sizes": {role: len(ids) for role, ids in parent_roles.items()},
        "paths": {"historical": str(historical),
                  "new_training_runs": {label: str(path) for label, path in train},
                  "new_test_runs": {label: str(path) for label, path in test}},
        "splits": roles,
        "counts": dict(counts),
    }
    (out / "split_plan.json").write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    summary = {"total": len(selected), "split_sizes": {r: len(ids) for r, ids in roles.items()},
               "added": {label: counts[f"{label}_included"] for label in labels},
               "rejected_or_deduplicated": {k: v for k, v in counts.items()
                                            if "duplicate" in k or "excluded" in k or "overlap" in k},
               "split_policy": plan["policy"]}
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--parent-gnn", type=Path, required=True)
    p.add_argument("--historical", type=Path, required=True)
    p.add_argument("--train-ga", type=_extra, action="append", required=True,
                   metavar="GROUP=/absolute/GA_run")
    p.add_argument("--test-ga", type=_extra, action="append", default=[],
                   metavar="GROUP=/absolute/heldout_GA_run")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--min-per-run", type=int, default=1)
    args = p.parse_args()
    if args.min_per_run < 1:
        p.error("--min-per-run must be positive")
    run(args)


if __name__ == "__main__":
    main()

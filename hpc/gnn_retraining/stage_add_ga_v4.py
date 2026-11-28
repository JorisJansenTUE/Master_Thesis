#!/usr/bin/env python3
"""Extend a previously VERIFIED source-aware GNN split without reshuffling it.

Each original train/val/test ID retains its existing split. For each new GA run,
whole *late* creation generations (~20%) are held out for VAL while earlier
generations remain TRAIN. An optional independent new-seed GA remains TEST.

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


def _generation_map(root: Path, label: str, scenario_ids: set[str],
                    override: Path | None) -> tuple[dict[str, int], dict]:
    """Recover GA *creation* generations from an actual saved optimizer archive.

    Search only the supplied GA root. If multiple state snapshots are present,
    use the one covering most included scenario IDs; conflicting equally complete
    snapshots fail closed rather than silently changing the validation set.
    """
    candidates = ([override.resolve(strict=True)] if override else
                  sorted(root.glob("optimization_state*.json")))
    if not candidates:
        raise FileNotFoundError(
            f"{label}: no optimization_state*.json in {root}; "
            f"supply --generation-state {label}=/absolute/path/to/state.json"
        )
    choices = []
    for path in candidates:
        state = _json(path)
        archive = state.get("archive")
        if not isinstance(archive, dict):
            continue
        mapping = {}
        for record in archive.values():
            if not isinstance(record, dict):
                continue
            sid = record.get("scenario")
            gen = record.get("created_generation")
            if sid not in scenario_ids:
                continue
            if not isinstance(gen, int) or isinstance(gen, bool) or gen < 0:
                continue
            if sid in mapping and mapping[sid] != gen:
                raise ValueError(f"{label}: conflicting creation generation for {sid} in {path}")
            mapping[sid] = gen
        choices.append((len(mapping), path, mapping))
    if not choices:
        raise ValueError(f"{label}: state files have no usable archive; candidates={candidates}")
    max_count = max(count for count, _, _ in choices)
    best = [x for x in choices if x[0] == max_count]
    if len(best) > 1 and any(x[2] != best[0][2] for x in best[1:]):
        raise ValueError(f"{label}: equally complete state files disagree on generations: "
                         f"{[str(x[1]) for x in best]}; specify --generation-state")
    _, state_path, mapping = best[0]
    coverage = len(mapping) / len(scenario_ids)
    return mapping, {"state_file": str(state_path), "state_sha256": _sha(state_path),
                     "generation_coverage": coverage,
                     "mapped_scenarios": len(mapping),
                     "unmapped_scenarios": len(scenario_ids)-len(mapping)}


def _reserve_generation_validation(
    selected: dict[str, dict], train: list[tuple[str, Path]],
    fraction: float, state_overrides: dict[str, Path], min_coverage: float,
    counts: Counter,
) -> dict:
    """Keep whole late creation generations together, preserving parent splits.

    Earlier generations train the GNN; a contiguous suffix of later generations
    is reserved for validation. Unmapped scenarios always remain in training.
    """
    details = {}
    for label, root in train:
        ids = {sid for sid, row in selected.items() if row["group"] == label}
        if not ids:
            raise ValueError(f"{label}: no distinct eligible training scenarios")
        generations, audit = _generation_map(root.resolve(strict=True), label, ids,
                                              state_overrides.get(label))
        if audit["generation_coverage"] < min_coverage:
            raise ValueError(
                f"{label}: only {len(generations)}/{len(ids)} selected scenarios "
                f"({100*audit['generation_coverage']:.1f}%) have recorded "
                f"created_generation; require {100*min_coverage:.1f}%. "
                f"Inspect {audit['state_file']} or specify --generation-state."
            )
        buckets: dict[int, list[str]] = {}
        for sid, generation in generations.items():
            buckets.setdefault(generation, []).append(sid)
        timeline = sorted(buckets)
        if len(timeline) < 2:
            raise ValueError(f"{label}: need at least two recorded creation generations "
                             f"to reserve complete generation blocks for validation")
        target = max(1, round(len(ids) * fraction))
        holdout = []
        chosen = []
        # Reserve a contiguous suffix, leaving at least the oldest generation
        # in training even for very short GA runs.
        for gen in reversed(timeline[1:]):
            holdout.extend(sorted(buckets[gen]))
            chosen.append(gen)
            if len(holdout) >= target:
                break
        if not holdout:
            raise ValueError(f"{label}: could not construct generation-based validation")
        for sid in holdout:
            selected[sid]["split"] = "val"
        for sid, gen in generations.items():
            selected[sid]["created_generation"] = gen
        counts[f"{label}_val"] = len(holdout)
        counts[f"{label}_train"] = len(ids) - len(holdout)
        counts[f"{label}_unmapped_generation"] = len(ids) - len(generations)
        details[label] = {
            **audit, "included": len(ids), "requested_validation_fraction": fraction,
            "actual_validation_fraction": len(holdout)/len(ids),
            "val_count": len(holdout), "train_count": len(ids)-len(holdout),
            "validation_generations": sorted(chosen),
            "train_generations": [g for g in timeline if g not in chosen],
            "mapping_policy": "contiguous latest creation generations, full blocks",
        }
    return details


def run(args: argparse.Namespace) -> dict:
    parent = args.parent_gnn.resolve(strict=True)
    historical = args.historical.resolve(strict=True)
    baseline = historical / "baseline_00000"
    segments = historical / "street_segments.csv.gz"
    for p in (baseline / "scenario.json", baseline / "network.xml.gz",
              baseline / "transit_schedule.xml.gz", segments,
              parent / "splits.json", parent / "fixed_provenance.json",
              parent / "preparation_metadata" / "source_manifest.csv"):
        if not p.is_file():
            raise FileNotFoundError(p)
    baseline_meta = _json(baseline / "scenario.json")
    baseline_hashes = baseline_meta.get("baseline_hashes")
    if not baseline_hashes:
        raise ValueError("Historical baseline has no baseline_hashes")
    # The historical baseline records hashes of original inputs; GA scenarios
    # record hashes of *completed* baseline_00000 network and schedule files.
    # This is the verified November 25 provenance fix: derive the alternative
    # identity from the actual files, never whitelist arbitrary metadata.
    completed_hashes = {
        "network": _sha(baseline / "network.xml.gz"),
        "schedule": _sha(baseline / "transit_schedule.xml.gz"),
    }
    reason = eligible(baseline, baseline_hashes, is_baseline=True,
                      completed_hashes=completed_hashes)
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
        reason = eligible(source, baseline_hashes, completed_hashes=completed_hashes)
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
                reason = eligible(source, baseline_hashes, completed_hashes=completed_hashes)
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

    states = {}
    for label, path in (args.generation_state or []):
        if label in states:
            raise ValueError(f"Duplicate --generation-state label {label}")
        states[label] = path
    if set(states) - {label for label, _ in train}:
        raise ValueError(f"Generation state labels not present in --train-ga: "
                         f"{sorted(set(states)-{label for label, _ in train})}")
    generation_validation = _reserve_generation_validation(
        selected, train, args.val_fraction, states, args.min_generation_coverage, counts,
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
    columns = ["scenario", "group", "split", "source", "fingerprint",
               "baseline_network_hash", "created_generation"]
    with (out / "source_manifest.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for sid in sorted(selected):
            writer.writerow({c: selected[sid].get(c, "") for c in columns})
    plan = {
        "policy": ("Immutable parent train/val/test; new same-seed GA runs split "
                   "by complete late creation generations into TRAIN and VAL; "
                   "optional independent seed GA TEST; intervention fingerprints deduplicated"),
        "new_ga_validation": generation_validation,
        "validation_fraction_requested": args.val_fraction,
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
               "new_ga_validation": {label: {"train": entry["train_count"],
                                                "val": entry["val_count"],
                                                "generations": entry["validation_generations"],
                                                "generation_coverage": entry["generation_coverage"]}
                                     for label, entry in generation_validation.items()},
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
    p.add_argument("--val-fraction", type=float, default=0.20,
                   help="Approx. fraction of each new GA root allocated to VAL by whole generations")
    p.add_argument("--min-generation-coverage", type=float, default=0.80,
                   help="Fail if optimizer state does not identify enough included scenarios")
    p.add_argument("--generation-state", type=_extra, action="append", default=[],
                   metavar="GROUP=/absolute/path/to/optimization_state.json",
                   help="Override automatic optimizer-state discovery for one GA root")
    p.add_argument("--min-per-run", type=int, default=1)
    args = p.parse_args()
    if args.min_per_run < 1:
        p.error("--min-per-run must be positive")
    if not 0 < args.val_fraction < 0.5:
        p.error("--val-fraction must be >0 and <0.5")
    if not 0 < args.min_generation_coverage <= 1:
        p.error("--min-generation-coverage must be >0 and <=1")
    run(args)


if __name__ == "__main__":
    main()

"""Score existing MATSim scenarios and select diverse GA warm-start individuals.

Run from the thesis repository root:
    python -m optimization.warm_start --limit 20 --count 4    # quick pilot
    python -m optimization.warm_start                         # all scenarios

Only a completed scenario's saved linkstats are used; MATSim never runs here.
The resulting JSON contains complete intervention lists and historical fitness.
Use load_warm_start() with ga.initial_population(..., initial=individuals).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from accessibility.inputs.grid import destination_settings as dest
from network_sampling.dataset.generate import DEFAULT_SETTINGS, load_baseline
from network_sampling.interventions.intervention_types import (
    Intervention, InterventionType, RoadSpaceVariant,
)
from optimization.fitness import FITNESS_VERSION, evaluate_scenario
from optimization.ga import configuration_hash, validate_individual

# ---------------------------- Editable settings ----------------------------
DATASET = Path("data/Simulated_Runs_v2")
OUTPUT = Path("data/GA_Pilot/warm_start.json")
BASELINE_NAME = "baseline_00000"
SPATIAL_DIR = dest.spatial_dir()
REFERENCE_FILE = dest.baseline_reference_file(BASELINE_NAME).with_name(
    "sufficiency_reference_15min.json"
)
BUDGET_M = 5000.0
WARM_START_COUNT = 8                  # for example, 8 of a 40-individual population
MAX_SEGMENT_OVERLAP = 0.70           # maximum Jaccard similarity of street segments

# Match your existing numerical fitness settings (Step 4).
MODE_WEIGHTS = {"car": 1.0, "bike": 1.0, "pt": 1.0}
TIME_WEIGHTS = {15.0: 1.0}
SUFFICIENCY_LEVEL = "high"
HOUR_START = 8
LINKSTATS_STATISTIC = "avg"
BIKE_SPEED_M_S = 4.17
PT_TRANSFER_PENALTY_S = 300.0
ORIGIN_SNAP_M = 500.0
JOB_SNAP_M = 300.0
# ---------------------------------------------------------------------------


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _individual_from_records(records: list[dict]) -> list[Intervention]:
    return [
        Intervention(
            segment_id=str(row["segment_id"]),
            intervention_type=InterventionType(row["intervention_type"]),
            road_space_variant=(RoadSpaceVariant(row["road_space_variant"])
                                if row.get("road_space_variant") is not None else None),
            target_link_ids=tuple(str(x) for x in row.get("target_link_ids", [])),
        )
        for row in records
    ]


def load_warm_start(path: str | Path) -> tuple[list[list[Intervention]], dict[str, float]]:
    """Return GA-ready individuals and historical fitness indexed by config hash.

    The existing ga.initial_population() validates seeds against its baseline and
    fills the remaining positions using the established samplers. Historical
    fitness can only be reused after confirming MATSim/evaluation comparability.
    """
    payload = _read_json(Path(path))
    if payload.get("version") != 1 or payload.get("fitness_version") != FITNESS_VERSION:
        raise ValueError("Warm-start file schema or fitness version is not current")
    individuals = []
    historical_fitness = {}
    for seed in payload["seeds"]:
        individual = _individual_from_records(seed["interventions"])
        fingerprint = configuration_hash(individual)
        if fingerprint != seed["config_hash"] or fingerprint in historical_fitness:
            raise ValueError("Invalid/duplicate warm-start configuration fingerprint")
        individuals.append(individual)
        historical_fitness[fingerprint] = float(seed["fitness"])
    return individuals, historical_fitness


def _completed_status(path: Path) -> dict | None:
    status_path = path / "run_status.json"
    if not status_path.is_file():
        return None
    status = _read_json(status_path)
    return status if status.get("state", status.get("status")) in {"completed", "complete"} else None


def _same_simulation_settings(baseline: dict, candidate: dict) -> bool:
    # Different iteration counts or linkstats windows are not comparable.
    for field in ("last_iteration", "linkstats_interval", "linkstats_average_iterations"):
        a, b = baseline.get(field), candidate.get(field)
        if a is not None and b is not None and a != b:
            return False
    return True


def _evaluate(path: Path, spatial_dir: Path, reference_file: Path,
              existing_only: bool) -> dict:
    cached = ((path / "accessibility" / "sufficiency_summary.csv").is_file()
              and (path / "accessibility" / "accessibility_by_hexagon.csv.gz").is_file())
    if existing_only and not cached:
        raise FileNotFoundError("no saved accessibility; --existing-only was requested")

    kwargs = dict(
        baseline_name=BASELINE_NAME,
        mode_weights=MODE_WEIGHTS, time_weights=TIME_WEIGHTS,
        level=SUFFICIENCY_LEVEL, hour_start=HOUR_START,
        linkstats_statistic=LINKSTATS_STATISTIC,
        bike_speed_m_s=BIKE_SPEED_M_S, pt_transfer_penalty_s=PT_TRANSFER_PENALTY_S,
        origin_snap_m=ORIGIN_SNAP_M, job_snap_m=JOB_SNAP_M,
    )
    if cached:
        try:
            return evaluate_scenario(path, spatial_dir, reference_file,
                                     score_only=True, **kwargs)
        except ValueError:
            # A saved 15/20/30-minute or squared-shortfall CSV must be rebuilt.
            if existing_only:
                raise
            print(f"  Recalculating incompatible accessibility: {path.name}", flush=True)
    if existing_only:
        raise FileNotFoundError("cannot calculate accessibility with --existing-only")
    return evaluate_scenario(path, spatial_dir, reference_file,
                             score_only=False, **kwargs)


def _score_worker(task: tuple[Path, Path, Path, bool]):
    """Score one *independent* scenario; workers never write shared GA outputs.

    The fixed baseline reference and attachment mask must exist before workers
    start. Only the main process writes the warm-start ranking and manifest.
    """
    path, spatial_dir, reference_file, existing_only = task
    try:
        return _evaluate(path, spatial_dir, reference_file, existing_only), None
    except (OSError, ValueError, KeyError, TypeError, ZeroDivisionError, RuntimeError) as exc:
        return None, (type(exc).__name__, str(exc))


def _segment_similarity(a: set[str], b: set[str]) -> float:
    union = a | b
    return len(a & b) / len(union) if union else 1.0


def _write_ranking(path: Path, rows: list[dict], selected_hashes: set[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = (
        "rank", "scenario", "strategy", "fitness", "budget_used_m",
        "intervention_count", "config_hash", "car_score", "bike_score",
        "pt_score", "new_bike_attachments", "lost_bike_attachments", "selected",
    )
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for rank, row in enumerate(rows, 1):
            component = {x["mode"]: x for x in row["score"]["components"]}
            writer.writerow({
                "rank": rank, "scenario": row["scenario"], "strategy": row["strategy"],
                "fitness": row["fitness"], "budget_used_m": row["budget_used_m"],
                "intervention_count": len(row["interventions"]),
                "config_hash": row["config_hash"],
                **{f"{mode}_score": component[mode]["mode_score"] for mode in ("car", "bike", "pt")},
                "new_bike_attachments": component["bike"]["n_newly_attached"],
                "lost_bike_attachments": component["bike"]["n_lost_attachment"],
                "selected": row["config_hash"] in selected_hashes,
            })
    os.replace(tmp, path)


def build_warm_start(
    dataset: Path, output: Path, spatial_dir: Path, reference_file: Path,
    *, count: int, budget_m: float, max_overlap: float, limit: int = 0,
    existing_only: bool = False, workers: int = 1,
) -> dict:
    """Score historical completed scenarios and export valid, diverse GA seeds."""
    if workers < 1:
        raise ValueError("workers must be at least 1")
    dataset = dataset.resolve(strict=True)
    baseline_dir = dataset / BASELINE_NAME
    baseline_metadata = _read_json(baseline_dir / "scenario.json")
    if not baseline_metadata.get("baseline_hashes"):
        raise ValueError("Baseline scenario.json has no baseline hashes; cannot compare datasets")
    baseline_status = _completed_status(baseline_dir)
    if baseline_status is None:
        raise RuntimeError(f"Baseline simulation must be completed: {baseline_dir}")
    print("Scoring the fixed baseline...", flush=True)
    baseline_score = _evaluate(baseline_dir, spatial_dir, reference_file, existing_only)

    # Stage 1: filter cheap metadata and deduplicate *before* expensive routing.
    # Each unique scenario is then independent and can run in a process pool.
    queued_hashes: set[str] = set()
    candidates: list[dict] = []
    rows: list[dict] = []
    skips = Counter()
    considered = 0
    paths = sorted(p for p in dataset.iterdir() if p.is_dir() and p.name != BASELINE_NAME
                   and (p / "scenario.json").is_file())
    warnings_shown = 0
    for path in paths:
        try:
            metadata = _read_json(path / "scenario.json")
            if metadata.get("baseline_hashes") != baseline_metadata.get("baseline_hashes"):
                skips["different baseline"] += 1
                continue
            if not (path / "interventions.json").is_file():
                skips["no interventions.json"] += 1
                continue
            records = _read_json(path / "interventions.json")
            if not isinstance(records, list) or not records:
                skips["empty intervention list"] += 1
                continue
            budget = sum(float(entry["length_m"]) for entry in records)
            if not math.isfinite(budget) or budget < 0:
                raise ValueError("invalid intervention lengths")
            if abs(budget - float(metadata.get("budget_used_m", budget))) > 1e-3:
                skips["budget metadata mismatch"] += 1
                continue
            if budget > budget_m + 1e-6:
                skips["above GA budget"] += 1
                continue
            status = _completed_status(path)
            if status is None:
                skips["not completed"] += 1
                continue
            if not _same_simulation_settings(baseline_status, status):
                skips["different MATSim settings"] += 1
                continue
            individual = _individual_from_records(records)
            key = configuration_hash(individual)
            if key in queued_hashes:
                skips["duplicate configuration"] += 1
                continue
            if limit and considered >= limit:
                break
            considered += 1
            queued_hashes.add(key)
            candidates.append({
                "path": path, "scenario": path.name,
                "strategy": metadata.get("strategy", "unknown"),
                "config_hash": key, "budget_used_m": budget,
                "interventions": records,
                "segments": {iv.segment_id for iv in individual},
            })
        except (OSError, ValueError, KeyError, TypeError, ZeroDivisionError, RuntimeError) as exc:
            skips[type(exc).__name__] += 1
            if warnings_shown < 10:
                print(f"  SKIP {path.name}: {exc}", flush=True)
                warnings_shown += 1

    print(f"Eligible unique candidates: {len(candidates)}; "
          f"scoring with {workers} process(es)", flush=True)
    tasks = [(row["path"], spatial_dir, reference_file, existing_only)
             for row in candidates]

    def collect(scored):
        nonlocal warnings_shown
        for candidate, (score, error) in zip(candidates, scored):
            if error is not None:
                category, message = error
                skips[category] += 1
                if warnings_shown < 10:
                    print(f"  SKIP {candidate['scenario']}: {message}", flush=True)
                    warnings_shown += 1
                continue
            row = {k: v for k, v in candidate.items() if k != "path"}
            row.update(fitness=float(score["fitness"]), score=score)
            rows.append(row)
            if len(rows) % 20 == 0:
                print(f"  Scored {len(rows)}/{len(candidates)} candidates", flush=True)

    # Baseline has already been evaluated, so all workers only read the common
    # sufficiency reference and attachment mask. Each writes its *own* folder.
    if workers == 1:
        collect(map(_score_worker, tasks))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            collect(pool.map(_score_worker, tasks, chunksize=1))

    if not rows:
        raise RuntimeError("No comparable scenarios could be scored. Check saved "
                           "accessibility/linkstats, baseline references, and the GA budget.")

    rows.sort(key=lambda x: (-x["fitness"], x["scenario"]))
    print(f"Loading baseline network to validate candidate seeds...", flush=True)
    settings = {**DEFAULT_SETTINGS, "budget_m": float(budget_m)}
    base = load_baseline(baseline_dir / "network.xml.gz",
                         baseline_dir / "transit_schedule.xml.gz", settings)
    selected = []
    for row in rows:
        if len(selected) == count:
            break
        if any(_segment_similarity(row["segments"], prev["segments"]) > max_overlap
               for prev in selected):
            continue
        individual = _individual_from_records(row["interventions"])
        valid, reason = validate_individual(
            individual, eligible=base.eligible, network=base.network,
            index=base.index, budget_m=budget_m,
        )
        if not valid:
            skips["GA invalid: " + reason] += 1
            continue
        selected.append(row)
    selected_hashes = {row["config_hash"] for row in selected}
    ranking_path = output.with_name(output.stem + "_ranking.csv")
    _write_ranking(ranking_path, rows, selected_hashes)

    payload = {
        "version": 1, "fitness_version": FITNESS_VERSION,
        "source_dataset": str(dataset), "baseline_name": BASELINE_NAME,
        "baseline_hashes": baseline_metadata.get("baseline_hashes"),
        "reference_sha256": hashlib.sha256(reference_file.read_bytes()).hexdigest(),
        "baseline_attachment_sha256": baseline_score["baseline_attachment_sha256"],
        "budget_m": budget_m, "selection_max_segment_overlap": max_overlap,
        "fitness_settings": {
            "mode_weights": MODE_WEIGHTS, "time_weights": {"15": 1.0},
            "sufficiency_level": SUFFICIENCY_LEVEL,
        },
        "historical_baseline_fitness": baseline_score["fitness"],
        "scored_candidates": len(rows), "skipped": dict(skips),
        "seeds": [
            {
                "scenario": row["scenario"], "strategy": row["strategy"],
                "config_hash": row["config_hash"], "fitness": row["fitness"],
                "budget_used_m": row["budget_used_m"],
                "interventions": row["interventions"],
            }
            for row in selected
        ],
    }
    _write_json(output, payload)
    print(f"Scored {len(rows)} comparable configurations; skipped {sum(skips.values())}.")
    print(f"Baseline fitness: {baseline_score['fitness']:.6f}")
    for i, row in enumerate(selected, 1):
        print(f"  {i:2d}. {row['scenario']:<24} fitness={row['fitness']:.6f}  "
              f"budget={row['budget_used_m']:.1f} m  [{row['strategy']}]")
    print(f"Warm-start file: {output}")
    print(f"Full ranking:    {ranking_path}")
    if len(selected) < count:
        print(f"Only {len(selected)}/{count} diverse, valid seeds found; "
              "ga.initial_population() can fill the remaining positions using samplers.")
    print("No MATSim simulations were run.")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--spatial-dir", type=Path, default=SPATIAL_DIR)
    parser.add_argument("--reference", type=Path, default=REFERENCE_FILE)
    parser.add_argument("--budget", type=float, default=BUDGET_M)
    parser.add_argument("--count", type=int, default=WARM_START_COUNT)
    parser.add_argument("--max-overlap", type=float, default=MAX_SEGMENT_OVERLAP)
    parser.add_argument("--limit", type=int, default=0, help="Scoring pilot: process only first N eligible scenarios (0=all)")
    parser.add_argument("--existing-only", action="store_true",
                        help="Only rescore scenarios with saved accessibility CSVs; do not route")
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel historical accessibility processes (default: 1)")
    args = parser.parse_args()
    if args.budget < 0 or args.count < 1 or not 0 <= args.max_overlap <= 1 or args.limit < 0 or args.workers < 1:
        parser.error("Budget must be nonnegative, count positive, overlap in [0,1], limit >= 0, workers >= 1")
    build_warm_start(args.dataset, args.output, args.spatial_dir, args.reference,
                     count=args.count, budget_m=args.budget,
                     max_overlap=args.max_overlap, limit=args.limit,
                     existing_only=args.existing_only, workers=args.workers)


if __name__ == "__main__":
    main()

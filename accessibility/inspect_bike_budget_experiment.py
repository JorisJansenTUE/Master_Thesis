"""Screen cycling-only road-space scenarios without running MATSim.

Reuse network_sampling's random/corridor/cluster samplers and the same bicycle
routing/linear sufficiency calculation used for thesis accessibility.

Run locally:
    python -m accessibility.inspect_bike_budget_experiment --workers 1
On Slurm, use hpc/GA_optimization/bike_budget_experiment.slurm.

Results are saved after EACH candidate. Re-running safely skips completed cases.
No network_sampling, optimization, or MATSim output files are modified.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

from accessibility.calculations.cumulative import CumulativeAccessibility
from accessibility.calculations.sufficiency import SufficiencyReference
from accessibility.inputs.grid.snapping import ScenarioRoadSnapper
from accessibility.inputs.links.network import BikeNetwork
from accessibility.routing.road import RoadRouter
from network_sampling.dataset.generate import DEFAULT_SETTINGS, load_baseline, sample
from network_sampling.interventions.intervention_types import InterventionType, RoadSpaceVariant
from network_sampling.interventions.matsim_app.apply_network import (
    BICYCLE_STREET_SPEED_M_S, DESTINATION_ONLY_SPEED_M_S, apply_interventions,
)
from network_sampling.network.matsim_network import write_matsim_network


# ============================================================================
# SETTINGS — edit these, or override environment variables in the Slurm script.
# ============================================================================
BASE = Path(os.environ.get("BASE", "/home/20201733/locarno_ETH_Model"))
DATASET_MANIFEST = Path(os.environ.get(
    "BIKE_DATASET_MANIFEST", str(BASE / "training_dataset_v4/dataset_manifest.json")))
# Normally read from DATASET_MANIFEST; set either Path explicitly to override.
BASELINE_NETWORK: Path | None = None
BASELINE_SCHEDULE: Path | None = None
SPATIAL_DIR = Path(os.environ.get("SPATIAL_DIR", str(BASE / "spatial_baseline")))
REFERENCE_FILE = Path(os.environ.get(
    "BIKE_REFERENCE_FILE", str(BASE / "GA_Optimization/references/sufficiency_reference_15min.json")))
OUTPUT_DIR = Path(os.environ.get("BIKE_EXPERIMENT_OUT", str(BASE / "Bike_Accessibility_Budgets")))

BUDGETS_KM = (3.0, 7.0, 15.0)
STRATEGIES = ("random", "corridor", "cluster")
SAMPLES_PER_STRATEGY_PER_BUDGET = 50  # 150 per budget, 450 total
SEED_BASE = 20261122
MAX_SAMPLE_ATTEMPTS = 12
THRESHOLD_MIN = 15.0
BIKE_SPEED_M_S = 4.17
ORIGIN_SNAP_M = 500.0
JOB_SNAP_M = 300.0
# Prefer PBL + keep_car where eligible; otherwise eligible bicycle street.
# No forced/ineligible interventions, bus lanes, or destination-only roads.
# ============================================================================

_WORKER = None


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _baseline_paths() -> tuple[Path, Path]:
    if BASELINE_NETWORK is not None and BASELINE_SCHEDULE is not None:
        return BASELINE_NETWORK.resolve(strict=True), BASELINE_SCHEDULE.resolve(strict=True)
    if BASELINE_NETWORK is not None or BASELINE_SCHEDULE is not None:
        raise ValueError("Set both BASELINE_NETWORK and BASELINE_SCHEDULE, or neither")
    data = json.loads(DATASET_MANIFEST.read_text(encoding="utf-8"))
    return (Path(data["baseline_network"]).resolve(strict=True),
            Path(data["baseline_schedule"]).resolve(strict=True))


def _cycling_only_baseline(base):
    # One preferred, eligible intervention per physical segment; non-cycling
    # interventions and car-removal variants are deliberately excluded.
    by_segment = {}
    for intervention in base.eligible:
        by_segment.setdefault(intervention.segment_id, []).append(intervention)
    selected = []
    for sid in sorted(by_segment):
        options = by_segment[sid]
        best_pbl = next((iv for iv in options
                         if iv.intervention_type == InterventionType.PROTECTED_BIKE_LANE
                         and iv.road_space_variant == RoadSpaceVariant.KEEP_CAR), None)
        best_street = next((iv for iv in options
                            if iv.intervention_type == InterventionType.BICYCLE_STREET), None)
        if best_pbl is not None or best_street is not None:
            selected.append(best_pbl if best_pbl is not None else best_street)
    if not selected:
        raise RuntimeError("No eligible PBL keep_car or bicycle-street interventions")
    print(f"Cycling-only eligible segments: {len(selected)} "
          f"(PBL keep_car={sum(iv.intervention_type == InterventionType.PROTECTED_BIKE_LANE for iv in selected)}, "
          f"bicycle street={sum(iv.intervention_type == InterventionType.BICYCLE_STREET for iv in selected)})",
          flush=True)
    return replace(base, eligible=selected)


def _bike_results(network: Path, origins: pd.DataFrame, jobs: pd.DataFrame) -> pd.DataFrame:
    bike = BikeNetwork.from_matsim_network(network, bike_speed_m_s=BIKE_SPEED_M_S)
    connections = ScenarioRoadSnapper(
        origins, jobs,
        max_origin_distance_m=ORIGIN_SNAP_M,
        max_job_distance_m=JOB_SNAP_M,
    ).attach(bike.graph)
    router = RoadRouter(
        bike.graph, connections.origins.connectors,
        connections.jobs.connectors, jobs,
    )
    return CumulativeAccessibility((THRESHOLD_MIN,)).run(
        router, origins["hex_id"].astype(str), mode="bike")


def _metrics(frame: pd.DataFrame, base_frame: pd.DataFrame, ref_jobs: float, denominator: int):
    column = CumulativeAccessibility.column_name(THRESHOLD_MIN)
    data = frame.set_index("hex_id").sort_index()
    base = base_frame.set_index("hex_id").sort_index()
    if not data.index.equals(base.index):
        raise ValueError("Baseline and scenario origin IDs differ")
    attached = data["attached"].astype(bool)
    before_attached = base["attached"].astype(bool)
    vals = pd.to_numeric(data.loc[attached, column], errors="raise").to_numpy(dtype=float)
    if not np.isfinite(vals).all() or (vals < 0).any():
        raise ValueError("Invalid accessibility for attached bicycle origin")
    score = float(np.minimum(vals / ref_jobs, 1.0).sum() / denominator)
    paired = attached & before_attached
    differences = (data.loc[paired, column] - base.loc[paired, column]).to_numpy(dtype=float)
    return {
        "bike_score": score,
        "mean_jobs_attached": float(vals.mean()) if len(vals) else None,
        "attached_origins": int(attached.sum()),
        "new_attached_origins": int((attached & ~before_attached).sum()),
        "lost_attached_origins": int((~attached & before_attached).sum()),
        "paired_origins_improved": int((differences > 1e-9).sum()),
        "paired_origins_worsened": int((differences < -1e-9).sum()),
        "paired_origins_unchanged": int((abs(differences) <= 1e-9).sum()),
        "mean_change_jobs_paired": float(differences.mean()) if len(differences) else None,
    }


def _task_key(budget: float, strategy: str, index: int) -> str:
    return f"{budget:g}km_{strategy}_{index:04d}"


def _task_seed(budget_index: int, strategy_index: int, sample_index: int) -> int:
    return SEED_BASE + budget_index * 1_000_000 + strategy_index * 100_000 + sample_index


def _tasks() -> list[tuple[float, str, int, int]]:
    return [(budget, strategy, idx, _task_seed(bi, si, idx))
            for bi, budget in enumerate(BUDGETS_KM)
            for si, strategy in enumerate(STRATEGIES)
            for idx in range(SAMPLES_PER_STRATEGY_PER_BUDGET)]


def _run_candidate(task: tuple[float, str, int, int]) -> dict:
    if _WORKER is None:
        raise RuntimeError("Worker context not initialised; Linux fork is required for parallel mode")
    base, schedule, origins, jobs, baseline_access, reference_jobs, denominator, base_score = _WORKER
    budget_km, strategy, index, seed = task
    settings = dict(DEFAULT_SETTINGS)
    settings["budget_m"] = 1000.0 * budget_km
    settings["corridor_bus_share"] = 0.0
    settings["corridor_destination_share"] = 0.0
    scenario = None
    attempts = 0
    last_reason = None
    for attempts in range(1, MAX_SAMPLE_ATTEMPTS + 1):
        current_seed = seed + (attempts - 1) * 1_000_003
        scenario = sample(strategy, current_seed, base, settings)
        if scenario:
            break
        last_reason = "Sampler produced no eligible cycling interventions"
    if not scenario:
        return {"key": _task_key(budget_km, strategy, index),
                "budget_km": budget_km, "strategy": strategy, "index": index,
                "seed": seed, "status": "no_sample", "reason": last_reason,
                "attempts": attempts}
    actual_budget_m = sum(float(base.index.get(iv.segment_id).length_m) for iv in scenario)
    if actual_budget_m > budget_km * 1000.0 + 1e-6:
        raise AssertionError("Sampler exceeded requested budget")
    if len({iv.segment_id for iv in scenario}) != len(scenario):
        raise AssertionError("Multiple interventions selected for the same segment")
    modified, bus_mapping = apply_interventions(base.network, base.index, scenario)
    if bus_mapping:
        raise AssertionError("Unexpected dedicated bus lanes in cycling-only experiment")
    with tempfile.TemporaryDirectory(prefix="bike_budget_") as tmp:
        network_file = Path(tmp) / "network.xml.gz"
        write_matsim_network(modified, network_file)
        access = _bike_results(network_file, origins, jobs)
    metrics = _metrics(access, baseline_access, reference_jobs, denominator)
    metrics["bike_score_delta"] = metrics["bike_score"] - base_score
    metrics["bike_score_delta_pct"] = (100.0 * (metrics["bike_score"] / base_score - 1)
                                       if base_score > 0 else None)
    return {
        "key": _task_key(budget_km, strategy, index),
        "budget_km": budget_km, "strategy": strategy, "index": index,
        "seed": seed, "effective_seed": current_seed, "attempts": attempts,
        "status": "ok", "used_budget_km": actual_budget_m / 1000.0,
        "intervention_count": len(scenario),
        "pbl_count": sum(iv.intervention_type == InterventionType.PROTECTED_BIKE_LANE for iv in scenario),
        "bicycle_street_count": sum(iv.intervention_type == InterventionType.BICYCLE_STREET for iv in scenario),
        **metrics,
        "interventions": [{"segment_id": str(iv.segment_id),
                           "intervention_type": iv.intervention_type.value,
                           "road_space_variant": iv.road_space_variant.value if iv.road_space_variant else None,
                           "target_link_ids": list(iv.target_link_ids)} for iv in scenario],
    }


def _save_summary(folder: Path) -> None:
    records = [json.loads(path.read_text(encoding="utf-8"))
               for path in sorted((folder / "results").glob("*.json"))]
    if not records:
        print("No completed results yet", flush=True)
        return
    table = pd.DataFrame([{k: v for k, v in row.items() if k != "interventions"}
                          for row in records]).sort_values(["budget_km", "strategy", "index"])
    table.to_csv(folder / "all_candidates.csv", index=False)
    good = table.loc[table["status"] == "ok"].copy()
    print(f"\nCandidates: {len(table)} total, {len(good)} valid", flush=True)
    if len(good):
        grouped = (good.groupby(["budget_km", "strategy"], sort=True)
                   .agg(n=("bike_score", "size"),
                        best_score=("bike_score", "max"),
                        mean_score=("bike_score", "mean"),
                        median_score=("bike_score", "median"),
                        mean_used_km=("used_budget_km", "mean"),
                        best_gain_pct=("bike_score_delta_pct", "max"))
                   .reset_index())
        grouped.to_csv(folder / "summary_by_budget_strategy.csv", index=False)
        print(grouped.round(5).to_string(index=False), flush=True)
        top = good.nlargest(10, "bike_score")
        print("\nTop 10 cycling-only candidates:", flush=True)
        print(top[["key", "used_budget_km", "bike_score", "bike_score_delta_pct",
                   "pbl_count", "bicycle_street_count"]].to_string(index=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    parser.add_argument("--max-candidates", type=int, default=None,
                        help="Optional quick test: evaluate only first N candidates")
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be >= 1")
    out = OUTPUT_DIR.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    if args.summary_only:
        _save_summary(out)
        return

    network, schedule = _baseline_paths()
    origins_file = (SPATIAL_DIR / "origins.csv").resolve(strict=True)
    jobs_file = (SPATIAL_DIR / "jobs.csv").resolve(strict=True)
    reference_file = REFERENCE_FILE.resolve(strict=True)
    reference = SufficiencyReference.load(reference_file)
    reference_jobs = reference.threshold_jobs(CumulativeAccessibility.column_name(THRESHOLD_MIN), "high")
    if reference_jobs <= 0:
        raise ValueError("Nonpositive baseline car sufficiency reference")
    identity = {
        "experiment": "bike_budget_screen_v1",
        "network": str(network), "network_sha256": _sha(network),
        "schedule": str(schedule), "schedule_sha256": _sha(schedule),
        "origins": str(origins_file), "origins_sha256": _sha(origins_file),
        "jobs": str(jobs_file), "jobs_sha256": _sha(jobs_file),
        "reference": str(reference_file), "reference_sha256": _sha(reference_file),
        "budgets_km": BUDGETS_KM, "strategies": STRATEGIES,
        "samples_per_strategy_per_budget": SAMPLES_PER_STRATEGY_PER_BUDGET,
        "seed_base": SEED_BASE, "max_attempts": MAX_SAMPLE_ATTEMPTS,
        "threshold_min": THRESHOLD_MIN, "bike_speed_m_s": BIKE_SPEED_M_S,
        "origin_snap_m": ORIGIN_SNAP_M, "job_snap_m": JOB_SNAP_M,
        "bike_street_speed_m_s": BICYCLE_STREET_SPEED_M_S,
        "destination_only_speed_m_s": DESTINATION_ONLY_SPEED_M_S,
        "preferred_interventions": "pbl_keep_car_else_bicycle_street_eligible",
        "normalization": "baseline_attached_origins",
    }
    manifest = out / "experiment_manifest.json"
    if manifest.exists():
        existing = json.loads(manifest.read_text(encoding="utf-8"))
        if existing != json.loads(json.dumps(identity)):
            raise RuntimeError(f"Experiment inputs/settings changed. Use a new OUTPUT_DIR: {out}")
    else:
        if any((out / "results").glob("*.json")):
            raise RuntimeError("Results exist without an experiment manifest; use a new output directory")
        _atomic_json(manifest, identity)
    (out / "results").mkdir(exist_ok=True)

    base = _cycling_only_baseline(load_baseline(network, schedule, dict(DEFAULT_SETTINGS)))
    origins = pd.read_csv(origins_file, dtype={"hex_id": str})
    jobs = pd.read_csv(jobs_file, dtype={"job_id": str})
    print("Calculating baseline bicycle accessibility once...", flush=True)
    baseline_access = _bike_results(network, origins, jobs)
    denominator = int(baseline_access["attached"].astype(bool).sum())
    if denominator == 0:
        raise ValueError("Baseline contains no bicycle-attached origins")
    baseline_metrics = _metrics(baseline_access, baseline_access, reference_jobs, denominator)
    _atomic_json(out / "baseline_metrics.json", {
        **baseline_metrics, "baseline_attached_denominator": denominator,
        "total_origins": len(baseline_access), "reference_jobs": reference_jobs,
    })
    print(f"Baseline bicycle score={baseline_metrics['bike_score']:.6f}; "
          f"baseline-attached origins={denominator}/{len(baseline_access)}; "
          f"baseline-car reference={reference_jobs:.1f} jobs", flush=True)

    global _WORKER
    _WORKER = (base, schedule, origins, jobs, baseline_access, reference_jobs,
               denominator, baseline_metrics["bike_score"])
    tasks = _tasks()
    if args.max_candidates is not None:
        tasks = tasks[:args.max_candidates]
    pending = [task for task in tasks
               if not (out / "results" / (_task_key(task[0], task[1], task[2]) + ".json")).is_file()]
    print(f"Experiment: {len(tasks)} planned, {len(pending)} remaining; "
          f"workers={args.workers}; output={out}", flush=True)

    if args.workers == 1:
        for index, task in enumerate(pending, 1):
            record = _run_candidate(task)
            _atomic_json(out / "results" / (record["key"] + ".json"), record)
            print(f"[{index}/{len(pending)}] {record['key']} "
                  f"{record['status']} score={record.get('bike_score', float('nan')):.6f}", flush=True)
    else:
        if os.name != "posix":
            raise RuntimeError("Parallel workers require Linux fork; use --workers 1 locally")
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("fork")) as executor:
            future_map = {executor.submit(_run_candidate, task): task for task in pending}
            for index, future in enumerate(as_completed(future_map), 1):
                record = future.result()  # Fail visibly on unexpected implementation errors.
                _atomic_json(out / "results" / (record["key"] + ".json"), record)
                print(f"[{index}/{len(pending)}] {record['key']} "
                      f"{record['status']} score={record.get('bike_score', float('nan')):.6f}", flush=True)
    _save_summary(out)


if __name__ == "__main__":
    main()

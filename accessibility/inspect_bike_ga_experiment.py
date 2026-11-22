"""MATSim-free genetic-algorithm experiment for bicycle employment accessibility.

Uses the thesis's existing GA mutation/crossover/repair and network_sampling
samplers; fitness is the 15-min bicycle sufficiency score computed from the
modified network and fixed facilities-based employment opportunities.

Run from the repository root:
    python -m accessibility.inspect_bike_ga_experiment --workers 1
    python -m accessibility.inspect_bike_ga_experiment --workers 8 --max-generations 1

Every evaluated configuration is saved immediately, allowing interrupted
SLURM jobs to resume without rerouting completed individuals. No MATSim simulations, Java validator, or production GA checkpoints are used.
Eligible variants and directed road connectivity are validated in Python.

This module reuses the scoring helpers in inspect_bike_budget_experiment.py;
install BOTH Python files from the supplied package.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
import hashlib
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import random
import tempfile

import pandas as pd

from accessibility.calculations.cumulative import CumulativeAccessibility
from accessibility.calculations.sufficiency import SufficiencyReference
from accessibility.inspect_bike_budget_experiment import (
    _atomic_json, _bike_results, _metrics, _sha,
)
from network_sampling.dataset.generate import DEFAULT_SETTINGS, load_baseline, sample
from network_sampling.interventions.intervention_types import InterventionType
from network_sampling.interventions.matsim_app.apply_network import (
    BICYCLE_STREET_SPEED_M_S, apply_interventions,
)
from network_sampling.network.matsim_network import write_matsim_network
from optimization.ga import (
    configuration_hash, crossover, mutate, validate_individual,
)


# ===========================================================================
# EDIT SETTINGS HERE. HPC paths are inherited from hpc_config.sh when present.
# ===========================================================================
BASE = Path(os.environ.get("BASE", "/home/20201733/locarno_ETH_Model"))
DATASET_MANIFEST = Path(os.environ.get(
    "BIKE_DATASET_MANIFEST", str(BASE / "training_dataset_v4/dataset_manifest.json")))
BASELINE_NETWORK: Path | None = None     # Read from DATASET_MANIFEST if None
BASELINE_SCHEDULE: Path | None = None    # Read from DATASET_MANIFEST if None
SPATIAL_DIR = Path(os.environ.get("SPATIAL_DIR", str(BASE / "spatial_baseline")))
REFERENCE_FILE = Path(os.environ.get(
    "BIKE_REFERENCE_FILE", str(BASE / "GA_Optimization/references/sufficiency_reference_15min.json")))
OUTPUT_DIR = Path(os.environ.get("BIKE_GA_OUT", str(BASE / "Bike_Accessibility_GA_AllBike")))

BUDGETS_KM = (15.0)
POPULATION_SIZE = 100
GENERATIONS = 80                # Includes generation 0; up to 24,000 evaluations over 3 budgets
SEED = int(os.environ.get("BIKE_GA_SEED", "20261122"))
ELITES = 2
TOURNAMENT_SIZE = 3
CROSSOVER_PROBABILITY = 0.75
MUTATION_PROBABILITY = 0.85
IMMIGRANT_FRACTION = 0.10
# A budget is a MAXIMUM, not a target. Seed a range of configuration sizes.
INITIAL_BUDGET_FRACTIONS = (0.10, 0.25, 0.50, 1.00)
MAX_SAMPLE_ATTEMPTS_PER_SLOT = 100

THRESHOLD_MIN = 15.0
BIKE_SPEED_M_S = 4.17
ORIGIN_SNAP_M = 500.0
JOB_SNAP_M = 300.0
# All eligible bicycle streets and ALL eligible PBL variants (KEEP_CAR,
# REMOVE_ONE_LANE, REMOVE_CAR), using the existing width/budget/car-bike-truck
# connectivity and compatibility rules. No bus lanes or destination-only.
# No Java connectivity check / MATSim simulation: this is a fast diagnostic GA.
# ===========================================================================

_WORKER = None


def _all_eligible_bike_baseline(base):
    """Retain every cycling option produced by the real eligibility rules.

    Do NOT collapse alternatives to one PBL variant per street: this experiment
    needs to explore both low- and high-car-impact cycling infrastructure.
    Existing GA validation enforces budget, compatibility and directed modal
    connectivity for each complete individual.
    """
    bike_types = {
        InterventionType.PROTECTED_BIKE_LANE,
        InterventionType.BICYCLE_STREET,
    }
    selected = [iv for iv in base.eligible if iv.intervention_type in bike_types]
    if not selected:
        raise RuntimeError("No eligible cycling interventions in the baseline")
    from collections import Counter
    counts = Counter(
        (iv.intervention_type.value,
         iv.road_space_variant.value if iv.road_space_variant else "none")
        for iv in selected
    )
    print(f"Eligible cycling gene options: {len(selected)} "
          f"across {len({iv.segment_id for iv in selected})} segments", flush=True)
    for (intervention_type, variant), count in sorted(counts.items()):
        print(f"  {intervention_type}/{variant}: {count}", flush=True)
    return replace(base, eligible=selected)


def _paths() -> tuple[Path, Path]:
    if BASELINE_NETWORK is not None or BASELINE_SCHEDULE is not None:
        if BASELINE_NETWORK is None or BASELINE_SCHEDULE is None:
            raise ValueError("Set both BASELINE_NETWORK and BASELINE_SCHEDULE")
        return BASELINE_NETWORK.resolve(strict=True), BASELINE_SCHEDULE.resolve(strict=True)
    data = json.loads(DATASET_MANIFEST.read_text(encoding="utf-8"))
    return (Path(data["baseline_network"]).resolve(strict=True),
            Path(data["baseline_schedule"]).resolve(strict=True))


def _genes(individual, index: dict) -> list[int]:
    return sorted(index[iv] for iv in individual)


def _individual(genes: list[int], eligible: list):
    return [eligible[int(i)] for i in genes]


def _as_candidate(individual, index: dict, origin: str) -> dict:
    return {"key": configuration_hash(individual),
            "genes": _genes(individual, index), "origin": origin}


def _sample_population(base, *, budget_m: float, count: int, seen: set,
                       rng: random.Random, index: dict) -> list[dict]:
    """Seed and immigrate with varied lengths and existing samplers.

    Each sample is validated by optimization.ga.validate_individual; neither
    the Java validator nor the regular MATSim generator is called.
    """
    candidates = []
    strategies = ("random", "corridor", "cluster")
    max_attempts = MAX_SAMPLE_ATTEMPTS_PER_SLOT * count
    for attempt in range(max_attempts):
        if len(candidates) >= count:
            return candidates
        strategy = strategies[(attempt // len(INITIAL_BUDGET_FRACTIONS)) % len(strategies)]
        fraction = INITIAL_BUDGET_FRACTIONS[attempt % len(INITIAL_BUDGET_FRACTIONS)]
        settings = dict(DEFAULT_SETTINGS)
        settings.update(budget_m=budget_m * fraction,
                        corridor_bus_share=0.0, corridor_destination_share=0.0)
        try:
            individual = sample(strategy, rng.randrange(2**31), base, settings)
        except (ValueError, RuntimeError):
            continue
        if not individual:
            continue
        key = configuration_hash(individual)
        if key in seen:
            continue
        ok, _ = validate_individual(
            individual, eligible=base.eligible, network=base.network,
            index=base.index, budget_m=budget_m)
        if not ok:
            continue
        seen.add(key)
        candidates.append(_as_candidate(individual, index, f"sample:{strategy}:{fraction:g}"))
    raise RuntimeError(f"Only generated {len(candidates)}/{count} valid unique samples "
                       f"with max budget {budget_m/1000:g} km")


def _new_generation(generation: int, previous: dict | None, *,
                    folder: Path, budget_m: float, base, iv_index: dict) -> dict:
    rng = random.Random(f"bike-ga:{SEED}:{budget_m:.0f}:{generation}")
    if previous is None:
        population = _sample_population(base, budget_m=budget_m,
                                        count=POPULATION_SIZE, seen=set(),
                                        rng=rng, index=iv_index)
        counters = {"initial_samples": len(population)}
    else:
        old = previous["population"]
        def score(candidate):
            path = folder / "evaluations" / (candidate["key"] + ".json")
            if not path.is_file():
                raise RuntimeError(f"Previous generation has unscored individual: {path}")
            return float(json.loads(path.read_text(encoding="utf-8"))["bike_score"])
        fitness = {candidate["key"]: score(candidate) for candidate in old}
        ranked = sorted(old, key=lambda c: (-fitness[c["key"]], c["key"]))
        population = [dict(c, origin="elite") for c in ranked[:ELITES]]
        seen = {c["key"] for c in population}
        immigrant_slots = math.ceil(POPULATION_SIZE * IMMIGRANT_FRACTION)
        goal = POPULATION_SIZE - immigrant_slots
        counters = {"elites": len(population), "ga_new": 0,
                    "ga_attempts": 0, "repair_failures": 0, "duplicates": 0,
                    "immigrants_or_fallback": 0}

        def tournament():
            picked = rng.sample(old, min(TOURNAMENT_SIZE, len(old)))
            return max(picked, key=lambda c: (fitness[c["key"]], c["key"]))

        while len(population) < goal and counters["ga_attempts"] < POPULATION_SIZE * 100:
            counters["ga_attempts"] += 1
            left = _individual(tournament()["genes"], base.eligible)
            try:
                if rng.random() < CROSSOVER_PROBABILITY:
                    right = _individual(tournament()["genes"], base.eligible)
                    child = crossover(left, right, eligible=base.eligible,
                                      network=base.network, index=base.index,
                                      budget_m=budget_m, rng=rng)
                else:
                    child = left
                if rng.random() < MUTATION_PROBABILITY:
                    child = mutate(child, eligible=base.eligible,
                                   network=base.network, index=base.index,
                                   budget_m=budget_m, rng=rng)
            except ValueError as exc:
                if str(exc) != "Cannot repair offspring with current operators":
                    raise
                counters["repair_failures"] += 1
                continue
            key = configuration_hash(child)
            if key in seen:
                counters["duplicates"] += 1
                continue
            seen.add(key)
            population.append(_as_candidate(child, iv_index, "ga"))
            counters["ga_new"] += 1

        extras = POPULATION_SIZE - len(population)
        if extras:
            immigration = _sample_population(base, budget_m=budget_m,
                                             count=extras, seen=seen,
                                             rng=rng, index=iv_index)
            population.extend(immigration)
            counters["immigrants_or_fallback"] = len(immigration)

    if len(population) != POPULATION_SIZE:
        raise AssertionError("Incorrect GA population size")
    return {"generation": generation, "budget_km": budget_m/1000,
            "population": population, "operators": counters}


def _evaluate(task: dict) -> dict:
    if _WORKER is None:
        raise RuntimeError("Worker context not initialised")
    base, origins, jobs, baseline_frame, reference_jobs, denominator, base_score = _WORKER
    individual = _individual(task["genes"], base.eligible)
    modified, bus_mapping = apply_interventions(base.network, base.index, individual)
    if bus_mapping:
        raise AssertionError("Bike-only GA unexpectedly modified bus routes")
    with tempfile.TemporaryDirectory(prefix="bike_ga_") as tmp:
        path = Path(tmp) / "network.xml.gz"
        write_matsim_network(modified, path)
        accessibility = _bike_results(path, origins, jobs)
    metrics = _metrics(accessibility, baseline_frame, reference_jobs, denominator)
    budget_m = sum(base.index.get(iv.segment_id).length_m for iv in individual)
    result = {"key": task["key"], "genes": task["genes"],
              "used_budget_km": float(budget_m)/1000,
              "intervention_count": len(individual),
              **metrics}
    result["bike_score_delta"] = result["bike_score"] - base_score
    result["bike_score_delta_pct"] = (100 * (result["bike_score"] / base_score - 1)
                                      if base_score else None)
    if not math.isfinite(result["bike_score"]):
        raise ValueError("Nonfinite bicycle fitness")
    return result


def _evaluate_generation(folder: Path, generation: dict, workers: int) -> None:
    eval_dir = folder / "evaluations"
    eval_dir.mkdir(exist_ok=True)
    tasks = [candidate for candidate in generation["population"]
             if not (eval_dir / (candidate["key"] + ".json")).is_file()]
    print(f"  Generation {generation['generation']:02d}: "
          f"{len(tasks)} new accessibility evaluations; "
          f"{POPULATION_SIZE-len(tasks)} reused", flush=True)
    if workers == 1:
        for candidate in tasks:
            result = _evaluate(candidate)
            _atomic_json(eval_dir / (result["key"] + ".json"), result)
    else:
        if os.name != "posix":
            raise RuntimeError("Parallel operation requires Linux fork; use --workers 1 locally")
        with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("fork")) as pool:
            pending = {pool.submit(_evaluate, task): task for task in tasks}
            for future in as_completed(pending):
                result = future.result()  # Don't confuse code errors with poor fitness.
                _atomic_json(eval_dir / (result["key"] + ".json"), result)


def _summarise(folder: Path, base, budget_km: float) -> None:
    rows = []
    all_results = {}
    for file in (folder / "evaluations").glob("*.json"):
        result = json.loads(file.read_text(encoding="utf-8"))
        all_results[result["key"]] = result
    seen_earlier = set()
    for file in sorted(folder.glob("generation_*.json")):
        generation = json.loads(file.read_text(encoding="utf-8"))
        records = [all_results.get(candidate["key"]) for candidate in generation["population"]]
        if any(record is None for record in records):
            continue
        scored = [float(record["bike_score"]) for record in records]
        novel = [record for candidate, record in zip(generation["population"], records)
                 if candidate["key"] not in seen_earlier]
        seen_earlier.update(candidate["key"] for candidate in generation["population"])
        mean_budget = sum(float(record["used_budget_km"]) for record in records) / len(records)
        rows.append({"generation": generation["generation"],
                     "best_fitness": max(scored),
                     "mean_fitness": sum(scored)/len(scored),
                     "best_new_fitness": max((r["bike_score"] for r in novel), default=None),
                     "mean_used_km": mean_budget,
                     "unique_individuals": len({c["key"] for c in generation["population"]}),
                     "new_evaluations": len(novel),
                     **generation["operators"]})
    if rows:
        pd.DataFrame(rows).to_csv(folder / "history.csv", index=False)
        print(f"  Completed generations={len(rows)} "
              f"best={max(row['best_fitness'] for row in rows):.6f} "
              f"latest_mean={rows[-1]['mean_fitness']:.6f}", flush=True)

    if all_results:
        best = max(all_results.values(), key=lambda r: (r["bike_score"], r["key"]))
        individual = _individual(best["genes"], base.eligible)
        _atomic_json(folder / "best_interventions.json", {
            "budget_limit_km": budget_km, "bike_score": best["bike_score"],
            "improvement_pct": best["bike_score_delta_pct"],
            "used_budget_km": best["used_budget_km"],
            "interventions": [
                {"segment_id": str(iv.segment_id),
                 "intervention_type": iv.intervention_type.value,
                 "road_space_variant": iv.road_space_variant.value if iv.road_space_variant else None,
                 "target_link_ids": [str(v) for v in iv.target_link_ids]}
                for iv in individual
            ],
        })
        modified, _ = apply_interventions(base.network, base.index, individual)
        write_matsim_network(modified, folder / "best_network.xml.gz")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    parser.add_argument("--max-generations", type=int, default=GENERATIONS,
                        help="Evaluate only this many generations per budget (resume later)")
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()
    if args.workers < 1 or not 1 <= args.max_generations <= GENERATIONS:
        parser.error("Invalid worker count or --max-generations")

    out = OUTPUT_DIR.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    if args.summary_only:
        for budget in BUDGETS_KM:
            folder = out / f"budget_{budget:g}km"
            if folder.is_dir():
                # Requires baseline to reconstruct interventions; simply show history here.
                path = folder / "history.csv"
                if path.is_file():
                    print(f"{folder.name}:\n{pd.read_csv(path).tail(4).to_string(index=False)}")
        return

    network, schedule = _paths()
    origins_file = (SPATIAL_DIR / "origins.csv").resolve(strict=True)
    jobs_file = (SPATIAL_DIR / "jobs.csv").resolve(strict=True)
    ref_file = REFERENCE_FILE.resolve(strict=True)
    reference = SufficiencyReference.load(ref_file)
    ref_jobs = reference.threshold_jobs(CumulativeAccessibility.column_name(THRESHOLD_MIN), "high")
    if not math.isfinite(ref_jobs) or ref_jobs <= 0:
        raise ValueError("Invalid baseline car sufficiency reference")

    identity = {
        "version": "bike_ga_v2_all_eligible_bike", "network": str(network), "network_sha256": _sha(network),
        "schedule": str(schedule), "schedule_sha256": _sha(schedule),
        "origins": str(origins_file), "origins_sha256": _sha(origins_file),
        "jobs": str(jobs_file), "jobs_sha256": _sha(jobs_file),
        "reference": str(ref_file), "reference_sha256": _sha(ref_file),
        "budgets_km": BUDGETS_KM, "population": POPULATION_SIZE,
        "generations": GENERATIONS, "seed": SEED, "elites": ELITES,
        "tournament": TOURNAMENT_SIZE, "crossover": CROSSOVER_PROBABILITY,
        "mutation": MUTATION_PROBABILITY, "immigrant_fraction": IMMIGRANT_FRACTION,
        "initial_budget_fractions": INITIAL_BUDGET_FRACTIONS,
        "speed_m_s": BIKE_SPEED_M_S, "threshold_min": THRESHOLD_MIN,
        "origin_snap_m": ORIGIN_SNAP_M, "job_snap_m": JOB_SNAP_M,
        "bicycle_street_speed_m_s": BICYCLE_STREET_SPEED_M_S,
        "selection": "all_baseline_eligible_pbl_variants_and_bicycle_streets",
        "normalisation": "fixed_baseline_bicycle_attached_origin_denominator",
    }
    # Freeze the eligible-gene mapping: generation checkpoints store gene indices.
    base = _all_eligible_bike_baseline(
        load_baseline(network, schedule, dict(DEFAULT_SETTINGS))
    )
    eligible = sorted(base.eligible, key=lambda iv: (
        str(iv.segment_id), iv.intervention_type.value,
        iv.road_space_variant.value if iv.road_space_variant else "",
        tuple(str(i) for i in iv.target_link_ids)))
    base = replace(base, eligible=eligible)
    iv_index = {iv: index for index, iv in enumerate(eligible)}
    eligible_data = [[str(iv.segment_id), iv.intervention_type.value,
                      iv.road_space_variant.value if iv.road_space_variant else None,
                      list(map(str, iv.target_link_ids))] for iv in eligible]
    identity["eligible_gene_sha256"] = hashlib.sha256(
        json.dumps(eligible_data, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    identity = json.loads(json.dumps(identity))
    manifest = out / "experiment_manifest.json"
    if manifest.exists():
        if json.loads(manifest.read_text(encoding="utf-8")) != identity:
            raise RuntimeError(f"Experiment settings or inputs changed: use a new output directory: {out}")
    else:
        if any(out.iterdir()):
            raise RuntimeError("Nonempty output directory without experiment_manifest.json")
        _atomic_json(manifest, identity)

    origins = pd.read_csv(origins_file, dtype={"hex_id": str})
    jobs = pd.read_csv(jobs_file, dtype={"job_id": str})
    baseline = _bike_results(network, origins, jobs)
    denominator = int(baseline["attached"].astype(bool).sum())
    if denominator <= 0:
        raise RuntimeError("Baseline has no attached bicycle origins")
    base_metrics = _metrics(baseline, baseline, ref_jobs, denominator)
    _atomic_json(out / "baseline_metrics.json", {
        **base_metrics, "baseline_attached_origins": denominator,
        "all_origins": len(baseline), "car_reference_jobs": ref_jobs,
    })
    print(f"Baseline bicycle fitness={base_metrics['bike_score']:.6f}; "
          f"attached={denominator}/{len(baseline)}", flush=True)

    global _WORKER
    _WORKER = (base, origins, jobs, baseline, ref_jobs,
               denominator, base_metrics["bike_score"])
    for budget_km in BUDGETS_KM:
        folder = out / f"budget_{budget_km:g}km"
        folder.mkdir(exist_ok=True)
        budget_m = budget_km * 1000.0
        print(f"\n=== BUDGET {budget_km:g} km: population={POPULATION_SIZE}, "
              f"generations={args.max_generations} ===", flush=True)
        previous = None
        for generation in range(args.max_generations):
            path = folder / f"generation_{generation:03d}.json"
            if path.is_file():
                current = json.loads(path.read_text(encoding="utf-8"))
                if current["generation"] != generation or current["budget_km"] != budget_km:
                    raise RuntimeError(f"Bad saved generation manifest: {path}")
            else:
                current = _new_generation(generation, previous, folder=folder,
                                          budget_m=budget_m, base=base,
                                          iv_index=iv_index)
                _atomic_json(path, current)
            _evaluate_generation(folder, current, args.workers)
            previous = current
            _summarise(folder, base, budget_km)

    print(f"\nResults saved to {out}", flush=True)


if __name__ == "__main__":
    main()

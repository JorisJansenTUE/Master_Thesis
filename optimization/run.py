"""Restartable, generation-by-generation GA controller for MATSim road-space scenarios.

From the thesis repository root:
    python -m optimization.run init --population 6 --generations 2 --dry-run
    python -m optimization.run init --population 40 --generations 10
    python -m optimization.run status
    # Submit generation_NNN.txt on the HPC via optimization/submit_generation.sh.
    # After simulation outputs are available locally:
    python -m optimization.run step

The controller never executes MATSim or submits jobs. The existing SLURM
script executes an explicit manifest; this controller collects completed
fitness results, selects offspring, and prepares the next generation.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
from pathlib import Path

from network_sampling.dataset.generate import DEFAULT_SETTINGS, load_baseline, sample
from network_sampling.validation.matsim_connectivity import MatsimConnectivityError
from optimization.ga import (
    build_spatial_context, configuration_hash, crossover, initial_population,
    mutate, validate_individual,
)
from optimization.prepare_scenario import prepare_one
from optimization import warm_start as warm
from optimization.fitness_trial08 import FITNESS_VERSION, SETTINGS, validate_reference, scenario_hashes
from concurrent.futures import ProcessPoolExecutor, as_completed

# -------------------- Editable local settings -----------------------------
DATASET = Path("data/GA_Trial08")
BASELINE = Path("data/Simulated_Runs_v2/baseline_00000")
WARM_START = DATASET / "warm_start.json"
SPATIAL_DIR = warm.SPATIAL_DIR
REFERENCE_FILE = warm.REFERENCE_FILE
BUDGET_M = 50000.0
POPULATION_SIZE = 80
GENERATIONS = 80               # Total, including generation 0
SEED = 1234
ELITES = 2
IMMIGRANT_FRACTION = 0.10
TOURNAMENT_SIZE = 3
CROSSOVER_PROBABILITY = 0.40  # Prefer local improvements over mixing patches
MUTATION_PROBABILITY = 0.80
OPERATOR_VERSION = "spatial_v1_multistep_trial08"
MUTATION_STEPS = (1, 2, 4, 8, 12)
PREFLIGHT_MAX_REPLACEMENTS = 80  # Bound resampling per invalid candidate
# Used by existing HPC overnight runner, which invokes `step` without jar args.
MATSIM_JAR = Path(os.environ.get(
    "MATSIM_JAR", "/home/20201733/eqasim-java-Fork/switzerland/target/switzerland.jar"
))
# --------------------------------------------------------------------------

STATE_FILE = "optimization_state.json"
REQUIRED_TARGETS = (
    "linkstats.txt.gz", "output_detailed_link_statistics.csv.gz", "modestats.csv",
)
SIMULATION_SETTINGS = (
    "last_iteration", "linkstats_interval", "linkstats_average_iterations",
)


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _mask_path(reference: Path) -> Path:
    return reference.with_name(reference.stem + "_attachments.json")


def _fitness_settings() -> dict:
    """Freeze scoring choices so a resumed GA cannot mix objectives."""
    return SETTINGS


def _records(individual) -> list[dict]:
    """Store canonical interventions (no new intervention data model)."""
    return [
        dict(
            segment_id=str(iv.segment_id),
            intervention_type=iv.intervention_type.value,
            road_space_variant=(iv.road_space_variant.value if iv.road_space_variant else None),
            target_link_ids=sorted(str(link) for link in iv.target_link_ids),
        )
        for iv in sorted(individual, key=lambda iv: (
            str(iv.segment_id), iv.intervention_type.value,
            iv.road_space_variant.value if iv.road_space_variant else "",
            tuple(sorted(str(x) for x in iv.target_link_ids)),
        ))
    ]


def _individual(records):
    # The warm-start module already converts the established JSON schema.
    return warm._individual_from_records(records)


def _validate_inputs(baseline: Path, reference: Path, warm_file: Path | None,
                     budget: float) -> tuple[dict, dict, dict | None]:
    metadata = _json(baseline / "scenario.json")
    if not metadata.get("baseline_hashes"):
        raise ValueError("Baseline scenario lacks source hashes")
    status = _json(baseline / "run_status.json")
    if status.get("state", status.get("status")) not in ("completed", "complete"):
        raise RuntimeError("The reference baseline MATSim simulation is not completed")
    if not reference.is_file() or not _mask_path(reference).is_file():
        raise FileNotFoundError(
            f"Baseline reference or attachment mask missing. Evaluate {baseline.name} "
            "with optimization.inspect_fitness first."
        )
    warm_payload = None
    if warm_file is not None:
        warm_payload = _json(warm_file)
        if warm_payload.get("fitness_version") != FITNESS_VERSION:
            raise ValueError("Warm start was scored with another fitness version")
        if warm_payload.get("baseline_hashes") != metadata["baseline_hashes"]:
            raise ValueError("Warm-start and current baseline MATSim networks differ")
        if float(warm_payload.get("budget_m", -1)) != float(budget):
            raise ValueError("Warm-start budget differs from current GA budget")
        if warm_payload.get("reference_sha256") != _hash(reference):
            raise ValueError("Warm-start baseline sufficiency reference differs")
        if warm_payload.get("baseline_attachment_sha256") != _hash(_mask_path(reference)):
            raise ValueError("Warm-start baseline attachment mask differs")
        expected = _fitness_settings()
        if warm_payload.get("fitness_settings") != expected:
            raise ValueError("Warm-start fitness settings differ from current GA settings")
    return metadata, status, warm_payload


def _entry(individual, *, source: str, created_generation: int,
           fitness: float | None = None, ancestry: dict | None = None) -> dict:
    entry = {
        "interventions": _records(individual),
        "source": source,
        "created_generation": created_generation,
        "scenario": None if fitness is not None else "ga_" + configuration_hash(individual)[:16],
        "fitness": fitness,
    }
    if ancestry:
        entry["ancestry"] = ancestry
    return entry


def _generation_manifest(dataset: Path, generation: int, population: list[str],
                         archive: dict) -> Path:
    target = dataset / f"generation_{generation:03d}.txt"
    # Only candidates without known fitness need MATSim evaluation. Keep this
    # generation manifest immutable while SLURM may have tasks using it.
    names = [archive[key]["scenario"] for key in population
             if archive[key]["fitness"] is None]
    if len(names) != len(set(names)):
        raise ValueError("Duplicate scenario in a generation")
    if target.exists():
        old = [line.strip() for line in target.read_text(encoding="utf-8").splitlines()
               if line.strip()]
        if old != names:
            raise RuntimeError(f"Existing generation manifest has changed: {target}")
    else:
        target.write_text("".join(f"{name}\n" for name in names), encoding="utf-8")
    return target


def _preflight_settings(jar: Path | None, *, java: str = "java") -> dict:
    """Refuse to silently disable authoritative validation in production."""
    if jar is None:
        return {"enabled": False, "jar": None, "java": java}
    jar = Path(jar).expanduser().resolve(strict=True)
    if not jar.is_file():
        raise FileNotFoundError(f"MATSim validator JAR is missing: {jar}")
    return {"enabled": True, "jar": str(jar), "java": java}


def _new_candidate_after_preflight_failure(
    state: dict, base, *, parent_keys: list[str], seen: set[str],
    rejected: set[str], rng: random.Random, diagnostics: dict,
) -> list | None:
    """Draw a genuinely new feasible individual without modifying network_sampling.

    First try a local mutation of a successfully evaluated parent, then a
    spatial crossover. Occasionally sample a new immigrant with a mixed target
    budget. Never substitute a failed child with a duplicate in this generation.
    """
    archive = state["archive"]
    valid_parents = [key for key in parent_keys
                     if key in archive and archive[key]["fitness"] is not None
                     and archive[key]["fitness"] >= 0]
    ctx = build_spatial_context(base.eligible, base.index)
    full_budget = state["budget_m"]
    for attempt in range(100):
        try:
            choice = rng.random()
            if valid_parents and choice < 0.68:
                left_key = _tournament(valid_parents, archive, rng)
                parent = _individual(archive[left_key]["interventions"])
                candidate = mutate(parent, eligible=base.eligible, network=base.network,
                                   index=base.index, budget_m=full_budget, rng=rng,
                                   context=ctx, diagnostics=diagnostics)
            elif len(valid_parents) > 1 and choice < 0.82:
                left_key = _tournament(valid_parents, archive, rng)
                right_key = _tournament(valid_parents, archive, rng)
                candidate = crossover(
                    _individual(archive[left_key]["interventions"]),
                    _individual(archive[right_key]["interventions"]),
                    eligible=base.eligible, network=base.network,
                    index=base.index, budget_m=full_budget, rng=rng,
                    context=ctx, diagnostics=diagnostics,
                )
            else:
                strategy = rng.choice(("random", "corridor", "cluster"))
                # Target can be short; the GA budget itself remains a MAXIMUM.
                low, high = rng.choice(((0.05, 0.20), (0.20, 0.47), (0.47, 1.0)))
                target = max(min(250.0, full_budget), rng.uniform(low, high) * full_budget)
                settings = {**DEFAULT_SETTINGS, "budget_m": target}
                candidate = sample(strategy, rng.randrange(2**31), base, settings)
            if not candidate:
                continue
            key = configuration_hash(candidate)
            if key in seen or key in rejected:
                continue
            ok, _ = validate_individual(candidate, eligible=base.eligible,
                                        network=base.network, index=base.index,
                                        budget_m=full_budget)
            if ok:
                return candidate
        except ValueError as exc:
            if str(exc) != "Cannot repair offspring with current operators":
                raise
        except RuntimeError:
            # Existing independent samplers can run out of feasible additions.
            continue
    return None


def _prepare_generation(
    state: dict, dataset: Path, base, *, parent_keys: list[str] | None = None,
    matsim_jar: Path | None = None, java: str = "java",
) -> Path:
    """Publish ONLY successfully prevalidated candidates into the SLURM manifest.

    A disconnected MATSim network is rejected while still inside the atomic
    scenario staging directory; its candidate is replaced before publishing
    the generation checkpoint or submitting the simulation array. A validator
    execution error is NOT treated as a candidate failure and is propagated.
    """
    settings = {**DEFAULT_SETTINGS, "budget_m": state["budget_m"]}
    network = Path(state["baseline"]) / "network.xml.gz"
    schedule = Path(state["baseline"]) / "transit_schedule.xml.gz"
    archive = state["archive"]
    seen = set(state["population"])
    rejected: set[str] = set()
    parents = list(parent_keys or ())
    diagnostics = state.setdefault("preflight_stats", {})
    gen_key = str(state["generation"])
    stats = diagnostics.setdefault(gen_key, {"passed": 0, "rejected": 0,
                                             "replacement_attempts": 0})
    for slot in range(len(state["population"])):
        rng = random.Random(f"preflight:{state['seed']}:{state['generation']}:{slot}")
        for attempt in range(PREFLIGHT_MAX_REPLACEMENTS + 1):
            key = state["population"][slot]
            item = archive[key]
            if item["fitness"] is not None:
                break  # Historical fitness, elite or already evaluated child.
            individual = _individual(item["interventions"])
            try:
                path = prepare_one(
                    network, schedule, dataset, individual, base, settings,
                    source=f"GA generation {state['generation']}",
                    matsim_jar=matsim_jar, java=java,
                )
            except MatsimConnectivityError as exc:
                # This is a configuration rejection, NOT a Java execution error.
                rejected.add(key)
                stats["rejected"] += 1
                item["preflight_rejection"] = str(exc)
                print(f"PREFLIGHT REJECT {item['scenario']}: {exc}", flush=True)
                if attempt == PREFLIGHT_MAX_REPLACEMENTS:
                    raise RuntimeError(
                        f"Could not replace slot {slot} after {attempt} invalid candidates"
                    ) from exc
                replacement = _new_candidate_after_preflight_failure(
                    state, base, parent_keys=parents,
                    seen=seen, rejected=rejected, rng=rng, diagnostics=stats,
                )
                if replacement is None:
                    raise RuntimeError(
                        f"No unique feasible replacement for generation {state['generation']} "
                        f"slot {slot}; check sampler and GA connectivity restrictions"
                    ) from exc
                replacement_key = configuration_hash(replacement)
                seen.discard(key)
                seen.add(replacement_key)
                state["population"][slot] = replacement_key
                if replacement_key not in archive:
                    archive[replacement_key] = _entry(
                        replacement, source="ga",
                        created_generation=state["generation"],
                        ancestry={"operator": "preflight_replacement", "parents": []},
                    )
                stats["replacement_attempts"] += 1
                continue
            if path.name != item["scenario"]:
                raise RuntimeError("Generated scenario ID does not match its fingerprint")
            stats["passed"] += 1
            break
        else:
            raise RuntimeError(f"Preflight retries exceeded for slot {slot}")

    manifest = _generation_manifest(dataset, state["generation"],
                                    state["population"], archive)
    print(f"Generation {state['generation']} preflight: "
          f"passed={stats['passed']}, rejected={stats['rejected']}, "
          f"replacements={stats['replacement_attempts']}", flush=True)
    return manifest


def initialize(args: argparse.Namespace) -> None:
    dataset = args.dataset
    state_path = dataset / STATE_FILE
    if state_path.exists():
        raise FileExistsError(f"Existing optimisation state: {state_path}. Use 'status' or 'step'.")
    if args.population < 2 or args.generations < 1 or not 0 <= args.elites < args.population:
        raise ValueError("Population must be >=2; generations >=1; elites in [0, population)")
    if not 0 <= args.immigrants <= 1 or args.budget <= 0:
        raise ValueError("Immigrant fraction must be in [0,1], budget positive")

    warm_file = None if args.no_warm_start else args.warm_start
    _, baseline_status, warm_payload = _validate_inputs(args.baseline, args.reference,
                                                        warm_file, args.budget)
    _, mask = validate_reference(args.reference, args.spatial_dir)
    if mask['baseline_scenario_sha256'] != scenario_hashes(args.baseline):
        raise ValueError('Trial 08 reference belongs to another baseline simulation')
    baseline_score = warm._evaluate(args.baseline, args.spatial_dir, args.reference, True)
    if abs(baseline_score['fitness'] - 0.89555067) > 1e-8:
        raise ValueError('Trial 08 baseline benchmark mismatch')
    settings = {**DEFAULT_SETTINGS, "budget_m": args.budget}
    print("Loading the baseline and eligible interventions...", flush=True)
    base = load_baseline(args.baseline / "network.xml.gz",
                         args.baseline / "transit_schedule.xml.gz", settings)
    seeds = []
    historical = {}
    if warm_file is not None:
        seeds, historical = warm.load_warm_start(warm_file)
        for candidate in seeds:
            valid, reason = validate_individual(candidate, eligible=base.eligible,
                                                network=base.network, index=base.index,
                                                budget_m=args.budget)
            if not valid:
                raise ValueError(f"Warm-start individual invalid: {reason}")
        if len(seeds) > args.population:
            raise ValueError("Warm-start seed count exceeds population; use smaller warm manifest for pilot")
        for seed in warm_payload['seeds']:
            if seed['fitness'] is not None:
                current = warm._evaluate(Path(seed['source_path']), args.spatial_dir, args.reference, True)
                if abs(current['fitness'] - float(seed['fitness'])) > 1e-8:
                    raise ValueError(f"Historical seed cache changed: {seed['scenario']}")
        print(f"Warm start: {len(historical)} historical, {len(seeds)-len(historical)} unevaluated seeds", flush=True)
    population = initial_population(
        base, size=args.population, budget_m=args.budget,
        rng=random.Random(args.seed), settings=settings, initial=seeds,
    )
    fingerprints = [configuration_hash(item) for item in population]
    archive = {}
    for item, key in zip(population, fingerprints):
        score = historical.get(key)
        if score is not None and not math.isfinite(score):
            raise ValueError("Non-finite historical fitness")
        archive[key] = _entry(item, source="historical" if score is not None else "ga",
                              created_generation=0, fitness=score)
        if score is not None and warm_payload is not None:
            origin = next((seed.get("scenario") for seed in warm_payload["seeds"]
                           if seed["config_hash"] == key), None)
            archive[key]["historical_scenario"] = origin
            archive[key]["historical_path"] = next(seed.get("source_path") for seed in warm_payload["seeds"] if seed["config_hash"] == key)
            archive[key]["budget_used_m"] = next(seed["budget_used_m"] for seed in warm_payload["seeds"] if seed["config_hash"] == key)

    state = {
        "version": 1,
        "operator_version": OPERATOR_VERSION,
        "preflight": ({"enabled": False, "jar": None, "java": args.java}
                      if args.dry_run else _preflight_settings(
                          None if args.skip_preflight else args.matsim_jar,
                          java=args.java)),
        "fitness_version": FITNESS_VERSION,
        "fitness_settings": _fitness_settings(),
        "baseline": str(args.baseline),
        "baseline_hashes": _json(args.baseline / "scenario.json")["baseline_hashes"],
        "reference": str(args.reference),
        "reference_sha256": _hash(args.reference),
        "attachment_sha256": _hash(_mask_path(args.reference)),
        "spatial_dir": str(args.spatial_dir),
        "expected_simulation": {field: baseline_status.get(field)
                                for field in SIMULATION_SETTINGS},
        "budget_m": args.budget,
        "seed": args.seed,
        "population_size": args.population,
        "generations": args.generations,
        "elites": args.elites,
        "immigrants": args.immigrants,
        "generation": 0,
        "phase": "waiting",
        "population": fingerprints,
        "archive": archive,
        "history": [],
    }
    # Use the same physical cost calculation as the GA; no routing changes.
    from network_sampling.sampling.random import intervention_length
    lengths = [intervention_length(item, base.index) for item in population]
    print(f"Initial intervention lengths (km): min={min(lengths)/1000:.2f}, mean={sum(lengths)/len(lengths)/1000:.2f}, max={max(lengths)/1000:.2f}", flush=True)
    needed = sum(archive[key]["fitness"] is None for key in fingerprints)
    print(f"Generation 0: {len(population)} individuals, {len(population)-needed} known, "
          f"{needed} new MATSim evaluations")
    if args.dry_run:
        print("DRY RUN: No scenario files or optimisation state written.")
        return
    check = state["preflight"]
    manifest = _prepare_generation(
        state, dataset, base,
        matsim_jar=Path(check["jar"]) if check["enabled"] else None,
        java=check["java"],
    )
    _write(state_path, state)
    print(f"State: {state_path}\nSubmit manifest: {manifest}")
    print("No MATSim simulations were launched.")


def _status_for(scenario: Path) -> str:
    status_file = scenario / "run_status.json"
    if not status_file.is_file():
        return "prepared"
    status = _json(status_file)
    stage = status.get("state", status.get("status"))
    if stage not in ("completed", "complete"):
        return str(stage)
    if not all((scenario / "targets" / target).is_file() for target in REQUIRED_TARGETS):
        return "needs_compaction"
    return "completed"


def _counts(state: dict, dataset: Path) -> dict[str, int]:
    counts = {"scored": 0, "prepared": 0, "running": 0,
              "failed": 0, "needs_compaction": 0, "completed": 0}
    for key in state["population"]:
        item = state["archive"][key]
        if item["fitness"] is not None:
            counts["scored"] += 1
        else:
            stage = _status_for(dataset / item["scenario"])
            counts[stage] = counts.get(stage, 0) + 1
    return counts


def report(state: dict, dataset: Path) -> None:
    counts = _counts(state, dataset)
    known = [v["fitness"] for v in state["archive"].values()
             if v["fitness"] is not None and v["fitness"] >= 0]
    print(f"Generation {state['generation'] + 1}/{state['generations']} "
          f"({state['phase']}), total archived configurations: {len(state['archive'])}")
    print("  " + ", ".join(f"{k}={v}" for k, v in counts.items() if v))
    if known:
        print(f"  Best known fitness: {max(known):.6f}")
    manifest = dataset / f"generation_{state['generation']:03d}.txt"
    print(f"  Generation manifest: {manifest}")


def _score_current(state: dict, dataset: Path, *, penalize_failed: bool, workers: int = 1) -> int:
    """Independent scenario processes; only the parent writes the checkpoint."""
    if workers < 1:
        raise ValueError('fitness workers must be positive')
    ref = Path(state['reference'])
    mask = _mask_path(ref)
    if _hash(ref) != state['reference_sha256'] or _hash(mask) != state['attachment_sha256']:
        raise ValueError('Fixed baseline sufficiency reference/attachments changed since init')
    spatial = Path(state['spatial_dir'])
    validate_reference(ref, spatial)
    new = 0
    tasks = []
    for key in state['population']:
        entry = state['archive'][key]
        if entry['fitness'] is not None:
            continue
        scenario = dataset / entry['scenario']
        stage = _status_for(scenario)
        if stage == 'failed' and penalize_failed:
            entry['fitness'] = -1.0
            entry['error'] = 'MATSim failed; explicitly penalized by operator'
            _write(dataset / STATE_FILE, state)
            new += 1
            print(f'PENALIZE failed simulation: {scenario.name}', flush=True)
            continue
        if stage != 'completed':
            print(f'WAIT {scenario.name}: {stage}', flush=True)
            continue
        status = _json(scenario / 'run_status.json')
        for field, expected in state['expected_simulation'].items():
            if expected is not None and status.get(field) != expected:
                raise ValueError(f'{scenario.name}: {field} differs from reference baseline MATSim')
        tasks.append((key, (scenario, spatial, ref, False)))
    print(f'Scoring {len(tasks)} completed scenarios with {min(workers,len(tasks))} process(es)', flush=True)
    def collect(key, result):
        nonlocal new
        score, error = result
        scenario = state['archive'][key]['scenario']
        if error is not None:
            print(f'WAIT scoring {scenario}: {error}', flush=True)
            return
        fitness = float(score['fitness'])
        if score['fitness_version'] != FITNESS_VERSION or not math.isfinite(fitness) or not 0 <= fitness <= 1:
            raise ValueError('Invalid Trial 08 score')
        entry = state['archive'][key]
        entry.update(fitness=fitness, raw_fitness=fitness, feasible=True,
                     car_constraint=score['car_constraint'], components=score['components'])
        _write(dataset / STATE_FILE, state)
        new += 1
        print(f'SCORED {scenario}: fitness={fitness:.8f}', flush=True)
    if workers == 1 or len(tasks) <= 1:
        for key, task in tasks:
            collect(key, warm._score_worker(task))
    else:
        with ProcessPoolExecutor(max_workers=min(workers,len(tasks))) as pool:
            futures = {pool.submit(warm._score_worker, task): key for key,task in tasks}
            for future in as_completed(futures):
                collect(futures[future], future.result())
    return new


def _tournament(keys: list[str], archive: dict, rng: random.Random) -> str:
    contestants = rng.sample(keys, min(TOURNAMENT_SIZE, len(keys)))
    return max(contestants, key=lambda key: (archive[key]["fitness"], key))


def _offspring(state: dict, base, *, diagnostics: dict | None = None,
               provenance: dict | None = None) -> list:
    """Elitism, spatial crossover, predominantly local mutation, and immigrants.

    Candidate order is deterministic for the seed/generation. Connectivity
    preflight happens in _prepare_generation, before state/manifest publication.
    """
    diagnostics = diagnostics if diagnostics is not None else {}
    provenance = provenance if provenance is not None else {}
    archive = state["archive"]
    ranked = sorted(state["population"], key=lambda key: (-archive[key]["fitness"], key))
    target_size = state["population_size"]
    elite_keys = ranked[:state["elites"]]
    next_population = [_individual(archive[key]["interventions"]) for key in elite_keys]
    seen = set(elite_keys)
    next_generation = state["generation"] + 1
    rng = random.Random(f"ga:{state['seed']}:{next_generation}")
    settings = {**DEFAULT_SETTINGS, "budget_m": state["budget_m"]}
    context = build_spatial_context(base.eligible, base.index)
    immigrant_count = min(target_size - len(next_population),
                          math.ceil(target_size * state["immigrants"]))
    goal = target_size - immigrant_count

    attempts = 0
    while len(next_population) < goal and attempts < target_size * 100:
        attempts += 1
        left_key = _tournament(state["population"], archive, rng)
        left = _individual(archive[left_key]["interventions"])
        parents = [left_key]
        label = "mutation"
        try:
            if rng.random() < CROSSOVER_PROBABILITY:
                right_key = _tournament(state["population"], archive, rng)
                right = _individual(archive[right_key]["interventions"])
                child = crossover(left, right, eligible=base.eligible,
                                  network=base.network, index=base.index,
                                  budget_m=state["budget_m"], rng=rng,
                                  context=context, diagnostics=diagnostics)
                parents.append(right_key)
                label = "spatial_crossover"
            else:
                child = left
            if rng.random() < MUTATION_PROBABILITY:
                steps = rng.choice(MUTATION_STEPS)
                for _ in range(steps):
                    child = mutate(child, eligible=base.eligible, network=base.network,
                                   index=base.index, budget_m=state["budget_m"],
                                   rng=rng, context=context, diagnostics=diagnostics)
                diagnostics["mutation_steps"] = diagnostics.get("mutation_steps", 0) + steps
                label = ("spatial_crossover+" if len(parents) > 1 else "") + "local_mutation"
        except ValueError as exc:
            if str(exc) != "Cannot repair offspring with current operators":
                raise
            diagnostics["repair_failed"] = diagnostics.get("repair_failed", 0) + 1
            continue
        key = configuration_hash(child)
        if key in seen or not child:
            diagnostics["duplicate_or_empty"] = diagnostics.get("duplicate_or_empty", 0) + 1
            continue
        next_population.append(child)
        seen.add(key)
        diagnostics["accepted_evolutionary"] = diagnostics.get("accepted_evolutionary", 0) + 1
        provenance[key] = {
            "operator": label, "parents": parents,
            "best_parent_fitness": max(archive[k]["fitness"] for k in parents),
        }

    diagnostics["breeding_attempts"] = attempts
    if len(next_population) < target_size:
        evolution_count = len(next_population)
        next_population = initial_population(
            base, size=target_size, budget_m=state["budget_m"],
            rng=rng, settings=settings, initial=next_population,
        )
        diagnostics["immigrants"] = len(next_population) - evolution_count
        for child in next_population[evolution_count:]:
            provenance[configuration_hash(child)] = {"operator": "immigrant", "parents": []}
    return next_population


def _write_best(dataset: Path, state: dict) -> Path:
    ranked = [(item["fitness"], key, item) for key, item in state["archive"].items()
              if item["fitness"] is not None and item["fitness"] >= 0]
    if not ranked:
        print("No feasible configurations to export (all evaluations failed).")
        return None
    score, key, item = max(ranked, key=lambda record: (record[0], record[1]))
    path = dataset / "optimization_best.json"
    _write(path, {
        "fitness": score, "config_hash": key, "fitness_version": FITNESS_VERSION,
        "source": item["source"], "scenario": item["scenario"],
        "historical_scenario": item.get("historical_scenario"),
        "historical_path": item.get("historical_path"),
        "interventions": item["interventions"],
        "baseline_hashes": state["baseline_hashes"],
    })
    return path


def _write_operator_outcomes(dataset: Path, archive: dict) -> None:
    """One row per evaluated offspring, with observed gain vs best parent.

    This makes actual operator quality measurable rather than inferred from
    population averages; failed simulations appear with success=False.
    """
    out = dataset / "optimization_operator_outcomes.csv"
    tmp = out.with_name(out.name + ".tmp")
    fields = ("generation", "scenario", "operator", "fitness", "valid",
              "best_parent_fitness", "gain_over_best_parent")
    with tmp.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for key, item in sorted(archive.items(), key=lambda kv: (
                kv[1]["created_generation"], kv[0])):
            ancestry = item.get("ancestry")
            if not ancestry or item.get("fitness") is None:
                continue
            score = item["fitness"]
            best_parent = ancestry.get("best_parent_fitness")
            writer.writerow({
                "generation": item["created_generation"],
                "scenario": item.get("scenario") or "ga_" + key[:16],
                "operator": ancestry["operator"],
                "fitness": score,
                "valid": score >= 0,
                "best_parent_fitness": best_parent,
                "gain_over_best_parent": (score - best_parent if score >= 0
                                          and best_parent is not None else None),
            })
    os.replace(tmp, out)


def _write_history(dataset: Path, history: list[dict]) -> None:
    path = dataset / "optimization_history.csv"
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=(
            "generation", "best_fitness", "mean_fitness", "unique_individuals",
            "evaluated_individuals", "new_simulations",
            "mean_budget_km", "max_budget_km", "mean_segment_jaccard",
            "failed_evaluations", "preflight_rejections",
        ))
        writer.writeheader()
        writer.writerows(history)
    os.replace(tmp, path)


def advance(args: argparse.Namespace) -> None:
    dataset = args.dataset
    state_path = dataset / STATE_FILE
    state = _json(state_path)
    if state["fitness_version"] != FITNESS_VERSION or state["version"] != 1:
        raise ValueError("Unsupported optimisation checkpoint/fitness version")
    if state.get("operator_version") != OPERATOR_VERSION:
        raise ValueError("This checkpoint uses different GA operators. Start a new dataset/run.")
    if state["fitness_settings"] != _fitness_settings():
        raise ValueError("Current fitness settings differ from this optimisation run")
    if state["phase"] == "complete":
        print("Optimisation already completed.")
        report(state, dataset)
        return
    _score_current(state, dataset, penalize_failed=args.penalize_failed, workers=args.fitness_workers)
    counts = _counts(state, dataset)
    if counts["scored"] != state["population_size"]:
        report(state, dataset)
        print("Generation incomplete. Submit/finish its manifest, then run 'step' again.")
        return

    entries = [state['archive'][key] for key in state['population']]
    scores = [entry['fitness'] for entry in entries]
    segments = [{iv['segment_id'] for iv in entry['interventions']} for entry in entries]
    similarities = [len(a & b)/len(a | b) if a | b else 1.0
                    for i,a in enumerate(segments) for b in segments[i+1:]]
    lengths = []
    for entry in entries:
        path = dataset / entry['scenario'] if entry['scenario'] else Path(entry['historical_path'])
        value = entry.get('budget_used_m')
        if value is None:
            value = _json(path/'scenario.json')['budget_used_m']
        lengths.append(float(value)/1000)
    summary = {
        'mean_budget_km': sum(lengths)/len(lengths),
        'max_budget_km': max(lengths),
        'mean_segment_jaccard': sum(similarities)/len(similarities),
        'failed_evaluations': sum(score < 0 for score in scores),
        'preflight_rejections': state.get('preflight_stats', {}).get(str(state['generation']), {}).get('rejected',0),
        "generation": state["generation"],
        "best_fitness": max(scores),
        "mean_fitness": sum(scores) / len(scores),
        "unique_individuals": len(set(state["population"])),
        "evaluated_individuals": len(scores),
        "new_simulations": sum(
            state["archive"][key]["created_generation"] == state["generation"]
            and state["archive"][key]["source"] == "ga"
            and state["archive"][key]["fitness"] >= 0
            for key in state["population"]
        ),
    }
    if not any(item["generation"] == state["generation"] for item in state["history"]):
        state["history"].append(summary)
    _write_history(dataset, state["history"])
    _write_operator_outcomes(dataset, state["archive"])
    _write_best(dataset, state)
    print(f"Completed generation {state['generation']}: best={summary['best_fitness']:.6f}, "
          f"mean={summary['mean_fitness']:.6f}; mean budget={summary['mean_budget_km']:.2f} km; "
          f"mean segment Jaccard={summary['mean_segment_jaccard']:.3f}", flush=True)
    if state["generation"] + 1 == state["generations"]:
        state["phase"] = "complete"
        _write(state_path, state)
        best_file = _write_best(dataset, state)
        print("Finished all generations. History:", dataset / "optimization_history.csv")
        if best_file is not None:
            print("Best configuration:", best_file)
        return

    # Prepare the next generation before publishing the checkpoint. A failure
    # during preparation leaves the previous checkpoint retriable and existing
    # scenario folders reusable via their deterministic hashes.
    baseline = Path(state["baseline"])
    if _json(baseline / "scenario.json").get("baseline_hashes") != state["baseline_hashes"]:
        raise ValueError("Source baseline changed since the optimisation was initialized")
    base_settings = {**DEFAULT_SETTINGS, "budget_m": state["budget_m"]}
    base = load_baseline(baseline / "network.xml.gz",
                         baseline / "transit_schedule.xml.gz", base_settings)
    diag, provenance = {}, {}
    next_population = _offspring(state, base, diagnostics=diag, provenance=provenance)
    next_state = json.loads(json.dumps(state))  # simple independent JSON snapshot
    next_state["generation"] += 1
    next_state["population"] = [configuration_hash(x) for x in next_population]
    for item, key in zip(next_population, next_state["population"]):
        if key not in next_state["archive"]:
            next_state["archive"][key] = _entry(
                item, source="ga", created_generation=next_state["generation"],
                ancestry=provenance.get(key),
            )
    next_state.setdefault("operator_stats", {})[str(next_state["generation"])] = diag
    check = next_state["preflight"]
    manifest = _prepare_generation(
        next_state, dataset, base, parent_keys=state["population"],
        matsim_jar=Path(check["jar"]) if check["enabled"] else None,
        java=check["java"],
    )
    _write(state_path, next_state)
    print(f"Prepared generation {next_state['generation']} with "
          f"{sum(next_state['archive'][k]['fitness'] is None for k in next_state['population'])} "
          f"new simulations.")
    print("Submit:", manifest)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    init = sub.add_parser("init", help="Build first population, including warm start")
    init.add_argument("--baseline", type=Path, default=BASELINE)
    init.add_argument("--warm-start", type=Path, default=WARM_START)
    init.add_argument("--no-warm-start", action="store_true")
    init.add_argument("--reference", type=Path, default=REFERENCE_FILE)
    init.add_argument("--spatial-dir", type=Path, default=SPATIAL_DIR)
    init.add_argument("--population", type=int, default=POPULATION_SIZE)
    init.add_argument("--generations", type=int, default=GENERATIONS)
    init.add_argument("--budget", type=float, default=BUDGET_M)
    init.add_argument("--seed", type=int, default=SEED)
    init.add_argument("--elites", type=int, default=ELITES)
    init.add_argument("--immigrants", type=float, default=IMMIGRANT_FRACTION)
    init.add_argument("--dry-run", action="store_true")
    init.add_argument("--matsim-jar", type=Path, default=MATSIM_JAR,
                      help="Switzerland JAR containing ValidateNetworkConnectivity")
    init.add_argument("--java", default="java")
    init.add_argument("--skip-preflight", action="store_true",
                      help="Only for local tests: skip authoritative Java preflight")
    status = sub.add_parser("status", help="Read checkpoint and scenario completion")
    step = sub.add_parser("step", help="Score completed jobs and prepare next generation")
    step.add_argument("--fitness-workers", type=int, default=1)
    step.add_argument("--penalize-failed", action="store_true",
                      help="Explicitly assign a low score to failed MATSim jobs")
    for p in (init, status, step):
        p.add_argument("--dataset", type=Path, default=DATASET)
    args = parser.parse_args()
    if args.command == "init":
        initialize(args)
    elif args.command == "status":
        report(_json(args.dataset / STATE_FILE), args.dataset)
    elif args.command == "step":
        advance(args)


if __name__ == "__main__":
    main()

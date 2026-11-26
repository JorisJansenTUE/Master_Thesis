"""Feasibility-constrained bicycle accessibility potential (no MATSim simulation).

Four HPC experiments: one unlimited-budget multi-order greedy retrofit, and
three independently restartable bicycle GAs (15, 50, 150 km).  Each evaluation
uses the *exact* 20-minute population-weighted quadratic bicycle component of
a frozen fitness_experiments/trial_03.  This is bicycle-only optimization;
PT and car accessibility are NOT recomputed.  Network eligibility and directed
mode connectivity are checked in Python; by default the Java MATSim routing
preflight additionally checks greedy batches and the best GA solutions.

Does not modify existing network_sampling, accessibility, optimization GA,
MATSim scenarios, fitness trials or input files.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
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
import time

import numpy as np
import pandas as pd

from accessibility.calculations.cumulative import CumulativeAccessibility
from accessibility.inputs.grid.snapping import ScenarioRoadSnapper
from accessibility.inputs.links.network import BikeNetwork
from accessibility.routing.road import RoadRouter
from network_sampling.dataset.generate import DEFAULT_SETTINGS, load_baseline, sample
from network_sampling.interventions.intervention_types import InterventionType
from network_sampling.interventions.matsim_app.apply_network import apply_interventions
from network_sampling.network.matsim_network import write_matsim_network
from optimization.ga import (
    build_spatial_context, configuration_hash, crossover, mutate, validate_individual,
)
from optimization.bicycle_extreme_diagnostics import (
    cache_bike, prepared_accessibility, read_population, score_bike, weighted_mean,
)

VERSION = "bicycle_budget_potential_v1"
CUTOFF_MIN = 20.0
SEED = 20261127
DEFAULT_ORDERS = ("long_first", "short_first", "random_17", "random_43")
GA_TYPES = {
    InterventionType.PROTECTED_BIKE_LANE,
    InterventionType.BICYCLE_STREET,
    InterventionType.DESTINATION_ONLY,
}
PBL_VARIANTS = {
    "keep_car": 0, "remove_one_lane": 1,
    "remove_one_direction": 2, "remove_car": 3,
}
_G = None


def _sha(file: Path) -> str:
    h = hashlib.sha256()
    with file.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _get_component(result: dict, mode: str, cutoff: float) -> float:
    hits = [v for v in result["components"]
            if v["mode"] == mode and abs(float(v["time_min"]) - cutoff) < 1e-8]
    if len(hits) != 1:
        raise ValueError(f"Expected one trial component for {mode}/{cutoff}min")
    return float(hits[0]["score"])


def _validate_trial(config: dict) -> tuple[float, float, float]:
    cfg = config["config"]
    time_w = {float(k): float(v) for k, v in cfg["TIME_WEIGHTS"].items()}
    if time_w != {20.0: 1.0}:
        raise ValueError("Expected trial_03 single 20-minute cutoff")
    weights = {str(k): float(v) for k, v in cfg["MODE_WEIGHTS"].items()}
    if weights != {"car": 0.0, "bike": 1.0, "pt": 1.0}:
        raise ValueError("Expected trial_03: bike+pt in objective, car weight zero")
    if not math.isclose(float(cfg["SUFFICIENCY_FRACTION"]), 1.0):
        raise ValueError("Expected trial_03 high sufficiency threshold")
    if not math.isclose(float(cfg["SHORTFALL_EXPONENT"]), 2.0):
        raise ValueError("Expected trial_03 squared shortfall")
    if not cfg["CAR_CONSTRAINT_ENABLED"]:
        raise ValueError("Expected trial_03 car safeguard in source experiment")
    jobs_col = CumulativeAccessibility.column_name(CUTOFF_MIN)
    reference = float(config["reference"][jobs_col])
    threshold = reference * float(cfg["SUFFICIENCY_FRACTION"])
    if not math.isfinite(threshold) or threshold <= 0:
        raise ValueError("Invalid baseline reference threshold")
    return threshold, 2.0, _get_component(config["baseline_score"], "bike", CUTOFF_MIN)


def _gene_label(iv) -> tuple:
    return (str(iv.segment_id), iv.intervention_type.value,
            iv.road_space_variant.value if iv.road_space_variant else "",
            tuple(str(x) for x in iv.target_link_ids))


def _serialized(iv) -> dict:
    return {
        "segment_id": str(iv.segment_id),
        "intervention_type": iv.intervention_type.value,
        "road_space_variant": iv.road_space_variant.value if iv.road_space_variant else None,
        "target_link_ids": [str(i) for i in iv.target_link_ids],
    }


def _initialize(args) -> dict:
    trial_path = args.trial.resolve(strict=True) / "experiment.json"
    conf = json.loads(trial_path.read_text(encoding="utf-8"))
    threshold, exponent, expected_baseline = _validate_trial(conf)
    if args.network:
        network = args.network.resolve(strict=True)
    else:
        network = (Path(conf["baseline_source"]).parent.parent / "network.xml.gz").resolve(strict=True)
    if args.schedule:
        schedule = args.schedule.resolve(strict=True)
    else:
        schedule = (network.parent / "transit_schedule.xml.gz").resolve(strict=True)
    spatial = args.spatial.resolve(strict=True)
    origins_path = (spatial / "origins.csv").resolve(strict=True)
    jobs_path = (spatial / "jobs.csv").resolve(strict=True)
    population_path = args.population.resolve(strict=True)
    if conf.get("population_sha256") and _sha(population_path) != conf["population_sha256"]:
        raise ValueError("STATPOP file differs from frozen Trial 03 population input")
    cached_baseline_path = Path(conf["baseline_source"]).resolve(strict=True)
    if conf.get("baseline_sha256") and _sha(cached_baseline_path) != conf["baseline_sha256"]:
        raise ValueError("Baseline accessibility cache differs from frozen Trial 03 input")
    pop = read_population(population_path)
    origins = pd.read_csv(origins_path, dtype={"hex_id": str})
    jobs = pd.read_csv(jobs_path, dtype={"job_id": str})
    if len(origins) != len(pop) or set(origins.hex_id) != set(pop.hex_id):
        raise ValueError("STATPOP population does not match spatial origins")
    base = load_baseline(network, schedule, dict(DEFAULT_SETTINGS))
    eligible = sorted((iv for iv in base.eligible if iv.intervention_type in GA_TYPES),
                      key=_gene_label)
    if not eligible:
        raise ValueError("No eligible bicycle/PBL/destination interventions")
    base = replace(base, eligible=eligible)
    gene_index = {iv: n for n, iv in enumerate(eligible)}
    spatial_context = build_spatial_context(eligible, base.index)
    baseline_cache = cache_bike(Path(conf["baseline_source"]), pop,
                               CumulativeAccessibility.column_name(CUTOFF_MIN))
    snapper = ScenarioRoadSnapper(
        origins, jobs,
        max_origin_distance_m=float(conf["config"]["ORIGIN_SNAP_M"]),
        max_job_distance_m=float(conf["config"]["JOB_SNAP_M"]),
    )
    normal = _bike_access(network, origins, jobs, conf["config"], snapper)
    checked = prepared_accessibility(normal, "bike", CumulativeAccessibility.column_name(CUTOFF_MIN), pop)
    col = CumulativeAccessibility.column_name(CUTOFF_MIN)
    if not np.array_equal(checked.hex_id.to_numpy(), baseline_cache.hex_id.to_numpy()):
        raise ValueError("Baseline origin IDs disagree with trial_03")
    if not np.allclose(checked[col].to_numpy(float), baseline_cache[col].to_numpy(float), atol=1e-7):
        delta = np.max(abs(checked[col].to_numpy(float)-baseline_cache[col].to_numpy(float)))
        raise RuntimeError(f"Rerouted bicycle baseline disagrees with trial_03 by {delta:.3f} jobs")
    baseline_score = score_bike(checked[col].to_numpy(float), checked.population.to_numpy(float),
                                threshold, exponent)
    if not math.isclose(baseline_score, expected_baseline, abs_tol=1e-8):
        raise RuntimeError(f"Trial_03 baseline score {expected_baseline} != rerouted {baseline_score}")
    identity = {
        "version": VERSION,
        "code_sha256": _sha(Path(__file__)),
        "trial": str(trial_path), "trial_sha256": _sha(trial_path),
        "network": str(network), "network_sha256": _sha(network),
        "schedule": str(schedule), "schedule_sha256": _sha(schedule),
        "origins": str(origins_path), "origins_sha256": _sha(origins_path),
        "jobs": str(jobs_path), "jobs_sha256": _sha(jobs_path),
        "population": str(population_path), "population_sha256": _sha(population_path),
        "eligible_sha256": hashlib.sha256(json.dumps(
            [_gene_label(iv) for iv in eligible], separators=(",", ":")).encode()).hexdigest(),
        "objective": "resident_weighted_bike_only_20min_high_squared_via_trial03",
        "threshold_jobs": threshold,
        "baseline_bike_score": baseline_score,
        "n_eligible": len(eligible),
    }
    print(f"Trial 03 matched: baseline bike={baseline_score:.8f}, threshold={threshold:.2f} jobs; "
          f"eligible genes={len(eligible)}, segments={len(set(str(iv.segment_id) for iv in eligible))}",
          flush=True)
    return dict(base=base, eligible=eligible, gene_index=gene_index,
                spatial_context=spatial_context,
                network=network, schedule=schedule, origins=origins, jobs=jobs, pop=pop,
                baseline=checked, threshold=threshold, exponent=exponent,
                baseline_score=baseline_score, cfg=conf["config"], identity=identity)


def _bike_access(path: Path, origins: pd.DataFrame, jobs: pd.DataFrame,
                 cfg: dict, snapper: ScenarioRoadSnapper | None = None):
    bike = BikeNetwork.from_matsim_network(path, bike_speed_m_s=float(cfg["BIKE_SPEED_M_S"]))
    if snapper is None:
        snapper = ScenarioRoadSnapper(
            origins, jobs,
            max_origin_distance_m=float(cfg["ORIGIN_SNAP_M"]),
            max_job_distance_m=float(cfg["JOB_SNAP_M"]),
        )
    attach = snapper.attach(bike.graph)
    router = RoadRouter(bike.graph, attach.origins.connectors, attach.jobs.connectors, jobs)
    return CumulativeAccessibility((CUTOFF_MIN,)).run(router, origins.hex_id.astype(str), mode="bike")


def _score_network(path: Path, ctx: dict) -> dict:
    raw = _bike_access(path, ctx["origins"], ctx["jobs"], ctx["cfg"])
    col = CumulativeAccessibility.column_name(CUTOFF_MIN)
    processed = prepared_accessibility(raw, "bike", col, ctx["pop"])
    original = ctx["baseline"][col].to_numpy(float)
    now = processed[col].to_numpy(float)
    if not np.array_equal(ctx["baseline"].hex_id.to_numpy(), processed.hex_id.to_numpy()):
        raise ValueError("Mismatch in scenario vs baseline hexagon order")
    people = processed.population.to_numpy(float)
    fitness = score_bike(now, people, ctx["threshold"], ctx["exponent"])
    change = now - original
    return {
        "bike_score": fitness,
        "bike_score_delta": fitness - ctx["baseline_score"],
        "mean_jobs": weighted_mean(now, people),
        "mean_job_gain": weighted_mean(change, people),
        "sufficient_resident_share": weighted_mean((now >= ctx["threshold"]).astype(float), people),
        "share_residents_improved": float(people[change > 1e-7].sum()/people.sum()),
        "share_residents_worsened": float(people[change < -1e-7].sum()/people.sum()),
        "attached_origins": int(processed.attached.sum()),
    }


def _intervention_km(individual, base) -> float:
    return float(sum(float(base.index.get(iv.segment_id).length_m) for iv in individual))/1000.0


def _modified_path(ctx: dict, individual, target: Path) -> None:
    modified, bus_mapping = apply_interventions(ctx["base"].network, ctx["base"].index, individual)
    if bus_mapping:
        raise RuntimeError("Unexpected bus-lane modification in bike experiment")
    write_matsim_network(modified, target)


def _java_network(path: Path, jar: Path | None, java: str) -> tuple[bool, str]:
    if jar is None:
        return True, "SKIPPED (Python network connectivity only)"
    from network_sampling.validation.matsim_connectivity import check_matsim_connectivity
    try:
        check_matsim_connectivity(path, jar, java=java)
        return True, "Java MATSim routing preflight passed"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def _java_individual(ctx: dict, individual, jar: Path | None, java: str) -> tuple[bool, str]:
    if jar is None:
        return True, "SKIPPED (Python only)"
    with tempfile.TemporaryDirectory(prefix="bike_java_") as d:
        path = Path(d) / "network.xml.gz"
        _modified_path(ctx, individual, path)
        return _java_network(path, jar, java)


def _ensure_identity(folder: Path, identity: dict):
    folder.mkdir(parents=True, exist_ok=True)
    manifest = folder / "experiment_manifest.json"
    if manifest.exists():
        old = json.loads(manifest.read_text(encoding="utf-8"))
        if old != identity:
            raise RuntimeError(f"Configuration/input changed for {folder}; use a NEW output folder")
    else:
        if any(folder.iterdir()):
            raise RuntimeError(f"Nonempty folder without manifest: {folder}")
        _atomic_json(manifest, identity)


def _valid(individual, ctx: dict, budget_km: float) -> bool:
    ok, _ = validate_individual(individual, eligible=ctx["eligible"],
                                network=ctx["base"].network, index=ctx["base"].index,
                                budget_m=budget_km*1000.0)
    return ok


def _segment_candidates(ctx: dict) -> dict[str, list]:
    grouped = defaultdict(list)
    for iv in ctx["eligible"]:
        grouped[str(iv.segment_id)].append(iv)
    result = {}
    for sid, ivs in grouped.items():
        pbl = sorted((iv for iv in ivs if iv.intervention_type == InterventionType.PROTECTED_BIKE_LANE),
                     key=lambda iv: PBL_VARIANTS.get(iv.road_space_variant.value if iv.road_space_variant else "", 9))
        destination = sorted((iv for iv in ivs if iv.intervention_type == InterventionType.DESTINATION_ONLY), key=_gene_label)
        # For GREEDY: prefer a feasible PBL; otherwise destination-only.
        result[sid] = pbl + destination
    return result


def _greedy_order(ctx: dict, name: str, keys: list[str]) -> list[str]:
    length = lambda sid: float(ctx["base"].index.get(sid).length_m)
    if name == "short_first":
        return sorted(keys, key=lambda x: (length(x), x))
    if name == "long_first":
        return sorted(keys, key=lambda x: (-length(x), x))
    if name.startswith("random_"):
        order = sorted(keys)
        random.Random(int(name.split("_", 1)[1])).shuffle(order)
        return order
    raise ValueError(f"Unsupported greedy order: {name}")


def _choose_greedy(ctx: dict, individual: list, candidates: list, *, jar: Path | None,
                   java: str, each_java: bool) -> tuple[object | None, Counter]:
    stats = Counter()
    for iv in candidates:
        test = individual + [iv]
        if not _valid(test, ctx, budget_km=1e15):
            stats["python_rejected"] += 1
            continue
        if each_java:
            ok, _ = _java_individual(ctx, test, jar, java)
            if not ok:
                stats["java_rejected"] += 1
                continue
        return iv, stats
    return None, stats


def _greedy_search(task: tuple[str, str, str, int, str]) -> dict:
    """One independent search; greedy steps are sequential, orders parallelizable."""
    ctx = _G
    if ctx is None:
        raise RuntimeError("Greedy worker context missing")
    name, folder_text, jar_text, checkpoint_every, java = task
    folder = Path(folder_text)
    jar = Path(jar_text) if jar_text else None
    by_segment = _segment_candidates(ctx)
    order = _greedy_order(ctx, name, [sid for sid, options in by_segment.items() if options])
    checkpoint = folder / "checkpoints" / (name + ".json")
    total = len(order)
    if checkpoint.is_file():
        last = json.loads(checkpoint.read_text(encoding="utf-8"))
        if last["order"] != name or last["total"] != total:
            raise RuntimeError(f"Greedy checkpoint incompatible: {checkpoint}")
        next_index = int(last["next_index"])
        chosen = [ctx["eligible"][i] for i in last["genes"]]
        counts = Counter(last["diagnostics"])
        # Every saved checkpoint has passed its full Java check (if enabled).
        print(f"  GREEDY {name}: resume at {next_index}/{total}", flush=True)
    else:
        next_index, chosen, counts = 0, [], Counter()
    if not _valid(chosen, ctx, 1e15):
        raise RuntimeError("Invalid resumed greedy individual")
    if jar is not None and next_index:
        ok, why = _java_individual(ctx, chosen, jar, java)
        if not ok:
            raise RuntimeError(f"Previously validated greedy checkpoint now fails Java: {why}")
    batch_before = list(chosen)
    batch_start = next_index
    for pos in range(next_index, total):
        sid = order[pos]
        iv, failures = _choose_greedy(ctx, chosen, by_segment[sid],
                                      jar=None, java=java, each_java=False)
        counts.update(failures)
        if iv is not None:
            chosen.append(iv)
            counts["added_" + iv.intervention_type.value] += 1
        else:
            counts["segments_skipped"] += 1
        at_check = ((pos+1) % checkpoint_every == 0 or pos == total-1)
        if at_check:
            if jar is not None and chosen != batch_before:
                ok, why = _java_individual(ctx, chosen, jar, java)
                counts["java_batch_checks"] += 1
                if not ok:
                    counts["java_failed_batches"] += 1
                    print(f"  GREEDY {name}: Java rejected batch through {pos+1}/{total}; "
                          f"replay {pos+1-batch_start} segments with Java per addition", flush=True)
                    chosen = list(batch_before)
                    # Enforce Java feasibility at every addition to identify which
                    # PBL fails, and fall back to destination_only when possible.
                    for sid2 in order[batch_start:pos+1]:
                        iv2, stats2 = _choose_greedy(ctx, chosen, by_segment[sid2],
                                                     jar=jar, java=java, each_java=True)
                        counts.update(stats2)
                        if iv2 is not None:
                            chosen.append(iv2)
                        else:
                            counts["java_replay_segments_skipped"] += 1
                    ok, why = _java_individual(ctx, chosen, jar, java)
                    if not ok:
                        raise RuntimeError(f"Greedy replay still violates Java preflight: {why}")
            # Checkpoints are written only after the full batch passed Java.
            _atomic_json(checkpoint, {
                "order": name, "total": total, "next_index": pos+1,
                "genes": [ctx["gene_index"][iv] for iv in chosen],
                "diagnostics": dict(counts), "java_validated": jar is not None,
            })
            batch_before, batch_start = list(chosen), pos+1
            if (pos+1) % max(checkpoint_every*4, 1) == 0 or pos == total-1:
                print(f"  GREEDY {name}: {pos+1}/{total} scanned; "
                      f"{len(chosen)} interventions; {_intervention_km(chosen, ctx['base']):.1f} km",
                      flush=True)
    out = folder / "results" / (name + ".json")
    out.parent.mkdir(parents=True, exist_ok=True)
    network_file = folder / "networks" / (name + ".network.xml.gz")
    network_file.parent.mkdir(parents=True, exist_ok=True)
    _modified_path(ctx, chosen, network_file)
    ok, why = _java_network(network_file, jar, java)
    if not ok:
        raise RuntimeError(f"Greedy final Java preflight failed: {why}")
    metric = _score_network(network_file, ctx)
    result = {
        "strategy": name, "budget_limit_km": None,
        "used_budget_km": _intervention_km(chosen, ctx["base"]),
        "intervention_count": len(chosen),
        "intervention_types": dict(Counter(iv.intervention_type.value for iv in chosen)),
        "java_passed": jar is not None, "java_status": why,
        "diagnostics": dict(counts),
        "genes": [ctx["gene_index"][iv] for iv in chosen],
        "interventions": [_serialized(iv) for iv in chosen],
        **metric,
    }
    _atomic_json(out, result)
    print(f"GREEDY {name} DONE: bike={metric['bike_score']:.8f} "
          f"delta={metric['bike_score_delta']:+.8f}, km={result['used_budget_km']:.2f}, "
          f"Java={result['java_passed']}", flush=True)
    return {k:v for k,v in result.items() if k not in ("genes", "interventions")}


def _greedy_report(folder: Path) -> None:
    results = [json.loads(p.read_text(encoding="utf-8")) for p in
               sorted((folder/"results").glob("*.json"))]
    if not results:
        return
    fields = ["strategy", "bike_score", "bike_score_delta", "mean_job_gain",
              "used_budget_km", "intervention_count", "java_passed"]
    pd.DataFrame([{k:r[k] for k in fields} for r in results]).sort_values(
        "bike_score", ascending=False).to_csv(folder/"summary.csv", index=False)
    best = max(results, key=lambda x:x["bike_score"])
    _atomic_json(folder/"best_greedy.json", best)
    print(f"BEST GREEDY: {best['strategy']} F_bike={best['bike_score']:.8f} "
          f"delta={best['bike_score_delta']:+.8f}, "
          f"budget={best['used_budget_km']:.2f} km", flush=True)


def _sample_population(ctx: dict, n: int, budget_km: float, seen: set,
                       rng: random.Random, seed_file: Path, attempts_limit: int) -> list[dict]:
    """Different-size scenario sampling plus mutated greedy seeds for large budgets."""
    result = []
    base = ctx["base"]
    sources = []
    for p in sorted((seed_file/"results").glob("*.json")):
        item = json.loads(p.read_text(encoding="utf-8"))
        if not item.get("java_passed", False):
            # Python-only experiment seeds are permitted if user explicitly skipped Java.
            pass
        source = [ctx["eligible"][k] for k in item["genes"]]
        sources.append((item["strategy"], source))
    if not sources:
        raise RuntimeError(f"No completed greedy seeds under {seed_file}; run greedy first")
    def add(individual, origin):
        if not individual:
            return False
        key = configuration_hash(individual)
        if key in seen or not _valid(individual, ctx, budget_km):
            return False
        seen.add(key)
        result.append({"key": key, "genes": sorted(ctx["gene_index"][v] for v in individual),
                       "origin": origin})
        return True
    # Prefixes of validated greedy orders retain Python connectivity. No need
    # for the original greedy solution to fit the budget.
    for name, source in sources:
        for fraction in (0.25, 0.50, 0.75, 0.95, 1.0):
            limit = budget_km * 1000.0 * fraction
            prefix, used = [], 0.0
            for iv in source:
                cost = float(base.index.get(iv.segment_id).length_m)
                if used + cost > limit + 1e-6:
                    break
                prefix.append(iv)
                used += cost
            add(prefix, f"greedy:{name}:{fraction}")
            if len(result) >= n:
                return result
    # The old samplers often fail when asked directly for 150 km. Start with
    # smaller targets and then use spatial mutation to diversify seeds.
    strategies = ("random", "cluster", "corridor")
    lengths = (0.05, 0.15, 0.35, 0.65, 1.0)
    for attempt in range(attempts_limit):
        if len(result) >= n:
            break
        strategy = strategies[attempt % len(strategies)]
        fraction = lengths[(attempt // len(strategies)) % len(lengths)]
        settings = {**DEFAULT_SETTINGS,
                    "budget_m": budget_km*1000.0*fraction,
                    "corridor_bus_share": 0.0,
                    "corridor_destination_share": 0.0}
        try:
            individual = sample(strategy, rng.randrange(2**31), base, settings)
        except (ValueError, RuntimeError):
            individual = None
        if individual is not None:
            add(individual, f"sample:{strategy}:{fraction:g}")
        # Diversity safeguard: mutate a known feasible source even if the
        # 150km sampler rarely returns valid individuals.
        if len(result) and attempt % 3 == 0 and len(result) < n:
            parent = [ctx["eligible"][i] for i in rng.choice(result)["genes"]]
            for _ in range(rng.randint(2, 8)):
                try:
                    parent = mutate(parent, eligible=ctx["eligible"], network=base.network,
                                    index=base.index, budget_m=budget_km*1000.0, rng=rng,
                                    context=ctx["spatial_context"])
                except ValueError:
                    break
            add(parent, "mutated_seed")
    if len(result) < n:
        raise RuntimeError(f"Only {len(result)}/{n} feasible unique individuals initialized; "
                           "increase --seed-attempts")
    return result


def _offspring(ctx: dict, previous: dict, budget_km: float,
               seed: int, generation: int, n: int, elites: int,
               seed_folder: Path, seed_attempts: int) -> dict:
    rng = random.Random(f"bicycle-ga:{seed}:{budget_km:g}:{generation}")
    folder = _ga_folder(ctx["output"], budget_km)
    if previous is None:
        population = _sample_population(ctx, n, budget_km, set(), rng,
                                        seed_folder, seed_attempts)
        counters = {"initial": len(population)}
    else:
        old = previous["population"]
        fitness = {}
        for candidate in old:
            path = folder / "evaluations" / (candidate["key"] + ".json")
            fitness[candidate["key"]] = float(json.loads(path.read_text(encoding="utf-8"))["bike_score"])
        ranked = sorted(old, key=lambda x: (-fitness[x["key"]], x["key"]))
        population = [dict(x, origin="elite") for x in ranked[:elites]]
        seen = {x["key"] for x in population}
        counters = Counter()
        base = ctx["base"]
        def tournament():
            return max(rng.sample(old, min(3, len(old))), key=lambda x: fitness[x["key"]])
        target = int(0.90*n)
        for attempt in range(n*160):
            if len(population) >= target:
                break
            a = [ctx["eligible"][i] for i in tournament()["genes"]]
            try:
                if rng.random() < 0.75:
                    b = [ctx["eligible"][i] for i in tournament()["genes"]]
                    child = crossover(a, b, eligible=ctx["eligible"],
                                      network=base.network, index=base.index,
                                      budget_m=budget_km*1000.0, rng=rng,
                                      context=ctx["spatial_context"])
                    counters["crossovers"] += 1
                else:
                    child = a
                if rng.random() < 0.9:
                    # Macro mutation to change more than one street per generation.
                    steps = rng.choice((1, 2, 4, 8, 12))
                    for _ in range(steps):
                        child = mutate(child, eligible=ctx["eligible"],
                                       network=base.network, index=base.index,
                                       budget_m=budget_km*1000.0, rng=rng,
                                      context=ctx["spatial_context"])
                    counters["mutations"] += steps
            except ValueError as exc:
                if "Cannot repair offspring" not in str(exc):
                    raise
                counters["repair_rejections"] += 1
                continue
            key = configuration_hash(child)
            if key in seen:
                counters["duplicates"] += 1
                continue
            if not _valid(child, ctx, budget_km):
                counters["invalid_offspring"] += 1
                continue
            seen.add(key)
            population.append({"key": key, "genes": sorted(ctx["gene_index"][v] for v in child),
                               "origin": "GA"})
            counters["accepted_offspring"] += 1
        if len(population) < n:
            extra = _sample_population(ctx, n-len(population), budget_km, seen,
                                       rng, seed_folder, seed_attempts)
            population.extend(extra)
            counters["immigrants"] += len(extra)
    if len(population) != n:
        raise AssertionError(f"Generation size {len(population)} != {n}")
    return {"generation": generation, "population": population,
            "budget_km": budget_km, "operators": dict(counters)}


def _ga_folder(root: Path, budget_km: float) -> Path:
    return root / f"ga_{budget_km:g}km"


def _evaluate(task: dict) -> dict:
    ctx = _G
    if ctx is None:
        raise RuntimeError("GA worker context missing")
    individual = [ctx["eligible"][i] for i in task["genes"]]
    with tempfile.TemporaryDirectory(prefix="bicycle_ga_") as d:
        path = Path(d) / "network.xml.gz"
        _modified_path(ctx, individual, path)
        java_ok, java_reason = _java_network(path, ctx["jar"], ctx["java"])
        if java_ok:
            metric = _score_network(path, ctx)
        else:
            # Never let an individual rejected by the actual MATSim Java
            # routing cleaner receive a positive selection score.
            metric = {"bike_score": -1.0,
                      "bike_score_delta": -1.0 - ctx["baseline_score"],
                      "mean_jobs": None, "mean_job_gain": None,
                      "sufficient_resident_share": None,
                      "share_residents_improved": None,
                      "share_residents_worsened": None,
                      "attached_origins": None}
    return {"key": task["key"], "genes": task["genes"],
            "used_budget_km": _intervention_km(individual, ctx["base"]),
            "intervention_count": len(individual),
            "intervention_types": dict(Counter(iv.intervention_type.value for iv in individual)),
            "java_passed": ctx["jar"] is not None and java_ok,
            "java_status": java_reason,
            "feasible": bool(java_ok),
            **metric}


def _evaluate_population(folder: Path, population: list, workers: int,
                         executor: ProcessPoolExecutor | None) -> None:
    dest = folder/"evaluations"
    dest.mkdir(exist_ok=True)
    missing = [c for c in population if not (dest/(c["key"]+".json")).is_file()]
    print(f"  to evaluate={len(missing)}, cache reused={len(population)-len(missing)}", flush=True)
    if executor is None:
        for i, candidate in enumerate(missing, 1):
            r = _evaluate(candidate)
            _atomic_json(dest/(r["key"]+".json"), r)
    else:
        future_map = {executor.submit(_evaluate, c): c for c in missing}
        for i, future in enumerate(as_completed(future_map), 1):
            result = future.result()  # Unexpected errors must not be hidden.
            _atomic_json(dest/(result["key"]+".json"), result)
            if i % 25 == 0 or i == len(missing):
                print(f"  evaluated {i}/{len(missing)}", flush=True)


def _history(folder: Path) -> list[dict]:
    rows = []
    for p in sorted(folder.glob("generation_*.json")):
        gen = json.loads(p.read_text(encoding="utf-8"))
        scores = []
        lengths = []
        for c in gen["population"]:
            f = folder/"evaluations"/(c["key"]+".json")
            if not f.exists():
                break
            row = json.loads(f.read_text(encoding="utf-8"))
            scores.append(float(row["bike_score"]))
            lengths.append(float(row["used_budget_km"]))
        if len(scores) != len(gen["population"]):
            break
        rows.append({"generation": gen["generation"],
                     "best_fitness": max(scores), "mean_fitness": float(np.mean(scores)),
                     "mean_used_km": float(np.mean(lengths)),
                     "unique": len({c["key"] for c in gen["population"]}),
                     **gen["operators"]})
    if rows:
        pd.DataFrame(rows).to_csv(folder/"history.csv", index=False)
    return rows


def _best_ga(folder: Path, ctx: dict, budget_km: float, jar: Path | None,
             java: str, max_java_attempts: int) -> dict:
    scored = [json.loads(p.read_text(encoding="utf-8"))
              for p in (folder/"evaluations").glob("*.json")]
    if not scored:
        raise RuntimeError("GA has no completed accessibility evaluations")
    ranked = sorted(scored, key=lambda r:(-r["bike_score"], r["key"]))
    if ranked[0]["bike_score"] < 0:
        raise RuntimeError("No Java-MATSim-feasible GA individual in evaluated population")
    _atomic_json(folder/"best_python_connectivity.json", ranked[0])
    failures = []
    for candidate in ranked[:max_java_attempts] if jar is not None else ranked[:1]:
        if not candidate.get("feasible", True):
            failures.append({"key": candidate["key"], "reason": candidate.get("java_status")})
            continue
        individual = [ctx["eligible"][k] for k in candidate["genes"]]
        ok, reason = _java_individual(ctx, individual, jar, java)
        if ok:
            result = {**candidate, "java_passed": jar is not None,
                      "java_status": reason, "budget_limit_km": budget_km,
                      "java_candidates_rejected_before_selected": len(failures),
                      "interventions": [_serialized(iv) for iv in individual]}
            _atomic_json(folder/"best_validated.json", result)
            _modified_path(ctx, individual, folder/"best_network.xml.gz")
            _atomic_json(folder/"java_rejections.json", {"rejected": failures})
            print(f"  BEST GA {budget_km:g}km: F_bike={candidate['bike_score']:.8f} "
                  f"delta={candidate['bike_score_delta']:+.8f}, "
                  f"km={candidate['used_budget_km']:.1f}, Java={result['java_passed']}", flush=True)
            return result
        failures.append({"key": candidate["key"], "reason": reason})
    _atomic_json(folder/"java_rejections.json", {"rejected": failures})
    raise RuntimeError(f"None of the top {max_java_attempts} GA solutions passed Java preflight; "
                       "increase --max-java-attempts or examine java_rejections.json")


def _run_greedy(args, ctx, jar):
    folder = args.out.resolve()/"greedy"
    ident = {**ctx["identity"], "experiment": "unlimited_greedy",
             "orders": list(args.orders), "checkpoint_every": args.checkpoint_every,
             "java_jar": str(jar) if jar else None,
             "java_jar_sha256": _sha(jar) if jar else None}
    _ensure_identity(folder, ident)
    finished = {p.stem for p in (folder/"results").glob("*.json")}
    tasks = [(o, str(folder), str(jar) if jar else "", args.checkpoint_every, args.java)
             for o in args.orders if o not in finished]
    print(f"GREEDY: {len(args.orders)} orders; pending={len(tasks)}; "
          f"workers={min(args.workers, len(tasks)) if tasks else 0}", flush=True)
    if args.workers == 1:
        for t in tasks:
            _greedy_search(t)
    elif tasks:
        with ProcessPoolExecutor(max_workers=min(args.workers, len(tasks)),
                                 mp_context=mp.get_context("fork")) as pool:
            futures = {pool.submit(_greedy_search, t): t[0] for t in tasks}
            for future in as_completed(futures):
                future.result()
    _greedy_report(folder)


def _run_ga(args, ctx, jar):
    budget_km = float(args.budget_km)
    folder = _ga_folder(args.out.resolve(), budget_km)
    greedy_folder = args.out.resolve()/"greedy"
    seed_results = sorted((greedy_folder/"results").glob("*.json"))
    if not seed_results:
        raise RuntimeError(f"Run greedy first: no seeds at {greedy_folder/'results'}")
    ident = {**ctx["identity"], "experiment": "budget_ga",
             "budget_km": budget_km, "population_size": args.population_size,
             "generations": args.generations, "seed": args.seed,
             "seed_attempts": args.seed_attempts,
             "greedy_seed_hashes": {p.name: _sha(p) for p in seed_results},
             "java_jar": str(jar) if jar else None,
             "java_jar_sha256": _sha(jar) if jar else None,
             "max_java_attempts": args.max_java_attempts}
    _ensure_identity(folder, ident)
    ctx["output"] = args.out.resolve()
    ctx["jar"] = jar
    ctx["java"] = args.java
    previous = None
    global _G
    _G = ctx
    if args.workers == 1:
        executor = None
    else:
        if os.name != "posix":
            raise RuntimeError("Multiple workers require Linux fork")
        executor = ProcessPoolExecutor(max_workers=args.workers,
                                       mp_context=mp.get_context("fork"))
    try:
        for generation in range(args.max_generations):
            file = folder/f"generation_{generation:03d}.json"
            if file.is_file():
                current = json.loads(file.read_text(encoding="utf-8"))
                if current["generation"] != generation or current["budget_km"] != budget_km:
                    raise RuntimeError("Mismatched generation checkpoint")
            else:
                current = _offspring(ctx, previous, budget_km, args.seed, generation,
                                     args.population_size, args.elites,
                                     greedy_folder, args.seed_attempts)
                _atomic_json(file, current)
            print(f"GA {budget_km:g}km generation={generation}/{args.max_generations-1}", flush=True)
            _evaluate_population(folder, current["population"], args.workers, executor)
            previous = current
            rows = _history(folder)
            print(f"  current best={max(x['best_fitness'] for x in rows):.8f}; "
                  f"last mean={rows[-1]['mean_fitness']:.8f}", flush=True)
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
    _best_ga(folder, ctx, budget_km, jar, args.java, args.max_java_attempts)


def _args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="experiment", required=True)
    for name in ("greedy", "ga"):
        q = sub.add_parser(name)
        q.add_argument("--trial", type=Path, required=True)
        q.add_argument("--spatial", type=Path, required=True)
        q.add_argument("--population", type=Path, required=True)
        q.add_argument("--out", type=Path, required=True)
        q.add_argument("--network", type=Path, help="Baseline network; defaults to trial baseline location")
        q.add_argument("--schedule", type=Path, help="Baseline schedule; defaults to trial baseline location")
        q.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
        q.add_argument("--matsim-jar", type=Path, help="Existing Switzerland JAR for Java routing preflight")
        q.add_argument("--skip-java", action="store_true", help="Explicitly skip Java preflight: Python only")
        q.add_argument("--java", default="java")
    sub.choices["greedy"].add_argument("--orders", nargs="+", default=list(DEFAULT_ORDERS))
    sub.choices["greedy"].add_argument("--checkpoint-every", type=int, default=50)
    g = sub.choices["ga"]
    g.add_argument("--budget-km", type=float, required=True, choices=(15.0, 50.0, 150.0))
    g.add_argument("--population-size", type=int, default=100)
    g.add_argument("--generations", type=int, default=80)
    g.add_argument("--max-generations", type=int, default=None,
                   help="Complete at most this many generations, then resume later")
    g.add_argument("--elites", type=int, default=3)
    g.add_argument("--seed", type=int, default=SEED)
    g.add_argument("--seed-attempts", type=int, default=30000)
    g.add_argument("--max-java-attempts", type=int, default=40)
    return p.parse_args(argv)


def main(argv=None):
    args = _args(argv)
    if args.workers < 1:
        raise ValueError("--workers must be >=1")
    if args.experiment == "ga":
        if not 1 <= args.elites <= args.population_size:
            raise ValueError("Invalid elite/population count")
        if args.generations < 1:
            raise ValueError("--generations must be positive")
        args.max_generations = args.generations if args.max_generations is None else args.max_generations
        if not 1 <= args.max_generations <= args.generations:
            raise ValueError("--max-generations must be in [1,generations]")
    elif args.checkpoint_every < 1:
        raise ValueError("--checkpoint-every must be >=1")
    if args.skip_java:
        if args.matsim_jar is not None:
            raise ValueError("Do not give --matsim-jar with --skip-java")
        jar = None
        print("WARNING: Java routing preflight SKIPPED. Python connectivity is NOT identical to Java MATSim routing feasibility.", flush=True)
    else:
        if args.matsim_jar is None:
            raise ValueError("Provide --matsim-jar (default strict Java check) or explicitly --skip-java")
        jar = args.matsim_jar.resolve(strict=True)
    ctx = _initialize(args)
    if jar is not None:
        ok, reason = _java_network(ctx["network"], jar, args.java)
        if not ok:
            raise RuntimeError(f"Baseline fails MATSim Java routing preflight: {reason}")
        print("Baseline Java MATSim routing preflight passed", flush=True)
    global _G
    _G = ctx
    if args.experiment == "greedy":
        _run_greedy(args, ctx, jar)
    else:
        _run_ga(args, ctx, jar)


if __name__ == "__main__":
    main()

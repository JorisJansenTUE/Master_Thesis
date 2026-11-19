"""Generate reproducible MATSim intervention scenarios (no surrogate/GNN features).

Uses the existing samplers and existing matsim_app. Each scenario contains:
    network.xml.gz, transit_schedule.xml.gz, interventions.json,
    link_mapping.csv.gz, scenario.json.
A dataset contains street_segments.csv.gz and dataset_manifest.json once.

Run from Master_Thesis as:
    python -m network_sampling.dataset.generate \
        --network "$NETWORK" --schedule "$SCHEDULE" --out "$DATASET" \
        --samples 10 --strategies baseline random corridor cluster
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import hashlib
import json
import random
import tempfile
from dataclasses import dataclass
from pathlib import Path

import networkx as nx

from network_sampling.dataset.link_mapping import write_link_mapping, write_street_segments
from network_sampling.interventions.eligibility import get_eligible_interventions
from network_sampling.network.matsim_network import load_matsim_network
from network_sampling.network.street_segments import StreetSegmentIndex
from network_sampling.network.transit_usage import TransitUsageIndex
from network_sampling.sampling.cluster import sample_clusters
from network_sampling.sampling.corridor_mixed import sample_corridor_package
from network_sampling.sampling.random import sample_random
from network_sampling.validation.connectivity import check_connectivity

from network_sampling.interventions.matsim_app.apply_scenario import apply_matsim_scenario



DEFAULT_SETTINGS = dict(
    budget_m=5000.0,
    width_attribute="carriageway_width_m",
    cluster_length_m=1000.0,
    cluster_pbl_weight=2.0,
    cluster_destination_weight=0.5,
    corridor_pbl_weight=3.0,
    corridor_intrusion_strength=0.5,
    corridor_bus_share=0.15,
    corridor_destination_share=0.10,
    corridor_bus_length_m=1500.0,
)
# Budget strata are balanced within each strategy, even when generation is batched
# with --start-seed. Each individual budget is drawn uniformly within its stratum.
BUDGET_STRATA_M = (
    (500.0, 3000.0),
    (3000.0, 7000.0),
    (7000.0, 15000.0),
    (15000.0, 30000.0),
    (30000.0, 50000.0),
)

# These are sampler parameters that already exist in generate.py. Do not vary
# MATSim demand, calibration or connectivity/validity rules.
PARAMETER_RANGES = {
    "cluster_length_m": (500.0, 5000.0),
    "cluster_pbl_weight": (0.5, 4.5),
    "cluster_destination_weight": (0.3, 2.0),
    "corridor_pbl_weight": (0.5, 5.0),
    "corridor_intrusion_strength": (0.0, 2.5),
    "corridor_bus_share": (0.0, 0.25),
    "corridor_destination_share": (0.0, 0.25),
    "corridor_bus_length_m": (500.0, 5000.0),
}
PARAMETER_SAMPLING_VERSION = 2
STRATEGY_STRATUM_OFFSET = {"random": 0, "corridor": 1, "cluster": 2}

SCENARIO_FILES = (
    "network.xml.gz", "transit_schedule.xml.gz", "interventions.json",
    "link_mapping.csv.gz", "scenario.json",
)
SCHEMA_VERSION = 1


@dataclass
class Baseline:
    network: object
    index: StreetSegmentIndex
    usage: TransitUsageIndex
    eligible: list
    graph: nx.MultiGraph


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_baseline(network_path: Path, schedule_path: Path, settings: dict) -> Baseline:
    network = load_matsim_network(network_path)
    index = StreetSegmentIndex.from_network(network, width_attribute=settings["width_attribute"])
    usage = TransitUsageIndex.from_schedule(schedule_path, network, index)
    eligible = [iv for seg in index.all()
                for iv in get_eligible_interventions(seg, network, usage)]
    graph = nx.MultiGraph()
    for seg in index.all():
        graph.add_edge(seg.node_a, seg.node_b, key=seg.id,
                       segment_id=seg.id, length_m=seg.length_m)
    print(f"Baseline: {len(index)} street segments, {len(eligible)} intervention configurations", flush=True)
    return Baseline(network, index, usage, eligible, graph)


def sample(strategy: str, seed: int, base: Baseline, settings: dict):
    network, index, usage, eligible, graph = (
        base.network, base.index, base.usage, base.eligible, base.graph
    )
    budget = settings["budget_m"]
    if strategy == "baseline":
        return []
    if strategy == "random":
        return sample_random(eligible, index, network, budget, seed=seed)
    if strategy == "corridor":
        return sample_corridor_package(
            graph, index, network, eligible, budget, transit_usage=usage, seed=seed,
            bus_share=settings["corridor_bus_share"],
            destination_share=settings["corridor_destination_share"],
            max_bus_corridor_length_m=settings["corridor_bus_length_m"],
            intrusion_strength=settings["corridor_intrusion_strength"],
            pbl_weight_multiplier=settings["corridor_pbl_weight"],
        )
    if strategy == "cluster":
        return sample_clusters(
            graph, index, network, eligible, budget, seed=seed,
            max_cluster_length_m=settings["cluster_length_m"],
            pbl_weight_multiplier=settings["cluster_pbl_weight"],
            destination_weight_multiplier=settings["cluster_destination_weight"],
        )
    raise ValueError(f"Unknown strategy: {strategy}")


def _scenario_settings(strategy: str, seed: int, global_settings: dict) -> tuple[dict, dict | None]:
    """Draw one stable configuration per scenario (not per connectivity retry).

    Deterministic under job restarts, batching and different strategy orderings.
    No Python process-global hash or random state is involved.
    """
    distribution = global_settings.get("parameter_sampling")
    if distribution is None:
        return global_settings, None  # Preserve original fixed-settings behavior.

    if strategy == "baseline":
        return {
            "budget_m": 0.0,
            "width_attribute": global_settings["width_attribute"],
        }, {"budget_stratum_index": None, "budget_stratum_m": None}

    strata = distribution["budget_strata_m"]
    offset = STRATEGY_STRATUM_OFFSET[strategy]
    stratum_index = (seed + offset) % len(strata)
    low, high = strata[stratum_index]
    key = f"v{distribution['version']}:{distribution['seed']}:{strategy}:{seed}"
    rng = random.Random(int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest(), "big"))

    sampled = {
        "budget_m": rng.uniform(low, high),
        "width_attribute": global_settings["width_attribute"],
    }
    if strategy == "cluster":
        names = (
            "cluster_length_m", "cluster_pbl_weight", "cluster_destination_weight",
        )
    elif strategy == "corridor":
        names = (
            "corridor_pbl_weight", "corridor_intrusion_strength",
            "corridor_bus_share", "corridor_destination_share",
            "corridor_bus_length_m",
        )
    elif strategy == "random":
        names = ()  # sample_random currently exposes budget + seed only.
    else:
        raise ValueError(f"Unknown sampling strategy {strategy!r}")

    for name in names:
        lower, upper = distribution["ranges"][name]
        sampled[name] = rng.uniform(lower, upper)

    # Both sampled corridor shares are <= 0.25, hence sum <= 0.50.
    return sampled, {
        "budget_stratum_index": stratum_index,
        "budget_stratum_m": [low, high],
    }


def _intervention_summary(records: list[dict]) -> dict:
    by_type: dict[str, dict] = {}
    for entry in records:
        category = entry["intervention_type"]
        summary = by_type.setdefault(category, {"count": 0, "length_m": 0.0})
        summary["count"] += 1
        summary["length_m"] += entry["length_m"]
    return by_type


def _serialized_interventions(interventions, index):
    """Keep the exact applied intervention, not just its sampler seed."""
    return [{
        "segment_id": str(iv.segment_id),
        "intervention_type": iv.intervention_type.value,
        "road_space_variant": iv.road_space_variant.value if iv.road_space_variant else None,
        "target_link_ids": [str(lid) for lid in iv.target_link_ids],
        "length_m": float(index.get(iv.segment_id).length_m),
    } for iv in interventions]


def _dataset_manifest(out: Path, network_path: Path, schedule_path: Path, settings: dict):
    """Refuse to mix baselines/sampling settings in one training dataset."""
    identity = {
        "schema_version": SCHEMA_VERSION,
        "baseline_network": str(network_path),
        "baseline_schedule": str(schedule_path),
        "baseline_hashes": {
            "network": _sha256(network_path),
            "schedule": _sha256(schedule_path),
        },
        "settings": settings,
    }
    path = out / "dataset_manifest.json"
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != identity:
            raise RuntimeError("Dataset has different baseline inputs or sampling settings. Use a new --out directory.")
    elif any(p.is_dir() and (p / "scenario.json").exists() for p in out.iterdir()):
        raise RuntimeError("Dataset contains scenarios but no dataset_manifest.json. Use a clean --out directory.")
    else:
        # The user requested fresh datasets, not backfill support.
        path.write_text(json.dumps(identity, indent=2) + "\n", encoding="utf-8")
    return identity


def generate_one(strategy: str, seed: int, out: Path, schedule: Path,
                 base: Baseline, identity: dict, max_attempts: int) -> None:
    scenario_id = f"{strategy}_{seed:05d}"
    settings, stratum_info = _scenario_settings(strategy, seed, identity["settings"])
    randomize = stratum_info is not None
    target = out / scenario_id
    if target.exists():
        if not all((target / f).is_file() for f in SCENARIO_FILES):
            raise RuntimeError(f"Incomplete existing scenario: {target}")
        old = json.loads((target / "scenario.json").read_text(encoding="utf-8"))
        same_settings = (old.get("settings") == settings)
        if randomize:
            same_settings &= old.get("generation_settings") == identity["settings"]
        if old.get("baseline_hashes") != identity["baseline_hashes"] or not same_settings:
            raise RuntimeError(f"Existing {scenario_id} uses different input files/settings")
        print(f"SKIP {scenario_id}", flush=True)
        return

    with tempfile.TemporaryDirectory(prefix=f".{scenario_id}_", dir=out) as temp:
        stage = Path(temp)
        network_file = stage / "network.xml.gz"
        schedule_file = stage / "transit_schedule.xml.gz"
        reason = "No valid intervention sample"
        for attempt in range(1 if strategy == "baseline" else max_attempts):
            actual_seed = seed + attempt * 1_000_003
            interventions = sample(strategy, actual_seed, base, settings)
            if strategy != "baseline" and not interventions:
                reason = "Sampler returned no interventions"
                continue
            records = _serialized_interventions(interventions, base.index)
            used_m = sum(row["length_m"] for row in records)
            if used_m > settings["budget_m"] + 1e-6:
                raise RuntimeError(f"Sampler exceeded budget in {scenario_id}: {used_m:.3f}m")

            network_file.unlink(missing_ok=True)
            schedule_file.unlink(missing_ok=True)
            try:
                apply_matsim_scenario(base.network, schedule, base.index, interventions,
                                      network_file, schedule_file)
            except ValueError as exc:
                if not str(exc).startswith("Interventions disconnect the following modes:"):
                    raise
                reason = str(exc)
                print(f"REJECT {scenario_id} attempt={attempt + 1}: {reason}", flush=True)
                continue

            if not network_file.is_file() or not schedule_file.is_file():
                raise RuntimeError(f"MATSim network or schedule missing for {scenario_id}")
            modified = load_matsim_network(network_file)
            connectivity = check_connectivity(base.network, modified)
            if not connectivity or not all(connectivity.values()):
                reason = f"Final connectivity check failed: {connectivity}"
                print(f"REJECT {scenario_id} attempt={attempt + 1}: {reason}", flush=True)
                continue
            # No train-time graph features or target arrays are written here.
            n_mapped = write_link_mapping(stage / "link_mapping.csv.gz", base.index,
                                          base.network, modified)
            (stage / "interventions.json").write_text(
                json.dumps(records, indent=2) + "\n", encoding="utf-8"
            )
            metadata = {
                "schema_version": SCHEMA_VERSION,
                "id": scenario_id, "strategy": strategy, "seed": seed,
                "effective_seed": actual_seed, "attempts": attempt + 1,
                "budget_used_m": used_m,
                "intervention_count": len(records),
                "mapped_road_links": n_mapped,
                "connectivity": connectivity,
                "baseline_hashes": identity["baseline_hashes"],
                "settings": settings,
            }
            if randomize:
                metadata.update({
                    "generation_settings": identity["settings"],
                    "requested_budget_m": settings["budget_m"],
                    "realized_budget_m": used_m,
                    "budget_utilization": (used_m / settings["budget_m"]
                                           if settings["budget_m"] else None),
                    "parameter_sampling": stratum_info,
                    "interventions_by_type": _intervention_summary(records),
                })
            (stage / "scenario.json").write_text(
                json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
            )
            stage.rename(target)  # Publish only after all validations and files succeeded.
            print(f"READY {scenario_id}: {len(records)} interventions; {used_m:.1f} m; "
                  f"{n_mapped} mapped road links", flush=True)
            return
    raise RuntimeError(f"No valid sample for {scenario_id} after {max_attempts} attempts: {reason}")



# Each generation task publishes its own scenario directory atomically.
# The baseline is prepared once in the parent and inherited by workers via
# fork on Linux; it is intentionally not pickled or rebuilt per scenario.
_PARALLEL_CONTEXT = None


def _generate_worker(task: tuple[str, int]) -> None:
    if _PARALLEL_CONTEXT is None:
        raise RuntimeError("Generation worker missing inherited baseline context")
    base, schedule_path, out, identity, max_attempts = _PARALLEL_CONTEXT
    strategy, seed = task
    generate_one(strategy, seed, out, schedule_path, base, identity, max_attempts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--network", required=True, type=Path)
    parser.add_argument("--schedule", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--samples", type=int, default=10, help="Samples PER strategy; baseline only once")
    parser.add_argument("--start-seed", type=int, default=0)
    parser.add_argument("--max-attempts", type=int, default=30)
    parser.add_argument("--workers", type=int, default=1, help="Parallel scenario-generation processes (Linux fork; default 1)")
    parser.add_argument("--budget", type=float, default=DEFAULT_SETTINGS["budget_m"])
    parser.add_argument("--randomize-parameters", action="store_true",
                        help="Stratify budget (0.5–7 km) and randomize existing strategy-specific sampler parameters")
    parser.add_argument("--parameter-seed", type=int, default=42,
                        help="Deterministic seed for scenario parameter draws (independent of sampler seeds)")
    # Expose the existing sampler settings so experiments are reproducible from CLI.
    parser.add_argument("--width-attribute", default=DEFAULT_SETTINGS["width_attribute"])
    parser.add_argument("--cluster-length", type=float, default=DEFAULT_SETTINGS["cluster_length_m"])
    parser.add_argument("--cluster-pbl-weight", type=float, default=DEFAULT_SETTINGS["cluster_pbl_weight"])
    parser.add_argument("--cluster-destination-weight", type=float, default=DEFAULT_SETTINGS["cluster_destination_weight"])
    parser.add_argument("--corridor-pbl-weight", type=float, default=DEFAULT_SETTINGS["corridor_pbl_weight"])
    parser.add_argument("--corridor-intrusion-strength", type=float, default=DEFAULT_SETTINGS["corridor_intrusion_strength"])
    parser.add_argument("--corridor-bus-share", type=float, default=DEFAULT_SETTINGS["corridor_bus_share"])
    parser.add_argument("--corridor-destination-share", type=float, default=DEFAULT_SETTINGS["corridor_destination_share"])
    parser.add_argument("--corridor-bus-length", type=float, default=DEFAULT_SETTINGS["corridor_bus_length_m"])
    parser.add_argument("--strategies", nargs="+", choices=("baseline", "random", "corridor", "cluster"),
                        default=("baseline", "random", "corridor", "cluster"))
    args = parser.parse_args()
    if args.samples < 1 or args.start_seed < 0 or args.max_attempts < 1 or args.budget < 0:
        parser.error("--samples and --max-attempts must be positive; --start-seed and --budget nonnegative")
    if args.parameter_seed < 0:
        parser.error("--parameter-seed must be nonnegative")
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.randomize_parameters and args.budget != DEFAULT_SETTINGS["budget_m"]:
        parser.error("--budget is overridden by --randomize-parameters; edit BUDGET_STRATA_M instead")

    network_path = args.network.resolve(strict=True)
    schedule_path = args.schedule.resolve(strict=True)
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    settings = dict(
        budget_m=float(args.budget),
        width_attribute=args.width_attribute,
        cluster_length_m=args.cluster_length,
        cluster_pbl_weight=args.cluster_pbl_weight,
        cluster_destination_weight=args.cluster_destination_weight,
        corridor_pbl_weight=args.corridor_pbl_weight,
        corridor_intrusion_strength=args.corridor_intrusion_strength,
        corridor_bus_share=args.corridor_bus_share,
        corridor_destination_share=args.corridor_destination_share,
        corridor_bus_length_m=args.corridor_bus_length,
    )
    if not 0 <= args.corridor_bus_share <= 1 or not 0 <= args.corridor_destination_share <= 1:
        parser.error("corridor shares must be in [0,1]")
    if args.corridor_bus_share + args.corridor_destination_share > 1:
        parser.error("corridor bus/destination shares cannot sum above 1")
    if args.randomize_parameters:
        settings["parameter_sampling"] = {
            "version": PARAMETER_SAMPLING_VERSION,
            "seed": args.parameter_seed,
            "budget_strata_m": [list(bounds) for bounds in BUDGET_STRATA_M],
            "ranges": {name: list(bounds) for name, bounds in PARAMETER_RANGES.items()},
        }
    identity = _dataset_manifest(out, network_path, schedule_path, settings)
    base = load_baseline(network_path, schedule_path, settings)

    segment_file = out / "street_segments.csv.gz"
    # Do not silently overwrite a physical index when resuming a dataset.
    if not segment_file.exists():
        n = write_street_segments(segment_file, base.index)
        print(f"Saved {n} baseline physical street segments", flush=True)

    tasks = [
        (strategy, seed)
        for strategy in dict.fromkeys(args.strategies)
        for seed in ([0] if strategy == "baseline" else
                     range(args.start_seed, args.start_seed + args.samples))
    ]
    if args.workers == 1:
        for strategy, seed in tasks:
            generate_one(strategy, seed, out, schedule_path, base, identity, args.max_attempts)
    else:
        # Linux/HPC only: fork lets workers inherit the read-only baseline
        # without reloading it for each task. Every task has a unique directory.
        global _PARALLEL_CONTEXT
        _PARALLEL_CONTEXT = (base, schedule_path, out, identity, args.max_attempts)
        print(f"Generating {len(tasks)} scenarios with {args.workers} processes", flush=True)
        with ProcessPoolExecutor(
            max_workers=args.workers,
            mp_context=mp.get_context("fork"),
        ) as pool:
            for _ in pool.map(_generate_worker, tasks, chunksize=1):
                pass
        _PARALLEL_CONTEXT = None

    names = sorted(p.name for p in out.iterdir()
                   if p.is_dir() and (p / "scenario.json").is_file())
    (out / "scenarios.txt").write_text("\n".join(names) + "\n", encoding="utf-8")
    print(f"Done: {len(names)} scenarios in {out}", flush=True)


if __name__ == "__main__":
    main()

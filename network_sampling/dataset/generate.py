"""Step 1: Generate valid MATSim scenarios from existing network_sampling code.

Only samples interventions, applies them with matsim_app, validates and saves.
No route checking, simulation, output extraction or GNN code belongs here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path

import networkx as nx

from network_sampling.interventions.eligibility import get_eligible_interventions
from network_sampling.interventions.matsim_app.apply_scenario import apply_matsim_scenario
from network_sampling.network.matsim_network import load_matsim_network
from network_sampling.network.street_segments import StreetSegmentIndex
from network_sampling.network.transit_usage import TransitUsageIndex
from network_sampling.sampling.cluster import sample_clusters
from network_sampling.sampling.corridor_mixed import sample_corridor_package
from network_sampling.sampling.random import sample_random
from network_sampling.validation.connectivity import check_connectivity


# These are the sampling settings used for the entire dataset.
SETTINGS = dict(
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
RANDOM_MAX_ATTEMPTS = 30
FILES = ("network.xml.gz", "transit_schedule.xml.gz", "interventions.json", "scenario.json")


@dataclass
class Baseline:
    network: object
    index: StreetSegmentIndex
    usage: TransitUsageIndex
    eligible: list
    graph: nx.MultiGraph


def load_baseline(network_path: Path, schedule_path: Path) -> Baseline:
    """Load and prepare shared sampling inputs just once."""
    network = load_matsim_network(network_path)
    index = StreetSegmentIndex.from_network(
        network, width_attribute=SETTINGS["width_attribute"]
    )
    usage = TransitUsageIndex.from_schedule(schedule_path, network, index)
    eligible = [iv for seg in index.all()
                for iv in get_eligible_interventions(seg, network, usage)]
    graph = nx.MultiGraph()
    for seg in index.all():
        graph.add_edge(seg.node_a, seg.node_b, key=seg.id, segment_id=seg.id)
    print(f"Baseline: {len(index)} street segments, {len(eligible)} configurations")
    return Baseline(network, index, usage, eligible, graph)


def sample(strategy: str, seed: int, base: Baseline):
    """Call the already implemented samplers, without changing their logic."""
    network, index, usage, eligible, graph = (
        base.network, base.index, base.usage, base.eligible, base.graph
    )
    budget = SETTINGS["budget_m"]
    if strategy == "baseline":
        return []
    if strategy == "random":
        return sample_random(eligible, index, network, budget, seed=seed)
    if strategy == "corridor":
        return sample_corridor_package(
            graph, index, network, eligible, budget, transit_usage=usage, seed=seed,
            bus_share=SETTINGS["corridor_bus_share"],
            destination_share=SETTINGS["corridor_destination_share"],
            max_bus_corridor_length_m=SETTINGS["corridor_bus_length_m"],
            intrusion_strength=SETTINGS["corridor_intrusion_strength"],
            pbl_weight_multiplier=SETTINGS["corridor_pbl_weight"],
        )
    if strategy == "cluster":
        return sample_clusters(
            graph, index, network, eligible, budget, seed=seed,
            max_cluster_length_m=SETTINGS["cluster_length_m"],
            pbl_weight_multiplier=SETTINGS["cluster_pbl_weight"],
            destination_weight_multiplier=SETTINGS["cluster_destination_weight"],
        )
    raise ValueError(f"Unknown strategy: {strategy}")


def generate_one(
    strategy: str, seed: int, out: Path, schedule: Path,
    base: Baseline, baseline_hashes: dict,
):
    """Generate one scenario; publish the folder only after it passes checks."""
    scenario_id = f"{strategy}_{seed:05d}"
    target = out / scenario_id
    if target.exists():
        if not all((target / name).is_file() for name in FILES):
            raise RuntimeError(f"Incomplete existing scenario: {target}")
        manifest = json.loads((target / "scenario.json").read_text())
        if manifest["settings"] != SETTINGS or manifest["baseline_hashes"] != baseline_hashes:
            raise RuntimeError(f"Existing {scenario_id} uses other inputs/settings")
        print(f"SKIP {scenario_id}")
        return

    # A failed scenario is not committed to the dataset.
    with tempfile.TemporaryDirectory(dir=out, prefix=f".{scenario_id}_") as temp:
        stage = Path(temp)
        network_file = stage / "network.xml.gz"
        schedule_file = stage / "transit_schedule.xml.gz"
        n_attempts = RANDOM_MAX_ATTEMPTS if strategy == "random" else 1
        for attempt in range(n_attempts):
            effective_seed = seed + attempt * 1_000_003
            interventions = sample(strategy, effective_seed, base)
            if strategy != "baseline" and not interventions:
                continue
            used_m = sum(base.index.get(iv.segment_id).length_m for iv in interventions)
            if used_m > SETTINGS["budget_m"] + 1e-6:
                raise RuntimeError(f"Budget exceeded: {scenario_id} ({used_m:.1f} m)")

            network_file.unlink(missing_ok=True)
            schedule_file.unlink(missing_ok=True)
            apply_matsim_scenario(
                base.network, schedule, base.index, interventions, network_file, schedule_file
            )
            if not network_file.is_file() or not schedule_file.is_file():
                raise RuntimeError(f"Missing modified XML in {scenario_id}")
            connectivity = check_connectivity(base.network, load_matsim_network(network_file))
            if connectivity and all(connectivity.values()):
                break
        else:
            raise RuntimeError(f"No valid {strategy} sample for {scenario_id}")

        (stage / "interventions.json").write_text(json.dumps([
            dict(
                segment_id=iv.segment_id,
                intervention_type=iv.intervention_type.value,
                road_space_variant=(iv.road_space_variant.value if iv.road_space_variant else None),
                target_link_ids=list(iv.target_link_ids),
            ) for iv in interventions
        ], indent=2) + "\n")
        (stage / "scenario.json").write_text(json.dumps(dict(
            id=scenario_id, strategy=strategy, seed=seed, effective_seed=effective_seed,
            attempts=attempt + 1, intervention_count=len(interventions),
            budget_used_m=round(used_m, 3), connectivity=connectivity,
            baseline_hashes=baseline_hashes, settings=SETTINGS,
        ), indent=2) + "\n")
        stage.rename(target)
    print(f"READY {scenario_id}: {len(interventions)} interventions, {used_m:.0f} m")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--network", required=True, type=Path)
    parser.add_argument("--schedule", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--samples", type=int, default=10,
                        help="Number of samples per strategy; baseline only once")
    parser.add_argument("--start-seed", type=int, default=0)
    parser.add_argument("--strategies", nargs="+",
                        choices=("baseline", "random", "corridor", "cluster"),
                        default=("baseline", "random", "corridor", "cluster"))
    args = parser.parse_args()
    if args.samples < 1 or args.start_seed < 0:
        parser.error("--samples must be >=1 and --start-seed must be >=0")

    network_path = args.network.resolve(strict=True)
    schedule_path = args.schedule.resolve(strict=True)
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    base = load_baseline(network_path, schedule_path)
    hashes = {"network": _sha256(network_path), "schedule": _sha256(schedule_path)}

    for strategy in args.strategies:
        seeds = [0] if strategy == "baseline" else range(
            args.start_seed, args.start_seed + args.samples
        )
        for seed in seeds:
            generate_one(strategy, seed, out, schedule_path, base, hashes)

    names = sorted(p.name for p in out.iterdir()
                   if p.is_dir() and (p / "scenario.json").is_file())
    (out / "scenarios.txt").write_text("\n".join(names) + "\n")
    print(f"Done: {len(names)} scenarios indexed in {out / 'scenarios.txt'}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    main()

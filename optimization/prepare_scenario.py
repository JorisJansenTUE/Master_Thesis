"""Step 2: prepare one GA individual as a standard MATSim dataset scenario.

No MATSim simulation is executed. Defaults can be edited below, or overridden:
    python -m optimization.prepare_scenario
    python -m optimization.prepare_scenario --candidate crossover --seed 42
    python -m optimization.prepare_scenario --interventions-json PATH

The resulting scenario can later be evaluated by the existing run_one.py.
"""
from __future__ import annotations

import argparse
import json
import random
import tempfile
from pathlib import Path

from network_sampling.dataset.generate import (
    DEFAULT_SETTINGS, SCENARIO_FILES, SCHEMA_VERSION,
    _dataset_manifest, _intervention_summary, load_baseline,
    write_intervention_scenario_files,
)
from network_sampling.dataset.link_mapping import write_street_segments
from network_sampling.interventions.intervention_types import (
    Intervention, InterventionType, RoadSpaceVariant,
)
from optimization.ga import (
    configuration_hash, crossover, initial_population, mutate, validate_individual,
)
from preprocessing.utils import PROJECT_ROOT

# ------------------------- Editable settings -------------------------------
BASELINE_DIR = PROJECT_ROOT / "data" / "Simulated_Runs_v2" / "baseline_00000"
NETWORK_FILE = BASELINE_DIR / "network.xml.gz"
SCHEDULE_FILE = BASELINE_DIR / "transit_schedule.xml.gz"
OUTPUT_DIR = PROJECT_ROOT / "data" / "GA_Pilot"  # Separate from the existing training dataset!
WIDTH_ATTRIBUTE = "carriageway_width_m"
BUDGET_M = 5000.0
SEED = 42
CANDIDATE = "mutation"  # initial, mutation, crossover
# ---------------------------------------------------------------------------


def read_interventions(path: Path) -> list[Intervention]:
    """Read the existing interventions.json format (no new schema)."""
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError("interventions.json must contain a list")
    return [
        Intervention(
            segment_id=str(row["segment_id"]),
            intervention_type=InterventionType(row["intervention_type"]),
            road_space_variant=(RoadSpaceVariant(row["road_space_variant"])
                                if row.get("road_space_variant") is not None else None),
            target_link_ids=tuple(str(link) for link in row.get("target_link_ids", [])),
        ) for row in rows
    ]


def make_example_individual(base, settings: dict, seed: int, kind: str):
    """For this local test only; the eventual GA will supply its own individual."""
    rng = random.Random(seed)
    parents = initial_population(
        base, size=2 if kind == "crossover" else 1,
        budget_m=settings["budget_m"], rng=rng, settings=settings,
    )
    if kind == "initial":
        return parents[0]
    if kind == "mutation":
        return mutate(parents[0], eligible=base.eligible, network=base.network,
                      index=base.index, budget_m=settings["budget_m"], rng=rng)
    if kind == "crossover":
        return crossover(parents[0], parents[1], eligible=base.eligible,
                         network=base.network, index=base.index,
                         budget_m=settings["budget_m"], rng=rng)
    raise ValueError(f"Unknown candidate kind: {kind}")


def prepare_one(network_path: Path, schedule_path: Path, out: Path,
                interventions: list[Intervention], base, settings: dict,
                source: str) -> Path:
    """Publish one self-contained MATSim scenario; never overwrite completed work."""
    out.mkdir(parents=True, exist_ok=True)
    identity = _dataset_manifest(out, network_path, schedule_path, settings)
    segment_file = out / "street_segments.csv.gz"
    if not segment_file.is_file():
        write_street_segments(segment_file, base.index)

    ok, reason = validate_individual(
        interventions, eligible=base.eligible, network=base.network,
        index=base.index, budget_m=settings["budget_m"],
    )
    if not ok:
        raise ValueError(f"GA individual failed validation: {reason}")

    fingerprint = configuration_hash(interventions)
    scenario_id = "ga_" + fingerprint[:16]
    target = out / scenario_id
    if target.exists():
        if not all((target / name).is_file() for name in SCENARIO_FILES):
            raise RuntimeError(f"Existing incomplete scenario, refusing to overwrite: {target}")
        old = json.loads((target / "scenario.json").read_text(encoding="utf-8"))
        if (old.get("config_hash") != fingerprint or
                old.get("baseline_hashes") != identity["baseline_hashes"] or
                old.get("settings") != settings):
            raise RuntimeError(f"Existing scenario has different identity: {target}")
        print(f"SKIP already prepared: {target}")
    else:
        with tempfile.TemporaryDirectory(prefix=f".{scenario_id}_", dir=out) as temp:
            stage = Path(temp)
            records, used_m, mapped, connectivity = write_intervention_scenario_files(
                stage, base=base, schedule=schedule_path, interventions=interventions,
                budget_m=settings["budget_m"],
            )
            metadata = {
                "schema_version": SCHEMA_VERSION,
                "id": scenario_id,
                "strategy": "ga",
                "source": source,
                "config_hash": fingerprint,
                "budget_used_m": used_m,
                "requested_budget_m": settings["budget_m"],
                "budget_utilization": (used_m / settings["budget_m"]
                                       if settings["budget_m"] else None),
                "intervention_count": len(records),
                "interventions_by_type": _intervention_summary(records),
                "mapped_road_links": mapped,
                "connectivity": connectivity,
                "baseline_hashes": identity["baseline_hashes"],
                "settings": settings,
            }
            (stage / "scenario.json").write_text(
                json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
            )
            stage.rename(target)
        print(f"READY {scenario_id}: {len(interventions)} interventions, "
              f"{used_m:.1f} m/{settings['budget_m']:.1f} m; {mapped} mapped links")

    # Compatibility with dataset generation's scenario index. No SLURM jobs yet.
    scenario_names = sorted(p.name for p in out.iterdir()
                            if p.is_dir() and (p / "scenario.json").is_file())
    (out / "scenarios.txt").write_text("\n".join(scenario_names) + "\n",
                                        encoding="utf-8")
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--network", type=Path, default=NETWORK_FILE)
    parser.add_argument("--schedule", type=Path, default=SCHEDULE_FILE)
    parser.add_argument("--out", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--budget", type=float, default=BUDGET_M)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--candidate", choices=("initial", "mutation", "crossover"),
                        default=CANDIDATE)
    parser.add_argument("--interventions-json", type=Path,
                        help="Use an existing interventions.json instead of creating a test candidate")
    parser.add_argument("--width-attribute", default=WIDTH_ATTRIBUTE)
    args = parser.parse_args()
    if args.budget < 0:
        parser.error("--budget must be nonnegative")
    network_path = args.network.resolve(strict=True)
    schedule_path = args.schedule.resolve(strict=True)
    settings = {**DEFAULT_SETTINGS, "budget_m": float(args.budget),
                "width_attribute": args.width_attribute}

    print("Loading baseline and eligible interventions...")
    base = load_baseline(network_path, schedule_path, settings)
    if args.interventions_json is not None:
        individual = read_interventions(args.interventions_json)
        source = str(args.interventions_json)
    else:
        individual = make_example_individual(base, settings, args.seed, args.candidate)
        source = f"{args.candidate}, seed={args.seed}"

    target = prepare_one(network_path, schedule_path, args.out.resolve(),
                         individual, base, settings, source)
    print("Scenario directory:", target)
    print("No MATSim simulation was run.")


if __name__ == "__main__":
    main()

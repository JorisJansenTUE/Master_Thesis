"""Local GA inspection on the actual baseline; no MATSim run or output files.

From the Master_Thesis repository root:
    python -m optimization.inspect_ga --network PATH --schedule PATH

You may edit the default paths/settings below instead of using CLI flags.
"""
from __future__ import annotations

import argparse
import random
from collections import Counter
from pathlib import Path

from network_sampling.dataset.generate import DEFAULT_SETTINGS, load_baseline
from network_sampling.sampling.random import intervention_length
from optimization.ga import (
    configuration_hash, crossover, initial_population, mutate,
    validate_individual,
)

# ------------------------ Editable settings ------------------------
BASELINE_DIR = Path("data/Simulated_Runs_v2/baseline_00000")
NETWORK_FILE = BASELINE_DIR / "network.xml.gz"
SCHEDULE_FILE = BASELINE_DIR / "transit_schedule.xml.gz"
WIDTH_ATTRIBUTE = "carriageway_width_m"
BUDGET_M = 5000.0
POPULATION_SIZE = 6
CHILDREN_TO_TEST = 12
RANDOM_SEED = 42
# -------------------------------------------------------------------


def _summary(label, individuals, base, budget_m):
    ids = [configuration_hash(iv) for iv in individuals]
    counts = [len(iv) for iv in individuals]
    lengths = [intervention_length(iv, base.index) for iv in individuals]
    print(f"{label}: {len(individuals)} individuals; {len(set(ids))} unique")
    print(f"  interventions: min={min(counts)} max={max(counts)} mean={sum(counts)/len(counts):.1f}")
    print(f"  budget used:   min={min(lengths):.1f}m max={max(lengths):.1f}m "
          f"mean={sum(lengths)/len(lengths):.1f}m of {budget_m:.1f}m")
    for num, (iv, identifier, length) in enumerate(zip(individuals, ids, lengths), 1):
        kinds = dict(sorted(Counter(x.intervention_type.value for x in iv).items()))
        print(f"  {num:2d}: {identifier[:12]}  n={len(iv):3d}  "
              f"budget={length:8.1f}m  types={kinds}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--network", type=Path, default=NETWORK_FILE)
    parser.add_argument("--schedule", type=Path, default=SCHEDULE_FILE)
    parser.add_argument("--budget", type=float, default=BUDGET_M)
    parser.add_argument("--population", type=int, default=POPULATION_SIZE)
    parser.add_argument("--children", type=int, default=CHILDREN_TO_TEST)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--width-attribute", default=WIDTH_ATTRIBUTE)
    args = parser.parse_args()
    if args.population < 2 or args.children < 1 or args.budget <= 0:
        parser.error("--population >= 2, --children >= 1 and --budget > 0 are required")
    if not args.network.is_file() or not args.schedule.is_file():
        parser.error("Network/schedule missing: set NETWORK_FILE and SCHEDULE_FILE "
                     "at the top of the script, or pass --network/--schedule")

    rng = random.Random(args.seed)
    settings = dict(DEFAULT_SETTINGS)
    settings["budget_m"] = args.budget
    settings["width_attribute"] = args.width_attribute
    print("Loading baseline and eligible interventions...")
    base = load_baseline(args.network, args.schedule, settings)
    parents = initial_population(base, size=args.population, budget_m=args.budget,
                                 rng=rng, settings=settings)
    _summary("Initial population", parents, base, args.budget)

    children = []
    mutation_changes = 0
    for i in range(args.children):
        if i % 2 == 0:
            parent = parents[rng.randrange(len(parents))]
            child = mutate(parent, eligible=base.eligible, network=base.network,
                           index=base.index, budget_m=args.budget, rng=rng)
            mutation_changes += configuration_hash(child) != configuration_hash(parent)
            label = "mutation"
        else:
            a, b = rng.sample(parents, 2)
            child = crossover(a, b, eligible=base.eligible, network=base.network,
                              index=base.index, budget_m=args.budget, rng=rng)
            label = "crossover"
        valid, reason = validate_individual(child, eligible=base.eligible,
                                            network=base.network, index=base.index,
                                            budget_m=args.budget)
        if not valid:
            raise AssertionError(f"Child {i+1} ({label}) invalid: {reason}")
        children.append(child)
    print()
    _summary("Valid offspring", children, base, args.budget)
    print(f"\nMutation changes: {mutation_changes}/{(args.children+1)//2}")
    print(f"Unique across parents and offspring: "
          f"{len({configuration_hash(x) for x in parents + children})}")
    print("PASS: all offspring eligible, compatible, within budget, and connected "
          "for car/bike/truck.")
    print("No MATSim simulations or scenario files were created.")


if __name__ == "__main__":
    main()

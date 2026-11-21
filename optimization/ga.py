"""Small, source-compatible genetic operators for MATSim intervention lists.

An individual is a list[network_sampling.interventions.intervention_types.Intervention].
No networks, scenarios, or MATSim jobs are created here.
"""
from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict

from network_sampling.interventions.compatibility import interventions_compatible
from network_sampling.interventions.intervention_types import Intervention, RoadSpaceVariant
from network_sampling.interventions.matsim_app.apply_network import apply_interventions
from network_sampling.sampling.random import intervention_length
from network_sampling.validation.connectivity import check_connectivity


def configuration_hash(individual: list[Intervention]) -> str:
    """Order-independent ID for an intervention *configuration*.

    Intended for comparisons within one baseline/model version. Before reusing
    MATSim results, the evaluation system must also check baseline/settings IDs.
    """
    records = [
        (
            str(iv.segment_id),
            iv.intervention_type.value,
            iv.road_space_variant.value if iv.road_space_variant is not None else None,
            tuple(sorted(str(link) for link in iv.target_link_ids)),
        )
        for iv in individual
    ]
    data = json.dumps(sorted(records, key=lambda row: json.dumps(row)), separators=(",", ":"))
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def _basic_check(individual, *, eligible, network, index, budget_m) -> str | None:
    """Cheap checks before applying a full MATSim network."""
    if budget_m < 0:
        raise ValueError("budget_m must be non-negative")
    if len(individual) != len(set(individual)):
        return "duplicate intervention"
    allowed = set(eligible)
    if any(iv not in allowed for iv in individual):
        return "intervention not in baseline eligibility list"
    if intervention_length(individual, index) > budget_m + 1e-6:
        return "road-space budget exceeded"

    # Compatible bus lanes in opposite directions can share a segment.
    groups: dict[str, list[Intervention]] = defaultdict(list)
    for iv in individual:
        for existing in groups[iv.segment_id]:
            if not interventions_compatible(iv, existing, network):
                return "incompatible interventions on the same segment"
        groups[iv.segment_id].append(iv)
    return None


def validate_individual(
    individual: list[Intervention], *, eligible: list[Intervention],
    network, index, budget_m: float,
) -> tuple[bool, str]:
    """Validate eligibility, compatibility, budget and directed modal connectivity.

    Uses the established network_sampling application and connectivity rules.
    An empty list is allowed (valid baseline configuration).
    """
    problem = _basic_check(individual, eligible=eligible, network=network,
                           index=index, budget_m=budget_m)
    if problem is not None:
        return False, problem
    try:
        modified, _ = apply_interventions(network, index, individual)
        connected = check_connectivity(network, modified)
    except ValueError as exc:
        # A disconnected baseline is an invalid input, not a repairable child.
        if str(exc).startswith("Baseline "):
            raise
        return False, str(exc)
    failed = [mode for mode, ok in connected.items() if not ok]
    if failed:
        return False, "disconnected modes: " + ", ".join(failed)
    return True, "ok"


def repair_individual(
    proposed: list[Intervention], *, eligible: list[Intervention],
    network, index, budget_m: float,
) -> list[Intervention]:
    """Repair by dropping conflicts/over-budget genes, then car-removal genes.

    Conservative rather than optimal: no new intervention is invented. The
    original order is respected, so operators can put preferred genes first.
    """
    allowed = set(eligible)
    chosen: list[Intervention] = []
    groups: dict[str, list[Intervention]] = defaultdict(list)
    used = 0.0
    for iv in proposed:
        if iv not in allowed or iv in chosen:
            continue
        cost = index.get(iv.segment_id).length_m
        if used + cost > budget_m + 1e-6:
            continue
        if not all(interventions_compatible(iv, old, network) for old in groups[iv.segment_id]):
            continue
        chosen.append(iv)
        groups[iv.segment_id].append(iv)
        used += cost

    while True:
        ok, _ = validate_individual(chosen, eligible=eligible, network=network,
                                    index=index, budget_m=budget_m)
        if ok:
            return chosen
        # Only REMOVE_CAR changes can disconnect baseline car/truck access.
        removable = [i for i, iv in enumerate(chosen)
                     if iv.road_space_variant == RoadSpaceVariant.REMOVE_CAR]
        if not removable:
            # Defensive fallback: if another issue arose, do not return an
            # invalid candidate or hide it as a successful repair.
            raise ValueError("Cannot repair offspring with current operators")
        chosen.pop(removable[-1])


def mutate(
    parent: list[Intervention], *, eligible: list[Intervention], network,
    index, budget_m: float, rng: random.Random, max_attempts: int = 15,
) -> list[Intervention]:
    """One random add, remove or replace mutation; return a valid offspring.

    If every trial fails, return the unchanged parent (also valid).
    """
    if max_attempts < 1:
        raise ValueError("max_attempts must be positive")
    parent_key = configuration_hash(parent)
    for _ in range(max_attempts):
        operation = rng.choice(("add", "remove", "replace", "replace"))
        proposed = list(parent)
        if operation == "remove":
            if not proposed:
                continue
            proposed.pop(rng.randrange(len(proposed)))
        elif operation == "add":
            if not eligible:
                continue
            proposed.append(rng.choice(eligible))
        else:  # replace an existing gene, usually on the same segment
            if not proposed or not eligible:
                continue
            pos = rng.randrange(len(proposed))
            old = proposed.pop(pos)
            local = [iv for iv in eligible if iv.segment_id == old.segment_id
                     and iv != old]
            options = local if local and rng.random() < 0.5 else eligible
            proposed.insert(pos, rng.choice(options))

        # Prefer the intended new gene when repairing; avoid arbitrary
        # order effects during duplicate/compatibility removal.
        if operation == "add":
            proposed = [proposed[-1], *proposed[:-1]]
        child = repair_individual(proposed, eligible=eligible, network=network,
                                  index=index, budget_m=budget_m)
        if configuration_hash(child) != parent_key:
            return child
    return list(parent)


def crossover(
    a: list[Intervention], b: list[Intervention], *,
    eligible: list[Intervention], network, index, budget_m: float,
    rng: random.Random,
) -> list[Intervention]:
    """Uniform crossover of variable-length intervention lists, with repair."""
    selected = [iv for iv in a if rng.random() < 0.5]
    selected.extend(iv for iv in b if rng.random() < 0.5)
    rng.shuffle(selected)
    return repair_individual(selected, eligible=eligible, network=network,
                             index=index, budget_m=budget_m)


def initial_population(
    base, *, size: int, budget_m: float, rng: random.Random,
    settings: dict, initial: list[list[Intervention]] | None = None,
    max_attempts: int | None = None,
) -> list[list[Intervention]]:
    """Distinct, validated population from optional seeds and existing samplers.

    The 'initial' argument permits warm starts later without changing the GA.
    No fitness values are inferred here.
    """
    from network_sampling.dataset.generate import sample

    if size < 1:
        raise ValueError("size must be positive")
    settings = {**settings, "budget_m": float(budget_m)}
    population: list[list[Intervention]] = []
    seen: set[str] = set()

    def consider(candidate: list[Intervention]) -> bool:
        if not candidate:
            return False
        identifier = configuration_hash(candidate)
        if identifier in seen:
            return False
        valid, _ = validate_individual(candidate, eligible=base.eligible,
                                       network=base.network, index=base.index,
                                       budget_m=budget_m)
        if not valid:
            return False
        population.append(list(candidate))
        seen.add(identifier)
        return True

    for candidate in initial or []:
        consider(candidate)
        if len(population) >= size:
            return population

    strategies = ("random", "corridor", "cluster")
    attempts = size * 40 if max_attempts is None else max_attempts
    for attempt in range(attempts):
        if len(population) == size:
            return population
        strategy = strategies[attempt % len(strategies)]
        seed = rng.randrange(2**31)
        try:
            candidate = sample(strategy, seed, base, settings)
        except (ValueError, RuntimeError):
            continue  # A sampler can fail to produce a feasible scenario.
        consider(candidate)

    if len(population) < size:
        raise RuntimeError(
            f"Only {len(population)}/{size} distinct valid individuals after "
            f"{attempts} attempts; try a larger budget or more attempts"
        )
    return population

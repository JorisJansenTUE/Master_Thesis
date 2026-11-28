"""Spatially aware genetic operators for MATSim road-space interventions.

Individuals retain the established list[Intervention] representation. All
eligibility, width, compatibility and directed-connectivity rules continue to
come from network_sampling; this module never implements a second sampler.
"""
from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from dataclasses import dataclass

from network_sampling.interventions.compatibility import interventions_compatible
from network_sampling.interventions.intervention_types import Intervention, RoadSpaceVariant
from network_sampling.interventions.matsim_app.apply_network import apply_interventions
from network_sampling.sampling.random import intervention_length
from network_sampling.validation.connectivity import check_connectivity


# Target budgets for immigrants: fractions of the GA's MAXIMUM budget.
# The independent samplers remain unchanged; they receive a smaller target.
IMMIGRANT_BUDGET_FRACTIONS = ((0.50, 0.67), (0.67, 0.83), (0.83, 1.0))


def configuration_hash(individual: list[Intervention]) -> str:
    """Order-independent configuration identifier, compatible with prior GA."""
    records = [
        (str(iv.segment_id), iv.intervention_type.value,
         iv.road_space_variant.value if iv.road_space_variant is not None else None,
         tuple(sorted(str(link) for link in iv.target_link_ids)))
        for iv in individual
    ]
    data = json.dumps(sorted(records, key=lambda row: json.dumps(row)), separators=(",", ":"))
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def _basic_check(individual, *, eligible, network, index, budget_m) -> str | None:
    if budget_m < 0:
        raise ValueError("budget_m must be non-negative")
    if len(individual) != len(set(individual)):
        return "duplicate intervention"
    allowed = set(eligible)
    if any(iv not in allowed for iv in individual):
        return "intervention not in baseline eligibility list"
    if intervention_length(individual, index) > budget_m + 1e-6:
        return "road-space budget exceeded"
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
    """Fast eligibility and directed-modal check. Java checking happens on XML."""
    problem = _basic_check(individual, eligible=eligible, network=network,
                           index=index, budget_m=budget_m)
    if problem is not None:
        return False, problem
    try:
        modified, _ = apply_interventions(network, index, individual)
        connected = check_connectivity(network, modified)
    except ValueError as exc:
        if str(exc).startswith("Baseline "):
            raise
        return False, str(exc)
    failed = [mode for mode, ok in connected.items() if not ok]
    if failed:
        return False, "disconnected modes: " + ", ".join(failed)
    return True, "ok"


@dataclass
class SpatialContext:
    """Physical street adjacency and eligibility, built once per GA generation."""
    neighbors: dict[str, tuple[str, ...]]
    by_segment: dict[str, tuple[Intervention, ...]]


def build_spatial_context(eligible: list[Intervention], index) -> SpatialContext:
    by_segment: dict[str, list[Intervention]] = defaultdict(list)
    for iv in eligible:
        by_segment[str(iv.segment_id)].append(iv)
    by_node: dict[str, set[str]] = defaultdict(set)
    for segment in index.all():
        by_node[str(segment.node_a)].add(str(segment.id))
        by_node[str(segment.node_b)].add(str(segment.id))
    neighbors = {}
    for segment in index.all():
        sid = str(segment.id)
        neighbors[sid] = tuple(sorted((by_node[str(segment.node_a)] |
                                       by_node[str(segment.node_b)]) - {sid}))
    return SpatialContext(neighbors=neighbors,
                          by_segment={k: tuple(v) for k, v in by_segment.items()})


def _segment_ids(individual):
    return {str(iv.segment_id) for iv in individual}


def _connected_patch(individual, ctx: SpatialContext, index, rng: random.Random,
                     target_m: float) -> set[str]:
    """Select one contiguous patch of the parent's OWN physical segments."""
    present = _segment_ids(individual)
    if not present:
        return set()
    seed = rng.choice(sorted(present))
    patch, frontier = set(), [seed]
    used = 0.0
    while frontier:
        at = rng.randrange(len(frontier))
        sid = frontier.pop(at)
        if sid in patch or sid not in present:
            continue
        patch.add(sid)
        used += float(index.get(sid).length_m)
        if used >= target_m:
            break
        nxt = [v for v in ctx.neighbors.get(sid, ())
               if v in present and v not in patch and v not in frontier]
        rng.shuffle(nxt)
        frontier.extend(nxt)
    return patch


def _segment_cost(individual, index) -> float:
    return sum(float(index.get(iv.segment_id).length_m) for iv in individual)


def repair_individual(
    proposed: list[Intervention], *, eligible: list[Intervention],
    network, index, budget_m: float,
    protected: list[Intervention] | None = None,
) -> list[Intervention]:
    """Preserve the primary parent's unchanged genes wherever possible.

    Newly introduced genes are attempted after protected genes. If connectivity
    fails, unprotected additions are removed first; only then is a disruptive
    inherited REMOVE_CAR gene considered. Never return an invalid individual.
    """
    allowed = set(eligible)
    protected = list(protected or [])
    priority = protected + [iv for iv in proposed if iv not in protected]
    chosen: list[Intervention] = []
    groups: dict[str, list[Intervention]] = defaultdict(list)
    used = 0.0
    for iv in priority:
        if iv not in allowed or iv in chosen:
            continue
        cost = float(index.get(iv.segment_id).length_m)
        if used + cost > budget_m + 1e-6:
            continue
        if not all(interventions_compatible(iv, old, network)
                   for old in groups[str(iv.segment_id)]):
            continue
        chosen.append(iv)
        groups[str(iv.segment_id)].append(iv)
        used += cost

    protected_set = set(protected)
    while True:
        ok, _ = validate_individual(chosen, eligible=eligible, network=network,
                                    index=index, budget_m=budget_m)
        if ok:
            return chosen
        # Remove newly inserted invasive genes first, preserving an inherited
        # corridor even if the new variant is invalid or connectivity-breaking.
        unprotected = [i for i, iv in enumerate(chosen) if iv not in protected_set]
        invasive = [i for i, iv in enumerate(chosen)
                    if iv.road_space_variant == RoadSpaceVariant.REMOVE_CAR]
        removable = ([i for i in unprotected if i in invasive]
                     or unprotected or invasive)
        if not removable:
            # A genuinely invalid protected set should never be silently accepted.
            raise ValueError("Cannot repair offspring with current operators")
        chosen.pop(removable[-1])


def _eligible_neighbor(ctx, present, rng):
    frontier = set()
    for sid in present:
        frontier.update(n for n in ctx.neighbors.get(sid, ())
                        if n not in present and n in ctx.by_segment)
    return rng.choice(sorted(frontier)) if frontier else None


def _variants_on_segment(ctx, sid, old=None):
    return [iv for iv in ctx.by_segment.get(sid, ()) if iv != old]


def _grow_addition_patch(parent, ctx: SpatialContext, index, budget_m: float,
                         rng: random.Random) -> list[Intervention]:
    """Propose one connected patch of new physical streets within spare budget.

    Usually grow next to an existing intervention; sometimes seed a new area.
    Eligibility is inherited from ctx. The caller performs the usual repair
    and verifies the surviving patch before accepting it.
    """
    remaining = budget_m - _segment_cost(parent, index)
    target = min(remaining, rng.uniform(0.03, 0.12) * budget_m)
    if target <= 0:
        return []
    present = _segment_ids(parent)
    available = {sid for sid in ctx.by_segment if sid not in present
                 and 0 < float(index.get(sid).length_m) <= target + 1e-6}
    if not available:
        return []
    adjacent = {sid for old in present for sid in ctx.neighbors.get(old, ())
                if sid in available}
    seed_pool = adjacent if adjacent and rng.random() < 0.70 else available
    frontier = {rng.choice(sorted(seed_pool))}
    patch = []
    used = 0.0
    while frontier:
        sid = rng.choice(sorted(frontier))
        frontier.remove(sid)
        available.discard(sid)
        cost = float(index.get(sid).length_m)
        if used + cost > target + 1e-6:
            continue
        patch.append(rng.choice(ctx.by_segment[sid]))
        used += cost
        frontier.update(n for n in ctx.neighbors.get(sid, ()) if n in available)
        if used >= target - 1e-6:
            break
    return patch


def _one_connected_patch(segments: set[str], ctx: SpatialContext) -> bool:
    if not segments:
        return False
    visited = set()
    frontier = [min(segments)]
    while frontier:
        sid = frontier.pop()
        if sid in visited:
            continue
        visited.add(sid)
        frontier.extend(n for n in ctx.neighbors.get(sid, ())
                        if n in segments and n not in visited)
    return visited == segments


def mutate(
    parent: list[Intervention], *, eligible: list[Intervention], network,
    index, budget_m: float, rng: random.Random, max_attempts: int = 18,
    context: SpatialContext | None = None, diagnostics: dict | None = None,
) -> list[Intervention]:
    """Local changes and connected patch addition/removal."""
    if max_attempts < 1:
        raise ValueError("max_attempts must be positive")
    ctx = context or build_spatial_context(eligible, index)
    parent_key = configuration_hash(parent)
    by_present = _segment_ids(parent)
    ops = ["extend", "extend", "trim", "trim", "shift", "shift",
           "variant", "variant", "new_component", "remove_patch",
           "add_patch", "add_patch"]
    for _ in range(max_attempts):
        op = rng.choice(ops)
        proposed = list(parent)
        protected: list[Intervention] = list(parent)

        if op == "extend":
            sid = _eligible_neighbor(ctx, by_present, rng)
            if sid is None:
                continue
            proposed.append(rng.choice(_variants_on_segment(ctx, sid)))
        elif op == "add_patch":
            additions = _grow_addition_patch(parent, ctx, index, budget_m, rng)
            if not additions:
                continue
            proposed.extend(additions)
        elif op == "new_component":
            if not eligible:
                continue
            choices = [iv for iv in eligible if str(iv.segment_id) not in by_present]
            if not choices:
                continue
            proposed.append(rng.choice(choices))
        elif op == "variant":
            if not proposed:
                continue
            old = rng.choice(proposed)
            variants = _variants_on_segment(ctx, str(old.segment_id), old)
            if not variants:
                continue
            proposed.remove(old)
            proposed.append(rng.choice(variants))
            protected.remove(old)
        elif op in ("trim", "shift", "remove_patch"):
            if not by_present:
                continue
            if op == "remove_patch":
                patch = _connected_patch(parent, ctx, index, rng,
                                         max(1.0, rng.uniform(0.03, 0.12) * budget_m))
            else:
                # Prefer an endpoint in a corridor to preserve its middle.
                ends = [sid for sid in by_present
                        if sum(n in by_present for n in ctx.neighbors.get(sid, ())) <= 1]
                patch = {rng.choice(sorted(ends or by_present))}
            proposed = [iv for iv in proposed if str(iv.segment_id) not in patch]
            protected = list(proposed)
            if op == "shift":
                sid = _eligible_neighbor(ctx, _segment_ids(proposed), rng)
                if sid is None:
                    # Relocation can also shift the whole 1-segment component.
                    possible = set()
                    for removed_sid in patch:
                        possible.update(ctx.neighbors.get(removed_sid, ()))
                    options = sorted(s for s in possible if s in ctx.by_segment and
                                     s not in _segment_ids(proposed))
                    sid = rng.choice(options) if options else None
                if sid is not None:
                    proposed.append(rng.choice(_variants_on_segment(ctx, sid)))
        else:
            raise AssertionError(op)

        try:
            child = repair_individual(proposed, eligible=eligible, network=network,
                                      index=index, budget_m=budget_m, protected=protected)
        except ValueError as exc:
            if str(exc) != "Cannot repair offspring with current operators":
                raise
            continue
        if op == "add_patch":
            # Repair may remove additions (including an interior street). Do
            # not accept a disconnected patch or loss of inherited genes.
            if not set(parent).issubset(child):
                continue
            added_segments = _segment_ids(child) - by_present
            if not _one_connected_patch(added_segments, ctx):
                continue
            if diagnostics is not None:
                added_m = sum(float(index.get(sid).length_m) for sid in added_segments)
                diagnostics["mutation_add_patch_total_m"] = (
                    diagnostics.get("mutation_add_patch_total_m", 0.0) + added_m)
        if configuration_hash(child) != parent_key:
            if diagnostics is not None:
                label = "mutation_" + op
                diagnostics[label] = diagnostics.get(label, 0) + 1
            return child
    if diagnostics is not None:
        diagnostics["mutation_no_change"] = diagnostics.get("mutation_no_change", 0) + 1
    return list(parent)


def crossover(
    a: list[Intervention], b: list[Intervention], *,
    eligible: list[Intervention], network, index, budget_m: float,
    rng: random.Random, context: SpatialContext | None = None,
    diagnostics: dict | None = None,
) -> list[Intervention]:
    """Replace a compact connected patch of parent A with a patch from B.

    Unlike uniform crossover, existing corridors outside the replaced physical
    street patch are retained intact. Both patches may be in different areas:
    this also enables trading space between disconnected intervention clusters.
    """
    ctx = context or build_spatial_context(eligible, index)
    if not b:
        return list(a)
    for _ in range(10):
        fraction = rng.uniform(0.04, 0.25)
        patch_b = _connected_patch(b, ctx, index, rng,
                                   max(1.0, fraction * budget_m))
        if not patch_b:
            continue
        patch_a = _connected_patch(a, ctx, index, rng,
                                   max(1.0, fraction * budget_m))
        retained_a = [iv for iv in a if str(iv.segment_id) not in patch_a]
        block_b = [iv for iv in b if str(iv.segment_id) in patch_b]
        # Try newly transferred genes before discarding any retained A genes.
        candidate = retained_a + block_b
        try:
            child = repair_individual(candidate, eligible=eligible, network=network,
                                      index=index, budget_m=budget_m,
                                      protected=retained_a)
        except ValueError as exc:
            if str(exc) != "Cannot repair offspring with current operators":
                raise
            continue
        # A crossover must transfer at least one gene from B, not merely
        # shorten A after exceeding the budget.
        if (configuration_hash(child) != configuration_hash(a)
                and any(iv in child for iv in block_b if iv not in retained_a)):
            if diagnostics is not None:
                diagnostics["crossover_changed"] = diagnostics.get("crossover_changed", 0) + 1
            return child
    if diagnostics is not None:
        diagnostics["crossover_no_change"] = diagnostics.get("crossover_no_change", 0) + 1
    return list(a)


def _sample_target_budget(budget_m: float, attempt: int, rng: random.Random) -> float:
    if budget_m <= 0:
        return 0.0
    lo, hi = IMMIGRANT_BUDGET_FRACTIONS[attempt % len(IMMIGRANT_BUDGET_FRACTIONS)]
    return max(min(250.0, budget_m), rng.uniform(lo, hi) * budget_m)


def initial_population(
    base, *, size: int, budget_m: float, rng: random.Random,
    settings: dict, initial: list[list[Intervention]] | None = None,
    max_attempts: int | None = None,
) -> list[list[Intervention]]:
    """Distinct validated population, including mixed-budget immigrants.

    Existing samplers receive a sampled TARGET budget <= the GA's limit;
    final validity always uses the original GA maximum budget.
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
    attempts = size * 50 if max_attempts is None else max_attempts
    offset = rng.randrange(len(IMMIGRANT_BUDGET_FRACTIONS))
    for attempt in range(attempts):
        if len(population) == size:
            return population
        strategy = strategies[attempt % len(strategies)]
        seed = rng.randrange(2**31)
        local_settings = {**settings,
                          "budget_m": _sample_target_budget(budget_m, attempt + offset, rng)}
        try:
            candidate = sample(strategy, seed, base, local_settings)
        except (ValueError, RuntimeError):
            continue
        consider(candidate)

    if len(population) < size:
        raise RuntimeError(
            f"Only {len(population)}/{size} distinct valid individuals after "
            f"{attempts} attempts; try more attempts or inspect sampler constraints"
        )
    return population

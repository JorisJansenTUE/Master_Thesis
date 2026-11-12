"""Connectivity-aware cycling/bus corridors with random destination roads.

Budgets count each intervention separately: opposite-direction dedicated bus
lanes on one physical StreetSegment therefore cost twice its length.
Neither the baseline MATSim network nor the physical graph is modified.
"""
from __future__ import annotations

import random
from collections import Counter, defaultdict
from math import isfinite, sqrt

import networkx as nx

from network_sampling.interventions.compatibility import interventions_compatible
from network_sampling.interventions.intervention_types import (
    Intervention, InterventionType, RoadSpaceVariant,
)
from network_sampling.network.matsim_network import MatsimNetwork
from network_sampling.network.street_segments import StreetSegmentIndex
from network_sampling.network.transit_usage import TransitUsageIndex
from network_sampling.sampling.random import WEIGHT_KEEP_CAR, intervention_weight
from network_sampling.validation.connectivity import build_mode_graph


BIKE_TYPES = {InterventionType.PROTECTED_BIKE_LANE, InterventionType.BICYCLE_STREET}


def physical_length(interventions: list[Intervention], index: StreetSegmentIndex) -> float:
    """Sum intervention lengths; opposite-direction bus lanes count twice."""
    return sum(index.get(iv.segment_id).length_m for iv in interventions)


def _is_protected_bike_link(link) -> bool:
    return (
        set(link.allowed_modes) == {"bike"}
        or str(link.attributes.get("cyclestreet", "")).strip().lower() == "yes"
    )


class _CarState:
    """Incremental directed car connectivity; baseline MATSim is read only.

    ``support[(u, v)]`` counts car-permitted MATSim links with these endpoints,
    so a parallel directed graph edge is removed only after its last link.
    """

    def __init__(self, network: MatsimNetwork):
        self.graph = build_mode_graph(network, "car")
        if not nx.is_strongly_connected(self.graph):
            raise ValueError("Baseline car network is not strongly connected")
        self.links = {
            lid: (link.from_node, link.to_node)
            for lid, link in network.links.items()
            if "car" in link.allowed_modes
        }
        self.support = Counter(self.links.values())
        self.removed: set[str] = set()

    def copy(self) -> "_CarState":
        state = object.__new__(_CarState)
        state.graph = self.graph.copy()
        state.links = self.links  # Immutable baseline mapping, shared by copies.
        state.support = self.support.copy()
        state.removed = self.removed.copy()
        return state

    def try_intervention(self, iv: Intervention) -> bool:
        """Commit REMOVE_CAR when strongly connected; otherwise undo it."""
        if iv.road_space_variant != RoadSpaceVariant.REMOVE_CAR:
            return True

        targets = set(iv.target_link_ids)
        if any(lid in self.removed for lid in targets):
            return False
        affected = Counter(self.links[lid] for lid in targets if lid in self.links)
        if any(self.support[edge] < count for edge, count in affected.items()):
            return False

        removed_edges = []
        for edge, count in affected.items():
            self.support[edge] -= count
            if self.support[edge] == 0:
                self.graph.remove_edge(*edge)
                removed_edges.append(edge)

        if nx.is_strongly_connected(self.graph):
            self.removed.update(targets)
            return True

        for edge, count in affected.items():
            self.support[edge] += count
        self.graph.add_edges_from(removed_edges)
        return False


def sample_corridors(
    graph: nx.MultiGraph,
    index: StreetSegmentIndex,
    network: MatsimNetwork,
    candidates: list[Intervention],
    budget_m: float,
    *,
    max_corridor_length_m: float | None = 2000.0,
    intrusion_strength: float = 1.0,
    pbl_weight_multiplier: float = 1.0,
    rng: random.Random | None = None,
    preselected: list[Intervention] | None = None,
) -> list[Intervention]:
    """Sample new cycling corridors, considering previously selected bus removals.

    Returns cycling interventions only; preselected interventions are not repeated.
    """
    if not isfinite(budget_m) or budget_m < 0:
        raise ValueError("budget_m must be finite and nonnegative")
    if max_corridor_length_m is not None and (
        not isfinite(max_corridor_length_m) or max_corridor_length_m <= 0
    ):
        raise ValueError("max_corridor_length_m must be finite and positive")
    if not isfinite(intrusion_strength) or intrusion_strength < 0:
        raise ValueError("intrusion_strength must be finite and nonnegative")
    if not isfinite(pbl_weight_multiplier) or pbl_weight_multiplier <= 0:
        raise ValueError("pbl_weight_multiplier must be finite and positive")
    if not isinstance(graph, nx.MultiGraph) or graph.is_directed():
        raise TypeError("Expected an undirected nx.MultiGraph")

    rng = rng if rng is not None else random.Random()
    preselected = list(preselected or [])
    occupied = {iv.segment_id for iv in preselected}
    car = _CarState(network)
    for iv in preselected:
        if not car.try_intervention(iv):
            raise ValueError("Preselected interventions disconnect the car graph")

    by_segment = defaultdict(list)
    for iv in candidates:
        if iv.intervention_type not in BIKE_TYPES or iv.segment_id in occupied:
            continue
        weight = intervention_weight(iv, index, network)
        if weight <= 0:
            raise ValueError(f"Nonpositive intervention weight: {iv}")
        severity = sqrt(WEIGHT_KEEP_CAR / weight) - 1.0
        by_segment[iv.segment_id].append((iv, weight, severity))

    available = nx.MultiGraph()
    for u, v, key, data in graph.edges(keys=True, data=True):
        sid = data.get("segment_id")
        if sid is None:
            sid = getattr(data.get("segment"), "id", key)
        if sid not in by_segment or u == v:
            continue
        segment = index.get(sid)
        if not isfinite(segment.length_m) or segment.length_m <= 0:
            raise ValueError(f"Invalid physical length: {sid}")
        if {u, v} != {segment.node_a, segment.node_b}:
            raise ValueError(f"Physical graph endpoints differ for {sid}")
        opts = by_segment[sid]
        total = sum(w for _, w, _ in opts)
        expected_intrusion = sum(w * penalty for _, w, penalty in opts) / total
        factor = 1.0 + intrusion_strength * expected_intrusion
        available.add_edge(u, v, key=sid, length_m=segment.length_m,
                           routing_cost=segment.length_m * factor, routing_factor=factor)

    if by_segment and available.number_of_edges() == 0:
        raise ValueError("No eligible cycling candidates match physical graph edges")

    anchors = {
        node for link in network.links.values()
        if _is_protected_bike_link(link)
        for node in (link.from_node, link.to_node)
    } & set(available.nodes)

    selected = []
    spent = 0.0

    while spent < budget_m - 1e-9:
        anchors &= set(available.nodes)
        if not anchors or available.number_of_edges() == 0:
            break
        remaining = budget_m - spent
        target_m = min(remaining, max_corridor_length_m or remaining)

        distance, paths = nx.multi_source_dijkstra(
            available, sorted(anchors), weight="routing_cost"
        )
        physical_distance = {}
        parent_edges = {}
        for node in sorted(distance, key=distance.__getitem__):
            path = paths[node]
            if len(path) == 1:
                physical_distance[node] = 0.0
                continue
            parent = path[-2]
            edge_id = min(available[parent][node], key=lambda k: (
                available[parent][node][k]["routing_cost"], str(k)))
            parent_edges[node] = edge_id
            physical_distance[node] = (
                physical_distance[parent] + available[parent][node][edge_id]["length_m"]
            )

        seeds = []
        for u, v, sid, data in available.edges(keys=True, data=True):
            if u not in distance or v not in distance:
                continue
            near, far = (u, v) if distance[u] <= distance[v] else (v, u)
            length = physical_distance[near] + data["length_m"]
            if length <= target_m + 1e-9 and far not in paths[near]:
                seeds.append((sid, near, far, far not in anchors))
        if not seeds:
            break

        remote_seeds = [seed for seed in seeds if seed[3]]
        seed_sid, near, far, _ = rng.choice(sorted(remote_seeds or seeds))
        node_path = list(paths[near]) + [far]
        segment_path = [parent_edges[node] for node in paths[near][1:]] + [seed_sid]
        length_m = sum(index.get(sid).length_m for sid in segment_path)
        visited = set(node_path)

        while True:
            end = node_path[-1]
            last_way = index.get(segment_path[-1]).osm_way_id
            steps, weights = [], []
            for next_node, edges in available[end].items():
                if next_node in visited:
                    continue
                for sid, data in edges.items():
                    segment = index.get(sid)
                    if length_m + segment.length_m > target_m + 1e-9:
                        continue
                    same_way = (3.0 if last_way and segment.osm_way_id == last_way else 1.0)
                    steps.append((sid, next_node))
                    weights.append(same_way / data["routing_factor"])
            if not steps:
                break
            sid, next_node = rng.choices(steps, weights=weights, k=1)[0]
            segment_path.append(sid)
            node_path.append(next_node)
            visited.add(next_node)
            length_m += index.get(sid).length_m

        pending = []
        trial_car = car.copy()
        previous_type = None
        blocked_sid = None

        for sid in segment_path:
            options = by_segment[sid]
            weights = [
                w * (pbl_weight_multiplier if iv.intervention_type == InterventionType.PROTECTED_BIKE_LANE else 1.0)
                * (3.0 if iv.intervention_type == previous_type else 1.0)
                for iv, w, _ in options
            ]
            order = sorted(range(len(options)), key=lambda i: rng.expovariate(weights[i]))
            chosen = None
            for i in order:
                iv = options[i][0]
                if any(
                    not interventions_compatible(iv, other, network)
                    for other in preselected + selected + pending
                    if other.segment_id == sid
                ):
                    continue
                if not trial_car.try_intervention(iv):
                    continue
                chosen = iv
                break
            if chosen is None:
                blocked_sid = sid
                break
            pending.append(chosen)
            previous_type = chosen.intervention_type

        if blocked_sid is not None:
            seg = index.get(blocked_sid)
            available.remove_edge(seg.node_a, seg.node_b, key=blocked_sid)
            continue

        selected.extend(pending)
        car = trial_car
        spent += length_m
        for iv in pending:
            seg = index.get(iv.segment_id)
            anchors.update((seg.node_a, seg.node_b))
        for sid in segment_path:
            seg = index.get(sid)
            available.remove_edge(seg.node_a, seg.node_b, key=sid)

    return selected


def sample_bus_corridors(
    transit_usage: TransitUsageIndex,
    index: StreetSegmentIndex,
    network: MatsimNetwork,
    candidates: list[Intervention],
    selected: list[Intervention],
    budget_m: float,
    *,
    rng: random.Random,
    max_corridor_length_m: float | None = 1500.0,
    min_links: int = 3,
) -> list[Intervention]:
    """Grow directional bus-lane chains along scheduled route sequences.

    Each bus-lane intervention consumes its segment's full length, including
    opposite directions on the same street. A connectivity failure ends the
    current chain only when no compatible, connectivity-safe variant remains.
    Retain its feasible prefix if len(prefix) >= min_links, then start another.
    """
    if not isfinite(budget_m) or budget_m < 0:
        raise ValueError("Bus budget must be finite and nonnegative")
    if max_corridor_length_m is not None and (
        not isfinite(max_corridor_length_m) or max_corridor_length_m <= 0
    ):
        raise ValueError("Invalid max_corridor_length_m")
    if min_links < 1:
        raise ValueError("min_links must be positive")

    occupied = {
        iv.segment_id for iv in selected
        if iv.intervention_type != InterventionType.DEDICATED_BUS_LANE
    }
    taken = {
        iv.target_link_ids[0] for iv in selected
        if iv.intervention_type == InterventionType.DEDICATED_BUS_LANE
    }
    by_link = defaultdict(list)
    for iv in candidates:
        if iv.intervention_type != InterventionType.DEDICATED_BUS_LANE:
            continue
        if len(iv.target_link_ids) != 1:
            raise ValueError(f"Bus lane must target one directional link: {iv}")
        lid = iv.target_link_ids[0]
        if iv.segment_id not in occupied and lid not in taken and lid in network.links:
            by_link[lid].append(iv)

    if not by_link or budget_m == 0:
        return []

    # Split scheduled routes at ineligible, repeated or disconnected links.
    # Unlike graph-based growth, these chains cannot jump between bus services.
    runs = []
    seen_runs = set()

    def add_run(run):
        key = tuple(run)
        if len(key) >= min_links and key not in seen_runs:
            seen_runs.add(key)
            runs.append(key)

    for _, route in sorted(transit_usage.bus_routes.items()):
        run = []
        for lid in route:
            if lid not in by_link:
                add_run(run)
                run = []
                continue
            if run and (
                lid in run
                or network.links[run[-1]].to_node != network.links[lid].from_node
            ):
                add_run(run)
                run = []
            run.append(lid)
        add_run(run)

    if not runs:
        return []

    lengths = {
        lid: index.get(options[0].segment_id).length_m
        for lid, options in by_link.items()
    }
    active = set(by_link)  # Directional links not yet selected or blocked.
    attempted = set()      # (route index, position, direction) to avoid retries.
    car = _CarState(network)
    for iv in selected:
        if not car.try_intervention(iv):
            raise ValueError("Preselected interventions disconnect the car graph")
    result = []
    remaining = budget_m

    while remaining > 1e-9:
        limit = min(remaining, max_corridor_length_m or remaining)

        # Only start where at least min_links consecutive eligible directional
        # links could fit. This prevents useless very short chain attempts.
        starts = []
        for route_idx, run in enumerate(runs):
            for direction in (1, -1):
                for pos in range(len(run)):
                    start_key = (route_idx, pos, direction)
                    if start_key in attempted:
                        continue
                    positions = range(pos, pos + direction * min_links, direction)
                    if not all(0 <= i < len(run) and run[i] in active for i in positions):
                        continue
                    if sum(lengths[run[i]] for i in positions) <= limit + 1e-9:
                        starts.append(start_key)

        if not starts:
            break

        route_idx, pos, direction = rng.choice(starts)
        attempted.add((route_idx, pos, direction))
        run = runs[route_idx]
        pending = []
        trial_car = car.copy()
        spent = 0.0
        failed_link = None

        # Extend along the chosen route. Try safer variants before ending
        # the chain at a link that cannot preserve car connectivity.
        while 0 <= pos < len(run):
            lid = run[pos]
            cost = lengths[lid]
            if lid not in active or spent + cost > limit + 1e-9:
                break

            options = by_link[lid]
            order = sorted(
                options,
                key=lambda iv: rng.expovariate(intervention_weight(iv, index, network)),
            )
            chosen = None
            previous = selected + result + pending

            for iv in order:
                # Never select the same intervention twice, even when
                # multiple scheduled bus routes share a directional link.
                if iv in previous:
                    continue
                if not all(
                    interventions_compatible(iv, other, network)
                    for other in previous
                ):
                    continue

                if not trial_car.try_intervention(iv):
                    # Try another eligible variant before ending the chain.
                    continue

                chosen = iv
                break

            if chosen is None:
                # No compatible, connectivity-safe variant remains.
                # Keep a valid prefix only if it reaches min_links.
                failed_link = lid
                break

            pending.append(chosen)
            spent += cost
            pos += direction

        if failed_link is not None:
            # A failed link cannot extend another chain in this scenario.
            # Other parts of this same route can still form new chains.
            active.discard(failed_link)

        if len(pending) < min_links:
            # Discard short prefixes without committing their car removals.
            continue

        result.extend(pending)
        car = trial_car
        remaining -= spent
        active.difference_update(iv.target_link_ids[0] for iv in pending)

    return result


def sample_destination_roads(
    index: StreetSegmentIndex,
    candidates: list[Intervention],
    selected: list[Intervention],
    budget_m: float,
    *,
    rng: random.Random,
) -> list[Intervention]:
    """Random destination-only streets, unrelated to corridor locations."""
    used = {iv.segment_id for iv in selected}
    options = [iv for iv in candidates
               if iv.intervention_type == InterventionType.DESTINATION_ONLY
               and iv.segment_id not in used]
    options.sort(key=lambda iv: iv.segment_id)
    rng.shuffle(options)
    result = []
    for iv in options:
        sid = iv.segment_id
        if sid in used:
            continue
        cost = index.get(sid).length_m
        if cost <= budget_m + 1e-9:
            result.append(iv)
            budget_m -= cost
            used.add(sid)
    return result


def sample_corridor_package(
    graph: nx.MultiGraph,
    index: StreetSegmentIndex,
    network: MatsimNetwork,
    candidates: list[Intervention],
    budget_m: float,
    *,
    transit_usage: TransitUsageIndex,
    seed: int | None = None,
    destination_share: float | None = None,
    bus_share: float | None = None,
    max_destination_share: float = 0.20,
    max_bus_share: float = 0.20,
    max_corridor_length_m: float | None = None,
    max_bus_corridor_length_m: float | None = 1500.0,
    intrusion_strength: float = 0.5,
    pbl_weight_multiplier: float = 3.0,
) -> list[Intervention]:
    """Build a mixed package with a shared per-intervention length budget.

    Scheduled bus-route corridors are sampled first; cycling corridors account
    for their car removals, then random destination roads are added.
    """
    if not isfinite(budget_m) or budget_m < 0:
        raise ValueError("Budget must be finite and nonnegative")
    if not 0 <= max_destination_share <= 1 or not 0 <= max_bus_share <= 1:
        raise ValueError("Maximum shares must be in [0, 1]")
    if max_destination_share + max_bus_share > 1:
        raise ValueError("Maximum bus and destination shares must sum to at most 1")

    rng = random.Random(seed)
    dest = rng.uniform(0, max_destination_share) if destination_share is None else destination_share
    bus = rng.uniform(0, max_bus_share) if bus_share is None else bus_share
    if not (isfinite(dest) and isfinite(bus) and dest >= 0 and bus >= 0 and dest + bus <= 1):
        raise ValueError("Invalid bus/destination shares")

    selected = sample_bus_corridors(
        transit_usage, index, network, candidates, [], budget_m * bus,
        rng=rng, max_corridor_length_m=max_bus_corridor_length_m,
    )

    # Unspent bus allocation is available to cycling corridors.
    cycling_budget = max(0.0, budget_m - physical_length(selected, index) - budget_m * dest)
    selected.extend(sample_corridors(
        graph, index, network, candidates, cycling_budget, rng=rng,
        max_corridor_length_m=max_corridor_length_m,
        intrusion_strength=intrusion_strength,
        pbl_weight_multiplier=pbl_weight_multiplier, preselected=selected,
    ))

    # Destination roads use their reserved share, independently of location.
    remaining = budget_m - physical_length(selected, index)
    destination_budget = min(budget_m * dest, remaining)
    selected.extend(sample_destination_roads(
        index, candidates, selected, destination_budget, rng=rng,
    ))
    if physical_length(selected, index) > budget_m + 1e-6:
        raise AssertionError("Intervention package exceeds intervention length budget")
    return selected

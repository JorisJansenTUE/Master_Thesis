"""Spatially clustered, multimodal intervention sampling on abstract graphs.

MATSim is read only to construct baseline graphs and obtain intervention metadata.
No modified MATSim network is built during sampling. The final intervention package
is still validated by the application/inspection layer.
"""
from __future__ import annotations

import random
from collections import Counter, defaultdict
from math import isfinite

import networkx as nx

from network_sampling.interventions.intervention_types import (
    Intervention, InterventionType, RoadSpaceVariant,
)
from network_sampling.network.matsim_network import MatsimNetwork
from network_sampling.network.street_segments import StreetSegmentIndex
from network_sampling.sampling.random import intervention_weight
from network_sampling.validation.connectivity import build_mode_graph


BIKE_TYPES = {InterventionType.BICYCLE_STREET, InterventionType.PROTECTED_BIKE_LANE}


def _length(interventions: list[Intervention], index: StreetSegmentIndex) -> float:
    """Per-intervention length: bus lanes in opposite directions count twice."""
    return sum(index.get(iv.segment_id).length_m for iv in interventions)


def _try_car_removal(
    intervention: Intervention,
    car: nx.DiGraph,
    support: Counter,
    car_links: dict[str, tuple[str, str]],
) -> bool:
    """Test and, if safe, commit one REMOVE_CAR on a working car graph.

    ``support[(u, v)]`` counts remaining car-permitted MATSim links between
    the two nodes. The DiGraph edge disappears only once the last is removed.
    A rejected intervention rolls back both the graph and the counts.
    """
    if intervention.road_space_variant != RoadSpaceVariant.REMOVE_CAR:
        return True

    affected = Counter()
    for link_id in set(intervention.target_link_ids):
        if edge := car_links.get(link_id):
            affected[edge] += 1

    if not affected:
        return True
    if any(support[edge] < count for edge, count in affected.items()):
        return False  # A targeted car link is already removed or inconsistent.

    removed_edges = []
    for edge, count in affected.items():
        support[edge] -= count
        if support[edge] == 0:
            car.remove_edge(*edge)
            removed_edges.append(edge)

    if nx.is_strongly_connected(car):
        return True

    for edge, count in affected.items():
        support[edge] += count
    car.add_edges_from(removed_edges)
    return False


def _choose(
    options: list[tuple[Intervention, float]],
    previous: list[Intervention],
    car: nx.DiGraph,
    support: Counter,
    car_links: dict[str, tuple[str, str]],
    rng: random.Random,
    *,
    safest: bool = False,
) -> Intervention | None:
    """Choose an eligible intervention, testing safer variants on failure.

    The working car graph is updated only when a choice is accepted.
    """
    if safest:
        priority = {
            None: 0,
            RoadSpaceVariant.KEEP_CAR: 0,
            RoadSpaceVariant.REMOVE_ONE_LANE: 1,
            RoadSpaceVariant.REMOVE_CAR: 2,
        }
        ordered = sorted(options, key=lambda pair: (
            priority.get(pair[0].road_space_variant, 3),
            pair[0].intervention_type != InterventionType.BICYCLE_STREET,
            pair[0].segment_id,
        ))
    else:
        ordered = sorted(options, key=lambda pair: rng.expovariate(pair[1]))

    for iv, _ in ordered:
        # Cluster sampling allows only one intervention per physical segment.
        # Thus no opposite-direction bus-lane pair is selected within a cluster.
        if any(iv.segment_id == old.segment_id for old in previous):
            continue
        if _try_car_removal(iv, car, support, car_links):
            return iv
    return None


def _add_bicycle(
    bike: nx.DiGraph,
    intervention: Intervention,
    index: StreetSegmentIndex,
) -> None:
    """Mirror the bidirectional bike access created by the application layer."""
    if intervention.intervention_type in BIKE_TYPES:
        segment = index.get(intervention.segment_id)
        bike.add_edge(segment.node_a, segment.node_b)
        bike.add_edge(segment.node_b, segment.node_a)


def _grow_footprint(seed, available, graph, index, target_m, rng):
    """Build a physically connected, branching cluster of StreetSegments."""
    footprint = {seed}
    seg = index.get(seed)
    length = seg.length_m
    depth = {seg.node_a: 0, seg.node_b: 0}

    while length < target_m - 1e-9:
        choices = {}
        for node, distance in list(depth.items()):
            if node not in graph:
                continue
            for neighbor, edges in graph[node].items():
                for key, attrs in edges.items():
                    sid = attrs.get("segment_id", key)
                    if sid not in available or sid in footprint:
                        continue
                    candidate = index.get(sid)
                    if length + candidate.length_m > target_m + 1e-9:
                        continue
                    touches = int(candidate.node_a in depth) + int(candidate.node_b in depth)
                    weight = (2.0 if touches == 2 else 1.0) / (1.0 + distance) ** 2
                    choices[sid] = max(choices.get(sid, 0.0), weight)

        if not choices:
            break

        sids = sorted(choices)
        sid = rng.choices(sids, weights=[choices[s] for s in sids], k=1)[0]
        candidate = index.get(sid)
        dist = min(depth[n] for n in (candidate.node_a, candidate.node_b) if n in depth)
        depth.setdefault(candidate.node_a, dist + 1)
        depth.setdefault(candidate.node_b, dist + 1)
        footprint.add(sid)
        length += candidate.length_m

    return footprint


def _repair_bike(
    graph: nx.MultiGraph,
    index: StreetSegmentIndex,
    by_segment: dict[str, list[tuple[Intervention, float]]],
    selected: list[Intervention],
    pending: list[Intervention],
    max_extra_m: float,
    rng: random.Random,
    car: nx.DiGraph,
    support: Counter,
    car_links: dict[str, tuple[str, str]],
    bike: nx.DiGraph,
    anchor: str,
) -> tuple[list[Intervention], nx.DiGraph, Counter, nx.DiGraph] | None:
    """Join new bicycle SCCs with cheap eligible bidirectional connections.

    Operations use working graphs; failing connector paths are rolled back.
    """
    added = []
    excluded = set()

    while True:
        if nx.is_strongly_connected(bike):
            return added, car, support, bike

        sccs = list(nx.strongly_connected_components(bike))
        scc_index = {node: i for i, comp in enumerate(sccs) for node in comp}
        root_nodes = sccs[scc_index[anchor]]
        occupied = {iv.segment_id for iv in selected + pending + added}
        available_budget = max_extra_m - _length(added, index)
        if available_budget <= 1e-9:
            return None

        search = nx.MultiGraph()
        # Free graph traversals only within each existing strongly connected
        # component. Inter-SCC one-way bike arcs must not count as free repair.
        for u, v in bike.edges():
            if scc_index[u] == scc_index[v]:
                search.add_edge(u, v, key=("existing", u, v), cost=0.0, sid=None)

        for u, v, key, attrs in graph.edges(keys=True, data=True):
            sid = attrs.get("segment_id", key)
            if sid in occupied or sid in excluded:
                continue
            if not any(iv.intervention_type in BIKE_TYPES for iv, _ in by_segment.get(sid, ())):
                continue
            length = index.get(sid).length_m
            if length <= available_budget + 1e-9:
                search.add_edge(u, v, key=("candidate", sid), cost=length, sid=sid)

        sources = sorted(root_nodes & set(search.nodes))
        if not sources:
            return None

        distances, paths = nx.multi_source_dijkstra(search, sources, weight="cost")
        targets = [node for node in distances if scc_index.get(node, -1) != scc_index[anchor]]
        if not targets:
            return None
        target = min(targets, key=lambda node: (distances[node], str(node)))
        if distances[target] > available_budget + 1e-9:
            return None

        connector_sids = []
        for u, v in zip(paths[target], paths[target][1:]):
            edge = min(search[u][v].values(), key=lambda data: (
                data["cost"], str(data["sid"])
            ))
            if edge["sid"] is not None and edge["sid"] not in connector_sids:
                connector_sids.append(edge["sid"])
        if not connector_sids:
            return None

        # Path-level transaction: do not keep failed partial connector paths.
        trial_car = car.copy()
        trial_support = support.copy()
        trial_bike = bike.copy()
        trial = []
        failed = None
        for sid in connector_sids:
            options = [pair for pair in by_segment[sid] if pair[0].intervention_type in BIKE_TYPES]
            iv = _choose(
                options, selected + pending + added + trial,
                trial_car, trial_support, car_links, rng, safest=True,
            )
            if iv is None:
                failed = sid
                break
            trial.append(iv)
            _add_bicycle(trial_bike, iv, index)

        if failed is not None:
            excluded.add(failed)
            continue
        if _length(added + trial, index) > max_extra_m + 1e-9:
            return None
        added.extend(trial)
        car, support, bike = trial_car, trial_support, trial_bike


def sample_clusters(
    graph: nx.MultiGraph,
    index: StreetSegmentIndex,
    network: MatsimNetwork,
    candidates: list[Intervention],
    budget_m: float,
    *,
    seed: int | None = None,
    max_cluster_length_m: float | None = 1000.0,
    pbl_weight_multiplier: float = 3.0,
    max_failed_attempts: int = 60,
) -> list[Intervention]:
    """Sample physically connected multimodal clusters using graph-only checks.

    Each StreetSegment gets one weighted-random intervention (including individual
    bus lanes and destination-only roads). New bicycle infrastructure may be
    disconnected while the cluster grows; repairs are attempted before commit.

    The baseline MATSim network is read-only throughout sampling. For exactly
    the configured connectivity constraints, no MATSim network is rebuilt.
    """
    if not isinstance(graph, nx.MultiGraph) or graph.is_directed():
        raise TypeError("Expected an undirected nx.MultiGraph")
    if not isfinite(budget_m) or budget_m < 0:
        raise ValueError("budget_m must be finite and nonnegative")
    if max_cluster_length_m is not None and (
        not isfinite(max_cluster_length_m) or max_cluster_length_m <= 0
    ):
        raise ValueError("max_cluster_length_m must be finite and positive")
    if not isfinite(pbl_weight_multiplier) or pbl_weight_multiplier <= 0:
        raise ValueError("pbl_weight_multiplier must be finite and positive")
    if max_failed_attempts < 1:
        raise ValueError("max_failed_attempts must be positive")

    rng = random.Random(seed)
    by_segment = defaultdict(list)
    for iv in candidates:
        if iv.intervention_type == InterventionType.DEDICATED_BUS_LANE and len(iv.target_link_ids) != 1:
            raise ValueError(f"Bus lane must target one directional link: {iv}")
        weight = intervention_weight(iv, index, network) * (
            pbl_weight_multiplier
            if iv.intervention_type == InterventionType.PROTECTED_BIKE_LANE else 1.0
        )
        if weight <= 0 or not isfinite(weight):
            raise ValueError(f"Invalid weight for intervention {iv}: {weight}")
        by_segment[iv.segment_id].append((iv, weight))

    graph_segments = {
        attrs.get("segment_id", key)
        for _, _, key, attrs in graph.edges(keys=True, data=True)
    }
    available = graph_segments & set(by_segment)
    for sid in available:
        length = index.get(sid).length_m
        if not isfinite(length) or length <= 0:
            raise ValueError(f"Invalid physical segment length: {sid}")

    # Convert baseline modal links once, then never apply/copy MATSim in sampling.
    car = build_mode_graph(network, "car")
    bike = build_mode_graph(network, "bike")
    if not nx.is_strongly_connected(car) or not nx.is_strongly_connected(bike):
        raise ValueError("Baseline car and bike networks must be strongly connected")
    anchor = next(iter(bike.nodes))
    car_links = {
        link.id: (link.from_node, link.to_node)
        for link in network.links.values() if "car" in link.allowed_modes
    }
    support = Counter(car_links.values())

    selected: list[Intervention] = []
    selected_length = 0.0
    failed_attempts = 0

    while available and failed_attempts < max_failed_attempts:
        remaining = budget_m - selected_length
        if remaining <= 1e-9:
            break

        cap = min(remaining, max_cluster_length_m or remaining)
        seeds = [sid for sid in available if index.get(sid).length_m <= cap + 1e-9]
        if not seeds:
            break
        seed_sid = rng.choice(sorted(seeds))

        footprint_target = max(
            index.get(seed_sid).length_m,
            cap * rng.uniform(0.65, 0.85),
        )
        footprint = _grow_footprint(seed_sid, available, graph, index, footprint_target, rng)

        # One copy per proposed cluster, not once per variant feasibility test.
        working_car = car.copy()
        working_bike = bike.copy()
        working_support = support.copy()
        pending: list[Intervention] = []
        for sid in sorted(footprint):
            iv = _choose(
                by_segment[sid], selected + pending,
                working_car, working_support, car_links, rng,
            )
            if iv is None:
                break
            pending.append(iv)
            _add_bicycle(working_bike, iv, index)

        if len(pending) != len(footprint):
            failed_attempts += 1
            continue

        extra = cap - _length(pending, index)
        repaired = _repair_bike(
            graph, index, by_segment, selected, pending,
            extra, rng, working_car, working_support, car_links, working_bike, anchor,
        )
        if repaired is None:
            failed_attempts += 1
            continue

        connectors, working_car, working_support, working_bike = repaired
        cluster = pending + connectors
        cluster_cost = _length(cluster, index)
        if cluster_cost > cap + 1e-9 or not (
            nx.is_strongly_connected(working_car)
            and nx.is_strongly_connected(working_bike)
        ):
            failed_attempts += 1
            continue

        selected.extend(cluster)
        selected_length += cluster_cost
        available.difference_update(iv.segment_id for iv in cluster)
        car, support, bike = working_car, working_support, working_bike
        failed_attempts = 0

    if selected_length > budget_m + 1e-6:
        raise AssertionError("Cluster sampler exceeded its length budget")
    return selected

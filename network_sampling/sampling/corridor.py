
"""Sample connected cycling corridors, favoring less intrusive routing options.

Corridors are paths on the undirected physical StreetSegment MultiGraph.
Interventions are selected separately, using the random sampler's weights.
The returned package must still pass directed MATSim connectivity validation.
"""
from __future__ import annotations

import random
from collections import defaultdict
from math import isfinite, sqrt

import networkx as nx

from network_sampling.interventions.compatibility import interventions_compatible
from network_sampling.interventions.intervention_types import Intervention, InterventionType
from network_sampling.network.matsim_network import MatsimNetwork
from network_sampling.network.street_segments import StreetSegmentIndex
from network_sampling.sampling.random import WEIGHT_KEEP_CAR, intervention_weight


BIKE_TYPES = {InterventionType.PROTECTED_BIKE_LANE, InterventionType.BICYCLE_STREET}


def _is_protected_bike_link(link) -> bool:
    """Baseline anchor: bike-only link or an existing bicycle street."""
    return (
        set(link.allowed_modes) == {"bike"}
        or str(link.attributes.get("cyclestreet", "")).strip().lower() == "yes"
    )


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
) -> list[Intervention]:
    """Sample physical corridor paths within budget, without modifying MATSim.

    Routing uses length * (1 + intrusion_strength * expected_intrusion).
    Expected intrusion is calculated from the available intervention choices
    using their original sampling weights. Selection remains stochastic.

    pbl_weight_multiplier adjusts PBL frequency without changing the
    relative road-space-variant weights or the physical corridor routing.
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

    # Each option stores (intervention, original_weight, intrusion_penalty).
    # Original weights 4, 2, 1, 0.25 yield penalties 0, 0.41, 1, 3.
    # The optional PBL multiplier affects selection only, not routing.
    by_segment = defaultdict(list)

    for iv in candidates:
        if iv.intervention_type not in BIKE_TYPES:
            continue

        base_weight = intervention_weight(iv, index, network)
        if base_weight <= 0:
            raise ValueError(f"Nonpositive intervention weight: {iv}")

        severity = sqrt(WEIGHT_KEEP_CAR / base_weight) - 1.0
        by_segment[iv.segment_id].append((iv, base_weight, severity))

    # Build a graph containing eligible cycling-intervention segments.
    available = nx.MultiGraph()

    for u, v, key, data in graph.edges(keys=True, data=True):
        sid = data.get("segment_id")
        if sid is None:
            sid = getattr(data.get("segment"), "id", key)

        if sid not in by_segment or u == v:
            continue

        segment = index.get(sid)

        if not isfinite(segment.length_m) or segment.length_m <= 0:
            raise ValueError(f"Invalid length for segment {sid}")
        if {u, v} != {segment.node_a, segment.node_b}:
            raise ValueError(f"Physical graph endpoints differ for {sid}")

        options = by_segment[sid]
        total_weight = sum(weight for _, weight, _ in options)
        expected_intrusion = sum(
            weight * severity for _, weight, severity in options
        ) / total_weight

        routing_factor = 1.0 + intrusion_strength * expected_intrusion

        available.add_edge(
            u, v,
            key=sid,
            length_m=segment.length_m,
            routing_cost=segment.length_m * routing_factor,
            routing_factor=routing_factor,
        )

    if by_segment and available.number_of_edges() == 0:
        raise ValueError("No candidate segment IDs match physical MultiGraph edges")

    # Only protected cycling infrastructure provides baseline anchors.
    protected_nodes = {
        node
        for link in network.links.values()
        if _is_protected_bike_link(link)
        for node in (link.from_node, link.to_node)
    }

    anchors = protected_nodes & set(available.nodes)
    selected: list[Intervention] = []
    spent = 0.0

    # Continue creating corridors until the budget or feasible paths run out.
    while spent < budget_m - 1e-9:
        anchors &= set(available.nodes)

        if not anchors or available.number_of_edges() == 0:
            break

        remaining = budget_m - spent
        target_m = min(remaining, max_corridor_length_m or remaining)

        # Dijkstra minimizes intrusion-weighted distance.
        # Physical distance is calculated separately for budget accounting.
        distance, paths = nx.multi_source_dijkstra(
            available,
            sorted(anchors),
            weight="routing_cost",
        )

        physical_distance = {}
        parent_edges = {}

        # Recover the actual MultiGraph edge chosen between each node pair.
        for node in sorted(distance, key=distance.__getitem__):
            path = paths[node]

            if len(path) == 1:
                physical_distance[node] = 0.0
                continue

            parent = path[-2]

            edge_id = min(
                available[parent][node],
                key=lambda k: (
                    available[parent][node][k]["routing_cost"],
                    str(k),
                ),
            )

            parent_edges[node] = edge_id
            physical_distance[node] = (
                physical_distance[parent]
                + available[parent][node][edge_id]["length_m"]
            )

        # Identify seed segments reachable within the physical budget.
        options = []

        for u, v, sid, data in available.edges(keys=True, data=True):
            if u not in distance or v not in distance:
                continue

            near, far = (
                (u, v) if distance[u] <= distance[v] else (v, u)
            )

            physical_cost = physical_distance[near] + data["length_m"]

            if physical_cost <= target_m + 1e-9 and far not in paths[near]:
                options.append((sid, near, far, far not in anchors))

        if not options:
            break

        # Prefer seeds away from protected infrastructure.
        remote_options = [option for option in options if option[3]]
        seed_sid, near, far, _ = rng.choice(sorted(remote_options or options))

        # Reconstruct the weighted shortest path connecting the seed.
        node_path = list(paths[near]) + [far]
        segment_path = [parent_edges[node] for node in paths[near][1:]]
        segment_path.append(seed_sid)

        length_m = sum(index.get(sid).length_m for sid in segment_path)
        visited = set(node_path)

        # Extend the corridor stochastically, favoring the same OSM way
        # and neighboring segments with lower expected intrusion.
        while True:
            end = node_path[-1]
            last_way = index.get(segment_path[-1]).osm_way_id

            steps, weights = [], []

            for next_node, edges in available[end].items():
                if next_node in visited:
                    continue

                for edge_id, data in edges.items():
                    segment = index.get(edge_id)

                    if length_m + segment.length_m > target_m + 1e-9:
                        continue

                    same_way = (
                        3.0 if last_way and segment.osm_way_id == last_way
                        else 1.0
                    )

                    steps.append((edge_id, next_node))
                    weights.append(same_way / data["routing_factor"])

            if not steps:
                break

            next_sid, next_node = rng.choices(
                steps, weights=weights, k=1
            )[0]

            segment_path.append(next_sid)
            node_path.append(next_node)
            visited.add(next_node)
            length_m += index.get(next_sid).length_m

        # Select interventions independently of the routing decision.
        # Retain the original random sampler's less-intrusive weights.
        # The optional PBL multiplier controls representation in the dataset.
        previous_type = None

        for edge_id in segment_path:
            options = by_segment[edge_id]

            weights = [
                weight
                * (
                    pbl_weight_multiplier
                    if iv.intervention_type == InterventionType.PROTECTED_BIKE_LANE
                    else 1.0
                )
                * (3.0 if iv.intervention_type == previous_type else 1.0)
                for iv, weight, _ in options
            ]

            iv = rng.choices(
                [item[0] for item in options],
                weights=weights,
                k=1,
            )[0]

            if any(
                not interventions_compatible(iv, prev, network)
                for prev in selected if prev.segment_id == iv.segment_id
            ):
                raise RuntimeError(
                    f"Unexpected compatibility conflict on {edge_id}"
                )

            selected.append(iv)
            previous_type = iv.intervention_type

        spent += length_m

        # Newly selected cycling infrastructure can anchor later corridors.
        for iv in selected[-len(segment_path):]:
            if iv.intervention_type in BIKE_TYPES:
                segment = index.get(iv.segment_id)
                anchors.update((segment.node_a, segment.node_b))

        # Do not sample the same physical segment twice.
        for edge_id in segment_path:
            segment = index.get(edge_id)
            available.remove_edge(
                segment.node_a,
                segment.node_b,
                key=edge_id,
            )

    return selected

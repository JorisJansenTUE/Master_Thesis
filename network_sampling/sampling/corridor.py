
"""Sample cycling corridors with intrusion-weighted routing and car connectivity.

Corridors use the physical StreetSegment MultiGraph. Intervention choices retain
weighted randomness. Only car-removing variants require connectivity checks,
since other cycling interventions preserve car/truck access.
"""
from __future__ import annotations

import random
from collections import defaultdict
from math import isfinite, sqrt

import networkx as nx

from network_sampling.interventions.compatibility import interventions_compatible
from network_sampling.interventions.intervention_types import (
    Intervention,
    InterventionType,
    RoadSpaceVariant,
)
from network_sampling.interventions.matsim_app.apply_network import apply_interventions
from network_sampling.network.matsim_network import MatsimNetwork
from network_sampling.network.street_segments import StreetSegmentIndex
from network_sampling.sampling.random import WEIGHT_KEEP_CAR, intervention_weight
from network_sampling.validation.connectivity import check_connectivity


BIKE_TYPES = {
    InterventionType.PROTECTED_BIKE_LANE,
    InterventionType.BICYCLE_STREET,
}


def _is_protected_bike_link(link) -> bool:
    """Baseline anchor: bike-only link or existing bicycle street."""
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
    """Sample cycling corridors while preserving directed car connectivity.

    Routing favors less intrusive streets. Intervention selection retains
    the original weighted probabilities.

    REMOVE_CAR candidates are checked using the existing MATSim application
    and connectivity validator. Corridors are accepted atomically.

    Assumes the baseline car and truck modal graphs are identical.
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

    # -------------------------------------------------------------
    # Eligible cycling interventions and routing penalties
    # -------------------------------------------------------------

    # Each option contains:
    # (intervention, original selection weight, intrusion penalty)
    by_segment = defaultdict(list)

    for iv in candidates:
        if iv.intervention_type not in BIKE_TYPES:
            continue

        weight = intervention_weight(iv, index, network)

        if weight <= 0:
            raise ValueError(f"Nonpositive intervention weight: {iv}")

        severity = sqrt(WEIGHT_KEEP_CAR / weight) - 1.0
        by_segment[iv.segment_id].append((iv, weight, severity))

    # Build the physical graph containing eligible cycling streets.
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

        factor = 1.0 + intrusion_strength * expected_intrusion

        available.add_edge(
            u,
            v,
            key=sid,
            length_m=segment.length_m,
            routing_cost=segment.length_m * factor,
            routing_factor=factor,
        )

    if by_segment and available.number_of_edges() == 0:
        raise ValueError(
            "No candidate segment IDs match physical MultiGraph edges"
        )

    # -------------------------------------------------------------
    # Initial protected cycling infrastructure
    # -------------------------------------------------------------

    anchors = {
        node
        for link in network.links.values()
        if _is_protected_bike_link(link)
        for node in (link.from_node, link.to_node)
    } & set(available.nodes)

    selected: list[Intervention] = []
    accepted_removals: list[Intervention] = []
    spent = 0.0

    # -------------------------------------------------------------
    # Generate corridors until the available budget is exhausted
    # -------------------------------------------------------------

    while spent < budget_m - 1e-9:
        anchors &= set(available.nodes)

        if not anchors or available.number_of_edges() == 0:
            break

        remaining = budget_m - spent
        target_m = min(
            remaining,
            max_corridor_length_m or remaining,
        )

        # ---------------------------------------------------------
        # Weighted shortest paths to protected infrastructure
        # ---------------------------------------------------------

        distance, paths = nx.multi_source_dijkstra(
            available,
            sorted(anchors),
            weight="routing_cost",
        )

        physical_distance = {}
        parent_edges = {}

        # Recover actual MultiGraph edges used by weighted Dijkstra.
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

        # ---------------------------------------------------------
        # Select corridor seed
        # ---------------------------------------------------------

        seeds = []

        for u, v, sid, data in available.edges(keys=True, data=True):
            if u not in distance or v not in distance:
                continue

            near, far = (
                (u, v) if distance[u] <= distance[v] else (v, u)
            )

            physical_cost = (
                physical_distance[near] + data["length_m"]
            )

            if (
                physical_cost <= target_m + 1e-9
                and far not in paths[near]
            ):
                seeds.append((sid, near, far, far not in anchors))

        if not seeds:
            break

        remote_seeds = [seed for seed in seeds if seed[3]]

        seed_sid, near, far, _ = rng.choice(
            sorted(remote_seeds or seeds)
        )

        # Reconstruct the connection from the anchor to the seed.
        node_path = list(paths[near]) + [far]

        segment_path = [
            parent_edges[node]
            for node in paths[near][1:]
        ]
        segment_path.append(seed_sid)

        length_m = sum(
            index.get(sid).length_m
            for sid in segment_path
        )

        visited = set(node_path)

        # ---------------------------------------------------------
        # Extend corridor through neighboring StreetSegments
        # ---------------------------------------------------------

        while True:
            end = node_path[-1]
            last_way = index.get(segment_path[-1]).osm_way_id

            steps = []
            weights = []

            for next_node, edges in available[end].items():
                if next_node in visited:
                    continue

                for sid, data in edges.items():
                    segment = index.get(sid)

                    if length_m + segment.length_m > target_m + 1e-9:
                        continue

                    same_way = (
                        3.0
                        if last_way and segment.osm_way_id == last_way
                        else 1.0
                    )

                    steps.append((sid, next_node))
                    weights.append(
                        same_way / data["routing_factor"]
                    )

            if not steps:
                break

            sid, next_node = rng.choices(
                steps,
                weights=weights,
                k=1,
            )[0]

            segment_path.append(sid)
            node_path.append(next_node)
            visited.add(next_node)

            length_m += index.get(sid).length_m

        # ---------------------------------------------------------
        # Connectivity-aware intervention selection
        # ---------------------------------------------------------

        # Keep this corridor temporary until every segment has a
        # feasible intervention.
        pending: list[Intervention] = []
        pending_removals: list[Intervention] = []

        previous_type = None
        blocked_sid = None

        for sid in segment_path:
            options = by_segment[sid]

            weights = [
                weight
                * (
                    pbl_weight_multiplier
                    if iv.intervention_type
                    == InterventionType.PROTECTED_BIKE_LANE
                    else 1.0
                )
                * (
                    3.0
                    if iv.intervention_type == previous_type
                    else 1.0
                )
                for iv, weight, _ in options
            ]

            # Weighted random ordering without replacement.
            # The first option has the highest expected priority,
            # but alternatives remain available if it is infeasible.
            ordered = sorted(
                range(len(options)),
                key=lambda i: rng.expovariate(weights[i]),
            )

            chosen = None

            for i in ordered:
                iv = options[i][0]

                if any(
                    not interventions_compatible(iv, prev, network)
                    for prev in selected + pending
                    if prev.segment_id == sid
                ):
                    continue

                # Only REMOVE_CAR can disconnect the existing car
                # network. Test all accumulated removals together.
                if iv.road_space_variant == RoadSpaceVariant.REMOVE_CAR:
                    removals = (
                        accepted_removals
                        + pending_removals
                        + [iv]
                    )

                    modified, _ = apply_interventions(
                        network,
                        index,
                        removals,
                    )

                    if not check_connectivity(network, modified)["car"]:
                        continue

                chosen = iv
                break

            if chosen is None:
                blocked_sid = sid
                break

            pending.append(chosen)

            if chosen.road_space_variant == RoadSpaceVariant.REMOVE_CAR:
                pending_removals.append(chosen)

            previous_type = chosen.intervention_type

        # ---------------------------------------------------------
        # Reject infeasible corridors
        # ---------------------------------------------------------

        if blocked_sid is not None:
            # No intervention on this segment preserves connectivity
            # with the already accepted removals and current trial.
            #
            # Discard the entire corridor and exclude this segment
            # from subsequent corridor searches in this sample.
            segment = index.get(blocked_sid)

            available.remove_edge(
                segment.node_a,
                segment.node_b,
                key=blocked_sid,
            )
            continue

        # ---------------------------------------------------------
        # Accept complete corridor
        # ---------------------------------------------------------

        selected.extend(pending)
        accepted_removals.extend(pending_removals)

        spent += length_m

        # Newly selected cycling infrastructure can act as an anchor
        # for subsequent corridors.
        for iv in pending:
            segment = index.get(iv.segment_id)

            anchors.update((
                segment.node_a,
                segment.node_b,
            ))

        # Prevent duplicate intervention selection.
        for sid in segment_path:
            segment = index.get(sid)

            available.remove_edge(
                segment.node_a,
                segment.node_b,
                key=sid,
            )

    return selected

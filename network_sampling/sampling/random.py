from __future__ import annotations

import random
import networkx as nx

from network_sampling.interventions.compatibility import interventions_compatible
from network_sampling.interventions.intervention_types import (
    Intervention,
    InterventionType,
    RoadSpaceVariant,
)
from network_sampling.network.matsim_network import MatsimNetwork
from network_sampling.network.street_segments import StreetSegmentIndex


WEIGHT_KEEP_CAR = 4.0
WEIGHT_REMOVE_ONE_LANE = 2.0
WEIGHT_REMOVE_ONE_DIRECTION = 1.0
WEIGHT_REMOVE_ALL_CAR = 0.25


def sample_random(
    interventions: list[Intervention],
    segment_index: StreetSegmentIndex,
    network: MatsimNetwork,
    budget_m: float,
    seed: int | None = None,
) -> list[Intervention]:
    """
    Randomly sample compatible interventions within a length budget.

    Less invasive interventions receive a higher probability. Bike-creating
    interventions must already be part of, or directly connect to, the
    baseline bicycle network.
    """
    if budget_m < 0:
        raise ValueError("budget_m must be non-negative.")

    rng = random.Random(seed)

    bike_nodes = {
        node
        for link in network.links.values()
        if "bike" in link.allowed_modes
        for node in (link.from_node, link.to_node)
    }

    candidates = [
        intervention
        for intervention in interventions
        if _bike_candidate_allowed(intervention, segment_index, network, bike_nodes)
    ]

    # Weighted random order without replacement.
    candidates.sort(
        key=lambda intervention: rng.expovariate(
            intervention_weight(intervention, segment_index, network)
        )
    )

    selected: list[Intervention] = []
    used_budget = 0.0

    # Maintain directed car/truck graphs throughout sampling.
    # These are constructed only once.
    car_graphs = _build_car_graphs(network)

    for candidate in candidates:
        cost = segment_index.get(candidate.segment_id).length_m

        if used_budget + cost > budget_m:
            continue

        if not all(
            interventions_compatible(candidate, existing, network)
            for existing in selected
        ):
            continue

        # Only REMOVE_CAR interventions can disconnect the road network.
        # The function updates the graphs if accepted and restores them
        # if connectivity would be lost.
        if candidate.road_space_variant == RoadSpaceVariant.REMOVE_CAR:
            if not _try_remove_car_links(candidate, network, car_graphs):
                continue

        selected.append(candidate)
        used_budget += cost

    return selected


def intervention_weight(
    intervention: Intervention,
    segment_index: StreetSegmentIndex,
    network: MatsimNetwork,
) -> float:
    """Return the random sampling weight for an intervention."""

    variant = intervention.road_space_variant

    if variant is None or variant == RoadSpaceVariant.KEEP_CAR:
        return WEIGHT_KEEP_CAR

    if variant == RoadSpaceVariant.REMOVE_ONE_LANE:
        return WEIGHT_REMOVE_ONE_LANE

    if variant == RoadSpaceVariant.REMOVE_CAR:
        if _removes_all_car(intervention, segment_index, network):
            return WEIGHT_REMOVE_ALL_CAR

        return WEIGHT_REMOVE_ONE_DIRECTION

    raise ValueError(f"Unsupported road-space variant: {variant}")


def _bike_candidate_allowed(
    intervention: Intervention,
    segment_index: StreetSegmentIndex,
    network: MatsimNetwork,
    bike_nodes: set[str],
) -> bool:
    """
    Restrict random bike interventions to the existing bike network or one
    directly adjacent street.
    """
    if intervention.intervention_type not in {
        InterventionType.PROTECTED_BIKE_LANE,
        InterventionType.BICYCLE_STREET,
    }:
        return True
    segment = segment_index.get(intervention.segment_id)

    # Street already carries bicycle traffic.
    if any(
        "bike" in network.links[link_id].allowed_modes
        for link_id in segment.link_ids
    ):
        return True

    # New bike infrastructure directly attaches to the baseline bike network.
    return segment.node_a in bike_nodes or segment.node_b in bike_nodes


def _removes_all_car(
    intervention: Intervention,
    segment_index: StreetSegmentIndex,
    network: MatsimNetwork,
) -> bool:
    """Return whether REMOVE_CAR eliminates every car direction on the street."""

    segment = segment_index.get(intervention.segment_id)

    car_directions = {
        (link.from_node, link.to_node)
        for link_id in segment.link_ids
        if "car" in (link := network.links[link_id]).allowed_modes
    }

    removed_directions = {
        (network.links[link_id].from_node, network.links[link_id].to_node)
        for link_id in intervention.target_link_ids
    }

    return car_directions <= removed_directions

def _build_car_graphs(
    network: MatsimNetwork,
) -> dict[str, nx.MultiDiGraph]:
    """
    Build directed car and truck graphs once.

    MultiDiGraph preserves parallel MATSim links, so removing one
    link does not accidentally remove another link with the same
    endpoints.
    """
    graphs = {
        "car": nx.MultiDiGraph(),
        "truck": nx.MultiDiGraph(),
    }

    for link_id, link in network.links.items():
        for mode, graph in graphs.items():
            if mode in link.allowed_modes:
                graph.add_edge(
                    link.from_node,
                    link.to_node,
                    key=link_id,
                )

    return graphs


def _try_remove_car_links(
    intervention: Intervention,
    network: MatsimNetwork,
    graphs: dict[str, nx.MultiDiGraph],
) -> bool:
    """
    Tentatively remove car/truck links from the current graphs.

    Accept only if an alternative directed path remains between
    the endpoints of every removed link.

    If accepted, the graphs remain modified.
    If rejected, all removed links are restored.

    This preserves the baseline network's existing directed
    reachability without requiring a full connectivity check.
    """
    removed = []

    # Apply candidate removals temporarily.
    for link_id in intervention.target_link_ids:
        link = network.links[link_id]

        u = link.from_node
        v = link.to_node

        for mode, graph in graphs.items():
            if graph.has_edge(u, v, key=link_id):
                graph.remove_edge(u, v, key=link_id)
                removed.append((mode, u, v, link_id))

    # Check only the removed connections.
    valid = True

    for mode, u, v, _ in removed:
        graph = graphs[mode]

        # Another parallel link still connects these nodes.
        if graph.has_edge(u, v):
            continue

        # Otherwise check whether a directed alternative exists.
        if not nx.has_path(graph, u, v):
            valid = False
            break

    if not valid:
        # Roll back this candidate.
        for mode, u, v, link_id in removed:
            graphs[mode].add_edge(u, v, key=link_id)

    return valid


def intervention_length(
    interventions: list[Intervention],
    segment_index: StreetSegmentIndex,
) -> float:
    """Return the total sampled intervention length in metres."""

    return sum(
        segment_index.get(intervention.segment_id).length_m
        for intervention in interventions
    )
from __future__ import annotations

import random

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

    for candidate in candidates:
        cost = segment_index.get(candidate.segment_id).length_m

        if used_budget + cost > budget_m:
            continue

        if not all(
            interventions_compatible(candidate, existing, network)
            for existing in selected
        ):
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

    if intervention.intervention_type == InterventionType.PT_BIKE_STREET:
        return WEIGHT_REMOVE_ALL_CAR

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
        InterventionType.PT_BIKE_STREET,
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


def intervention_length(
    interventions: list[Intervention],
    segment_index: StreetSegmentIndex,
) -> float:
    """Return the total sampled intervention length in metres."""

    return sum(
        segment_index.get(intervention.segment_id).length_m
        for intervention in interventions
    )
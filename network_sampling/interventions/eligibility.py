from __future__ import annotations

from collections import defaultdict

from network_sampling.interventions.intervention_types import (
    Intervention,
    InterventionType,
    RoadSpaceVariant,
)
from network_sampling.network.matsim_network import MatsimLink, MatsimNetwork
from network_sampling.network.street_segments import StreetSegment
from network_sampling.network.transit_usage import TransitUsageIndex


# Placeholder values, to be finalized later.
BICYCLE_STREET_MIN_WIDTH_M = 5.0
PT_BIKE_STREET_MIN_WIDTH_ONE_WAY_BUS_M = 5.0
PT_BIKE_STREET_MIN_WIDTH_TWO_WAY_BUS_M = 7.0

CAR_LANE_WIDTH_M = 3.0
PBL_BOTH_DIRECTIONS_WIDTH_M = 4.0
BUS_LANE_WIDTH_M = 3.0


def get_eligible_interventions(
    segment: StreetSegment,
    network: MatsimNetwork,
    transit_usage: TransitUsageIndex,
) -> list[Intervention]:
    """
    Return all interventions that can be applied to a street segment.

    Only street segments containing links that currently allow car traffic
    are considered.

    Parameters
    ----------
    segment
        Physical street segment being evaluated.

    network
        Baseline MATSim network.

    transit_usage
        Scheduled bus usage of the baseline network.

    Returns
    -------
    list[Intervention]
        All eligible intervention configurations for the street segment.
    """

    car_links = [
        network.links[link_id]
        for link_id in segment.link_ids
        if "car" in network.links[link_id].allowed_modes
    ]

    if not car_links:
        return []

    interventions = [Intervention(segment.id, InterventionType.DESTINATION_ONLY)]

    if segment.total_width_m is not None and segment.total_width_m >= BICYCLE_STREET_MIN_WIDTH_M:
        interventions.append(Intervention(segment.id, InterventionType.BICYCLE_STREET))

    interventions.extend(_pt_bike_street(segment, transit_usage))
    interventions.extend(_bus_lanes(segment, car_links, transit_usage))
    interventions.extend(_protected_bike_lanes(segment, car_links, transit_usage))

    return interventions


def _pt_bike_street(
    segment: StreetSegment,
    transit_usage: TransitUsageIndex,
) -> list[Intervention]:
    """Return a PT+bicycle street intervention when width and PT use allow it."""

    if segment.total_width_m is None:
        return []

    bus_directions = transit_usage.directions(segment.id)

    if not bus_directions:
        return []

    required_width = (
        PT_BIKE_STREET_MIN_WIDTH_TWO_WAY_BUS_M
        if len(bus_directions) > 1
        else PT_BIKE_STREET_MIN_WIDTH_ONE_WAY_BUS_M
    )

    if segment.total_width_m < required_width:
        return []

    return [Intervention(segment.id, InterventionType.PT_BIKE_STREET)]


def _bus_lanes(
    segment: StreetSegment,
    car_links: list[MatsimLink],
    transit_usage: TransitUsageIndex,
) -> list[Intervention]:
    """
    Return dedicated bus-lane interventions.

    Bus lanes are only created on car links already traversed by a
    scheduled bus route.
    """

    if segment.total_width_m is None:
        return []

    total_lanes = sum(link.permlanes for link in car_links)
    interventions: list[Intervention] = []

    for link in car_links:
        if not transit_usage.uses_link(link.id):
            continue

        required_width = total_lanes * CAR_LANE_WIDTH_M + BUS_LANE_WIDTH_M

        if segment.total_width_m >= required_width:
            variant = RoadSpaceVariant.KEEP_CAR
        elif link.permlanes > 1:
            variant = RoadSpaceVariant.REMOVE_ONE_LANE
        else:
            variant = RoadSpaceVariant.REMOVE_CAR

        interventions.append(
            Intervention(
                segment.id,
                InterventionType.DEDICATED_BUS_LANE,
                road_space_variant=variant,
                target_link_id=link.id,
            )
        )

    return interventions


def _protected_bike_lanes(
    segment: StreetSegment,
    car_links: list[MatsimLink],
    transit_usage: TransitUsageIndex,
) -> list[Intervention]:
    """
    Return eligible protected bicycle lane configurations.

    A PBL is created while retaining all car lanes when sufficient width is
    available. Otherwise, one car lane or one complete car direction is
    removed from the direction containing the most lanes.
    """

    width = segment.total_width_m

    if width is None or width < PBL_BOTH_DIRECTIONS_WIDTH_M:
        return []

    links_by_direction: dict[tuple[str, str], list[MatsimLink]] = defaultdict(list)

    for link in car_links:
        links_by_direction[(link.from_node, link.to_node)].append(link)

    lanes_by_direction = {
        direction: sum(link.permlanes for link in links)
        for direction, links in links_by_direction.items()
    }

    total_lanes = sum(lanes_by_direction.values())

    if width >= total_lanes * CAR_LANE_WIDTH_M + PBL_BOTH_DIRECTIONS_WIDTH_M:
        return [
            Intervention(
                segment.id,
                InterventionType.PROTECTED_BIKE_LANE,
                road_space_variant=RoadSpaceVariant.KEEP_CAR,
            )
        ]

    if width < (total_lanes - 1) * CAR_LANE_WIDTH_M + PBL_BOTH_DIRECTIONS_WIDTH_M:
        return []

    max_lanes = max(lanes_by_direction.values())
    interventions: list[Intervention] = []

    for direction, lanes in lanes_by_direction.items():
        if lanes != max_lanes:
            continue

        links = links_by_direction[direction]
        target_link = max(links, key=lambda link: link.permlanes)

        if lanes > 1:
            variant = RoadSpaceVariant.REMOVE_ONE_LANE

        else:
            if any(transit_usage.uses_link(link.id) for link in links):
                continue

            variant = RoadSpaceVariant.REMOVE_CAR

        interventions.append(
            Intervention(
                segment.id,
                InterventionType.PROTECTED_BIKE_LANE,
                road_space_variant=variant,
                target_link_id=target_link.id,
            )
        )

    return interventions
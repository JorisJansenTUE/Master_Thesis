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

LANE_WIDTH_M = 3.0
PBL_WIDTH_M = 4.0
BUS_LANE_WIDTH_M = 3.0


def get_eligible_interventions(
    segment: StreetSegment,
    network: MatsimNetwork,
    transit_usage: TransitUsageIndex,
) -> list[Intervention]:
    """
    Return all interventions that can be applied to a street segment.

    Only street segments containing links that currently allow car traffic
    can be modified.

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

    interventions.extend(_bicycle_street(segment))
    interventions.extend(_pt_bike_street(segment, transit_usage))
    interventions.extend(_bus_lanes(segment, car_links, transit_usage))
    interventions.extend(_protected_bike_lanes(segment, car_links, transit_usage))

    return interventions


def _bicycle_street(segment: StreetSegment) -> list[Intervention]:
    """Return a bicycle-street intervention when sufficient width is available."""

    if segment.total_width_m is None or segment.total_width_m < BICYCLE_STREET_MIN_WIDTH_M:
        return []

    return [Intervention(segment.id, InterventionType.BICYCLE_STREET)]


def _pt_bike_street(
    segment: StreetSegment,
    transit_usage: TransitUsageIndex,
) -> list[Intervention]:
    """
    Return a PT+bicycle street intervention when width and PT use allow it.

    A different minimum width is used depending on whether scheduled buses
    traverse the segment in one or both directions.
    """

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
    Return eligible dedicated bus-lane interventions.

    A bus lane can only be created on a car link already used by a scheduled
    bus. If sufficient extra width exists, all car lanes are retained.
    Otherwise, one car lane on the target link is converted to a bus lane.
    """

    if segment.total_width_m is None:
        return []

    total_lanes = sum(link.permlanes for link in car_links)
    required_width_keep_car = total_lanes * LANE_WIDTH_M + BUS_LANE_WIDTH_M

    interventions: list[Intervention] = []

    for link in car_links:
        if not transit_usage.uses_link(link.id):
            continue

        if segment.total_width_m >= required_width_keep_car:
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
                target_link_ids=(link.id,),
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

    A bidirectional PBL requires 4 m. Existing car lanes are assumed to
    require at least 3 m each. Car space is progressively removed until
    sufficient width is available.

    Parameters
    ----------
    segment
        Physical street segment being evaluated.

    car_links
        MATSim links on the segment that currently allow car traffic.

    transit_usage
        Scheduled bus usage of the baseline network.

    Returns
    -------
    list[Intervention]
        Least disruptive eligible PBL configurations.
    """

    width = segment.total_width_m

    if width is None or width < PBL_WIDTH_M:
        return []

    links_by_direction: dict[tuple[str, str], list[MatsimLink]] = defaultdict(list)

    for link in car_links:
        links_by_direction[(link.from_node, link.to_node)].append(link)

    lanes_by_direction = {
        direction: sum(link.permlanes for link in links)
        for direction, links in links_by_direction.items()
    }

    total_lanes = sum(lanes_by_direction.values())

    # 1. Keep all existing car lanes.
    if width >= total_lanes * LANE_WIDTH_M + PBL_WIDTH_M:
        return [
            Intervention(
                segment.id,
                InterventionType.PROTECTED_BIKE_LANE,
                road_space_variant=RoadSpaceVariant.KEEP_CAR,
            )
        ]

    max_lanes = max(lanes_by_direction.values())

    # 2. Remove one lane from one direction with the most lanes.
    required_width = (total_lanes - 1) * LANE_WIDTH_M + PBL_WIDTH_M

    if width >= required_width:
        interventions = []

        for direction, lanes in lanes_by_direction.items():
            if lanes != max_lanes:
                continue

            target_link = max(links_by_direction[direction], key=lambda link: link.permlanes)

            if target_link.permlanes > 1:
                interventions.append(
                    Intervention(
                        segment.id,
                        InterventionType.PROTECTED_BIKE_LANE,
                        road_space_variant=RoadSpaceVariant.REMOVE_ONE_LANE,
                        target_link_ids=(target_link.id,),
                    )
                )

        if interventions:
            return interventions

    # 3. Remove one lane from both directions.
    if len(links_by_direction) == 2:
        target_links = [
            max(links, key=lambda link: link.permlanes)
            for links in links_by_direction.values()
        ]

        required_width = (total_lanes - 2) * LANE_WIDTH_M + PBL_WIDTH_M

        if (
            all(link.permlanes > 1 for link in target_links)
            and width >= required_width
        ):
            return [
                Intervention(
                    segment.id,
                    InterventionType.PROTECTED_BIKE_LANE,
                    road_space_variant=RoadSpaceVariant.REMOVE_ONE_LANE,
                    target_link_ids=tuple(link.id for link in target_links),
                )
            ]

    # 4. Remove one complete car direction.
    interventions = []

    for direction, lanes in lanes_by_direction.items():
        if lanes != max_lanes:
            continue

        links = links_by_direction[direction]
        required_width = (total_lanes - lanes) * LANE_WIDTH_M + PBL_WIDTH_M

        if width < required_width:
            continue

        if any(transit_usage.uses_link(link.id) for link in links):
            continue

        interventions.append(
            Intervention(
                segment.id,
                InterventionType.PROTECTED_BIKE_LANE,
                road_space_variant=RoadSpaceVariant.REMOVE_CAR,
                target_link_ids=tuple(link.id for link in links),
            )
        )
    # Ensures that only the least disruptive interventions are returned. If multiple directions have the same number of lanes, all are returned.
    if interventions:
        return interventions

    # 5. Remove all car traffic from the street.
    if any(transit_usage.uses_link(link.id) for link in car_links):
        return []

    return [
        Intervention(
            segment.id,
            InterventionType.PROTECTED_BIKE_LANE,
            road_space_variant=RoadSpaceVariant.REMOVE_CAR,
            target_link_ids=tuple(link.id for link in car_links),
        )
    ]
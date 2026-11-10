from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from itertools import combinations

from network_sampling.interventions.compatibility import interventions_compatible
from network_sampling.interventions.eligibility import BUS_LANE_WIDTH_M, PBL_WIDTH_M
from network_sampling.interventions.intervention_types import (
    Intervention,
    InterventionType,
    RoadSpaceVariant,
)
from network_sampling.network.matsim_network import MatsimLink, MatsimNetwork
from network_sampling.network.street_segments import StreetSegment, StreetSegmentIndex


GENERAL_CAR_MODES = {"car", "car_passenger", "truck", "taxi"}

DESTINATION_ATTRIBUTE = "destination_road"
BICYCLE_STREET_ATTRIBUTE = "cyclestreet"
WIDTH_ATTRIBUTE = "carriageway_width_m"

BICYCLE_STREET_SPEED_M_S = 20.0 / 3.6
DESTINATION_ONLY_SPEED_M_S = 30.0 / 3.6


def apply_interventions(
    baseline_network: MatsimNetwork,
    segment_index: StreetSegmentIndex,
    interventions: list[Intervention],
) -> tuple[MatsimNetwork, dict[str, str]]:
    """
    Apply interventions to a copy of the baseline MATSim network.

    Each physical street segment can receive at most one intervention.
    The baseline network and StreetSegmentIndex remain unchanged.

    Parameters
    ----------
    baseline_network
        Original MATSim network.

    segment_index
        Mapping between physical street segments and baseline MATSim links.

    interventions
        Selected interventions to apply.

    Returns
    -------
    tuple[MatsimNetwork, dict[str, str]]
        Modified network and mapping from original bus-route links to newly
        created dedicated bus-lane links.
    """

    _validate_interventions(baseline_network, segment_index, interventions)

    network = deepcopy(baseline_network)
    bus_link_replacements: dict[str, str] = {}

    for intervention in interventions:
        segment = segment_index.get(intervention.segment_id)

        if intervention.intervention_type == InterventionType.DESTINATION_ONLY:
            _apply_destination_only(network, segment)

        elif intervention.intervention_type == InterventionType.BICYCLE_STREET:
            _apply_bicycle_street(network, segment)

        elif intervention.intervention_type == InterventionType.PROTECTED_BIKE_LANE:
            _apply_protected_bike_lane(network, segment, intervention)

        elif intervention.intervention_type == InterventionType.DEDICATED_BUS_LANE:
            replacement = _apply_dedicated_bus_lane(network, intervention)
            bus_link_replacements.update(replacement)

        else:
            raise ValueError(
                f"Unsupported intervention type: {intervention.intervention_type}"
            )

    return network, bus_link_replacements


def _apply_destination_only(network: MatsimNetwork, segment: StreetSegment) -> None:
    """Mark car links as destination-only and limit their free speed to 30 km/h."""

    for link in _car_links(network, segment):
        attributes = dict(link.attributes)
        attributes[DESTINATION_ATTRIBUTE] = "Yes"

        network.links[link.id] = replace(
            link,
            freespeed=DESTINATION_ONLY_SPEED_M_S,
            attributes=attributes,
        )


def _apply_bicycle_street(network: MatsimNetwork, segment: StreetSegment) -> None:
    """Create a bidirectional bicycle street, preserving existing car directions."""

    car_links = _car_links(network, segment)

    # Allow bicycles on every existing car link.
    for link in car_links:
        attributes = dict(link.attributes)
        attributes[BICYCLE_STREET_ATTRIBUTE] = "yes"

        network.links[link.id] = replace(
            link,
            allowed_modes=frozenset(set(link.allowed_modes) | {"bike"}),
            freespeed=BICYCLE_STREET_SPEED_M_S,
            attributes=attributes,
        )

    # Determine which cycling directions exist on this physical segment.
    bike_directions = {
        (network.links[link_id].from_node, network.links[link_id].to_node)
        for link_id in segment.link_ids
        if "bike" in network.links[link_id].allowed_modes
    }

    # Add a bicycle-only reverse link when necessary.
    for from_node, to_node in (
        (segment.node_a, segment.node_b),
        (segment.node_b, segment.node_a),
    ):
        if (from_node, to_node) in bike_directions:
            continue

        template = car_links[0]

        attributes = dict(template.attributes)
        attributes[BICYCLE_STREET_ATTRIBUTE] = "yes"

        # This is a virtual reverse cycling connection, not extra road width.
        attributes.pop(WIDTH_ATTRIBUTE, None)

        link_id = _new_link_id(
            network,
            f"{segment.id}__bike__{from_node}__{to_node}",
        )

        network.links[link_id] = replace(
            template,
            id=link_id,
            from_node=from_node,
            to_node=to_node,
            allowed_modes=frozenset({"bike"}),
            freespeed=BICYCLE_STREET_SPEED_M_S,
            permlanes=1.0,
            capacity=template.capacity / template.permlanes,
            attributes=attributes,
            xml_attributes=dict(template.xml_attributes),
        )

        bike_directions.add((from_node, to_node))



def _apply_protected_bike_lane(
    network: MatsimNetwork,
    segment: StreetSegment,
    intervention: Intervention,
) -> None:
    """
    Apply the car-space change and create a bidirectional protected bicycle lane.

    REMOVE_CAR links are reused as bicycle-only links. Otherwise new parallel
    bicycle links are created in both directions.
    """

    if intervention.road_space_variant is None:
        raise ValueError("Protected bicycle lane requires a road-space variant.")

    _apply_road_space(network, intervention)

    reused_directions: set[tuple[str, str]] = set()

    if intervention.road_space_variant == RoadSpaceVariant.REMOVE_CAR:
        for link_id in intervention.target_link_ids:
            link = network.links[link_id]
            reused_directions.add((link.from_node, link.to_node))
            network.links[link_id] = _make_pbl_link(link, link.id)

    for from_node, to_node in (
        (segment.node_a, segment.node_b),
        (segment.node_b, segment.node_a),
    ):
        if (from_node, to_node) in reused_directions:
            continue

        template = _direction_template(network, segment, from_node, to_node)
        link_id = _new_link_id(network, f"{segment.id}__pbl__{from_node}__{to_node}")

        network.links[link_id] = _make_pbl_link(
            template,
            link_id,
            from_node=from_node,
            to_node=to_node,
        )


def _apply_dedicated_bus_lane(
    network: MatsimNetwork,
    intervention: Intervention,
) -> dict[str, str]:
    """Create a dedicated bus link on the targeted directional car link."""

    if len(intervention.target_link_ids) != 1:
        raise ValueError("Dedicated bus lane requires exactly one target link.")

    if intervention.road_space_variant is None:
        raise ValueError("Dedicated bus lane requires a road-space variant.")

    target_id = intervention.target_link_ids[0]
    original_link = deepcopy(network.links[target_id])

    # Reduce/remove general car space where required.
    _apply_road_space(network, intervention)

    # PT2MATSim convention for dedicated public-transport links.
    new_link_id = f"{target_id}_spec"

    if new_link_id in network.links:
        raise ValueError(
            f"Dedicated bus lane already exists for link {target_id}: {new_link_id}"
        )

    # Bus should use the new dedicated link instead of the general road link.
    general_link = network.links[target_id]
    general_modes = set(general_link.allowed_modes)
    general_modes.discard("bus")

    network.links[target_id] = replace(
        general_link,
        allowed_modes=frozenset(general_modes),
    )

    # Create the dedicated bus/PT link.
    attributes = dict(original_link.attributes)
    attributes.pop(WIDTH_ATTRIBUTE, None)
    attributes["dedicated_bus_lane"] = "yes"

    capacity_per_lane = original_link.capacity / original_link.permlanes

    network.links[new_link_id] = replace(
        original_link,
        id=new_link_id,
        permlanes=1.0,
        capacity=capacity_per_lane,
        allowed_modes=frozenset({"bus", "pt"}),
        attributes=attributes,
        xml_attributes=dict(original_link.xml_attributes),
    )

    return {target_id: new_link_id}


def _apply_road_space(network: MatsimNetwork, intervention: Intervention) -> None:
    """Apply KEEP_CAR, REMOVE_ONE_LANE or REMOVE_CAR to targeted MATSim links."""

    variant = intervention.road_space_variant

    if variant == RoadSpaceVariant.KEEP_CAR:
        return

    if not intervention.target_link_ids:
        raise ValueError(f"{variant.value} requires at least one target link.")

    for link_id in intervention.target_link_ids:
        link = network.links[link_id]

        if variant == RoadSpaceVariant.REMOVE_ONE_LANE:
            if link.permlanes <= 1:
                raise ValueError(
                    f"Cannot remove a lane from link {link.id} with "
                    f"{link.permlanes} lane(s)."
                )

            new_lanes = link.permlanes - 1
            new_capacity = link.capacity * new_lanes / link.permlanes

            network.links[link.id] = replace(
                link,
                permlanes=new_lanes,
                capacity=new_capacity,
            )

        elif variant == RoadSpaceVariant.REMOVE_CAR:
            allowed_modes = set(link.allowed_modes) - GENERAL_CAR_MODES

            network.links[link.id] = replace(
                link,
                allowed_modes=allowed_modes,
            )

        else:
            raise ValueError(f"Unsupported road-space variant: {variant}")


def _make_pbl_link(
    template: MatsimLink,
    link_id: str,
    from_node: str | None = None,
    to_node: str | None = None,
) -> MatsimLink:
    """Create or convert a MATSim link representing a protected bicycle lane."""

    attributes = dict(template.attributes)
    attributes["highway"] = "bicycle_designated"
    attributes["protected_bike_lane"] = "yes"
    attributes[WIDTH_ATTRIBUTE] = PBL_WIDTH_M / 2

    capacity_per_lane = template.capacity / template.permlanes

    return replace(
        template,
        id=link_id,
        from_node=from_node or template.from_node,
        to_node=to_node or template.to_node,
        permlanes=1.0,
        capacity=capacity_per_lane,
        allowed_modes={"bike"},
        attributes=attributes,
        xml_attributes=dict(template.xml_attributes),
    )


def _direction_template(
    network: MatsimNetwork,
    segment: StreetSegment,
    from_node: str,
    to_node: str,
) -> MatsimLink:
    """Return a segment link in the requested direction, or another link as fallback."""

    for link_id in segment.link_ids:
        link = network.links[link_id]

        if link.from_node == from_node and link.to_node == to_node:
            return link

    return network.links[segment.link_ids[0]]


def _car_links(network: MatsimNetwork, segment: StreetSegment) -> list[MatsimLink]:
    """Return baseline segment links that currently allow car traffic."""

    return [
        network.links[link_id]
        for link_id in segment.link_ids
        if "car" in network.links[link_id].allowed_modes
    ]


def _new_link_id(network: MatsimNetwork, base_id: str) -> str:
    """Return a unique MATSim link ID based on the requested ID."""

    if base_id not in network.links:
        return base_id

    index = 2

    while f"{base_id}_{index}" in network.links:
        index += 1

    return f"{base_id}_{index}"


def _validate_interventions(
    network: MatsimNetwork,
    segment_index: StreetSegmentIndex,
    interventions: list[Intervention],
) -> None:
    """Check that selected interventions are valid and mutually compatible."""

    if len(interventions) != len(set(interventions)):
        raise ValueError("The same intervention was selected more than once.")

    for a, b in combinations(interventions, 2):
        if not interventions_compatible(a, b, network):
            raise ValueError(
                f"Incompatible interventions on segment {a.segment_id}: "
                f"{a.intervention_type.value} and {b.intervention_type.value}"
            )

    for intervention in interventions:
        segment = segment_index.get(intervention.segment_id)

        for link_id in intervention.target_link_ids:
            if link_id not in network.links:
                raise ValueError(f"Unknown target MATSim link: {link_id}")

            if link_id not in segment.link_ids:
                raise ValueError(
                    f"Target link {link_id} does not belong to "
                    f"StreetSegment {segment.id}."
                )

            if "car" not in network.links[link_id].allowed_modes:
                raise ValueError(f"Target link {link_id} does not allow car.")
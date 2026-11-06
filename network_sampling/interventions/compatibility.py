from __future__ import annotations

from network_sampling.interventions.intervention_types import Intervention, InterventionType
from network_sampling.network.matsim_network import MatsimNetwork


def interventions_compatible(
    a: Intervention,
    b: Intervention,
    network: MatsimNetwork,
) -> bool:
    """Return whether two interventions may coexist in one scenario."""

    if a.segment_id != b.segment_id:
        return True

    if (
        a.intervention_type != InterventionType.DEDICATED_BUS_LANE
        or b.intervention_type != InterventionType.DEDICATED_BUS_LANE
    ):
        return False

    link_a = network.links[a.target_link_ids[0]]
    link_b = network.links[b.target_link_ids[0]]

    return (
        link_a.from_node != link_b.from_node
        or link_a.to_node != link_b.to_node
    )
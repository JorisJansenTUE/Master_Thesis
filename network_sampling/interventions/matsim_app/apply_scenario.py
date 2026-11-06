from __future__ import annotations

from pathlib import Path

from network_sampling.interventions.intervention_types import Intervention
from network_sampling.interventions.matsim_app.apply_network import apply_interventions
from network_sampling.interventions.matsim_app.apply_schedule import apply_bus_lane_replacements
from network_sampling.network.matsim_network import (
    MatsimNetwork,
    write_matsim_network,
)
from network_sampling.network.street_segments import StreetSegmentIndex
from network_sampling.validation.connectivity import validate_connectivity


def apply_matsim_scenario(
    baseline_network: MatsimNetwork,
    baseline_schedule: Path,
    segment_index: StreetSegmentIndex,
    interventions: list[Intervention],
    output_network: Path,
    output_schedule: Path,
) -> None:
    """
    Apply interventions and write a complete MATSim network and schedule.

    Parameters
    ----------
    baseline_network
        Original MATSim network.

    baseline_schedule
        Original MATSim transit schedule.

    segment_index
        Physical StreetSegment index based on the baseline network.

    interventions
        Selected interventions.

    output_network
        Path for the modified MATSim network.

    output_schedule
        Path for the modified MATSim transit schedule.
    """

    modified_network, bus_link_replacements = apply_interventions(
        baseline_network,
        segment_index,
        interventions,
    )

    validate_connectivity(baseline_network, modified_network)

    # Only continue with a valid scenario
    write_matsim_network(modified_network, output_network)

    apply_bus_lane_replacements(
        baseline_schedule,
        output_schedule,
        bus_link_replacements,
    )
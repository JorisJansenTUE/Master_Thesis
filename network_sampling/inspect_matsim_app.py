from __future__ import annotations

from dataclasses import replace
import gzip
from math import isclose
from pathlib import Path

from lxml import etree

from network_sampling.interventions.eligibility import get_eligible_interventions
from network_sampling.interventions.intervention_types import (
    Intervention,
    InterventionType,
    RoadSpaceVariant,
)
from network_sampling.interventions.matsim_app.apply_scenario import (
    apply_matsim_scenario,
)
from network_sampling.network.matsim_network import MatsimNetwork, load_matsim_network
from network_sampling.network.street_segments import StreetSegmentIndex
from network_sampling.network.transit_usage import TransitUsageIndex

from network_sampling.validation.connectivity import check_connectivity
from preprocessing.utils  import DATA_DIR

# ---------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------

SCENARIO_DIR = Path(DATA_DIR / "NDA_ETH_Model" / "Final_Baseline")

NETWORK_FILE = Path(DATA_DIR / "NDA_ETH_Model" / "Final_Baseline" / "network_with_widths.xml.gz")
TRANSIT_SCHEDULE_FILE = Path(DATA_DIR / "NDA_ETH_Model" / "Final_Baseline" / "locarno_transit_schedule.xml.gz")

OUTPUT_DIR = Path(SCENARIO_DIR / "test_matsim_apply")
OUTPUT_NETWORK = OUTPUT_DIR / "network.xml.gz"
OUTPUT_SCHEDULE = OUTPUT_DIR / "transitSchedule.xml.gz"

WIDTH_ATTRIBUTE = "carriageway_width_m"

GENERAL_CAR_MODES = {"car", "car_passenger", "truck", "taxi"}


def collect_interventions(
    network: MatsimNetwork,
    segment_index: StreetSegmentIndex,
    transit_usage: TransitUsageIndex,
) -> list[Intervention]:
    """Return all eligible interventions."""

    interventions = []

    for segment in segment_index.all():
        interventions.extend(
            get_eligible_interventions(segment, network, transit_usage)
        )

    return interventions


def select_test_interventions(
    interventions: list[Intervention],
) -> list[Intervention]:
    """Select one useful test case for each intervention type."""

    selected: list[Intervention] = []
    used_segments: set[str] = set()

    preferences = {
        InterventionType.DESTINATION_ONLY: [(None, 0)],
        InterventionType.BICYCLE_STREET: [(None, 0)],
        InterventionType.PT_BIKE_STREET: [(None, 0)],
        InterventionType.PROTECTED_BIKE_LANE: [
            (RoadSpaceVariant.REMOVE_CAR, 2),
            (RoadSpaceVariant.REMOVE_ONE_LANE, 2),
            (RoadSpaceVariant.REMOVE_CAR, 1),
            (RoadSpaceVariant.REMOVE_ONE_LANE, 1),
            (RoadSpaceVariant.KEEP_CAR, 0),
        ],
        InterventionType.DEDICATED_BUS_LANE: [
            (RoadSpaceVariant.REMOVE_ONE_LANE, 1),
            (RoadSpaceVariant.REMOVE_CAR, 1),
            (RoadSpaceVariant.KEEP_CAR, 1),
        ],
    }

    for intervention_type, options in preferences.items():
        choice = None

        for variant, target_count in options:
            choice = next(
                (
                    intervention
                    for intervention in interventions
                    if intervention.intervention_type == intervention_type
                    and intervention.road_space_variant == variant
                    and len(intervention.target_link_ids) == target_count
                    and intervention.segment_id not in used_segments
                ),
                None,
            )

            if choice is not None:
                break

        if choice is None:
            raise ValueError(f"No test intervention found for {intervention_type.value}")

        selected.append(choice)
        used_segments.add(choice.segment_id)

    return selected


def check_application(
    baseline: MatsimNetwork,
    modified: MatsimNetwork,
    segment_index: StreetSegmentIndex,
    interventions: list[Intervention],
    output_schedule: Path,
) -> None:
    """Check that each selected intervention was applied correctly."""

    bus_route_links = _read_bus_route_links(output_schedule)

    for intervention in interventions:
        segment = segment_index.get(intervention.segment_id)
        baseline_car_links = [
            baseline.links[link_id]
            for link_id in segment.link_ids
            if "car" in baseline.links[link_id].allowed_modes
        ]

        if intervention.intervention_type == InterventionType.DESTINATION_ONLY:
            for link in baseline_car_links:
                assert modified.links[link.id].attributes.get("destination_road") == "Yes"

        elif intervention.intervention_type == InterventionType.BICYCLE_STREET:
            for link in baseline_car_links:
                assert modified.links[link.id].attributes.get("cyclestreet") == "yes"

        elif intervention.intervention_type == InterventionType.PT_BIKE_STREET:
            for link in baseline_car_links:
                modes = modified.links[link.id].allowed_modes
                assert not GENERAL_CAR_MODES & set(modes)
                assert "bike" in modes

        elif intervention.intervention_type == InterventionType.PROTECTED_BIKE_LANE:
            _check_pbl(baseline, modified, segment, intervention)

        elif intervention.intervention_type == InterventionType.DEDICATED_BUS_LANE:
            _check_bus_lane(baseline, modified, intervention, bus_route_links)

        print(f"PASS: {intervention.intervention_type.value} on {segment.id}")


def _check_pbl(
    baseline: MatsimNetwork,
    modified: MatsimNetwork,
    segment,
    intervention: Intervention,
) -> None:
    """Check PBL links and road-space changes."""

    for link_id in intervention.target_link_ids:
        old = baseline.links[link_id]
        new = modified.links[link_id]

        if intervention.road_space_variant == RoadSpaceVariant.REMOVE_ONE_LANE:
            assert new.permlanes == old.permlanes - 1
            assert isclose(
                new.capacity,
                old.capacity * new.permlanes / old.permlanes,
            )

        elif intervention.road_space_variant == RoadSpaceVariant.REMOVE_CAR:
            assert set(new.allowed_modes) == {"bike"}
            assert new.attributes.get("highway") == "bicycle_designated"
            assert new.attributes.get("protected_bike_lane") == "yes"

    pbl_directions = {
        (link.from_node, link.to_node)
        for link in modified.links.values()
        if link.attributes.get("protected_bike_lane") == "yes"
        and {link.from_node, link.to_node} == {segment.node_a, segment.node_b}
    }

    assert (segment.node_a, segment.node_b) in pbl_directions
    assert (segment.node_b, segment.node_a) in pbl_directions


def _check_bus_lane(
    baseline: MatsimNetwork,
    modified: MatsimNetwork,
    intervention: Intervention,
    bus_route_links: set[str],
) -> None:
    """Check the dedicated bus lane and its schedule replacement."""

    target_id = intervention.target_link_ids[0]
    spec_id = f"{target_id}_spec"

    old = baseline.links[target_id]
    new = modified.links[target_id]

    assert spec_id in modified.links

    spec = modified.links[spec_id]

    assert spec.from_node == old.from_node
    assert spec.to_node == old.to_node
    assert spec.permlanes == 1
    assert isclose(spec.capacity, old.capacity / old.permlanes)
    assert set(spec.allowed_modes) == {"bus", "pt"}
    assert "carriageway_width_m" not in spec.attributes
    assert "bus" not in new.allowed_modes

    if intervention.road_space_variant == RoadSpaceVariant.REMOVE_ONE_LANE:
        assert new.permlanes == old.permlanes - 1
        assert isclose(
            new.capacity,
            old.capacity * new.permlanes / old.permlanes,
        )

    elif intervention.road_space_variant == RoadSpaceVariant.REMOVE_CAR:
        assert not GENERAL_CAR_MODES & set(new.allowed_modes)

    elif intervention.road_space_variant == RoadSpaceVariant.KEEP_CAR:
        assert new.permlanes == old.permlanes
        assert new.capacity == old.capacity

    assert spec_id in bus_route_links
    assert target_id not in bus_route_links


def _read_bus_route_links(path: Path) -> set[str]:
    """Return all MATSim link IDs used by scheduled bus routes."""

    parser = etree.XMLParser(remove_blank_text=False)

    if path.suffix == ".gz":
        with gzip.open(path, "rb") as file:
            tree = etree.parse(file, parser)
    else:
        tree = etree.parse(str(path), parser)

    link_ids: set[str] = set()

    for transit_route in tree.iterfind(".//transitRoute"):
        if transit_route.findtext("transportMode") != "bus":
            continue

        route = transit_route.find("route")

        if route is not None:
            link_ids.update(
                link.get("refId")
                for link in route.iterfind("link")
                if link.get("refId")
            )

    return link_ids


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading baseline...")

    baseline = load_matsim_network(NETWORK_FILE)
    segment_index = StreetSegmentIndex.from_network(
        baseline,
        width_attribute=WIDTH_ATTRIBUTE,
    )

    transit_usage = TransitUsageIndex.from_schedule(
        TRANSIT_SCHEDULE_FILE,
        baseline,
        segment_index,
    )

    candidates = collect_interventions(
        baseline,
        segment_index,
        transit_usage,
    )

    interventions = select_test_interventions(candidates)

    print("\nSelected interventions")
    print("----------------------")

    for intervention in interventions:
        print(
            f"{intervention.intervention_type.value:<25} "
            f"segment={intervention.segment_id} "
            f"variant={intervention.road_space_variant.value if intervention.road_space_variant else '-'} "
            f"targets={intervention.target_link_ids or '-'}"
        )

    apply_matsim_scenario(
        baseline,
        TRANSIT_SCHEDULE_FILE,
        segment_index,
        interventions,
        OUTPUT_NETWORK,
        OUTPUT_SCHEDULE,
    )

    # Reloading also tests that the written network is valid for our loader.
    modified = load_matsim_network(OUTPUT_NETWORK)

    print("\nApplication checks")
    print("------------------")

    check_application(
        baseline,
        modified,
        segment_index,
        interventions,
        OUTPUT_SCHEDULE,
    )

    print("\nAll intervention checks passed.")
    results = check_connectivity(
    baseline,
    modified,
)
    # TEMPORARY TEST: deliberately break car connectivity
    test_node = next(
        link.from_node
        for link in modified.links.values()
        if "car" in link.allowed_modes
    )

    for link_id, link in list(modified.links.items()):
        if "car" in link.allowed_modes and (
            link.from_node == test_node or link.to_node == test_node
        ):
            modes = set(link.allowed_modes)
            modes.remove("car")
            modified.links[link_id] = replace(
                link,
                allowed_modes=frozenset(modes),
            )


    results = check_connectivity(baseline, modified)

    print("\nConnectivity")
    print("------------")
    for mode, connected in results.items():
        print(f"{mode:<8} {'PASS' if connected else 'FAIL'}")

    print(f"Network:  {OUTPUT_NETWORK.resolve()}")
    print(f"Schedule: {OUTPUT_SCHEDULE.resolve()}")


if __name__ == "__main__":
    main()
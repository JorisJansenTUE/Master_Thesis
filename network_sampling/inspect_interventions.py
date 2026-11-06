from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path

from network_sampling.interventions.eligibility import get_eligible_interventions
from network_sampling.interventions.intervention_types import Intervention
from network_sampling.network.matsim_network import MatsimNetwork, load_matsim_network
from network_sampling.network.street_segments import StreetSegmentIndex
from network_sampling.network.transit_usage import TransitUsageIndex

from preprocessing.utils import DATA_DIR

# ---------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------

NETWORK_FILE = Path(DATA_DIR / "NDA_ETH_Model" / "Final_Baseline" / "network_with_widths.xml.gz")

TRANSIT_SCHEDULE_FILE = Path(DATA_DIR / "NDA_ETH_Model" / "Final_Baseline" / "locarno_transit_schedule.xml.gz")

OUTPUT_FILE = Path(DATA_DIR / "NDA_ETH_Model" / "Final_Baseline" / "eligible_interventions.csv")

WIDTH_ATTRIBUTE = "carriageway_width_m"
MAX_EXAMPLES = 5


def print_header(title: str) -> None:
    print()
    print(title)
    print("-" * len(title))


def collect_interventions(
    network: MatsimNetwork,
    segment_index: StreetSegmentIndex,
    transit_usage: TransitUsageIndex,
) -> list[Intervention]:
    """
    Generate all eligible intervention configurations.

    Returns
    -------
    list[Intervention]
        Eligible interventions for all physical street segments.
    """

    interventions = []

    for segment in segment_index.all():
        interventions.extend(
            get_eligible_interventions(segment, network, transit_usage)
        )

    return interventions


def write_interventions_csv(
    path: Path,
    interventions: list[Intervention],
    network: MatsimNetwork,
    segment_index: StreetSegmentIndex,
    transit_usage: TransitUsageIndex,
) -> None:
    """Write all eligible intervention configurations to CSV."""

    fieldnames = [
        "segment_id",
        "intervention_type",
        "road_space_variant",
        "target_link_ids",
        "number_target_links",
        "osm_way_id",
        "length_m",
        "total_width_m",
        "target_links",
        "target_permlanes",
        "target_modes",
        "target_used_by_bus",
    ]

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for intervention in interventions:
            segment = segment_index.get(intervention.segment_id)

            target_links = [
                network.links[link_id]
                for link_id in intervention.target_link_ids
            ]

            writer.writerow(
                {
                    "segment_id": segment.id,
                    "intervention_type": intervention.intervention_type.value,
                    "road_space_variant": (
                        intervention.road_space_variant.value
                        if intervention.road_space_variant else ""
                    ),
                    "target_link_ids": ",".join(intervention.target_link_ids),
                    "number_target_links": len(intervention.target_link_ids),
                    "osm_way_id": segment.osm_way_id or "",
                    "length_m": segment.length_m,
                    "total_width_m": segment.total_width_m,
                    "target_links": "; ".join(
                        f"{link.from_node}->{link.to_node}"
                        for link in target_links
                    ),
                    "target_permlanes": ",".join(
                        str(link.permlanes)
                        for link in target_links
                    ),
                    "target_modes": "; ".join(
                        ",".join(sorted(link.allowed_modes))
                        for link in target_links
                    ),
                    "target_used_by_bus": ",".join(
                        str(transit_usage.uses_link(link.id))
                        for link in target_links
                    ),
                }
            )


def main() -> None:
    print_header("Loading baseline")

    network = load_matsim_network(NETWORK_FILE)

    segment_index = StreetSegmentIndex.from_network(
        network,
        width_attribute=WIDTH_ATTRIBUTE,
    )

    transit_usage = TransitUsageIndex.from_schedule(
        TRANSIT_SCHEDULE_FILE,
        network,
        segment_index,
    )

    print(f"MATSim nodes:        {len(network.nodes):,}")
    print(f"MATSim links:        {len(network.links):,}")
    print(f"StreetSegments:      {len(segment_index):,}")
    print(f"Scheduled bus links: {len(transit_usage):,}")

    # -----------------------------------------------------------------
    # Generate interventions
    # -----------------------------------------------------------------

    interventions = collect_interventions(
        network,
        segment_index,
        transit_usage,
    )

    interventions.sort(
        key=lambda intervention: (
            intervention.intervention_type.value,
            intervention.segment_id,
            intervention.road_space_variant.value
            if intervention.road_space_variant else "",
            intervention.target_link_ids,
        )
    )

    print_header("Eligible interventions")

    print(f"Total configurations:  {len(interventions):,}")
    print(f"Unique configurations: {len(set(interventions)):,}")

    duplicate_count = len(interventions) - len(set(interventions))

    if duplicate_count:
        print(f"WARNING: {duplicate_count:,} duplicate configurations.")

    # -----------------------------------------------------------------
    # Counts by intervention type
    # -----------------------------------------------------------------

    type_counts = Counter(
        intervention.intervention_type.value
        for intervention in interventions
    )

    segments_by_type: dict[str, set[str]] = defaultdict(set)

    for intervention in interventions:
        segments_by_type[intervention.intervention_type.value].add(
            intervention.segment_id
        )

    print_header("By intervention type")

    for intervention_type, count in sorted(type_counts.items()):
        print(
            f"{intervention_type:<25} "
            f"{count:>6,} configurations | "
            f"{len(segments_by_type[intervention_type]):>6,} segments"
        )

    # -----------------------------------------------------------------
    # Road-space variants
    # -----------------------------------------------------------------

    variant_counts = Counter(
        (
            intervention.intervention_type.value,
            intervention.road_space_variant.value
            if intervention.road_space_variant else "none",
        )
        for intervention in interventions
    )

    print_header("Road-space variants")

    for (intervention_type, variant), count in sorted(variant_counts.items()):
        print(f"{intervention_type:<25} {variant:<20} {count:>6,}")

    # -----------------------------------------------------------------
    # Road-space variants by number of target links
    # -----------------------------------------------------------------

    variant_target_counts = Counter(
        (
            intervention.intervention_type.value,
            intervention.road_space_variant.value
            if intervention.road_space_variant else "none",
            len(intervention.target_link_ids),
        )
        for intervention in interventions
    )

    print_header("Road-space variants by target links")

    for (intervention_type, variant, targets), count in sorted(
        variant_target_counts.items()
    ):
        print(
            f"{intervention_type:<25} "
            f"{variant:<20} "
            f"{targets} target link(s): {count:>6,}"
        )


    # -----------------------------------------------------------------
    # Number of targeted links
    # -----------------------------------------------------------------

    target_counts = Counter(
        (
            intervention.intervention_type.value,
            len(intervention.target_link_ids),
        )
        for intervention in interventions
    )

    print_header("Target links")

    for (intervention_type, number_targets), count in sorted(target_counts.items()):
        print(
            f"{intervention_type:<25} "
            f"{number_targets} target link(s): {count:>6,}"
        )

    # -----------------------------------------------------------------
    # Consistency checks
    # -----------------------------------------------------------------

    invalid_targets: list[tuple[Intervention, str]] = []

    for intervention in interventions:
        segment = segment_index.get(intervention.segment_id)

        for link_id in intervention.target_link_ids:
            if link_id not in network.links:
                invalid_targets.append(
                    (intervention, f"Unknown MATSim link {link_id}")
                )
                continue

            if link_id not in segment.link_ids:
                invalid_targets.append(
                    (intervention, f"Link {link_id} not part of segment")
                )
                continue

            if "car" not in network.links[link_id].allowed_modes:
                invalid_targets.append(
                    (intervention, f"Link {link_id} does not allow car")
                )

    print_header("Consistency checks")

    print(f"Invalid target links: {len(invalid_targets):,}")

    for intervention, reason in invalid_targets[:MAX_EXAMPLES]:
        print(
            f"  {intervention.intervention_type.value}: "
            f"segment={intervention.segment_id}, "
            f"targets={intervention.target_link_ids}, "
            f"reason={reason}"
        )

    # -----------------------------------------------------------------
    # Examples
    # -----------------------------------------------------------------

    print_header("Examples")

    interventions_by_type: dict[str, list[Intervention]] = defaultdict(list)

    for intervention in interventions:
        interventions_by_type[intervention.intervention_type.value].append(
            intervention
        )

    for intervention_type, examples in sorted(interventions_by_type.items()):
        print()
        print(intervention_type)

        for intervention in examples[:MAX_EXAMPLES]:
            segment = segment_index.get(intervention.segment_id)

            print(
                f"  segment={segment.id}, "
                f"width={segment.total_width_m}, "
                f"variant="
                f"{intervention.road_space_variant.value if intervention.road_space_variant else '-'}, "
                f"targets="
                f"{','.join(intervention.target_link_ids) if intervention.target_link_ids else '-'}"
            )

    # -----------------------------------------------------------------
    # Export
    # -----------------------------------------------------------------

    write_interventions_csv(
        OUTPUT_FILE,
        interventions,
        network,
        segment_index,
        transit_usage,
    )

    print_header("Output")
    print(f"Written to: {OUTPUT_FILE.resolve()}")


if __name__ == "__main__":
    main()
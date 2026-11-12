
from __future__ import annotations

from collections import Counter
from pathlib import Path

import networkx as nx

from network_sampling.interventions.eligibility import get_eligible_interventions
from network_sampling.interventions.matsim_app.apply_network import apply_interventions
from network_sampling.interventions.intervention_types import InterventionType
from network_sampling.network.matsim_network import load_matsim_network
from network_sampling.network.street_segments import StreetSegmentIndex
from network_sampling.network.transit_usage import TransitUsageIndex
from network_sampling.sampling.cluster import sample_clusters
from network_sampling.validation.connectivity import check_connectivity

from preprocessing.utils import DATA_DIR


# ---------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------

SCENARIO_DIR = Path(DATA_DIR) / "NDA_ETH_Model" / "Final_Baseline"
NETWORK_FILE = SCENARIO_DIR / "network_with_widths.xml.gz"
TRANSIT_SCHEDULE_FILE = SCENARIO_DIR / "locarno_transit_schedule.xml.gz"

WIDTH_ATTRIBUTE = "carriageway_width_m"

BUDGET_M = 5000.0
MAX_CLUSTER_LENGTH_M = 1000.0  # None = unlimited
PBL_WEIGHT_MULTIPLIER = 3
DESTINATION_WEIGHT_MULTIPLIER = 0.25
MAX_FAILED_ATTEMPTS = 60

SEEDS = range(50)


def build_physical_graph(index: StreetSegmentIndex) -> nx.MultiGraph:
    """Build one undirected edge per physical StreetSegment."""
    graph = nx.MultiGraph()

    for segment in index.all():
        graph.add_edge(
            segment.node_a,
            segment.node_b,
            key=segment.id,
            segment_id=segment.id,
            length_m=segment.length_m,
        )

    return graph


def collect_interventions(network, index, transit_usage):
    """Collect all locally eligible intervention configurations."""
    return [
        iv
        for segment in index.all()
        for iv in get_eligible_interventions(
            segment, network, transit_usage
        )
    ]


def physical_components(interventions, index) -> int:
    """Count connected components among selected physical segments."""
    graph = nx.Graph()

    for iv in interventions:
        segment = index.get(iv.segment_id)
        graph.add_edge(segment.node_a, segment.node_b)

    return nx.number_connected_components(graph) if graph.number_of_edges() else 0


def intervention_length(interventions, index) -> float:
    """Count physical length per intervention, including directional bus lanes."""
    return sum(
        index.get(iv.segment_id).length_m
        for iv in interventions
    )


def sample(seed, graph, index, network, eligible):
    return sample_clusters(
        graph=graph,
        index=index,
        network=network,
        candidates=eligible,
        budget_m=BUDGET_M,
        seed=seed,
        max_cluster_length_m=MAX_CLUSTER_LENGTH_M,
        pbl_weight_multiplier=PBL_WEIGHT_MULTIPLIER,
        destination_weight_multiplier=DESTINATION_WEIGHT_MULTIPLIER,
        max_failed_attempts=MAX_FAILED_ATTEMPTS,
    )


def main() -> None:
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

    graph = build_physical_graph(segment_index)
    eligible = collect_interventions(
        baseline, segment_index, transit_usage
    )

    print(f"StreetSegments:          {len(segment_index):,}")
    print(
        f"Physical graph:          {graph.number_of_nodes():,} nodes, "
        f"{graph.number_of_edges():,} edges"
    )
    print(f"Eligible configurations: {len(eligible):,}")
    print(f"Budget:                  {BUDGET_M:,.0f} m")
    print(
        "Maximum cluster length:  "
        + (
            f"{MAX_CLUSTER_LENGTH_M:,.0f} m"
            if MAX_CLUSTER_LENGTH_M is not None
            else "unlimited"
        )
    )
    print(f"PBL weight multiplier:   {PBL_WEIGHT_MULTIPLIER:.1f}")
    print(f"Destination weight multiplier: {DESTINATION_WEIGHT_MULTIPLIER:.1f}")

    type_counts = Counter()
    variant_counts = Counter()
    variant_by_type = Counter()
    type_lengths = Counter()

    used_budgets = []
    component_counts = []
    bike_component_counts = []

    valid = invalid = empty = 0

    # -----------------------------------------------------------------
    # Cluster sampling
    # -----------------------------------------------------------------

    print("\nCluster samples")
    print("---------------")

    for seed in SEEDS:
        selected = sample(
            seed, graph, segment_index, baseline, eligible
        )

        used = intervention_length(selected, segment_index)
        assert used <= BUDGET_M + 1e-6, (
            f"Budget exceeded for seed {seed}"
        )

        modified, bus_replacements = apply_interventions(
            baseline,
            segment_index,
            selected,
        )

        connectivity = check_connectivity(baseline, modified)

        failed_modes = [
            mode
            for mode, connected in connectivity.items()
            if not connected
        ]

        components = physical_components(selected, segment_index)

        cycling = [
            iv for iv in selected
            if iv.intervention_type in (
                InterventionType.BICYCLE_STREET,
                InterventionType.PROTECTED_BIKE_LANE,
            )
        ]

        bike_components = physical_components(
            cycling, segment_index
        )

        used_budgets.append(used)
        component_counts.append(components)
        bike_component_counts.append(bike_components)

        empty += not selected
        invalid += bool(failed_modes)
        valid += not failed_modes

        # Accumulate statistics across all samples.
        for iv in selected:
            kind = iv.intervention_type.value

            variant = (
                iv.road_space_variant.value
                if iv.road_space_variant
                else "none"
            )

            type_counts[kind] += 1
            variant_counts[variant] += 1
            variant_by_type[kind, variant] += 1

            type_lengths[kind] += (
                segment_index.get(iv.segment_id).length_m
            )

        status = (
            f"FAIL ({', '.join(failed_modes)})"
            if failed_modes
            else "PASS"
        )

        print(
            f"seed {seed:<3} | "
            f"{used:>7.0f} m | "
            f"{len(selected):>3} interventions | "
            f"{components:>2} components | "
            f"{len(bus_replacements):>2} bus links | "
            f"{status}"
        )

    # -----------------------------------------------------------------
    # Reproducibility
    # -----------------------------------------------------------------

    print("\nReproducibility")
    print("---------------")

    first_seed = next(iter(SEEDS), None)

    if first_seed is not None:
        sample_a = sample(
            first_seed, graph, segment_index, baseline, eligible
        )
        sample_b = sample(
            first_seed, graph, segment_index, baseline, eligible
        )

        assert sample_a == sample_b, (
            f"Seed {first_seed} is not reproducible"
        )

        print(
            f"PASS: seed {first_seed} produces the same sample twice"
        )

    # -----------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------

    n = len(used_budgets)

    print("\nCluster sampling summary")
    print("------------------------")
    print(f"Samples:              {n}")
    print(f"Connectivity valid:   {valid}")
    print(f"Connectivity invalid: {invalid}")
    print(f"Empty samples:        {empty}")

    if not n:
        return

    print(f"Mean budget used:     {sum(used_budgets) / n:,.1f} m")
    print(f"Minimum budget used:  {min(used_budgets):,.1f} m")
    print(f"Maximum budget used:  {max(used_budgets):,.1f} m")
    print(f"Mean components:      {sum(component_counts) / n:.1f}")
    print(
        f"Mean cycling components: "
        f"{sum(bike_component_counts) / n:.1f}"
    )

    # -----------------------------------------------------------------
    # Intervention counts
    # -----------------------------------------------------------------

    print("\nSelected interventions")
    print("----------------------")

    for kind, count in sorted(type_counts.items()):
        print(
            f"{kind:<25} "
            f"{count:>5,} total | "
            f"{count / n:>5.1f} per sample"
        )

    print("\nRoad-space variants")
    print("-------------------")

    for variant, count in sorted(variant_counts.items()):
        print(
            f"{variant:<25} "
            f"{count:>5,} total | "
            f"{count / n:>5.1f} per sample"
        )

    print("\nVariants by intervention type")
    print("-----------------------------")

    for (kind, variant), count in sorted(variant_by_type.items()):
        print(
            f"{kind:<25} "
            f"{variant:<18} "
            f"{count:>5,} total | "
            f"{count / n:>5.1f} per sample"
        )

    # -----------------------------------------------------------------
    # Intervention lengths
    # -----------------------------------------------------------------

    total_length = sum(type_lengths.values())

    assert abs(total_length - sum(used_budgets)) < 1e-4

    print("\nIntervention lengths")
    print("--------------------")

    for kind, metres in sorted(type_lengths.items()):
        share = (
            100 * metres / total_length
            if total_length else 0.0
        )

        print(
            f"{kind:<25} "
            f"{metres / n:>7.1f} m per scenario | "
            f"{share:>5.1f}%"
        )


if __name__ == "__main__":
    main()

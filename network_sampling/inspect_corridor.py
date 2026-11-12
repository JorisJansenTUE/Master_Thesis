
from __future__ import annotations

import random
from collections import Counter
from pathlib import Path

import networkx as nx

from network_sampling.interventions.eligibility import get_eligible_interventions
from network_sampling.interventions.matsim_app.apply_network import apply_interventions
from network_sampling.network.matsim_network import load_matsim_network
from network_sampling.network.street_segments import StreetSegmentIndex
from network_sampling.network.transit_usage import TransitUsageIndex
from network_sampling.sampling.corridor import sample_corridors
from network_sampling.sampling.corridor_mixed import sample_corridor_package
from network_sampling.sampling.random import intervention_length
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
MAX_CORRIDOR_LENGTH_M = None
INTRUSION_STRENGTH = 0.5
PBL_WEIGHT_MULTIPLIER = 3.0
SEEDS = range(50)


def collect_interventions(network, segment_index, transit_usage):
    """Return all locally eligible configurations (no bike-adjacency filter)."""
    interventions = []

    for segment in segment_index.all():
        interventions.extend(
            get_eligible_interventions(segment, network, transit_usage)
        )

    return interventions


def build_physical_graph(segment_index) -> nx.MultiGraph:
    """One undirected edge per baseline physical StreetSegment."""
    graph = nx.MultiGraph()

    for segment in segment_index.all():
        graph.add_edge(
            segment.node_a,
            segment.node_b,
            key=segment.id,
            segment_id=segment.id,
            length_m=segment.length_m,
        )

    return graph


def count_physical_components(selected, segment_index) -> int:
    """Count components among selected segments (ignoring baseline edges)."""
    graph = nx.Graph()

    for intervention in selected:
        segment = segment_index.get(intervention.segment_id)
        graph.add_edge(segment.node_a, segment.node_b)

    return nx.number_connected_components(graph) if graph.number_of_edges() else 0


def diagnose_connectivity(selected, baseline, segment_index, failed_modes):
    """Find the first sampled intervention after which a mode is disconnected."""
    previous = {mode: True for mode in failed_modes}
    found = set()

    for i in range(1, len(selected) + 1):
        modified, _ = apply_interventions(
            baseline,
            segment_index,
            selected[:i],
        )
        results = check_connectivity(baseline, modified)
        intervention = selected[i - 1]

        for mode in failed_modes:
            if mode in found:
                continue

            if previous[mode] and not results[mode]:
                segment = segment_index.get(intervention.segment_id)
                variant = (
                    intervention.road_space_variant.value
                    if intervention.road_space_variant else "-"
                )

                print(
                    f"    {mode} first breaks at intervention {i}: "
                    f"{intervention.intervention_type.value}, "
                    f"segment={segment.id}, variant={variant}, "
                    f"targets={intervention.target_link_ids or '-'}"
                )
                print(f"      StreetSegment links: {segment.link_ids}")

                for link_id in segment.link_ids:
                    link = baseline.links[link_id]
                    print(
                        f"        {link.id}: "
                        f"{link.from_node} -> {link.to_node}, "
                        f"modes={sorted(link.allowed_modes)}, "
                        f"lanes={link.permlanes}"
                    )

                node_a = baseline.nodes[segment.node_a]
                node_b = baseline.nodes[segment.node_b]
                print(
                    f"      Nodes: "
                    f"{segment.node_a} ({node_a.x:.2f}, {node_a.y:.2f}) <-> "
                    f"{segment.node_b} ({node_b.x:.2f}, {node_b.y:.2f})"
                )
                found.add(mode)

            previous[mode] = results[mode]

        if len(found) == len(failed_modes):
            break


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
    physical_graph = build_physical_graph(segment_index)
    eligible = collect_interventions(baseline, segment_index, transit_usage)

    print(f"StreetSegments:          {len(segment_index):,}")
    print(
        f"Physical graph:          {physical_graph.number_of_nodes():,} nodes, "
        f"{physical_graph.number_of_edges():,} edges"
    )
    print(f"Eligible configurations: {len(eligible):,}")
    print(f"Budget:                  {BUDGET_M:,.0f} m")

    max_length_label = (
        f"{MAX_CORRIDOR_LENGTH_M:,.0f} m"
        if MAX_CORRIDOR_LENGTH_M is not None
        else "unlimited"
    )
    print(f"Maximum corridor length: {max_length_label}")

    type_lengths = Counter()
    total_type_counts = Counter()
    total_variant_counts = Counter()
    valid_samples = 0
    invalid_samples = 0
    empty_samples = 0
    used_budgets = []
    component_counts = []

    print("\nCorridor samples")
    print("----------------")

    for seed in SEEDS:
        selected = sample_corridor_package(
            graph=physical_graph,
            index=segment_index,
            network=baseline,
            candidates=eligible,
            transit_usage=transit_usage,
            budget_m=5000.0,
            seed=seed,
            destination_share=0.10,
            bus_share=0.15,
            max_corridor_length_m=None,
            max_bus_corridor_length_m=1500.0,
            intrusion_strength=0.5,
            pbl_weight_multiplier=3.0,
        )
        used_budget = intervention_length(selected, segment_index)
        assert used_budget <= BUDGET_M + 1e-6, "Corridor budget exceeded"

        used_budgets.append(used_budget)
        empty_samples += not selected

        components = count_physical_components(selected, segment_index)
        component_counts.append(components)

        for iv in selected:
            type_lengths[iv.intervention_type.value] += (
                segment_index.get(iv.segment_id).length_m
            )

        modified, bus_replacements = apply_interventions(
            baseline,
            segment_index,
            selected,
        )
        connectivity = check_connectivity(baseline, modified)
        failed_modes = [
            mode for mode, connected in connectivity.items() if not connected
        ]

        if failed_modes:
            status = f"FAIL ({', '.join(failed_modes)})"
            invalid_samples += 1
        else:
            status = "PASS (EMPTY)" if not selected else "PASS"
            valid_samples += 1

        total_type_counts.update(
            iv.intervention_type.value for iv in selected
        )
        total_variant_counts.update(
            iv.road_space_variant.value if iv.road_space_variant else "none"
            for iv in selected
        )

        print(
            f"seed {seed:<3} | "
            f"{used_budget:>7.0f} m | "
            f"{len(selected):>3} interventions | "
            f"{components:>2} components | "
            f"{len(bus_replacements):>2} bus links | "
            f"{status}"
        )

        # Enable to investigate invalid final networks.
        # Note that a partial package might disconnect a mode temporarily
        # even if a later intervention reconnects it.
        # if failed_modes:
        #     diagnose_connectivity(
        #         selected, baseline, segment_index, failed_modes
        #     )

    # -----------------------------------------------------------------
    # Reproducibility
    # -----------------------------------------------------------------

    first_seed = next(iter(SEEDS), None)
    if first_seed is not None:
        sample_a = sample_corridors(
            graph=physical_graph,
            index=segment_index,
            network=baseline,
            candidates=eligible,
            budget_m=BUDGET_M,
            max_corridor_length_m=MAX_CORRIDOR_LENGTH_M,
            intrusion_strength=INTRUSION_STRENGTH,
            pbl_weight_multiplier=PBL_WEIGHT_MULTIPLIER,
            rng=random.Random(first_seed),
        )
        sample_b = sample_corridors(
            graph=physical_graph,
            index=segment_index,
            network=baseline,
            candidates=eligible,
            budget_m=BUDGET_M,
            max_corridor_length_m=MAX_CORRIDOR_LENGTH_M,
            intrusion_strength=INTRUSION_STRENGTH,
            pbl_weight_multiplier=PBL_WEIGHT_MULTIPLIER,
            rng=random.Random(first_seed),
        )

        assert sample_a == sample_b

        print("\nReproducibility")
        print("---------------")
        print(f"PASS: seed {first_seed} produces the same sample twice")

    # -----------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------

    number_samples = len(used_budgets)

    print("\nCorridor sampling summary")
    print("-------------------------")
    print(f"Samples:              {number_samples:,}")
    print(f"Connectivity valid:   {valid_samples:,}")
    print(f"Connectivity invalid: {invalid_samples:,}")
    print(f"Empty samples:        {empty_samples:,}")

    if used_budgets:
        print(f"Mean budget used:     {sum(used_budgets) / number_samples:,.1f} m")
        print(f"Minimum budget used:  {min(used_budgets):,.1f} m")
        print(f"Maximum budget used:  {max(used_budgets):,.1f} m")
        print(f"Mean components:      {sum(component_counts) / number_samples:.1f}")

    print("\nSelected interventions")
    print("----------------------")

    for intervention_type, count in sorted(total_type_counts.items()):
        print(
            f"{intervention_type:<25} "
            f"{count:>5,} total | "
            f"{count / number_samples:>5.1f} per sample"
        )

    print("\nRoad-space variants")
    print("-------------------")

    for variant, count in sorted(total_variant_counts.items()):
        print(
            f"{variant:<25} "
            f"{count:>5,} total | "
            f"{count / number_samples:>5.1f} per sample"
        )


    # At the end:
    print("\nIntervention lengths")
    print("--------------------")

    for kind, length in sorted(type_lengths.items()):
        print(
            f"{kind:<25} "
            f"{length / len(SEEDS):>7.1f} m per scenario | "
            f"{100 * length / sum(type_lengths.values()):>5.1f}%"
        )


if __name__ == "__main__":
    main()

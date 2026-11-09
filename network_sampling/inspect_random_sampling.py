from __future__ import annotations

from collections import Counter
from pathlib import Path

from network_sampling.interventions.eligibility import get_eligible_interventions
from network_sampling.interventions.matsim_app.apply_network import apply_interventions
from network_sampling.network.matsim_network import load_matsim_network
from network_sampling.network.street_segments import StreetSegmentIndex
from network_sampling.network.transit_usage import TransitUsageIndex
from network_sampling.sampling.random import intervention_length, sample_random
from network_sampling.validation.connectivity import check_connectivity

from preprocessing.utils import DATA_DIR


# ---------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------

SCENARIO_DIR = Path(DATA_DIR / "NDA_ETH_Model" / "Final_Baseline")

NETWORK_FILE = Path(DATA_DIR / "NDA_ETH_Model" / "Final_Baseline" / "network_with_widths.xml.gz")
TRANSIT_SCHEDULE_FILE = Path(DATA_DIR / "NDA_ETH_Model" / "Final_Baseline" / "locarno_transit_schedule.xml.gz")
    

WIDTH_ATTRIBUTE = "carriageway_width_m"

BUDGET_M = 5000.0
SEEDS = range(0, 50)


def collect_interventions(network, segment_index, transit_usage):
    """Return all eligible intervention configurations."""

    interventions = []

    for segment in segment_index.all():
        interventions.extend(
            get_eligible_interventions(segment, network, transit_usage)
        )

    return interventions

def diagnose_connectivity(
    selected,
    baseline,
    segment_index,
    failed_modes: list[str],
) -> None:
    """Show where each failed modal network first becomes disconnected."""

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

                print(
                    f"    {mode} breaks at intervention {i}: "
                    f"{intervention.intervention_type.value}, "
                    f"segment={segment.id}, "
                    f"variant="
                    f"{intervention.road_space_variant.value if intervention.road_space_variant else '-'}, "
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

    eligible = collect_interventions(
        baseline,
        segment_index,
        transit_usage,
    )

    print(f"StreetSegments:          {len(segment_index):,}")
    print(f"Eligible configurations: {len(eligible):,}")
    print(f"Budget:                  {BUDGET_M:,.0f} m")

    total_type_counts = Counter()
    valid_samples = 0
    invalid_samples = 0
    used_budgets = []

    print("\nRandom samples")
    print("--------------")

    for seed in SEEDS:
        selected = sample_random(
            eligible,
            segment_index,
            baseline,
            budget_m=BUDGET_M,
            seed=seed,
        )

        used_budget = intervention_length(selected, segment_index)
        used_budgets.append(used_budget)

        modified, bus_replacements = apply_interventions(
            baseline,
            segment_index,
            selected,
        )

        connectivity = check_connectivity(
            baseline,
            modified,
        )

        failed_modes = [
            mode
            for mode, connected in connectivity.items()
            if not connected
        ]

        if failed_modes:
            status = f"FAIL ({', '.join(failed_modes)})"
            invalid_samples += 1
        else:
            status = "PASS"
            valid_samples += 1

        type_counts = Counter(
            intervention.intervention_type.value
            for intervention in selected
        )
        total_type_counts.update(type_counts)

        double_bus_segments = _count_double_bus_segments(selected)

        print(
            f"seed {seed:<3} | "
            f"{used_budget:>7.0f} m | "
            f"{len(selected):>3} interventions | "
            f"{len(bus_replacements):>2} bus lanes | "
            f"{double_bus_segments:>2} double bus | "
            f"{status}"
        )

        # if failed_modes:
        #     diagnose_connectivity(
        #         selected,
        #         baseline,
        #         segment_index,
        #         failed_modes,
        #     )

    # -----------------------------------------------------------------
    # Reproducibility
    # -----------------------------------------------------------------

    first_seed = next(iter(SEEDS))

    sample_a = sample_random(
        eligible,
        segment_index,
        baseline,
        budget_m=BUDGET_M,
        seed=first_seed,
    )

    sample_b = sample_random(
        eligible,
        segment_index,
        baseline,
        budget_m=BUDGET_M,
        seed=first_seed,
    )

    assert sample_a == sample_b

    print("\nReproducibility")
    print("---------------")
    print(f"PASS: seed {first_seed} produces the same sample twice")

    # -----------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------

    number_samples = len(used_budgets)

    print("\nRandom sampling summary")
    print("-----------------------")
    print(f"Samples:              {number_samples:,}")
    print(f"Connectivity valid:   {valid_samples:,}")
    print(f"Connectivity invalid: {invalid_samples:,}")

    if used_budgets:
        print(f"Mean budget used:     {sum(used_budgets) / number_samples:,.1f} m")
        print(f"Minimum budget used:  {min(used_budgets):,.1f} m")
        print(f"Maximum budget used:  {max(used_budgets):,.1f} m")

    print("\nSelected interventions")
    print("----------------------")

    for intervention_type, count in sorted(total_type_counts.items()):
        mean = count / number_samples

        print(
            f"{intervention_type:<25} "
            f"{count:>5,} total | "
            f"{mean:>5.1f} per sample"
        )


def _count_double_bus_segments(interventions) -> int:
    """Count segments receiving dedicated bus lanes in both directions."""

    bus_counts = Counter(
        intervention.segment_id
        for intervention in interventions
        if intervention.intervention_type.value == "dedicated_bus_lane"
    )

    return sum(count > 1 for count in bus_counts.values())


if __name__ == "__main__":
    main()
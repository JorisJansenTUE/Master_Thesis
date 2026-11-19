"""Inspect MATSim car links, matching linkstats, and directed connectivity.

Run from anywhere:
    python PROJECT_ROOT/accessibility/inputs/inspect_network.py

Only requires networkx (and the Step 1 linkstats.py beside these files).
"""

from __future__ import annotations

from pathlib import Path
from statistics import median
from time import perf_counter

import networkx as nx

from .linkstats import LinkTravelTimes
from .network import CarNetwork
from preprocessing.utils import PROJECT_ROOT

# -----------------------------------------------------------------------------
# LOCAL SETTINGS
# -----------------------------------------------------------------------------

RUN_DIRECTORY = PROJECT_ROOT / "data" / "Simulated_Runs" / "baseline_00000"
NETWORK_FILE = RUN_DIRECTORY / "network.xml.gz"
LINKSTATS_FILE = RUN_DIRECTORY / "200.linkstats.txt.gz"

HOUR_START = 8
STATISTIC = "avg"
AVERAGED_ITERATIONS = 8  # provenance metadata; no averaging is done here
MISSING_TIME_POLICY = "error"  # can explicitly select "free_flow"
# -----------------------------------------------------------------------------


def _percentile(values: list[float], percentile: float) -> float:
    """Small diagnostic helper; no NumPy needed."""
    ordered = sorted(values)
    index = (len(ordered) - 1) * percentile / 100
    lower = int(index)
    fraction = index - lower
    return ordered[lower] * (1 - fraction) + ordered[min(lower + 1, len(ordered) - 1)] * fraction


def main() -> None:
    for source in (NETWORK_FILE, LINKSTATS_FILE):
        if not source.is_file():
            raise FileNotFoundError(f"Input not found: {source}; adjust paths at the top")

    begin = perf_counter()
    snapshot = LinkTravelTimes.from_linkstats(
        LINKSTATS_FILE,
        hour_start=HOUR_START,
        statistic=STATISTIC,
        averaged_iterations=AVERAGED_ITERATIONS,
    )
    after_stats = perf_counter()
    car = CarNetwork.from_matsim_network(
        NETWORK_FILE, snapshot, missing_time_policy=MISSING_TIME_POLICY,
    )
    after_graph = perf_counter()

    G = car.graph

    # ------------------------------------------------------------
    # Inspect disconnected car components
    # ------------------------------------------------------------

    components = sorted(
        nx.weakly_connected_components(G),
        key=len,
        reverse=True,
    )

    main_component = components[0]

    print("\nDisconnected car components:")

    for i, component in enumerate(components[1:], start=1):
        subgraph = G.subgraph(component)

        print(f"\nComponent {i}:")
        print(f"  Nodes: {len(component)}")
        print(f"  Links: {subgraph.number_of_edges()}")

        for node_id in sorted(component):
            node = G.nodes[node_id]
            print(
                f"  Node {node_id}: "
                f"x={node['x']:.2f}, y={node['y']:.2f}"
            )

        for u, v, link_id, data in subgraph.edges(
            keys=True, data=True
        ):
            print(
                f"  Link {link_id}: {u} -> {v}, "
                f"length={data['length_m']:.1f} m"
            )

    report = car.report
    lengths = [d["length_m"] for _, _, _, d in G.edges(keys=True, data=True)]
    times = [d["travel_time_s"] for _, _, _, d in G.edges(keys=True, data=True)]
    weak = [len(c) for c in nx.weakly_connected_components(G)]
    strong = [len(c) for c in nx.strongly_connected_components(G)]

    # Illustrative smoke check: routing will be implemented separately.
    origin_node = next(iter(G.nodes))
    start_route = perf_counter()
    reached = nx.single_source_dijkstra_path_length(
        G, origin_node, cutoff=30 * 60, weight="travel_time_s",
    )
    end_route = perf_counter()

    print("=" * 68)
    print("MATSim car network — custom XML + NetworkX")
    print("=" * 68)
    print(f"Network:                 {NETWORK_FILE}")
    print(f"Linkstats:               {LINKSTATS_FILE}")
    print(f"Travel-time column:      {snapshot.travel_time_column}")
    print(f"Averaged iterations:     {snapshot.averaged_iterations} (metadata only)")
    print(f"Missing time policy:     {MISSING_TIME_POLICY}")
    print("\nMode filtering / network:")
    print(f"  MATSim nodes:          {report.total_network_nodes:>8,}")
    print(f"  MATSim links:          {report.total_network_links:>8,}")
    print(f"  Car links:             {report.car_links:>8,}")
    print(f"  Non-car links:         {report.non_car_links:>8,}")
    print(f"  Links without modes:   {report.links_without_modes:>8,}")
    print(f"  Car graph nodes:       {car.num_nodes:>8,}")
    print(f"  Car graph edges:       {car.num_links:>8,}")
    print(f"  Car directed length:   {sum(lengths)/1000:>8,.2f} km")

    print("\nTravel-time matching:")
    print(f"  Matched:               {report.car_links_matched:>8,}")
    print(f"  Missing IDs:           {report.car_links_missing_time:>8,}")
    print(f"  Invalid values:        {report.car_links_invalid_time:>8,}")
    print(f"  Free-flow fallbacks:   {report.car_links_free_flow_fallback:>8,}")
    print(f"  Stats IDs outside XML: {report.linkstats_ids_not_in_network:>8,}")
    if report.missing_examples:
        print(f"  Missing examples:      {report.missing_examples}")
    if report.invalid_examples:
        print(f"  Invalid examples:      {report.invalid_examples}")

    print("\nTravel times / link lengths:")
    print(f"  Link length median/p95: {median(lengths):.2f} / {_percentile(lengths,95):.2f} m")
    print(f"  Travel time median/p95:{median(times):.2f} / {_percentile(times,95):.2f} s")
    print(f"  Maximum travel time:   {max(times):.2f} s")
    for threshold in (5, 10, 20):
        print(f"  Links < {threshold} m:         {sum(length < threshold for length in lengths):>8,}")

    print("\nDirected connectivity:")
    print(f"  Weak components:       {len(weak):>8,}; largest {max(weak):,} nodes")
    print(f"  Strong components:     {len(strong):>8,}; largest {max(strong):,} nodes")
    print("\nNetworkX 30-minute Dijkstra smoke check:")
    print(f"  Origin:                {origin_node}")
    print(f"  Reachable nodes:       {len(reached):>8,}")
    print("\nTimings (local environment):")
    print(f"  Load linkstats:        {after_stats-begin:.3f} s")
    print(f"  Load car graph:        {after_graph-after_stats:.3f} s")
    print(f"  Dijkstra one origin:   {end_route-start_route:.3f} s")
    print("\nReady for Step 3: spatial origins and STATENT destinations.")


if __name__ == "__main__":
    main()

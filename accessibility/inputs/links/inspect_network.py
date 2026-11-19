"""Inspect car and bike accessibility networks from one MATSim scenario.

Run: python accessibility/inputs/inspect_network.py
Edit paths and bicycle speed at the top. No test suite required.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from statistics import median
from time import perf_counter

import networkx as nx

from .linkstats import LinkTravelTimes
from .network import BikeNetwork, CarNetwork, load_road_networks
from preprocessing.utils import PROJECT_ROOT

RUN_DIRECTORY = PROJECT_ROOT / "data" / "Simulated_Runs" / "baseline_00000"
NETWORK_FILE = RUN_DIRECTORY / "network.xml.gz"
LINKSTATS_FILE = RUN_DIRECTORY / "200.linkstats.txt.gz"
HOUR_START = 8
STATISTIC = "avg"
AVERAGED_ITERATIONS = 8  # provenance metadata, not recalculated
MISSING_TIME_POLICY = "error"
BIKE_SPEED_M_S = 4.17  # STILL PROVISIONAL: verify the actual eQasim bike speed


def _describe(name: str, network: CarNetwork | BikeNetwork, *, cutoff: float = 1800) -> None:
    graph = network.graph
    times = [data["travel_time_s"] for _, _, _, data in graph.edges(keys=True, data=True)]
    costs = [data["routing_cost_s"] for _, _, _, data in graph.edges(keys=True, data=True)]
    lengths = [data["length_m"] for _, _, _, data in graph.edges(keys=True, data=True)]
    weak = sorted((len(c) for c in nx.weakly_connected_components(graph)), reverse=True)
    strong = sorted((len(c) for c in nx.strongly_connected_components(graph)), reverse=True)
    start_node = next(iter(graph))
    start_time = perf_counter()
    reached = nx.single_source_dijkstra_path_length(
        graph, start_node, weight="routing_cost_s", cutoff=cutoff
    )
    route_elapsed = perf_counter() - start_time
    r = network.report
    print(f"\n{name}:")
    print(f"  MATSim nodes / links:     {r.total_network_nodes:,} / {r.total_network_links:,}")
    print(f"  Eligible directed links: {graph.number_of_edges():,}")
    print(f"  Graph nodes:             {graph.number_of_nodes():,}")
    print(f"  Directed length:         {sum(lengths) / 1000:.2f} km")
    print(f"  Links < 10 / 20 m:       {sum(x < 10 for x in lengths):,} / {sum(x < 20 for x in lengths):,}")
    print(f"  Weak components:         {len(weak)}; largest {weak[0]:,}")
    print(f"  Strong components:       {len(strong)}; largest {strong[0]:,}")
    print(f"  TT median/max:           {median(times):.2f} / {max(times):.2f} s")
    print(f"  Cost median/max:         {median(costs):.2f} / {max(costs):.2f} s")
    print(f"  30-min Dijkstra:         {len(reached):,} nodes in {route_elapsed:.4f} s")
    if name == "Car":
        print(f"  Linkstats matched:       {r.links_matched:,}")
        print(f"  Missing/invalid/fallback:{r.links_missing_time} / {r.links_invalid_time} / {r.links_free_flow_fallback}")
        print(f"  Stats outside network:   {r.linkstats_ids_not_in_network:,}")
    else:
        factors = Counter(data["safety_multiplier"] for _, _, _, data in graph.edges(keys=True, data=True))
        categories = Counter(data["safety_category"] for _, _, _, data in graph.edges(keys=True, data=True))
        missing_types = sum(data["link_type"] == "unknown"
                            for _, _, _, data in graph.edges(keys=True, data=True))
        print(f"  VTT categories:         {dict(categories)}")
        print(f"  Safety multipliers:     {dict(sorted(factors.items()))}")
        print(f"  Missing OSM highway:    {missing_types:,} (osm:way:highway)")
        print("  VTT rule: supplied eQasim Java values, precomputed per link")
        print("  NOTE: bicycle SPEED is still provisional")


def main() -> None:
    start = perf_counter()
    times = LinkTravelTimes.from_linkstats(
        LINKSTATS_FILE, hour_start=HOUR_START, statistic=STATISTIC,
        averaged_iterations=AVERAGED_ITERATIONS,
    )
    loaded = perf_counter()
    car, bike = load_road_networks(
        NETWORK_FILE, times, bike_speed_m_s=BIKE_SPEED_M_S,
        missing_time_policy=MISSING_TIME_POLICY,
    )
    graphs_ready = perf_counter()

    print("=" * 70)
    print("Accessibility road networks — car and bicycle")
    print("=" * 70)
    print(f"Network:      {NETWORK_FILE}")
    print(f"Linkstats:    {LINKSTATS_FILE}")
    print(f"Column:       {times.travel_time_column}")
    print(f"Iterations:   {times.averaged_iterations} (metadata)")
    _describe("Car", car)
    _describe("Bike", bike)
    print("\nLoading times:")
    print(f"  Linkstats: {loaded-start:.3f} s")
    print(f"  Car + bike (single XML parse): {graphs_ready-loaded:.3f} s")


if __name__ == "__main__":
    main()

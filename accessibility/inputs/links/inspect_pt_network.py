"""Inspect the route-aware, timetable-independent PT accessibility graph.

Run: python accessibility/inputs/inspect_pt_network.py
Requires linkstats.py and schedule.py next to network.py.
"""
from __future__ import annotations

from pathlib import Path
from statistics import median
from time import perf_counter

import networkx as nx

from .linkstats import LinkTravelTimes
from .network import PTNetwork
from .schedule import TransitSchedule

from preprocessing.utils import PROJECT_ROOT

RUN_DIRECTORY = PROJECT_ROOT / "data" / "Simulated_Runs" / "baseline_00000"
SCHEDULE_FILE = RUN_DIRECTORY / "transit_schedule.xml.gz"  # edit if needed
LINKSTATS_FILE = RUN_DIRECTORY / "200.linkstats.txt.gz"
HOUR_START = 8
STATISTIC = "avg"
AVERAGED_ITERATIONS = 8
TRANSFER_PENALTY_S = 300  # assumption for route changes, in addition to schedule minimum


def main() -> None:
    begin = perf_counter()
    schedule = TransitSchedule.from_matsim_schedule(SCHEDULE_FILE)
    times = LinkTravelTimes.from_linkstats(
        LINKSTATS_FILE, hour_start=HOUR_START, statistic=STATISTIC,
        averaged_iterations=AVERAGED_ITERATIONS,
    )
    read_done = perf_counter()
    pt = PTNetwork.from_schedule(schedule, times,
                                 transfer_penalty_s=TRANSFER_PENALTY_S)
    graph_done = perf_counter()
    g, report = pt.graph, pt.report
    rides = [(u, v, d) for u, v, d in g.edges(data=True) if d["kind"] == "ride"]
    slowest_bus = sorted(
        ((d["travel_time_s"], d["scheduled_time_s"], u, v)
         for u, v, d in rides if d["transport_mode"] == "bus"),
        reverse=True,
    )[:8]
    bus_times = [d["travel_time_s"] for _, _, d in rides if d["transport_mode"] == "bus"]
    weak = [len(c) for c in nx.weakly_connected_components(g)]
    src = next(iter(g))
    rt_start = perf_counter()
    reached = nx.single_source_dijkstra_path_length(
        g, src, cutoff=30*60, weight="routing_cost_s"
    )
    rt_end = perf_counter()
    print("=" * 70)
    print("PT accessibility network — bus linkstats (special links at free flow)")
    print("=" * 70)
    print(f"Schedule:                {SCHEDULE_FILE}")
    print(f"Linkstats:               {LINKSTATS_FILE}")
    print(f"Time column:             {times.travel_time_column}")
    print(f"Transfer penalty:        {TRANSFER_PENALTY_S:.0f} s")
    print(f"Route states:            {report.route_stop_states:,}")
    print(f"Edges total:             {g.number_of_edges():,}")
    print(f"Ride edges:              {report.ride_edges:,} {report.ride_edges_by_mode}")
    print(f"Transfer edges:          {report.constructed_transfer_edges:,}")
    print(f"Schedule transfer pairs: {report.scheduled_transfer_relations:,}")
    print(f"Weak components:         {len(weak)}; largest {max(weak):,}")
    print(f"Unique bus links used:   {report.bus_link_references:,}")
    print(f"Special bus links at FF: {report.spec_links_using_free_flow:,}")
    print(f"Artificial zero TT links:{len(report.zero_time_artificial_pt_links):,} "
          f"{report.zero_time_artificial_pt_links}")
    print(f"Same-link stop segments: {report.same_link_bus_segments:,}")
    print(f"Zero-time bus segments:  {report.zero_time_bus_segments:,}")
    print(f"Bus slower than schedule:{report.bus_segments_slower_than_schedule:,}")
    print(f"Worst bus/schedule ratio:{report.max_bus_to_schedule_ratio:.2f}x")
    print(f"Bus segment median TT:   {median(bus_times):.2f} s")
    print(f"PT 30-min Dijkstra:      {len(reached):,} states / {rt_end-rt_start:.4f} s")
    print("\nEight longest bus segments (adjusted bus cost vs schedule seconds):")
    for observed, scheduled, u, v in slowest_bus:
        print(f"  {observed:8.1f} / {scheduled:6.1f}  {u[:2]}  {u[2]} -> {v[2]}")
    print("\nTiming:")
    print(f"  Inputs: {read_done-begin:.3f} s; PT graph: {graph_done-read_done:.3f} s")
    print("\nLimitations: ignores frequencies/departures; ordinary bus linkstats may")
    print("include stop/holding behaviour; *_spec links use free flow. Artificial")
    print("pt_* links with explicit zero and same-link segments remain at zero.")
    print("Walking access/egress is not yet implemented (future Step 4).")


if __name__ == "__main__":
    main()

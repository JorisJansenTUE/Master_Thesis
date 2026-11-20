"""Inspect car, bike and PT shortest-path costs without computing metrics.

Run from the project root:
    python -m accessibility.routing.inspect_routing

Uses dynamic in-memory snapping for car/bike (apply the previous dynamic
snapping update first). PT uses the existing static walking connector CSVs.
All paths/settings are at the top; no scenario connector CSVs are written.
"""
from __future__ import annotations

from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from preprocessing.utils import PROJECT_ROOT
from accessibility.inputs.links.linkstats import LinkTravelTimes
from accessibility.inputs.links.network import PTNetwork, load_road_networks
from accessibility.inputs.links.schedule import TransitSchedule
from accessibility.inputs.grid.snapping import ScenarioRoadSnapper
from .road import RoadRouter
from .pt import PTRouter


# Change to cluster_00051 (or another simulated run) to inspect a scenario.
RUN_DIR = PROJECT_ROOT / "data" / "Simulated_Runs_v2" / "baseline_00000"
NETWORK_FILE = RUN_DIR / "network.xml.gz"
SCHEDULE_FILE = RUN_DIR / "transit_schedule.xml.gz"
LINKSTATS_FILE = RUN_DIR / "targets" / "linkstats.txt.gz"
SPATIAL_DIR = PROJECT_ROOT / "data" / "accessibility" /  "spatial_baseline"

HOUR_START = 8
BIKE_SPEED_M_S = 4.17  # Provisional: verify the actual MATSim/eQasim bike speed
PT_TRANSFER_PENALTY_S = 300.0
ORIGIN_SNAP_M = 500.0
JOB_SNAP_M = 300.0
MAX_COST_S = 30 * 60
NUM_SAMPLE_ORIGINS = 200


def _inspect(name: str, router, sample_ids: list[str]) -> None:
    start = perf_counter()
    print(f"\n{name}: {router.attached_origin_count:,} attached origins; "
          f"{router.attached_job_count:,} attached employment cells")
    for origin in sample_ids:
        costs = router.shortest_job_costs(origin, MAX_COST_S)
        if costs is None:
            print(f"  {origin}: not attached")
            continue
        reachable = np.isfinite(costs)
        print(f"  {origin}: {int(reachable.sum()):,} reachable employment cells; "
              f"minimum {costs[reachable].min():.1f}s" if reachable.any()
              else f"  {origin}: 0 reachable employment cells")
    print(f"  {len(sample_ids)} queries: {perf_counter()-start:.4f} s")


def main() -> None:
    t0 = perf_counter()
    origins = pd.read_csv(SPATIAL_DIR / "origins.csv", dtype={"hex_id": str})
    jobs = pd.read_csv(SPATIAL_DIR / "jobs.csv", dtype={"job_id": str})
    sample_ids = origins["hex_id"].astype(str).head(NUM_SAMPLE_ORIGINS).tolist()
    linkstats = LinkTravelTimes.from_linkstats(LINKSTATS_FILE, hour_start=HOUR_START,
                                                statistic="avg")
    car, bike = load_road_networks(NETWORK_FILE, linkstats,
                                    bike_speed_m_s=BIKE_SPEED_M_S)
    snapper = ScenarioRoadSnapper(origins, jobs,
                                 max_origin_distance_m=ORIGIN_SNAP_M,
                                 max_job_distance_m=JOB_SNAP_M)
    car_links = snapper.attach(car.graph)
    bike_links = snapper.attach(bike.graph)
    car_router = RoadRouter(car.graph, car_links.origins.connectors,
                            car_links.jobs.connectors, jobs)
    bike_router = RoadRouter(bike.graph, bike_links.origins.connectors,
                             bike_links.jobs.connectors, jobs)
    t1 = perf_counter()

    schedule = TransitSchedule.from_matsim_schedule(SCHEDULE_FILE)
    pt = PTNetwork.from_schedule(schedule, linkstats,
                                  transfer_penalty_s=PT_TRANSFER_PENALTY_S)
    pt_origin_links = pd.read_csv(SPATIAL_DIR / "pt_origin_connectors.csv",
                                  dtype={"point_id": str, "stop_id": str})
    pt_job_links = pd.read_csv(SPATIAL_DIR / "pt_job_connectors.csv",
                               dtype={"point_id": str, "stop_id": str})
    pt_router = PTRouter(pt, pt_origin_links, pt_job_links, jobs)
    t2 = perf_counter()

    print("=" * 68)
    print("Accessibility routing — shortest job costs (no aggregation)")
    print("=" * 68)
    print(f"Network: {NETWORK_FILE}")
    print(f"Schedule: {SCHEDULE_FILE}")
    print(f"Linkstats: {LINKSTATS_FILE} ({linkstats.travel_time_column})")
    print(f"Sample origins: {sample_ids}")
    print(f"Road preparation: {t1-t0:.3f}s; PT preparation: {t2-t1:.3f}s")
    print(f"PT skipped jobs absent from jobs.csv: {len(pt_router.unknown_job_ids)}")
    print(f"PT unserved walking stop IDs: {len(pt_router.unserved_stop_ids)}")
    _inspect("Car", car_router, sample_ids)
    _inspect("Bike (perceived time)", bike_router, sample_ids)
    _inspect("PT (walking + rides + scheduled transfers)", pt_router, sample_ids)
    print("No scenario connector CSVs or accessibility metrics created.")


if __name__ == "__main__":
    main()

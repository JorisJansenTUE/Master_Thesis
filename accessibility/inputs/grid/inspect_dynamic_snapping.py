"""Inspect in-memory car/bike snapping against an actual scenario network.

Uses the fixed spatial inputs from the baseline, but DOES NOT load or save
car/bike connector CSV files. Re-run after changing SCENARIO_NETWORK_FILE to
an intervention scenario. The same ScenarioRoadSnapper object can be kept
alive in an optimisation worker and reused for every scenario.

Run from the project root:
    python -m accessibility.inputs.grid.inspect_dynamic_snapping
"""
from __future__ import annotations

from pathlib import Path
from time import perf_counter

import pandas as pd

from accessibility.inputs.grid.snapping import ScenarioRoadSnapper
from accessibility.inputs.grid import destination_settings as dest
from accessibility.inputs.links.linkstats import LinkTravelTimes
from accessibility.inputs.links.network import load_road_networks

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SPATIAL_DIR = dest.spatial_dir()

SCENARIO_DIR = PROJECT_ROOT / "data" / "Simulated_Runs" / "baseline_00000"
SCENARIO_NETWORK_FILE = SCENARIO_DIR / "network.xml.gz"
SCENARIO_LINKSTATS_FILE = SCENARIO_DIR / "200.linkstats.txt.gz"

MAX_ORIGIN_SNAP_M = 500.0
MAX_JOB_SNAP_M = 300.0
BIKE_SPEED_M_S = 4.17  # Provisional: match the actual eQasim bike speed later
HOUR_START = 8
LINKSTATS_STATISTIC = "avg"
AVERAGED_ITERATIONS = 8


def describe(mode: str, attachments: object, jobs: pd.DataFrame) -> None:
    for name, result in (("origins", attachments.origins), ("jobs", attachments.jobs)):
        print(
            f"  {mode} {name}: {result.attached_points:,}/{result.total_points:,}"
            f" attached; {len(result.connectors):,} directed connectors"
        )
        if result.unsnapped_ids:
            print(f"    Unattached examples: {result.unsnapped_ids[:5]}")
    attached_ids = set(attachments.jobs.connectors["point_id"])
    attached_jobs = jobs.loc[jobs["job_id"].astype(str).isin(attached_ids), "jobs"].sum()
    print(f"    Attached opportunities: {attached_jobs:,.0f}")


def main() -> None:
    t0 = perf_counter()
    origins = pd.read_csv(SPATIAL_DIR / "origins.csv", dtype={"hex_id": str})
    jobs = pd.read_csv(SPATIAL_DIR / "jobs.csv", dtype={"job_id": str})
    snapper = ScenarioRoadSnapper(
        origins, jobs,
        max_origin_distance_m=MAX_ORIGIN_SNAP_M,
        max_job_distance_m=MAX_JOB_SNAP_M,
    )
    t1 = perf_counter()

    linkstats = LinkTravelTimes.from_linkstats(
        SCENARIO_LINKSTATS_FILE,
        hour_start=HOUR_START,
        statistic=LINKSTATS_STATISTIC,
        averaged_iterations=AVERAGED_ITERATIONS,
    )
    car, bike = load_road_networks(
        SCENARIO_NETWORK_FILE, linkstats, bike_speed_m_s=BIKE_SPEED_M_S,
    )
    t2 = perf_counter()

    car_connections = snapper.attach(car.graph)
    bike_connections = snapper.attach(bike.graph)
    t3 = perf_counter()

    print("=" * 66)
    print("Dynamic road snapping — in-memory scenario connectors")
    print("=" * 66)
    print(f"Scenario network: {SCENARIO_NETWORK_FILE}")
    print(f"Fixed locations: {len(origins):,} origins, {len(jobs):,} "
          f"{dest.opportunity_label()}")
    describe("car", car_connections, jobs)
    describe("bike", bike_connections, jobs)
    print(f"Times: fixed inputs {t1-t0:.3f}s; scenario networks {t2-t1:.3f}s; "
          f"car+bike snapping {t3-t2:.3f}s")
    print("No scenario connector CSV files created.")

    # Pass these directly to the future routing layer:
    # car_connections.origins.connectors; car_connections.jobs.connectors
    # bike_connections.origins.connectors; bike_connections.jobs.connectors


if __name__ == "__main__":
    main()

"""Inspect STATENT + hexagon attachments to car and bike networks.

This replaces the previous car-only spatial inspection. Roads are used only
for snapping; no off-network access/egress time is added to connectors.

Run: python accessibility/inputs/inspect_spatial.py
"""
from __future__ import annotations

from pathlib import Path
from time import perf_counter

from .employment import EmploymentDestinations
from .origins import HexagonOrigins
from .snapping import RoadSnapIndex
from .spatial_diagnostics import export_spatial_diagnostics

from accessibility.inputs.links.linkstats import LinkTravelTimes
from accessibility.inputs.links.network import load_road_networks

from preprocessing.utils import PROJECT_ROOT

RUN_DIR = PROJECT_ROOT / "data" / "Simulated_Runs" / "baseline_00000"
NETWORK_FILE = RUN_DIR / "network.xml.gz"
LINKSTATS_FILE = RUN_DIR / "200.linkstats.txt.gz"
STUDY_AREA_FILE = PROJECT_ROOT / "data" / "raw"/ "shapes" / "REV_Cintura_project.shp"
STATENT_FILE = Path(r"C:\Users\20201733\Documents\Thesis\Data\Swiss_FSO\Employment-STATENT2023.zip")
OUTPUT_DIR = PROJECT_ROOT / "accessibility" / "outputs" / "spatial_baseline"

HEXAGON_DIAMETER_M = 500.0
MIN_HEXAGON_COVERAGE = 0.0
MAX_ORIGIN_SNAP_M = 500.0
MAX_JOB_SNAP_M = 300.0
HOUR_START = 8
LINKSTATS_STATISTIC = "avg"
AVERAGED_ITERATIONS = 8
BIKE_SPEED_M_S = 4.17  # provisional, not calibrated


def _describe(mode: str, label: str, attachments) -> None:
    print(f"  {mode} {label}: {attachments.attached_points:,}/{attachments.total_points:,} attached, "
          f"{len(attachments.connectors):,} link connectors")
    if not attachments.connectors.empty:
        dist = attachments.connectors.groupby("point_id")["off_network_distance_m"].first()
        print(f"    Off-network distance: median {dist.median():.1f} m; "
              f"p95 {dist.quantile(.95):.1f} m; max {dist.max():.1f} m")
    if attachments.unsnapped_ids:
        print(f"    Unattached examples: {attachments.unsnapped_ids[:8]}")


def main() -> None:
    for filename in (NETWORK_FILE, LINKSTATS_FILE, STUDY_AREA_FILE, STATENT_FILE):
        if not filename.is_file():
            raise FileNotFoundError(f"Input missing: {filename}; edit paths at top")
    t0 = perf_counter()
    linkstats = LinkTravelTimes.from_linkstats(
        LINKSTATS_FILE, hour_start=HOUR_START, statistic=LINKSTATS_STATISTIC,
        averaged_iterations=AVERAGED_ITERATIONS,
    )
    car, bike = load_road_networks(
        NETWORK_FILE, linkstats, bike_speed_m_s=BIKE_SPEED_M_S
    )
    t1 = perf_counter()
    origins = HexagonOrigins.from_study_area(
        STUDY_AREA_FILE, diameter_m=HEXAGON_DIAMETER_M,
        min_coverage_fraction=MIN_HEXAGON_COVERAGE,
    )
    # Include STATENT cells around both road graphs; use complete available
    # network coverage rather than clipping job opportunities to the study area.
    all_nodes = [node for network in (car, bike) for node in network.graph.nodes(data=True)]
    xs = [float(data["x"]) for _, data in all_nodes]
    ys = [float(data["y"]) for _, data in all_nodes]
    jobs = EmploymentDestinations.from_statent(
        STATENT_FILE,
        bounds=(min(xs) - MAX_JOB_SNAP_M, min(ys) - MAX_JOB_SNAP_M,
                max(xs) + MAX_JOB_SNAP_M, max(ys) + MAX_JOB_SNAP_M),
    )
    t2 = perf_counter()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    origins.save(OUTPUT_DIR / "origins.gpkg", OUTPUT_DIR / "origins.csv")
    jobs.save(OUTPUT_DIR / "jobs.csv")

    print("=" * 68)
    print("Accessibility spatial inputs — car + bike")
    print("=" * 68)
    print(f"Hexagons: {len(origins.cells):,}; employment cells: {len(jobs.jobs):,}")
    print(f"Employment total (STATENT): {jobs.jobs['jobs'].sum():,.0f}")
    print("Off-network distance is only a snapping threshold, NOT a time penalty.")
    origin_attachments = {}
    job_attachments = {}
    for mode, network in (("car", car), ("bike", bike)):
        snap_index = RoadSnapIndex(network.graph)
        origin_connections = snap_index.attach(
            origins.cells, id_column="hex_id", x_column="origin_x",
            y_column="origin_y", max_distance_m=MAX_ORIGIN_SNAP_M,
        )
        job_connections = snap_index.attach(
            jobs.jobs, id_column="job_id", x_column="x", y_column="y",
            max_distance_m=MAX_JOB_SNAP_M,
        )
        origin_attachments[mode] = origin_connections
        job_attachments[mode] = job_connections
        origin_connections.save(OUTPUT_DIR / f"{mode}_origin_connectors.csv")
        job_connections.save(OUTPUT_DIR / f"{mode}_job_connectors.csv")
        _describe(mode, "origins", origin_connections)
        _describe(mode, "employment", job_connections)
        attached_ids = set(job_connections.connectors["point_id"])
        attached_jobs = jobs.jobs.loc[jobs.jobs["job_id"].isin(attached_ids), "jobs"].sum()
        print(f"    Attached jobs total: {attached_jobs:,.0f}")

    diagnostics_file = OUTPUT_DIR / "snapping_diagnostics.gpkg"
    export_spatial_diagnostics(
        diagnostics_file,
        origins=origins.cells,
        jobs=jobs.jobs,
        origin_attachments=origin_attachments,
        job_attachments=job_attachments,
    )
    print(f"Diagnostic GeoPackage: {diagnostics_file}")
    t3 = perf_counter()
    print(f"\nTimings: networks {t1-t0:.2f}s, spatial data {t2-t1:.2f}s, "
          f"snapping/writing {t3-t2:.2f}s")
    print(f"Outputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

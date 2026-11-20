"""Prepare and inspect PT origin/job -> stop walking connectors.

Run after the existing car+bike spatial preprocessing:
    python accessibility/inputs/inspect_pt_mapping.py

It reuses the cached origins.csv and jobs.csv, so it does not rebuild the road
networks or reload STATENT. All local settings are at the top of this file.
"""
from __future__ import annotations

from pathlib import Path
from time import perf_counter

import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString

from .pt_mapping import PTStopMapper
from accessibility.inputs.links.schedule import TransitSchedule

from preprocessing.utils import PROJECT_ROOT

RUN_DIR = PROJECT_ROOT / "data" / "Simulated_Runs_v2" / "baseline_00000"
SCHEDULE_FILE = RUN_DIR / "transit_schedule.xml.gz"
SPATIAL_DIR = PROJECT_ROOT / "data" / "accessibility" / "spatial_baseline"
OUTPUT_DIR = SPATIAL_DIR

# Straight-line catchment radius. Every served stop in the radius is retained.
MAX_ORIGIN_STOP_DISTANCE_M = 800.0
MAX_JOB_STOP_DISTANCE_M = 800.0
WALK_SPEED_M_S = 1.3
WALK_DETOUR_FACTOR = 1.3  # Matsim default approximation; raise if desired
EXPORT_GPKG = False


def _describe(label: str, result) -> None:
    print(f"  {label}: {result.attached_points:,}/{result.total_points:,} attached; "
          f"{len(result.connectors):,} stop connectors")
    if not result.connectors.empty:
        nearest = result.connectors.groupby("point_id", sort=False)["distance_m"].min()
        times = result.connectors["walk_time_s"]
        print(f"    Nearest stop: median {nearest.median():.1f} m; "
              f"p95 {nearest.quantile(0.95):.1f} m")
        print(f"    Walking time per connector: median {times.median()/60:.2f} min; "
              f"p95 {times.quantile(0.95)/60:.2f} min")
    if result.unattached_ids:
        print(f"    Unattached examples: {result.unattached_ids[:8]}")


def _attach_columns(points: pd.DataFrame, result, id_col: str) -> pd.DataFrame:
    out = points.copy()
    keys = out[id_col].astype(str)
    counts = result.connectors.groupby("point_id", sort=False).size()
    out["pt_attached"] = ~keys.isin(result.unattached_ids)
    out["pt_stop_count"] = keys.map(counts).fillna(0).astype(int).to_numpy()
    out["nearest_pt_stop_m"] = keys.map(result.nearest_distances_m).astype(float).to_numpy()
    return out


def _save_gpkg(path: Path, origins, jobs, mapper, origin_links, job_links) -> None:
    """Visual diagnosis; no changes to existing car/bike GeoPackages."""
    path.parent.mkdir(parents=True, exist_ok=True)
    origin_table = _attach_columns(origins, origin_links, "hex_id")
    job_table = _attach_columns(jobs, job_links, "job_id")
    origin_points = gpd.GeoDataFrame(
        origin_table, geometry=gpd.points_from_xy(origin_table.origin_x, origin_table.origin_y),
        crs="EPSG:2056",
    )
    employment = gpd.GeoDataFrame(
        job_table, geometry=gpd.points_from_xy(job_table.x, job_table.y),
        crs="EPSG:2056",
    )
    stops = pd.DataFrame({
        "stop_id": mapper.stop_ids,
        "name": [mapper.schedule.stops[sid].name for sid in mapper.stop_ids],
        "route_count": mapper.route_counts,
        "x": mapper.coordinates[:, 0],
        "y": mapper.coordinates[:, 1],
    })
    stops = gpd.GeoDataFrame(
        stops, geometry=gpd.points_from_xy(stops.x, stops.y), crs="EPSG:2056",
    )
    layers = [
        ("origin_points", origin_points),
        ("employment", employment),
        ("pt_stops", stops),
    ]
    if (SPATIAL_DIR / "origins.gpkg").is_file():
        hexagons = gpd.read_file(SPATIAL_DIR / "origins.gpkg", layer="origins")
        hexagons["hex_id"] = hexagons["hex_id"].astype(str)
        columns = origin_table[["hex_id", "pt_attached", "pt_stop_count", "nearest_pt_stop_m"]]
        layers.insert(0, ("hexagons", hexagons.merge(columns, on="hex_id", validate="one_to_one")))

    # In QGIS these lines show the actual Euclidean connections. They are
    # strictly access/egress connectors, NOT links for stop-to-stop transfers.
    for label, links, table, id_column, xcol, ycol in (
        ("origin_walks", origin_links, origins, "hex_id", "origin_x", "origin_y"),
        ("job_walks", job_links, jobs, "job_id", "x", "y"),
    ):
        if links.connectors.empty:
            continue
        lookup = table.set_index(id_column)[[xcol, ycol]]
        destination = {sid: mapper.coordinates[i] for i, sid in enumerate(mapper.stop_ids)}
        records = []
        lines = []
        for point_id, stop_id, distance, walk_time in links.connectors[
            ["point_id", "stop_id", "distance_m", "walk_time_s"]
        ].itertuples(index=False, name=None):
            source_xy = lookup.loc[point_id].to_numpy(dtype=float)
            lines.append(LineString([source_xy, destination[stop_id]]))
            records.append((point_id, stop_id, distance, walk_time))
        connections = gpd.GeoDataFrame(
            pd.DataFrame(records, columns=["point_id", "stop_id", "distance_m", "walk_time_s"]),
            geometry=lines, crs="EPSG:2056",
        )
        layers.append((label, connections))

    if path.exists():
        path.unlink()
    for layer_index, (name, frame) in enumerate(layers):
        frame.to_file(path, layer=name, driver="GPKG",
                      mode="w" if layer_index == 0 else "a")


def main() -> None:
    required = (SCHEDULE_FILE, SPATIAL_DIR / "origins.csv", SPATIAL_DIR / "jobs.csv")
    for filename in required:
        if not filename.is_file():
            raise FileNotFoundError(f"Missing input: {filename}")
    t0 = perf_counter()
    schedule = TransitSchedule.from_matsim_schedule(SCHEDULE_FILE)
    mapper = PTStopMapper(
        schedule, walking_speed_m_s=WALK_SPEED_M_S,
        walking_detour_factor=WALK_DETOUR_FACTOR,
    )
    t1 = perf_counter()
    origins = pd.read_csv(SPATIAL_DIR / "origins.csv", dtype={"hex_id": str})
    jobs = pd.read_csv(SPATIAL_DIR / "jobs.csv", dtype={"job_id": str})
    origin_links = mapper.attach(
        origins, id_column="hex_id", x_column="origin_x", y_column="origin_y",
        max_distance_m=MAX_ORIGIN_STOP_DISTANCE_M,
    )
    job_links = mapper.attach(
        jobs, id_column="job_id", x_column="x", y_column="y",
        max_distance_m=MAX_JOB_STOP_DISTANCE_M,
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    origin_links.save(OUTPUT_DIR / "pt_origin_connectors.csv")
    job_links.save(OUTPUT_DIR / "pt_job_connectors.csv")
    t2 = perf_counter()
    print("=" * 70)
    print("Accessibility spatial inputs — PT access and egress")
    print("=" * 70)
    print(f"Schedule: {SCHEDULE_FILE}")
    print(f"Served PT stop facilities: {mapper.num_stops:,}")
    print(f"Origins: {len(origins):,}; employment cells: {len(jobs):,}; jobs: {jobs.jobs.sum():,.0f}")
    print(f"Catchment: origin {MAX_ORIGIN_STOP_DISTANCE_M:g} m; "
          f"destination {MAX_JOB_STOP_DISTANCE_M:g} m")
    print(f"Walking time = crow-flight distance × {WALK_DETOUR_FACTOR:g} / {WALK_SPEED_M_S:g} m/s")
    print("Only scheduled PT transfers are permitted in PTNetwork; no unscheduled walk transfers.")
    _describe("PT origins", origin_links)
    _describe("PT employment", job_links)
    valid_jobs = set(job_links.connectors.point_id)
    included = jobs.loc[jobs.job_id.isin(valid_jobs), "jobs"].sum()
    print(f"  Employment with PT egress: {included:,.0f} jobs ({100*included/jobs.jobs.sum():.1f}%)")
    if EXPORT_GPKG:
        diag = OUTPUT_DIR / "pt_mapping_diagnostics.gpkg"
        _save_gpkg(diag, origins, jobs, mapper, origin_links, job_links)
        print(f"Diagnostic GeoPackage: {diag}")
    t3 = perf_counter()
    print(f"Times: schedule/index {t1-t0:.2f}s, mapping {t2-t1:.2f}s, "
          f"GIS/output {t3-t2:.2f}s")
    print(f"Outputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

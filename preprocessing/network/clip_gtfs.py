from __future__ import annotations

from types import SimpleNamespace
import logging
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Optional

import pandas as pd
from shapely.geometry import Point

from preprocessing.network.config import PipelineConfig, PipelinePaths
from preprocessing.network.geometry import read_project_polygon_wgs84
from preprocessing.utils import RAW_DIR, OSM_DIR, GTFS_DIR

REQUIRED_GTFS_FILES = {
    "agency.txt",
    "routes.txt",
    "trips.txt",
    "stop_times.txt",
    "stops.txt",
}

OPTIONAL_GTFS_FILES = {
    "calendar.txt",
    "calendar_dates.txt",
    "shapes.txt",
    "frequencies.txt",
    "transfers.txt",
    "feed_info.txt",
    "levels.txt",
    "pathways.txt",
    "translations.txt",
    "attributions.txt",
}

EXCLUDED_ROUTE_TYPES = {"107","1000","1300","1400"}


def read_gtfs_table(gtfs_dir: Path, filename: str, required: bool = False) -> Optional[pd.DataFrame]:
    path = gtfs_dir / filename

    if not path.exists():
        if required:
            raise FileNotFoundError(f"Required GTFS file missing: {filename}")
        return None

    return pd.read_csv(path, dtype=str, keep_default_na=False)


def write_gtfs_table(df: Optional[pd.DataFrame], gtfs_dir: Path, filename: str) -> None:
    if df is not None:
        df.to_csv(gtfs_dir / filename, index=False)


def copy_unknown_gtfs_files(input_dir: Path, output_dir: Path, known_files: set[str]) -> None:
    for src in input_dir.iterdir():
        if src.name in known_files:
            continue
        if src.is_file():
            shutil.copy2(src, output_dir / src.name)


def zip_directory(input_dir: Path, output_zip: Path) -> None:
    output_zip.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(input_dir.iterdir()):
            if path.is_file():
                zf.write(path, arcname=path.name)

def log_route_types(tables: dict[str, Optional[pd.DataFrame]]) -> None:
    routes = tables["routes.txt"]
    trips = tables["trips.txt"]

    assert routes is not None
    assert trips is not None

    summary = (
        trips[["trip_id", "route_id"]]
        .merge(routes[["route_id", "route_type"]], on="route_id", how="left")
        .groupby("route_type")
        .agg(routes=("route_id", "nunique"), trips=("trip_id", "nunique"))
        .sort_index()
    )

    logging.info("GTFS route types in project area:\n%s", summary.to_string())

def filter_route_types(
    tables: dict[str, Optional[pd.DataFrame]],
    excluded_route_types: set[str],
) -> dict[str, Optional[pd.DataFrame]]:
    routes = tables["routes.txt"]
    trips = tables["trips.txt"]
    stop_times = tables["stop_times.txt"]

    assert routes is not None
    assert trips is not None
    assert stop_times is not None

    route_types = routes["route_type"].str.strip()
    excluded_route_ids = set(routes.loc[route_types.isin(excluded_route_types), "route_id"])

    excluded_trip_ids = set(
        trips.loc[trips["route_id"].isin(excluded_route_ids), "trip_id"]
    )

    logging.info(
        "Removing %s routes and %s trips with route types %s.",
        len(excluded_route_ids),
        len(excluded_trip_ids),
        sorted(excluded_route_types),
    )

    filtered = dict(tables)
    filtered["routes.txt"] = routes[~routes["route_id"].isin(excluded_route_ids)].copy()
    filtered["trips.txt"] = trips[~trips["trip_id"].isin(excluded_trip_ids)].copy()
    filtered["stop_times.txt"] = stop_times[~stop_times["trip_id"].isin(excluded_trip_ids)].copy()

    if filtered.get("frequencies.txt") is not None:
        filtered["frequencies.txt"] = filtered["frequencies.txt"][~filtered["frequencies.txt"]["trip_id"].isin(excluded_trip_ids)].copy()

    return filtered


def get_stop_inside_mask(stops: pd.DataFrame, polygon) -> pd.Series:
    lat = pd.to_numeric(stops["stop_lat"], errors="coerce")
    lon = pd.to_numeric(stops["stop_lon"], errors="coerce")

    if lat.isna().any() or lon.isna().any():
        bad = stops.loc[lat.isna() | lon.isna(), "stop_id"].tolist()[:10]
        raise ValueError(f"Some stops have invalid coordinates. Examples: {bad}")

    points = [Point(xy) for xy in zip(lon, lat)]
    return pd.Series([polygon.covers(point) for point in points], index=stops.index)

def get_candidate_stop_ids_by_bbox(
    stops: pd.DataFrame,
    polygon,
    margin_degrees: float = 0.20,
) -> set[str]:
    """
    Quickly select stops in a larger bounding box around the project polygon.

    This is only a coarse prefilter. The exact inside/outside decision is still
    done using the project polygon.
    """
    minx, miny, maxx, maxy = polygon.bounds

    lon = pd.to_numeric(stops["stop_lon"], errors="coerce")
    lat = pd.to_numeric(stops["stop_lat"], errors="coerce")

    mask = (
        (lon >= minx - margin_degrees)
        & (lon <= maxx + margin_degrees)
        & (lat >= miny - margin_degrees)
        & (lat <= maxy + margin_degrees)
    )

    return set(stops.loc[mask, "stop_id"])

def get_rail_trip_ids(routes: pd.DataFrame, trips: pd.DataFrame, candidate_trip_ids: set[str]) -> set[str]:
    """Return rail trips among the spatially prefiltered candidate trips."""
    candidate_trips = trips[trips["trip_id"].isin(candidate_trip_ids)].copy()

    route_type = pd.to_numeric(routes["route_type"], errors="coerce")
    rail_route_ids = set(
        routes.loc[(route_type == 2) | route_type.between(100, 199), "route_id"]
    )

    return set(
        candidate_trips.loc[
            candidate_trips["route_id"].isin(rail_route_ids),
            "trip_id",
        ]
    )

def clip_stop_times(
    stop_times: pd.DataFrame,
    trips: pd.DataFrame,
    routes: pd.DataFrame,
    inside_stop_ids: set[str],
    nearby_outside_stop_ids: set[str],
    candidate_stop_ids: set[str]
) -> pd.DataFrame:
    """
    Fast GTFS trip clipping.

    Logic:
    1. First keep only trips that touch a larger candidate area.
    2. Then keep only trips that have at least one stop inside the exact polygon.
    3. Keep all stops inside the polygon.
    4. For busses: Keep one adjacent outside stop at each boundary crossing, but only when
       that stop is within approximately 1 km of the polygon.
    5. Remove trips with fewer than two retained stops.
    """

    logging.info("Prefiltering GTFS trips using candidate stop area.")

    candidate_trip_ids = set(
        stop_times.loc[
            stop_times["stop_id"].isin(candidate_stop_ids),
            "trip_id",
        ]
    )

    logging.info(
        "Trips touching candidate area: %s / %s",
        len(candidate_trip_ids),
        stop_times["trip_id"].nunique(),
    )

    rail_trip_ids = get_rail_trip_ids(
        routes=routes,
        trips=trips,
        candidate_trip_ids=candidate_trip_ids,
    )

    logging.info(
        "Rail trips touching the candidate area: %s",
        len(rail_trip_ids),
    )

    st = stop_times[stop_times["trip_id"].isin(candidate_trip_ids)].copy()

    if st.empty:
        raise ValueError("No GTFS trips touch the candidate clipping area.")

    st["stop_sequence_num"] = pd.to_numeric(st["stop_sequence"], errors="coerce")

    if st["stop_sequence_num"].isna().any():
        bad = st.loc[st["stop_sequence_num"].isna(), "trip_id"].head(10).tolist()
        raise ValueError(f"Invalid stop_sequence values. Example trip_ids: {bad}")

    st = st.sort_values(["trip_id", "stop_sequence_num"]).copy()
    st["_pos"] = st.groupby("trip_id").cumcount()
    st["_inside"] = st["stop_id"].isin(inside_stop_ids)
    st["_rail"] = st["trip_id"].isin(rail_trip_ids)

    inside_trip_ids = set(st.loc[st["_inside"], "trip_id"])

    logging.info(
        "Trips entering exact project polygon: %s / %s",
        len(inside_trip_ids),
        len(candidate_trip_ids),
    )

    if not inside_trip_ids:
        raise ValueError("No GTFS trips enter the exact project polygon.")

    st = st[st["trip_id"].isin(inside_trip_ids)].copy()

    # An outside stop is adjacent when the previous or next stop is inside.
    st["_previous_inside"] = st.groupby("trip_id")["_inside"].shift(1, fill_value=False)
    st["_next_inside"] = st.groupby("trip_id")["_inside"].shift(-1, fill_value=False)

    adjacent_outside_mask = (
        ~st["_inside"]
        & ~st["_rail"]
        & (st["_previous_inside"] | st["_next_inside"])
        & st["stop_id"].isin(nearby_outside_stop_ids)
    )

    clipped = st.loc[st["_inside"] | adjacent_outside_mask].copy()

    # Remove trips that still have fewer than two retained stops.
    stop_counts = clipped.groupby("trip_id").size()
    valid_trip_ids = set(stop_counts[stop_counts >= 2].index)
    invalid_trip_count = (stop_counts < 2).sum()

    if invalid_trip_count:
        logging.warning(
            "Removing %s GTFS trips with fewer than two stops after clipping.",
            invalid_trip_count,
        )

    clipped = clipped[clipped["trip_id"].isin(valid_trip_ids)].copy()

    if clipped.empty:
        raise ValueError("No valid GTFS trips remain after clipping.")

    outside_stop_time_count = (~clipped["_inside"]).sum()

    clipped = clipped.sort_values(["trip_id", "_pos"]).copy()
    clipped["stop_sequence"] = (clipped.groupby("trip_id").cumcount() + 1).astype(str)

    clipped = clipped.drop(
        columns=[
            "stop_sequence_num",
            "_pos",
            "_inside",
            "_rail",
            "_previous_inside",
            "_next_inside",
        ]
    )

    logging.info(
        "GTFS trips retained after exact clipping: %s",
        clipped["trip_id"].nunique(),
    )
    logging.info("Nearby outside stop_times retained: %s", outside_stop_time_count)
    logging.info("GTFS stop_times retained after clipping: %s", len(clipped))

    return clipped

def filter_gtfs_tables(
    tables: dict[str, Optional[pd.DataFrame]],
    clipped_stop_times: pd.DataFrame,
) -> dict[str, Optional[pd.DataFrame]]:
    kept_trip_ids = set(clipped_stop_times["trip_id"])
    kept_stop_ids = set(clipped_stop_times["stop_id"])

    stops = tables["stops.txt"]
    trips = tables["trips.txt"]
    routes = tables["routes.txt"]
    agency = tables["agency.txt"]

    assert stops is not None
    assert trips is not None
    assert routes is not None
    assert agency is not None

    trips = trips[trips["trip_id"].isin(kept_trip_ids)].copy()
    kept_route_ids = set(trips["route_id"])
    routes = routes[routes["route_id"].isin(kept_route_ids)].copy()

    if "agency_id" in routes.columns and "agency_id" in agency.columns:
        kept_agency_ids = set(routes["agency_id"])
        agency = agency[agency["agency_id"].isin(kept_agency_ids)].copy()

    stops = stops[stops["stop_id"].isin(kept_stop_ids)].copy()

    filtered = dict(tables)
    filtered["agency.txt"] = agency
    filtered["routes.txt"] = routes
    filtered["trips.txt"] = trips
    filtered["stop_times.txt"] = clipped_stop_times
    filtered["stops.txt"] = stops

    if filtered.get("calendar.txt") is not None and "service_id" in trips.columns:
        kept_service_ids = set(trips["service_id"])
        filtered["calendar.txt"] = filtered["calendar.txt"][
            filtered["calendar.txt"]["service_id"].isin(kept_service_ids)
        ].copy()

    if filtered.get("calendar_dates.txt") is not None and "service_id" in trips.columns:
        kept_service_ids = set(trips["service_id"])
        filtered["calendar_dates.txt"] = filtered["calendar_dates.txt"][
            filtered["calendar_dates.txt"]["service_id"].isin(kept_service_ids)
        ].copy()

    if filtered.get("shapes.txt") is not None and "shape_id" in trips.columns:
        kept_shape_ids = set(trips.loc[trips["shape_id"] != "", "shape_id"])
        filtered["shapes.txt"] = filtered["shapes.txt"][
            filtered["shapes.txt"]["shape_id"].isin(kept_shape_ids)
        ].copy()

    if filtered.get("frequencies.txt") is not None:
        filtered["frequencies.txt"] = filtered["frequencies.txt"][
            filtered["frequencies.txt"]["trip_id"].isin(kept_trip_ids)
        ].copy()
    # Always drop transfers.txt as it can cause issues and is not needed for the schedule.
    if "transfers.txt" in filtered:
        logging.info("Dropping transfers.txt from clipped GTFS.")
        filtered["transfers.txt"] = None
        return filtered

def clip_gtfs_to_shape(config: PipelineConfig, paths: PipelinePaths) -> None:
    if paths.clipped_gtfs_dir.exists():
        shutil.rmtree(paths.clipped_gtfs_dir)
    paths.clipped_gtfs_dir.mkdir(parents=True, exist_ok=True)
    # Exact polygon used to determine which stops are inside the study area.
    polygon = read_project_polygon_wgs84(
        config.project_shape,
        config.shape_buffer_degrees
    )
    # Temporary GTFS-only buffer of approximately 1km
    nearby_polygon = polygon.buffer(0.01) # ~1km buffer in degrees

    with tempfile.TemporaryDirectory(prefix="gtfs_pipeline_") as tmp_str:
        tmp = Path(tmp_str)
        input_dir = tmp / "input_gtfs"
        input_dir.mkdir()

        logging.info("Extracting GTFS zip: %s", config.gtfs_zip)
        with zipfile.ZipFile(config.gtfs_zip, "r") as zf:
            zf.extractall(input_dir)

        all_known_files = REQUIRED_GTFS_FILES | OPTIONAL_GTFS_FILES
        tables: dict[str, Optional[pd.DataFrame]] = {}

        for filename in REQUIRED_GTFS_FILES:
            tables[filename] = read_gtfs_table(input_dir, filename, required=True)

        for filename in OPTIONAL_GTFS_FILES:
            tables[filename] = read_gtfs_table(input_dir, filename, required=False)
        
        if EXCLUDED_ROUTE_TYPES is not None and len(EXCLUDED_ROUTE_TYPES) > 0:
            logging.info("Filtering out excluded route types: %s", sorted(EXCLUDED_ROUTE_TYPES))
            tables = filter_route_types(
                tables=tables,
                excluded_route_types=EXCLUDED_ROUTE_TYPES, #TO DO: make this configurable
            )

        stops = tables["stops.txt"]
        stop_times = tables["stop_times.txt"]
        routes = tables["routes.txt"]
        trips = tables["trips.txt"]

        assert stops is not None
        assert stop_times is not None
        assert routes is not None
        assert trips is not None

        inside_mask = get_stop_inside_mask(stops, polygon)
        nearby_mask = get_stop_inside_mask(stops, nearby_polygon)

        inside_stop_ids = set(stops.loc[inside_mask, "stop_id"])
        nearby_outside_stop_ids = set(
            stops.loc[nearby_mask & ~inside_mask, "stop_id"]
        )

        logging.info(
            "GTFS stops inside exact project area: %s / %s",
            len(inside_stop_ids),
            len(stops),
        )
        logging.info(
            "Outside GTFS stops within approximate 1 km buffer: %s",
            len(nearby_outside_stop_ids),
        )

        if not inside_stop_ids:
            raise ValueError("No GTFS stops found inside the project shapefile.")

        candidate_stop_ids = get_candidate_stop_ids_by_bbox(
            stops=stops,
            polygon=polygon,
            margin_degrees=0.02,
        )

        logging.info(
            "GTFS stops in candidate bbox: %s / %s",
            len(candidate_stop_ids),
            len(stops),
        )

        clipped_stop_times = clip_stop_times(
            stop_times=stop_times,
            inside_stop_ids=inside_stop_ids,
            nearby_outside_stop_ids=nearby_outside_stop_ids,
            candidate_stop_ids=candidate_stop_ids,
            trips=tables["trips.txt"],
            routes=tables["routes.txt"],
        )
        filtered_tables = filter_gtfs_tables(tables, clipped_stop_times)
        log_route_types(filtered_tables)

        for filename in REQUIRED_GTFS_FILES | OPTIONAL_GTFS_FILES:
            write_gtfs_table(filtered_tables.get(filename), paths.clipped_gtfs_dir, filename)

        copy_unknown_gtfs_files(input_dir, paths.clipped_gtfs_dir, all_known_files)

    zip_directory(paths.clipped_gtfs_dir, paths.clipped_gtfs_zip_copy)

    logging.info("Wrote clipped GTFS folder: %s", paths.clipped_gtfs_dir)
    logging.info("Wrote debug copy of clipped GTFS zip: %s", paths.clipped_gtfs_zip_copy)

if __name__ == "__main__":
    # Standalone test run for debugging and development.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    config = SimpleNamespace(
        project_shape=Path(f"{RAW_DIR}/shapes/locarno_project_boundary.shp"),
        gtfs_zip=Path(f"{GTFS_DIR}/gtfs_fp2026_20260422.zip"),
        shape_buffer_degrees=0.0
    )

    paths = SimpleNamespace(
        clipped_gtfs_dir=Path("data/interim/test/clipped_gtfs"),
        clipped_gtfs_zip_copy=Path("data/interim/test/clipped_gtfs.zip"),
    )

    clip_gtfs_to_shape(config, paths)
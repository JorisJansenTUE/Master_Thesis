import requests
import pandas as pd
import geopandas as gpd
import osmium

from pathlib import Path
from collections import Counter
from shapely.geometry import LineString

from utils import PROJECT_ROOT,DATA_DIR, OSM_DIR

def Get_Shapefile(api_url: str, outputname: str, path: str = f"{OSM_DIR}/shapes", target_crs: str = None):
    """
    Fetches GeoJSON-like data from an API and converts it into a shapefile.
    Or instead of an API, you can also directly load a local GeoJSON file by using "file://path/to/file.geojson" as the api_url.

    Parameters
    ----------
    api_url : str
        API endpoint returning GeoJSON-like data (e.g. geo.admin.ch)
    outputname : str
        Name of the output shapefile (without extension)
    path : str, optional
        Output directory (default = "data/osm")

    Returns
    -------
    str
        Full path to the created shapefile
    """

    # Ensure output directory exists
    output_dir = Path(path)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Request data
    response = requests.get(api_url)
    response.raise_for_status()
    data = response.json()

    # Handle Swiss API structure ("feature" instead of "features")
    if "feature" in data:
        features = [data["feature"]]
    elif "features" in data:
        features = data["features"]
    else:
        raise ValueError("Invalid GeoJSON structure: no 'feature(s)' found")

    # Convert to GeoDataFrame
    gdf = gpd.GeoDataFrame.from_features(features, crs="EPSG:4326")  # Assuming API returns WGS84
    # Reproject if target CRS is specified
    if target_crs:
        gdf = gdf.to_crs(target_crs)

    # Output path
    output_file = output_dir/f"{outputname}.shp"

    # Save shapefile
    gdf.to_file(output_file)

    print(f"Shapefile saved to: {output_file}")

    return str(output_file)




def extract_gtfs_routes_through_area(
    gtfs_folder: str | Path,
    boundary_shp: str | Path,
    output_folder: str | Path,
) -> None:
    """
    Extract GTFS routes/trips that pass through a polygon boundary.

    Keeps:
    - routes with at least one stop inside the boundary
    - all trips belonging to those routes
    - all stop_times for those trips
    - all stops used by those trips
    - related calendar, calendar_dates, shapes, agency files when present

    Parameters
    ----------
    gtfs_folder : path
        Folder containing GTFS .txt files.
    boundary_shp : path
        Shapefile polygon of study area, e.g. Locarno_Boundary.shp.
    output_folder : path
        Folder where filtered GTFS files will be written.
    """

    gtfs_folder = Path(gtfs_folder)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    # -------------------------
    # Load boundary
    # -------------------------
    boundary = gpd.read_file(boundary_shp)

    # Ensure it's WGS84
    if boundary.crs != "EPSG:4326":
        print(f"Reprojecting boundary from {boundary.crs} to EPSG:4326")
        boundary = boundary.to_crs("EPSG:4326")

    

    # -------------------------
    # Load required GTFS files
    # -------------------------
    stops = pd.read_csv(gtfs_folder / "stops.txt")
    routes = pd.read_csv(gtfs_folder / "routes.txt")
    trips = pd.read_csv(gtfs_folder / "trips.txt")
    stop_times = pd.read_csv(gtfs_folder / "stop_times.txt")

    stops_gdf = gpd.GeoDataFrame(
        stops,
        geometry=gpd.points_from_xy(stops["stop_lon"], stops["stop_lat"]),
        crs="EPSG:4326",
    )

    
    
    # -------------------------
    # Find stops inside Locarno boundary
    # -------------------------
    stops_in_area = stops_gdf[stops_gdf.geometry.within(boundary.geometry.union_all())]

    if stops_in_area.empty:
        raise ValueError(
            "No GTFS stops found inside the boundary. "
            "Check CRS, shapefile location, or add a buffer."
        )

    stop_ids_in_area = set(stops_in_area["stop_id"].astype(str))

    # Ensure IDs are strings for reliable matching
    stop_times["stop_id"] = stop_times["stop_id"].astype(str)
    stop_times["trip_id"] = stop_times["trip_id"].astype(str)
    trips["trip_id"] = trips["trip_id"].astype(str)
    trips["route_id"] = trips["route_id"].astype(str)
    routes["route_id"] = routes["route_id"].astype(str)

    # Trips with at least one stop inside the polygon
    trip_ids_in_area = set(
        stop_times.loc[
            stop_times["stop_id"].isin(stop_ids_in_area),
            "trip_id",
        ]
    )

    filtered_trips = trips[trips["trip_id"].isin(trip_ids_in_area)].copy()
    route_ids_in_area = set(filtered_trips["route_id"])

    filtered_routes = routes[routes["route_id"].isin(route_ids_in_area)].copy()

    # Keep complete stop sequence for selected trips
    filtered_stop_times = stop_times[stop_times["trip_id"].isin(trip_ids_in_area)].copy()

    # Keep all stops used by selected trips, including stops outside Locarno
    used_stop_ids = set(filtered_stop_times["stop_id"])
    filtered_stops = stops[stops["stop_id"].astype(str).isin(used_stop_ids)].copy()

    # -------------------------
    # Write core GTFS files
    # -------------------------
    filtered_stops.to_csv(output_folder / "stops.txt", index=False)
    filtered_routes.to_csv(output_folder / "routes.txt", index=False)
    filtered_trips.to_csv(output_folder / "trips.txt", index=False)
    filtered_stop_times.to_csv(output_folder / "stop_times.txt", index=False)

    # -------------------------
    # Optional GTFS files
    # -------------------------
    optional_files = [
        "agency.txt",
        "calendar.txt",
        "calendar_dates.txt",
        "shapes.txt",
        "feed_info.txt",
        "frequencies.txt",
        "transfers.txt",
    ]

    for filename in optional_files:
        path = gtfs_folder / filename
        if not path.exists():
            continue

        df = pd.read_csv(path)

        if filename == "calendar.txt" and "service_id" in df.columns:
            service_ids = set(filtered_trips["service_id"])
            df = df[df["service_id"].isin(service_ids)]

        elif filename == "calendar_dates.txt" and "service_id" in df.columns:
            service_ids = set(filtered_trips["service_id"])
            df = df[df["service_id"].isin(service_ids)]

        elif filename == "shapes.txt" and "shape_id" in df.columns and "shape_id" in filtered_trips.columns:
            shape_ids = set(filtered_trips["shape_id"].dropna())
            df = df[df["shape_id"].isin(shape_ids)]

        elif filename == "frequencies.txt" and "trip_id" in df.columns:
            df = df[df["trip_id"].astype(str).isin(trip_ids_in_area)]

        elif filename == "transfers.txt":
            if {"from_stop_id", "to_stop_id"}.issubset(df.columns):
                df = df[
                    df["from_stop_id"].astype(str).isin(used_stop_ids)
                    & df["to_stop_id"].astype(str).isin(used_stop_ids)
                ]

        df.to_csv(output_folder / filename, index=False)

    print("Filtered GTFS written to:", output_folder)
    print("Stops inside boundary:", len(stops_in_area))
    print("Trips kept:", len(filtered_trips))
    print("Routes kept:", len(filtered_routes))


def gtfs_stops_to_gpkg(gtfs_folder, output_file):
    """
    Extracts GTFS stops and saves them as a GeoPackage layer useful for visualizing GTFS routes in QGIS or similar tools.

    Parameters
    ---------- 
    gtfs_folder : path
        Folder containing GTFS .txt files.
    output_file : path
        Output file path for the GeoPackage containing the stops layer.
    """
    gtfs_folder = Path(gtfs_folder)

    stops = pd.read_csv(gtfs_folder / "stops.txt")

    gdf = gpd.GeoDataFrame(
        stops,
        geometry=gpd.points_from_xy(stops["stop_lon"], stops["stop_lat"]),
        crs="EPSG:4326",
    )

    gdf.to_file(output_file, layer="gtfs_stops", driver="GPKG")

    print(f"Saved GTFS stops to {output_file}")

def gtfs_rough_routes_to_gpkg(gtfs_folder, output_file):
    """
    !!! Includes mistake, as it only connects the stops of one representative trip per route, which may not capture the full route geometry if there are multiple trips with different stop sequences. !!!

    Creates rough route geometries by connecting the stops of a representative trip for each route, useful for visualizing GTFS routes in QGIS or similar tools.
    
    Parameters
    ----------  
    gtfs_folder : path
        Folder containing GTFS .txt files.
    output_file : path
        Output file path for the GeoPackage containing the route geometries.
    """
    gtfs_folder = Path(gtfs_folder)

    stops = pd.read_csv(gtfs_folder / "stops.txt", dtype=str)
    stop_times = pd.read_csv(gtfs_folder / "stop_times.txt", dtype=str)
    trips = pd.read_csv(gtfs_folder / "trips.txt", dtype=str)
    routes = pd.read_csv(gtfs_folder / "routes.txt", dtype=str)

    stops["stop_lat"] = stops["stop_lat"].astype(float)
    stops["stop_lon"] = stops["stop_lon"].astype(float)
    stop_times["stop_sequence"] = stop_times["stop_sequence"].astype(int)

    # Use one representative trip per route
    representative_trips = (
        trips
        .drop_duplicates(subset="route_id")
        [["route_id", "trip_id"]]
    )

    merged = (
        representative_trips
        .merge(stop_times, on="trip_id", how="left")
        .merge(stops[["stop_id", "stop_lon", "stop_lat"]], on="stop_id", how="left")
        .merge(routes, on="route_id", how="left")
    )

    lines = []

    for route_id, group in merged.groupby("route_id"):
        group = group.sort_values("stop_sequence")

        coords = list(zip(group["stop_lon"], group["stop_lat"]))
        coords = [c for c in coords if pd.notna(c[0]) and pd.notna(c[1])]

        if len(coords) >= 2:
            lines.append({
                "route_id": route_id,
                "route_short_name": group["route_short_name"].iloc[0] if "route_short_name" in group else None,
                "route_long_name": group["route_long_name"].iloc[0] if "route_long_name" in group else None,
                "route_type": group["route_type"].iloc[0] if "route_type" in group else None,
                "trip_id": group["trip_id"].iloc[0],
                "geometry": LineString(coords),
            })

    gdf = gpd.GeoDataFrame(lines, crs="EPSG:4326")
    gdf.to_file(output_file, layer="gtfs_routes_from_stops", driver="GPKG")

    print(f"Saved rough GTFS route lines to {output_file}")

def find_unused_stops(gtfs_folder):
    """
    Identifies GTFS stops that are not used in any stop_times, which can indicate orphaned stops that are not part of any active route.
    This can help with data cleaning and understanding the GTFS dataset.
    """

    gtfs_folder = Path(gtfs_folder)

    stops = pd.read_csv(gtfs_folder / "stops.txt", dtype=str)
    stop_times = pd.read_csv(gtfs_folder / "stop_times.txt", dtype=str)

    used_stop_ids = set(stop_times["stop_id"])
    unused = stops[~stops["stop_id"].isin(used_stop_ids)]

    print("Total stops:", len(stops))
    print("Used stops:", len(used_stop_ids))
    print("Unused stops:", len(unused))

    return unused

class OSM_TagCounter(osmium.SimpleHandler):
    """
    Counts the occurrences of 'highway' and 'railway' tags in OSM data and stores the results in two separate Counter objects.

    Usage: 
        - Create an instance of OSM_TagCounter and call apply_file() with the path to your OSM file. 
        - After processing, access the highway_counter and railway_counter attributes to see the counts.
    """
    def __init__(self):
        super().__init__()
        self.highway_counter = Counter()
        self.railway_counter = Counter()

    def way(self, w):
        tags = w.tags

        if 'highway' in tags:
            self.highway_counter[tags['highway']] += 1

        if 'railway' in tags:
            self.railway_counter[tags['railway']] += 1

if __name__ == "__main__":
    # ------ Handler for counting OSM tags (e.g. to understand which highway types are present in the area) ------

    # osm_file = f"{OSM_DIR}/locarno.osm.pbf"  # <-- change this

    # handler = OSM_TagCounter()
    # handler.apply_file(osm_file, locations=False)

    # print("\n=== HIGHWAY TAGS ===")
    # for tag, count in handler.highway_counter.most_common():
    #     print(f"{tag:20} {count}")

    # print("\n=== RAILWAY TAGS ===")
    # for tag, count in handler.railway_counter.most_common():
    #     print(f"{tag:20} {count}")

    # ------ Example usage of GTFS extraction and conversion functions ------

    # extract_gtfs_routes_through_area(
    #     gtfs_folder=f"{DATA_DIR}/gtfs/gtfs_swiss_20260422",
    #     boundary_shp=f"{DATA_DIR}/osm/shapes/Locarno_QGIS_2056.shp",
    #     output_folder=f"{DATA_DIR}/gtfs/locarno_20260422",
    # )

    gtfs_rough_routes_to_gpkg(
        gtfs_folder=f"{DATA_DIR}/interim/locarno_617/gtfs/locarno_617_clipped_gtfs",
        output_file=f"{DATA_DIR}/interim/locarno_617/gtfs/locarno_617_clipped_gtfs_routes.gpkg",
    )
    # gtfs_stops_to_gpkg(
    #     gtfs_folder=f"{DATA_DIR}/gtfs/locarno_20260422",
    #     output_file=f"{DATA_DIR}/osm/shapes/locarno_gtfs_stops.gpkg",
    # )

    # unused=find_unused_stops(gtfs_folder=f"{DATA_DIR}/gtfs/locarno_20260422",)

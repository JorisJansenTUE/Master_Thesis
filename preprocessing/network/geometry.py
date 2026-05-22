from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd
from shapely.ops import unary_union


def read_project_polygon_wgs84(shape_path: Path, buffer_degrees: float = 0.0):
    logging.info("Reading project shapefile: %s", shape_path)
    gdf = gpd.read_file(shape_path)

    if gdf.empty:
        raise ValueError(f"Project shapefile contains no features: {shape_path}")

    if gdf.crs is None:
        raise ValueError(
            f"Project shapefile has no CRS: {shape_path}. "
            f"Define the CRS before running the pipeline."
        )

    gdf_wgs84 = gdf.to_crs("EPSG:4326")
    polygon = unary_union(gdf_wgs84.geometry)

    if buffer_degrees != 0.0:
        polygon = polygon.buffer(buffer_degrees)

    if polygon.is_empty:
        raise ValueError("The merged project polygon is empty.")

    return polygon


def write_boundary_geojson(
    shape_path: Path,
    output_geojson: Path,
    buffer_degrees: float = 0.0,
) -> None:
    polygon = read_project_polygon_wgs84(shape_path, buffer_degrees)
    output_geojson.parent.mkdir(parents=True, exist_ok=True)

    boundary_gdf = gpd.GeoDataFrame(
        {"id": [1]},
        geometry=[polygon],
        crs="EPSG:4326",
    )
    boundary_gdf.to_file(output_geojson, driver="GeoJSON")

    logging.info("Wrote WGS84 boundary GeoJSON: %s", output_geojson)
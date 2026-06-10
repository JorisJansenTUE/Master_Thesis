from __future__ import annotations

import random

import geopandas as gpd
from shapely.geometry import Point
from shapely.geometry.base import BaseGeometry


def assign_municipality_to_point_cells(
    cells: gpd.GeoDataFrame,
    municipalities: gpd.GeoDataFrame,
    municipality_id_col: str,
) -> gpd.GeoDataFrame:
    """
    Assign each STATPOP/STATENT point to a municipality polygon.

    This assumes the STATPOP/STATENT coordinates represent hectare-cell centroids.
    """
    joined = gpd.sjoin(
        cells,
        municipalities[[municipality_id_col, "geometry"]],
        how="left",
        predicate="within",
    )

    joined = joined.drop(columns=["index_right"], errors="ignore")

    missing = joined[municipality_id_col].isna().sum()
    if missing > 0:
        print(
            f"Warning: {missing:,} cells were not assigned to a municipality. "
            "They will be ignored."
        )

    joined = joined.dropna(subset=[municipality_id_col]).copy()

    return gpd.GeoDataFrame(joined, geometry="geometry", crs=cells.crs)


def sample_point_from_geometry(geom: BaseGeometry, rng: random.Random) -> Point:
    """
    Sample a point from a geometry.

    For STATPOP/STATENT point geometries, this jitters the coordinate within
    an approximate 100 m x 100 m hectare cell.
    """
    if geom.is_empty:
        raise ValueError("Cannot sample from empty geometry.")

    if geom.geom_type == "Point":
        dx = rng.uniform(-50.0, 50.0)
        dy = rng.uniform(-50.0, 50.0)
        return Point(geom.x + dx, geom.y + dy)

    minx, miny, maxx, maxy = geom.bounds

    for _ in range(10_000):
        point = Point(
            rng.uniform(minx, maxx),
            rng.uniform(miny, maxy),
        )

        if geom.contains(point):
            return point

    return geom.representative_point()
"""Export spatial snapping coverage for GIS inspection.

This module only visualises static origin/job attachments. It does not change
snapping, modal networks, or accessibility routing. Coordinates are LV95.
"""
from __future__ import annotations

from pathlib import Path
from typing import Mapping

import geopandas as gpd
import pandas as pd

try:
    from .snapping import PointAttachments
except ImportError:
    from snapping import PointAttachments

CRS = "EPSG:2056"
MODES = ("car", "bike")


def _add_coverage_columns(
    frame: pd.DataFrame,
    *,
    id_column: str,
    attachments: Mapping[str, PointAttachments],
) -> pd.DataFrame:
    """Report attachment status and nearest-road distance for every point.

    Unlike the connector CSVs, distances are available also for points that
    exceeded the snapping threshold. They are straight-line distances to the
    closest network segment, not path lengths or walking times.
    """
    output = frame.copy()
    ids = output[id_column].astype(str)
    if ids.duplicated().any():
        raise ValueError(f"Duplicate IDs in {id_column}")
    for mode in MODES:
        if mode not in attachments:
            raise KeyError(f"Missing {mode} attachments")
        result = attachments[mode]
        if len(result.nearest_distances_m) != len(output):
            raise ValueError(
                f"{mode} nearest-distance count mismatch for {id_column}: "
                f"{len(result.nearest_distances_m)} vs {len(output)}"
            )
        if set(result.nearest_distances_m) != set(ids):
            raise ValueError(f"{mode} nearest-distance IDs do not match {id_column}")
        unsnapped = set(result.unsnapped_ids)
        output[f"{mode}_attached"] = (~ids.isin(unsnapped)).to_numpy()
        output[f"nearest_{mode}_m"] = ids.map(result.nearest_distances_m).astype(float).to_numpy()

    car = output["car_attached"]
    bike = output["bike_attached"]
    output["attachment_class"] = "neither"
    output.loc[car & bike, "attachment_class"] = "both"
    output.loc[car & ~bike, "attachment_class"] = "car_only"
    output.loc[~car & bike, "attachment_class"] = "bike_only"
    return output


def export_spatial_diagnostics(
    path: str | Path,
    *,
    origins: gpd.GeoDataFrame,
    jobs: pd.DataFrame,
    origin_attachments: Mapping[str, PointAttachments],
    job_attachments: Mapping[str, PointAttachments],
) -> Path:
    """Write one GeoPackage with all origin/job snapping diagnostics.

    Layers:
      hexagons: clipped origin polygons with car/bike attachment status.
      origin_points: origin representative points (where snapping occurs).
      employment: every STATENT cell, including rejected ones, with job counts.
      unattached_car_jobs: rejected car employment locations.
      unattached_bike_jobs: rejected bicycle employment locations.

    The GeoPackage is replaced on each invocation; other outputs are untouched.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if origins.crs is None:
        raise ValueError("Origins must have a declared CRS")
    cells = origins.to_crs(CRS)
    cells = gpd.GeoDataFrame(
        _add_coverage_columns(cells, id_column="hex_id", attachments=origin_attachments),
        geometry="geometry", crs=CRS,
    )

    origin_points = gpd.GeoDataFrame(
        cells.drop(columns="geometry").copy(),
        geometry=gpd.points_from_xy(cells["origin_x"], cells["origin_y"]),
        crs=CRS,
    )
    job_table = _add_coverage_columns(
        jobs, id_column="job_id", attachments=job_attachments,
    )
    employment = gpd.GeoDataFrame(
        job_table,
        geometry=gpd.points_from_xy(job_table["x"], job_table["y"]),
        crs=CRS,
    )

    if path.exists():
        path.unlink()
    layers = (
        ("hexagons", cells),
        ("origin_points", origin_points),
        ("employment", employment),
        ("unattached_car_jobs", employment.loc[~employment["car_attached"]]),
        ("unattached_bike_jobs", employment.loc[~employment["bike_attached"]]),
    )
    for layer_name, layer_data in layers:
        layer_data.to_file(path, layer=layer_name, driver="GPKG", mode="a")

    return path

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pandas as pd

from preprocessing.demand.config import RunnerConfig
from preprocessing.demand.spatial import assign_municipality_to_point_cells

from .data import InputData

def read_csv_as_points(
    path: Path,
    x_col: str,
    y_col: str,
    source_crs: str,
    target_crs: str,
    sep: str | None = None,
) -> gpd.GeoDataFrame:
    """
    Read a coordinate-based CSV file, such as STATPOP or STATENT,
    and convert it to a GeoDataFrame.

    The Swiss geodata usually uses LV95 coordinates, so EPSG:2056
    is normally the correct CRS.
    """
    if sep is None:
        df = pd.read_csv(path)
    else:
        df = pd.read_csv(path, sep=sep)

    missing = [column for column in [x_col, y_col] if column not in df.columns]
    if missing:
        raise ValueError(
            f"Missing coordinate columns {missing} in {path}. "
            f"Available columns are: {list(df.columns)}"
        )

    df[x_col] = pd.to_numeric(df[x_col], errors="coerce")
    df[y_col] = pd.to_numeric(df[y_col], errors="coerce")

    before = len(df)
    df = df.dropna(subset=[x_col, y_col]).copy()
    after = len(df)

    if after < before:
        print(
            f"Warning: dropped {before - after:,} rows from {path} "
            "because of missing coordinates."
        )

    geometry = gpd.points_from_xy(df[x_col], df[y_col])

    gdf = gpd.GeoDataFrame(
        df,
        geometry=geometry,
        crs=source_crs,
    )

    return gdf.to_crs(target_crs)


def read_municipalities(
    path: Path,
    layer: str | None,
    target_crs: str,
) -> gpd.GeoDataFrame:
    if layer:
        gdf = gpd.read_file(path, layer=layer)
    else:
        gdf = gpd.read_file(path)

    if gdf.crs is None:
        raise ValueError(f"{path} has no CRS. Please define it before running.")

    return gdf.to_crs(target_crs)


def read_project_boundary(
    path: Path,
    target_crs: str,
) -> gpd.GeoDataFrame:
    boundary = gpd.read_file(path)

    if boundary.crs is None:
        raise ValueError(f"{path} has no CRS. Please define it before running.")

    boundary = boundary.to_crs(target_crs)
    # Make sure it is a single dissolved geometry.
    boundary_geom = boundary.geometry.union_all()

    return gpd.GeoDataFrame(
        {"name": ["project_boundary"]},
        geometry=[boundary_geom],
        crs=boundary.crs,
    )


def clip_to_boundary(
    gdf: gpd.GeoDataFrame,
    boundary: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """
    Clip any GeoDataFrame to the project boundary.

    For point data, this keeps points inside the boundary.
    For polygon/line data, this geometrically clips features.
    """
    if gdf.crs != boundary.crs:
        boundary = boundary.to_crs(gdf.crs)

    clipped = gpd.clip(gdf, boundary)

    clipped = clipped.reset_index(drop=True)

    return clipped



def prepare_inputs(cfg: RunnerConfig) -> InputData:
    columns = cfg.columns
    muni_col = columns["municipality_id"]

    boundary = read_project_boundary(
        path=cfg.path("input", "project_boundary"),
        target_crs=cfg.crs,
    )

    municipalities = read_municipalities(
        path=cfg.path("input", "municipalities"),
        layer=cfg.raw.get("layers", {}).get("municipalities"),
        target_crs=cfg.crs,
    )

    municipalities = municipalities[municipalities.geometry.centroid.within(boundary.geometry.iloc[0])].copy()
    municipalities[muni_col] = municipalities[muni_col].astype(str)

    statpop = prepare_point_grid(
        cfg=cfg,
        file_key="statpop",
        x_col=columns["x_col"],
        y_col=columns["y_col"],
        value_col=columns["statpop_population"],
        municipalities=municipalities,
        boundary=boundary,
    )

    statent = prepare_point_grid(
        cfg=cfg,
        file_key="statent",
        x_col=columns["x_col"],
        y_col=columns["y_col"],
        value_col=columns["statent_jobs"],
        municipalities=municipalities,
        boundary=boundary,
    )

    od = prepare_od(
        cfg=cfg,
        project_municipality_ids=set(municipalities[muni_col]),
    )

    modal_splits = prepare_modal_splits(
        cfg=cfg,
        project_municipality_ids=set(municipalities[muni_col]),
    )

    input_data = InputData(
        statpop=statpop,
        statent=statent,
        municipalities=municipalities,
        od=od,
        modal_splits=modal_splits,
    )

    print_input_summary(input_data)
    return input_data


def prepare_point_grid(
    cfg: RunnerConfig,
    file_key: str,
    x_col: str,
    y_col: str,
    value_col: str,
    municipalities: gpd.GeoDataFrame,
    boundary: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    muni_col = cfg.columns["municipality_id"]

    grid = read_csv_as_points(
        path=cfg.path("input", file_key),
        x_col=x_col,
        y_col=y_col,
        source_crs=cfg.crs,
        target_crs=cfg.crs,
        sep=";"
    )

    grid = clip_to_boundary(grid, boundary)

    grid = assign_municipality_to_point_cells(
        cells=grid,
        municipalities=municipalities,
        municipality_id_col=muni_col,
    )

    grid[muni_col] = grid[muni_col].astype(str)

    grid[value_col] = pd.to_numeric(
        grid[value_col],
        errors="coerce",
    ).fillna(0.0)

    return grid


def prepare_od(
    cfg: RunnerConfig,
    project_municipality_ids: set[str],
) -> pd.DataFrame:
    columns = cfg.columns

    od = pd.read_csv(cfg.path("input", "commuter_od"))
    for col, value in cfg.raw.get("od_filters", {}).items():
        od = od[od[col].astype(str) == str(value)].copy()

    origin_col = columns["od_origin"]
    destination_col = columns["od_destination"]
    flow_col = columns["od_flow"]

    od[origin_col] = od[origin_col].astype(str)
    od[destination_col] = od[destination_col].astype(str)

    od[flow_col] = pd.to_numeric(
        od[flow_col],
        errors="coerce",
    ).fillna(0.0)

    policy = cfg.raw.get("demand", {}).get("od_boundary_policy", "internal_only")

    # Reads only OD pairs where both origin and destination are within the project municipalities.
    if policy == "internal_only":
        od = od[
            od[origin_col].isin(project_municipality_ids)
            & od[destination_col].isin(project_municipality_ids)
        ].copy()
    else:
        raise ValueError(
            f"Unsupported od_boundary_policy: {policy}. "
            "Currently supported: internal_only."
        )

    origin_col = cfg.columns["od_origin"]
    destination_col = cfg.columns["od_destination"]
    flow_col = cfg.columns["od_flow"]

    from itertools import product

    origin_col = cfg.columns["od_origin"]
    destination_col = cfg.columns["od_destination"]

    municipalities = sorted(project_municipality_ids)

    expected_pairs = set(product(municipalities, municipalities))

    actual_pairs = set(
        zip(
            od[origin_col].astype(str),
            od[destination_col].astype(str),
        )
    )

    missing_pairs = sorted(expected_pairs - actual_pairs)

    print("\nMissing OD pairs")
    print("----------------")
    print(f"Expected pairs: {len(expected_pairs)}")
    print(f"Actual pairs:   {len(actual_pairs)}")
    print(f"Missing pairs:  {len(missing_pairs)}")

    for origin, destination in missing_pairs:
        print(f"  {origin} -> {destination}")

    return od


def prepare_modal_splits(
    cfg: RunnerConfig,
    project_municipality_ids: set[str],
) -> pd.DataFrame | None:
    modal_path = cfg.raw["input"].get("modal_splits")

    if not modal_path:
        return None

    columns = cfg.columns
    muni_col = columns["modal_split_municipality"]

    modal_splits = pd.read_csv(cfg.path("input", "modal_splits"))
    modal_splits[muni_col] = modal_splits[muni_col].astype(str)

    return modal_splits[
        modal_splits[muni_col].isin(project_municipality_ids)
    ].copy()


def print_input_summary(input_data: InputData) -> None:
    print("\nInput summary")
    print("-------------")
    print(f"STATPOP cells: {len(input_data.statpop):,}")
    print(f"STATENT cells: {len(input_data.statent):,}")
    print(f"Municipalities: {len(input_data.municipalities):,}")
    print(f"OD rows: {len(input_data.od):,}")
    print(
        "Modal split table: "
        f"{'yes' if input_data.modal_splits is not None else 'no'}"
    )



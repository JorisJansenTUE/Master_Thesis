
from pathlib import Path
import shutil
import subprocess

import geopandas as gpd

TARGET_CRS = "EPSG:2056"
OSM_BOUNDARY_CRS = "EPSG:4326"


def require_executable(name: str) -> None:
    if shutil.which(name) is None:
        raise RuntimeError(
            f"Required executable '{name}' was not found on PATH."
        )

def read_project_boundary(
    shape_path: Path,
    target_crs: str,
    buffer_m: float = 0.0,
) -> gpd.GeoDataFrame:
    """
    Read, reproject, dissolve and optionally buffer the project boundary.
    """
    boundary = gpd.read_file(shape_path)

    if boundary.empty:
        raise ValueError(f"Project boundary is empty: {shape_path}")

    if boundary.crs is None:
        raise ValueError(
            f"Project boundary has no CRS: {shape_path}"
        )

    boundary = boundary.to_crs(target_crs)
    boundary = boundary[["geometry"]].dissolve()

    # Repair simple invalid geometries if necessary.
    if not boundary.geometry.is_valid.all():
        boundary["geometry"] = boundary.geometry.buffer(0)

    if buffer_m > 0:
        boundary["geometry"] = boundary.geometry.buffer(buffer_m)

    return boundary

def write_osmium_boundary_geojson(
    boundary: gpd.GeoDataFrame,
    output_geojson: Path,
) -> None:
    """
    Write the project boundary as WGS84 GeoJSON for osmium.
    """
    output_geojson.parent.mkdir(parents=True, exist_ok=True)

    boundary_wgs84 = boundary.to_crs(OSM_BOUNDARY_CRS)

    boundary_wgs84.to_file(
        output_geojson,
        driver="GeoJSON",
        index=False,
    )


def clip_osm_to_boundary(
    input_osm: Path,
    boundary_geojson: Path,
    output_osm: Path,
) -> None:
    """
    Clip an OSM file using osmium extract.
    """
    require_executable("osmium")

    if not input_osm.exists():
        raise FileNotFoundError(f"OSM input does not exist: {input_osm}")

    output_osm.parent.mkdir(parents=True, exist_ok=True)

    command = [
        "osmium",
        "extract",
        "--polygon",
        str(boundary_geojson),
        "--strategy",
        "smart",
        "--overwrite",
        "-o",
        str(output_osm),
        str(input_osm),
    ]

    print("\nClipping OSM data...")
    print(" ".join(command))

    subprocess.run(command, check=True)

    print(f"Saved clipped OSM to:\n{output_osm}")


def clip_cadastral_layers(
    input_gpkg: Path,
    boundary: gpd.GeoDataFrame,
    output_gpkg: Path,
    layers: list[str],
    target_crs: str,
) -> None:
    """
    Clip selected cadastral GeoPackage layers to the project boundary.
    """
    if not input_gpkg.exists():
        raise FileNotFoundError(
            f"Cadastral GeoPackage does not exist: {input_gpkg}"
        )

    output_gpkg.parent.mkdir(parents=True, exist_ok=True)

    if output_gpkg.exists():
        output_gpkg.unlink()

    bbox = tuple(boundary.total_bounds)
    written_layers = 0

    for layer in layers:
        print(f"\nProcessing cadastral layer: {layer}")

        # First restrict the read operation to the boundary bounding box.
        gdf = gpd.read_file(
            input_gpkg,
            layer=layer,
            bbox=bbox,
        )

        if gdf.empty:
            print(f"No features found for layer '{layer}'. Skipping.")
            continue

        if gdf.crs is None:
            print(f"Layer '{layer}' has no CRS. Assuming {target_crs}.")
            gdf = gdf.set_crs(target_crs)
        else:
            gdf = gdf.to_crs(target_crs)

        print(f"Features read from bounding box: {len(gdf)}")

        clipped = gpd.clip(gdf, boundary,keep_geom_type=True)

        clipped = clipped[
            clipped.geometry.notna()
            & ~clipped.geometry.is_empty
        ]

        if clipped.empty:
            print(f"No features remain after clipping layer '{layer}'. Skipping.")
            continue

        print(f"Features after exact clipping: {len(clipped)}")

        clipped.to_file(
            output_gpkg,
            layer=layer,
            driver="GPKG",
            index=False,
        )

        written_layers += 1

    if written_layers == 0:
        raise RuntimeError(
            "None of the requested cadastral layers produced output."
        )

    print(f"\nSaved clipped cadastral data to:\n{output_gpkg}")

"""One-off STATPOP 2024 100 m population grid -> fixed 500 m origin hexagons.

Uses the same EPSG:2056 / southwest-grid-corner +50 m interpretation as
accessibility.inputs.grid.employment. Only BBTOT (total residents) is used.
Does NOT modify existing spatial files, accessibilities, or GA scores.

Example from Master_Thesis root:
 python -m accessibility.inputs.grid.population \
   --statpop data/raw/Population-STATPOTP2024.zip \
   --hexagons data/accessibility/base_mapping/spatial_baseline/origins.gpkg \
   --out data/accessibility/base_mapping/spatial_baseline/population_by_hexagon.csv
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

import geopandas as gpd
import numpy as np
import pandas as pd

FIELDS = ("E_KOORD", "N_KOORD", "BBTOT")
LV95 = "EPSG:2056"


def _csv_stream(path: Path):
    if path.suffix.lower() == ".zip":
        archive = ZipFile(path)
        candidates = [name for name in archive.namelist() if Path(name).name.upper() == "STATPOP2024.CSV"]
        if len(candidates) != 1:
            archive.close()
            raise ValueError(f"Expected exactly one STATPOP2024.csv in {path}, got {candidates}")
        return archive, archive.open(candidates[0])
    if path.suffix.lower() == ".csv":
        return None, path.open("rb")
    raise ValueError("STATPOP source must be Population-STATPOTP2024.zip or STATPOP2024.csv")


def load_population_cells(path: str | Path, bounds: tuple[float, float, float, float],
                          *, chunksize: int = 100_000) -> tuple[pd.DataFrame, dict]:
    """Retain populated 100 m cells whose CENTRES fall in LV95 bounding box."""
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    xmin, ymin, xmax, ymax = map(float, bounds)
    if not np.isfinite([xmin, ymin, xmax, ymax]).all() or xmin >= xmax or ymin >= ymax:
        raise ValueError("Invalid hexagon bounds")
    archive, stream = _csv_stream(path)
    frames = []
    records_total = within_bounds = 0
    try:
        chunks = pd.read_csv(stream, sep=";", encoding="utf-8-sig", usecols=list(FIELDS),
                             chunksize=chunksize, low_memory=False)
        for chunk in chunks:
            records_total += len(chunk)
            x = pd.to_numeric(chunk["E_KOORD"], errors="coerce") + 50.0
            y = pd.to_numeric(chunk["N_KOORD"], errors="coerce") + 50.0
            pop = pd.to_numeric(chunk["BBTOT"], errors="coerce")
            nearby = x.between(xmin, xmax) & y.between(ymin, ymax)
            within_bounds += int(nearby.sum())
            if (nearby & (pop.isna() | (pop < 0))).any():
                raise ValueError("Invalid STATPOP BBTOT count inside study-area bounds")
            selected = nearby & pop.gt(0)
            if selected.any():
                frames.append(pd.DataFrame({"x": x[selected].astype(float),
                                            "y": y[selected].astype(float),
                                            "population": pop[selected].astype(float)}))
    finally:
        stream.close()
        if archive is not None:
            archive.close()
    points = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["x", "y", "population"])
    if points.duplicated(["x", "y"]).any():
        raise ValueError("STATPOP source contains duplicate grid cells")
    info = {"source": str(path), "grid_year": 2024, "crs": LV95,
            "source_population_field": "BBTOT", "records_read": records_total,
            "cells_within_bbox": within_bounds, "populated_cells_within_bbox": len(points)}
    return points, info


def aggregate_population(statpop: str | Path, origins_gpkg: str | Path, *,
                         layer: str | None = None) -> tuple[pd.DataFrame, dict]:
    """Assign STATPOP cell centres to clipped origin hexagons (point-in-polygon).

    This is exact assignment of grid CENTRES, not area-proportional redistribution.
    Population in cells whose centres fall outside clipped study-area polygons is
    omitted. Hexagons without populated centres have weight zero.
    """
    origins_path = Path(origins_gpkg).expanduser().resolve()
    polygons = gpd.read_file(origins_path, layer=layer)
    if "hex_id" not in polygons:
        raise ValueError(f"Missing hex_id in {origins_path}")
    if polygons.crs is None:
        raise ValueError("Hexagon GeoPackage is missing CRS")
    polygons = polygons.to_crs(LV95)
    polygons["hex_id"] = polygons["hex_id"].astype(str)
    if polygons["hex_id"].duplicated().any() or polygons.geometry.isna().any():
        raise ValueError("Duplicate hexagon IDs or missing geometry")
    if polygons.empty:
        raise ValueError("No origin hexagons")
    cells, info = load_population_cells(statpop, tuple(polygons.total_bounds))
    points = gpd.GeoDataFrame(cells, geometry=gpd.points_from_xy(cells.x, cells.y), crs=LV95)
    assigned = gpd.sjoin(points[["population", "geometry"]], polygons[["hex_id", "geometry"]],
                         how="left", predicate="within")
    if assigned.index.duplicated().any():
        raise ValueError("Overlapping hexagons: at least one STATPOP cell assigned to multiple origins")
    hits = assigned["hex_id"].notna()
    totals = assigned.loc[hits].groupby("hex_id", observed=True)["population"].agg(["sum", "size"])
    totals.columns = ["population", "statpop_cells"]
    result = polygons[["hex_id"]].merge(totals, left_on="hex_id", right_index=True,
                                       how="left", validate="one_to_one")
    result["population"] = result["population"].fillna(0.0).astype(float)
    result["statpop_cells"] = result["statpop_cells"].fillna(0).astype(int)
    result = result.sort_values("hex_id").reset_index(drop=True)
    info.update({"origin_geopackage": str(origins_path), "origin_hexagons": len(result),
                 "population_hexagons": int((result.population > 0).sum()),
                 "total_assigned_residents": float(result.population.sum()),
                 "within_bbox_residents": float(cells.population.sum()),
                 "cells_assigned": int(hits.sum()), "cells_outside_clipped_hexagons": int((~hits).sum()),
                 "assignment_policy": "100m_cell_centre_within_clipped_500m_hexagon"})
    return result, info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--statpop", required=True, type=Path)
    parser.add_argument("--hexagons", required=True, type=Path,
                        help="Fixed baseline origins.gpkg with hex_id polygons")
    parser.add_argument("--layer", help="GeoPackage layer name (default: first layer)")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    table, info = aggregate_population(args.statpop, args.hexagons, layer=args.layer)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False)
    info["statpop_sha256"] = hashlib.sha256(args.statpop.read_bytes()).hexdigest()
    info["origin_geopackage_sha256"] = hashlib.sha256(args.hexagons.read_bytes()).hexdigest()
    meta = args.out.with_suffix(".metadata.json")
    meta.write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {args.out}: {len(table)} hexagons, {info['total_assigned_residents']:,.0f} residents, "
          f"{info['population_hexagons']} populated hexagons", flush=True)
    print(f"Metadata: {meta}", flush=True)
    print(f"Unassigned 100m cell centres: {info['cells_outside_clipped_hexagons']}", flush=True)


if __name__ == "__main__":
    main()

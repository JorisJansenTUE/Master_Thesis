"""Prepare fixed hexagonal accessibility origins from a study-area polygon.

The class owns its spatial loading and hexagon construction. The resulting
origin points and clipped hexagons are independent of MATSim scenarios.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import cos, pi, sin, sqrt
from pathlib import Path

import geopandas as gpd
from shapely.geometry import Point, Polygon

TARGET_CRS = "EPSG:2056"  # Swiss LV95; metres


@dataclass
class HexagonOrigins:
    cells: gpd.GeoDataFrame
    diameter_m: float

    @classmethod
    def from_study_area(
        cls,
        path: str | Path,
        *,
        diameter_m: float = 500.0,
        min_coverage_fraction: float = 0.0,
    ) -> "HexagonOrigins":
        """Generate flat-top hexagons; diameter is vertex-to-vertex distance.

        Hexagons are clipped to the exact study-area geometry. Origins use the
        original centre if it lies inside the study area; for a boundary cell,
        a representative point within the clipped polygon is used instead.
        """
        if diameter_m <= 0:
            raise ValueError("diameter_m must be positive")
        if not 0 <= min_coverage_fraction < 1:
            raise ValueError("min_coverage_fraction must be in [0, 1)")
        source = Path(path).expanduser()
        if not source.is_file():
            raise FileNotFoundError(f"Study-area geometry not found: {source}")
        polygons = gpd.read_file(source)
        if polygons.empty:
            raise ValueError(f"Study area has no geometry: {source}")
        if polygons.crs is None:
            raise ValueError("Study-area CRS is missing; declare it before reprojecting")
        polygons = polygons.to_crs(TARGET_CRS)
        area = polygons.geometry.dropna().union_all()
        if area.is_empty:
            raise ValueError("Study area is empty after geometry union")
        if not area.is_valid:
            area = area.buffer(0)
        if area.is_empty:
            raise ValueError("Study area is invalid and could not be repaired")

        radius = diameter_m / 2
        step_x = 1.5 * radius
        step_y = sqrt(3) * radius
        minx, miny, maxx, maxy = area.bounds
        records: list[dict] = []
        # Extend one cell beyond each bound to cover all boundary polygons.
        col = 0
        cx = minx - diameter_m
        while cx <= maxx + diameter_m:
            row = 0
            cy = miny - step_y + (step_y / 2 if col % 2 else 0.0)
            while cy <= maxy + step_y:
                hexagon = Polygon([
                    (cx + radius * cos(pi * k / 3),
                     cy + radius * sin(pi * k / 3))
                    for k in range(6)
                ])
                if area.intersects(hexagon):
                    clipped = area.intersection(hexagon)
                    coverage = clipped.area / hexagon.area
                    if not clipped.is_empty and coverage > min_coverage_fraction:
                        original_centre = Point(cx, cy)
                        origin = (original_centre if area.covers(original_centre)
                                  else clipped.representative_point())
                        records.append({
                            "hex_id": f"H{col:04d}_{row:04d}",
                            "origin_x": float(origin.x),
                            "origin_y": float(origin.y),
                            "coverage_fraction": float(coverage),
                            "geometry": clipped,
                        })
                cy += step_y
                row += 1
            cx += step_x
            col += 1

        if not records:
            raise ValueError("No hexagons intersected the study area")
        cells = gpd.GeoDataFrame(records, geometry="geometry", crs=TARGET_CRS)
        return cls(cells=cells, diameter_m=float(diameter_m))

    def save(self, geopackage: str | Path, origins_csv: str | Path) -> None:
        """Export clipped hexagons for maps and simple origin points for routing."""
        geopackage, origins_csv = Path(geopackage), Path(origins_csv)
        geopackage.parent.mkdir(parents=True, exist_ok=True)
        origins_csv.parent.mkdir(parents=True, exist_ok=True)
        self.cells.to_file(geopackage, layer="origins", driver="GPKG")
        self.cells.drop(columns="geometry").to_csv(origins_csv, index=False)

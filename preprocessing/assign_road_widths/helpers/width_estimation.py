import gzip
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Tuple, List, Set

import geopandas as gpd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pyproj import Transformer
from shapely.geometry import LineString
from shapely.geometry.base import BaseGeometry

from preprocessing.assign_road_widths.width_config import WidthEstimationConfig



# =============================================================================
# WIDTH ESTIMATION
# =============================================================================

@dataclass
class WidthStats:
    median: float
    mean: float
    p10: float
    p90: float
    minimum: float
    maximum: float
    n: int


@dataclass
class WayWidthResult:
    way_id: str
    highway: Optional[str]
    name: Optional[str]
    carriageway: Optional[WidthStats]
    street_space: Optional[WidthStats]


class WidthEstimator:
    """
    Estimates road width by intersecting perpendicular transects with
    cadastral surface polygons.
    """

    def __init__(
        self,
        config: WidthEstimationConfig,
        carriageway_surface: BaseGeometry,
        street_space_surface: BaseGeometry,
    ):
        self.config = config
        self.carriageway_surface = carriageway_surface
        self.street_space_surface = street_space_surface

    def estimate_for_line(self, line: LineString, surface: BaseGeometry) -> Optional[WidthStats]:
        if line is None or line.is_empty or line.length < 2.0:
            return None

        distances = np.arange(
            self.config.sample_spacing_m / 2.0,
            line.length,
            self.config.sample_spacing_m,
        )

        widths = []

        for distance in distances:
            transect = self._make_transect(line, distance)

            if transect is None:
                continue

            intersection = transect.intersection(surface)
            width = self._intersection_length(intersection)

            if self.config.min_width_m <= width <= self.config.max_width_m:
                widths.append(width)

        if not widths:
            return None

        widths_array = np.array(widths)

        return WidthStats(
            median=float(np.median(widths_array)),
            mean=float(np.mean(widths_array)),
            p10=float(np.percentile(widths_array, 10)),
            p90=float(np.percentile(widths_array, 90)),
            minimum=float(np.min(widths_array)),
            maximum=float(np.max(widths_array)),
            n=int(len(widths_array)),
        )

    def estimate_for_way(
        self,
        way_id: str,
        tags: Dict[str, str],
        line: LineString,
    ) -> Optional[WayWidthResult]:

        carriageway_stats = self.estimate_for_line(
            line,
            self.carriageway_surface,
        )

        street_space_stats = self.estimate_for_line(
            line,
            self.street_space_surface,
        )

        if carriageway_stats is None and street_space_stats is None:
            return None

        return WayWidthResult(
            way_id=way_id,
            highway=tags.get("highway"),
            name=tags.get("name"),
            carriageway=carriageway_stats,
            street_space=street_space_stats,
        )

    def _make_transect(
        self,
        line: LineString,
        distance_along_line: float,
    ) -> Optional[LineString]:

        point = line.interpolate(distance_along_line)

        eps = min(0.5, line.length / 10.0)

        d1 = max(distance_along_line - eps, 0.0)
        d2 = min(distance_along_line + eps, line.length)

        p1 = line.interpolate(d1)
        p2 = line.interpolate(d2)

        dx = p2.x - p1.x
        dy = p2.y - p1.y

        norm = np.hypot(dx, dy)

        if norm == 0:
            return None

        ux = dx / norm
        uy = dy / norm

        px = -uy
        py = ux

        half = self.config.transect_length_m / 2.0

        start = (
            point.x - px * half,
            point.y - py * half,
        )

        end = (
            point.x + px * half,
            point.y + py * half,
        )

        return LineString([start, end])

    @staticmethod
    def _intersection_length(geom: BaseGeometry) -> float:
        if geom.is_empty:
            return 0.0

        if geom.geom_type in {"LineString", "MultiLineString"}:
            return geom.length

        if geom.geom_type == "GeometryCollection":
            return sum(
                part.length
                for part in geom.geoms
                if part.geom_type in {"LineString", "MultiLineString"}
            )

        return 0.0

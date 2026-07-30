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

        distances = self._get_sample_distances(line.length)

        widths = []

        for distance in distances:
            # This is the point where the transect crosses the OSM way.
            center_point = line.interpolate(distance)

            transect = self._make_transect(
                line,
                distance,
            )

            if transect is None:
                continue

            intersection = transect.intersection(surface)

            width = self._closest_intersection_length(
                intersection,
                center_point,
            )

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

        # we don't currently use street-space widths, but the code is left here for future use.
        # street_space_stats = self.estimate_for_line(
        #     line,
        #     self.street_space_surface,
        # )

        if carriageway_stats is None: #and street_space_stats is None:
            return None

        return WayWidthResult(
            way_id=way_id,
            highway=tags.get("highway"),
            name=tags.get("name"),
            carriageway=carriageway_stats,
            #street_space=street_space_stats,
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
    
    def _get_sample_distances(self, line_length: float) -> np.ndarray:
        """
        Generate transect locations along an OSM way.

        Longer ways use approximately the configured sampling spacing.
        Short ways receive at least a minimum number of samples.
        Samples are kept away from the way endpoints, where junction
        measurements are more likely.
        """
        spacing = self.config.sample_spacing_m
        min_samples = self.config.min_samples_per_way
        max_samples = self.config.max_samples_per_way

        if line_length < 2.0:
            return np.array([])

        # Normal fixed-spacing samples.
        distances = np.arange(
            spacing / 2.0,
            line_length,
            spacing,
        )

        # Long ways already have enough samples.
        if len(distances) >= min_samples:
            if len(distances) > max_samples:
                indices = np.linspace(
                    0,
                    len(distances) - 1,
                    max_samples,
                    dtype=int,
                )
                distances = distances[indices]

            return distances

        # Short ways: distribute a minimum number of samples over the
        # interior of the line rather than sampling near the endpoints.
        return np.linspace(
            line_length / (min_samples + 1),
            line_length * min_samples / (min_samples + 1),
            min_samples,
        )

    @classmethod
    def _closest_intersection_length(
        cls,
        geom: BaseGeometry,
        center_point: BaseGeometry,
        max_offset_m: float = 2.0,
    ) -> float:

        line_parts = cls._extract_line_parts(geom)

        if not line_parts:
            return 0.0

        closest_part = min(
            line_parts,
            key=lambda part: part.distance(center_point),
        )

        offset = closest_part.distance(center_point)

        if offset > max_offset_m:
            return 0.0

        return float(closest_part.length)
    
    @classmethod
    def _extract_line_parts(
        cls,
        geom: BaseGeometry,
    ) -> List[LineString]:
        """
        Recursively extract individual LineStrings from an intersection
        geometry.
        """
        if geom is None or geom.is_empty:
            return []

        if geom.geom_type == "LineString":
            return [geom]

        if geom.geom_type in {
            "MultiLineString",
            "GeometryCollection",
        }:
            line_parts: List[LineString] = []

            for part in geom.geoms:
                line_parts.extend(
                    cls._extract_line_parts(part)
                )

            return line_parts

        return []

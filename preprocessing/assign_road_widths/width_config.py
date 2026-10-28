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


# =============================================================================
# CONFIGURATION
# =============================================================================

FALLBACK_LANE_WIDTHS = { #These were measured using analyze_width_per_lane.py, the link roads were given identical values to the main, motorways were not measured)
    "residential": 2.80,
    "unclassified": 2.01,
    "tertiary": 3.50,
    "secondary": 3.69,
    "secondary_link": 3.69,
    "primary": 4.45,
    "primary_link": 4.45,
    "trunk": 4.90,
    "trunk_link": 4.90,
    "motorway": 4.90,
    "motorway_link":4.90,
    "living_street": 4.75,
}


@dataclass(frozen=True)
class WidthEstimationConfig:
    input_osm_gz: Path
    cadastral_gpkg: Path
    output_osm_gz: Path

    cadastral_layer: str = "lcsf"
    cadastral_class_field: str = "Art"

    source_crs: str = "EPSG:4326"   # OSM lon/lat
    target_crs: str = "EPSG:2056"   # Swiss LV95, metres

    min_samples_per_way: int = 2
    max_samples_per_way: int = 20
    sample_spacing_m: float = 10.0
    transect_length_m: float = 25.0

    min_width_m: float = 1.0
    max_width_m: float = 22.0

    large_width_threshold_m: float = 12.0
    small_width_threshold_m: float = 2.5

    main_road_classes: frozenset = frozenset({
        "motorway",
        "motorway_link",
        "trunk",
        "trunk_link",
        "primary",
        "primary_link",
        "secondary",
        "secondary_link",
        "tertiary",
        "unclassified",
        "residential",
        "living_street",
        "road",

    })

    service_values: frozenset = frozenset({
        "bus",
        "busway",
    })

    carriageway_classes: frozenset = frozenset({
        "Strasse_Weg",
    })

    street_space_classes: frozenset = frozenset({
        "Strasse_Weg",
        "Trottoir",
        "Verkehrsinsel",
        "uebrige_befestigte",
    })

    # Unreliability parameters
    default_lanes=2
    
    default_lane_width_m: float = 3.0

    max_unreliable_samples=5
    min_unreliable_spread_m=2

    @property
    def plot_dir(self) -> Path:
        return self.output_osm_gz.parent / "width_sanity_plots"
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

@dataclass(frozen=True)
class WidthEstimationConfig:
    input_osm_gz: Path
    cadastral_gpkg: Path
    output_osm_gz: Path

    cadastral_layer: str = "lcsf"
    cadastral_class_field: str = "Art"

    source_crs: str = "EPSG:4326"   # OSM lon/lat
    target_crs: str = "EPSG:2056"   # Swiss LV95, metres

    sample_spacing_m: float = 15.0
    transect_length_m: float = 40.0

    min_width_m: float = 1.0
    max_width_m: float = 25.0

    large_width_threshold_m: float = 12.0
    small_width_threshold_m: float = 2.5

    main_road_classes: frozenset = frozenset({
        "motorway",
        "trunk",
        "primary",
        "secondary",
        "tertiary",
        "unclassified",
        "residential",
        "living_street",
        "road",
    })

    service_values: frozenset = frozenset({
        "bus",
        "busway",
        "alley",
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

    @property
    def plot_dir(self) -> Path:
        return self.output_osm_gz.parent / "width_sanity_plots"
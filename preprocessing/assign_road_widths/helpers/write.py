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
from preprocessing.assign_road_widths.helpers.width_estimation import WidthStats, WayWidthResult


# =============================================================================
# OSM XML WRITING
# =============================================================================

class OsmXmlWriter:
    """
    Writes a new .osm.gz file with width tags added to ways.
    """

    def __init__(self, config: WidthEstimationConfig):
        self.config = config

    def write_with_widths(
        self,
        width_results: Dict[str, WayWidthResult],
    ) -> None:

        print("Writing output .osm.gz with width tags...")

        tree = ET.parse(gzip.open(self.config.input_osm_gz, "rb"))
        root = tree.getroot()

        for way in root.findall("way"):
            way_id = way.attrib.get("id")

            if way_id not in width_results:
                continue

            result = width_results.get(way_id)
            self._add_width_tags(way, result)

        with gzip.open(self.config.output_osm_gz, "wb") as file:
            tree.write(
                file,
                encoding="utf-8",
                xml_declaration=True,
            )

        print(f"Written output file:\n{self.config.output_osm_gz}")

    def _add_width_tags(
        self,
        way: ET.Element,
        result: WayWidthResult,
    ) -> None:

        if result.final_width_m is not None:
            self._add_or_replace_tag(
                way,
                "width:carriageway:estimated",
                f"{result.final_width_m:.2f}",
            )

        if result.carriageway is not None:
            self._add_stats_tags(
                way,
                prefix="width:carriageway",
                stats=result.carriageway,
            )
        else:
            self._add_or_replace_tag(
                way,
                "width:carriageway:n_samples",
                "0",
            )

        # if result.street_space is not None:
        #     self._add_stats_tags(
        #         way,
        #         prefix="width:street_space",
        #         stats=result.street_space,
        #     )

        self._add_or_replace_tag(
            way,
            "width:carriageway:method",
            result.method,
        )

        self._add_or_replace_tag(
            way,
            "width:carriageway:lanes_used",
            result.lanes_used,
        )
        
        

        self._add_or_replace_tag(
            way,
            "width:carriageway:optimizable",
            "yes" if result.optimizable else "no",
        )

    def _add_stats_tags(
        self,
        way: ET.Element,
        prefix: str,
        stats: WidthStats,
    ) -> None:

        self._add_or_replace_tag(
            way,
            f"{prefix}:raw",
            f"{stats.median:.2f}",
        )

        self._add_or_replace_tag(
            way,
            f"{prefix}:p10",
            f"{stats.p10:.2f}",
        )

        self._add_or_replace_tag(
            way,
            f"{prefix}:p90",
            f"{stats.p90:.2f}",
        )
        

        self._add_or_replace_tag(
            way,
            f"{prefix}:n_samples",
            stats.n,
        )

    @staticmethod
    def _add_or_replace_tag(
        way_elem: ET.Element,
        key: str,
        value,
    ) -> None:

        for tag in way_elem.findall("tag"):
            if tag.attrib.get("k") == key:
                tag.set("v", str(value))
                return

        new_tag = ET.Element("tag")
        new_tag.set("k", key)
        new_tag.set("v", str(value))
        way_elem.append(new_tag)

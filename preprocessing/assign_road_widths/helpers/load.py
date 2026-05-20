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
# CADASTRAL SURFACE LOADING
# =============================================================================

class CadastralSurfaceLoader:
    """
    Loads cadastral land-cover polygons and creates dissolved surface geometries.
    """

    def __init__(self, config: WidthEstimationConfig):
        self.config = config

    def load_surfaces(self) -> Tuple[BaseGeometry, BaseGeometry]:
        print("Reading cadastral polygons...")

        lcsf = gpd.read_file(
            self.config.cadastral_gpkg,
            layer=self.config.cadastral_layer,
        )

        if lcsf.crs is None:
            raise ValueError(
                "Cadastral layer has no CRS. Expected EPSG:2056."
            )

        lcsf = lcsf.to_crs(self.config.target_crs)

        class_field = self.config.cadastral_class_field

        if class_field not in lcsf.columns:
            raise ValueError(
                f"Column '{class_field}' not found in cadastral layer.\n"
                f"Available columns are:\n{list(lcsf.columns)}"
            )

        carriageway = lcsf[
            lcsf[class_field].isin(self.config.carriageway_classes)
        ].copy()

        street_space = lcsf[
            lcsf[class_field].isin(self.config.street_space_classes)
        ].copy()

        print(f"Carriageway polygons: {len(carriageway)}")
        print(f"Street-space polygons: {len(street_space)}")

        if carriageway.empty:
            raise ValueError(
                "No carriageway polygons found. Check cadastral_class_field "
                "and carriageway_classes."
            )

        if street_space.empty:
            raise ValueError(
                "No street-space polygons found. Check cadastral_class_field "
                "and street_space_classes."
            )

        print("Dissolving cadastral surfaces...")

        carriageway_union = carriageway.geometry.union_all()
        street_space_union = street_space.geometry.union_all()

        return carriageway_union, street_space_union


# =============================================================================
# OSM XML READING
# =============================================================================

class OsmXmlReader:
    """
    Reads nodes and ways from compressed OSM XML.
    """

    def __init__(self, osm_path: Path):
        self.osm_path = osm_path

    @staticmethod
    def parse_way_tags(way_elem: ET.Element) -> Dict[str, str]:
        tags = {}

        for tag in way_elem.findall("tag"):
            key = tag.attrib.get("k")
            value = tag.attrib.get("v")

            if key is not None and value is not None:
                tags[key] = value

        return tags

    def read_nodes(self) -> Dict[str, Tuple[float, float]]:
        """
        Reads all OSM nodes as lon/lat coordinates.
        Returns:
            node_id -> (lon, lat)
        """

        print("Reading OSM nodes...")

        nodes: Dict[str, Tuple[float, float]] = {}

        context = ET.iterparse(
            gzip.open(self.osm_path, "rb"),
            events=("end",),
        )

        for _, elem in context:
            if elem.tag == "node":
                node_id = elem.attrib.get("id")
                lat = elem.attrib.get("lat")
                lon = elem.attrib.get("lon")

                if node_id is not None and lat is not None and lon is not None:
                    nodes[node_id] = (float(lon), float(lat))

                elem.clear()

        print(f"Nodes read: {len(nodes)}")
        return nodes

    def iter_ways(self):
        """
        Streams OSM way elements from the .osm.gz file.
        """

        context = ET.iterparse(
            gzip.open(self.osm_path, "rb"),
            events=("end",),
        )

        for _, elem in context:
            if elem.tag == "way":
                yield elem
                elem.clear()


# =============================================================================
# OSM GEOMETRY BUILDER
# =============================================================================

class OsmGeometryBuilder:
    """
    Converts OSM ways into projected Shapely LineStrings.
    """

    def __init__(
        self,
        nodes: Dict[str, Tuple[float, float]],
        source_crs: str,
        target_crs: str,
    ):
        self.nodes = nodes
        self.transformer = Transformer.from_crs(
            source_crs,
            target_crs,
            always_xy=True,
        )

    def build_way_line(self, way_elem: ET.Element) -> Optional[LineString]:
        coords = []

        for nd in way_elem.findall("nd"):
            ref = nd.attrib.get("ref")

            if ref not in self.nodes:
                continue

            lon, lat = self.nodes[ref]
            x, y = self.transformer.transform(lon, lat)
            coords.append((x, y))

        if len(coords) < 2:
            return None

        return LineString(coords)
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

from preprocessing.assign_road_widths.helpers.load import CadastralSurfaceLoader, OsmXmlReader, OsmGeometryBuilder
from preprocessing.assign_road_widths.helpers.sanity_checks import SanityChecker
from preprocessing.assign_road_widths.helpers.write import OsmXmlWriter
from preprocessing.assign_road_widths.helpers.width_estimation import WidthStats, WayWidthResult, WidthEstimator

from preprocessing.utils import PROJECT_ROOT,DATA_DIR, OSM_DIR


# =============================================================================
# OSM TAG LOGIC
# =============================================================================

class OsmRoadFilter:
    """
    Decides which OSM ways should receive width estimates.
    """

    def __init__(self, config: WidthEstimationConfig):
        self.config = config

    def should_process(self, tags: Dict[str, str]) -> bool:
        highway = tags.get("highway")

        if highway in self.config.main_road_classes:
            return True

        if highway == "service" and self._is_correct_service(tags):
            return True

        return False

    def _is_correct_service(self, tags: Dict[str, str]) -> bool:

        service = tags.get("service")
        bus = tags.get("bus")
        psv = tags.get("psv")
        access = tags.get("access")

        if service in self.config.service_values:
            return True

        if bus in {"yes", "designated"}:
            return True

        if psv in {"yes", "designated"}:
            return True

        if access == "no" and (bus == "yes" or psv == "yes"):
            return True

        if tags.get("busway") is not None:
            return True

        if tags.get("bus:lanes") is not None:
            return True

        if tags.get("psv:lanes") is not None:
            return True

        return False


# =============================================================================
#  WORKFLOW SERVICE
# =============================================================================

class OsmWidthTaggingWorkflow:
    """
    Coordinates the full workflow.
    """

    def __init__(self, config: WidthEstimationConfig):
        self.config = config
        self.road_filter = OsmRoadFilter(config)
        self.osm_reader = OsmXmlReader(config.input_osm_gz)

    def run(self) -> None:
        carriageway_surface, street_space_surface = (
            CadastralSurfaceLoader(self.config).load_surfaces()
        )

        nodes = self.osm_reader.read_nodes()

        geometry_builder = OsmGeometryBuilder(
            nodes=nodes,
            source_crs=self.config.source_crs,
            target_crs=self.config.target_crs,
        )

        estimator = WidthEstimator(
            config=self.config,
            carriageway_surface=carriageway_surface,
            street_space_surface=street_space_surface,
        )

        expected_way_ids, width_results = self._estimate_widths(
            geometry_builder=geometry_builder,
            estimator=estimator,
        )

        SanityChecker(
            config=self.config,
            expected_way_ids=expected_way_ids,
            width_results=width_results,
        ).run()

        OsmXmlWriter(self.config).write_with_widths(width_results)

    def _estimate_widths(
        self,
        geometry_builder: OsmGeometryBuilder,
        estimator: WidthEstimator,
    ) -> Tuple[Set[str], Dict[str, WayWidthResult]]:

        print("Estimating widths for OSM ways...")

        expected_way_ids: Set[str] = set()
        width_results: Dict[str, WayWidthResult] = {}

        for way in self.osm_reader.iter_ways():
            way_id = way.attrib.get("id")

            if way_id is None:
                continue

            tags = self.osm_reader.parse_way_tags(way)

            if not self.road_filter.should_process(tags):
                continue

            expected_way_ids.add(way_id)

            line = geometry_builder.build_way_line(way)

            if line is None:
                continue

            result = estimator.estimate_for_way(
                way_id=way_id,
                tags=tags,
                line=line,
            )

            if result is not None:
                width_results[way_id] = result

        print(f"Relevant road ways: {len(expected_way_ids)}")
        print(f"Ways with width estimates: {len(width_results)}")

        return expected_way_ids, width_results



def main() -> None:
    config = WidthEstimationConfig(
        input_osm_gz=Path(
            f"{OSM_DIR}/locarno_smallest.osm.gz"
        ),
        cadastral_gpkg=Path(
            r"C:\Users\20201733\Documents\Thesis\Data\locarno_cadastral_clipped.gpkg"
        ),
        output_osm_gz=Path(
            f"{OSM_DIR}/locarno_smallest_with_widths.osm.gz"
        ),


        cadastral_class_field="Art",

        # Optional tuning.
        sample_spacing_m=15.0,
        transect_length_m=40.0,
        min_width_m=1.0,
        max_width_m=25.0,
    )

    workflow = OsmWidthTaggingWorkflow(config)
    workflow.run()


if __name__ == "__main__":
    main()
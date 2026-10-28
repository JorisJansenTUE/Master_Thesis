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

from preprocessing.assign_road_widths.helpers.clipping import TARGET_CRS, read_project_boundary, write_osmium_boundary_geojson, clip_osm_to_boundary, clip_cadastral_layers
from preprocessing.assign_road_widths.helpers.load import CadastralSurfaceLoader, OsmXmlReader, OsmGeometryBuilder
from preprocessing.assign_road_widths.helpers.sanity_checks import SanityChecker
from preprocessing.assign_road_widths.helpers.write import OsmXmlWriter
from preprocessing.assign_road_widths.helpers.width_estimation import WidthStats, WayWidthResult, WidthEstimator
from preprocessing.assign_road_widths.helpers.fix_errors import WidthResultFixer

from preprocessing.utils import PROJECT_ROOT,DATA_DIR, OSM_DIR, RAW_DIR


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
        self.width_postprocessor = WidthResultFixer(config)

    def run(self) -> None:
        carriageway_surface, _ = (              #previously also street_space_surface, but we don't currently use street-space widths
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

            result = self.width_postprocessor.process(
                result=result,
                tags=tags,
            )

            
            width_results[way_id] = result


        print(f"Relevant road ways: {len(expected_way_ids)}")
        print(f"Ways with width estimates: {len(width_results)}")

        return expected_way_ids, width_results


def main(clip=True) -> None:
    # -----------------------------------------------------------------
    # Inputs
    # -----------------------------------------------------------------

    project_shape = Path(
        f"{RAW_DIR}/shapes/REV_Cintura_project.shp"
    )

    input_osm = Path(
        f"{OSM_DIR}/Switzerland-260413.osm.pbf"
    )

    input_cadastral = Path(
        r"C:\Users\20201733\Documents\Thesis\Data\Cadastral_TI_gpkg_lv95\geopackage\av_TI_2056.gpkg"
    )
    # -----------------------------------------------------------------
    # Intermediates
    # -----------------------------------------------------------------
    boundary_geojson = Path(f"{RAW_DIR}/shapes/REV_boundary_buffered_2D.geojson")
    clipped_osm = Path(f"{OSM_DIR}/REV_network_buffered.osm.gz")
    clipped_cadastral = Path(f"{RAW_DIR}/cadastral/REV_cadastral_buffered.gpkg")
    output_osm = Path(f"{OSM_DIR}/REV_width_tagged_buffered/REV_with_widths_buffered.osm.gz")

    # -----------------------------------------------------------------
    # Clipping
    # -----------------------------------------------------------------
    if clip is True: 
        boundary = read_project_boundary(
            shape_path=project_shape,
            target_crs=TARGET_CRS,

            # Prevent edge effects during cadastral width estimation.
            buffer_m=8000,
        )
        if boundary_geojson.exists():
            print(f"Boundary GeoJSON already exists: {boundary_geojson}")

        else:
            write_osmium_boundary_geojson(
                boundary=boundary,
                output_geojson=boundary_geojson,
            )

        clip_osm_to_boundary(
            input_osm=input_osm,
            boundary_geojson=boundary_geojson,
            output_osm=clipped_osm,
        )

        clip_cadastral_layers(
            input_gpkg=input_cadastral,
            boundary=boundary,
            output_gpkg=clipped_cadastral,
            layers=[
                "lcsf",
                "soli",
            ],
            target_crs=TARGET_CRS,
        )



    config = WidthEstimationConfig(
        input_osm_gz=clipped_osm,
        cadastral_gpkg=clipped_cadastral,
        output_osm_gz=output_osm,


        cadastral_class_field="Art",

        # Optional tuning.
        sample_spacing_m=10.0,
        transect_length_m=25.0,
        min_width_m=1.0,
        max_width_m=22.0,
    )

    workflow = OsmWidthTaggingWorkflow(config)
    workflow.run()


if __name__ == "__main__":
    main(clip=False)  # Set to True to clip OSM and cadastral data to project boundary
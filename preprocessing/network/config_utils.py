from __future__ import annotations

import logging
from pathlib import Path

from lxml import etree

from preprocessing.network.config import PipelineConfig, PipelinePaths


def load_xml(path: Path) -> etree._ElementTree:
    parser = etree.XMLParser(remove_blank_text=False)
    return etree.parse(str(path), parser)


def write_xml(tree: etree._ElementTree, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(
        str(output_path),
        pretty_print=True,
        xml_declaration=True,
        encoding="UTF-8",
    )


def set_param(tree: etree._ElementTree, name: str, value: str) -> None:
    matches = tree.xpath(f"//param[@name='{name}']")

    if not matches:
        logging.warning("Config parameter not found and therefore not set: %s", name)
        return

    for param in matches:
        param.attrib["value"] = value
        logging.info("Set config param %s = %s", name, value)


def apply_overrides(tree: etree._ElementTree, overrides: dict[str, str]) -> None:
    for key, value in overrides.items():
        set_param(tree, key, value)


def generate_osm_config(config: PipelineConfig, paths: PipelinePaths) -> None:
    """
    Generate OsmConverter config from the tested template.

    The template remains the source of truth for modelling settings.
    The pipeline only changes file paths and output CRS.
    """
    tree = load_xml(config.osm_config_template)
    if config.process_bike_tags:
        pipeline_managed_params = {
        "osmFile": paths.processed_tags_osm.resolve(),
        "outputNetworkFile": paths.unmapped_network.resolve(),
        "outputDetailedLinkGeometryFile": paths.unmapped_detailed_link_geometry.resolve(),
        "outputCoordinateSystem": config.target_crs,
        }
           
    else:
        pipeline_managed_params = {
        "osmFile": paths.clipped_osm.resolve(),
        "outputNetworkFile": paths.unmapped_network.resolve(),
        "outputDetailedLinkGeometryFile": paths.unmapped_detailed_link_geometry.resolve(),
        "outputCoordinateSystem": config.target_crs,
    }

    for key, value in pipeline_managed_params.items():
        set_param(tree, key, str(value))

    apply_overrides(tree, config.osm_config_overrides)
    write_xml(tree, paths.osm_config)


def generate_mapper_config(config: PipelineConfig, paths: PipelinePaths) -> None:
    """
    Generate PublicTransitMapper config from the tested template.

    The exact parameter names depend on your mapper XML. The common PT2MATSim
    names are covered here. Missing names only produce warnings.
    """
    tree = load_xml(config.mapper_config_template)

    pipeline_managed_params = {
        "inputNetworkFile": paths.unmapped_network.resolve(),
        "inputScheduleFile": paths.unmapped_schedule.resolve(),
        "outputNetworkFile": paths.mapped_network.resolve(),
        "outputScheduleFile": paths.mapped_schedule.resolve(),
    }

    for key, value in pipeline_managed_params.items():
        set_param(tree, key, str(value))

    apply_overrides(tree, config.mapper_config_overrides)
    write_xml(tree, paths.mapper_config)

    
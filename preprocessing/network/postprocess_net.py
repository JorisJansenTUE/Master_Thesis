from __future__ import annotations

import gzip
import logging
import tempfile
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterable, Optional

from preprocessing.network.common import create_parent_dirs, run_command
from preprocessing.network.config import PipelineConfig, PipelinePaths

BIKE_MODE = "bike"
HIGHWAY_ATTRIBUTE = "osm:way:highway"

TARGET_HIGHWAY_TYPES = {
    "primary",
    "primary_link",
    "secondary",
    "secondary_link",
}

BIKE_SPEED_THRESHOLD_KMH = 50
BIKE_CLEANER_MAIN_CLASS = "thesis.network.CleanModeSubnetwork"

def read_matsim_network_xml(path: Path) -> ET.ElementTree:
    logging.info("Reading MATSim network: %s", path)

    if path.suffix == ".gz":
        with gzip.open(path, "rb") as file:
            return ET.parse(file)

    return ET.parse(path)


def write_matsim_network_xml(tree: ET.ElementTree, path: Path) -> None:
    logging.info("Writing MATSim network: %s", path)
    path.parent.mkdir(parents=True, exist_ok=True)

    xml_body = ET.tostring(
        tree.getroot(),
        encoding="unicode",
        short_empty_elements=True,
    )

    xml_text = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<!DOCTYPE network SYSTEM "http://www.matsim.org/files/dtd/network_v2.dtd">\n'
        f"{xml_body}\n"
    )

    if path.suffix == ".gz":
        with gzip.open(path, "wt", encoding="utf-8", newline="\n") as file:
            file.write(xml_text)
    else:
        with path.open("w", encoding="utf-8", newline="\n") as file:
            file.write(xml_text)


def get_links_element(root: ET.Element) -> ET.Element:
    links = root.find("links")

    if links is None:
        raise ValueError("Could not find <links> element in MATSim network.")

    return links


def get_modes(link: ET.Element) -> set[str]:
    modes_raw = link.get("modes", "")

    return {
        mode.strip()
        for mode in modes_raw.split(",")
        if mode.strip()
    }


def set_modes(link: ET.Element, modes: Iterable[str]) -> None:
    link.set("modes", ",".join(sorted(set(modes))))


def get_link_attribute(link: ET.Element, attribute_name: str) -> Optional[str]:
    attributes = link.find("attributes")

    if attributes is None:
        return None

    for attribute in attributes.findall("attribute"):
        if attribute.get("name") == attribute_name:
            return attribute.text

    return None

def should_keep_bike_as_exception(link: ET.Element) -> bool:

    relation_route = get_link_attribute(link, "osm:relation:route")

    if relation_route is not None:
        route_values = {
            value.strip().lower()
            for value in relation_route.split(",")
            if value.strip()
        }

        if {"bicycle", "mtb"} & route_values:
            return True

    return False


def remove_bike_from_fast_primary_secondary_links(
    input_network: Path,
    output_network: Path,
    speed_threshold_kmh: float = 50.0,
    bike_mode: str = BIKE_MODE,
) -> None:
    """
    Remove bicycle access from primary/secondary MATSim links above a speed threshold.

    This does not clean the network. It only modifies the allowed modes.
    Run the Java MultimodalNetworkCleaner afterwards.
    """
    tree = read_matsim_network_xml(input_network)
    root = tree.getroot()
    links = get_links_element(root)

    checked_target_links = 0
    fast_target_links = 0
    removed_bike_links = 0
    exception_link = 0
    already_without_bike = 0
    missing_highway_attribute = 0

    for link in links.findall("link"):
        highway_type = get_link_attribute(link, HIGHWAY_ATTRIBUTE)

        if highway_type is None:
            missing_highway_attribute += 1
            continue

        if highway_type not in TARGET_HIGHWAY_TYPES:
            continue

        checked_target_links += 1

        freespeed_raw = link.get("freespeed")

        if freespeed_raw is None:
            raise ValueError(f"Link {link.get('id')} has no freespeed attribute.")

        freespeed_kmh = float(freespeed_raw) * 3.6

        if freespeed_kmh <= speed_threshold_kmh:
            continue
        
        fast_target_links += 1

        if should_keep_bike_as_exception(link):
            logging.info("Keeping bike on exception link %s, due to MTB exception, highway=%s, freespeed=%.1f km/h", link.get("id"), highway_type, freespeed_kmh)
            exception_link+=1
            continue

        modes = get_modes(link)

        if bike_mode not in modes:
            already_without_bike += 1
            continue

        modes.remove(bike_mode)
        set_modes(link, modes)
        removed_bike_links += 1

    if checked_target_links == 0:
        raise ValueError(
            "No primary/secondary links were found. "
            f"Expected PT2MATSim link attribute '{HIGHWAY_ATTRIBUTE}'. "
            "Check that keepTagsAsAttributes=true in the OsmConverter config."
        )

    logging.info("Bike access removal from fast primary/secondary links:")
    logging.info("  Checked primary/secondary links: %s", checked_target_links)
    logging.info("  Fast primary/secondary links:    %s", fast_target_links)
    logging.info("  Removed bike from links:         %s", removed_bike_links)
    logging.info("  Exception links:                 %s", exception_link)
    logging.info("  Already without bike:            %s", already_without_bike)
    logging.info("  Missing highway attribute:       %s", missing_highway_attribute)

    write_matsim_network_xml(tree, output_network)

def apply_bike_speed_heuristic_and_clean(
    config: PipelineConfig,
    paths: PipelinePaths,
) -> None:    
    """
    Remove bicycle access from fast primary/secondary links and clean the bike network.

    This function does not expose intermediate files in PipelinePaths.
    It safely replaces paths.mapped_network only after the heuristic and cleaner succeed.

    Expected flow:
        paths.mapped_network
            -> temporary bike-access-modified network
            -> temporary bike-cleaned network
            -> replace paths.mapped_network
    """
    input_network = paths.unmapped_network

    if not input_network.exists():
        raise FileNotFoundError(f"Mapped network does not exist: {input_network}")

    input_network.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(
        prefix="bike_speed_heuristic_",
        dir=input_network.parent,
    ) as tmp_dir:
        tmp_dir_path = Path(tmp_dir)

        modified_network = tmp_dir_path / "network_bike_access_modified.xml.gz"
        cleaned_network = tmp_dir_path / "network_bike_cleaned.xml.gz"

        logging.info("Applying bicycle speed heuristic to mapped network: %s", input_network)

        remove_bike_from_fast_primary_secondary_links(
            input_network=input_network,
            output_network=modified_network,
            speed_threshold_kmh=getattr(
                config,
                "bike_speed_threshold_kmh",
                BIKE_SPEED_THRESHOLD_KMH,
            ),
            bike_mode=getattr(
                config,
                "bike_mode",
                BIKE_MODE,
            ),
        )

        logging.info("Cleaning bicycle network with Java cleaner.")

        run_command(
            [
                config.java_runner,
                "exec:java",
                f"-Dexec.mainClass={getattr(config, 'bike_cleaner_main_class', BIKE_CLEANER_MAIN_CLASS)}",
                "-Dexec.args="
                f"{modified_network} "
                f"{cleaned_network} "
                f"{getattr(config, 'bike_mode', BIKE_MODE)}",
            ],
            cwd=config.matsim_project_dir,
        )

        if not cleaned_network.exists():
            raise FileNotFoundError(
                "Java bicycle cleaner finished, but did not create output network: "
                f"{cleaned_network}"
            )

        shutil.move(cleaned_network, input_network)

    logging.info("Updated mapped network with bike-speed-cleaned network: %s", input_network)
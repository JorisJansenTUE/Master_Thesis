from __future__ import annotations

import gzip
from pathlib import Path

from lxml import etree


def apply_bus_lane_replacements(
    input_schedule: Path,
    output_schedule: Path,
    bus_link_replacements: dict[str, str],
) -> None:
    """
    Replace general road links by dedicated _spec links in scheduled bus routes.

    Parameters
    ----------
    input_schedule
        Baseline MATSim transitSchedule.xml or transitSchedule.xml.gz.

    output_schedule
        Path for the modified transit schedule.

    bus_link_replacements
        Mapping from original MATSim link IDs to newly created dedicated
        bus-lane link IDs.
    """

    tree = _read_xml(input_schedule)

    replaced_routes = 0
    replaced_links = 0
    used_replacements: set[str] = set()

    for transit_route in tree.iterfind(".//transitRoute"):
        mode = transit_route.findtext("transportMode")

        if mode != "bus":
            continue

        route = transit_route.find("route")

        if route is None:
            continue

        route_changed = False

        for link_ref in route.iterfind("link"):
            old_link_id = link_ref.get("refId")

            if old_link_id not in bus_link_replacements:
                continue

            link_ref.set("refId", bus_link_replacements[old_link_id])
            used_replacements.add(old_link_id)
            replaced_links += 1
            route_changed = True

        if route_changed:
            replaced_routes += 1

    # Update stop facilities on links replaced by dedicated bus lanes.
    replaced_stops = 0

    for stop_facility in tree.iterfind(".//transitStops/stopFacility"):
        old_link_id = stop_facility.get("linkRefId")

        if old_link_id in used_replacements:
            stop_facility.set(
                "linkRefId",
                bus_link_replacements[old_link_id],
            )
            replaced_stops += 1

    missing = set(bus_link_replacements) - used_replacements

    if missing:
        raise ValueError(
            "Dedicated bus-lane replacements were not found in any scheduled "
            f"bus route: {sorted(missing)}"
        )

    _write_xml(tree, output_schedule)

    print(f"Bus routes modified: {replaced_routes:,}")
    print(f"Route link references replaced: {replaced_links:,}")
    print(f"Stop facilities updated: {replaced_stops:,}")


def _read_xml(path: Path) -> etree._ElementTree:
    """Read an XML or XML.GZ file."""

    parser = etree.XMLParser(remove_blank_text=False)

    if path.suffix == ".gz":
        with gzip.open(path, "rb") as file:
            return etree.parse(file, parser)

    return etree.parse(str(path), parser)


def _write_xml(tree: etree._ElementTree, path: Path) -> None:
    """Write an XML or XML.GZ file while preserving the document type."""

    doctype = tree.docinfo.doctype or None

    if path.suffix == ".gz":
        with gzip.open(path, "wb") as file:
            tree.write(
                file,
                encoding="UTF-8",
                xml_declaration=True,
                doctype=doctype,
                pretty_print=False,
            )
    else:
        tree.write(
            str(path),
            encoding="UTF-8",
            xml_declaration=True,
            doctype=doctype,
            pretty_print=False,
        )
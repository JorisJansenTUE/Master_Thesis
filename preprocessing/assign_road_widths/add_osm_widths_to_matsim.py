"""Transfer OSM carriageway widths and road classifications to MATSim links.

Example (run from repository root):
  python -m preprocessing.assign_road_widths.add_osm_widths_to_matsim \
      --osm data/raw/osm/REV_with_widths.osm.gz \
      --network data/processed/locarno_network.xml.gz \
      --output data/processed/locarno_network_with_widths.xml.gz

Attributes assigned:
  carriageway_width_m: java.lang.Double, allocated by MATSim/OSM lane ratio
  cyclestreet: java.lang.Boolean, initially false
  destination_road: java.lang.Boolean, inferred from OSM access tags and name overrides

Name overrides apply to ALL OSM ways sharing that exact name (case/whitespace
insensitive). Use with care for streets with different access restrictions along
individual segments. If needed, use OSM IDs for segment-specific exceptions.
"""

import argparse
import gzip
import math
from dataclasses import dataclass
from pathlib import Path
from lxml import etree

# Street-name overrides: fill in names verified manually on the map.
# False = not destination-only; True = destination-only.
# Example: "Via Example": False,
DESTINATION_ROAD_NAME_OVERRIDES: dict[str, bool] = {"Via alle Vigne": False, "Strada Collina": False, "Via delle Querce": False, "Via alla Vignascia": False}

# More specific motor-vehicle tags take priority over general access tags.
ACCESS_TAG_PRIORITY = ("motorcar", "motor_vehicle", "vehicle", "access")


@dataclass(frozen=True)
class OsmRoadData:
    width_m: float | None
    lanes_used: int | None
    destination_road: bool


def open_binary(path: Path, mode: str):
    return gzip.open(path, mode) if path.suffix == ".gz" else open(path, mode)


def normalize_name(name: str) -> str:
    return " ".join(name.casefold().split())


def is_destination_road(tags: dict[str, str], overrides: dict[str, bool]) -> bool:
    name = normalize_name(tags.get("name", ""))
    if name in overrides:
        return overrides[name]

    # A more specific restriction/permission overrides a less specific one.
    for key in ACCESS_TAG_PRIORITY:
        value = tags.get(key)
        if value is not None:
            return value.strip().casefold() == "destination"
    return False


def load_osm_roads(path: Path) -> dict[str, OsmRoadData]:
    """Map OSM way IDs to width, lanes_used, and destination-road classification."""
    roads = {}
    overrides = {normalize_name(k): v for k, v in DESTINATION_ROAD_NAME_OVERRIDES.items()}
    override_matches = {name: 0 for name in overrides}

    with open_binary(path, "rb") as source:
        for _, way in etree.iterparse(source, events=("end",), tag="way"):
            tags = {tag.get("k"): tag.get("v") for tag in way.iterchildren("tag")}
            way_id = way.get("id")
            if way_id is not None:
                name = normalize_name(tags.get("name", ""))
                if name in override_matches:
                    override_matches[name] += 1

                width_m = None
                lanes_used = None
                try:
                    candidate_width = float(tags.get("width:carriageway:estimated"))
                    candidate_lanes = int(tags.get("width:carriageway:lanes_used"))
                    if math.isfinite(candidate_width) and candidate_width > 0 and candidate_lanes > 0:
                        width_m = candidate_width
                        lanes_used = candidate_lanes
                except (TypeError, ValueError):
                    pass

                roads[way_id] = OsmRoadData(
                    width_m=width_m,
                    lanes_used=lanes_used,
                    destination_road=is_destination_road(tags, overrides),
                )

            way.clear()
            while way.getprevious() is not None:
                del way.getparent()[0]

    for name, count in override_matches.items():
        print(f"Name override '{name}': {count} OSM way(s) matched")
        if count == 0:
            print("  WARNING: No matching OSM way; check spelling and source OSM coverage.")
    return roads


def set_attribute(attributes, name: str, value: str, java_class: str) -> None:
    for old in attributes.findall("attribute"):
        if old.get("name") == name:
            attributes.remove(old)
    new = etree.SubElement(attributes, "attribute", name=name)
    new.set("class", java_class)
    new.text = value


def add_widths(
    network: Path,
    output: Path,
    roads: dict[str, OsmRoadData],
    osm_id_attribute: str,
    width_attribute: str,
) -> None:
    if network.resolve() == output.resolve():
        raise ValueError("Output must be different from input (to protect original).")

    with open_binary(network, "rb") as source:
        tree = etree.parse(source, parser=etree.XMLParser(huge_tree=True))

    matched = missing_id = missing_osm_way = missing_width = 0
    destination_count = 0
    total = 0

    for link in tree.iterfind(".//links/link"):
        total += 1
        attributes = link.find("attributes")
        osm_id = None
        if attributes is not None:
            for attr in attributes.findall("attribute"):
                if attr.get("name") == osm_id_attribute:
                    osm_id = (attr.text or "").strip()
                    break

        road = roads.get(osm_id) if osm_id else None
        if not osm_id:
            missing_id += 1
        elif road is None:
            missing_osm_way += 1

        if attributes is None:
            attributes = etree.SubElement(link, "attributes")

        # These flags are defined on ALL MATSim links, even without a matched OSM way, adds cyclestreet and destination_road attributes to all links which are then used by router
        set_attribute(attributes, "cyclestreet", "false", "java.lang.Boolean")
        destination = road.destination_road if road is not None else False
        set_attribute(attributes, "destination_road", str(destination).lower(), "java.lang.Boolean")
        destination_count += int(destination)

        if road is None or road.width_m is None or road.lanes_used is None:
            # Avoid leaving a stale width attribute from a previous run.
            for old in attributes.findall("attribute"):
                if old.get("name") == width_attribute:
                    attributes.remove(old)
            if road is not None:
                missing_width += 1
            continue

        matsim_lanes = float(link.get("permlanes"))
        link_width = road.width_m * matsim_lanes / road.lanes_used
        set_attribute(attributes, width_attribute, str(link_width), "java.lang.Double")
        matched += 1

    output.parent.mkdir(parents=True, exist_ok=True)
    with open_binary(output, "wb") as dest:
        tree.write(dest, encoding="UTF-8", xml_declaration=True,
                   doctype=tree.docinfo.doctype or None)

    print(f"OSM ways read: {len(roads):,}")
    print(f"MATSim links processed: {total:,}")
    print(f"MATSim links with widths: {matched:,}")
    print(f"MATSim links marked destination_road=true: {destination_count:,}")
    print(f"Links without '{osm_id_attribute}': {missing_id:,}")
    print(f"Links with OSM ID absent from source OSM: {missing_osm_way:,}")
    print(f"Links with matched OSM way but no usable width: {missing_width:,}")
    print(f"Wrote {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--osm", type=Path, required=True,
                        help="Width-tagged .osm or .osm.gz")
    parser.add_argument("--network", type=Path, required=True,
                        help="Input MATSim network .xml or .xml.gz")
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output MATSim network .xml or .xml.gz",
    )
    parser.add_argument(
        "--osm-id-attribute",
        default="osm:way:id",
        help="MATSim link attribute containing the OSM way ID",
    )
    parser.add_argument(
        "--width-attribute",
        default="carriageway_width_m",
        help="MATSim link attribute for estimated carriageway width",
    )

    args = parser.parse_args()

    print(f"Reading OSM widths from: {args.osm}", flush=True)
    roads = load_osm_roads(args.osm)

    print(f"Reading MATSim network: {args.network}", flush=True)
    add_widths(
        network=args.network,
        output=args.output,
        roads=roads,
        osm_id_attribute=args.osm_id_attribute,
        width_attribute=args.width_attribute,
    )


if __name__ == "__main__":
    main()
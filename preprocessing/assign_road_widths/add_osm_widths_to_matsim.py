"""Copy final carriageway widths from OSM XML onto MATSim network links.

Example:
  python add_osm_widths_to_matsim.py --osm REV_with_widths.osm.gz \
      --network locarno_network.xml.gz --output locarno_network_with_widths.xml.gz
"""

import argparse
import gzip
from pathlib import Path
from lxml import etree


def open_binary(path: Path, mode: str):
    return gzip.open(path, mode) if path.suffix == ".gz" else open(path, mode)


def load_osm_widths(path: Path) -> dict[str, float]:
    widths = {}
    with open_binary(path, "rb") as source:
        for _, way in etree.iterparse(source, events=("end",), tag="way"):
            tags = {
                tag.get("k"): tag.get("v")
                for tag in way.iterchildren("tag")
            }
            raw = tags.get("width:carriageway:estimated")
            if raw is not None:
                try:
                    value = float(raw)
                    if value > 0:
                        widths[way.get("id")] = value
                except (TypeError, ValueError):
                    pass
            way.clear()
            while way.getprevious() is not None:
                del way.getparent()[0]
    return widths


def add_widths(network: Path, output: Path, widths: dict[str, float],
               osm_id_attribute: str, width_attribute: str) -> None:
    with open_binary(network, "rb") as f:
        tree = etree.parse(f, parser=etree.XMLParser(huge_tree=True))

    matched = missing_id = missing_width = 0
    for link in tree.iterfind(".//links/link"):
        attributes = link.find("attributes")
        osm_id = None
        if attributes is not None:
            for attr in attributes.findall("attribute"):
                if attr.get("name") == osm_id_attribute:
                    osm_id = (attr.text or "").strip()
                    break

        if not osm_id:
            missing_id += 1
            continue
        value = widths.get(osm_id)
        if value is None:
            missing_width += 1
            continue

        if attributes is None:
            attributes = etree.SubElement(link, "attributes")
        old = [attr for attr in attributes.findall("attribute")
               if attr.get("name") == width_attribute]
        for attr in old:
            attributes.remove(attr)
        new = etree.SubElement(attributes, "attribute", name=width_attribute)
        new.set("class", "java.lang.Double")
        new.text = str(value)
        matched += 1

    output.parent.mkdir(parents=True, exist_ok=True)
    if network.resolve() == output.resolve():
        raise ValueError("Output must be different from input (to protect original).")
    with open_binary(output, "wb") as f:
        tree.write(f, encoding="UTF-8", xml_declaration=True,
                   doctype=tree.docinfo.doctype or None)

    print(f"OSM ways with widths: {len(widths):,}")
    print(f"MATSim links updated: {matched:,}")
    print(f"Links without '{osm_id_attribute}': {missing_id:,}")
    print(f"Links with OSM ID but no matching width: {missing_width:,}")
    print(f"Wrote {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--osm", type=Path, required=True,
                        help="Width-tagged .osm or .osm.gz")
    parser.add_argument("--network", type=Path, required=True,
                        help="Input MATSim network .xml or .xml.gz")
    parser.add_argument("--output", type=Path, required=True,
                        help="Output MATSim network .xml or .xml.gz")
    parser.add_argument("--osm-id-attribute", default="osm:way:id",
                        help="Existing MATSim link attribute containing OSM way ID")
    parser.add_argument("--width-attribute", default="carriageway_width_m",
                        help="Name of the added MATSim link attribute")
    args = parser.parse_args()
    widths = load_osm_widths(args.osm)
    add_widths(args.network, args.output, widths,
               args.osm_id_attribute, args.width_attribute)


if __name__ == "__main__":
    main()

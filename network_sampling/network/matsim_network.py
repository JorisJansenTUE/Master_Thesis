from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import gzip

from lxml import etree


# ---------------------------------------------------------------------
# Domain objects
# ---------------------------------------------------------------------

@dataclass
class MatsimNode:
    id: str
    x: float
    y: float

    attributes: dict[str, Any] = field(default_factory=dict)
    xml_attributes: dict[str, str] = field(default_factory=dict)


@dataclass
class MatsimLink:
    id: str
    from_node: str
    to_node: str

    length: float
    freespeed: float
    capacity: float
    permlanes: float

    allowed_modes: set[str]

    # MATSim <attributes>...</attributes>
    attributes: dict[str, Any] = field(default_factory=dict)

    # Any additional XML attributes on <link ...>
    xml_attributes: dict[str, str] = field(default_factory=dict)


@dataclass
class MatsimNetwork:
    nodes: dict[str, MatsimNode]
    links: dict[str, MatsimLink]

    # Attributes stored on <links ...>, such as capperiod
    links_xml_attributes: dict[str, str] = field(default_factory=dict)

    # Optional network-level attributes
    attributes: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------
# MATSim XML loading and Writing
# ---------------------------------------------------------------------

def load_matsim_network(path: str | Path) -> MatsimNetwork:
    """
    Load a MATSim network (.xml or .xml.gz).

    The loader preserves:
      - nodes
      - links
      - standard MATSim link properties
      - allowed modes
      - custom MATSim attributes
      - additional XML attributes

    Parameters
    ----------
    path
        Path to MATSim network XML or gzipped XML.

    Returns
    -------
    MatsimNetwork
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"MATSim network does not exist: {path}")

    parser = etree.XMLParser(
        load_dtd=False,
        no_network=True,
        resolve_entities=False,
        remove_blank_text=False,
    )

    with _open_xml(path) as file:
        tree = etree.parse(file, parser)

    root = tree.getroot()

    nodes_element = root.find("nodes")
    links_element = root.find("links")

    if nodes_element is None:
        raise ValueError("MATSim network does not contain a <nodes> element.")

    if links_element is None:
        raise ValueError("MATSim network does not contain a <links> element.")

    nodes = _parse_nodes(nodes_element)
    links = _parse_links(links_element)

    _validate_link_nodes(nodes, links)

    network_attributes = _parse_attributes(root)

    return MatsimNetwork(
        nodes=nodes,
        links=links,
        links_xml_attributes=dict(links_element.attrib),
        attributes=network_attributes,
    )

def write_matsim_network(network: MatsimNetwork, path: str | Path) -> None:
    """
    Write a MATSim network to XML or XML.GZ.

    Parameters
    ----------
    network
        MATSim network to write.

    path
        Output .xml or .xml.gz file.
    """
    path = Path(path)

    root = etree.Element("network")
    _write_attributes(root, network.attributes)

    nodes_element = etree.SubElement(root, "nodes")

    for node in network.nodes.values():
        xml_attributes = dict(node.xml_attributes)
        xml_attributes.update({
            "id": node.id,
            "x": _format_number(node.x),
            "y": _format_number(node.y),
        })

        node_element = etree.SubElement(nodes_element, "node", **xml_attributes)
        _write_attributes(node_element, node.attributes)

    links_element = etree.SubElement(
        root,
        "links",
        **{key: str(value) for key, value in network.links_xml_attributes.items()},
    )

    for link in network.links.values():
        xml_attributes = dict(link.xml_attributes)

        xml_attributes.update({
            "id": link.id,
            "from": link.from_node,
            "to": link.to_node,
            "length": _format_number(link.length),
            "freespeed": _format_number(link.freespeed),
            "capacity": _format_number(link.capacity),
            "permlanes": _format_number(link.permlanes),
            "modes": ",".join(sorted(link.allowed_modes)),
        })

        link_element = etree.SubElement(links_element, "link", **xml_attributes)
        _write_attributes(link_element, link.attributes)

    tree = etree.ElementTree(root)

    doctype = '<!DOCTYPE network SYSTEM "http://www.matsim.org/files/dtd/network_v2.dtd">'

    if path.suffix == ".gz":
        with gzip.open(path, "wb") as file:
            tree.write(
                file,
                encoding="UTF-8",
                xml_declaration=True,
                doctype=doctype,
                pretty_print=True,
            )
    else:
        tree.write(
            str(path),
            encoding="UTF-8",
            xml_declaration=True,
            doctype=doctype,
            pretty_print=True,
        )



# ---------------------------------------------------------------------
# Internal parsing
# ---------------------------------------------------------------------

def _parse_nodes(
    nodes_element: etree._Element,
) -> dict[str, MatsimNode]:

    result: dict[str, MatsimNode] = {}

    for element in nodes_element.findall("node"):
        node_id = _required(element, "id")

        if node_id in result:
            raise ValueError(f"Duplicate MATSim node ID: {node_id}")

        standard = {"id", "x", "y"}

        result[node_id] = MatsimNode(
            id=node_id,
            x=float(_required(element, "x")),
            y=float(_required(element, "y")),
            attributes=_parse_attributes(element),
            xml_attributes={
                key: value
                for key, value in element.attrib.items()
                if key not in standard
            },
        )

    return result


def _parse_links(
    links_element: etree._Element,
) -> dict[str, MatsimLink]:

    result: dict[str, MatsimLink] = {}

    standard = {
        "id",
        "from",
        "to",
        "length",
        "freespeed",
        "capacity",
        "permlanes",
        "modes",
    }

    for element in links_element.findall("link"):
        link_id = _required(element, "id")

        if link_id in result:
            raise ValueError(f"Duplicate MATSim link ID: {link_id}")

        modes_text = element.get("modes", "")

        allowed_modes = {
            mode.strip()
            for mode in modes_text.split(",")
            if mode.strip()
        }

        result[link_id] = MatsimLink(
            id=link_id,
            from_node=_required(element, "from"),
            to_node=_required(element, "to"),
            length=float(_required(element, "length")),
            freespeed=float(_required(element, "freespeed")),
            capacity=float(_required(element, "capacity")),
            permlanes=float(_required(element, "permlanes")),
            allowed_modes=allowed_modes,
            attributes=_parse_attributes(element),
            xml_attributes={
                key: value
                for key, value in element.attrib.items()
                if key not in standard
            },
        )

    return result


def _parse_attributes(
    parent: etree._Element,
) -> dict[str, Any]:
    """
    Parse MATSim:

        <attributes>
            <attribute name="..." class="...">value</attribute>
        </attributes>
    """

    attributes_element = parent.find("attributes")

    if attributes_element is None:
        return {}

    result: dict[str, Any] = {}

    for element in attributes_element.findall("attribute"):
        name = element.get("name")

        if name is None:
            continue

        class_name = element.get("class", "java.lang.String")
        value = element.text or ""

        result[name] = _convert_attribute(
            value=value,
            class_name=class_name,
        )

    return result


def _convert_attribute(
    value: str,
    class_name: str,
) -> Any:

    value = value.strip()

    if class_name in {
        "java.lang.Integer",
        "java.lang.Long",
        "java.lang.Short",
    }:
        return int(value)

    if class_name in {
        "java.lang.Double",
        "java.lang.Float",
    }:
        return float(value)

    if class_name == "java.lang.Boolean":
        return value.lower() == "true"

    return value


def _validate_link_nodes(
    nodes: dict[str, MatsimNode],
    links: dict[str, MatsimLink],
) -> None:

    for link in links.values():
        if link.from_node not in nodes:
            raise ValueError(
                f"Link {link.id} references missing from-node "
                f"{link.from_node}"
            )

        if link.to_node not in nodes:
            raise ValueError(
                f"Link {link.id} references missing to-node "
                f"{link.to_node}"
            )


def _required(
    element: etree._Element,
    key: str,
) -> str:

    value = element.get(key)

    if value is None:
        raise ValueError(
            f"<{element.tag}> is missing required attribute '{key}'."
        )

    return value


def _open_xml(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rb")

    return path.open("rb")

def _write_attributes(parent: etree._Element, attributes: dict[str, Any]) -> None:
    """Write MATSim typed attributes."""

    if not attributes:
        return

    attributes_element = etree.SubElement(parent, "attributes")

    for name, value in attributes.items():
        attribute = etree.SubElement(
            attributes_element,
            "attribute",
            name=name,
            **{"class": _attribute_class(value)},
        )
        attribute.text = _attribute_text(value)


def _attribute_class(value: Any) -> str:
    """Return the MATSim Java class for a Python value."""

    if isinstance(value, bool):
        return "java.lang.Boolean"
    if isinstance(value, int):
        return "java.lang.Integer"
    if isinstance(value, float):
        return "java.lang.Double"

    return "java.lang.String"


def _attribute_text(value: Any) -> str:
    """Convert a Python value to MATSim attribute text."""

    if isinstance(value, bool):
        return str(value).lower()

    return str(value)


def _format_number(value: float) -> str:
    """Format numeric MATSim XML values compactly."""

    return f"{value:.15g}"
"""Read a MATSim network as a directed, car-only NetworkX graph.

This intentionally stays independent of the network_sampling package. Unlike
its general network reader, it only loads the fields needed for accessibility:
network topology, node coordinates and a few link properties. XML is streamed
using the Python standard library; no MATSim-specific dependency is required.

Every MATSim car link becomes a separate MultiDiGraph edge with the original
link ID as its key. Link travel times are supplied by Step 1's LinkTravelTimes.
"""

from __future__ import annotations

import gzip
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol
from xml.etree import ElementTree as ET

import networkx as nx

MissingTimePolicy = Literal["error", "free_flow"]


class TravelTimeSource(Protocol):
    """Only the fields used from the Step 1 linkstats reader."""

    travel_times_s: dict[str, float]
    invalid_travel_times: dict[str, object]


@dataclass(frozen=True, slots=True)
class NetworkLoadReport:
    total_network_nodes: int
    total_network_links: int
    links_without_modes: int
    car_links: int
    non_car_links: int
    car_links_matched: int
    car_links_missing_time: int
    car_links_invalid_time: int
    car_links_free_flow_fallback: int
    linkstats_ids_not_in_network: int
    missing_examples: tuple[str, ...]
    invalid_examples: tuple[str, ...]


@dataclass(slots=True)
class CarNetwork:
    """A routable network of car-permitted directed MATSim links.

    ``graph`` is a NetworkX MultiDiGraph.
    - Nodes: original MATSim node IDs (strings), with ``x`` and ``y``.
    - Edges: keyed by original MATSim link ID, with ``travel_time_s``,
      ``length_m``, ``freespeed_m_s``, ``capacity``, and ``permlanes``.

    The class owns only data loading and graph construction. Routing belongs
    to NetworkX or to a separate routing component later.
    """

    source_path: Path
    graph: nx.MultiDiGraph
    report: NetworkLoadReport

    @classmethod
    def from_matsim_network(
        cls,
        path: str | Path,
        travel_times: TravelTimeSource,
        *,
        missing_time_policy: MissingTimePolicy = "error",
    ) -> CarNetwork:
        """Read a MATSim .xml/.xml.gz network and attach linkstats times.

        The default ``error`` policy requires valid positive travel times for
        every car link. ``free_flow`` explicitly substitutes length/freespeed
        only for missing or invalid entries; replacements are counted.
        """
        if missing_time_policy not in ("error", "free_flow"):
            raise ValueError("missing_time_policy must be 'error' or 'free_flow'")

        source = Path(path).expanduser()
        if not source.is_file():
            raise FileNotFoundError(f"MATSim network not found: {source}")

        nodes, car_links, total_links, missing_modes, all_link_ids = _read_car_links(source)
        if not car_links:
            raise ValueError("No links explicitly permitting mode 'car' were found")

        # Only include nodes actually used by car links; no PT-only snap points.
        graph = nx.MultiDiGraph()
        used_nodes = {node for u, v, *_ in car_links for node in (u, v)}
        undefined = used_nodes.difference(nodes)
        if undefined:
            raise ValueError(f"Car links reference undefined nodes: {sorted(undefined)[:8]}")
        for node_id in used_nodes:
            x, y = nodes[node_id]
            graph.add_node(node_id, x=x, y=y)

        valid_times = travel_times.travel_times_s
        invalid_times = travel_times.invalid_travel_times
        missing_ids: list[str] = []
        invalid_ids: list[str] = []
        fallback_count = 0
        matched_count = 0

        for u, v, link_id, length, freespeed, capacity, permlanes in car_links:
            tt = valid_times.get(link_id)
            if tt is not None and math.isfinite(tt) and tt > 0:
                matched_count += 1
            else:
                if link_id in invalid_times or tt is not None:
                    invalid_ids.append(link_id)
                else:
                    missing_ids.append(link_id)
                if missing_time_policy == "free_flow":
                    if not (math.isfinite(length) and length > 0
                            and math.isfinite(freespeed) and freespeed > 0):
                        raise ValueError(
                            f"Cannot derive free-flow time for car link {link_id!r}: "
                            f"length={length}, freespeed={freespeed}"
                        )
                    tt = length / freespeed
                    fallback_count += 1
                else:
                    # Collect all discrepancies before returning an error.
                    continue

            graph.add_edge(
                u, v, key=link_id,
                link_id=link_id,
                travel_time_s=float(tt),
                length_m=length,
                freespeed_m_s=freespeed,
                capacity=capacity,
                permlanes=permlanes,
            )

        if missing_time_policy == "error" and (missing_ids or invalid_ids):
            raise ValueError(
                "Car links without valid linkstats travel times: "
                f"missing={len(missing_ids)} {missing_ids[:8]}, "
                f"invalid={len(invalid_ids)} {invalid_ids[:8]}. "
                "Check that the scenario network and linkstats match, or "
                "explicitly use missing_time_policy='free_flow' for a fallback."
            )

        report = NetworkLoadReport(
            total_network_nodes=len(nodes),
            total_network_links=total_links,
            links_without_modes=missing_modes,
            car_links=len(car_links),
            non_car_links=total_links - len(car_links),
            car_links_matched=matched_count,
            car_links_missing_time=len(missing_ids),
            car_links_invalid_time=len(invalid_ids),
            car_links_free_flow_fallback=fallback_count,
            linkstats_ids_not_in_network=len(
                (set(valid_times) | set(invalid_times)).difference(all_link_ids)
            ),
            missing_examples=tuple(missing_ids[:8]),
            invalid_examples=tuple(invalid_ids[:8]),
        )
        graph.graph.update(source_path=str(source), weight="travel_time_s")
        return cls(source, graph, report)

    @property
    def num_nodes(self) -> int:
        return self.graph.number_of_nodes()

    @property
    def num_links(self) -> int:
        return self.graph.number_of_edges()


def _read_car_links(path: Path):
    """Stream minimal car-network data, ignoring detailed MATSim attributes.

    We never construct the XML tree in memory. MATSim node records normally
    appear before links. Parsing each link at its closing tag also supports
    networks containing nested <attributes> elements.
    """
    nodes: dict[str, tuple[float, float]] = {}
    car_links: list[tuple[str, str, str, float, float, float, float]] = []
    all_link_ids: set[str] = set()
    count_links = 0
    missing_modes = 0
    root = None

    opener = gzip.open if path.suffix.lower() == ".gz" else open
    with opener(path, "rb") as stream:
        for event, elem in ET.iterparse(stream, events=("start", "end")):
            if event == "start":
                if root is None:
                    root = elem
                continue

            name = elem.tag.rsplit("}", 1)[-1]  # permits namespaced MATSim XML
            if name == "node":
                node_id = _required(elem, "id")
                if node_id in nodes:
                    raise ValueError(f"Duplicate MATSim node ID: {node_id}")
                nodes[node_id] = (
                    float(_required(elem, "x")), float(_required(elem, "y"))
                )
                elem.clear()
                if root is not None:
                    root.clear()
            elif name == "link":
                count_links += 1
                link_id = _required(elem, "id")
                if link_id in all_link_ids:
                    raise ValueError(f"Duplicate MATSim link ID: {link_id}")
                all_link_ids.add(link_id)
                modes = elem.get("modes", "").strip()
                if not modes:
                    missing_modes += 1
                if _permits_car(modes):
                    car_links.append((
                        _required(elem, "from"),
                        _required(elem, "to"),
                        link_id,
                        float(_required(elem, "length")),
                        float(_required(elem, "freespeed")),
                        float(elem.get("capacity", "nan")),
                        float(elem.get("permlanes", "nan")),
                    ))
                elem.clear()
                if root is not None:
                    root.clear()

    if not nodes:
        raise ValueError(f"No network nodes found in {path}")
    return nodes, car_links, count_links, missing_modes, all_link_ids


def _permits_car(modes: str) -> bool:
    return "car" in {
        mode.strip()
        for mode in modes.split(",")
    }


def _required(element: ET.Element, name: str) -> str:
    value = element.get(name)
    if value is None:
        raise ValueError(f"<{element.tag}> missing required attribute {name!r}")
    return value

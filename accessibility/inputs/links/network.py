"""Small, read-only MATSim network loaders for accessibility.

- CarNetwork: directed MATSim car links weighted by experienced linkstats.
- BikeNetwork: directed bike links weighted by explicitly configured cycling
  time and the exact supplied eQasim infrastructure VTT multiplier.
- PTNetwork: route-state transit graph from schedule.py, with absolute
  linkstats times on bus routes and scheduled times for other PT modes.

All network representations use NetworkX. XML reading uses the Python standard
library, not network_sampling or matsim-tools. Routing/STATENT aggregation will
be added in a separate step.
"""
from __future__ import annotations

import gzip
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol
from xml.etree import ElementTree as ET

import networkx as nx

from .schedule import TransitSchedule, TransitRoute
from .bike_costs import bike_vtt


MissingTimePolicy = Literal["error", "free_flow"]


class TravelTimeSource(Protocol):
    travel_times_s: dict[str, float]
    invalid_travel_times: dict[str, object]


@dataclass(frozen=True, slots=True)
class NetworkLoadReport:
    mode: str
    total_network_nodes: int
    total_network_links: int
    links_without_modes: int
    mode_links: int
    non_mode_links: int
    links_matched: int
    links_missing_time: int
    links_invalid_time: int
    links_free_flow_fallback: int
    linkstats_ids_not_in_network: int
    missing_examples: tuple[str, ...]
    invalid_examples: tuple[str, ...]



@dataclass(slots=True)
class ModeNetwork:
    source_path: Path
    graph: nx.MultiDiGraph
    report: NetworkLoadReport
    mode: str

    @property
    def num_nodes(self) -> int:
        return self.graph.number_of_nodes()

    @property
    def num_links(self) -> int:
        return self.graph.number_of_edges()


@dataclass(frozen=True, slots=True)
class _Link:
    link_id: str
    u: str
    v: str
    length_m: float
    freespeed_m_s: float
    capacity: float
    permlanes: float
    modes: frozenset[str]
    attributes: dict[str, str]


@dataclass(frozen=True, slots=True)
class _RawNetwork:
    nodes: dict[str, tuple[float, float]]
    links: tuple[_Link, ...]
    missing_modes: int


def _load_matsim_network(path: Path) -> _RawNetwork:
    if not path.is_file():
        raise FileNotFoundError(f"MATSim network not found: {path}")
    nodes: dict[str, tuple[float, float]] = {}
    links: list[_Link] = []
    ids: set[str] = set()
    missing_modes = 0
    root = None
    opener = gzip.open if path.suffix.lower() == ".gz" else open
    with opener(path, "rb") as stream:
        for event, element in ET.iterparse(stream, events=("start", "end")):
            if event == "start":
                if root is None:
                    root = element
                continue
            tag = element.tag.rsplit("}", 1)[-1]
            if tag == "node":
                nid = _required(element, "id")
                if nid in nodes:
                    raise ValueError(f"Duplicate node ID {nid!r}")
                nodes[nid] = (
                    float(_required(element, "x")),
                    float(_required(element, "y")),
                )
                element.clear()
                if root is not None:
                    root.clear()
            elif tag == "link":
                lid = _required(element, "id")
                if lid in ids:
                    raise ValueError(f"Duplicate link ID {lid!r}")
                ids.add(lid)
                modes = frozenset(
                    part.strip() for part in element.get("modes", "").split(",")
                    if part.strip()
                )
                if not modes:
                    missing_modes += 1
                attrs: dict[str, str] = dict(element.attrib)
                attributes_element = element.find("attributes")
                if attributes_element is not None:
                    for attr in attributes_element.findall("attribute"):
                        name = attr.get("name")
                        if name:
                            attrs[name] = (attr.text or "").strip()
                links.append(_Link(
                    link_id=lid,
                    u=_required(element, "from"),
                    v=_required(element, "to"),
                    length_m=float(_required(element, "length")),
                    freespeed_m_s=float(_required(element, "freespeed")),
                    capacity=float(element.get("capacity", "nan")),
                    permlanes=float(element.get("permlanes", "nan")),
                    modes=modes,
                    attributes=attrs,
                ))
                element.clear()
                if root is not None:
                    root.clear()
    if not nodes:
        raise ValueError(f"Network contains no nodes: {path}")
    for link in links:
        if link.u not in nodes or link.v not in nodes:
            raise ValueError(f"Link {link.link_id} references nonexistent nodes")
    return _RawNetwork(nodes, tuple(links), missing_modes)


def _required(element: ET.Element, key: str) -> str:
    value = element.get(key)
    if value is None:
        raise ValueError(f"<{element.tag}> missing required attribute {key!r}")
    return value


def _select_graph(raw: _RawNetwork, source: Path, mode: str) -> nx.MultiDiGraph:
    graph = nx.MultiDiGraph()
    relevant = (link for link in raw.links if mode in link.modes)
    for link in relevant:
        for node_id in (link.u, link.v):
            if node_id not in graph:
                x, y = raw.nodes[node_id]
                graph.add_node(node_id, x=x, y=y)
        graph.add_edge(
            link.u, link.v, key=link.link_id,
            link_id=link.link_id,
            length_m=link.length_m,
            freespeed_m_s=link.freespeed_m_s,
            capacity=link.capacity,
            permlanes=link.permlanes,
            allowed_modes=link.modes,
            # Retain original values needed to reproduce bicycle disutility.
            attributes=link.attributes,
        )
    if not graph.number_of_edges():
        raise ValueError(f"No links explicitly permitting mode {mode!r}")
    graph.graph.update(source_path=str(source), mode=mode, weight="routing_cost_s")
    return graph


@dataclass(slots=True)
class CarNetwork(ModeNetwork):
    @classmethod
    def from_matsim_network(
        cls,
        path: str | Path,
        travel_times: TravelTimeSource,
        *,
        missing_time_policy: MissingTimePolicy = "error",
    ) -> CarNetwork:
        source = Path(path).expanduser()
        raw = _load_matsim_network(source)
        return cls._from_raw(raw, source, travel_times, missing_time_policy)

    @classmethod
    def _from_raw(
        cls,
        raw: _RawNetwork,
        source: Path,
        travel_times: TravelTimeSource,
        missing_time_policy: MissingTimePolicy,
    ) -> CarNetwork:
        if missing_time_policy not in ("error", "free_flow"):
            raise ValueError("missing_time_policy must be 'error' or 'free_flow'")
        graph = _select_graph(raw, source, "car")
        missing: list[str] = []
        invalid: list[str] = []
        matched = fallback = 0
        for _, _, lid, data in graph.edges(keys=True, data=True):
            tt = travel_times.travel_times_s.get(str(lid))
            if tt is not None and math.isfinite(tt) and tt > 0:
                matched += 1
            else:
                if lid in travel_times.invalid_travel_times or tt is not None:
                    invalid.append(lid)
                else:
                    missing.append(lid)
                if missing_time_policy == "error":
                    continue
                length, speed = data["length_m"], data["freespeed_m_s"]
                if not (math.isfinite(length) and length > 0 and math.isfinite(speed) and speed > 0):
                    raise ValueError(f"Cannot calculate free-flow fallback for car link {lid}")
                tt = length / speed
                fallback += 1
            data["travel_time_s"] = float(tt)
            data["routing_cost_s"] = float(tt)
        if missing_time_policy == "error" and (missing or invalid):
            raise ValueError(
                f"Car linkstats mismatch: missing={len(missing)} {missing[:8]}, "
                f"invalid={len(invalid)} {invalid[:8]}"
            )
        all_ids = {link.link_id for link in raw.links}
        report = NetworkLoadReport(
            "car", len(raw.nodes), len(raw.links), raw.missing_modes,
            graph.number_of_edges(), len(raw.links)-graph.number_of_edges(),
            matched, len(missing), len(invalid), fallback,
            len((set(travel_times.travel_times_s) | set(travel_times.invalid_travel_times)) - all_ids),
            tuple(missing[:8]), tuple(invalid[:8]),
        )
        return cls(source, graph, report, "car")


@dataclass(slots=True)
class BikeNetwork(ModeNetwork):
    @classmethod
    def from_matsim_network(
        cls,
        path: str | Path,
        *,
        bike_speed_m_s: float,
    ) -> BikeNetwork:
        """Load bike-only links and precompute the eQasim perceived-time cost."""
        source = Path(path).expanduser()
        raw = _load_matsim_network(source)
        return cls._from_raw(raw, source, bike_speed_m_s)

    @classmethod
    def _from_raw(
        cls, raw: _RawNetwork, source: Path, bike_speed_m_s: float
    ) -> BikeNetwork:
        if not math.isfinite(bike_speed_m_s) or bike_speed_m_s <= 0:
            raise ValueError("bike_speed_m_s must be finite and positive")
        graph = _select_graph(raw, source, "bike")
        for _, _, lid, data in graph.edges(keys=True, data=True):
            length = data["length_m"]
            if not math.isfinite(length) or length <= 0:
                raise ValueError(f"Invalid bicycle link length for {lid!r}: {length}")
            factor, category, link_type = bike_vtt(
                data["attributes"], data["freespeed_m_s"]
            )
            physical_s = length / bike_speed_m_s
            data["travel_time_s"] = physical_s
            data["safety_multiplier"] = factor
            data["safety_category"] = category
            data["link_type"] = link_type
            data["infrastructure_disutility_s"] = physical_s * (factor - 1.0)
            data["routing_cost_s"] = physical_s * factor
        report = NetworkLoadReport(
            "bike", len(raw.nodes), len(raw.links), raw.missing_modes,
            graph.number_of_edges(), len(raw.links)-graph.number_of_edges(),
            0, 0, 0, 0, 0, (), (),
        )
        graph.graph.update(
            bike_speed_m_s=bike_speed_m_s,
            vtt_rule="supplied_eqasim_java",
            link_type_lookup="osm:way:highway",
        )
        return cls(source, graph, report, "bike")


def load_road_networks(
    path: str | Path,
    travel_times: TravelTimeSource,
    *,
    bike_speed_m_s: float,
    missing_time_policy: MissingTimePolicy = "error",
) -> tuple[CarNetwork, BikeNetwork]:
    """Build car and bike graphs with a *single* MATSim XML parse.

    Use when both modes are needed for a scenario. Individual class loaders
    remain available when only one graph is required.
    """
    source = Path(path).expanduser()
    raw = _load_matsim_network(source)
    car = CarNetwork._from_raw(raw, source, travel_times, missing_time_policy)
    bike = BikeNetwork._from_raw(raw, source, bike_speed_m_s)
    return car, bike


@dataclass(frozen=True, slots=True)
class PTNetworkReport:
    route_count: int
    route_stop_states: int
    ride_edges: int
    bus_ride_edges: int
    fixed_mode_ride_edges: int
    scheduled_transfer_relations: int
    constructed_transfer_edges: int
    bus_link_references: int
    zero_time_artificial_pt_links: tuple[str, ...]
    same_link_bus_segments: int
    zero_time_bus_segments: int
    bus_segments_slower_than_schedule: int
    max_bus_to_schedule_ratio: float
    ride_edges_by_mode: dict[str, int]


@dataclass(slots=True)
class PTNetwork:
    """PT route-state graph, without departure times or unscheduled transfers.

    Each node represents an occurrence of a stop on a particular route. Ride
    edges go forward along that route. Transfer edges exist *only* where a
    `minimalTransferTimes` relation explicitly permits them and link two
    distinct routes. No walking/boarding/egress edges are inserted here:
    future accessibility routing must connect origins/destinations to route
    states without allowing free route changes through a common stop node.
    """
    graph: nx.DiGraph
    report: PTNetworkReport
    states_by_stop: dict[str, tuple[tuple[str, str, int], ...]]

    @classmethod
    def from_schedule(
        cls,
        schedule: TransitSchedule,
        travel_times: TravelTimeSource,
        *,
        transfer_penalty_s: float = 180.0,
    ) -> PTNetwork:
        if not math.isfinite(transfer_penalty_s) or transfer_penalty_s < 0:
            raise ValueError("transfer_penalty_s must be nonnegative and finite")
        graph = nx.DiGraph()
        by_stop: dict[str, list[tuple[str, str, int]]] = defaultdict(list)
        counts: Counter[str] = Counter()
        bus_links_used: set[str] = set()
        zero_pt_links: set[str] = set()
        same_link_segments = 0
        zero_time_bus_segments = 0
        bus_slower = 0
        max_ratio = 0.0

        def link_time(lid: str) -> float:
            if lid in travel_times.travel_times_s:
                value = travel_times.travel_times_s[lid]
                if math.isfinite(value) and value > 0:
                    return float(value)
                if lid.startswith("pt_") and value == 0:
                    zero_pt_links.add(lid)
                    return 0.0
            # Some MATSim artificial PT connectors report zero because they
            # have no physical travel distance. Allow ONLY explicit zero on
            # pt_* links; nonzero invalid data remains an error.
            invalid = travel_times.invalid_travel_times.get(lid)
            if (lid.startswith("pt_") and invalid is not None
                    and getattr(invalid, "reason", None) == "nonpositive"
                    and _is_exact_zero(getattr(invalid, "raw_value", ""))):
                zero_pt_links.add(lid)
                return 0.0
            raise ValueError(
                f"Bus route requires link {lid!r} without a positive linkstats "
                "time (or explicit zero-time artificial pt_ connector). "
                "Check that the network, linkstats and schedule match."
            )

        for key, route in schedule.routes.items():
            line_id, route_id = key
            stops = route.stops
            states: list[tuple[str, str, int]] = []
            for idx, stop in enumerate(stops):
                state = (line_id, route_id, idx)
                states.append(state)
                graph.add_node(state, stop_id=stop.stop_id, route_key=key,
                               transport_mode=route.mode, stop_index=idx)
                by_stop[stop.stop_id].append(state)

            link_positions = _match_route_stop_links(route, schedule)
            for i, (prev_stop, next_stop) in enumerate(zip(stops, stops[1:])):
                scheduled_s = next_stop.arrival_offset_s - prev_stop.departure_offset_s
                if scheduled_s < 0:
                    raise ValueError(
                        f"Negative scheduled segment duration for {key} stops {i}->{i+1}"
                    )
                traversed_links: tuple[str, ...] = ()
                if route.mode == "bus":
                    start, end = link_positions[i], link_positions[i+1]
                    if end == start:
                        same_link_segments += 1
                    traversed_links = route.link_ids[start+1:end+1]
                    segment_time_s = 0.0
                    for lid in traversed_links:
                        bus_links_used.add(lid)
                        segment_time_s += link_time(lid)
                    if segment_time_s == 0:
                        zero_time_bus_segments += 1
                    if scheduled_s > 0 and segment_time_s > scheduled_s:
                        bus_slower += 1
                        max_ratio = max(max_ratio, segment_time_s / scheduled_s)
                    counts["bus"] += 1
                else:
                    # Fixed PT modes retain scheduled stop-to-stop durations.
                    segment_time_s = scheduled_s
                    counts[route.mode] += 1
                graph.add_edge(states[i], states[i+1], kind="ride",
                               transport_mode=route.mode,
                               travel_time_s=segment_time_s,
                               routing_cost_s=segment_time_s,
                               scheduled_time_s=scheduled_s,
                               link_ids=traversed_links)

        transfer_edges = 0
        for relation in schedule.transfers:
            origins = by_stop.get(relation.from_stop, ())
            destinations = by_stop.get(relation.to_stop, ())
            for origin in origins:
                for destination in destinations:
                    if origin[:2] == destination[:2]:
                        continue  # staying on the same route is not a transfer
                    weight = relation.minimum_time_s + transfer_penalty_s
                    graph.add_edge(
                        origin, destination,
                        kind="transfer", travel_time_s=weight,
                        routing_cost_s=weight,
                        scheduled_minimum_s=relation.minimum_time_s,
                        transfer_penalty_s=transfer_penalty_s,
                        from_stop=relation.from_stop,
                        to_stop=relation.to_stop,
                    )
                    transfer_edges += 1

        graph.graph.update(
            weight="routing_cost_s", transfer_penalty_s=transfer_penalty_s,
            no_departure_times=True, transfers_from_schedule_only=True,
        )
        report = PTNetworkReport(
            route_count=len(schedule.routes),
            route_stop_states=graph.number_of_nodes(),
            ride_edges=sum(counts.values()),
            bus_ride_edges=counts["bus"],
            fixed_mode_ride_edges=sum(v for mode, v in counts.items() if mode != "bus"),
            scheduled_transfer_relations=len(schedule.transfers),
            constructed_transfer_edges=transfer_edges,
            bus_link_references=len(bus_links_used),
            zero_time_artificial_pt_links=tuple(sorted(zero_pt_links)),
            same_link_bus_segments=same_link_segments,
            zero_time_bus_segments=zero_time_bus_segments,
            bus_segments_slower_than_schedule=bus_slower,
            max_bus_to_schedule_ratio=max_ratio,
            ride_edges_by_mode=dict(counts),
        )
        return cls(graph, report, {k: tuple(v) for k, v in by_stop.items()})


def _is_exact_zero(value: str) -> bool:
    try:
        return float(value) == 0.0
    except (ValueError, TypeError):
        return False


def _match_route_stop_links(route: TransitRoute, schedule: TransitSchedule) -> tuple[int, ...]:
    """Match stop linkRefs to ordered route links, preserving repeats/loops.

    MATSim routeProfile stops refer to stop facilities, each with a linkRefId.
    Search monotonically through that route's link sequence. Equal positions
    are allowed when consecutive stop facilities share one link. We attribute
    the link entering a stop to the corresponding previous stop-to-stop leg:
    (index of origin stop, index of destination stop].
    """
    indices: list[int] = []
    lower = 0
    for i, stop in enumerate(route.stops):
        link_id = schedule.stops[stop.stop_id].link_ref_id
        found = next((j for j in range(lower, len(route.link_ids))
                      if route.link_ids[j] == link_id), None)
        if found is None:
            raise ValueError(
                f"Route {route.line_id}/{route.route_id}: stop #{i} "
                f"{stop.stop_id} linkRefId {link_id!r} not in remaining route links"
            )
        indices.append(found)
        lower = found
    return tuple(indices)

"""Read MATSim transit schedule (.xml or .xml.gz) for PT accessibility.

Only static timetable topology is retained: facilities, route profiles, link
sequences, scheduled *offsets* and explicit minimalTransferTimes relations.
Departures/frequencies are intentionally not used by the planned approximate
PT accessibility metric.
"""
from __future__ import annotations

import gzip
import math
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree as ET


@dataclass(frozen=True, slots=True)
class StopFacility:
    stop_id: str
    x: float
    y: float
    link_ref_id: str
    name: str
    stop_area_id: str | None


@dataclass(frozen=True, slots=True)
class RouteStop:
    stop_id: str
    arrival_offset_s: float
    departure_offset_s: float


@dataclass(frozen=True, slots=True)
class TransitRoute:
    line_id: str
    route_id: str
    mode: str
    line_name: str
    stops: tuple[RouteStop, ...]
    link_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class TransferRelation:
    from_stop: str
    to_stop: str
    minimum_time_s: float


@dataclass(slots=True)
class TransitSchedule:
    source_path: Path
    stops: dict[str, StopFacility]
    routes: dict[tuple[str, str], TransitRoute]
    transfers: tuple[TransferRelation, ...]

    @classmethod
    def from_matsim_schedule(cls, path: str | Path) -> TransitSchedule:
        source = Path(path).expanduser()
        if not source.is_file():
            raise FileNotFoundError(f"MATSim transit schedule not found: {source}")
        opener = gzip.open if source.suffix.lower() == ".gz" else open
        with opener(source, "rb") as stream:
            root = ET.parse(stream).getroot()

        stops: dict[str, StopFacility] = {}
        transit_stops = root.find("transitStops")
        if transit_stops is None:
            raise ValueError("Transit schedule lacks <transitStops>")
        for item in transit_stops.findall("stopFacility"):
            sid = _req(item, "id")
            if sid in stops:
                raise ValueError(f"Duplicate stop facility {sid}")
            stops[sid] = StopFacility(
                stop_id=sid,
                x=float(_req(item, "x")),
                y=float(_req(item, "y")),
                link_ref_id=_req(item, "linkRefId"),
                name=item.get("name", ""),
                stop_area_id=item.get("stopAreaId"),
            )

        routes: dict[tuple[str, str], TransitRoute] = {}
        for line in root.findall("transitLine"):
            line_id = _req(line, "id")
            line_name = line.get("name", "")
            for route in line.findall("transitRoute"):
                route_id = _req(route, "id")
                key = (line_id, route_id)
                if key in routes:
                    raise ValueError(f"Duplicate transit route {key}")
                mode = (route.findtext("transportMode") or "").strip()
                if not mode:
                    raise ValueError(f"Missing transportMode on route {key}")
                profile = route.find("routeProfile")
                physical_route = route.find("route")
                if profile is None or physical_route is None:
                    raise ValueError(f"Route {key} missing routeProfile or route links")
                route_stops = tuple(RouteStop(
                    stop_id=_req(stop, "refId"),
                    arrival_offset_s=_parse_time(_req(stop, "arrivalOffset")),
                    departure_offset_s=_parse_time(_req(stop, "departureOffset")),
                ) for stop in profile.findall("stop"))
                link_ids = tuple(_req(link, "refId")
                                 for link in physical_route.findall("link"))
                if len(route_stops) < 2 or not link_ids:
                    raise ValueError(f"Route {key} needs >=2 stops and its link sequence")
                for stop in route_stops:
                    if stop.stop_id not in stops:
                        raise ValueError(f"Route {key} refers to missing stop {stop.stop_id}")
                routes[key] = TransitRoute(
                    line_id=line_id, route_id=route_id, mode=mode,
                    line_name=line_name, stops=route_stops, link_ids=link_ids,
                )

        transfer_times: dict[tuple[str, str], float] = {}
        minimal = root.find("minimalTransferTimes")
        if minimal is not None:
            for relation in minimal.findall("relation"):
                first, last = _req(relation, "fromStop"), _req(relation, "toStop")
                if first not in stops or last not in stops:
                    raise ValueError(f"Transfer references missing stops {first} -> {last}")
                amount = float(_req(relation, "transferTime"))
                if not math.isfinite(amount) or amount < 0:
                    raise ValueError(f"Invalid transfer time for {first}->{last}: {amount}")
                key = (first, last)
                if key in transfer_times:
                    raise ValueError(f"Duplicate transfer relation: {key}")
                transfer_times[key] = amount
        transfers = tuple(TransferRelation(a, b, time_s)
                          for (a, b), time_s in transfer_times.items())
        return cls(source, stops, routes, transfers)

    @property
    def modes(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for route in self.routes.values():
            counts[route.mode] = counts.get(route.mode, 0) + 1
        return counts


def _req(node: ET.Element, key: str) -> str:
    value = node.get(key)
    if value is None:
        raise ValueError(f"<{node.tag}> missing attribute {key!r}")
    return value


def _parse_time(value: str) -> float:
    """MATSim schedule time offset: HH:MM:SS, possibly beyond 24:00:00."""
    parts = value.split(":")
    if len(parts) != 3:
        raise ValueError(f"Expected HH:MM:SS offset; received {value!r}")
    hours, minutes, seconds = (float(part) for part in parts)
    time = hours * 3600 + minutes * 60 + seconds
    if not math.isfinite(time) or time < 0 or minutes >= 60 or seconds >= 60:
        raise ValueError(f"Invalid MATSim offset: {value!r}")
    return time

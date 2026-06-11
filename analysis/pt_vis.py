from __future__ import annotations

import gzip
import json
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

from pyproj import Transformer


SCENARIO_DIR = Path(
    r"C:\Users\20201733\Documents\Thesis\Git_Master_Thesis\MATsim\output\simple_run"
)

NETWORK_FILE = SCENARIO_DIR / "output_network.xml.gz"
EVENTS_FILE = SCENARIO_DIR / "output_events.xml.gz"

OUTPUT_GEOJSON = SCENARIO_DIR / "animation" / "pt_vehicle_trips.geojson"

SOURCE_CRS = "EPSG:2056"
TARGET_CRS = "EPSG:4326"


def get_attr(event: ET.Element, *names: str) -> str | None:
    for name in names:
        value = event.attrib.get(name)
        if value is not None:
            return value
    return None


def read_network_link_midpoints(network_file: Path) -> dict[str, tuple[float, float]]:
    """
    Read a MATSim network and return link midpoint coordinates in EPSG:2056.

    This uses the straight line between from-node and to-node. That is enough
    for animation/debugging. If your network contains detailed link geometries,
    those are usually not stored directly in standard MATSim network XML anyway.
    """
    nodes: dict[str, tuple[float, float]] = {}
    links: dict[str, tuple[str, str]] = {}

    with gzip.open(network_file, "rt", encoding="utf-8") as file:
        for _, elem in ET.iterparse(file, events=("end",)):
            if elem.tag == "node":
                node_id = elem.attrib["id"]
                x = float(elem.attrib["x"])
                y = float(elem.attrib["y"])
                nodes[node_id] = (x, y)

            elif elem.tag == "link":
                link_id = elem.attrib["id"]
                from_node = elem.attrib["from"]
                to_node = elem.attrib["to"]
                links[link_id] = (from_node, to_node)

            elem.clear()

    link_midpoints: dict[str, tuple[float, float]] = {}

    for link_id, (from_node, to_node) in links.items():
        if from_node not in nodes or to_node not in nodes:
            continue

        x1, y1 = nodes[from_node]
        x2, y2 = nodes[to_node]

        link_midpoints[link_id] = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    return link_midpoints


def read_pt_passenger_vehicles(events_file: Path):
    """
    Find transit vehicles, PT passenger boardings, and vehicle modes.

    Uses:
    - TransitDriverStarts
    - PersonEntersPtVehicle
    - PersonLeavesPtVehicle
    """
    transit_vehicles: set[str] = set()
    vehicle_modes: dict[str, str] = {}

    boardings_by_vehicle: dict[str, int] = defaultdict(int)
    alightings_by_vehicle: dict[str, int] = defaultdict(int)

    with gzip.open(events_file, "rt", encoding="utf-8") as file:
        for _, elem in ET.iterparse(file, events=("end",)):
            if elem.tag != "event":
                elem.clear()
                continue

            event_type = elem.attrib.get("type")

            if event_type == "TransitDriverStarts":
                vehicle_id = get_attr(elem, "vehicle", "vehicleId")
                transit_line = get_attr(elem, "transitLineId", "line")
                transit_route = get_attr(elem, "transitRouteId", "route")

                if vehicle_id is not None:
                    transit_vehicles.add(vehicle_id)

                    # Your vehicle IDs seem to contain bus/rail suffixes, e.g. veh_1144_bus.
                    if vehicle_id.endswith("_bus"):
                        vehicle_modes[vehicle_id] = "bus"
                    elif vehicle_id.endswith("_rail"):
                        vehicle_modes[vehicle_id] = "rail"
                    else:
                        vehicle_modes[vehicle_id] = "pt"

            elif event_type == "PersonEntersPtVehicle":
                vehicle_id = get_attr(elem, "vehicle", "vehicleId")
                if vehicle_id is not None:
                    boardings_by_vehicle[vehicle_id] += 1

            elif event_type == "PersonLeavesPtVehicle":
                vehicle_id = get_attr(elem, "vehicle", "vehicleId")
                if vehicle_id is not None:
                    alightings_by_vehicle[vehicle_id] += 1

            elem.clear()

    vehicles_with_passengers = {
        vehicle_id
        for vehicle_id, boardings in boardings_by_vehicle.items()
        if boardings > 0
    }

    return transit_vehicles, vehicles_with_passengers, vehicle_modes, boardings_by_vehicle, alightings_by_vehicle


def read_vehicle_link_trajectories(
    events_file: Path,
    vehicles_to_keep: set[str],
    link_midpoints: dict[str, tuple[float, float]],
    start_time_s: float | None = None,
    end_time_s: float | None = None,
):
    """
    Extract timestamped link midpoint positions from entered-link events.
    """
    trajectories: dict[str, list[tuple[float, float, float]]] = defaultdict(list)

    with gzip.open(events_file, "rt", encoding="utf-8") as file:
        for _, elem in ET.iterparse(file, events=("end",)):
            if elem.tag != "event":
                elem.clear()
                continue

            event_type = elem.attrib.get("type")

            if event_type != "entered link":
                elem.clear()
                continue

            vehicle_id = get_attr(elem, "vehicle", "vehicleId")
            link_id = get_attr(elem, "link", "linkId")
            time_s = float(elem.attrib["time"])

            if vehicle_id not in vehicles_to_keep:
                elem.clear()
                continue

            if start_time_s is not None and time_s < start_time_s:
                elem.clear()
                continue

            if end_time_s is not None and time_s > end_time_s:
                elem.clear()
                continue

            if link_id not in link_midpoints:
                elem.clear()
                continue

            x, y = link_midpoints[link_id]
            trajectories[vehicle_id].append((time_s, x, y))

            elem.clear()

    return trajectories


def write_kepler_trip_geojson(
    trajectories: dict[str, list[tuple[float, float, float]]],
    vehicle_modes: dict[str, str],
    boardings_by_vehicle: dict[str, int],
    alightings_by_vehicle: dict[str, int],
    output_geojson: Path,
) -> None:
    transformer = Transformer.from_crs(SOURCE_CRS, TARGET_CRS, always_xy=True)

    features = []

    for vehicle_id, points in trajectories.items():
        points = sorted(points, key=lambda p: p[0])

        # Need at least two points for a line animation.
        if len(points) < 2:
            continue

        coordinates = []

        for time_s, x, y in points:
            lon, lat = transformer.transform(x, y)

            # Kepler Trip Layer accepts [lon, lat, altitude, timestamp].
            coordinates.append([lon, lat, 0, time_s])

        mode = vehicle_modes.get(vehicle_id, "pt")

        features.append(
            {
                "type": "Feature",
                "properties": {
                    "vehicle_id": vehicle_id,
                    "mode": mode,
                    "boardings": boardings_by_vehicle.get(vehicle_id, 0),
                    "alightings": alightings_by_vehicle.get(vehicle_id, 0),
                },
                "geometry": {
                    "type": "LineString",
                    "coordinates": coordinates,
                },
            }
        )

    output_geojson.parent.mkdir(parents=True, exist_ok=True)

    with open(output_geojson, "w", encoding="utf-8") as file:
        json.dump(
            {
                "type": "FeatureCollection",
                "features": features,
            },
            file,
        )

    print(f"Wrote {len(features):,} animated PT vehicle trips to:")
    print(output_geojson)


def main() -> None:
    print("Reading network...")
    link_midpoints = read_network_link_midpoints(NETWORK_FILE)
    print(f"Network link midpoints: {len(link_midpoints):,}")

    print("Reading PT passenger vehicles...")
    (
        transit_vehicles,
        vehicles_with_passengers,
        vehicle_modes,
        boardings_by_vehicle,
        alightings_by_vehicle,
    ) = read_pt_passenger_vehicles(EVENTS_FILE)

    print(f"Transit vehicles found:      {len(transit_vehicles):,}")
    print(f"Vehicles with passengers:    {len(vehicles_with_passengers):,}")
    print(f"PT passenger boardings:      {sum(boardings_by_vehicle.values()):,}")
    print(f"PT passenger alightings:     {sum(alightings_by_vehicle.values()):,}")

    print("Extracting vehicle trajectories...")

    # Optional: restrict to morning peak.
    # Use None, None for full day.
    trajectories = read_vehicle_link_trajectories(
        events_file=EVENTS_FILE,
        vehicles_to_keep=vehicles_with_passengers,
        link_midpoints=link_midpoints,
        start_time_s=6 * 3600,
        end_time_s=10 * 3600,
    )

    print(f"Vehicle trajectories extracted: {len(trajectories):,}")

    write_kepler_trip_geojson(
        trajectories=trajectories,
        vehicle_modes=vehicle_modes,
        boardings_by_vehicle=boardings_by_vehicle,
        alightings_by_vehicle=alightings_by_vehicle,
        output_geojson=OUTPUT_GEOJSON,
    )


if __name__ == "__main__":
    main()
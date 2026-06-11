from __future__ import annotations

import gzip
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path


EVENTS_FILE = Path(
    r"C:\Users\20201733\Documents\Thesis\Git_Master_Thesis\MATsim\output\simple_run\output_events.xml.gz"
)


def get_attr(event: ET.Element, *names: str) -> str | None:
    for name in names:
        value = event.attrib.get(name)
        if value is not None:
            return value
    return None


def main() -> None:

    boardings_by_vehicle = defaultdict(int)
    alightings_by_vehicle = defaultdict(int)
    boardings_by_person = defaultdict(int)
    alightings_by_person = defaultdict(int)

    total_boardings = 0
    total_alightings = 0
    waiting_for_pt = 0

    with gzip.open(EVENTS_FILE, "rt", encoding="utf-8") as f:
        for _, elem in ET.iterparse(f, events=("end",)):
            if elem.tag != "event":
                elem.clear()
                continue

            event_type = elem.attrib.get("type")

            if event_type == "waitingForPt":
                waiting_for_pt += 1

            elif event_type == "PersonEntersPtVehicle":
                person_id = elem.attrib.get("person")
                vehicle_id = elem.attrib.get("vehicle")

                total_boardings += 1

                if vehicle_id is not None:
                    boardings_by_vehicle[vehicle_id] += 1

                if person_id is not None:
                    boardings_by_person[person_id] += 1

            elif event_type == "PersonLeavesPtVehicle":
                person_id = elem.attrib.get("person")
                vehicle_id = elem.attrib.get("vehicle")

                total_alightings += 1

                if vehicle_id is not None:
                    alightings_by_vehicle[vehicle_id] += 1

                if person_id is not None:
                    alightings_by_person[person_id] += 1

            elem.clear()

    print("\nPT passenger check")
    print("------------------")
    print(f"waitingForPt events:      {waiting_for_pt:,}")
    print(f"PT passenger boardings:   {total_boardings:,}")
    print(f"PT passenger alightings:  {total_alightings:,}")
    print(f"PT users boarding:        {len(boardings_by_person):,}")
    print(f"PT vehicles with riders:  {len(boardings_by_vehicle):,}")

    print("\nTop 20 PT vehicles by boardings:")
    for vehicle_id, boardings in sorted(
        boardings_by_vehicle.items(),
        key=lambda item: item[1],
        reverse=True,
    )[:20]:
        print(
            f"{vehicle_id}: "
            f"boardings={boardings}, "
            f"alightings={alightings_by_vehicle[vehicle_id]}"
        )
if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""Measure baseline car-accessible physical street length against intervention budgets.

A physical street segment is included if at least one of its baseline MATSim
links allows the `car` mode. Each segment is counted once, regardless of
whether car access exists in one or both directions.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import statistics
import xml.etree.ElementTree as ET
from pathlib import Path


def open_text(path: Path):
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, "r", encoding="utf-8")


def car_links_in_network(path: Path) -> tuple[set[str], int, float]:
    car_ids: set[str] = set()
    directed_length_m = 0.0
    with open_text(path) as stream:
        for _, elem in ET.iterparse(stream, events=("end",)):
            if elem.tag.rsplit("}", 1)[-1] != "link":
                elem.clear()
                continue
            modes = {mode.strip() for mode in elem.get("modes", "").split(",")}
            if "car" in modes:
                car_ids.add(elem.attrib["id"])
                directed_length_m += float(elem.attrib["length"])
            elem.clear()
    return car_ids, len(car_ids), directed_length_m


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("network", type=Path, help="Baseline network.xml[.gz]")
    parser.add_argument("segments", type=Path, help="Shared street_segments.csv.gz")
    parser.add_argument(
        "--budgets", type=float, nargs="+",
        default=[500, 1500, 3000, 5000, 7000], metavar="METRES",
    )
    args = parser.parse_args()

    car_ids, directed_count, directed_car_m = car_links_in_network(args.network)

    car_lengths: list[float] = []
    all_segment_count = 0
    missing_car_mapping = set(car_ids)
    with open_text(args.segments) as stream:
        reader = csv.DictReader(stream)
        required = {"length_m", "baseline_link_ids"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Expected columns {sorted(required)}; found {reader.fieldnames}")
        for row in reader:
            all_segment_count += 1
            link_ids = {str(link_id) for link_id in json.loads(row["baseline_link_ids"])}
            matched = link_ids & car_ids
            if matched:
                car_lengths.append(float(row["length_m"]))
                missing_car_mapping.difference_update(matched)

    total_m = sum(car_lengths)
    if total_m <= 0:
        raise ValueError("No car-accessible StreetSegments matched the baseline network")

    print("BASELINE CAR NETWORK")
    print(f"  Directed car links:          {directed_count:,}")
    print(f"  Directed car link length:    {directed_car_m/1000:,.2f} km")
    print("\nCAR-ACCESSIBLE PHYSICAL STREET NETWORK (count each segment once)")
    print(f"  Total indexed segments:     {all_segment_count:,}")
    print(f"  Car-accessible segments:    {len(car_lengths):,}")
    print(f"  Car-accessible length:      {total_m/1000:,.2f} km")
    print(f"  Median segment length:      {statistics.median(car_lengths):,.0f} m")
    print("\nINTERVENTION BUDGET / CAR-ACCESSIBLE PHYSICAL LENGTH")
    for budget in args.budgets:
        print(f"  {budget/1000:5.2f} km : {100*budget/total_m:6.2f}%")

    if missing_car_mapping:
        print(f"\nNote: {len(missing_car_mapping):,} directed car links are not represented in the "
              "StreetSegmentIndex; their length is excluded from the physical denominator.")
    print("\nDefinition: a segment qualifies if >=1 of its baseline links permits 'car'.")
    print("This is NOT the same as full intervention eligibility (road width, PT rules, etc.).")


if __name__ == "__main__":
    main()


import argparse
import csv
import gzip
import math
import re
import xml.etree.ElementTree as ET


def read_network(path):
    opener = gzip.open if str(path).endswith(".gz") else open

    with opener(path, "rb") as f:
        root = ET.parse(f).getroot()

    nodes = {}

    for node in root.findall("./nodes/node"):
        node_id = node.get("id")
        z = node.get("z")

        nodes[node_id] = {
            "x": float(node.get("x")),
            "y": float(node.get("y")),
            "z": float(z) if z is not None else None,
        }

    return root, nodes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("network", help="MATSim network.xml or network.xml.gz")
    parser.add_argument(
        "--output",
        default="bike_gradients.csv",
        help="Output CSV file",
    )
    args = parser.parse_args()

    root, nodes = read_network(args.network)

    results = []

    n_bike = 0
    n_missing = 0
    n_zero = 0
    n_short = 0
    n_invalid_length = 0

    for link in root.findall("./links/link"):

        modes = set(
            re.split(r"[\s,]+", link.get("modes", ""))
        )

        if "bike" not in modes:
            continue

        n_bike += 1

        from_node = nodes[link.get("from")]
        to_node = nodes[link.get("to")]

        z_from = from_node["z"]
        z_to = to_node["z"]

        # Missing or invalid elevations
        if (
            z_from is None
            or z_to is None
            or not math.isfinite(z_from)
            or not math.isfinite(z_to)
        ):
            n_missing += 1
            continue

        zero_elevation = z_from == 0 or z_to == 0

        if zero_elevation:
            n_zero += 1

        length = float(link.get("length"))

        if length <= 0:
            n_invalid_length += 1
            continue

        if length < 1:
            n_short += 1

        elevation_difference = z_to - z_from

        gradient = 100 * elevation_difference / length

        # Midpoint for locating the link in QGIS
        mid_x = (from_node["x"] + to_node["x"]) / 2
        mid_y = (from_node["y"] + to_node["y"]) / 2

        results.append({
            "link_id": link.get("id"),
            "from_node": link.get("from"),
            "to_node": link.get("to"),
            "length_m": length,
            "z_from_m": z_from,
            "z_to_m": z_to,
            "elevation_difference_m": elevation_difference,
            "gradient_pct": gradient,
            "abs_gradient_pct": abs(gradient),
            "zero_elevation": zero_elevation,
            "mid_x": mid_x,
            "mid_y": mid_y,
        })

    # Sort by absolute gradient: steepest first
    results.sort(
        key=lambda r: r["abs_gradient_pct"],
        reverse=True,
    )

    print("\n===== BICYCLE NETWORK GRADIENT CHECK =====")

    print(f"Total nodes:             {len(nodes):,}")
    print(f"Bike-accessible links:   {n_bike:,}")
    print(f"Missing elevation links: {n_missing:,}")
    print(f"Zero elevation links:    {n_zero:,}")
    print(f"Links shorter than 1 m:  {n_short:,}")
    print(f"Invalid-length links:    {n_invalid_length:,}")

    if not results:
        print("\nNo bicycle links with valid elevations found.")
        return

    print("\n===== GRADIENT DISTRIBUTION =====")

    for threshold in [5, 10, 15, 20, 25, 30, 50, 100]:
        count = sum(
            r["abs_gradient_pct"] >= threshold
            for r in results
        )

        percentage = 100 * count / len(results)

        print(
            f"|Gradient| >= {threshold:3d}%: "
            f"{count:7,d} links ({percentage:6.2f}%)"
        )

    print("\n===== 30 STEEPEST LINKS =====")

    print(
        f"{'Link ID':<25}"
        f"{'Length (m)':>12}"
        f"{'Delta Z (m)':>14}"
        f"{'Gradient (%)':>15}"
    )

    for r in results[:30]:
        print(
            f"{r['link_id']:<25}"
            f"{r['length_m']:>12.1f}"
            f"{r['elevation_difference_m']:>14.1f}"
            f"{r['gradient_pct']:>15.1f}"
        )

    # Export every bike link with valid elevations
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(results[0].keys()),
        )

        writer.writeheader()
        writer.writerows(results)

    print(f"\nSaved {len(results):,} links to: {args.output}")


if __name__ == "__main__":
    main()
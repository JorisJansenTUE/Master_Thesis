
from pathlib import Path
import gzip
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

from preprocessing.utils import DATA_DIR, RAW_DIR
# --------------------------------------------------
# Settings
# --------------------------------------------------
NETWORK_BEFORE = Path(DATA_DIR / "NDA_Eth_Model" / "locarno" / "locarno_network.xml.gz")
#NETWORK_BEFORE = Path(RAW_DIR / "OSM" / "REV_width_tagged_buffered" / "REV_network_final.xml.gz")
NETWORK_AFTER = Path(r"C:\Users\20201733\Downloads\output_network.xml.gz")

OUTPUT = Path("calibration_comparison_new_v2.csv")

OSM_ATTRIBUTE = "osm:way:id"

CHANGE_TOLERANCE = 0.01  # 1% change threshold
LENGTH_TOLERANCE = 0.01   # 1% difference in OSM way length


# --------------------------------------------------
# Read MATSim network
# --------------------------------------------------
def read_network(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    records = []

    with opener(path, "rb") as f:
        for _, elem in ET.iterparse(f, events=("end",)):
            if elem.tag != "link":
                continue

            attrs = {
                a.get("name"): (a.text or "").strip()
                for a in elem.findall("./attributes/attribute")
            }

            osm_id = attrs.get(OSM_ATTRIBUTE)

            if osm_id:
                records.append({
                    "osm_id": osm_id,
                    "length": float(elem.get("length")),
                    "capacity": float(elem.get("capacity")),
                    "freespeed": float(elem.get("freespeed")),
                })

            elem.clear()

    if not records:
        raise ValueError(
            f"No links with attribute '{OSM_ATTRIBUTE}' found in {path}"
        )

    return pd.DataFrame(records)


# --------------------------------------------------
# Aggregate links by OSM way ID
# --------------------------------------------------
def aggregate(df):
    df = df.copy()

    df["weighted_capacity"] = df["capacity"] * df["length"]
    df["weighted_speed"] = df["freespeed"] * df["length"]

    grouped = df.groupby("osm_id").agg(
        n_links=("osm_id", "size"),
        length=("length", "sum"),
        capacity_sum=("weighted_capacity", "sum"),
        speed_sum=("weighted_speed", "sum"),
    )

    grouped["capacity"] = grouped["capacity_sum"] / grouped["length"]
    grouped["freespeed"] = grouped["speed_sum"] / grouped["length"]

    return grouped[["n_links", "length", "capacity", "freespeed"]]


# --------------------------------------------------
# Print statistics
# --------------------------------------------------
def print_statistics(df, variable):
    changes = df[f"{variable}_pct_change"]
    changed = df[f"{variable}_changed"]

    changed_values = changes[changed].dropna()

    print(f"\n--- {variable.upper()} ---")
    print(f"Changed:   {changed.sum():,} / {len(df):,} "
          f"({changed.mean() * 100:.2f}%)")
    print(f"Unchanged: {(~changed).sum():,}")

    if not changed_values.empty:
        print("\nChanges among changed OSM ways:")
        print(f"Mean:      {changed_values.mean():+.2f}%")
        print(f"Median:    {changed_values.median():+.2f}%")
        print(f"Minimum:   {changed_values.min():+.2f}%")
        print(f"Maximum:   {changed_values.max():+.2f}%")

        ratios = df.loc[changed, f"{variable}_ratio"].dropna()
        print(f"\nMean ratio:   {ratios.mean():.3f}")
        print(f"Median ratio: {ratios.median():.3f}")
    else:
        print("No changes detected.")


# --------------------------------------------------
# Main
# --------------------------------------------------
def main():
    print("Reading networks...")

    before_raw = read_network(NETWORK_BEFORE)
    after_raw = read_network(NETWORK_AFTER)

    before = aggregate(before_raw)
    after = aggregate(after_raw)

    # Match OSM ways that occur in both networks
    df = before.join(
        after,
        how="inner",
        lsuffix="_before",
        rsuffix="_after",
    )

    smaller_count = min(len(before), len(after))

    print("\n=== NETWORK MATCHING ===")
    print(f"Links before:     {len(before_raw):,}")
    print(f"Links after:      {len(after_raw):,}")
    print(f"OSM ways before:  {len(before):,}")
    print(f"OSM ways after:   {len(after):,}")
    print(f"Matched OSM ways: {len(df):,}")

    print(
        f"Coverage of smaller network: "
        f"{len(df) / smaller_count * 100:.2f}%"
    )

    # Check whether the matched OSM ways have similar lengths
    df["length_ratio"] = (
        df["length_after"] / df["length_before"]
    )

    df["length_matched"] = (
        (df["length_ratio"] - 1).abs() <= LENGTH_TOLERANCE
    )

    n_equal = df["length_matched"].sum()
    n_different = (~df["length_matched"]).sum()

    print("\n=== LENGTH COMPARISON ===")
    print(f"Similar length:   {n_equal:,}")
    print(f"Different length: {n_different:,}")

    # Keep all matched OSM ways in the output CSV,
    # but only use similarly sized ways for calibration statistics
    comparable = df[df["length_matched"]].copy()

    if comparable.empty:
        print("\nNo comparable OSM ways found.")
        df.to_csv(OUTPUT)
        return

    # Compare capacity and free-flow speed
    for variable in ["capacity", "freespeed"]:
        old = df[f"{variable}_before"]
        new = df[f"{variable}_after"]

        df[f"{variable}_ratio"] = (
            new / old.replace(0, np.nan)
        )

        df[f"{variable}_pct_change"] = (
            (df[f"{variable}_ratio"] - 1) * 100
        )

        df[f"{variable}_changed"] = (
            df[f"{variable}_pct_change"].abs()
            > CHANGE_TOLERANCE * 100
        )

    comparable = df[df["length_matched"]].copy()

    # Summary
    print("\n=== CALIBRATION COMPARISON ===")
    print(f"Comparable OSM ways: {len(comparable):,}")

    print_statistics(comparable, "capacity")
    print_statistics(comparable, "freespeed")

    capacity_changed = comparable["capacity_changed"]
    speed_changed = comparable["freespeed_changed"]

    print("\n--- COMBINED CHANGES ---")
    print(f"Both changed:         {(capacity_changed & speed_changed).sum():,}")
    print(f"Only capacity:        {(capacity_changed & ~speed_changed).sum():,}")
    print(f"Only free-flow speed: {(~capacity_changed & speed_changed).sum():,}")
    print(f"Neither changed:      {(~capacity_changed & ~speed_changed).sum():,}")

    # Save detailed comparison
    df.to_csv(OUTPUT)

    print(f"\nDetailed comparison saved to: {OUTPUT}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import numpy as np
import pandas as pd
import gzip
import xml.etree.ElementTree as ET
from pathlib import Path

from .data import SyntheticPerson


def print_summary(persons: list[SyntheticPerson]) -> None:
    print("\nPopulation synthesis summary")
    print("----------------------------")
    print(f"Persons: {len(persons):,}")

    if not persons:
        print("No persons were generated.")
        return

    mode_series = pd.Series([person.initial_mode for person in persons])
    mode_counts = mode_series.value_counts(normalize=False)
    mode_shares = mode_series.value_counts(normalize=True)

    print("\nInitial modes:")
    for mode in sorted(mode_counts.index):
        print(
            f"  {mode:>5}: "
            f"{mode_counts[mode]:>8,} "
            f"({100.0 * mode_shares[mode]:5.1f}%)"
        )

    distances = [
        person.home.distance_to(person.work)
        for person in persons
    ]

    print("\nHome-work distance:")
    print(f"  mean:   {np.mean(distances):,.0f} m")
    print(f"  median: {np.median(distances):,.0f} m")
    print(f"  p95:    {np.percentile(distances, 95):,.0f} m")

    origin_counts = pd.Series(
        [person.home.municipality for person in persons]
    ).value_counts()

    destination_counts = pd.Series(
        [person.work.municipality for person in persons]
    ).value_counts()

    print("\nTop 10 origin municipalities:")
    for municipality, count in origin_counts.head(10).items():
        print(f"  {municipality}: {count:,}")

    print("\nTop 10 destination municipalities:")
    for municipality, count in destination_counts.head(10).items():
        print(f"  {municipality}: {count:,}")

import matplotlib.pyplot as plt
from pathlib import Path


def plot_population_summary(
    modes: list[str],
    distances: list[float],
    output_folder: str | Path = "population_validation",
) -> None:
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------------
    # Mode-share pie chart
    # --------------------------------------------------------------

    mode_counts = pd.Series(modes).value_counts()

    plt.figure(figsize=(6, 6))

    mode_counts.plot.pie(
        autopct="%1.1f%%",
        startangle=90,
    )

    plt.title("Initial mode share")
    plt.ylabel("")
    plt.tight_layout()

    plt.savefig(
        output_folder / "initial_mode_share.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()

    # --------------------------------------------------------------
    # Home-work distance histogram
    # --------------------------------------------------------------

    distance_km = np.asarray(distances) / 1000

    # 1km distance brackets
    bins = np.arange(
        0,
        max(2, np.ceil(distance_km.max()) + 1),
        1,
    )

    plt.figure(figsize=(7, 5))

    plt.hist(
        distance_km,
        bins=bins,
        edgecolor="black",
    )

    plt.axvline(
        distance_km.mean(),
        linestyle="--",
        label=f"Mean: {distance_km.mean():.1f} km",
    )

    plt.axvline(
        np.median(distance_km),
        linestyle=":",
        label=f"Median: {np.median(distance_km):.1f} km",
    )

    plt.xlabel("Home-work distance (km)")
    plt.ylabel("Number of persons")
    plt.title("Home-work distance distribution")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        output_folder / "home_work_distance.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()


def print_xml_population_summary(folder: str | Path) -> None:

    folder = Path(folder)
    population_file = folder / "population.xml.gz"

    with gzip.open(population_file, "rb") as file:
        root = ET.parse(file).getroot()

    modes = []
    distances = []
    origins = []
    destinations = []

    for person in root.findall("person"):
        plans = person.findall("plan")

        plan = next(
            (
                plan for plan in plans
                if plan.get("selected", "").lower() in {"yes", "true"}
            ),
            plans[0],
        )

        activities = plan.findall("activity")
        legs = plan.findall("leg")

        home = next(
            (
                activity
                for activity in activities
                if activity.get("type", "").lower() == "h" or activity.get("type", "").lower().startswith("home")
            ),
            None
        )

        work = next(
            (
                activity
                for activity in activities
                if activity.get("type", "").lower() == "w" or activity.get("type", "").lower().startswith("work")
            ),
            None
        )

        if home is None or work is None:
            print(
                f"Skipping person {person.get('id')}: "
                f"activity types = {[a.get('type') for a in activities]}"
            )
            continue

        modes.append(legs[0].get("mode", "unknown"))

        distances.append(
            np.hypot(
                float(work.get("x")) - float(home.get("x")),
                float(work.get("y")) - float(home.get("y")),
            )
        )

        # Only works if municipality was written as an activity attribute
        if home.get("municipality"):
            origins.append(home.get("municipality"))

        if work.get("municipality"):
            destinations.append(work.get("municipality"))

    print("\nPopulation XML summary")
    print("----------------------")
    print(f"Persons: {len(modes):,}")

    mode_counts = pd.Series(modes).value_counts()

    print("\nInitial modes:")
    for mode, count in mode_counts.items():
        print(
            f"  {mode:>5}: {count:>8,} "
            f"({100 * count / len(modes):5.1f}%)"
        )

    print("\nHome-work distance:")
    print(f"  mean:   {np.mean(distances):,.0f} m")
    print(f"  median: {np.median(distances):,.0f} m")
    print(f"  p95:    {np.percentile(distances, 95):,.0f} m")

    if origins:
        print("\nTop 10 origin municipalities:")
        print(pd.Series(origins).value_counts().head(10).to_string())

    if destinations:
        print("\nTop 10 destination municipalities:")
        print(pd.Series(destinations).value_counts().head(10).to_string())

    if modes and distances:
        plot_population_summary(
            modes=modes,
            distances=distances,
            output_folder=folder,
        )

if __name__ == "__main__":
    folder = Path(
        r"data/scenarios/locarno_720"
    )

    print_xml_population_summary(folder)
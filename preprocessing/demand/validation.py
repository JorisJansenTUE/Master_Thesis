from __future__ import annotations

import numpy as np
import pandas as pd

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
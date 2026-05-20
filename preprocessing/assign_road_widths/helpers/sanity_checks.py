import gzip
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Tuple, List, Set

import geopandas as gpd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pyproj import Transformer
from shapely.geometry import LineString
from shapely.geometry.base import BaseGeometry

from preprocessing.assign_road_widths.width_config import WidthEstimationConfig
from preprocessing.assign_road_widths.helpers.width_estimation import WayWidthResult

class SanityChecker:
    """
    Creates tabular and visual sanity checks for the estimated widths.
    """

    def __init__(
        self,
        config: WidthEstimationConfig,
        expected_way_ids: Set[str],
        width_results: Dict[str, WayWidthResult],
    ):
        self.config = config
        self.expected_way_ids = expected_way_ids
        self.width_results = width_results

    def run(self) -> pd.DataFrame:
        print("\n===================== Sanity Checks =====================")
        print("Running sanity checks...")

        summary = self._build_summary_table()

        self._print_coverage(summary)
        self._print_missing_by_highway(summary)
        self._print_large_widths(summary)
        self._print_small_widths(summary)
        self._print_bad_street_space_relations(summary)

        self._save_outputs(summary)

        return summary

    def _build_summary_table(self) -> pd.DataFrame:
        rows = []

        for way_id in sorted(self.expected_way_ids):
            result = self.width_results.get(way_id)

            row = {
                "way_id": way_id,
                "has_width": result is not None,
                "highway": None,
                "name": None,
                "carriageway_width_m": np.nan,
                "carriageway_p10_m": np.nan,
                "carriageway_p90_m": np.nan,
                "carriageway_n": 0,
                "street_space_width_m": np.nan,
                "street_space_p10_m": np.nan,
                "street_space_p90_m": np.nan,
                "street_space_n": 0,
            }

            if result is not None:
                row["highway"] = result.highway
                row["name"] = result.name

                if result.carriageway is not None:
                    row["carriageway_width_m"] = result.carriageway.median
                    row["carriageway_p10_m"] = result.carriageway.p10
                    row["carriageway_p90_m"] = result.carriageway.p90
                    row["carriageway_n"] = result.carriageway.n

                if result.street_space is not None:
                    row["street_space_width_m"] = result.street_space.median
                    row["street_space_p10_m"] = result.street_space.p10
                    row["street_space_p90_m"] = result.street_space.p90
                    row["street_space_n"] = result.street_space.n

            rows.append(row)

        return pd.DataFrame(rows)

    def _print_coverage(self, summary: pd.DataFrame) -> None:
        n_expected = len(self.expected_way_ids)
        n_with_width = len(self.width_results)
        n_missing = n_expected - n_with_width

        print(f"Road ways that should have width: {n_expected}")
        print(f"Road ways with width estimate:    {n_with_width}")
        print(f"Road ways missing width estimate: {n_missing}")

        if n_expected > 0:
            coverage = 100.0 * n_with_width / n_expected
            print(f"Coverage: {coverage:.1f}%")

    @staticmethod
    def _print_missing_by_highway(summary: pd.DataFrame) -> None:
        missing = summary[~summary["has_width"]]

        print("\nMissing widths by highway type:")

        if missing.empty:
            print("None")
        else:
            print(missing["highway"].value_counts(dropna=False).to_string())

    def _print_large_widths(self, summary: pd.DataFrame) -> None:
        threshold = self.config.large_width_threshold_m

        large = summary[
            summary["carriageway_width_m"] > threshold
        ].copy()

        print(f"\nRoads with carriageway width > {threshold} m: {len(large)}")

        if not large.empty:
            print(
                large[
                    [
                        "way_id",
                        "highway",
                        "name",
                        "carriageway_width_m",
                        "carriageway_p10_m",
                        "carriageway_p90_m",
                        "carriageway_n",
                    ]
                ]
                .sort_values("carriageway_width_m", ascending=False)
                .head(30)
                .to_string(index=False)
            )

    def _print_small_widths(self, summary: pd.DataFrame) -> None:
        threshold = self.config.small_width_threshold_m

        small = summary[
            summary["carriageway_width_m"] < threshold
        ].copy()

        print(f"\nRoads with carriageway width < {threshold} m: {len(small)}")

        if not small.empty:
            print(
                small[
                    [
                        "way_id",
                        "highway",
                        "name",
                        "carriageway_width_m",
                        "carriageway_n",
                    ]
                ]
                .sort_values("carriageway_width_m")
                .head(30)
                .to_string(index=False)
            )

    @staticmethod
    def _print_bad_street_space_relations(summary: pd.DataFrame) -> None:
        bad = summary[
            summary["street_space_width_m"] < summary["carriageway_width_m"]
        ].copy()

        print(
            "\nRoads where street-space width is smaller than carriageway width: "
            f"{len(bad)}"
        )

        if not bad.empty:
            print(
                bad[
                    [
                        "way_id",
                        "highway",
                        "name",
                        "carriageway_width_m",
                        "street_space_width_m",
                    ]
                ]
                .head(30)
                .to_string(index=False)
            )

    def _save_outputs(self, summary: pd.DataFrame) -> None:
        self.config.plot_dir.mkdir(exist_ok=True)

        summary_csv = self.config.plot_dir / "width_summary_by_way.csv"
        summary.to_csv(summary_csv, index=False)

        print(f"\nSaved summary table:\n{summary_csv}")

        self._plot_carriageway_histogram(summary)
        self._plot_carriageway_by_highway(summary)
        self._plot_carriageway_vs_street_space(summary)

    def _plot_carriageway_histogram(self, summary: pd.DataFrame) -> None:
        values = summary["carriageway_width_m"].dropna()

        if values.empty:
            return

        plt.figure(figsize=(9, 5))
        plt.hist(values, bins=40)
        plt.axvline(
            self.config.large_width_threshold_m,
            linestyle="--",
            label="large-width threshold",
        )
        plt.xlabel("Estimated carriageway width [m]")
        plt.ylabel("Number of OSM ways")
        plt.title("Distribution of estimated carriageway widths")
        plt.legend()
        plt.tight_layout()

        out = self.config.plot_dir / "carriageway_width_histogram.png"
        plt.savefig(out, dpi=200)
        plt.close()

        print(f"Saved histogram:\n{out}")

    def _plot_carriageway_by_highway(self, summary: pd.DataFrame) -> None:
        plot_df = summary.dropna(subset=["carriageway_width_m"]).copy()

        if plot_df.empty:
            return

        highway_order = (
            plot_df.groupby("highway")["carriageway_width_m"]
            .median()
            .sort_values()
            .index
            .tolist()
        )

        data = [
            plot_df.loc[
                plot_df["highway"] == highway,
                "carriageway_width_m",
            ].values
            for highway in highway_order
        ]

        plt.figure(figsize=(11, 6))
        plt.boxplot(data, labels=highway_order, showfliers=True)
        plt.axhline(
            self.config.large_width_threshold_m,
            linestyle="--",
            label="large-width threshold",
        )
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("Estimated carriageway width [m]")
        plt.title("Estimated carriageway widths by OSM highway class")
        plt.legend()
        plt.tight_layout()

        out = self.config.plot_dir / "carriageway_width_by_highway_boxplot.png"
        plt.savefig(out, dpi=200)
        plt.close()

        print(f"Saved boxplot:\n{out}")

    def _plot_carriageway_vs_street_space(self, summary: pd.DataFrame) -> None:
        plot_df = summary.dropna(
            subset=["carriageway_width_m", "street_space_width_m"]
        ).copy()

        if plot_df.empty:
            return

        plt.figure(figsize=(6, 6))
        plt.scatter(
            plot_df["carriageway_width_m"],
            plot_df["street_space_width_m"],
            s=10,
            alpha=0.6,
        )

        max_val = max(
            plot_df["carriageway_width_m"].max(),
            plot_df["street_space_width_m"].max(),
        )

        plt.plot(
            [0, max_val],
            [0, max_val],
            linestyle="--",
            label="1:1 line",
        )

        plt.xlabel("Carriageway width [m]")
        plt.ylabel("Street-space width [m]")
        plt.title("Carriageway width vs. street-space width")
        plt.legend()
        plt.tight_layout()

        out = self.config.plot_dir / "carriageway_vs_street_space.png"
        plt.savefig(out, dpi=200)
        plt.close()

        print(f"Saved scatter plot:\n{out}")
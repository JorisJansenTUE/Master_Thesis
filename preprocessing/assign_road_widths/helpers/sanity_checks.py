# =============================================================================
# SANITY CHECKS
# =============================================================================

from typing import Dict, Set

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from preprocessing.assign_road_widths.width_config import (
    WidthEstimationConfig,
)
from preprocessing.assign_road_widths.helpers.width_estimation import (
    WayWidthResult,
)


class SanityChecker:
    """
    Creates tabular and visual sanity checks for carriageway widths.

    The checker distinguishes between:
    - raw cadastral width;
    - final post-processed width;
    - corrected and protected roads.
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
        print(
            "\n===================== Sanity Checks "
            "====================="
        )
        print("Running sanity checks...")

        summary = self._build_summary_table()

        self._print_coverage(summary)
        self._print_missing_by_highway(summary)
        self._print_methods(summary)
        self._print_optimizable(summary)
        self._print_large_widths(summary)
        self._print_small_widths(summary)
        self._print_largest_corrections(summary)

        self._save_outputs(summary)

        return summary

    def _build_summary_table(self) -> pd.DataFrame:
        rows = []

        for way_id in sorted(self.expected_way_ids):
            result = self.width_results.get(way_id)

            row = {
                "way_id": way_id,
                "has_result": result is not None,
                "has_raw_width": False,
                "has_final_width": False,
                "highway": None,
                "name": None,
                "raw_width_m": np.nan,
                "final_width_m": np.nan,
                "width_change_m": np.nan,
                "carriageway_p10_m": np.nan,
                "carriageway_p90_m": np.nan,
                "carriageway_spread_m": np.nan,
                "carriageway_n": 0,
                "method": None,
                "optimizable": False,
                "width_changed": False,
            }

            if result is None:
                rows.append(row)
                continue

            row["highway"] = result.highway
            row["name"] = result.name
            row["method"] = result.method
            row["optimizable"] = result.optimizable

            if result.final_width_m is not None:
                row["has_final_width"] = True
                row["final_width_m"] = result.final_width_m

            stats = result.carriageway

            if stats is not None:
                row["has_raw_width"] = True
                row["raw_width_m"] = stats.median
                row["carriageway_p10_m"] = stats.p10
                row["carriageway_p90_m"] = stats.p90
                row["carriageway_spread_m"] = stats.spread
                row["carriageway_n"] = stats.n

            if (
                result.final_width_m is not None
                and stats is not None
            ):
                change = result.final_width_m - stats.median

                row["width_change_m"] = change
                row["width_changed"] = not np.isclose(
                    result.final_width_m,
                    stats.median,
                    atol=0.01,
                )

            elif (
                result.final_width_m is not None
                and stats is None
            ):
                # Final width was generated without a raw cadastral estimate.
                row["width_changed"] = True

            rows.append(row)

        return pd.DataFrame(rows)

    def _print_coverage(
        self,
        summary: pd.DataFrame,
    ) -> None:

        n_expected = len(summary)
        n_raw = int(summary["has_raw_width"].sum())
        n_final = int(summary["has_final_width"].sum())
        n_missing = n_expected - n_final

        print(f"Road ways that should have width: {n_expected}")
        print(f"Road ways with raw measurement:   {n_raw}")
        print(f"Road ways with final width:       {n_final}")
        print(f"Road ways missing final width:    {n_missing}")

        if n_expected > 0:
            raw_coverage = 100.0 * n_raw / n_expected
            final_coverage = 100.0 * n_final / n_expected

            print(f"Raw cadastral coverage:   {raw_coverage:.1f}%")
            print(f"Final width coverage:     {final_coverage:.1f}%")

    @staticmethod
    def _print_missing_by_highway(
        summary: pd.DataFrame,
    ) -> None:

        missing = summary[
            ~summary["has_final_width"]
        ]

        print("\nMissing final widths by highway type:")

        if missing.empty:
            print("None")
            return

        print(
            missing["highway"]
            .value_counts(dropna=False)
            .to_string()
        )

    @staticmethod
    def _print_methods(
        summary: pd.DataFrame,
    ) -> None:

        methods = (
            summary["method"]
            .fillna("no_result")
            .value_counts(dropna=False)
        )

        print("\nFinal width methods:")
        print(methods.to_string())

        n_changed = int(summary["width_changed"].sum())

        print(
            f"\nRoads whose final width differs "
            f"from the raw width: {n_changed}"
        )

    @staticmethod
    def _print_optimizable(
        summary: pd.DataFrame,
    ) -> None:

        with_width = summary[
            summary["has_final_width"]
        ]

        n_optimizable = int(
            with_width["optimizable"].sum()
        )
        n_protected = len(with_width) - n_optimizable

        print(f"\nOptimizable roads:     {n_optimizable}")
        print(f"Non-optimizable roads: {n_protected}")

        protected = with_width[
            ~with_width["optimizable"]
        ]

        if not protected.empty:
            print(
                protected[
                    [
                        "way_id",
                        "highway",
                        "name",
                        "final_width_m",
                        "method",
                    ]
                ]
                .head(30)
                .to_string(index=False)
            )

    def _print_large_widths(
        self,
        summary: pd.DataFrame,
    ) -> None:

        threshold = self.config.large_width_threshold_m

        large = summary[
            summary["final_width_m"] > threshold
        ].copy()

        print(
            f"\nRoads with final carriageway width "
            f"> {threshold} m: {len(large)}"
        )

        if large.empty:
            return

        print(
            large[
                [
                    "way_id",
                    "highway",
                    "name",
                    "raw_width_m",
                    "final_width_m",
                    "carriageway_spread_m",
                    "carriageway_n",
                    "method",
                    "optimizable",
                ]
            ]
            .sort_values(
                "final_width_m",
                ascending=False,
            )
            .head(30)
            .to_string(index=False)
        )

    def _print_small_widths(
        self,
        summary: pd.DataFrame,
    ) -> None:

        threshold = self.config.small_width_threshold_m

        small = summary[
            summary["has_final_width"]
            & (
                summary["final_width_m"] < threshold
            )
        ].copy()

        print(
            f"\nRoads with final carriageway width "
            f"< {threshold} m: {len(small)}"
        )

        if small.empty:
            return

        print(
            small[
                [
                    "way_id",
                    "highway",
                    "name",
                    "raw_width_m",
                    "final_width_m",
                    "carriageway_spread_m",
                    "carriageway_n",
                    "method",
                ]
            ]
            .sort_values("final_width_m")
            .head(30)
            .to_string(index=False)
        )

    @staticmethod
    def _print_largest_corrections(
        summary: pd.DataFrame,
    ) -> None:

        corrected = summary[
            summary["width_changed"]
        ].copy()

        print(
            f"\nPost-processed width corrections: "
            f"{len(corrected)}"
        )

        comparable = corrected.dropna(
            subset=[
                "raw_width_m",
                "final_width_m",
            ]
        ).copy()

        if comparable.empty:
            return

        comparable["width_change_m"] = (
            comparable["final_width_m"]
            - comparable["raw_width_m"]
        )

        comparable["absolute_width_change_m"] = (
            comparable["width_change_m"].abs()
        )

        # Sort before selecting the columns to print.
        comparable = comparable.sort_values(
            "absolute_width_change_m",
            ascending=False,
        )

        print("\nLargest absolute corrections:")

        print(
            comparable[
                [
                    "way_id",
                    "highway",
                    "name",
                    "raw_width_m",
                    "final_width_m",
                    "width_change_m",
                    "carriageway_spread_m",
                    "carriageway_n",
                    "method",
                ]
            ]
            .head(30)
            .to_string(index=False)
        )

    def _save_outputs(
        self,
        summary: pd.DataFrame,
    ) -> None:

        self.config.plot_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        summary_csv = (
            self.config.plot_dir
            / "width_summary_by_way.csv"
        )

        summary.to_csv(
            summary_csv,
            index=False,
        )

        print(
            f"\nSaved summary table:\n"
            f"{summary_csv}"
        )

        self._plot_final_width_histogram(summary)
        self._plot_final_width_by_highway(summary)
        self._plot_raw_vs_final_width(summary)
        self._plot_width_methods(summary)

    def _plot_final_width_histogram(
        self,
        summary: pd.DataFrame,
    ) -> None:

        values = summary["final_width_m"].dropna()

        if values.empty:
            return

        plt.figure(figsize=(9, 5))
        plt.hist(values, bins=40)

        plt.axvline(
            self.config.large_width_threshold_m,
            linestyle="--",
            label="large-width threshold",
        )

        plt.xlabel("Final carriageway width [m]")
        plt.ylabel("Number of OSM ways")
        plt.title("Distribution of final carriageway widths")
        plt.legend()
        plt.tight_layout()

        output = (
            self.config.plot_dir
            / "final_carriageway_width_histogram.png"
        )

        plt.savefig(
            output,
            dpi=200,
        )
        plt.close()

        print(f"Saved histogram:\n{output}")

    def _plot_final_width_by_highway(
        self,
        summary: pd.DataFrame,
    ) -> None:

        plot_df = summary.dropna(
            subset=[
                "final_width_m",
                "highway",
            ]
        ).copy()

        if plot_df.empty:
            return

        highway_order = (
            plot_df
            .groupby("highway")["final_width_m"]
            .median()
            .sort_values()
            .index
            .tolist()
        )

        data = [
            plot_df.loc[
                plot_df["highway"] == highway,
                "final_width_m",
            ].values
            for highway in highway_order
        ]

        plt.figure(figsize=(11, 6))

        plt.boxplot(
            data,
            labels=highway_order,
            showfliers=True,
        )

        plt.axhline(
            self.config.large_width_threshold_m,
            linestyle="--",
            label="large-width threshold",
        )

        plt.xticks(
            rotation=45,
            ha="right",
        )
        plt.ylabel("Final carriageway width [m]")
        plt.title("Final carriageway widths by highway class")
        plt.legend()
        plt.tight_layout()

        output = (
            self.config.plot_dir
            / "final_width_by_highway_boxplot.png"
        )

        plt.savefig(
            output,
            dpi=200,
        )
        plt.close()

        print(f"Saved boxplot:\n{output}")

    def _plot_raw_vs_final_width(
        self,
        summary: pd.DataFrame,
    ) -> None:

        plot_df = summary.dropna(
            subset=[
                "raw_width_m",
                "final_width_m",
            ]
        ).copy()

        if plot_df.empty:
            return

        plt.figure(figsize=(6, 6))

        plt.scatter(
            plot_df["raw_width_m"],
            plot_df["final_width_m"],
            s=12,
            alpha=0.6,
        )

        max_value = max(
            plot_df["raw_width_m"].max(),
            plot_df["final_width_m"].max(),
        )

        plt.plot(
            [0, max_value],
            [0, max_value],
            linestyle="--",
            label="unchanged width",
        )

        plt.xlabel("Raw cadastral median [m]")
        plt.ylabel("Final carriageway width [m]")
        plt.title("Raw versus post-processed width")
        plt.legend()
        plt.tight_layout()

        output = (
            self.config.plot_dir
            / "raw_vs_final_width.png"
        )

        plt.savefig(
            output,
            dpi=200,
        )
        plt.close()

        print(
            f"Saved raw/final comparison:\n"
            f"{output}"
        )

    def _plot_width_methods(
        self,
        summary: pd.DataFrame,
    ) -> None:

        method_counts = (
            summary.loc[
                summary["has_final_width"],
                "method",
            ]
            .fillna("unknown")
            .value_counts()
        )

        if method_counts.empty:
            return

        plt.figure(figsize=(9, 5))

        plt.bar(
            method_counts.index,
            method_counts.values,
        )

        plt.xticks(
            rotation=45,
            ha="right",
        )
        plt.xlabel("Width method")
        plt.ylabel("Number of OSM ways")
        plt.title("Final carriageway-width methods")
        plt.tight_layout()

        output = (
            self.config.plot_dir
            / "width_method_counts.png"
        )

        plt.savefig(
            output,
            dpi=200,
        )
        plt.close()

        print(f"Saved method counts:\n{output}")
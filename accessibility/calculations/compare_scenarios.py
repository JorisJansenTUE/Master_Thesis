
"""Compare accessibility scenarios against the baseline.

Run:
    python -m accessibility.calculations.compare_scenarios
"""

from pathlib import Path

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.cm import ScalarMappable
import numpy as np
import pandas as pd


# ============================================================
# SETTINGS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

OUTPUT_ROOT = PROJECT_ROOT / "data" / "accessibility" / "outputs"

BASELINE = "baseline_00000"
SCENARIOS = (
    "cluster_00051",
    "corridor_00792",
)

MODES = ("car", "bike", "pt")
THRESHOLDS_MIN = (15, 20, 30)

# Use a robust colour scale to make local changes visible.
# 100 = full range; 98 = clip the most extreme 2% of values.
COLOUR_PERCENTILE = 98

SAVE_DIR = OUTPUT_ROOT / "scenario_comparisons"


# ============================================================
# LOAD RESULTS
# ============================================================

def load_results(scenario):
    path = (
        OUTPUT_ROOT
        / f"results_{scenario}"
        / "accessibility_by_hexagon.csv"
    )
    return pd.read_csv(path, dtype={"hex_id": str})


def load_hexagons():
    path = (
        OUTPUT_ROOT
        / f"results_{BASELINE}"
        / "accessibility_hexagons.gpkg"
    )

    hexagons = gpd.read_file(
        path,
        layer="accessibility_hexagons",
    )

    return hexagons[["hex_id", "geometry"]].copy()


def compare_mode(baseline, scenario, hexagons, mode, column):
    """Calculate scenario minus baseline for each hexagon."""

    base = baseline.loc[
        baseline["mode"] == mode,
        ["hex_id", "attached", column],
    ].rename(columns={
        "attached": "baseline_attached",
        column: "baseline_jobs",
    })

    new = scenario.loc[
        scenario["mode"] == mode,
        ["hex_id", "attached", column],
    ].rename(columns={
        "attached": "scenario_attached",
        column: "scenario_jobs",
    })

    result = hexagons.merge(
        base, on="hex_id", how="left", validate="one_to_one"
    ).merge(
        new, on="hex_id", how="left", validate="one_to_one"
    )

    # Compare only origins attached in both scenarios.
    valid = (
        result["baseline_attached"].eq(True)
        & result["scenario_attached"].eq(True)
    )

    result["delta_jobs"] = (
        result["scenario_jobs"] - result["baseline_jobs"]
    ).where(valid)

    return result


# ============================================================
# PLOTTING
# ============================================================

def plot_comparison(baseline, scenarios, hexagons, minutes):
    column = f"jobs_{minutes:g}min"

    n_modes = len(MODES)
    n_scenarios = len(scenarios)

    fig, axes = plt.subplots(
        n_modes,
        n_scenarios,
        figsize=(7 * n_scenarios, 4.8 * n_modes),
        squeeze=False,
        constrained_layout=True,
    )

    for row, mode in enumerate(MODES):

        comparisons = {
            name: compare_mode(
                baseline, data, hexagons, mode, column
            )
            for name, data in scenarios.items()
        }

        # Same colour scale for the mode across all scenarios.
        all_changes = np.concatenate([
            df["delta_jobs"].dropna().to_numpy(dtype=float)
            for df in comparisons.values()
        ])

        if len(all_changes):
            limit = np.percentile(
                np.abs(all_changes),
                COLOUR_PERCENTILE,
            )
            limit = max(float(limit), 1.0)
        else:
            limit = 1.0

        norm = TwoSlopeNorm(
            vmin=-limit,
            vcenter=0,
            vmax=limit,
        )

        for col, (name, result) in enumerate(comparisons.items()):
            ax = axes[row, col]

            result.plot(
                column="delta_jobs",
                ax=ax,
                cmap="RdBu",
                norm=norm,
                linewidth=0.08,
                edgecolor="#666666",
                missing_kwds={"color": "#dddddd"},
            )

            changes = result["delta_jobs"].dropna()

            improved = int((changes > 0).sum())
            worsened = int((changes < 0).sum())
            unchanged = int((changes == 0).sum())

            mean = changes.mean()
            median = changes.median()

            ax.set_title(
                f"{mode.upper()} — {name}\n"
                f"Mean: {mean:+,.0f} jobs | "
                f"Median: {median:+,.0f}\n"
                f"Improved: {improved} | "
                f"Worsened: {worsened} | "
                f"Unchanged: {unchanged}",
                fontsize=10,
            )

            ax.set_axis_off()

            print(
                f"{minutes:2g} min | {mode:4s} | {name:16s} | "
                f"mean {mean:+9.1f} | median {median:+9.1f} | "
                f"min {changes.min():+9.0f} | "
                f"max {changes.max():+9.0f}"
            )

        # One shared colourbar per mode.
        sm = ScalarMappable(norm=norm, cmap="RdBu")
        sm.set_array([])

        fig.colorbar(
            sm,
            ax=axes[row, :].tolist(),
            fraction=0.025,
            pad=0.02,
            label=f"{mode.upper()}: change in accessible jobs",
        )

    fig.suptitle(
        f"Accessibility change relative to {BASELINE}\n"
        f"Employment reachable within {minutes:g} minutes",
        fontsize=15,
        fontweight="bold",
    )

    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    output = SAVE_DIR / f"accessibility_difference_{minutes:g}min.png"

    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {output}")


# ============================================================
# MAIN
# ============================================================

def main():
    baseline = load_results(BASELINE)
    hexagons = load_hexagons()

    scenarios = {}
    for name in SCENARIOS:
        path = (
            OUTPUT_ROOT
            / f"results_{name}"
            / "accessibility_by_hexagon.csv"
        )
        if path.exists():
            scenarios[name] = load_results(name)
        else:
            print(f"Skipping {name}: results not found")

    if not scenarios:
        raise FileNotFoundError("No scenario accessibility results found.")

    for minutes in THRESHOLDS_MIN:
        print(f"\n--- {minutes:g}-minute accessibility ---")
        plot_comparison(
            baseline, scenarios, hexagons, minutes
        )


if __name__ == "__main__":
    main()

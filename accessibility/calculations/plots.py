"""Quick diagnostic figures for accessibility and sufficiency (no basemap)."""
from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")  # Works headlessly on Windows/SLURM.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

from .cumulative import CumulativeAccessibility


def plot_sufficiency_bars(summary: pd.DataFrame, output_dir: str | Path) -> None:
    """One chart per mode; percentage of attached origins above low/high levels."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for mode, group in summary.groupby("mode", sort=False):
        group = group.sort_values("time_min")
        ticks = np.arange(len(group))
        fig, ax = plt.subplots(figsize=(7.4, 4.4))
        ax.bar(ticks - 0.19, group["pct_sufficient_low"], width=0.38,
               label="Low: 50% of baseline car mean", color="#168477")
        ax.bar(ticks + 0.19, group["pct_sufficient_high"], width=0.38,
               label="High: 100% of baseline car mean", color="#5473ab")
        ax.set_xticks(ticks, [f"{t:g} min" for t in group["time_min"]])
        ax.set_ylim(0, 105)
        ax.set_ylabel("Attached origins meeting sufficiency (%)")
        ax.set_title(f"Job accessibility sufficiency — {mode.upper()}")
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
        ax.legend(loc="upper left", fontsize=9)
        n = group["n_attached"].iloc[0]
        total = group["n_origins"].iloc[0]
        ax.text(0.99, 0.02, f"{n:,} / {total:,} origins attached",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=9)
        fig.tight_layout()
        fig.savefig(output / f"sufficiency_{mode}.png", dpi=170)
        plt.close(fig)


def create_hexagon_geopackage(origins_gpkg: str | Path, scored: pd.DataFrame,
                               output_file: str | Path) -> gpd.GeoDataFrame:
    """Join all mode-level results to one hexagon per ID, then export as GPKG."""
    hexagons = gpd.read_file(origins_gpkg, layer="origins")
    if "hex_id" not in hexagons or hexagons["hex_id"].duplicated().any():
        raise ValueError("Input GeoPackage must contain unique hex_id values")
    hexagons["hex_id"] = hexagons["hex_id"].astype(str)
    expected = set(hexagons["hex_id"])
    for mode in scored["mode"].unique():
        mode_data = scored.loc[scored["mode"] == mode].drop(columns="mode").copy()
        if set(mode_data["hex_id"]) != expected:
            raise ValueError(f"{mode} results do not match hexagon IDs in origins.gpkg")
        columns = {col: f"{mode}_{col}" for col in mode_data if col != "hex_id"}
        mode_data = mode_data.rename(columns=columns)
        hexagons = hexagons.merge(mode_data, on="hex_id", how="left", validate="one_to_one")
    destination = Path(output_file)
    destination.parent.mkdir(parents=True, exist_ok=True)
    hexagons.to_file(destination, layer="accessibility_hexagons", driver="GPKG")
    return hexagons


def plot_accessibility_maps(hexagons: gpd.GeoDataFrame,
                            modes: tuple[str, ...], travel_time_min: float,
                            output_dir: str | Path) -> None:
    """Map absolute accessible jobs at one time cutoff; comparable colour scale."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    col = CumulativeAccessibility.column_name(travel_time_min)
    values_by_mode = [f"{mode}_{col}" for mode in modes]
    for value_col in values_by_mode:
        if value_col not in hexagons:
            raise ValueError(f"Missing map column {value_col}")
    all_values = np.concatenate([
        hexagons[value_col].dropna().to_numpy(dtype=float) for value_col in values_by_mode
    ])
    vmax = max(1.0, float(np.max(all_values))) if all_values.size else 1.0
    for mode, value_col in zip(modes, values_by_mode, strict=True):
        fig, ax = plt.subplots(figsize=(9.0, 6.0))
        hexagons.plot(column=value_col, ax=ax, cmap="viridis", vmin=0, vmax=vmax,
                      linewidth=0.08, edgecolor="#555555",
                      missing_kwds={"color": "#e3e3e3", "label": "Not attached"})
        ax.set_title(f"Accessible jobs within {travel_time_min:g} min — {mode.upper()}")
        ax.set_axis_off()
        scalar = ScalarMappable(norm=Normalize(vmin=0, vmax=vmax), cmap="viridis")
        scalar.set_array([])
        fig.colorbar(scalar, ax=ax, fraction=0.035, pad=0.02,
                     label="Reachable STATENT jobs (same scale for all modes)")
        fig.tight_layout()
        fig.savefig(output / f"accessibility_map_{mode}_{travel_time_min:g}min.png", dpi=180)
        plt.close(fig)

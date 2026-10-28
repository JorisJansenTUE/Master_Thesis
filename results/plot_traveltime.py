
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from preprocessing.utils import DATA_DIR, RAW_DIR

# ------------------------------------------------------------------
# Settings
# ------------------------------------------------------------------

SCENARIO = Path(
    DATA_DIR / "NDA_Eth_Model" / "Updated_router_results"
)

TRIPS_FILE = SCENARIO / "output_trips_REV.csv.gz"

TIME_OUTPUT_FILE = SCENARIO / "travel_times_by_mode_REV-only.png"
DISTANCE_OUTPUT_FILE = SCENARIO / "travel_distances_by_mode_REV-only.png"
SUMMARY_OUTPUT_FILE = SCENARIO / "travel_time_distance_histograms_REV-only.csv"

MODES_TO_PLOT = {
    "car": "Car",
    "bike": "Bicycle",
    "pt": "Public transport",
    "walk": "Walk",
}

MODE_MAPPING = {
    "car": "car",
    "car_loop": "car",
    "car_passenger": "car",
    "car_passenger_loop": "car",
    "truck": "car",

    "bike": "bike",
    "bike_loop": "bike",

    "pt": "pt",

    "walk": "walk",
    "walk_loop": "walk",
    "remote_walk": "walk",
}

# Histogram settings
TIME_BIN_WIDTH = 1          # minutes
DISTANCE_BIN_WIDTH = 0.5    # kilometres

REFERENCE_TIME = 20         # minutes
REFERENCE_DISTANCE = 5      # kilometres

# TIME_X_LIMITS = {
#     "car": (0, 45),
#     "bike": (0, 45),
#     "pt": (0, 60),
#     "walk": (0, 45),
# }

# DISTANCE_X_LIMITS = {
#     "car": (0, 22),
#     "bike": (0, 15),
#     "pt": (0, 22),
#     "walk": (0, 1),
# }

# Set either dictionary to None for automatic limits.
TIME_X_LIMITS = None
DISTANCE_X_LIMITS = None    

# ------------------------------------------------------------------
# Geographic filtering
# ------------------------------------------------------------------

FILTER_PROJECT_AREA = True

PROJECT_SHAPE = Path(
    RAW_DIR / "shapes" / "REV_Cintura_project.shp"
)

TRIPS_CRS = "EPSG:2056"


# ------------------------------------------------------------------
# Data processing
# ------------------------------------------------------------------

def matsim_time_to_seconds(values: pd.Series) -> pd.Series:
    """
    Convert MATSim travel times to seconds.

    Supports:
        452.0
        00:07:32
    """
    numeric_seconds = pd.to_numeric(values, errors="coerce")

    duration_seconds = pd.to_timedelta(
        values.astype(str),
        errors="coerce",
    ).dt.total_seconds()

    return numeric_seconds.fillna(duration_seconds)


def load_trips() -> pd.DataFrame:

    if not TRIPS_FILE.exists():
        raise FileNotFoundError(
            f"Trips file not found:\n{TRIPS_FILE.resolve()}"
        )

    trips = pd.read_csv(
        TRIPS_FILE,
        sep=";",
        compression="infer",
        low_memory=False,
    )

    required_columns = {"main_mode", "trav_time"}
    missing = required_columns - set(trips.columns)

    if missing:
        raise ValueError(
            f"Missing columns: {sorted(missing)}\n"
            f"Available columns: {list(trips.columns)}"
        )

    # Find distance column
    distance_candidates = [
        "traveled_distance",
        "travel_distance",
        "distance",
    ]

    distance_column = next(
        (c for c in distance_candidates if c in trips.columns),
        None,
    )

    if distance_column is None:
        raise ValueError(
            "No distance column found.\n"
            f"Available columns: {list(trips.columns)}"
        )

    print(f"Using distance column: {distance_column}")

    trips["main_mode_raw"] = (
        trips["main_mode"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    print("\nOriginal MATSim modes:")
    print(trips["main_mode_raw"].value_counts())

    trips["main_mode"] = (
        trips["main_mode_raw"].map(MODE_MAPPING)
    )

    unmapped = trips.loc[
        trips["main_mode"].isna(),
        "main_mode_raw",
    ].value_counts()

    if not unmapped.empty:
        print("\nExcluded/unmapped modes:")
        print(unmapped)

    trips = trips.loc[
        trips["main_mode"].notna()
    ].copy()

    # Travel time in minutes
    trips["travel_time_minutes"] = (
        matsim_time_to_seconds(trips["trav_time"]) / 60
    )

    # Distance in kilometres
    trips["travel_distance_km"] = (
        pd.to_numeric(
            trips[distance_column],
            errors="coerce",
        ) / 1000
    )

    # Keep trips with valid positive travel times.
    # Distance validity is handled independently.
    trips = trips.loc[
        np.isfinite(trips["travel_time_minutes"])
        & (trips["travel_time_minutes"] > 0)
    ].copy()

    print("\nAggregated modes:")
    print(trips["main_mode"].value_counts())

    return trips


def filter_trips_by_project_area(
    trips: pd.DataFrame,
) -> pd.DataFrame:
    """
    Keep trips for which both the origin and destination
    are inside the project boundary.

    Coordinates are taken from:
        start_x, start_y, end_x, end_y
    """

    if not FILTER_PROJECT_AREA:
        print("\nGeographic filtering disabled.")
        return trips

    if not PROJECT_SHAPE.exists():
        raise FileNotFoundError(
            f"Project shape not found: {PROJECT_SHAPE}"
        )

    required = {"start_x", "start_y", "end_x", "end_y"}
    missing = required - set(trips.columns)

    if missing:
        raise ValueError(
            f"Missing coordinate columns: {sorted(missing)}"
        )

    # Load project boundary
    boundary = gpd.read_file(PROJECT_SHAPE)

    if boundary.crs is None:
        raise ValueError(
            "Project shapefile has no defined CRS."
        )

    boundary = boundary.to_crs(TRIPS_CRS)

    # Combine all boundary geometries
    project_area = boundary.geometry.union_all()

    # Convert coordinates to points
    start_points = gpd.GeoSeries(
        gpd.points_from_xy(
            pd.to_numeric(trips["start_x"], errors="coerce"),
            pd.to_numeric(trips["start_y"], errors="coerce"),
        ),
        index=trips.index,
        crs=TRIPS_CRS,
    )

    end_points = gpd.GeoSeries(
        gpd.points_from_xy(
            pd.to_numeric(trips["end_x"], errors="coerce"),
            pd.to_numeric(trips["end_y"], errors="coerce"),
        ),
        index=trips.index,
        crs=TRIPS_CRS,
    )

    # Include points inside or exactly on the boundary
    start_inside = start_points.intersects(project_area)
    end_inside = end_points.intersects(project_area)

    # Both endpoints must be inside
    keep = start_inside & end_inside

    print("\nGeographic trip filtering:")
    print(f"Total trips: {len(trips):,}")
    print(f"Start inside: {start_inside.sum():,}")
    print(f"End inside: {end_inside.sum():,}")
    print(f"Both inside: {keep.sum():,}")
    print(f"Excluded: {(~keep).sum():,}")

    return trips.loc[keep].copy()

# ------------------------------------------------------------------
# Plotting
# ------------------------------------------------------------------


def plot_histograms(
    trips: pd.DataFrame,
    value_column: str,
    bin_width: float,
    reference_value: float,
    x_limits: dict | None,
    x_label: str,
    output_file: Path,
) -> pd.DataFrame:

    number_of_modes = len(MODES_TO_PLOT)

    fig, axes = plt.subplots(
        1,
        number_of_modes,
        figsize=(5.5 * number_of_modes, 4.5),
        squeeze=False,
    )

    axes = axes.flatten()

    histogram_data = {}

    for ax, (mode, title) in zip(
        axes,
        MODES_TO_PLOT.items(),
    ):

        mode_values = trips.loc[
            trips["main_mode"] == mode,
            value_column,
        ]

        mode_values = mode_values.loc[
            np.isfinite(mode_values) & (mode_values >= 0)
        ]

        if x_limits is not None:
            limits = x_limits.get(mode)
        else:
            limits = None

        if limits is None:
            minimum_value = 0
            maximum_value = (
                np.ceil(mode_values.max() / bin_width) * bin_width
                if not mode_values.empty else bin_width
            )
            maximum_value = max(maximum_value, bin_width)
        else:
            minimum_value, maximum_value = limits

        bins = np.arange(
            minimum_value,
            maximum_value + bin_width * 1.001,
            bin_width,
        )

        counts, edges = np.histogram(
            mode_values,
            bins=bins,
        )

        # Store the same counts used for plotting
        histogram_data[title] = pd.Series(
            counts,
            index=np.round(edges[:-1], 6),
        )

        if mode_values.empty:
            ax.set_title(f"{title}\n(no trips found)")
            ax.set_axis_off()
            continue

        bar_colours = [
            "#bdbdbd"
            if left_edge < reference_value
            else "#2a9d8f"
            for left_edge in edges[:-1]
        ]

        ax.bar(
            edges[:-1],
            counts,
            width=np.diff(edges),
            align="edge",
            color=bar_colours,
            edgecolor="white",
            linewidth=0.5,
        )

        ax.axvline(
            reference_value,
            color="#f2b705",
            linestyle="--",
            linewidth=2,
        )

        ax.set_title(title, loc="left")
        ax.set_xlabel(x_label)
        ax.set_ylabel("# trips")

        ax.set_xlim(minimum_value, maximum_value)
        ax.grid(axis="y", alpha=0.3)
        ax.set_axisbelow(True)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        print(
            f"{title}: {len(mode_values):,} trips, "
            f"mean = {mode_values.mean():.2f}, "
            f"median = {mode_values.median():.2f}"
        )

    fig.tight_layout()
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)

    print(f"\nFigure written to: {output_file.resolve()}")

    # Combine counts from all modes into one DataFrame
    result = pd.DataFrame(histogram_data).fillna(0).astype(int)
    result.index.name = "bracket_start"

    result = result.reset_index()
    result["bracket_end"] = result["bracket_start"] + bin_width

    return result[
        ["bracket_start", "bracket_end", *MODES_TO_PLOT.values()]
    ]





# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------


def main() -> None:

    trips = load_trips()

    print("\n--- Travel times ---")

    # Optionally keep only trips entirely within the project area
    trips = filter_trips_by_project_area(trips)

    time_data = plot_histograms(
        trips=trips,
        value_column="travel_time_minutes",
        bin_width=TIME_BIN_WIDTH,
        reference_value=REFERENCE_TIME,
        x_limits=TIME_X_LIMITS,
        x_label="Travel time (min)",
        output_file=TIME_OUTPUT_FILE,
    )

    time_data.insert(0, "metric", "time_min")

    print("\n--- Travel distances ---")

    distance_data = plot_histograms(
        trips=trips,
        value_column="travel_distance_km",
        bin_width=DISTANCE_BIN_WIDTH,
        reference_value=REFERENCE_DISTANCE,
        x_limits=DISTANCE_X_LIMITS,
        x_label="Travel distance (km)",
        output_file=DISTANCE_OUTPUT_FILE,
    )

    distance_data.insert(0, "metric", "distance_km")

    # Combine both histograms into one CSV
    histogram_data = pd.concat(
        [time_data, distance_data],
        ignore_index=True,
    )

    histogram_data.to_csv(
        SUMMARY_OUTPUT_FILE,
        index=False,
    )

    print(
        f"\nHistogram data written to: "
        f"{SUMMARY_OUTPUT_FILE.resolve()}"
    )


if __name__ == "__main__":
    main()

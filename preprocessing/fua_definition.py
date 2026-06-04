from pathlib import Path
from collections import deque

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import contextily as ctx

from shapely.geometry import box
from matplotlib.patches import Patch

from preprocessing.utils import PROJECT_ROOT, DATA_DIR, RAW_DIR

# =============================================================================
# SETTINGS
# =============================================================================

INPUT_CSV = Path(r"C:\Users\20201733\Downloads\ag-b-00.03-vz2024statpop\STATPOP2024.csv")
DISTRICT_SHAPE = Path(f"{RAW_DIR}/shapes/Locarno_Boundary.shp")

OUTPUT_PNG = Path(f"{RAW_DIR}/clusters/statpop_smoothed_1ha_clusters_v2.png")

INPUT_CRS = "EPSG:2056"

X_COL = "E_KOORD"
Y_COL = "N_KOORD"
POP_COL = "BBTOT"

GRID_SIZE = 100  # 1 ha = 100 m × 100 m

# Moving window: 10 × 10 cells = 1 km²
WINDOW_CELLS_BEFORE = 5
WINDOW_CELLS_AFTER = 4

# OECD/Eurostat-inspired thresholds applied to local 1 km² window population
URBAN_CLUSTER_DENSITY_THRESHOLD = 300
URBAN_CLUSTER_POP_THRESHOLD = 5000

URBAN_CENTRE_DENSITY_THRESHOLD = 1500
URBAN_CENTRE_POP_THRESHOLD = 25000

GAP_FILL_MIN_NEIGHBOURS = 5

USE_QUEEN_CONTIGUITY = False  # If True, use 8-neighbour contiguity. If False, use 4-neighbour contiguity.


# =============================================================================
# LOADING
# =============================================================================

def load_statpop_csv(path: Path) -> pd.DataFrame:
    """
    Load STATPOP CSV.
    Swiss STATPOP CSVs are usually semicolon-separated.
    """

    if not path.exists():
        raise FileNotFoundError(f"STATPOP CSV not found: {path}")

    df = pd.read_csv(
        path,
        sep=";",
        quotechar='"',
        encoding="utf-8",
    )

    required_cols = [X_COL, Y_COL, POP_COL]
    missing = [col for col in required_cols if col not in df.columns]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}\n"
            f"Available columns:\n{list(df.columns)}"
        )

    df = df[[X_COL, Y_COL, POP_COL]].copy()

    df[X_COL] = pd.to_numeric(df[X_COL], errors="coerce")
    df[Y_COL] = pd.to_numeric(df[Y_COL], errors="coerce")
    df[POP_COL] = pd.to_numeric(df[POP_COL], errors="coerce").fillna(0)

    df = df.dropna(subset=[X_COL, Y_COL])

    return df


def load_study_extent_as_bbox(path: Path, target_crs: str = INPUT_CRS) -> gpd.GeoDataFrame:
    """
    Load the district shape, but return its rectangular bounding box.

    This avoids the U-shaped district geometry cutting out the area in the middle.
    """

    if not path.exists():
        raise FileNotFoundError(f"District shape not found: {path}")

    district = gpd.read_file(path)

    if district.crs is None:
        raise ValueError("District shape has no CRS. Assign the CRS first.")

    district = district.to_crs(target_crs)
    district = district.dissolve()

    minx, miny, maxx, maxy = district.total_bounds

    bbox = gpd.GeoDataFrame(
        {"name": ["study_extent_bbox"]},
        geometry=[box(minx, miny, maxx, maxy)],
        crs=target_crs,
    )

    return bbox


# =============================================================================
# GRID CREATION
# =============================================================================

def create_complete_1ha_grid(study_extent: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Create a complete 100 m × 100 m grid over the study extent.

    This is needed because STATPOP only contains cells with data.
    Missing cells should still exist in the grid and receive population = 0.
    """

    minx, miny, maxx, maxy = study_extent.total_bounds

    x_start = int(np.floor(minx / GRID_SIZE) * GRID_SIZE)
    y_start = int(np.floor(miny / GRID_SIZE) * GRID_SIZE)
    x_end = int(np.ceil(maxx / GRID_SIZE) * GRID_SIZE)
    y_end = int(np.ceil(maxy / GRID_SIZE) * GRID_SIZE)

    cells = []

    for x in range(x_start, x_end, GRID_SIZE):
        for y in range(y_start, y_end, GRID_SIZE):
            cells.append(
                {
                    "grid_x": x,
                    "grid_y": y,
                    "geometry": box(x, y, x + GRID_SIZE, y + GRID_SIZE),
                }
            )

    grid = gpd.GeoDataFrame(cells, geometry="geometry", crs=study_extent.crs)

    return grid


def prepare_statpop_grid_table(statpop: pd.DataFrame) -> pd.DataFrame:
    """
    Convert STATPOP coordinate rows to grid indices.


    Note: E_KOORD and N_KOORD are upper-right coordinates of the 100 m cell, should be shifted.
    """

    df = statpop.copy()

    #shift to bottom-left corner and convert to grid indices
    df["grid_x"] = df[X_COL].astype(int)-100
    df["grid_y"] = df[Y_COL].astype(int)-100
    df["population"] = df[POP_COL].fillna(0)

    return df[["grid_x", "grid_y", "population"]]


def join_population_to_complete_grid(
    grid: gpd.GeoDataFrame,
    statpop_table: pd.DataFrame,
) -> gpd.GeoDataFrame:
    """
    Join STATPOP population to complete grid.
    Missing cells receive population = 0.
    """

    result = grid.merge(
        statpop_table,
        on=["grid_x", "grid_y"],
        how="left",
    )

    result["population"] = result["population"].fillna(0)

    return gpd.GeoDataFrame(result, geometry="geometry", crs=grid.crs)


# =============================================================================
# SMOOTHING
# =============================================================================

def add_1km_window_population(grid: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Compute population in a moving 1 km × 1 km window around each 1 ha cell.

    Output:
        pop_1km_window

    Since the window is 1 km², this value is directly comparable to
    people/km² thresholds.
    """

    grid = grid.copy()

    x_values = np.sort(grid["grid_x"].unique())
    y_values = np.sort(grid["grid_y"].unique())

    x_to_col = {x: i for i, x in enumerate(x_values)}
    y_to_row = {y: i for i, y in enumerate(y_values)}

    n_rows = len(y_values)
    n_cols = len(x_values)

    pop_array = np.zeros((n_rows, n_cols), dtype=float)

    for row in grid.itertuples():
        r = y_to_row[row.grid_y]
        c = x_to_col[row.grid_x]
        pop_array[r, c] = row.population

    padded = np.pad(pop_array, pad_width=((1, 0), (1, 0)), mode="constant")
    integral = padded.cumsum(axis=0).cumsum(axis=1)

    def get_window_sum(r: int, c: int) -> float:
        r0 = max(r - WINDOW_CELLS_BEFORE, 0)
        r1 = min(r + WINDOW_CELLS_AFTER + 1, n_rows)

        c0 = max(c - WINDOW_CELLS_BEFORE, 0)
        c1 = min(c + WINDOW_CELLS_AFTER + 1, n_cols)

        return (
            integral[r1, c1]
            - integral[r0, c1]
            - integral[r1, c0]
            + integral[r0, c0]
        )

    values = []

    for row in grid.itertuples():
        r = y_to_row[row.grid_y]
        c = x_to_col[row.grid_x]
        values.append(get_window_sum(r, c))

    grid["pop_1km_window"] = values

    return grid


# =============================================================================
# CLASSIFICATION, GAP FILLING, CLUSTERING
# =============================================================================

def neighbour_offsets() -> list[tuple[int, int]]:
    """
    Return neighbour offsets for 1 ha grid cells.
    """

    if USE_QUEEN_CONTIGUITY:
        return [
            (-GRID_SIZE, -GRID_SIZE), (0, -GRID_SIZE), (GRID_SIZE, -GRID_SIZE),
            (-GRID_SIZE, 0),                            (GRID_SIZE, 0),
            (-GRID_SIZE, GRID_SIZE),  (0, GRID_SIZE),  (GRID_SIZE, GRID_SIZE),
        ]

    return [
                    (0, -GRID_SIZE),
        (-GRID_SIZE, 0),        (GRID_SIZE, 0),
                    (0, GRID_SIZE),
    ]


def add_candidate_column(
    grid: gpd.GeoDataFrame,
    threshold: float,
    output_col: str,
) -> gpd.GeoDataFrame:
    """
    Classify candidate cells based on smoothed 1 km² population.
    """

    grid = grid.copy()
    grid[output_col] = grid["pop_1km_window"] >= threshold
    return grid


def fill_candidate_gaps(
    grid: gpd.GeoDataFrame,
    candidate_col: str,
    output_col: str,
    min_neighbours: int = GAP_FILL_MIN_NEIGHBOURS,
) -> gpd.GeoDataFrame:
    """
    Fill one-cell gaps.

    A non-candidate cell becomes candidate if at least min_neighbours
    of its 8 neighbouring cells are candidates.
    """

    grid = grid.copy()

    candidate_lookup = {
        (row.grid_x, row.grid_y): bool(getattr(row, candidate_col))
        for row in grid.itertuples()
    }

    offsets = neighbour_offsets()
    filled_values = []
    added_by_gap_fill = []

    for row in grid.itertuples():
        key = (row.grid_x, row.grid_y)
        is_candidate = candidate_lookup[key]

        if is_candidate:
            filled_values.append(True)
            added_by_gap_fill.append(False)
            continue

        x, y = key

        neighbour_count = 0

        for dx, dy in offsets:
            neighbour_key = (x + dx, y + dy)

            if candidate_lookup.get(neighbour_key, False):
                neighbour_count += 1

        fill_cell = neighbour_count >= min_neighbours

        filled_values.append(fill_cell)
        added_by_gap_fill.append(fill_cell)

    grid[output_col] = filled_values
    grid[f"{output_col}_added_by_gap_fill"] = added_by_gap_fill

    return grid


def assign_clusters(
    grid: gpd.GeoDataFrame,
    candidate_col: str,
    cluster_col: str,
) -> gpd.GeoDataFrame:
    """
    Assign connected-component cluster IDs to candidate cells.
    """

    grid = grid.copy()
    grid[cluster_col] = np.nan

    candidates = grid[grid[candidate_col]].copy()

    cell_to_index = {
        (row.grid_x, row.grid_y): idx
        for idx, row in candidates.iterrows()
    }

    offsets = neighbour_offsets()
    visited = set()
    cluster_id = 1

    for start_key in cell_to_index.keys():
        if start_key in visited:
            continue

        queue = deque([start_key])
        visited.add(start_key)
        cluster_indices = []

        while queue:
            current_key = queue.popleft()
            current_idx = cell_to_index[current_key]
            cluster_indices.append(current_idx)

            x, y = current_key

            for dx, dy in offsets:
                neighbour_key = (x + dx, y + dy)

                if neighbour_key in cell_to_index and neighbour_key not in visited:
                    visited.add(neighbour_key)
                    queue.append(neighbour_key)

        grid.loc[cluster_indices, cluster_col] = cluster_id
        cluster_id += 1

    return grid


def mark_valid_clusters(
    grid: gpd.GeoDataFrame,
    candidate_col: str,
    cluster_col: str,
    output_col: str,
    population_threshold: float,
) -> gpd.GeoDataFrame:
    """
    Mark cells that belong to clusters above the total population threshold.

    Cluster population is based on actual population, not smoothed population.
    """

    grid = grid.copy()
    grid[output_col] = False

    candidates = grid[grid[candidate_col] & grid[cluster_col].notna()].copy()

    if candidates.empty:
        return grid

    cluster_pop = (
        candidates
        .groupby(cluster_col)["population"]
        .sum()
    )

    valid_cluster_ids = cluster_pop[
        cluster_pop >= population_threshold
    ].index.tolist()

    grid[output_col] = grid[cluster_col].isin(valid_cluster_ids)

    return grid


def classify_cluster_type(
    grid: gpd.GeoDataFrame,
    density_threshold: float,
    population_threshold: float,
    name: str,
) -> gpd.GeoDataFrame:
    """
    Full classification pipeline for one cluster type.

    Creates:
        {name}_candidate_raw
        {name}_candidate
        {name}_candidate_added_by_gap_fill
        {name}_cluster_id
        is_{name}
    """

    raw_col = f"{name}_candidate_raw"
    candidate_col = f"{name}_candidate"
    cluster_col = f"{name}_cluster_id"
    final_col = f"is_{name}"

    grid = add_candidate_column(
        grid,
        threshold=density_threshold,
        output_col=raw_col,
    )

    grid = fill_candidate_gaps(
        grid,
        candidate_col=raw_col,
        output_col=candidate_col,
        min_neighbours=GAP_FILL_MIN_NEIGHBOURS,
    )

    grid = assign_clusters(
        grid,
        candidate_col=candidate_col,
        cluster_col=cluster_col,
    )

    grid = mark_valid_clusters(
        grid,
        candidate_col=candidate_col,
        cluster_col=cluster_col,
        output_col=final_col,
        population_threshold=population_threshold,
    )

    return grid


# =============================================================================
# PLOTTING
# =============================================================================

def plot_clusters(grid: gpd.GeoDataFrame, study_extent: gpd.GeoDataFrame, output_png: Path) -> None:
    """
    Create final static map using Contextily.
    """

    output_png.parent.mkdir(parents=True, exist_ok=True)

    grid_web = grid.to_crs(epsg=3857)
    extent_web = study_extent.to_crs(epsg=3857)

    fig, ax = plt.subplots(figsize=(11, 11))

    grid_web.plot(
        ax=ax,
        color="lightgrey",
        edgecolor="white",
        linewidth=0.05,
        alpha=0.15,
    )

    grid_web[grid_web["urban_cluster_candidate"]].plot(
        ax=ax,
        color="lightskyblue",
        edgecolor="none",
        alpha=0.45,
    )

    grid_web[grid_web["urban_cluster_candidate_added_by_gap_fill"]].plot(
        ax=ax,
        color="purple",
        edgecolor="none",
        alpha=0.8,
    )

    grid_web[grid_web["is_urban_cluster"]].plot(
        ax=ax,
        color="royalblue",
        edgecolor="none",
        alpha=0.65,
    )

    grid_web[grid_web["urban_centre_candidate"]].plot(
        ax=ax,
        color="orange",
        edgecolor="none",
        alpha=0.75,
    )

    grid_web[grid_web["is_urban_centre"]].plot(
        ax=ax,
        color="red",
        edgecolor="none",
        alpha=0.85,
    )

    extent_web.boundary.plot(
        ax=ax,
        color="black",
        linewidth=1.2,
        alpha=0.9,
    )

    ctx.add_basemap(
        ax,
        source=ctx.providers.OpenStreetMap.Mapnik,
        alpha=0.7,
    )

    legend_handles = [
        Patch(facecolor="red", label="Urban centre cluster ≥ 50,000 pop"),
        Patch(facecolor="orange", label="Local 1 km² pop ≥ 1,500"),
        Patch(facecolor="royalblue", label="Urban cluster ≥ 5,000 pop"),
        Patch(facecolor="lightskyblue", label="Local 1 km² pop ≥ 300"),
        Patch(facecolor="purple", label="Added by ≥5-neighbour gap fill"),
        Patch(facecolor="lightgrey", label="Other 1 ha cells"),
        Patch(facecolor="none", edgecolor="black", label="Study extent bounding box"),
    ]

    ax.legend(
        handles=legend_handles,
        loc="upper right",
        frameon=True,
        framealpha=0.9,
    )

    ax.set_title(
        "Smoothed 1 ha STATPOP clustering with gap filling",
        fontsize=14,
    )

    ax.set_axis_off()

    plt.tight_layout()
    plt.savefig(output_png, dpi=300)
    plt.close()


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    print("Loading STATPOP CSV...")
    statpop = load_statpop_csv(INPUT_CSV)

    print("Loading study extent as bounding box...")
    study_extent = load_study_extent_as_bbox(DISTRICT_SHAPE)

    print("Creating complete 1 ha grid...")
    complete_grid = create_complete_1ha_grid(study_extent)

    print("Preparing STATPOP population table...")
    statpop_table = prepare_statpop_grid_table(statpop)

    print("Joining population to complete grid...")
    grid = join_population_to_complete_grid(
        complete_grid,
        statpop_table,
    )

    print("Computing moving 1 km² population window...")
    grid = add_1km_window_population(grid)

    print("Classifying urban cluster...")
    grid = classify_cluster_type(
        grid,
        density_threshold=URBAN_CLUSTER_DENSITY_THRESHOLD,
        population_threshold=URBAN_CLUSTER_POP_THRESHOLD,
        name="urban_cluster",
    )

    print("Classifying urban centre...")
    grid = classify_cluster_type(
        grid,
        density_threshold=URBAN_CENTRE_DENSITY_THRESHOLD,
        population_threshold=URBAN_CENTRE_POP_THRESHOLD,
        name="urban_centre",
    )

    print("Creating map...")
    plot_clusters(grid, study_extent, OUTPUT_PNG)

    print("\nDone.")
    print(f"Map written to: {OUTPUT_PNG}")


if __name__ == "__main__":
    main()
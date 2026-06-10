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

# Administrative boundary steps
#--------------------------------
ADMIN_GPKG = Path(r"C:\Users\20201733\Downloads\swissboundaries3d_2026-01_2056_5728.gpkg.zip")


# Names as found in swissBOUNDARIES3D municipality layer, could differ per version
ADMIN_LAYER = "TLM_HOHEITSGEBIET"

ADMIN_ID_COL = "bfs_nummer"
ADMIN_NAME_COL = "name_1"

# Which classified grid cells define the local centre?
# Options from your script include:
#   "is_urban_cluster" 
#   "urban_cluster_candidate"
#   "urban_centre_candidate"
#   "is_urban_centre" -> Most inline with the Eurostat definition, but may need to be adapted for the Locarnese case.
CENTER_CELL_COL = "is_urban_centre" 

# Eurostat city rule: LAU is assigned to city/centre if >= 50% of population
# lives in the urban centre.
ADMIN_ASSIGNMENT_THRESHOLD = 0.50

OUTPUT_ADMIN_PNG = Path(f"{RAW_DIR}/clusters/statpop_admin_assignment.png")

# Comutting Zone Settings
# --------------------------------
COMMUTING_CSV = Path(r"C:\Users\20201733\Downloads\employed_persons.csv")

# Column names as found in BFS commuting CSV, could differ per version
COMMUTING_RESIDENCE_COL = "geo_comm_resid"
COMMUTING_WORK_COL = "geo_comm_work"
COMMUTING_YEAR_COL = "ref_year"
COMMUTING_COUNT_COL = "value"
COMMUTING_PERSPECTIVE_COL = "perspective"

COMMUTING_YEAR = 2020       #Other options in found data file are 2018 or 2014

#Eurostat commuting zone rule: LAU is assigned to commuting zone if >= 15% of employed residents commute to the centre.
COMMUTING_THRESHOLD = 0.15 

#Determine which perspective to use. 
#If the CSV contains both residence and work perspectives, use residence perspective
#Avoid double counting and is inline with Eurostat.
COMMUTING_PERSPECTIVE_VALUE = "R"
OUTPUT_FUA_PNG = Path(f"{RAW_DIR}/clusters/statpop_fua_assignment.png")


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

def load_admin_boundaries(
    path: Path,
    layer: str,
    study_extent: gpd.GeoDataFrame,
    target_crs: str = INPUT_CRS,
) -> gpd.GeoDataFrame:
    """
    Load local administrative boundaries from swissBOUNDARIES3D.

    The layer is clipped to the study extent bounding box for efficiency.
    """

    if not path.exists():
        raise FileNotFoundError(f"Administrative boundary file not found: {path}")

    admin = gpd.read_file(path, layer=layer)

    if admin.crs is None:
        raise ValueError("Administrative boundary layer has no CRS.")
    
    admin.columns = admin.columns.str.lower()
    admin = admin.to_crs(target_crs)

    # Keep only boundaries intersecting the study extent
    admin = gpd.overlay(
        admin,
        study_extent.to_crs(target_crs),
        how="intersection",
        keep_geom_type=True,
    )

    if ADMIN_ID_COL not in admin.columns:
        raise ValueError(
            f"Missing ADMIN_ID_COL '{ADMIN_ID_COL}'. "
            f"Available columns are:\n{list(admin.columns)}"
        )

    if ADMIN_NAME_COL not in admin.columns:
        raise ValueError(
            f"Missing ADMIN_NAME_COL '{ADMIN_NAME_COL}'. "
            f"Available columns are:\n{list(admin.columns)}"
        )

    admin = admin[[ADMIN_ID_COL, ADMIN_NAME_COL, "geometry"]].copy()

    return admin

def load_commuting_csv(path: Path) -> pd.DataFrame:
    """
    Load BFS residence-work commuting matrix.

    Expected structure:
        one row per residence municipality, work municipality, year
        with a commuter/employed-person count.
    """

    if not path.exists():
        raise FileNotFoundError(f"Commuting CSV not found: {path}")

    df = pd.read_csv(
        path,
        sep=",",
        quotechar='"',
        encoding="utf-8",
    )

    df.columns = (
        df.columns
        .str.strip()
        .str.replace('"', "", regex=False)
        .str.lower()
    )

    print("Available commuting columns:")
    print(df.columns.tolist())

    return df


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


    Note: E_KOORD and N_KOORD are bottom-left coordinates of the 100 m cell
    """

    df = statpop.copy()

    #convert to grid indices
    df["grid_x"] = df[X_COL].astype(int)
    df["grid_y"] = df[Y_COL].astype(int)
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
# Administrative Area Assignment
# =============================================================================

def assign_grid_cells_to_admin(
    grid: gpd.GeoDataFrame,
    admin: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """
    Assign each 1 ha grid cell to an administrative unit using its centroid.
    """

    grid = grid.copy()

    if grid.crs != admin.crs:
        admin = admin.to_crs(grid.crs)

    centroids = grid.copy()
    centroids["geometry"] = centroids.geometry.centroid

    joined = gpd.sjoin(
        centroids,
        admin[[ADMIN_ID_COL, ADMIN_NAME_COL, "geometry"]],
        how="left",
        predicate="within",
    )

    grid[ADMIN_ID_COL] = joined[ADMIN_ID_COL].values
    grid[ADMIN_NAME_COL] = joined[ADMIN_NAME_COL].values

    return grid

def classify_admin_units_by_centre_share(
    grid: gpd.GeoDataFrame,
    admin: gpd.GeoDataFrame,
    centre_cell_col: str = CENTER_CELL_COL,
    threshold: float = ADMIN_ASSIGNMENT_THRESHOLD,
) -> gpd.GeoDataFrame:
    """
    Assign administrative units to the local centre based on the share of their
    population living in centre grid cells.

    This follows the Eurostat logic:
        LAU is part of the city if at least 50% of its population lives
        in the urban centre.

    For the Locarnese case, centre_cell_col can be adapted because the area may
    not meet the formal 50,000-person urban-centre threshold.
    """

    if centre_cell_col not in grid.columns:
        raise ValueError(
            f"Missing centre cell column '{centre_cell_col}'. "
            f"Available columns are:\n{list(grid.columns)}"
        )

    grid = grid.copy()

    if ADMIN_ID_COL not in grid.columns:
        raise ValueError(
            "Grid cells have not yet been assigned to administrative units. "
            "Run assign_grid_cells_to_admin() first."
        )

    grid["centre_population"] = np.where(
        grid[centre_cell_col],
        grid["population"],
        0,
    )

    pop_by_admin = (
        grid.dropna(subset=[ADMIN_ID_COL])
        .groupby(ADMIN_ID_COL)
        .agg(
            total_population=("population", "sum"),
            centre_population=("centre_population", "sum"),
        )
        .reset_index()
    )

    pop_by_admin["centre_population_share"] = np.where(
        pop_by_admin["total_population"] > 0,
        pop_by_admin["centre_population"] / pop_by_admin["total_population"],
        0,
    )

    pop_by_admin["assigned_to_centre"] = (
        pop_by_admin["centre_population_share"] >= threshold
    )

    admin_out = admin.merge(
        pop_by_admin,
        on=ADMIN_ID_COL,
        how="left",
    )

    admin_out["total_population"] = admin_out["total_population"].fillna(0)
    admin_out["centre_population"] = admin_out["centre_population"].fillna(0)
    admin_out["centre_population_share"] = admin_out["centre_population_share"].fillna(0)
    admin_out["assigned_to_centre"] = admin_out["assigned_to_centre"].fillna(False)

    return admin_out

# =============================================================================
# Commuting Zone and FUA Assignment
# =============================================================================

def prepare_commuting_table(
    commuting_raw: pd.DataFrame,
    residence_col: str = COMMUTING_RESIDENCE_COL,
    work_col: str = COMMUTING_WORK_COL,
    count_col: str = COMMUTING_COUNT_COL,
    year_col: str = COMMUTING_YEAR_COL,
    year: int = COMMUTING_YEAR,
    perspective_col: str = COMMUTING_PERSPECTIVE_COL,
    perspective_value: str | None = None,
) -> pd.DataFrame:
    """
    Prepare BFS residence-work commuting matrix.

    Returns a clean table with:
        residence_bfs, work_bfs, commuters

    Notes:
        - Uses the residence perspective if perspective_value is provided.
        - Drops grouped 'other communes' / 'other cantons' rows by requiring
          numeric commune IDs.
    """

    df = commuting_raw.copy()

    df.columns = (
        df.columns
        .str.strip()
        .str.replace('"', "", regex=False)
        .str.lower()
    )

    required_cols = [residence_col, work_col, count_col, year_col]
    missing = [col for col in required_cols if col not in df.columns]

    if missing:
        raise ValueError(
            f"Missing commuting columns: {missing}\n"
            f"Available columns are:\n{list(df.columns)}"
        )

    # Filter year
    df[year_col] = pd.to_numeric(df[year_col], errors="coerce")
    df = df[df[year_col] == year].copy()

    # Filter residence/work perspective if requested
    if perspective_value is not None:
        if perspective_col not in df.columns:
            raise ValueError(
                f"Missing perspective column '{perspective_col}'. "
                f"Available columns are:\n{list(df.columns)}"
            )

        df[perspective_col] = df[perspective_col].astype(str).str.strip()
        df = df[df[perspective_col] == perspective_value].copy()

    out = df[[residence_col, work_col, count_col]].copy()

    out = out.rename(
        columns={
            residence_col: "residence_bfs",
            work_col: "work_bfs",
            count_col: "commuters",
        }
    )

    # Convert to numeric. Grouped categories such as "other communes" become NaN.
    out["residence_bfs"] = pd.to_numeric(out["residence_bfs"], errors="coerce")
    out["work_bfs"] = pd.to_numeric(out["work_bfs"], errors="coerce")
    out["commuters"] = pd.to_numeric(out["commuters"], errors="coerce").fillna(0)

    # Remove grouped / non-municipality rows
    out = out.dropna(subset=["residence_bfs", "work_bfs"])

    out["residence_bfs"] = out["residence_bfs"].astype(int)
    out["work_bfs"] = out["work_bfs"].astype(int)

    # If duplicate rows exist, aggregate them
    out = (
        out.groupby(["residence_bfs", "work_bfs"], as_index=False)
        .agg(commuters=("commuters", "sum"))
    )

    return out


def add_commuting_zone_assignment(
    admin_assigned: gpd.GeoDataFrame,
    commuting: pd.DataFrame,
    admin_id_col: str = ADMIN_ID_COL,
    threshold: float = COMMUTING_THRESHOLD,
) -> gpd.GeoDataFrame:
    """
    Add commuting-zone and FUA classification to administrative units.

    Assumes admin_assigned already contains:
        assigned_to_centre

    Creates:
        employed_residents
        commuters_to_centre
        share_to_centre
        assigned_to_commuting_zone
        assigned_to_fua
    """

    admin = admin_assigned.copy()

    if "assigned_to_centre" not in admin.columns:
        raise ValueError(
            "admin_assigned must contain 'assigned_to_centre'. "
            "Run classify_admin_units_by_centre_share() first."
        )

    city_ids = set(
        admin.loc[admin["assigned_to_centre"], admin_id_col]
        .dropna()
        .astype(int)
        .tolist()
    )

    if not city_ids:
        raise ValueError(
            "No administrative units are assigned to the centre. "
            "Cannot compute commuting zone."
        )

    commuting = commuting.copy()
    commuting["residence_bfs"] = commuting["residence_bfs"].astype(int)
    commuting["work_bfs"] = commuting["work_bfs"].astype(int)

    total_by_residence = (
        commuting
        .groupby("residence_bfs", as_index=False)
        .agg(employed_residents=("commuters", "sum"))
    )

    to_centre_by_residence = (
        commuting[commuting["work_bfs"].isin(city_ids)]
        .groupby("residence_bfs", as_index=False)
        .agg(commuters_to_centre=("commuters", "sum"))
    )

    shares = total_by_residence.merge(
        to_centre_by_residence,
        on="residence_bfs",
        how="left",
    )

    shares["commuters_to_centre"] = shares["commuters_to_centre"].fillna(0)

    shares["share_to_centre"] = np.where(
        shares["employed_residents"] > 0,
        shares["commuters_to_centre"] / shares["employed_residents"],
        0,
    )

    admin = admin.merge(
        shares,
        left_on=admin_id_col,
        right_on="residence_bfs",
        how="left",
    )

    admin["employed_residents"] = admin["employed_residents"].fillna(0)
    admin["commuters_to_centre"] = admin["commuters_to_centre"].fillna(0)
    admin["share_to_centre"] = admin["share_to_centre"].fillna(0)

    admin["assigned_to_commuting_zone"] = (
        (~admin["assigned_to_centre"])
        & (admin["share_to_centre"] >= threshold)
    )

    admin["assigned_to_fua"] = (
        admin["assigned_to_centre"]
        | admin["assigned_to_commuting_zone"]
    )

    return admin


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
        Patch(facecolor="red", label="Urban centre cluster ≥ 25,000 pop"),
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

def plot_admin_assignment(
    grid: gpd.GeoDataFrame,
    admin: gpd.GeoDataFrame,
    study_extent: gpd.GeoDataFrame,
    output_png: Path,
    centre_cell_col: str = CENTER_CELL_COL,
) -> None:
    """
    Plot administrative units assigned to the local centre.
    """

    output_png.parent.mkdir(parents=True, exist_ok=True)

    grid_web = grid.to_crs(epsg=3857)
    admin_web = admin.to_crs(epsg=3857)
    extent_web = study_extent.to_crs(epsg=3857)

    fig, ax = plt.subplots(figsize=(11, 11))

    # Assigned administrative units
    admin_web[~admin_web["assigned_to_centre"]].plot(
        ax=ax,
        color="lightgrey",
        edgecolor="white",
        linewidth=0.6,
        alpha=0.35,
    )

    admin_web[admin_web["assigned_to_centre"]].plot(
        ax=ax,
        color="lightskyblue",
        edgecolor="black",
        linewidth=1.0,
        alpha=0.55,
    )

    # Centre grid cells
    grid_web[grid_web[centre_cell_col]].plot(
        ax=ax,
        color="red",
        edgecolor="none",
        alpha=0.65,
    )

    # Administrative boundaries
    admin_web.boundary.plot(
        ax=ax,
        color="black",
        linewidth=0.5,
        alpha=0.8,
    )

    # Study extent
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
        Patch(facecolor="red", label=f"Centre cells: {centre_cell_col}"),
        Patch(facecolor="lightskyblue", edgecolor="black", label="Administrative unit assigned to centre"),
        Patch(facecolor="lightgrey", edgecolor="white", label="Other administrative unit"),
        Patch(facecolor="none", edgecolor="black", label="Study extent"),
    ]

    ax.legend(
        handles=legend_handles,
        loc="upper right",
        frameon=True,
        framealpha=0.9,
    )

    ax.set_title(
        "Administrative units assigned to the Locarnese centre",
        fontsize=14,
    )

    ax.set_axis_off()

    plt.tight_layout()
    plt.savefig(output_png, dpi=300)
    plt.close()

def plot_fua_assignment(
    admin_fua: gpd.GeoDataFrame,
    study_extent: gpd.GeoDataFrame,
    output_png: Path,
) -> None:
    """
    Plot adapted city, commuting zone, and resulting FUA.
    """

    output_png.parent.mkdir(parents=True, exist_ok=True)

    admin_web = admin_fua.to_crs(epsg=3857)
    extent_web = study_extent.to_crs(epsg=3857)

    fig, ax = plt.subplots(figsize=(11, 11))

    # Other municipalities
    admin_web[~admin_web["assigned_to_fua"]].plot(
        ax=ax,
        color="lightgrey",
        edgecolor="white",
        linewidth=0.6,
        alpha=0.35,
    )

    # Commuting zone
    admin_web[admin_web["assigned_to_commuting_zone"]].plot(
        ax=ax,
        color="lightskyblue",
        edgecolor="black",
        linewidth=0.8,
        alpha=0.6,
    )

    # City / centre municipalities
    admin_web[admin_web["assigned_to_centre"]].plot(
        ax=ax,
        color="red",
        edgecolor="black",
        linewidth=1.0,
        alpha=0.75,
    )

    admin_web.boundary.plot(
        ax=ax,
        color="black",
        linewidth=0.5,
        alpha=0.8,
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
        Patch(facecolor="red", edgecolor="black", label="Adapted Locarnese centre"),
        Patch(facecolor="lightskyblue", edgecolor="black", label="Commuting zone ≥ 15%"),
        Patch(facecolor="lightgrey", edgecolor="white", label="Outside adapted FUA"),
        Patch(facecolor="none", edgecolor="black", label="Study extent"),
    ]

    ax.legend(
        handles=legend_handles,
        loc="upper right",
        frameon=True,
        framealpha=0.9,
    )

    ax.set_title(
        "Adapted Locarnese functional urban area",
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

    print("Loading administrative boundaries...")
    admin = load_admin_boundaries(
        ADMIN_GPKG,
        ADMIN_LAYER,
        study_extent,
    )

    print("Assigning 1 ha cells to administrative units...")
    grid = assign_grid_cells_to_admin(
        grid,
        admin,
    )

    print("Classifying administrative units by centre population share...")
    admin_assigned = classify_admin_units_by_centre_share(
        grid,
        admin,
        centre_cell_col=CENTER_CELL_COL,
        threshold=ADMIN_ASSIGNMENT_THRESHOLD,
    )

    print("Creating administrative assignment map...")
    plot_admin_assignment(
        grid,
        admin_assigned,
        study_extent,
        OUTPUT_ADMIN_PNG,
        centre_cell_col=CENTER_CELL_COL,
    )

    print(f"Administrative assignment map written to: {OUTPUT_ADMIN_PNG}")

    print("Loading commuting data...")
    commuting_raw = load_commuting_csv(COMMUTING_CSV)

    print("Preparing commuting matrix...")
    commuting = prepare_commuting_table(
        commuting_raw,
        residence_col=COMMUTING_RESIDENCE_COL,
        work_col=COMMUTING_WORK_COL,
        count_col=COMMUTING_COUNT_COL,
        year_col=COMMUTING_YEAR_COL,
        year=COMMUTING_YEAR,
    )

    print("Assigning commuting zone...")
    admin_fua = add_commuting_zone_assignment(
        admin_assigned,
        commuting,
        admin_id_col=ADMIN_ID_COL,
        threshold=COMMUTING_THRESHOLD,
    )

    print("Creating FUA map...")
    plot_fua_assignment(
        admin_fua,
        study_extent,
        OUTPUT_FUA_PNG,
    )

    print(f"FUA map written to: {OUTPUT_FUA_PNG}")


if __name__ == "__main__":
    main()
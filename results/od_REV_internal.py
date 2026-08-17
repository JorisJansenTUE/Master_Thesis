from pathlib import Path

import geopandas as gpd
import pandas as pd

from preprocessing.utils import DATA_DIR


# ------------------------------------------------------------------
# Settings
# ------------------------------------------------------------------

SCENARIO = Path(
    DATA_DIR / "NDA_Eth_Model" / "Locarno" / "simulation_output"
)

TRIPS_FILE = SCENARIO / "output_trips.csv.gz"

BOUNDARIES_FILE = Path(
    DATA_DIR / "raw" / "shapes" / "swissBOUNDARIES3D.gpkg"
)

BOUNDARIES_LAYER = "tlm_hoheitsgebiet"
MUNICIPALITY_NAME_COLUMN = "name"


OUTPUT_CSV = SCENARIO / "mode_choice_od_table.csv"


# Municipalities shown in the table
MUNICIPALITIES = [
    "Ascona",
    "Locarno",
    "Minusio",
    "Muralto",
]


# Detailed MATSim modes -> four aggregated modes
MODE_MAPPING = {
    # Car
    "car": "Car",
    "car_loop": "Car",
    "car_passenger": "Car",
    "car_passenger_loop": "Car",

    # Bicycle
    "bike": "Bicycle",
    "bike_loop": "Bicycle",

    # Public transport
    "pt": "Public transport",

    # Walk
    "walk": "Walk",
    "walk_loop": "Walk",
    "remote_walk": "Walk",
}


MODE_ORDER = [
    "Walk",
    "Bicycle",
    "Car",
    "Public transport",
]


# ------------------------------------------------------------------
# Load municipality polygons
# ------------------------------------------------------------------

def load_municipalities() -> gpd.GeoDataFrame:

    municipalities = gpd.read_file(
        BOUNDARIES_FILE,
        layer=BOUNDARIES_LAYER,
    )

    municipalities = municipalities.loc[
        municipalities[MUNICIPALITY_NAME_COLUMN].isin(MUNICIPALITIES),
        [MUNICIPALITY_NAME_COLUMN, "geometry"],
    ].copy()

    # MATSim coordinates are Swiss LV95
    municipalities = municipalities.to_crs("EPSG:2056")

    found = set(municipalities[MUNICIPALITY_NAME_COLUMN])
    missing = set(MUNICIPALITIES) - found

    if missing:
        raise ValueError(
            f"Municipalities not found in boundary file: {sorted(missing)}"
        )

    return municipalities


# ------------------------------------------------------------------
# Assign municipality to coordinates
# ------------------------------------------------------------------

def assign_municipality(
    trips: pd.DataFrame,
    municipalities: gpd.GeoDataFrame,
    x_column: str,
    y_column: str,
) -> pd.Series:

    result = pd.Series(
        pd.NA,
        index=trips.index,
        dtype="object",
    )

    valid = (
        trips[x_column].notna()
        & trips[y_column].notna()
    )

    points = gpd.GeoDataFrame(
        index=trips.loc[valid].index,
        geometry=gpd.points_from_xy(
            trips.loc[valid, x_column],
            trips.loc[valid, y_column],
        ),
        crs="EPSG:2056",
    )

    joined = gpd.sjoin(
        points,
        municipalities,
        how="left",
        predicate="within",
    )

    result.loc[joined.index] = joined[
        MUNICIPALITY_NAME_COLUMN
    ]

    return result


# ------------------------------------------------------------------
# Create OD-mode table
# ------------------------------------------------------------------

def create_od_mode_table(trips: pd.DataFrame) -> pd.DataFrame:

    columns = pd.MultiIndex.from_product(
        [
            MUNICIPALITIES + ["Total"],
            ["n", "%"],
        ],
        names=["Destination", ""],
    )

    rows = []
    index = []

    # --------------------------------------------------------------
    # Individual origin municipalities
    # --------------------------------------------------------------

    for origin in MUNICIPALITIES:

        origin_trips = trips.loc[
            trips["origin"] == origin
        ]

        origin_total = len(origin_trips)

        # ----------------------------------------------------------
        # Mode rows
        # ----------------------------------------------------------

        for mode in MODE_ORDER:

            row = {}

            for destination in MUNICIPALITIES:

                destination_trips = origin_trips.loc[
                    origin_trips["destination"] == destination
                ]

                od_total = len(destination_trips)

                n = (
                    destination_trips["mode"] == mode
                ).sum()

                percentage = (
                    100 * n / od_total
                    if od_total > 0
                    else 0
                )

                row[(destination, "n")] = int(n)
                row[(destination, "%")] = percentage

            # Total across all destinations for this origin
            mode_total = (
                origin_trips["mode"] == mode
            ).sum()

            total_percentage = (
                100 * mode_total / origin_total
                if origin_total > 0
                else 0
            )

            row[("Total", "n")] = int(mode_total)
            row[("Total", "%")] = total_percentage

            rows.append(row)
            index.append((origin, mode))

        # ----------------------------------------------------------
        # Total row for this origin
        # ----------------------------------------------------------

        row = {}

        for destination in MUNICIPALITIES:

            n = (
                origin_trips["destination"] == destination
            ).sum()

            row[(destination, "n")] = int(n)
            row[(destination, "%")] = (
                100.0 if n > 0 else 0.0
            )

        row[("Total", "n")] = origin_total
        row[("Total", "%")] = (
            100.0 if origin_total > 0 else 0.0
        )

        rows.append(row)
        index.append((origin, "Total"))

    # --------------------------------------------------------------
    # Overall total across all origins
    # --------------------------------------------------------------

    total_trips = len(trips)

    for mode in MODE_ORDER:

        row = {}

        for destination in MUNICIPALITIES:

            destination_trips = trips.loc[
                trips["destination"] == destination
            ]

            destination_total = len(destination_trips)

            n = (
                destination_trips["mode"] == mode
            ).sum()

            percentage = (
                100 * n / destination_total
                if destination_total > 0
                else 0
            )

            row[(destination, "n")] = int(n)
            row[(destination, "%")] = percentage

        mode_total = (
            trips["mode"] == mode
        ).sum()

        total_percentage = (
            100 * mode_total / total_trips
            if total_trips > 0
            else 0
        )

        row[("Total", "n")] = int(mode_total)
        row[("Total", "%")] = total_percentage

        rows.append(row)
        index.append(("Total", mode))

    # Final total row
    row = {}

    for destination in MUNICIPALITIES:

        n = (
            trips["destination"] == destination
        ).sum()

        row[(destination, "n")] = int(n)
        row[(destination, "%")] = (
            100.0 if n > 0 else 0.0
        )

    row[("Total", "n")] = total_trips
    row[("Total", "%")] = (
        100.0 if total_trips > 0 else 0.0
    )

    rows.append(row)
    index.append(("Total", "Total"))

    table = pd.DataFrame(
        rows,
        index=pd.MultiIndex.from_tuples(
            index,
            names=["Origin", "Mode"],
        ),
    )

    return table.reindex(columns=columns)


# ------------------------------------------------------------------
# Write table
# ------------------------------------------------------------------

def write_outputs(table: pd.DataFrame) -> None:

    
    # --------------------------------------------------------------
    # CSV
    # Flatten the two-level column names
    # --------------------------------------------------------------

    csv_table = table.copy()

    csv_table.columns = [
        f"{destination}_{stat}"
        for destination, stat in csv_table.columns
    ]

    csv_table.to_csv(
        OUTPUT_CSV,
        sep=";",
    )

    print(f"\nCSV written to:\n{OUTPUT_CSV.resolve()}")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main() -> None:

    if not TRIPS_FILE.exists():
        raise FileNotFoundError(
            f"Trips file not found:\n{TRIPS_FILE.resolve()}"
        )

    if not BOUNDARIES_FILE.exists():
        raise FileNotFoundError(
            f"Boundary file not found:\n{BOUNDARIES_FILE.resolve()}"
        )

    # --------------------------------------------------------------
    # Read MATSim trips
    # --------------------------------------------------------------

    trips = pd.read_csv(
        TRIPS_FILE,
        sep=";",
        compression="infer",
        low_memory=False,
    )

    # Keep only home-to-work commuting trips
    trips = trips.loc[
        (trips["start_activity_type"] == "home")
        & (trips["end_activity_type"] == "work")
    ].copy()

    print(f"Home-to-work commuting trips: {len(trips):,}")

    required_columns = {
        "main_mode",
        "start_x",
        "start_y",
        "end_x",
        "end_y",
    }

    missing_columns = required_columns - set(trips.columns)

    if missing_columns:
        raise ValueError(
            f"Missing columns: {sorted(missing_columns)}"
        )

    print(f"Total MATSim trips: {len(trips):,}")
    print()

    # --------------------------------------------------------------
    # Aggregate MATSim modes
    # --------------------------------------------------------------

    trips["main_mode_raw"] = (
        trips["main_mode"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    print("Original MATSim modes:")
    print(trips["main_mode_raw"].value_counts())
    print()

    trips["mode"] = trips[
        "main_mode_raw"
    ].map(MODE_MAPPING)

    excluded_modes = trips.loc[
        trips["mode"].isna(),
        "main_mode_raw",
    ].value_counts()

    print("Excluded modes:")
    print(excluded_modes)
    print()

    # Removes outside, truck, and any other unmapped modes
    trips = trips.loc[
        trips["mode"].notna()
    ].copy()

    print("Aggregated modes:")
    print(trips["mode"].value_counts())
    print()

    # --------------------------------------------------------------
    # Municipality assignment
    # --------------------------------------------------------------

    municipalities = load_municipalities()

    print("Assigning origin municipalities...")

    trips["origin"] = assign_municipality(
        trips,
        municipalities,
        "start_x",
        "start_y",
    )

    print("Assigning destination municipalities...")

    trips["destination"] = assign_municipality(
        trips,
        municipalities,
        "end_x",
        "end_y",
    )

    # --------------------------------------------------------------
    # Keep only trips between selected municipalities
    # --------------------------------------------------------------

    before_filter = len(trips)

    trips = trips.loc[
        trips["origin"].isin(MUNICIPALITIES)
        & trips["destination"].isin(MUNICIPALITIES)
    ].copy()

    print()
    print(
        f"Trips between selected municipalities: "
        f"{len(trips):,}"
    )

    print(
        f"Trips removed because origin or destination "
        f"is outside selected municipalities: "
        f"{before_filter - len(trips):,}"
    )

    print()

    print("Origin municipalities:")
    print(trips["origin"].value_counts())
    print()

    print("Destination municipalities:")
    print(trips["destination"].value_counts())
    print()

    # --------------------------------------------------------------
    # Create table
    # --------------------------------------------------------------

    table = create_od_mode_table(trips)

    # Round percentages for display
    percentage_columns = [
        column
        for column in table.columns
        if column[1] == "%"
    ]

    table[percentage_columns] = (
        table[percentage_columns].round(1)
    )

    print(table)
    print()

    # --------------------------------------------------------------
    # Save
    # --------------------------------------------------------------

    write_outputs(table)


if __name__ == "__main__":
    main()
import geopandas as gpd
from pathlib import Path
from utils import PROJECT_ROOT,DATA_DIR, OSM_DIR

# -------------------------------------------------------------------
# INPUTS
# -------------------------------------------------------------------

cadastral_gpkg = Path(
    r"C:\Users\20201733\Documents\Thesis\Data\Cadastral_TI_gpkg_lv95\geopackage\av_TI_2056.gpkg"
)

study_area_file = Path(
    f"{DATA_DIR}/Raw_Inputs/shapes/Locarno_QGIS_2056.shp"
)

output_gpkg = Path(
    r"C:\Users\20201733\Documents\Thesis\Data\locarno_cadastral_clipped.gpkg"
)

layers_to_clip = [
    "lcsf",  # land-cover polygons: roads, sidewalks, buildings, etc.
    "soli",      # line objects
]

target_crs = "EPSG:2056"  # Swiss LV95


# -------------------------------------------------------------------
# READ AND PREPARE STUDY AREA
# -------------------------------------------------------------------

study_area = gpd.read_file(study_area_file)

if study_area.crs is None:
    raise ValueError(
        "The study-area shapefile has no CRS. Define its CRS in QGIS first."
    )

study_area = study_area.to_crs(target_crs)
study_area = study_area.dissolve()

# Bounding box of your study area in LV95
bbox = tuple(study_area.total_bounds)

print("Study area CRS:", study_area.crs)


# -------------------------------------------------------------------
# CLIP SELECTED LAYERS
# -------------------------------------------------------------------

if output_gpkg.exists():
    output_gpkg.unlink()

for layer in layers_to_clip:
    print(f"\nProcessing layer: {layer}")

    # Fast read: only read features whose bounding boxes intersect the study-area bbox
    gdf = gpd.read_file(
        cadastral_gpkg,
        layer=layer,
        bbox=bbox,
    )

    if gdf.empty:
        print(f"No features found in bbox for layer {layer}. Skipping.")
        continue

    if gdf.crs is None:
        print(f"Layer {layer} has no CRS. Assuming {target_crs}.")
        gdf = gdf.set_crs(target_crs)
    else:
        gdf = gdf.to_crs(target_crs)

    print(f"Features read from bbox: {len(gdf)}")

    # Exact clip to your study-area polygon
    clipped = gpd.clip(gdf, study_area)

    if clipped.empty:
        print(f"No features after exact clip for layer {layer}. Skipping.")
        continue

    print(f"Features after clip: {len(clipped)}")

    clipped.to_file(
        output_gpkg,
        layer=layer,
        driver="GPKG",
    )

print(f"\nDone. Saved clipped data to:\n{output_gpkg}")
import requests
import geopandas as gpd
from pathlib import Path

from utils import PROJECT_ROOT, OSM_DIR

def Get_Shapefile(api_url: str, outputname: str, path: str = f"{OSM_DIR}/shapes", target_crs: str = None):
    """
    Fetches GeoJSON-like data from an API and converts it into a shapefile.

    Parameters
    ----------
    api_url : str
        API endpoint returning GeoJSON-like data (e.g. geo.admin.ch)
    outputname : str
        Name of the output shapefile (without extension)
    path : str, optional
        Output directory (default = "data/osm")

    Returns
    -------
    str
        Full path to the created shapefile
    """

    # Ensure output directory exists
    output_dir = Path(path)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Request data
    response = requests.get(api_url)
    response.raise_for_status()
    data = response.json()

    # Handle Swiss API structure ("feature" instead of "features")
    if "feature" in data:
        features = [data["feature"]]
    elif "features" in data:
        features = data["features"]
    else:
        raise ValueError("Invalid GeoJSON structure: no 'feature(s)' found")

    # Convert to GeoDataFrame
    gdf = gpd.GeoDataFrame.from_features(features, crs="EPSG:4326")  # Assuming API returns WGS84
    # Reproject if target CRS is specified
    if target_crs:
        gdf = gdf.to_crs(target_crs)

    # Output path
    output_file = output_dir/f"{outputname}.shp"

    # Save shapefile
    gdf.to_file(output_file)

    print(f"Shapefile saved to: {output_file}")

    return str(output_file)


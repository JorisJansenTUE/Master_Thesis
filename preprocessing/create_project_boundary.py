from pathlib import Path

import geopandas as gpd
from .utils import RAW_DIR

def create_project_boundary(
    municipalities_path: Path,
    output_path: Path,
    name_col: str = "NAME",
    layer: str | None = None,
    target_crs: str = "EPSG:2056",
) -> None:
    project_municipalities = [
        "Locarno",
        "Ascona",
        "Minusio",
        "Losone",
        "Muralto",
        "Terre di Pedemonte",
        "Tenero-Contra",
        "Orselina",
        "Brione sopra Minusio",
        "Gordola",
        "Lavertezzo",
    ]

    municipalities = gpd.read_file(municipalities_path, layer=layer)
    municipalities = municipalities.to_crs(target_crs)

    selected = municipalities[
        municipalities[name_col].isin(project_municipalities)
    ].copy()

    missing = set(project_municipalities) - set(selected[name_col])
    if missing:
        raise ValueError(f"Missing municipalities in input file: {sorted(missing)}")

    boundary = gpd.GeoDataFrame(
        {
            "name": ["locarno_project_boundary"],
            "n_muni": [len(selected)],
        },
        geometry=[selected.geometry.union_all()],
        crs=municipalities.crs,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    boundary.to_file(output_path)

    print(f"Written project boundary to: {output_path}")


if __name__ == "__main__":
    create_project_boundary(
        municipalities_path=Path(f"{RAW_DIR}/shapes/swissBOUNDARIES3D.gpkg"),
        output_path=Path(f"{RAW_DIR}/shapes/Locarno_project_boundary.shp"),
        name_col="name",
        layer="tlm_hoheitsgebiet",
        target_crs="EPSG:2056",
    )
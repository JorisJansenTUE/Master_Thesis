from pathlib import Path

# Path structure

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MATSIM_PROJECT_DIR = PROJECT_ROOT / "MATSIM"

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
OSM_DIR = RAW_DIR / "osm"
GTFS_DIR = RAW_DIR / "gtfs"
SHAPES_DIR = RAW_DIR / "shapes"
INTERIM_DIR = DATA_DIR / "interim"
SCENARIOS_DIR = DATA_DIR / "scenarios"

PT2MATSIM_DIR = PROJECT_ROOT / "configs"/ "pt2matsim"

MODEL_DIR = PROJECT_ROOT / "models"
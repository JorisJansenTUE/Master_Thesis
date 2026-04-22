from pathlib import Path

# Path structure

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OSM_DIR = DATA_DIR / "osm"
NETWORK_DIR = DATA_DIR / "network"
MODEL_DIR = PROJECT_ROOT / "models"
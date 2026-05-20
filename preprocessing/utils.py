from pathlib import Path

# Path structure

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OSM_DIR = DATA_DIR / "Raw_inputs" / "osm"
NETWORK_DIR = DATA_DIR / "MATSIM_Scenarios" / "network"
MODEL_DIR = PROJECT_ROOT / "models"
"""Single setting for the opportunity dataset used in accessibility.

Edit DESTINATION_SOURCE here, then run inspect_spatial, inspect_pt_mapping,
and inspect_accessibility. Facility counts are *work locations*, not workers.
All coordinates must use Swiss LV95 / EPSG:2056.
"""
from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]

# Select "statent" (employees per 100 m cell) or "facilities" (one work
# opportunity per MATSim facility containing the configured activity type).
DESTINATION_SOURCE = "statent"

STATENT_FILE = Path(
    r"C:\Users\20201733\Documents\Thesis\Data\Swiss_FSO\Employment-STATENT2023.zip"
)

# EDIT to your actual baseline/prepared MATSim facilities file.
# Supported input types: .xml and .xml.gz.
FACILITIES_FILE = (
    PROJECT_ROOT / "data" / "Simulated_Runs_v2"
    / "baseline_00000" / "facilities.xml.gz"
)

# Exact MATSim activity-option type names; adjust if your facility file uses
# e.g. "work_1" or another label. Only one opportunity per matching facility.
WORK_ACTIVITY_TYPES = ("work",)

OUTPUT_ROOT = PROJECT_ROOT / "accessibility" / "outputs"


def source_key() -> str:
    source = DESTINATION_SOURCE.strip().lower()
    if source not in {"statent", "facilities"}:
        raise ValueError("DESTINATION_SOURCE must be 'statent' or 'facilities'")
    return source


def opportunity_label() -> str:
    return "STATENT jobs" if source_key() == "statent" else "MATSim work facilities"


def spatial_dir() -> Path:
    return OUTPUT_ROOT / ("spatial_baseline" if source_key() == "statent" else "spatial_facilities")


def results_dir(run_name: str) -> Path:
    suffix = "" if source_key() == "statent" else "_facilities"
    return OUTPUT_ROOT / f"results_{run_name}{suffix}"


def baseline_reference_file(baseline_run_name: str = "baseline_00000") -> Path:
    return results_dir(baseline_run_name) / "sufficiency_reference.json"

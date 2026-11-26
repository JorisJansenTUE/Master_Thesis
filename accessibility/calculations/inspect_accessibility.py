"""Cumulative opportunity accessibility + baseline-referenced sufficiency.

Run from the project root:
    python -m accessibility.calculations.inspect_accessibility

Reuses existing road/PT routers. Car and bicycle snapping is dynamically
computed IN MEMORY for the actual scenario network. No connector CSVs are
written for scenarios. PT walking connectors remain the original baseline CSVs
unless stop IDs/coordinates or served stops change.

This inspection writes results/figures, not scenario-specific input files.
"""
from __future__ import annotations

from pathlib import Path
from time import perf_counter

import pandas as pd

from accessibility.inputs.grid import destination_settings as dest
from accessibility.calculations.cumulative import CumulativeAccessibility
from accessibility.calculations.evaluate import calculate_accessibility
from accessibility.calculations.sufficiency import SufficiencyReference, assess_sufficiency
from accessibility.calculations.population_metrics import weight_sha256
from accessibility.calculations.population_settings import POPULATION_WEIGHTED
from accessibility.calculations.plots import (
    create_hexagon_geopackage, plot_accessibility_maps, plot_sufficiency_bars,
)


# ------ Edit settings here (no CLI flags) ------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUN_DIR = PROJECT_ROOT / "data" / "Simulated_Runs_v2" / "corridor_00792"  
SPATIAL_DIR = dest.spatial_dir()
OUTPUT_DIR = PROJECT_ROOT / "data" / "accessibility" / "base_mapping" / "population_weight"

NETWORK_FILE = RUN_DIR / "network.xml.gz"
SCHEDULE_FILE = RUN_DIR / "transit_schedule.xml.gz"
LINKSTATS_FILE = RUN_DIR / "targets" / "linkstats.txt.gz"
BASELINE_RUN_NAME = "baseline_00000"
REFERENCE_FILE = dest.baseline_reference_file(BASELINE_RUN_NAME).with_name(
    "sufficiency_reference_20min_park5_statpop.json" if POPULATION_WEIGHTED
    else "sufficiency_reference_20min_park5.json"
)

THRESHOLDS_MIN = (20.0,)
MAP_THRESHOLD_MIN = 20.0
HOUR_START = 8
LINKSTATS_STATISTIC = "avg"
BIKE_SPEED_M_S = 4.17  # Provisional, as in existing routing inspection.
PT_TRANSFER_PENALTY_S = 300.0
ORIGIN_SNAP_M = 500.0
JOB_SNAP_M = 300.0
CAR_PARKING_PENALTY_S = 300.0
# ------ End settings ------


def main() -> None:
    t0 = perf_counter()
    if MAP_THRESHOLD_MIN not in THRESHOLDS_MIN:
        raise ValueError("MAP_THRESHOLD_MIN must be one of THRESHOLDS_MIN")
    origins = pd.read_csv(SPATIAL_DIR / "origins.csv", dtype={"hex_id": str})
    jobs = pd.read_csv(SPATIAL_DIR / "jobs.csv", dtype={"job_id": str})
    cumulative = CumulativeAccessibility(THRESHOLDS_MIN)
    results = calculate_accessibility(
        RUN_DIR, SPATIAL_DIR,
        thresholds_min=THRESHOLDS_MIN,
        hour_start=HOUR_START,
        linkstats_statistic=LINKSTATS_STATISTIC,
        bike_speed_m_s=BIKE_SPEED_M_S,
        pt_transfer_penalty_s=PT_TRANSFER_PENALTY_S,
        origin_snap_m=ORIGIN_SNAP_M,
        job_snap_m=JOB_SNAP_M,
        car_parking_penalty_s=CAR_PARKING_PENALTY_S,
        population_weighted=POPULATION_WEIGHTED,
    )
    t2 = perf_counter()

    if RUN_DIR.name == BASELINE_RUN_NAME:
        car_results = results.loc[results["mode"] == "car"]
        reference = SufficiencyReference.from_baseline_car(
            car_results, cumulative.thresholds_min,
            car_parking_penalty_s=CAR_PARKING_PENALTY_S,
            population_weighted=POPULATION_WEIGHTED,
            population_sha256=weight_sha256(SPATIAL_DIR) if POPULATION_WEIGHTED else None,
        )
        if REFERENCE_FILE.is_file() and SufficiencyReference.load(REFERENCE_FILE) != reference:
            raise ValueError(f"Existing reference changed: {REFERENCE_FILE}; use a fresh reference path")
        if not REFERENCE_FILE.is_file():
            reference.save(REFERENCE_FILE)
        print(f"Baseline car reference verified: {REFERENCE_FILE}")
    else:
        if not REFERENCE_FILE.exists():
            raise FileNotFoundError(
                f"The fixed baseline sufficiency reference is missing: {REFERENCE_FILE}. "
                "Run the baseline inspection first. Never derive a new reference "
                "from an intervention scenario."
            )
        reference = SufficiencyReference.load(REFERENCE_FILE)
        if reference.population_weighted != POPULATION_WEIGHTED:
            raise ValueError("Fixed reference has different population weighting")
        if POPULATION_WEIGHTED and reference.population_sha256 != weight_sha256(SPATIAL_DIR):
            raise ValueError("STATPOP file does not match frozen population reference")
        if reference.car_parking_penalty_s != CAR_PARKING_PENALTY_S:
            raise ValueError("Baseline reference parking penalty differs")
        print(f"Fixed baseline car reference loaded: {REFERENCE_FILE}")

    scored, summary = assess_sufficiency(results, reference, cumulative.thresholds_min)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    filename_suffix = "_statpop" if POPULATION_WEIGHTED else ""
    scored.to_csv(OUTPUT_DIR / f"accessibility_by_hexagon{filename_suffix}.csv", index=False)
    summary.to_csv(OUTPUT_DIR / f"sufficiency_summary{filename_suffix}.csv", index=False)
    plot_dir = OUTPUT_DIR / "statpop" if POPULATION_WEIGHTED else OUTPUT_DIR
    plot_sufficiency_bars(summary, plot_dir, opportunity_label=dest.opportunity_label(),
                          population_weighted=POPULATION_WEIGHTED)
    hexagons = create_hexagon_geopackage(
        SPATIAL_DIR / "origins.gpkg", scored,
        OUTPUT_DIR / f"accessibility_hexagons{filename_suffix}.gpkg",
    )
    plot_accessibility_maps(hexagons, ("car", "bike", "pt"),
                            MAP_THRESHOLD_MIN, plot_dir, opportunity_label=dest.opportunity_label())
    t3 = perf_counter()

    print("=" * 70)
    print(f"Accessibility — cumulative {dest.opportunity_label()} and sufficiency")
    print("=" * 70)
    print(f"Scenario: {RUN_DIR.name};  linkstats: {LINKSTATS_FILE}")
    print(f"Origins: {len(origins):,}; destinations: {len(jobs):,}; "
          f"total opportunities: {jobs.jobs.sum():,.0f}")
    print(f"Travel-time cutoffs: {cumulative.thresholds_min} min")
    for threshold in cumulative.thresholds_min:
        column = cumulative.column_name(threshold)
        print(f"  {threshold:g} min: baseline car mean = {reference.mean_car_jobs[column]:,.1f} opportunities; "
              f"low = {reference.threshold_jobs(column, 'low'):,.1f}; "
              f"high = {reference.threshold_jobs(column, 'high'):,.1f}")
    print("\nSufficiency summary:")
    summary_cols = (["mode", "time_min", "population_total", "pct_population_sufficient_low",
                     "pct_population_sufficient_high", "mean_shortfall_population_low",
                     "mean_shortfall_population_high"] if POPULATION_WEIGHTED else
                    ["mode", "time_min", "n_attached", "pct_sufficient_low",
                     "pct_sufficient_high", "mean_shortfall_low", "mean_shortfall_high"])
    print(summary[summary_cols].round(3).to_string(index=False))
    print(f"\nLoading+routing+aggregation: {t2-t0:.3f}s; "
          f"saving+plots: {t3-t2:.3f}s")
    print(f"Results: {OUTPUT_DIR}")
    print("Population-weighted sufficient shares include unattached residents as insufficient."
          if POPULATION_WEIGHTED else
          "Legacy reports use unweighted attached-origin shares; no residents weighted.")
    print("Accessibility-only adaptation; no PMI gate or mode-specific resident group counts.")


if __name__ == "__main__":
    main()

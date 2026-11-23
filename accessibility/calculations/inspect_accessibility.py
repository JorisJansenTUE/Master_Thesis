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
from accessibility.calculations.plots import (
    create_hexagon_geopackage, plot_accessibility_maps, plot_sufficiency_bars,
)


# ------ Edit settings here (no CLI flags) ------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUN_DIR = PROJECT_ROOT / "data" / "Simulated_Runs_v2" / "corridor_00792"  
SPATIAL_DIR = dest.spatial_dir()
OUTPUT_DIR = dest.results_dir(RUN_DIR.name)

NETWORK_FILE = RUN_DIR / "network.xml.gz"
SCHEDULE_FILE = RUN_DIR / "transit_schedule.xml.gz"
LINKSTATS_FILE = RUN_DIR / "targets" / "linkstats.txt.gz"
BASELINE_RUN_NAME = "baseline_00000"
REFERENCE_FILE = dest.baseline_reference_file(BASELINE_RUN_NAME)

THRESHOLDS_MIN = (15.0, 20.0, 30.0)
MAP_THRESHOLD_MIN = 15.0
HOUR_START = 8
LINKSTATS_STATISTIC = "avg"
BIKE_SPEED_M_S = 4.17  # Provisional, as in existing routing inspection.
PT_TRANSFER_PENALTY_S = 300.0
ORIGIN_SNAP_M = 500.0
JOB_SNAP_M = 300.0
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
    )
    t2 = perf_counter()

    if RUN_DIR.name == BASELINE_RUN_NAME:
        car_results = results.loc[results["mode"] == "car"]
        reference = SufficiencyReference.from_baseline_car(
            car_results, cumulative.thresholds_min,
        )
        reference.save(REFERENCE_FILE)
        print(f"Baseline car reference saved: {REFERENCE_FILE}")
    else:
        if not REFERENCE_FILE.exists():
            raise FileNotFoundError(
                f"The fixed baseline sufficiency reference is missing: {REFERENCE_FILE}. "
                "Run the baseline inspection first. Never derive a new reference "
                "from an intervention scenario."
            )
        reference = SufficiencyReference.load(REFERENCE_FILE)
        print(f"Fixed baseline car reference loaded: {REFERENCE_FILE}")

    scored, summary = assess_sufficiency(results, reference, cumulative.thresholds_min)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    scored.to_csv(OUTPUT_DIR / "accessibility_by_hexagon.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "sufficiency_summary.csv", index=False)
    plot_sufficiency_bars(summary, OUTPUT_DIR, opportunity_label=dest.opportunity_label())
    hexagons = create_hexagon_geopackage(
        SPATIAL_DIR / "origins.gpkg", scored,
        OUTPUT_DIR / "accessibility_hexagons.gpkg",
    )
    plot_accessibility_maps(hexagons, ("car", "bike", "pt"),
                            MAP_THRESHOLD_MIN, OUTPUT_DIR, opportunity_label=dest.opportunity_label())
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
    print("\nSufficient attached hexagons (%):")
    print(summary[["mode", "time_min", "n_attached", "pct_sufficient_low",
                   "pct_sufficient_high", "mean_shortfall_low", "mean_shortfall_high"]]
          .round(3).to_string(index=False))
    print(f"\nLoading+routing+aggregation: {t2-t0:.3f}s; "
          f"saving+plots: {t3-t2:.3f}s")
    print(f"Results: {OUTPUT_DIR}")
    print("Caution: origin means/shares are unweighted by residents; unattached "
          "hexagons excluded. PT lacks initial wait/frequency. Bike uses perceived time.")


if __name__ == "__main__":
    main()

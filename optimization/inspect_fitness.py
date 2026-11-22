"""Numerical accessibility + provisional fitness for one completed MATSim run.

Run from the thesis repository root:
    python -m optimization.inspect_fitness
    python -m optimization.inspect_fitness --scenario data/Simulated_Runs_v2/corridor_00792
    python -m optimization.inspect_fitness --scenario data/GA_Pilot/ga_... --score-only

First evaluate the baseline once to establish the fixed car reference.
No figures, scenario generation or MATSim runs occur here.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

from accessibility.inputs.grid import destination_settings as dest
from optimization.fitness import evaluate_scenario

# -------------------------- Editable settings ----------------------------
SCENARIO_DIR = Path("data/Simulated_Runs_v2/baseline_00000")
SPATIAL_DIR = dest.spatial_dir()  # Based on destination_settings.DESTINATION_SOURCE
BASELINE_NAME = "baseline_00000"
BASELINE_REFERENCE_FILE = dest.baseline_reference_file(BASELINE_NAME).with_name("sufficiency_reference_15min.json")

HOUR_START = 8
LINKSTATS_STATISTIC = "avg"
THRESHOLDS_MIN = (15.0,)
BIKE_SPEED_M_S = 4.17
PT_TRANSFER_PENALTY_S = 300.0
ORIGIN_SNAP_M = 500.0
JOB_SNAP_M = 300.0

# PROVISIONAL example objective. Change these later without rerunning MATSim.
MODE_WEIGHTS = {"car": 1.0, "bike": 1.0, "pt": 1.0}
TIME_WEIGHTS = {t: 1.0 for t in THRESHOLDS_MIN}
SUFFICIENCY_LEVEL = "high"  # "low" or "high"
# -------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", type=Path, default=SCENARIO_DIR)
    parser.add_argument("--spatial-dir", type=Path, default=SPATIAL_DIR)
    parser.add_argument("--reference", type=Path, default=BASELINE_REFERENCE_FILE)
    parser.add_argument("--baseline-name", default=BASELINE_NAME)
    parser.add_argument("--score-only", action="store_true",
                        help="Rescore existing accessibility/sufficiency CSV without routing")
    args = parser.parse_args()
    start = perf_counter()
    score = evaluate_scenario(
        args.scenario, args.spatial_dir, args.reference,
        baseline_name=args.baseline_name,
        mode_weights=MODE_WEIGHTS,
        time_weights=TIME_WEIGHTS,
        level=SUFFICIENCY_LEVEL,
        hour_start=HOUR_START,
        linkstats_statistic=LINKSTATS_STATISTIC,
        bike_speed_m_s=BIKE_SPEED_M_S,
        pt_transfer_penalty_s=PT_TRANSFER_PENALTY_S,
        origin_snap_m=ORIGIN_SNAP_M,
        job_snap_m=JOB_SNAP_M,
        score_only=args.score_only,
    )
    print(f"Scenario: {score['scenario']}")
    print(f"Provisional fitness: {score['fitness']:.6f} "
          f"(weighted shortfall: {score['weighted_shortfall']:.6f})")
    print(f"Sufficiency level: {score['sufficiency_level']}; "
          f"weights: {score['mode_weights']}")
    print(f"Output: {args.scenario / 'accessibility' / 'fitness.json'}")
    print(f"Elapsed: {perf_counter()-start:.2f}s")


if __name__ == "__main__":
    main()

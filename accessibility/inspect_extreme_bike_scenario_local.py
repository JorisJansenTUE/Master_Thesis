"""Build a synthetic cycling-everywhere network and compare bicycle accessibility.

Edit the LOCAL SETTINGS at the top of this file, then run it locally from
within the Master_Thesis repository or your configured Python environment:

    python inspect_extreme_bike_scenario_local.py

This does not run MATSim. Bike accessibility is computed using the existing
accessibility module and the generated intervention network.
"""
from __future__ import annotations

import json
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from accessibility.calculations.cumulative import CumulativeAccessibility
from accessibility.calculations.sufficiency import SufficiencyReference
from accessibility.inputs.grid.snapping import ScenarioRoadSnapper
from accessibility.inputs.links.network import BikeNetwork
from accessibility.routing.road import RoadRouter
from network_sampling.dataset.generate import (
    DEFAULT_SETTINGS,
    load_baseline,
    write_intervention_scenario_files,
)
from network_sampling.dataset.link_mapping import write_street_segments
from network_sampling.interventions.intervention_types import (
    Intervention,
    InterventionType,
    RoadSpaceVariant,
)

from preprocessing.utils import PROJECT_ROOT

# ============================================================================
# LOCAL SETTINGS — EDIT THESE PATHS ON YOUR COMPUTER
# ============================================================================
# Use raw Windows paths, e.g. Path(r"C:\Users\...\network.xml.gz"), or / paths.
BASELINE_NETWORK = Path( PROJECT_ROOT / "data" / "Simulated_Runs_v2" / "baseline_00000" / "network.xml.gz" )
BASELINE_SCHEDULE = Path( PROJECT_ROOT / "data" / "Simulated_Runs_v2" / "baseline_00000" / "transit_schedule.xml.gz" )
SPATIAL_DIR = Path( r"C:\Users\20201733\Documents\Thesis\Git_Master_Thesis\data\accessibility\base_mapping\spatial_baseline" )
REFERENCE_FILE = Path(r"C:\Users\20201733\Documents\Thesis\Git_Master_Thesis\data\accessibility\base_mapping\results_baseline_00000\sufficiency_reference.json")
OUTPUT_DIR = Path( PROJECT_ROOT / "data" / "accessibility" / "outputs" / "Extreme_Bike_Experiment" )

# False: every car segment gets PBL + keep_car when eligible; otherwise a
# bicycle street, including segments failing ordinary bicycle-street eligibility.
# True: skip the remaining ineligible bicycle-street segments.
RESPECT_BICYCLE_STREET_ELIGIBILITY = False

ACCESSIBILITY_THRESHOLD_MIN = 15.0
BIKE_SPEED_M_S = 4.17
# ============================================================================


def select_interventions(base, *, respect_eligibility: bool):
    eligible_by_segment = {}
    for iv in base.eligible:
        eligible_by_segment.setdefault(iv.segment_id, []).append(iv)

    selected = []
    counts = Counter()
    lengths = Counter()
    for segment in base.index.all():
        if not any("car" in base.network.links[lid].allowed_modes for lid in segment.link_ids):
            continue

        possibilities = eligible_by_segment.get(segment.id, ())
        pbl_keep_car = next((iv for iv in possibilities
                             if iv.intervention_type == InterventionType.PROTECTED_BIKE_LANE
                             and iv.road_space_variant == RoadSpaceVariant.KEEP_CAR), None)
        eligible_bicycle_street = next((iv for iv in possibilities
                                        if iv.intervention_type == InterventionType.BICYCLE_STREET), None)

        if pbl_keep_car is not None:
            intervention = pbl_keep_car
            category = "pbl_keep_car"
        elif eligible_bicycle_street is not None:
            intervention = eligible_bicycle_street
            category = "bicycle_street_eligible"
        elif not respect_eligibility:
            # Deliberate stress-test override: not a claim of physical feasibility.
            intervention = Intervention(segment.id, InterventionType.BICYCLE_STREET)
            category = "bicycle_street_forced_ineligible"
        else:
            counts["car_segments_skipped_ineligible"] += 1
            lengths["car_segments_skipped_ineligible"] += float(segment.length_m)
            continue

        selected.append(intervention)
        counts[category] += 1
        lengths[category] += float(segment.length_m)
    return selected, dict(counts), dict(lengths)


def bike_accessibility(network: Path, origins: pd.DataFrame,
                       jobs: pd.DataFrame, threshold_min: float,
                       bike_speed_m_s: float) -> pd.DataFrame:
    bike = BikeNetwork.from_matsim_network(network, bike_speed_m_s=bike_speed_m_s)
    attachments = ScenarioRoadSnapper(
        origins, jobs, max_origin_distance_m=500.0, max_job_distance_m=300.0,
    ).attach(bike.graph)
    router = RoadRouter(
        bike.graph, attachments.origins.connectors,
        attachments.jobs.connectors, jobs,
    )
    print(f"Bike graph {network.name}: {bike.num_links} links, "
          f"origins attached={attachments.origins.attached_points}/{len(origins)}, "
          f"jobs attached={attachments.jobs.attached_points}/{len(jobs)}", flush=True)
    return CumulativeAccessibility((threshold_min,)).run(
        router, origins["hex_id"].astype(str), mode="bike",
    )


def build_comparison(original: pd.DataFrame, modified: pd.DataFrame,
                     threshold_min: float, reference_jobs: float):
    column = CumulativeAccessibility.column_name(threshold_min)
    old = original[["hex_id", "attached", column]].rename(columns={
        "attached": "baseline_attached", column: "baseline_jobs",
    })
    new = modified[["hex_id", "attached", column]].rename(columns={
        "attached": "scenario_attached", column: "scenario_jobs",
    })
    combined = old.merge(new, on="hex_id", how="outer", validate="one_to_one")
    if combined[["baseline_attached", "scenario_attached"]].isna().any().any():
        raise ValueError("Origin sets differ between baseline and synthetic scenario")

    paired = combined["baseline_attached"] & combined["scenario_attached"]
    combined["delta_jobs"] = (combined["scenario_jobs"] - combined["baseline_jobs"]).where(paired)
    summary = {"origin_count": len(combined), "paired_attached_origins": int(paired.sum()),
               "reference_jobs_15min_from_baseline_car": reference_jobs}
    for prefix, col in (("baseline", "baseline_jobs"), ("scenario", "scenario_jobs")):
        attached = combined[f"{prefix}_attached"]
        vals = combined.loc[attached, col].astype(float)
        # The provisional_linear bike component: unattached -> 0,
        # attached -> min(jobs / baseline-car-high-threshold, 1).
        linear_score = float(np.minimum(vals.to_numpy() / reference_jobs, 1.0).sum() / len(combined))
        summary[prefix] = {
            "attached_origins": int(attached.sum()),
            "mean_jobs_attached": float(vals.mean()) if len(vals) else None,
            "linear_bike_score": linear_score,
            "sufficient_origins_pct_among_attached": float(100.0 * (vals >= reference_jobs).mean())
                if len(vals) else None,
        }
    delta = combined.loc[paired, "delta_jobs"].to_numpy(dtype=float)
    summary["change"] = {
        "linear_bike_score_absolute": summary["scenario"]["linear_bike_score"] - summary["baseline"]["linear_bike_score"],
        "linear_bike_score_relative_pct": (
            100 * (summary["scenario"]["linear_bike_score"] / summary["baseline"]["linear_bike_score"] - 1)
            if summary["baseline"]["linear_bike_score"] > 0 else None
        ),
        "mean_change_jobs_paired": float(delta.mean()) if len(delta) else None,
        "median_change_jobs_paired": float(np.median(delta)) if len(delta) else None,
        "origins_improved": int((delta > 0).sum()),
        "origins_worsened": int((delta < 0).sum()),
        "origins_unchanged": int((delta == 0).sum()),
    }
    return combined, summary


def main() -> None:
    network = BASELINE_NETWORK.resolve(strict=True)
    schedule = BASELINE_SCHEDULE.resolve(strict=True)
    spatial = SPATIAL_DIR.resolve(strict=True)
    reference = SufficiencyReference.load(REFERENCE_FILE.resolve(strict=True))
    threshold_min = ACCESSIBILITY_THRESHOLD_MIN
    bike_speed_m_s = BIKE_SPEED_M_S
    respect_eligibility = RESPECT_BICYCLE_STREET_ELIGIBILITY
    out = OUTPUT_DIR.resolve()

    threshold_col = CumulativeAccessibility.column_name(threshold_min)
    reference_jobs = reference.threshold_jobs(threshold_col, "high")
    if reference_jobs <= 0:
        raise ValueError("Baseline car accessibility reference must be positive")

    out.mkdir(parents=True, exist_ok=True)
    scenario_name = ("bike_eligible_00000" if respect_eligibility
                     else "bike_everywhere_00000")
    scenario_dir = out / scenario_name

    base = load_baseline(network, schedule, DEFAULT_SETTINGS)
    selected, counts, lengths = select_interventions(
        base, respect_eligibility=respect_eligibility,
    )
    if not selected:
        raise ValueError("No interventions selected")
    used_m = sum(float(base.index.get(iv.segment_id).length_m) for iv in selected)
    print(f"Selected {len(selected)} interventions, total {used_m/1000:.2f} km", flush=True)
    for category, count in sorted(counts.items()):
        print(f"  {category}: {count} segments, {lengths[category]/1000:.2f} km", flush=True)

    if scenario_dir.exists():
        raise FileExistsError(f"Refusing to overwrite an existing scenario: {scenario_dir}. "
                              "Change OUTPUT_DIR or explicitly remove this diagnostic scenario.")
    segments_path = out / "street_segments.csv.gz"
    if not segments_path.is_file():
        write_street_segments(segments_path, base.index)

    with tempfile.TemporaryDirectory(prefix=f".{scenario_name}_", dir=out) as tmp:
        stage = Path(tmp)
        records, verified_used, n_mapped, connectivity = write_intervention_scenario_files(
            stage, base=base, schedule=schedule, interventions=selected,
            budget_m=used_m + 1.0,
            matsim_jar=None,  # Quick bike-only experiment; run_one can check Java later.
        )
        (stage / "scenario.json").write_text(json.dumps({
            "schema_version": 1,
            "id": scenario_name,
            "strategy": "synthetic_network_wide_cycling",
            "baseline_network": str(network),
            "baseline_schedule": str(schedule),
            "forced_bicycle_streets": not respect_eligibility,
            "interventions_by_category": counts,
            "intervention_lengths_m": lengths,
            "budget_used_m": verified_used,
            "intervention_count": len(records),
            "mapped_road_links": n_mapped,
            "connectivity": connectivity,
        }, indent=2) + "\n", encoding="utf-8")
        stage.rename(scenario_dir)

    origins = pd.read_csv(spatial / "origins.csv", dtype={"hex_id": str})
    jobs = pd.read_csv(spatial / "jobs.csv", dtype={"job_id": str})
    print("Calculating baseline bicycle accessibility...", flush=True)
    baseline = bike_accessibility(network, origins, jobs, threshold_min, bike_speed_m_s)
    print("Calculating synthetic bicycle accessibility...", flush=True)
    synthetic = bike_accessibility(
        scenario_dir / "network.xml.gz", origins, jobs, threshold_min, bike_speed_m_s,
    )
    combined, summary = build_comparison(baseline, synthetic, threshold_min, reference_jobs)
    summary["scenario_name"] = scenario_name
    summary["intervention_counts"] = counts
    summary["intervention_length_m_by_type"] = lengths
    summary["intervention_length_m"] = used_m
    summary["note"] = (
        "Bicycle-only model estimate; exact under current bicycle accessibility routing "
        "inputs, not a MATSim outcome. Car/PT changes and feasibility of forced "
        "bicycle streets require separate checks."
    )
    combined.to_csv(out / f"{scenario_name}_bicycle_accessibility.csv.gz", index=False, compression="gzip")
    (out / f"{scenario_name}_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )

    print("\n15-minute bicycle accessibility diagnostic:", flush=True)
    for label in ("baseline", "scenario"):
        d = summary[label]
        print(f"  {label:8s}  score={d['linear_bike_score']:.6f} "
              f"mean_jobs_attached={d['mean_jobs_attached']:.1f} "
              f"attached={d['attached_origins']}/{summary['origin_count']}", flush=True)
    d = summary["change"]
    print(f"  bike score improvement: {d['linear_bike_score_absolute']:+.6f} "
          f"({d['linear_bike_score_relative_pct']:+.2f}%)", flush=True)
    print(f"  paired hexagons improved={d['origins_improved']}, "
          f"unchanged={d['origins_unchanged']}, worsened={d['origins_worsened']}", flush=True)
    print(f"  scenario files: {scenario_dir}", flush=True)
    print(f"  results: {out / (scenario_name + '_summary.json')}", flush=True)


if __name__ == "__main__":
    main()

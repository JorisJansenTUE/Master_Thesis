"""Baseline-referenced 20-minute accessibility and constrained sufficiency fitness.

The 5-minute car parking/egress allowance is charged once per OD journey,
leaving 15 minutes of in-network car travel inside a 20-minute budget.
The baseline reference is the mean car accessibility of car-attached origins.

Only origins attached to at least one mode in the BASELINE enter the fitness
and car-feasibility denominators. This fixed eligibility set avoids counting
isolated hexagons with no network attachment in any mode; it is a network-based
proxy, not proof of residential population. Within that set, loss of a mode's
attachment scores as zero accessibility and a gained attachment can improve the
score. Population weighting is intentionally deferred.

The GA maximizes 1 - mean(squared bike shortfall, squared PT shortfall) among
solutions preserving car-sufficient origin share within 5 percentage points of
baseline. Infeasible solutions receive a negative scalar ranking score so they
cannot outrank feasible solutions. Raw sustainable-mode fitness is kept
separately. --score-only never reuses old linear/unparked 15-minute CSVs.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from accessibility.calculations.cumulative import CumulativeAccessibility
from accessibility.calculations.evaluate import calculate_accessibility
from accessibility.calculations.sufficiency import SufficiencyReference, assess_sufficiency

FITNESS_VERSION = "squared_sustainable_car_constraint_v1"
MODES = ("car", "bike", "pt")
ATTACHMENT_POLICY = "fixed_baseline_any_mode_attached"
SHORTFALL_FORMULA = "squared_normalized_v2"


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _write_csv(path: Path, dataframe: pd.DataFrame, *, compression: str | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    dataframe.to_csv(tmp, index=False, compression=compression)
    os.replace(tmp, path)


def _load_hexagons(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path, dtype={"hex_id": str, "mode": str})
    required = {"hex_id", "mode", "attached"}
    if missing := required - set(data.columns):
        raise ValueError(f"Missing per-hexagon columns in {path}: {sorted(missing)}")
    if data[["hex_id", "mode", "attached"]].isna().any().any():
        raise ValueError(f"Missing hexagon ID/mode/attachment in {path}")
    if data.duplicated(["mode", "hex_id"]).any():
        raise ValueError(f"Duplicate mode/hexagon pair in {path}")
    if not pd.api.types.is_bool_dtype(data["attached"]):
        values = data["attached"].astype(str)
        if not values.isin(["True", "False"]).all():
            raise ValueError(f"Invalid attachment flags in {path}")
        data["attached"] = values.map({"True": True, "False": False})
    return data


def _baseline_mask_path(reference_file: Path) -> Path:
    return reference_file.with_name(reference_file.stem + "_attachments.json")


def _baseline_mask_payload(
    data: pd.DataFrame, reference_file: Path, baseline_name: str,
    reference: SufficiencyReference, thresholds: tuple[float, ...],
) -> dict:
    origins: dict[str, list[str]] = {}
    attached: dict[str, list[str]] = {}
    for mode in MODES:
        subset = data.loc[data["mode"] == mode]
        if subset.empty:
            raise ValueError(f"Missing baseline hexagons for mode {mode}")
        origins[mode] = sorted(subset["hex_id"].tolist())
        attached[mode] = sorted(subset.loc[subset["attached"], "hex_id"].tolist())
    if len({tuple(ids) for ids in origins.values()}) != 1:
        raise ValueError("Different baseline origin IDs between transport modes")

    eligible = sorted(set().union(*(set(ids) for ids in attached.values())))
    if not eligible:
        raise ValueError("No origin attaches to any transport mode in the baseline")
    car_rows = data.loc[data["mode"] == "car"].set_index("hex_id")
    baseline_share: dict[str, dict[str, float]] = {}
    for t in thresholds:
        col = CumulativeAccessibility.column_name(t)
        if col not in car_rows:
            raise ValueError(f"Missing baseline car accessibility column {col}")
        jobs = pd.to_numeric(car_rows.loc[eligible, col], errors="raise")
        # Car-unattached eligible origins are insufficient; they are NOT
        # indistinguishable from fully isolated baseline hexagons.
        jobs = jobs.where(car_rows.loc[eligible, "attached"], 0.0).fillna(0.0)
        if not np.isfinite(jobs.to_numpy()).all() or (jobs < 0).any():
            raise ValueError(f"Invalid baseline car accessibility in {col}")
        baseline_share[col] = {
            level: float((jobs >= reference.threshold_jobs(col, level)).mean())
            for level in ("low", "high")
        }
    return {
        "version": 2,
        "baseline_name": baseline_name,
        "baseline_reference_sha256": hashlib.sha256(reference_file.read_bytes()).hexdigest(),
        "all_origins_by_mode": origins,
        "attached_origins_by_mode": attached,
        "eligible_origins": eligible,
        "baseline_car_sufficient_share": baseline_share,
        "eligibility_policy": ATTACHMENT_POLICY,
    }


def _load_or_create_baseline_mask(
    path: Path, data: pd.DataFrame, reference_file: Path, baseline_name: str,
    reference: SufficiencyReference, thresholds: tuple[float, ...],
    *, is_baseline: bool,
) -> dict:
    if is_baseline:
        expected = _baseline_mask_payload(data, reference_file, baseline_name,
                                          reference, thresholds)
        if path.is_file():
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload != expected:
                raise ValueError(f"Existing baseline origin mask differs: {path}. "
                                 "Check that the baseline and spatial inputs have not changed.")
        else:
            _write_json(path, expected)
        return expected

    if not path.is_file():
        raise FileNotFoundError(f"Fixed baseline origin mask missing: {path}. "
                                f"Evaluate baseline {baseline_name} first.")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (payload.get("version") != 2 or payload.get("baseline_name") != baseline_name
            or payload.get("eligibility_policy") != ATTACHMENT_POLICY
            or payload.get("baseline_reference_sha256") !=
            hashlib.sha256(reference_file.read_bytes()).hexdigest()):
        raise ValueError("Baseline origin mask does not match the fixed baseline reference")
    if set(payload.get("all_origins_by_mode", {})) != set(MODES) or \
            set(payload.get("attached_origins_by_mode", {})) != set(MODES):
        raise ValueError("Baseline origin mask is missing transport modes")
    eligible = set(payload["eligible_origins"])
    attached = set().union(*(set(x) for x in payload["attached_origins_by_mode"].values()))
    if not eligible or eligible != attached:
        raise ValueError("Baseline eligible origins do not match union of attached origins")
    if set(payload.get("baseline_car_sufficient_share", {})) != {
        CumulativeAccessibility.column_name(t) for t in thresholds
    }:
        raise ValueError("Baseline car feasibility reference differs from selected cutoffs")
    return payload


def score_sufficiency(
    summary: pd.DataFrame,
    hexagons: pd.DataFrame,
    baseline_mask: dict,
    reference: SufficiencyReference,
    *,
    mode_weights: Mapping[str, float],
    time_weights: Mapping[float, float],
    level: str = "high",
    car_sufficiency_tolerance_pp: float = 5.0,
) -> dict:
    """Evaluate the squared sustainable objective and explicit car constraint.

    Only fixed baseline-any-mode-attached origins are evaluated. All modes use
    the same fixed denominator. Unattached mode-specific origins in that set
    have zero accessible opportunities; origins with no baseline attachment in
    any mode are excluded entirely, even if later newly attached.

    Feasible solutions have ``fitness = raw_fitness`` in [0,1]; infeasible
    solutions have ``fitness = -constraint_violation_fraction`` (<0). This
    preserves the existing scalar GA's rank ordering without treating the
    car outcome as a benefit within the bike/PT objective.
    """
    if level not in ("low", "high"):
        raise ValueError("level must be low or high")
    if set(mode_weights) != set(MODES) or not time_weights:
        raise ValueError(f"Require weights for exactly {MODES} and at least one time cutoff")
    if mode_weights["car"] != 0 or mode_weights["bike"] <= 0 or mode_weights["pt"] <= 0:
        raise ValueError("Car weight must be zero and bike/PT weights positive")
    if any(not math.isfinite(float(v)) or float(v) < 0 for v in mode_weights.values()):
        raise ValueError("Mode weights must be finite and nonnegative")
    if any(not math.isfinite(float(k)) or float(k) <= 0 or
           not math.isfinite(float(v)) or float(v) < 0 for k, v in time_weights.items()):
        raise ValueError("Time cutoffs must be positive; weights finite and nonnegative")
    if not sum(time_weights.values()) > 0:
        raise ValueError("At least one positive time weight required")
    if (not math.isfinite(car_sufficiency_tolerance_pp)
            or not 0 <= car_sufficiency_tolerance_pp <= 100):
        raise ValueError("Car constraint tolerance must be within 0–100 percentage points")
    required = {"mode", "time_min", "n_origins", "n_attached", "baseline_car_mean_jobs",
                "shortfall_formula", "car_parking_penalty_s"}
    if missing := required - set(summary.columns):
        raise ValueError(f"Incompatible cached sufficiency summary, missing {sorted(missing)}; "
                         "recalculate accessibility without --score-only")
    if summary.duplicated(["mode", "time_min"]).any():
        raise ValueError("Duplicate mode/time rows in sufficiency summary")
    if (summary["shortfall_formula"].isna().any()
            or set(summary["shortfall_formula"].astype(str)) != {SHORTFALL_FORMULA}):
        raise ValueError("Cached sufficiency summary is not squared; rerun accessibility")
    if (not np.isfinite(summary["car_parking_penalty_s"].to_numpy(dtype=float)).all()
            or not np.allclose(summary["car_parking_penalty_s"].to_numpy(dtype=float),
                               reference.car_parking_penalty_s, atol=1e-9)):
        raise ValueError("Cached summary parking allowance differs from baseline reference")
    expected = {(mode, float(t)) for mode in MODES for t in time_weights}
    actual = set(zip(summary["mode"].astype(str), summary["time_min"].astype(float)))
    if actual != expected:
        raise ValueError(f"Mode/time rows differ: missing={sorted(expected-actual)}, "
                         f"extra={sorted(actual-expected)}; rerun accessibility")
    if set(hexagons["mode"]) != set(MODES):
        raise ValueError("Per-hexagon file must contain car, bike and PT")

    eligible = set(baseline_mask["eligible_origins"])
    if not eligible:
        raise ValueError("No eligible origins")
    # We set index after filtering for deterministic indexing and validation.
    origins = sorted(eligible)
    components: list[dict] = []
    weighted_shortfall = 0.0
    total_weight = 0.0
    car_sufficient_share = None
    car_baseline_share = None

    for row in summary.itertuples(index=False):
        mode, minutes = str(row.mode), float(row.time_min)
        subset = hexagons.loc[hexagons["mode"] == mode].set_index("hex_id")
        baseline_all = set(baseline_mask["all_origins_by_mode"][mode])
        baseline_attached = set(baseline_mask["attached_origins_by_mode"][mode])
        if set(subset.index) != baseline_all or not eligible <= baseline_all:
            raise ValueError(f"Origin hexagons differ from baseline for {mode}")
        attached_now = set(subset.index[subset["attached"]])
        if int(row.n_origins) != len(subset) or int(row.n_attached) != len(attached_now):
            raise ValueError(f"Saved summary attachment counts differ for {mode}")
        col = CumulativeAccessibility.column_name(minutes)
        threshold = reference.threshold_jobs(col, level)
        if not math.isclose(float(row.baseline_car_mean_jobs),
                            reference.mean_car_jobs[col], rel_tol=1e-9):
            raise ValueError(f"Saved baseline reference differs for {mode}, {minutes}")
        if col not in subset:
            raise ValueError(f"Missing {col} in per-hexagon accessibility")
        jobs = pd.to_numeric(subset[col], errors="raise")
        if (jobs.loc[sorted(attached_now)].isna().any()
                or not np.isfinite(jobs.loc[sorted(attached_now)].to_numpy()).all()
                or (jobs.loc[sorted(attached_now)] < 0).any()):
            raise ValueError(f"Invalid attached jobs in {mode}/{minutes}")
        if jobs.loc[sorted(baseline_all - attached_now)].notna().any():
            raise ValueError(f"Unattached origins have nonmissing jobs in {mode}")
        selected = subset.loc[origins, "attached"].astype(bool)
        eligible_jobs = jobs.reindex(origins).where(selected, 0.0).fillna(0.0)
        shortfall = (1.0 - (eligible_jobs / threshold).clip(0.0, 1.0)) ** 2
        mean_shortfall = float(shortfall.mean())
        mode_score = 1.0 - mean_shortfall
        n_new = len((attached_now - baseline_attached) & eligible)
        n_lost = len((baseline_attached - attached_now) & eligible)
        sufficient_share = float((eligible_jobs >= threshold).mean())
        weight = float(mode_weights[mode]) * float(time_weights[minutes])
        weighted_shortfall += weight * mean_shortfall
        total_weight += weight
        components.append({
            "mode": mode, "time_min": minutes,
            "n_origins": len(subset), "n_eligible_origins": len(origins),
            "n_baseline_attached": len(baseline_attached),
            "n_attached": len(attached_now),
            "n_eligible_attached": int(selected.sum()),
            "n_newly_attached": n_new, "n_lost_attachment": n_lost,
            "mode_score": mode_score,
            "mean_squared_shortfall": mean_shortfall,
            "sufficient_share": sufficient_share,
            "weight": weight,
        })
        if mode == "car":
            if len(time_weights) != 1:
                raise ValueError("Current car feasibility constraint requires one time cutoff")
            car_sufficient_share = sufficient_share
            car_baseline_share = float(baseline_mask["baseline_car_sufficient_share"][col][level])

    if not total_weight or car_sufficient_share is None or car_baseline_share is None:
        raise ValueError("Missing sustainable-mode fitness or car feasibility reference")
    normalized_shortfall = weighted_shortfall / total_weight
    raw_fitness = 1.0 - normalized_shortfall
    minimum_share = max(0.0, car_baseline_share - car_sufficiency_tolerance_pp / 100.0)
    violation_fraction = max(0.0, minimum_share - car_sufficient_share)
    feasible = violation_fraction <= 1e-12
    fitness = raw_fitness if feasible else -violation_fraction
    return {
        "fitness_version": FITNESS_VERSION,
        "fitness": fitness,
        "raw_fitness": raw_fitness,
        "feasible": feasible,
        "weighted_shortfall": normalized_shortfall,
        "sufficiency_level": level,
        "mode_weights": {mode: float(mode_weights[mode]) for mode in MODES},
        "time_weights": {f"{t:g}": float(time_weights[t]) for t in sorted(time_weights)},
        "unattached_policy": ATTACHMENT_POLICY,
        "n_eligible_origins": len(origins),
        "n_excluded_all_mode_unattached": len(baseline_mask["all_origins_by_mode"]["car"]) - len(origins),
        "car_constraint": {
            "scenario_share": car_sufficient_share,
            "baseline_share": car_baseline_share,
            "minimum_share": minimum_share,
            "tolerance_pp": float(car_sufficiency_tolerance_pp),
            "violation_pp": 100.0 * violation_fraction,
            "feasible": feasible,
        },
        "components": components,
        "provisional": False,
    }


def evaluate_scenario(
    scenario_dir: str | Path,
    spatial_dir: str | Path,
    baseline_reference_file: str | Path,
    *,
    baseline_name: str = "baseline_00000",
    mode_weights: Mapping[str, float],
    time_weights: Mapping[float, float],
    level: str = "high",
    hour_start: int = 8,
    linkstats_statistic: str = "avg",
    bike_speed_m_s: float = 4.17,
    pt_transfer_penalty_s: float = 300.0,
    origin_snap_m: float = 500.0,
    job_snap_m: float = 300.0,
    car_parking_penalty_s: float = 300.0,
    car_sufficiency_tolerance_pp: float = 5.0,
    score_only: bool = False,
) -> dict:
    """Calculate and save numerical accessibility and constrained fitness."""
    scenario_dir = Path(scenario_dir)
    output_dir = scenario_dir / "accessibility"
    summary_file = output_dir / "sufficiency_summary.csv"
    hexagon_file = output_dir / "accessibility_by_hexagon.csv.gz"
    reference_file = Path(baseline_reference_file)
    thresholds = tuple(sorted(float(t) for t in time_weights))
    if len(thresholds) != 1 or thresholds != (20.0,):
        raise ValueError("This fitness version requires a single 20-minute time cutoff")
    if not math.isclose(car_parking_penalty_s, 300.0):
        raise ValueError("This fitness version requires a 5-minute car parking penalty")
    if not math.isclose(car_sufficiency_tolerance_pp, 5.0):
        raise ValueError("This fitness version requires a 5-percentage-point car constraint")

    if score_only:
        summary = pd.read_csv(summary_file)
    else:
        linkstats = scenario_dir / "targets" / "linkstats.txt.gz"
        if not linkstats.is_file():
            raise FileNotFoundError(f"Compacted MATSim linkstats missing: {linkstats}. Run MATSim first.")
        status_file = scenario_dir / "run_status.json"
        if status_file.is_file():
            status = json.loads(status_file.read_text(encoding="utf-8"))
            if status.get("state", status.get("status")) not in ("completed", "complete"):
                raise RuntimeError(f"Simulation is not completed: {status_file}")
        result = calculate_accessibility(
            scenario_dir, spatial_dir, thresholds_min=thresholds,
            hour_start=hour_start, linkstats_statistic=linkstats_statistic,
            bike_speed_m_s=bike_speed_m_s,
            pt_transfer_penalty_s=pt_transfer_penalty_s,
            origin_snap_m=origin_snap_m, job_snap_m=job_snap_m,
            car_parking_penalty_s=car_parking_penalty_s,
        )
        if scenario_dir.name == baseline_name:
            car_results = result.loc[result["mode"] == "car"]
            candidate_reference = SufficiencyReference.from_baseline_car(
                car_results, thresholds, car_parking_penalty_s=car_parking_penalty_s,
            )
            if reference_file.is_file():
                reference = SufficiencyReference.load(reference_file)
                if (reference.car_parking_penalty_s != car_parking_penalty_s
                        or set(reference.mean_car_jobs) != set(candidate_reference.mean_car_jobs)
                        or any(not math.isclose(reference.mean_car_jobs[k], v, rel_tol=1e-9)
                               for k, v in candidate_reference.mean_car_jobs.items())):
                    raise ValueError(f"Existing baseline reference differs: {reference_file}")
            else:
                reference = candidate_reference
                reference.save(reference_file)
        else:
            if not reference_file.is_file():
                raise FileNotFoundError(f"Fixed baseline car reference missing: {reference_file}. "
                                        f"Evaluate {baseline_name} first.")
            reference = SufficiencyReference.load(reference_file)
        if reference.car_parking_penalty_s != car_parking_penalty_s:
            raise ValueError("Baseline reference car parking allowance differs from current run")
        if set(reference.mean_car_jobs) != {
            CumulativeAccessibility.column_name(t) for t in thresholds
        }:
            raise ValueError("Baseline reference time threshold differs from fitness cutoff")
        scored, summary = assess_sufficiency(result, reference, thresholds)
        _write_csv(hexagon_file, scored, compression="gzip")
        _write_csv(summary_file, summary)

    if not reference_file.is_file():
        raise FileNotFoundError(f"Baseline car reference missing: {reference_file}")
    reference = SufficiencyReference.load(reference_file)
    if reference.car_parking_penalty_s != car_parking_penalty_s:
        raise ValueError("Baseline reference parking allowance differs from current run")
    hexagons = _load_hexagons(hexagon_file)
    mask_file = _baseline_mask_path(reference_file)
    mask = _load_or_create_baseline_mask(
        mask_file, hexagons, reference_file, baseline_name, reference, thresholds,
        is_baseline=(scenario_dir.name == baseline_name),
    )
    score = score_sufficiency(
        summary, hexagons, mask, reference, mode_weights=mode_weights,
        time_weights=time_weights, level=level,
        car_sufficiency_tolerance_pp=car_sufficiency_tolerance_pp,
    )
    score.update({
        "scenario": scenario_dir.name,
        "summary_file": str(summary_file),
        "hexagon_file": str(hexagon_file),
        "baseline_attachment_file": str(mask_file),
        "baseline_attachment_sha256": hashlib.sha256(mask_file.read_bytes()).hexdigest(),
        "reference_file": str(reference_file),
        "reference_sha256": hashlib.sha256(reference_file.read_bytes()).hexdigest(),
        "summary_sha256": hashlib.sha256(summary_file.read_bytes()).hexdigest(),
        "hexagon_sha256": hashlib.sha256(hexagon_file.read_bytes()).hexdigest(),
        "opportunity_spatial_dir": str(Path(spatial_dir)),
        "accessibility_parameters": {
            "hour_start": hour_start,
            "linkstats_statistic": linkstats_statistic,
            "bike_speed_m_s": bike_speed_m_s,
            "pt_transfer_penalty_s": pt_transfer_penalty_s,
            "origin_snap_m": origin_snap_m,
            "job_snap_m": job_snap_m,
            "car_parking_penalty_s": car_parking_penalty_s,
            "car_sufficiency_tolerance_pp": car_sufficiency_tolerance_pp,
        },
    })
    _write_json(output_dir / "fitness.json", score)
    return score

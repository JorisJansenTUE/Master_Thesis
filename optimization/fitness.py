"""Numerical MATSim accessibility and provisional, baseline-aware sufficiency fitness.

The baseline defines the fixed car accessibility threshold and, independently,
which origin hexagons were initially attached for each mode. Scoring excludes
permanently unattached baseline origins, penalises lost attachments, and rewards
newly attached origins using their achieved (15-minute) job accessibility.

Neither MATSim nor accessibility routing is required when --score-only is used.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Mapping

import pandas as pd

from accessibility.calculations.cumulative import CumulativeAccessibility
from accessibility.calculations.evaluate import calculate_accessibility
from accessibility.calculations.sufficiency import SufficiencyReference, assess_sufficiency

FITNESS_VERSION = "provisional_linear"
MODES = ("car", "bike", "pt")
ATTACHMENT_POLICY = "baseline_attached_plus_newly_attached"


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


def _baseline_mask_payload(data: pd.DataFrame, reference_file: Path, baseline_name: str) -> dict:
    origins: dict[str, list[str]] = {}
    attached: dict[str, list[str]] = {}
    for mode in MODES:
        subset = data.loc[data["mode"] == mode]
        if subset.empty:
            raise ValueError(f"Missing baseline hexagons for mode {mode}")
        origins[mode] = sorted(subset["hex_id"].tolist())
        attached[mode] = sorted(subset.loc[subset["attached"], "hex_id"].tolist())
        if not attached[mode]:
            raise ValueError(f"No baseline-attached origins for mode {mode}")
    return {
        "version": 1,
        "baseline_name": baseline_name,
        "baseline_reference_sha256": hashlib.sha256(reference_file.read_bytes()).hexdigest(),
        "all_origins_by_mode": origins,
        "attached_origins_by_mode": attached,
    }


def _load_or_create_baseline_mask(
    path: Path, data: pd.DataFrame, reference_file: Path, baseline_name: str,
    *, is_baseline: bool,
) -> dict:
    if is_baseline:
        expected = _baseline_mask_payload(data, reference_file, baseline_name)
        if path.is_file():
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload != expected:
                raise ValueError(
                    f"Existing baseline attachment mask differs: {path}. "
                    "Check that the baseline and spatial inputs have not changed."
                )
        else:
            _write_json(path, expected)
        return expected

    if not path.is_file():
        raise FileNotFoundError(
            f"Fixed baseline attachment mask missing: {path}. "
            f"Run inspect_fitness for {baseline_name} with --score-only first."
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (payload.get("version") != 1 or payload.get("baseline_name") != baseline_name
            or payload.get("baseline_reference_sha256") !=
            hashlib.sha256(reference_file.read_bytes()).hexdigest()):
        raise ValueError("Baseline attachment mask does not match the fixed baseline reference")
    if set(payload.get("attached_origins_by_mode", {})) != set(MODES) or \
            set(payload.get("all_origins_by_mode", {})) != set(MODES):
        raise ValueError("Baseline attachment mask is missing transport modes")
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
) -> dict:
    """Score achieved accessibility over a fixed baseline-attached denominator.

    For each mode/time, every scenario-attached origin receives credit
    min(1, reachable_jobs / threshold); disconnected origins receive zero.
    Divide the sum by the number attached *in the baseline* for that mode.

    Consequently new attachments earn credit, lost attachments earn zero, and
    origins unattached in both scenarios contribute nothing. Because new
    attachments are additional rewards, the total fitness can exceed 1.
    """
    if level not in {"low", "high"}:
        raise ValueError("level must be 'low' or 'high'")
    if set(mode_weights) != set(MODES) or not time_weights:
        raise ValueError(f"Require weights for exactly {MODES} and at least one time cutoff")
    if any(not math.isfinite(float(v)) or float(v) < 0 for v in mode_weights.values()):
        raise ValueError("Mode weights must be nonnegative and finite")
    if any(not math.isfinite(float(k)) or float(k) <= 0 or
           not math.isfinite(float(v)) or float(v) < 0
           for k, v in time_weights.items()):
        raise ValueError("Time cutoffs must be positive; weights nonnegative and finite")
    if sum(mode_weights.values()) <= 0 or sum(time_weights.values()) <= 0:
        raise ValueError("At least one positive mode weight and one positive time weight required")
    required = {"mode", "time_min", "n_origins", "n_attached", "baseline_car_mean_jobs"}
    if missing := required - set(summary.columns):
        raise ValueError(f"Missing sufficiency summary columns: {sorted(missing)}")
    if summary.duplicated(["mode", "time_min"]).any():
        raise ValueError("Duplicate mode/time rows in sufficiency summary")
    if "shortfall_formula" in summary:
        # Newer sufficiency.py writes this marker; older linear outputs have no marker.
        if set(summary["shortfall_formula"].dropna()) != {"linear_normalized_v1"} or \
                summary["shortfall_formula"].isna().any():
            raise ValueError("Saved sufficiency summary is not linear; recalculate accessibility")

    expected = {(mode, float(time)) for mode in MODES for time in time_weights}
    actual = set(zip(summary["mode"].astype(str), summary["time_min"].astype(float)))
    if actual != expected:
        raise ValueError(f"Mode/time rows differ: missing={sorted(expected-actual)}, "
                         f"extra={sorted(actual-expected)}")
    if set(hexagons["mode"]) != set(MODES):
        raise ValueError("Per-hexagon file must contain car, bike, and PT")

    components = []
    weighted_score = 0.0
    total_weight = 0.0
    for row in summary.itertuples(index=False):
        mode, minutes = str(row.mode), float(row.time_min)
        subset = hexagons.loc[hexagons["mode"] == mode].set_index("hex_id")
        all_baseline = set(baseline_mask["all_origins_by_mode"][mode])
        attached_baseline = set(baseline_mask["attached_origins_by_mode"][mode])
        if set(subset.index) != all_baseline:
            raise ValueError(f"Origin hexagons differ from baseline for {mode}")
        if not attached_baseline or not attached_baseline <= all_baseline:
            raise ValueError(f"Invalid baseline attachment mask for {mode}")
        attached_now = set(subset.index[subset["attached"]])
        if int(row.n_origins) != len(subset) or int(row.n_attached) != len(attached_now):
            raise ValueError(f"Saved summary attachment counts differ from per-hexagon file ({mode})")

        col = CumulativeAccessibility.column_name(minutes)
        threshold = reference.threshold_jobs(col, level)
        if not math.isclose(float(row.baseline_car_mean_jobs), reference.mean_car_jobs[col], rel_tol=1e-9):
            raise ValueError(f"Saved summary car reference differs for {mode}, {minutes} min")
        if col not in subset.columns:
            raise ValueError(f"Missing {col} in per-hexagon accessibility")
        jobs = pd.to_numeric(subset[col], errors="raise")
        if jobs.loc[list(attached_now)].isna().any() or \
                (jobs.loc[list(attached_now)] < 0).any() or \
                (~jobs.loc[list(attached_now)].map(math.isfinite)).any():
            raise ValueError(f"Invalid attached accessibility values for {mode}, {minutes} min")
        if jobs.loc[list(all_baseline - attached_now)].notna().any():
            raise ValueError(f"Unattached origins must have missing job counts for {mode}")

        credit = (jobs.loc[list(attached_now)] / threshold).clip(0.0, 1.0)
        # The attached-only shortfall in saved CSV should be linear, not squared.
        column_shortfall = f"mean_shortfall_{level}"
        if column_shortfall in summary.columns:
            stated = getattr(row, column_shortfall)
            actual_mean = float((1.0 - credit).mean()) if len(credit) else float("nan")
            if len(credit) and (not math.isfinite(float(stated)) or
                                not math.isclose(float(stated), actual_mean, abs_tol=1e-6)):
                raise ValueError(f"Saved {mode} shortfall is not linear/consistent. "
                                 "Recalculate accessibility without --score-only")
        n_baseline = len(attached_baseline)
        mode_score = float(credit.sum()) / n_baseline
        n_new = len(attached_now - attached_baseline)
        n_lost = len(attached_baseline - attached_now)
        new_credit = float(credit.loc[list(attached_now - attached_baseline)].sum()) / n_baseline

        weight = float(mode_weights[mode]) * float(time_weights[minutes])
        weighted_score += weight * mode_score
        total_weight += weight
        components.append({
            "mode": mode,
            "time_min": minutes,
            "n_origins": len(subset),
            "n_baseline_attached": n_baseline,
            "n_attached": len(attached_now),
            "n_newly_attached": n_new,
            "n_lost_attachment": n_lost,
            "mode_score": mode_score,
            "new_attachment_score_contribution": new_credit,
            "effective_shortfall": 1.0 - mode_score,
            "weight": weight,
        })
    fitness = weighted_score / total_weight
    return {
        "fitness_version": FITNESS_VERSION,
        "fitness": fitness,
        "weighted_shortfall": 1.0 - fitness,
        "sufficiency_level": level,
        "mode_weights": {mode: float(mode_weights[mode]) for mode in MODES},
        "time_weights": {f"{t:g}": float(time_weights[t]) for t in sorted(time_weights)},
        "unattached_policy": ATTACHMENT_POLICY,
        "components": components,
        "provisional": True,
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
    score_only: bool = False,
) -> dict:
    """Calculate and save numerical accessibility and its baseline-aware fitness."""
    scenario_dir = Path(scenario_dir)
    output_dir = scenario_dir / "accessibility"
    summary_file = output_dir / "sufficiency_summary.csv"
    hexagon_file = output_dir / "accessibility_by_hexagon.csv.gz"
    reference_file = Path(baseline_reference_file)
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
        thresholds = tuple(sorted(float(t) for t in time_weights))
        result = calculate_accessibility(
            scenario_dir, spatial_dir, thresholds_min=thresholds,
            hour_start=hour_start, linkstats_statistic=linkstats_statistic,
            bike_speed_m_s=bike_speed_m_s,
            pt_transfer_penalty_s=pt_transfer_penalty_s,
            origin_snap_m=origin_snap_m, job_snap_m=job_snap_m,
        )
        if scenario_dir.name == baseline_name:
            car_results = result.loc[result["mode"] == "car"]
            candidate_reference = SufficiencyReference.from_baseline_car(car_results, thresholds)
            if reference_file.is_file():
                reference = SufficiencyReference.load(reference_file)
                if (set(reference.mean_car_jobs) != set(candidate_reference.mean_car_jobs)
                    or any(not math.isclose(reference.mean_car_jobs[k], v, rel_tol=1e-9)
                           for k, v in candidate_reference.mean_car_jobs.items())):
                    raise ValueError(f"Existing baseline reference differs: {reference_file}")
            else:
                reference = candidate_reference
                reference.save(reference_file)
        else:
            if not reference_file.is_file():
                raise FileNotFoundError(
                    f"Fixed baseline car reference missing: {reference_file}. "
                    f"Evaluate {baseline_name} first with identical settings."
                )
            reference = SufficiencyReference.load(reference_file)
        if set(reference.mean_car_jobs) != {
            CumulativeAccessibility.column_name(t) for t in thresholds
        }:
            raise ValueError("Baseline reference time thresholds differ from scoring thresholds")
        scored, summary = assess_sufficiency(result, reference, thresholds)
        _write_csv(hexagon_file, scored, compression="gzip")
        _write_csv(summary_file, summary)

    if not reference_file.is_file():
        raise FileNotFoundError(f"Baseline car reference is missing: {reference_file}")
    reference = SufficiencyReference.load(reference_file)
    hexagons = _load_hexagons(hexagon_file)
    mask_file = _baseline_mask_path(reference_file)
    mask = _load_or_create_baseline_mask(
        mask_file, hexagons, reference_file, baseline_name,
        is_baseline=(scenario_dir.name == baseline_name),
    )
    score = score_sufficiency(
        summary, hexagons, mask, reference,
        mode_weights=mode_weights, time_weights=time_weights, level=level,
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
        } if not score_only else None,
    })
    _write_json(output_dir / "fitness.json", score)
    return score

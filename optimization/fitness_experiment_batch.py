"""Fast STATPOP-weighted GA/sample rescoring, including one-job --workers mode.

Recommended:
 python -m optimization.fitness_experiment_batch run --workers 16 \
    --baseline ... --population ... --root ... --out ...

The original prepare/worker/finalize commands remain available for SLURM arrays.
No MATSim executions, no change to any scenario's fitness.json. Missing cutoffs
require --compute-missing (which recalculates accessibility routing, not MATSim).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import math
import os
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

MODES = ("car", "bike", "pt")
VERSION = "statpop_fitness_experiment_v2"


def atomic_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def sha256(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _get_column(t):
    t = float(t)
    return f"jobs_{t:g}min"


def required_times(cfg):
    values = {float(k) for k in cfg["TIME_WEIGHTS"]}
    if cfg["CAR_CONSTRAINT_ENABLED"]:
        values.add(float(cfg["CAR_CONSTRAINT_TIME_MIN"]))
    return sorted(values)


def load_settings():
    from optimization import fitness_experiment_settings as s
    names = (
        "TIME_WEIGHTS", "MODE_WEIGHTS", "SUFFICIENCY_FRACTION", "SHORTFALL_EXPONENT",
        "CAR_CONSTRAINT_ENABLED", "CAR_CONSTRAINT_TIME_MIN",
        "CAR_CONSTRAINT_SUFFICIENCY_FRACTION", "CAR_CONSTRAINT_TOLERANCE_PP",
        "CAR_PARKING_PENALTY_S", "HOUR_START", "LINKSTATS_STATISTIC",
        "BIKE_SPEED_M_S", "PT_TRANSFER_PENALTY_S", "ORIGIN_SNAP_M", "JOB_SNAP_M",
    )
    raw = {k: getattr(s, k) for k in names}
    raw["TIME_WEIGHTS"] = {str(float(k)): float(v) for k, v in raw["TIME_WEIGHTS"].items()}
    raw["MODE_WEIGHTS"] = {str(k): float(v) for k, v in raw["MODE_WEIGHTS"].items()}
    validate_config(raw)
    return raw


def validate_config(cfg):
    if set(cfg["MODE_WEIGHTS"]) != set(MODES):
        raise ValueError(f"MODE_WEIGHTS must contain exactly {MODES}")
    if not cfg["TIME_WEIGHTS"]:
        raise ValueError("TIME_WEIGHTS must not be empty")
    for key in ("SUFFICIENCY_FRACTION", "SHORTFALL_EXPONENT", "CAR_CONSTRAINT_SUFFICIENCY_FRACTION"):
        v = float(cfg[key])
        if not math.isfinite(v) or v <= 0:
            raise ValueError(f"{key} must be finite and >0")
    for name in ("TIME_WEIGHTS", "MODE_WEIGHTS"):
        for k, value in cfg[name].items():
            v = float(value)
            if not math.isfinite(v) or v < 0:
                raise ValueError(f"Invalid weight {name}[{k!r}] = {value!r}")
    if sum(map(float, cfg["TIME_WEIGHTS"].values())) <= 0 or sum(map(float, cfg["MODE_WEIGHTS"].values())) <= 0:
        raise ValueError("At least one positive time and mode weight required")
    if not 0 <= float(cfg["CAR_CONSTRAINT_TOLERANCE_PP"]) <= 100:
        raise ValueError("CAR_CONSTRAINT_TOLERANCE_PP must be between 0 and 100")
    if min(required_times(cfg)) <= 0:
        raise ValueError("Travel cutoffs must be >0")
    if not (0 <= float(cfg["CAR_PARKING_PENALTY_S"]) < min(required_times(cfg))*60):
        raise ValueError("Car parking penalty must be less than the shortest cutoff")
    for k in ("BIKE_SPEED_M_S", "PT_TRANSFER_PENALTY_S", "ORIGIN_SNAP_M", "JOB_SNAP_M"):
        if not math.isfinite(float(cfg[k])) or float(cfg[k]) < 0:
            raise ValueError(f"Invalid {k}")


def locate_access(path: Path):
    p = Path(path).expanduser().resolve()
    if p.is_file():
        if p.name not in ("accessibility_by_hexagon.csv", "accessibility_by_hexagon.csv.gz"):
            raise ValueError(f"Not an accessibility cache: {p}")
        return p
    for folder in (p / "accessibility", p):
        for name in ("accessibility_by_hexagon.csv.gz", "accessibility_by_hexagon.csv"):
            if (folder / name).is_file():
                return (folder / name).resolve()
    raise FileNotFoundError(f"No accessibility_by_hexagon.csv[.gz] under {p}")


def scenario_dir(file: Path):
    return file.parent.parent if file.parent.name == "accessibility" else file.parent


def provenance(file: Path, cfg: dict, times, *, allow_missing=False, ignore_cache_provenance=False):
    if ignore_cache_provenance:
        return
    folder = file.parent
    summary = next((p for p in (folder / "sufficiency_summary_statpop.csv",
                                folder / "sufficiency_summary.csv",
                                folder / "sufficiency_summary.csv.gz") if p.is_file()), None)
    if summary is None:
        raise ValueError(f"No sufficiency_summary.csv[.gz] for verifying car parking: {folder}")
    data = pd.read_csv(summary)
    if "car_parking_penalty_s" not in data:
        raise ValueError(f"Cache lacks car-parking provenance: {summary}")
    park = pd.to_numeric(data["car_parking_penalty_s"], errors="raise")
    if park.isna().any() or not np.allclose(park.to_numpy(dtype=float), cfg["CAR_PARKING_PENALTY_S"]):
        raise ValueError(f"Cache car parking differs from {cfg['CAR_PARKING_PENALTY_S']}s: {summary}")
    if "time_min" not in data:
        raise ValueError(f"Missing cutoff metadata: {summary}")
    present = set(pd.to_numeric(data.time_min, errors="raise").to_numpy(float))
    if not allow_missing and not set(map(float, times)).issubset(present):
        raise ValueError(f"Cached sufficiency metadata lacks {sorted(set(times)-present)} min: {summary}")
    # Inspect any available evaluated MATSim fitness provenance as an additional
    # safeguard against mixing accessibility computed with different parameters.
    scen = scenario_dir(file)
    settings_file = next((p for p in (scen / "accessibility" / "fitness_statpop.json",
                                     scen / "accessibility" / "fitness.json") if p.is_file()), None)
    if settings_file is not None:
        data = json.loads(settings_file.read_text(encoding="utf-8"))
        params = data.get("accessibility_parameters", {})
        checks = {
            "hour_start": "HOUR_START", "linkstats_statistic": "LINKSTATS_STATISTIC",
            "bike_speed_m_s": "BIKE_SPEED_M_S",
            "pt_transfer_penalty_s": "PT_TRANSFER_PENALTY_S",
            "origin_snap_m": "ORIGIN_SNAP_M", "job_snap_m": "JOB_SNAP_M",
            "car_parking_penalty_s": "CAR_PARKING_PENALTY_S",
        }
        for field, key in checks.items():
            if field not in params:
                continue
            observed, expected = params[field], cfg[key]
            if isinstance(expected, (float, int)):
                match = math.isclose(float(observed), float(expected), abs_tol=1e-8, rel_tol=1e-9)
            else:
                match = str(observed) == str(expected)
            if not match:
                raise ValueError(f"Cached accessibility parameter {field}={observed!r} "
                                 f"does not match selected {expected!r}: {settings_file}")
    # If only summary exists, parking/time are checked, but OTHER routing
    # parameters are not independently verifiable. See README.


def read_population(population_path: Path):
    p = pd.read_csv(population_path, dtype={"hex_id": str})
    if not {"hex_id", "population"}.issubset(p):
        raise ValueError("STATPOP file must contain hex_id,population")
    if p.hex_id.isna().any() or p.hex_id.duplicated().any():
        raise ValueError("STATPOP hex_id missing or duplicated")
    p["population"] = pd.to_numeric(p["population"], errors="raise")
    a = p["population"].to_numpy(float)
    if np.isnan(a).any() or not np.isfinite(a).all() or (a < 0).any() or a.sum() <= 0:
        raise ValueError("STATPOP populations must be finite >= 0 with positive sum")
    return p[["hex_id", "population"]]


def parse_attached(s: pd.Series):
    if pd.api.types.is_bool_dtype(s):
        return s.astype(bool)
    values = s.astype(str).str.strip().str.lower()
    if not values.isin(("true", "false", "1", "0")).all():
        raise ValueError("Invalid values in attached column")
    return values.isin(("true", "1"))


def read_accessibility(file: Path, population: pd.DataFrame, times, cfg: dict,
                       *, compute_missing=False, force_recalculate=False,
                       spatial: Path | None = None):
    if force_recalculate or not file.is_file():
        if not compute_missing:
            raise FileNotFoundError(f"Missing accessibility cache: {file}; use --compute-missing to route")
        df = None
    else:
        df = pd.read_csv(file, dtype={"hex_id": str, "mode": str})
        if not set(_get_column(t) for t in times).issubset(df.columns):
            if not compute_missing:
                missing = sorted(set(_get_column(t) for t in times)-set(df.columns))
                raise ValueError(f"Missing accessibility cutoff columns {missing} in {file}; "
                                 "pass --compute-missing with complete MATSim scenario files")
            df = None
    if df is None:
        if spatial is None:
            raise ValueError("--compute-missing requires --spatial")
        scen = scenario_dir(file)
        for fname in ("network.xml.gz", "transit_schedule.xml.gz", "targets/linkstats.txt.gz"):
            if not (scen / fname).is_file():
                raise FileNotFoundError(f"Cannot recalculate accessibility: {scen/fname}")
        try:
            from accessibility.calculations.evaluate import calculate_accessibility
        except ImportError as e:
            raise RuntimeError("Cannot import accessibility pipeline for new cutoffs") from e
        kwargs = dict(thresholds_min=tuple(times), hour_start=cfg["HOUR_START"],
                      linkstats_statistic=cfg["LINKSTATS_STATISTIC"],
                      bike_speed_m_s=cfg["BIKE_SPEED_M_S"],
                      pt_transfer_penalty_s=cfg["PT_TRANSFER_PENALTY_S"],
                      origin_snap_m=cfg["ORIGIN_SNAP_M"], job_snap_m=cfg["JOB_SNAP_M"],
                      car_parking_penalty_s=cfg["CAR_PARKING_PENALTY_S"])
        try:
            import inspect
            sig = inspect.signature(calculate_accessibility)
            if "population_weighted" in sig.parameters:
                kwargs["population_weighted"] = False
        except (ValueError, TypeError):
            pass
        df = calculate_accessibility(scen, spatial, **kwargs)
    required = {"hex_id", "mode", "attached"} | {_get_column(t) for t in times}
    if required - set(df):
        raise ValueError(f"Missing columns {required-set(df)} in accessibility {file}")
    df = df.copy()
    if df[["hex_id", "mode", "attached"]].isna().any().any():
        raise ValueError(f"Missing origin metadata: {file}")
    if df.duplicated(["mode", "hex_id"]).any() or set(df["mode"]) != set(MODES):
        raise ValueError(f"Must have unique origin IDs in each mode car/bike/pt: {file}")
    df["attached"] = parse_attached(df["attached"])
    ids = set(population.hex_id)
    for mode, g in df.groupby("mode"):
        if set(g.hex_id) != ids:
            raise ValueError(f"Origin IDs differ from STATPOP grid for {mode} in {file}")
    if "population" in df:
        df = df.drop(columns="population")
    df = df.merge(population, on="hex_id", how="left", validate="many_to_one")
    df["population"] = df["population"].astype(float)
    if df["population"].isna().any():
        raise ValueError("Missing population after join")
    for col in (_get_column(t) for t in times):
        values = pd.to_numeric(df[col], errors="raise")
        if values[df["attached"]].isna().any() or not np.isfinite(values[df["attached"]].to_numpy(float)).all():
            raise ValueError(f"Invalid attached accessibility {col}: {file}")
        if (values[df["attached"]] < 0).any():
            raise ValueError(f"Negative accessibility {col}: {file}")
        if values[~df["attached"]].notna().any():
            raise ValueError(f"Unattached modes must have NaN jobs ({col}): {file}")
        df[col] = values.where(df["attached"], 0.0).fillna(0.0)
    return df


def weighted_mean(x, weights):
    x, w = np.asarray(x, dtype=float), np.asarray(weights, dtype=float)
    if x.shape != w.shape or x.ndim != 1 or not np.isfinite(x).all() or (w < 0).any() or w.sum() <= 0:
        raise ValueError("Invalid weighted mean")
    return float(np.dot(x, w)/w.sum())


def baseline_benchmarks(baseline: pd.DataFrame, times):
    car = baseline[baseline["mode"] == "car"]
    weights = car["population"].to_numpy(float)
    return {_get_column(t): weighted_mean(car[_get_column(t)].to_numpy(float), weights) for t in times}


def threshold(ref: dict, time: float, fraction: float):
    base = float(ref[_get_column(time)])
    if base <= 0 or fraction <= 0:
        raise ValueError(f"Unusable reference threshold at {time:g}min: {base} jobs")
    return base * fraction


def origin_score(accessible_jobs: np.ndarray, jobs_threshold: float, exponent: float) -> np.ndarray:
    """EDIT THIS MATH if you want to experiment beyond the settings above.

    This is Martens-style normalized shortfall, with a continuous score between 0 and 1.
    Population weighting occurs LATER in score(); don't multiply jobs here.
    """
    return 1.0 - np.maximum(0.0, 1.0 - accessible_jobs/jobs_threshold) ** exponent


def score(df: pd.DataFrame, reference: dict, cfg: dict, baseline_car_share: float):
    """Return BOTH raw continuous F(x) and constrained GA ranking fitness."""
    mode_w = cfg["MODE_WEIGHTS"]
    time_w = {float(k): float(v) for k, v in cfg["TIME_WEIGHTS"].items()}
    t_constraint = float(cfg["CAR_CONSTRAINT_TIME_MIN"])
    f_constraint = float(cfg["CAR_CONSTRAINT_SUFFICIENCY_FRACTION"])
    exponent = float(cfg["SHORTFALL_EXPONENT"])
    records = []
    numerator = 0.0
    denom = 0.0
    population_by_mode = []
    for mode in MODES:
        subset = df.loc[df["mode"] == mode].sort_values("hex_id")
        people = subset["population"].to_numpy(float)
        total_people = float(people.sum())
        population_by_mode.append(total_people)
        for t, tw in sorted(time_w.items()):
            jobs = subset[_get_column(t)].to_numpy(float)
            suff_jobs = threshold(reference, t, float(cfg["SUFFICIENCY_FRACTION"]))
            ind_score = origin_score(jobs, suff_jobs, exponent)
            score_m = weighted_mean(ind_score, people)
            share_m = weighted_mean((jobs >= suff_jobs).astype(float), people)
            weight = float(mode_w[mode])*tw
            numerator += score_m*weight
            denom += weight
            records.append({"mode": mode, "time_min": t, "weight": weight,
                            "threshold_jobs": suff_jobs, "score": score_m,
                            "sufficient_population_share": share_m,
                            "n_populated_hexagons": int((people > 0).sum()),
                            "population": total_people})
    if not np.allclose(population_by_mode, population_by_mode[0]):
        raise ValueError("Different population total across modes")
    raw = numerator/denom
    car = df.loc[df["mode"] == "car"]
    car_jobs = car[_get_column(t_constraint)].to_numpy(float) if cfg["CAR_CONSTRAINT_ENABLED"] else None
    if cfg["CAR_CONSTRAINT_ENABLED"]:
        car_people = car["population"].to_numpy(float)
        car_thresh = threshold(reference, t_constraint, f_constraint)
        share_car = weighted_mean((car_jobs >= car_thresh).astype(float), car_people)
        min_share = max(0.0, float(baseline_car_share)-float(cfg["CAR_CONSTRAINT_TOLERANCE_PP"])/100.0)
        violation = max(0.0, min_share-share_car)
        feasible = violation <= 1e-12
    else:
        car_thresh, share_car, min_share, violation, feasible = (None, None, None, 0.0, True)
    fitness = raw if feasible else -violation
    return {"fitness": float(fitness), "raw_fitness": float(raw),
            "feasible": bool(feasible), "violation_pp": float(100*violation),
            "car_sufficient_share": share_car, "car_minimum_share": min_share,
            "car_threshold_jobs": car_thresh, "baseline_car_share": baseline_car_share,
            "population": population_by_mode[0], "components": records}


def compute_baseline_car_share(df, reference, cfg):
    if not cfg["CAR_CONSTRAINT_ENABLED"]:
        return None
    car = df[df["mode"] == "car"]
    cutoff = float(cfg["CAR_CONSTRAINT_TIME_MIN"])
    jobs = car[_get_column(cutoff)].to_numpy(float)
    pop = car["population"].to_numpy(float)
    t = threshold(reference, cutoff, float(cfg["CAR_CONSTRAINT_SUFFICIENCY_FRACTION"]))
    return weighted_mean((jobs >= t).astype(float), pop)


def paths_manifest(files, baseline_path):
    rows = []
    for idx, file in enumerate(files):
        scenario = scenario_dir(file)
        h = hashlib.sha1(str(file).encode("utf-8")).hexdigest()[:12]
        rows.append({"index": idx, "scenario": scenario.name,
                     "source": str(file), "key": f"{scenario.name}_{h}"})
    return rows


def prepare(args):
    out = args.out.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Experiment output already populated: {out}; choose a NEW directory")
    out.mkdir(parents=True, exist_ok=True)
    cfg = load_settings()
    times = required_times(cfg)
    pop_path = args.population.resolve()
    pop = read_population(pop_path)
    try:
        baseline_file = locate_access(args.baseline)
    except FileNotFoundError:
        scen = args.baseline.resolve()
        if not args.compute_missing or not (scen / "network.xml.gz").is_file():
            raise
        baseline_file = scen / "accessibility" / "accessibility_by_hexagon.csv.gz"
    baseline_data = read_accessibility(baseline_file, pop, times, cfg,
                                      compute_missing=args.compute_missing,
                                      force_recalculate=args.force_recalculate,
                                      spatial=args.spatial)
    if not args.force_recalculate and baseline_file.is_file():
        provenance(baseline_file, cfg, times, allow_missing=args.compute_missing,
                   ignore_cache_provenance=args.ignore_cache_provenance)
    ref = baseline_benchmarks(baseline_data, times)
    base_share = compute_baseline_car_share(baseline_data, ref, cfg)
    base_score = score(baseline_data, ref, cfg, base_share)
    all_files = set()
    for root in args.root:
        root = root.resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"Dataset root does not exist: {root}")
        for p in root.rglob("accessibility_by_hexagon.csv*"):
            if p.name in ("accessibility_by_hexagon.csv", "accessibility_by_hexagon.csv.gz"):
                all_files.add(p.resolve())
    # When new time cutoffs require routing, include MATSim scenarios with no
    # existing accessibility cache (common for compacted training samples).
    if args.compute_missing:
        already = {scenario_dir(f) for f in all_files}
        for root in args.root:
            for ls in root.resolve().rglob("linkstats.txt.gz"):
                if ls.parent.name != "targets":
                    continue
                scen = ls.parent.parent
                if scen not in already and (scen / "network.xml.gz").is_file():
                    all_files.add(scen / "accessibility" / "accessibility_by_hexagon.csv.gz")
                    already.add(scen)
    for path in args.scenario:
        try:
            all_files.add(locate_access(path))
        except FileNotFoundError:
            scen = path.resolve()
            if not args.compute_missing or not (scen / "network.xml.gz").is_file():
                raise
            all_files.add(scen / "accessibility" / "accessibility_by_hexagon.csv.gz")
    all_files.discard(baseline_file)
    all_files = {f for f in all_files if scenario_dir(f) != scenario_dir(baseline_file)}
    # Eliminate duplicate csv/.csv.gz from the same directory; .gz preferred
    by_scenario = {}
    for file in sorted(all_files):
        scenario = scenario_dir(file)
        prior = by_scenario.get(scenario)
        if prior is None or file.name.endswith(".gz"):
            by_scenario[scenario] = file
    files = sorted(by_scenario.values(), key=str)
    if args.limit:
        files = files[:args.limit]
    if not files:
        raise ValueError("No scenario accessibility caches found")
    chunk = int(args.chunk_size)
    if chunk < 1:
        raise ValueError("chunk-size must be >=1")
    manifest = paths_manifest(files, baseline_file)
    cfg_file = out / "experiment.json"
    config = {"version": VERSION, "config": cfg, "reference": ref,
              "baseline_car_share": base_share, "baseline_score": base_score,
              "baseline_source": str(baseline_file),
              "baseline_sha256": sha256(baseline_file) if baseline_file.is_file() else None,
              "population_file": str(pop_path), "population_sha256": sha256(pop_path),
              "spatial": str(args.spatial.resolve()) if args.spatial else None,
              "compute_missing": args.compute_missing,
              "force_recalculate": args.force_recalculate,
              "ignore_cache_provenance": args.ignore_cache_provenance,
              "required_times": times, "n_scenarios": len(manifest),
              "chunk_size": chunk, "n_chunks": math.ceil(len(manifest)/chunk),
              "scorer_sha256": sha256(Path(__file__))}
    atomic_json(cfg_file, config)
    with (out / "manifest.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=("index", "scenario", "source", "key"))
        w.writeheader(); w.writerows(manifest)
    atomic_json(out / "baseline_score.json", base_score)
    print(f"STATPOP residents={pop.population.sum():,.0f}; populated origins={(pop.population>0).sum()}; "
          f"unpopulated={(pop.population==0).sum()}", flush=True)
    print(f"Baseline resident-weighted car reference (jobs) = {ref}", flush=True)
    print(f"Baseline fitness F0={base_score['raw_fitness']:.8f} (feasible={base_score['feasible']})", flush=True)
    print(f"Prepared {len(manifest)} scenarios in {config['n_chunks']} chunks of {chunk}. Output: {out}", flush=True)


def worker(args):
    out = args.out.resolve()
    config = json.loads((out / "experiment.json").read_text(encoding="utf-8"))
    if config["version"] != VERSION:
        raise ValueError("Unsupported experiment version")
    if config.get("scorer_sha256") != sha256(Path(__file__)):
        raise ValueError("Scorer source changed since prepare; abort to avoid mixed fitness calculations")
    pop_path = Path(config["population_file"])
    if sha256(pop_path) != config["population_sha256"]:
        raise ValueError("STATPOP input changed after experiment preparation")
    pop = read_population(pop_path)
    with (out / "manifest.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    index = int(args.task_index)
    if not 0 <= index < config["n_chunks"]:
        raise ValueError(f"task-index must be 0..{config['n_chunks']-1}")
    part = rows[index*config["chunk_size"]:(index+1)*config["chunk_size"]]
    target = out / "chunks" / f"chunk_{index:05d}.json"
    if target.exists():
        print(f"Chunk {index} already exists: {target}", flush=True)
        return
    completed, errors = [], []
    cfg = config["config"]
    times = config["required_times"]
    for row in part:
        file = Path(row["source"])
        try:
            if not config["force_recalculate"] and file.is_file():
                provenance(file, cfg, times, allow_missing=config["compute_missing"],
                           ignore_cache_provenance=config["ignore_cache_provenance"])
            data = read_accessibility(file, pop, times, cfg,
                                      compute_missing=config["compute_missing"],
                                      force_recalculate=config["force_recalculate"],
                                      spatial=Path(config["spatial"]) if config["spatial"] else None)
            result = score(data, config["reference"], cfg, config["baseline_car_share"])
            result["scenario"] = row["scenario"]
            result["source"] = row["source"]
            result["key"] = row["key"]
            result["delta_fitness"] = result["raw_fitness"] - config["baseline_score"]["raw_fitness"]
            result["delta_fitness_pp"] = 100 * result["delta_fitness"]
            result["delta_fitness_pct_of_baseline"] = (100 * result["delta_fitness"] / config["baseline_score"]["raw_fitness"]
                                                       if config["baseline_score"]["raw_fitness"] > 0 else None)
            result["delta_car_sufficiency_pp"] = (100 * (result["car_sufficient_share"]-config["baseline_car_share"])
                                                   if cfg["CAR_CONSTRAINT_ENABLED"] else None)
            completed.append(result)
        except Exception as e:
            errors.append({"scenario": row["scenario"], "source": row["source"],
                           "error": f"{type(e).__name__}: {e}"})
    atomic_json(target, {"task_index": index, "n_expected": len(part), "completed": completed,
                         "errors": errors})
    print(f"Chunk {index}: {len(completed)} scored, {len(errors)} failed", flush=True)
    if errors and args.fail_on_error:
        raise RuntimeError(f"Chunk {index} had {len(errors)} failures; see {target}")


def finalize(args):
    out = args.out.resolve()
    config = json.loads((out / "experiment.json").read_text(encoding="utf-8"))
    files = [out / "chunks" / f"chunk_{i:05d}.json" for i in range(config["n_chunks"])]
    missing = [str(x) for x in files if not x.is_file()]
    if missing:
        raise RuntimeError(f"Missing {len(missing)} chunk outputs; e.g. {missing[:3]}")
    records, errors = [], []
    for file in files:
        content = json.loads(file.read_text(encoding="utf-8"))
        records += content["completed"]
        errors += content["errors"]
    if len(records) + len(errors) != config["n_scenarios"]:
        raise RuntimeError("Scenario count mismatch; some results were lost or duplicated")
    df = pd.DataFrame([{k:v for k,v in row.items() if k != "components"} for row in records])
    if len(df):
        df = df.sort_values(["feasible", "raw_fitness"], ascending=[False,False])
    df.to_csv(out / "fitness_scores.csv.gz", index=False, compression="gzip")
    pd.DataFrame(errors, columns=["scenario","source","error"]).to_csv(out / "failures.csv",index=False)
    base = config["baseline_score"]
    feasible = [r for r in records if r["feasible"]]
    best = max(feasible, key=lambda x:x["raw_fitness"]) if feasible else None
    best_raw = max(records,key=lambda x:x["raw_fitness"]) if records else None
    baseline_beats_all = best is None or best["raw_fitness"] <= base["raw_fitness"]
    payload = {"total": config["n_scenarios"], "scored": len(records), "failed": len(errors),
               "feasible": len(feasible), "baseline": base, "best_feasible": best,
               "best_raw": best_raw, "baseline_beats_all_feasible_samples": baseline_beats_all,
               "n_feasible_above_baseline": sum(1 for r in feasible if r["delta_fitness"] > 0),
               "n_above_baseline": sum(1 for r in records if r["delta_fitness"] > 0),
               "n_at_baseline": sum(1 for r in records if abs(r["delta_fitness"]) < 1e-12),
               "n_below_baseline": sum(1 for r in records if r["delta_fitness"] < 0)}
    atomic_json(out/"best_vs_baseline.json",payload)
    lines = [
        "# STATPOP-weighted fitness: best candidate vs baseline", "",
        f"Scored **{len(records)}** / **{config['n_scenarios']}** (failed **{len(errors)}**).",
        f"Feasible candidates: **{len(feasible)}**.", "",
        f"Baseline continuous fitness: **{base['raw_fitness']:.8f}**.",
    ]
    if best:
        lines += ["", f"Best feasible: **{best['scenario']}** (`{best['source']}`).",
                  f"Fitness: **{best['raw_fitness']:.8f}**; difference **{best['delta_fitness']:+.8f}** "
                  f"({best['delta_fitness_pp']:+.5f} score percentage points; "
                  f"{best['delta_fitness_pct_of_baseline']:+.3f}% relative).", "",
                  "| Mode | Cutoff (min) | Baseline mode score | Best mode score | Difference |",
                  "|---|---:|---:|---:|---:|"]
        for c in best['components']:
            b = next(x for x in base['components'] if x['mode'] == c['mode'] and x['time_min'] == c['time_min'])
            lines.append(f"| {c['mode']} | {c['time_min']:g} | {b['score']:.6f} | {c['score']:.6f} | {c['score']-b['score']:+.6f} |")
        if best['car_sufficient_share'] is not None:
            lines += ["", f"Car-sufficient resident share: baseline **{100*config['baseline_car_share']:.2f}%**, "
                      f"best **{100*best['car_sufficient_share']:.2f}%**, "
                      f"change **{best['delta_car_sufficiency_pp']:+.2f} pp** "
                      f"(minimum **{100*best['car_minimum_share']:.2f}%**)."]
    else:
        lines += ["", "No feasible candidates under the selected car constraint."]
    (out / "best_vs_baseline.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    if len(df):
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(8, 4.5))
            ax.hist(df['delta_fitness'].to_numpy(float), bins=min(45, max(5, round(len(df)**0.5))),
                    edgecolor='white', linewidth=.5)
            ax.axvline(0, color='black', linestyle='--', lw=1.2, label='Baseline')
            ax.set_xlabel('Population-weighted fitness difference from baseline')
            ax.set_ylabel('Number of scenarios')
            ax.set_title('Baseline-relative fitness distribution')
            ax.legend(loc='best')
            fig.tight_layout()
            fig.savefig(out / 'fitness_delta_histogram.png', dpi=200)
            plt.close(fig)
        except ImportError:
            print('matplotlib unavailable; skipped optional histogram', flush=True)
    print(f"Scored {len(records)}/{config['n_scenarios']} scenarios; failed={len(errors)}, feasible={len(feasible)}", flush=True)
    print(f"BASELINE: F={base['raw_fitness']:.8f}", flush=True)
    if best:
        print(f"BEST FEASIBLE: {best['scenario']} | F={best['raw_fitness']:.8f} | "
              f"delta={best['delta_fitness']:+.8f} ({best['delta_fitness_pp']:+.5f} score-pp) | "
              f"relative={best['delta_fitness_pct_of_baseline']:+.3f}%", flush=True)
        for comp in best["components"]:
            b = next(x for x in base["components"] if x["mode"] == comp["mode"] and x["time_min"] == comp["time_min"])
            print(f"  {comp['mode']} {comp['time_min']:g}min: baseline={b['score']:.6f}, "
                  f"best={comp['score']:.6f}, delta={comp['score']-b['score']:+.6f}", flush=True)
        if best["car_sufficient_share"] is not None:
            print(f"  Car-sufficient residents: baseline={100*config['baseline_car_share']:.2f}%, "
                  f"best={100*best['car_sufficient_share']:.2f}%, "
                  f"change={best['delta_car_sufficiency_pp']:+.2f} pp; "
                  f"minimum={100*best['car_minimum_share']:.2f}%", flush=True)
    elif records:
        print("NO FEASIBLE CANDIDATES under chosen car constraint", flush=True)
    print(f"Improved feasible candidates: {payload['n_feasible_above_baseline']}/{len(feasible)}", flush=True)
    if baseline_beats_all:
        print("RESULT: baseline is at least as good as every scored feasible intervention.", flush=True)
    else:
        print("RESULT: at least one intervention improved resident-weighted F over baseline.", flush=True)
    if errors:
        print(f"WARNING: {len(errors)} errors. See {out/'failures.csv'}", flush=True)


def run_parallel(args):
    """Prepare, score each chunk in a process pool, and finalize in one SLURM job."""
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")
    prepare(args)  # Freezes settings and population hashes before parallel execution.
    output = args.out.resolve()
    config = json.loads((output / "experiment.json").read_text(encoding="utf-8"))
    n_chunks = int(config["n_chunks"])
    n_processes = min(args.workers, n_chunks)
    print(f"Scoring {config['n_scenarios']} scenarios in {n_chunks} chunks "
          f"using {n_processes} worker processes", flush=True)
    # Each worker writes only its own atomic chunk JSON; no output write contention.
    first_errors = []
    with ProcessPoolExecutor(max_workers=n_processes) as executor:
        jobs = {
            executor.submit(worker, SimpleNamespace(
                out=output, task_index=index, fail_on_error=True)): index
            for index in range(n_chunks)
        }
        for future in as_completed(jobs):
            index = jobs[future]
            try:
                future.result()
            except Exception as exc:
                first_errors.append(f"chunk {index}: {type(exc).__name__}: {exc}")
    if first_errors:
        raise RuntimeError(f"{len(first_errors)} chunks failed. Check {output/'chunks'} "
                           f"and scenario errors: {first_errors[:5]}")
    finalize(SimpleNamespace(out=output))



def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    s = p.add_subparsers(dest="command", required=True)
    def add_input_options(parser):
        parser.add_argument("--baseline", type=Path, required=True, help="Baseline scenario directory or accessibility cache CSV")
        parser.add_argument("--population", type=Path, required=True, help="STATPOP population_by_hexagon.csv")
        parser.add_argument("--spatial", type=Path, default=None, help="Required for recomputing accessibility at new cutoffs")
        parser.add_argument("--root", type=Path, action="append", default=[], help="Recursive scenario dataset root; repeatable")
        parser.add_argument("--scenario", type=Path, action="append", default=[], help="Single scenario directory; repeatable")
        parser.add_argument("--out", type=Path, required=True)
        parser.add_argument("--chunk-size", type=int, default=20, help="Scenarios per worker task (default: 20)")
        parser.add_argument("--limit", type=int, default=0)
        parser.add_argument("--compute-missing", action="store_true", help="Reroute only when required time cutoff is missing")
        parser.add_argument("--force-recalculate", action="store_true", help="Reroute even if cached cutoff exists")
        parser.add_argument("--ignore-cache-provenance", action="store_true", help="DANGEROUS: for only older provenance-less caches")

    pp = s.add_parser("prepare", help="Prepare snapshot for SLURM arrays (advanced)")
    add_input_options(pp)
    rr = s.add_parser("run", help="Run all scenarios with --workers in one job (recommended)")
    add_input_options(rr)
    rr.add_argument("--workers", type=int, default=1, help="Number of parallel Python processes")
    ww = s.add_parser("worker")
    ww.add_argument("--out", type=Path, required=True)
    ww.add_argument("--task-index", type=int, default=None)
    ww.add_argument("--fail-on-error", action="store_true")
    ff = s.add_parser("finalize")
    ff.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)
    if args.command in ("prepare", "run"):
        parser = rr if args.command == "run" else pp
        if not args.root and not args.scenario:
            parser.error("Provide --root and/or --scenario")
        if args.force_recalculate and not args.compute_missing:
            parser.error("--force-recalculate requires --compute-missing")
        if args.compute_missing and not args.spatial:
            parser.error("--compute-missing needs --spatial")
        if args.command == "run":
            run_parallel(args)
        else:
            prepare(args)
    elif args.command == "worker":
        if args.task_index is None:
            env = os.environ.get("SLURM_ARRAY_TASK_ID")
            if env is None:
                ww.error("Provide --task-index or SLURM_ARRAY_TASK_ID")
            args.task_index = int(env)
        worker(args)
    else:
        finalize(args)

if __name__ == "__main__":
    main()

"""Trial 01 sensitivity: zero car parking and a RE-CALIBRATED common sufficiency threshold.

Uses successful Trial 01 accessibility caches for bike/PT and MATSim network/linkstats
for CAR ONLY; never edits historical MATSim runs or accessibility caches.

  python -m optimization.experiment_parking0 --trial .../trial_01 \
      --population .../population_by_hexagon.csv --spatial .../spatial_baseline \
      --out .../parking0_low_all_modes --workers 16

Linux fork workers; resumes completed chunks. Saves per-hexagon car access by chunk.
"""
from __future__ import annotations

import argparse
import copy
import csv
import gzip
import hashlib
import json
import math
import multiprocessing as mp
import os
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from optimization.fitness_experiment_batch import (
    MODES, atomic_json, baseline_benchmarks, provenance, read_accessibility,
    read_population, score, sha256, weighted_mean,
)

VERSION = "trial01_zero_parking_car_only_recalibrated_v1"
_WORKER = None
TIME = 20.0
JOBS = "jobs_20min"


def require_trial_01(cfg):
    """Prevent inadvertently mixing this experiment with an unrelated objective."""
    times = {float(k): float(v) for k, v in cfg["TIME_WEIGHTS"].items()}
    weights = {str(k): float(v) for k, v in cfg["MODE_WEIGHTS"].items()}
    if (times != {20.0: 1.0}
            or weights != {"car": 1.0, "bike": 1.0, "pt": 1.0}
            or not math.isclose(float(cfg["SUFFICIENCY_FRACTION"]), 0.5)
            or not math.isclose(float(cfg["SHORTFALL_EXPONENT"]), 2.0)
            or cfg["CAR_CONSTRAINT_ENABLED"]
            or not math.isclose(float(cfg["CAR_PARKING_PENALTY_S"]), 300.0)):
        raise ValueError("--trial must use Trial 01 settings: equal modes, 20min, "
                         "low=.5, squared=2, no constraint, old parking=300s")


def car_accessibility(scenario: Path, origins: pd.DataFrame, jobs: pd.DataFrame,
                      cfg: dict, *, validate_old_parking=False) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Exact same car network, snapping and RoadRouter as calculate_accessibility()."""
    from accessibility.calculations.cumulative import CumulativeAccessibility
    from accessibility.inputs.grid.snapping import ScenarioRoadSnapper
    from accessibility.inputs.links.linkstats import LinkTravelTimes
    from accessibility.inputs.links.network import CarNetwork
    from accessibility.routing.road import RoadRouter

    network_file = scenario / "network.xml.gz"
    linkstats_file = scenario / "targets" / "linkstats.txt.gz"
    if not network_file.is_file() or not linkstats_file.is_file():
        missing = [str(x) for x in (network_file, linkstats_file) if not x.is_file()]
        raise FileNotFoundError(f"Missing completed MATSim network/linkstats: {missing}")
    linkstats = LinkTravelTimes.from_linkstats(
        linkstats_file, hour_start=int(cfg["HOUR_START"]),
        statistic=cfg["LINKSTATS_STATISTIC"])
    car = CarNetwork.from_matsim_network(network_file, linkstats)
    attachments = ScenarioRoadSnapper(
        origins, jobs,
        max_origin_distance_m=float(cfg["ORIGIN_SNAP_M"]),
        max_job_distance_m=float(cfg["JOB_SNAP_M"]),
    ).attach(car.graph)
    router = RoadRouter(car.graph, attachments.origins.connectors,
                        attachments.jobs.connectors, jobs)
    origin_ids = origins["hex_id"].astype(str)
    cutoff = CumulativeAccessibility((TIME,))
    no_parking = cutoff.run(router, origin_ids, mode="car", extra_time_s=0.0)
    old_parking = cutoff.run(router, origin_ids, mode="car", extra_time_s=300.0) if validate_old_parking else None
    return no_parking, old_parking


def assert_baseline_routing_match(cached: pd.DataFrame, rerouted: pd.DataFrame) -> None:
    """Stop if the exact original car calculation cannot be reproduced."""
    old = cached.loc[cached["mode"] == "car", ["hex_id", "attached", JOBS]].copy()
    new = rerouted[["hex_id", "attached", JOBS]].copy()
    old = old.sort_values("hex_id").reset_index(drop=True)
    new = new.sort_values("hex_id").reset_index(drop=True)
    if old["hex_id"].tolist() != new["hex_id"].tolist():
        raise ValueError("Baseline origin IDs differ after car rerouting")
    if not np.array_equal(old["attached"].to_numpy(bool), new["attached"].to_numpy(bool)):
        raise ValueError("Baseline car attachments differ after rerouting")
    old_jobs = old[JOBS].to_numpy(float)
    new_jobs = new[JOBS].fillna(0).to_numpy(float)
    if not np.allclose(old_jobs, new_jobs, rtol=0, atol=1e-7):
        n_bad = int((abs(old_jobs-new_jobs) > 1e-7).sum())
        max_error = float(np.max(abs(old_jobs-new_jobs)))
        raise ValueError(f"Baseline cached vs rerouted 5-min parking differ: "
                         f"{n_bad} origins, max error={max_error}; abort")


def replace_car(cached: pd.DataFrame, rerouted: pd.DataFrame) -> pd.DataFrame:
    """Replace only car's jobs and attachment; preserve original bike/PT exactly."""
    a = cached.copy()
    assert set(a["mode"]) == set(MODES)
    carrows = a.loc[a["mode"] == "car", ["hex_id", "population"]].copy()
    if carrows["hex_id"].duplicated().any() or rerouted["hex_id"].duplicated().any():
        raise ValueError("Duplicate car origin ID")
    merged = carrows.merge(rerouted[["hex_id", "attached", JOBS]], on="hex_id",
                           how="left", validate="one_to_one", indicator=True)
    if not (merged["_merge"] == "both").all() or len(merged) != len(rerouted):
        raise ValueError("Car reroute origin IDs do not match cache")
    if not merged["attached"].notna().all():
        raise ValueError("Missing car attachment in recalculated data")
    merged[JOBS] = merged[JOBS].where(merged["attached"].astype(bool), 0.0).fillna(0.0)
    if not np.isfinite(merged[JOBS].to_numpy(float)).all():
        raise ValueError("Recalculated car accessibility has nonfinite jobs")
    c = a["mode"] == "car"
    rmap = merged.set_index("hex_id")
    a.loc[c, JOBS] = a.loc[c, "hex_id"].map(rmap[JOBS]).to_numpy(float)
    a.loc[c, "attached"] = a.loc[c, "hex_id"].map(rmap["attached"]).to_numpy(bool)
    before_bp = cached.loc[cached["mode"].isin(("bike", "pt")), ["mode", "hex_id", "attached", JOBS]].reset_index(drop=True)
    after_bp = a.loc[a["mode"].isin(("bike", "pt")), ["mode", "hex_id", "attached", JOBS]].reset_index(drop=True)
    if not before_bp.equals(after_bp):
        raise AssertionError("Bike/PT cache changed while replacing car accessibility")
    return a


def _get_component(result, mode):
    return next(x for x in result["components"] if x["mode"] == mode and x["time_min"] == TIME)


def statpop_car_jobs(frame):
    c = frame.loc[frame["mode"] == "car"]
    return weighted_mean(c[JOBS].to_numpy(float), c["population"].to_numpy(float))


def save_csv_gzip_atomic(df: pd.DataFrame, dest: Path):
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + f".{os.getpid()}.tmp")
    try:
        df.to_csv(tmp, index=False, compression="gzip")
        os.replace(tmp, dest)
    finally:
        if tmp.is_file():
            tmp.unlink()


def read_trial_successes(trial: Path, limit: int = 0):
    df = pd.read_csv(trial / "fitness_scores.csv.gz", dtype={"scenario": str, "source": str, "key": str})
    required = {"key", "source", "scenario", "raw_fitness"}
    if not required.issubset(df):
        raise ValueError(f"Missing columns in Trial 01 scores: {required-set(df.columns)}")
    if df["key"].duplicated().any() or df["source"].duplicated().any():
        raise ValueError("Duplicate scenario source/key in input Trial 01 results")
    records = df[["key", "source", "scenario", "raw_fitness"]].sort_values("key")
    if limit:
        records = records.head(limit)
    return records.to_dict("records")


def prepare(args):
    trial = args.trial.expanduser().resolve(strict=True)
    spatial = args.spatial.expanduser().resolve(strict=True)
    pop_file = args.population.expanduser().resolve(strict=True)
    out = args.out.expanduser().resolve()
    old = json.loads((trial / "experiment.json").read_text(encoding="utf-8"))
    old_cfg = old["config"]
    require_trial_01(old_cfg)
    if sha256(pop_file) != old["population_sha256"]:
        raise ValueError("Population file differs from Trial 01: use identical STATPOP inputs")
    baseline_source = Path(old["baseline_source"]).resolve(strict=True)
    if old.get("baseline_sha256") and sha256(baseline_source) != old["baseline_sha256"]:
        raise ValueError("Trial 01 baseline cache changed")
    pop = read_population(pop_file)
    origins = pd.read_csv(spatial / "origins.csv", dtype={"hex_id": str})
    jobs = pd.read_csv(spatial / "jobs.csv", dtype={"job_id": str})
    if set(origins["hex_id"]) != set(pop["hex_id"]):
        raise ValueError("STATPOP population and spatial origins have different IDs")
    times = [TIME]
    provenance(baseline_source, old_cfg, times)
    old_baseline = read_accessibility(baseline_source, pop, times, old_cfg)
    base_scenario = baseline_source.parent.parent if baseline_source.parent.name == "accessibility" else baseline_source.parent
    zero_car, parking_car = car_accessibility(base_scenario, origins, jobs, old_cfg, validate_old_parking=True)
    assert_baseline_routing_match(old_baseline, parking_car)
    baseline = replace_car(old_baseline, zero_car)
    reference = baseline_benchmarks(baseline, times)
    new_cfg = copy.deepcopy(old_cfg)
    new_cfg["CAR_PARKING_PENALTY_S"] = 0.0
    old_ref = old["reference"]
    base_score = score(baseline, reference, new_cfg, None)
    old_base_score = old["baseline_score"]
    entries = read_trial_successes(trial, limit=args.limit)
    if not entries:
        raise ValueError("No previously successful Trial 01 scenarios")
    manifest = {
        "version": VERSION,
        "trial": str(trial),
        "trial_scores_sha256": sha256(trial / "fitness_scores.csv.gz"),
        "trial_settings": old_cfg,
        "new_settings": new_cfg,
        "baseline_source": str(baseline_source),
        "baseline_source_sha256": sha256(baseline_source),
        "population": str(pop_file), "population_sha256": sha256(pop_file),
        "spatial": str(spatial),
        "spatial_origins_sha256": sha256(spatial / "origins.csv"),
        "spatial_jobs_sha256": sha256(spatial / "jobs.csv"),
        "baseline_old_car_reference_jobs": old_ref[JOBS],
        "baseline_new_car_reference_jobs": reference[JOBS],
        "common_low_threshold_jobs": reference[JOBS]*0.5,
        "baseline_score": base_score,
        "old_baseline_score": old_base_score,
        "baseline_car_mean_jobs_with_parking": statpop_car_jobs(old_baseline),
        "baseline_car_mean_jobs_without_parking": statpop_car_jobs(baseline),
        "n_scenarios": len(entries),
        "chunk_size": args.chunk_size,
        "n_chunks": math.ceil(len(entries)/args.chunk_size),
        "script_sha256": sha256(Path(__file__)),
    }
    if out.exists() and (out / "experiment.json").exists():
        previous = json.loads((out / "experiment.json").read_text(encoding="utf-8"))
        if previous != manifest:
            raise RuntimeError(f"Experiment inputs or script changed; use a new --out: {out}")
        if not (out / "manifest.csv").is_file():
            raise RuntimeError("Existing experiment lacks manifest.csv")
    else:
        if out.exists() and any(out.iterdir()):
            raise RuntimeError(f"Nonempty --out without experiment.json: {out}")
        out.mkdir(parents=True, exist_ok=True)
        atomic_json(out / "experiment.json", manifest)
        pd.DataFrame(entries).to_csv(out / "manifest.csv", index=False)
        save_csv_gzip_atomic(zero_car, out / "baseline_car_no_parking.csv.gz")
    print(f"Verified original 300s car baseline exactly before switching to 0s", flush=True)
    print(f"Car mean jobs: old={manifest['baseline_car_mean_jobs_with_parking']:.2f} "
          f"new={manifest['baseline_car_mean_jobs_without_parking']:.2f}", flush=True)
    print(f"Common low threshold: old={float(old_ref[JOBS])*0.5:.2f} "
          f"new={manifest['common_low_threshold_jobs']:.2f}", flush=True)
    print(f"Baseline F: old={old_base_score['raw_fitness']:.8f} "
          f"new={base_score['raw_fitness']:.8f}", flush=True)
    return manifest, entries, pop, origins, jobs


def _evaluate_chunk(chunk_index: int, entries: list[dict]):
    if _WORKER is None:
        raise RuntimeError("Worker context not initialized")
    cfg, new_cfg, population, origins, jobs, reference, baseline_score, out = _WORKER
    done = out / "chunks" / f"chunk_{chunk_index:05d}.json"
    car_file = out / "car_accessibility" / f"chunk_{chunk_index:05d}.csv.gz"
    if done.is_file() and car_file.is_file():
        return chunk_index, "reused", 0, 0
    completed, errors, cars = [], [], []
    for entry in entries:
        source = Path(entry["source"])
        try:
            provenance(source, cfg, [TIME])
            cached = read_accessibility(source, population, [TIME], cfg)
            scenario = source.parent.parent if source.parent.name == "accessibility" else source.parent
            car, _ = car_accessibility(scenario, origins, jobs, cfg)
            joined = replace_car(cached, car)
            result = score(joined, reference, new_cfg, None)
            components = {m: _get_component(result, m) for m in MODES}
            oldcar = cached.loc[cached["mode"] == "car"]
            newcar = joined.loc[joined["mode"] == "car"]
            record = {
                "key": entry["key"], "scenario": entry["scenario"],
                "source": entry["source"],
                "old_fitness": float(entry["raw_fitness"]),
                "fitness": result["raw_fitness"],
                "delta_fitness": result["raw_fitness"]-baseline_score["raw_fitness"],
                "car_score": components["car"]["score"],
                "bike_score": components["bike"]["score"],
                "pt_score": components["pt"]["score"],
                "car_sufficient_share": components["car"]["sufficient_population_share"],
                "bike_sufficient_share": components["bike"]["sufficient_population_share"],
                "pt_sufficient_share": components["pt"]["sufficient_population_share"],
                "car_jobs_mean_no_parking": statpop_car_jobs(joined),
                "car_jobs_mean_old_parking": statpop_car_jobs(cached),
            }
            completed.append(record)
            c = car[["hex_id", "attached", JOBS]].copy()
            c.insert(0, "key", entry["key"])
            cars.append(c)
        except Exception as exc:
            errors.append({"key": entry["key"], "scenario": entry["scenario"],
                           "source": entry["source"], "error": f"{type(exc).__name__}: {exc}"})
    car_all = pd.concat(cars, ignore_index=True) if cars else pd.DataFrame(columns=["key", "hex_id", "attached", JOBS])
    save_csv_gzip_atomic(car_all, car_file)
    atomic_json(done, {"chunk_index": chunk_index, "completed": completed,
                       "errors": errors, "expected": len(entries)})
    return chunk_index, "calculated", len(completed), len(errors)


def _finalize(out: Path, manifest: dict):
    completed, errors = [], []
    for i in range(manifest["n_chunks"]):
        target = out / "chunks" / f"chunk_{i:05d}.json"
        car_file = out / "car_accessibility" / f"chunk_{i:05d}.csv.gz"
        if not target.is_file() or not car_file.is_file():
            raise RuntimeError(f"Missing chunk {i} (results or car accessibility cache)")
        content = json.loads(target.read_text(encoding="utf-8"))
        completed.extend(content["completed"])
        errors.extend(content["errors"])
    if len(completed) + len(errors) != manifest["n_scenarios"]:
        raise RuntimeError("Scored + failed scenario count does not equal manifest")
    scores = pd.DataFrame(completed)
    if not scores.empty:
        scores.sort_values("fitness", ascending=False, inplace=True)
    save_csv_gzip_atomic(scores, out / "fitness_scores.csv.gz")
    pd.DataFrame(errors, columns=["key", "scenario", "source", "error"]).to_csv(out / "failures.csv", index=False)
    baseline = manifest["baseline_score"]["raw_fitness"]
    old_baseline = manifest["old_baseline_score"]["raw_fitness"]
    summary = {
        "old_baseline_fitness": old_baseline,
        "new_baseline_fitness": baseline,
        "old_low_threshold_jobs": 0.5*float(manifest["baseline_old_car_reference_jobs"]),
        "new_low_threshold_jobs": manifest["common_low_threshold_jobs"],
        "old_car_mean_jobs": manifest["baseline_car_mean_jobs_with_parking"],
        "new_car_mean_jobs": manifest["baseline_car_mean_jobs_without_parking"],
        "n_scored": len(completed), "n_failed": len(errors),
    }
    lines = ["# Experiment C — zero car parking, recalibrated common low threshold", "",
             f"Compared with Trial 01 on {len(completed):,} common MATSim scenarios; "
             f"{len(errors)} failures.", "",
             f"Population-weighted mean baseline car jobs: "
             f"{summary['old_car_mean_jobs']:,.1f} (5-min parking) → "
             f"{summary['new_car_mean_jobs']:,.1f} (no parking)",
             f"Common low threshold: {summary['old_low_threshold_jobs']:,.1f} → "
             f"{summary['new_low_threshold_jobs']:,.1f} jobs", "",
             f"Trial 01 baseline fitness: {old_baseline:.8f}",
             f"New baseline fitness: {baseline:.8f}"]
    if not scores.empty:
        best = scores.iloc[0].to_dict()
        oldbest = max(completed, key=lambda x:x["old_fitness"])
        summary["best_new"] = best
        summary["old_trial_winner_on_new_scale"] = next(
            x for x in completed if x["key"] == oldbest["key"])
        summary["new_n_improving"] = int((scores["delta_fitness"] > 0).sum())
        summary["old_n_improving_on_common_sample"] = int((scores["old_fitness"] > old_baseline).sum())
        lines += ["", f"Best new scenario: **{best['scenario']}**",
                  f"F={best['fitness']:.8f}, ΔF={best['delta_fitness']:+.8f}",
                  f"Car={best['car_score']:.6f}; bike={best['bike_score']:.6f}; "
                  f"PT={best['pt_score']:.6f}",
                  f"New improved scenarios: {summary['new_n_improving']}/{len(completed)}", "",
                  "Interpret all changes against the corresponding baseline, because "
                  "the sufficiency standard changed together with parking time."]
    atomic_json(out / "comparison.json", summary)
    (out / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)
    if errors:
        print(f"WARNING: {len(errors)} failures; see {out/'failures.csv'}", flush=True)
    print(f"Output: {out}", flush=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trial", type=Path, required=True, help="Trial 01 result directory")
    p.add_argument("--population", type=Path, required=True)
    p.add_argument("--spatial", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    p.add_argument("--chunk-size", type=int, default=20)
    p.add_argument("--limit", type=int, default=0, help="Smoke test: process first N successful scenarios")
    args = p.parse_args(argv)
    if args.workers < 1 or args.chunk_size < 1 or args.limit < 0:
        p.error("workers/chunk-size must be >=1; limit must be >=0")
    manifest, entries, pop, origins, jobs = prepare(args)
    out = args.out.expanduser().resolve()
    global _WORKER
    _WORKER = (manifest["trial_settings"], manifest["new_settings"], pop,
               origins, jobs, {JOBS: manifest["baseline_new_car_reference_jobs"]},
               manifest["baseline_score"], out)
    chunks = [(i, entries[i*args.chunk_size:(i+1)*args.chunk_size])
              for i in range(manifest["n_chunks"])]
    print(f"Routing no-parking car accessibility: {len(entries)} scenarios "
          f"in {len(chunks)} chunks, {args.workers} workers", flush=True)
    if args.workers == 1:
        for ix, items in chunks:
            print("Chunk", *_evaluate_chunk(ix, items), flush=True)
    else:
        if os.name != "posix":
            raise RuntimeError("Parallel mode requires Linux 'fork'")
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("fork")) as pool:
            futures = {pool.submit(_evaluate_chunk, ix, items): ix for ix, items in chunks}
            for k, future in enumerate(as_completed(futures), start=1):
                print(f"[{k}/{len(chunks)}] chunk", *future.result(), flush=True)
    _finalize(out, manifest)


if __name__ == "__main__":
    main()

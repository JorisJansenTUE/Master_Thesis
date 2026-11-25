#!/usr/bin/env python3
"""GA ranking from frozen GNN *saved predictions* and original MATSim files.

Deliberately separate from the historical-test-only evaluate_fitness.py. No MATSim
runs or GNN inference are performed here. For each GA scenario, we reconstruct
observed and predicted car TT with the existing paired-time implementation, and
calculate the identical 20-minute, parking-5, bike/PT fitness/feasibility.

Requires predictions made with the ORIGINAL frozen GNN, matching the full prepared
GA inputs and saved in EVALUATION/predictions/<ga_id>.npz. All other inputs are
read-only; one output directory is produced for each evaluation group.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import hashlib
import json
import math
import os
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from gnn_surrogate.evaluation import evaluate_fitness as ev

GROUPS = ("ga_spatial_no_warm", "ga_spatial_warm", "ga_nonspatial_old")


def jread(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def writejson(path: Path, item):
    ev._write_json(path, item)


def hash_identity(baseline: Path):
    meta = jread(baseline / "scenario.json")
    return (meta.get("baseline_hashes"), {
        "network": sha(baseline / "network.xml.gz"),
        "schedule": sha(baseline / "transit_schedule.xml.gz"),
    })


def load_manifest(prepared: Path, group: str):
    manifest_file = prepared / "preparation_metadata" / "source_manifest.csv"
    if not manifest_file.is_file():
        raise FileNotFoundError(f"Missing source-aware expanded GNN manifest: {manifest_file}")
    with manifest_file.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    rows = [r for r in rows if r.get("group") == group]
    if not rows:
        raise ValueError(f"No scenarios for {group} in {manifest_file}")
    names = [r["scenario"] for r in rows]
    if len(set(names)) != len(names):
        raise ValueError("Duplicate GA scenario IDs in manifest")
    for row in rows:
        if not row["scenario"].startswith("ga_"):
            raise ValueError(f"Unexpected non-GA scenario {row['scenario']}")
        if row.get("split") != ("test" if group == "ga_spatial_no_warm" else row.get("split")):
            raise ValueError(f"No-warm test scenario assigned to another split: {row['scenario']}")
        if not Path(row["source"]).is_dir():
            raise FileNotFoundError(f"Missing source MATSim scenario: {row['source']}")
    splits = jread(prepared / "splits.json")
    for row in rows:
        membership = [s for s in ("train", "val", "test") if row["scenario"] in splits[s]]
        if membership != [row["split"]]:
            raise ValueError(f"Split mismatch for {row['scenario']}: {membership}, manifest {row['split']}")
    return sorted(rows, key=lambda row: row["scenario"]), manifest_file


def check_fixed_graph(old: Path, prepared: Path):
    oldp, newp = old / "fixed_provenance.json", prepared / "fixed_provenance.json"
    if not oldp.is_file() or not newp.is_file():
        raise FileNotFoundError("Both original and expanded GNN fixed_provenance.json are required")
    a, b = jread(oldp), jread(newp)
    keys = ("baseline_network_sha256", "street_segments_sha256", "baseline_flows_sha256",
            "baseline_linkstats_sha256", "travel_time_hour", "target_representation",
            "baseline_flow_window", "static_features")
    for key in keys:
        if a.get(key) != b.get(key):
            raise ValueError(f"Original/expanded GNN provenance differs in {key}: {a.get(key)} != {b.get(key)}")
    x = pd.read_csv(old / "gnn_static" / "dual_nodes.csv.gz")
    y = pd.read_csv(prepared / "gnn_static" / "dual_nodes.csv.gz")
    pd.testing.assert_frame_equal(x, y, check_dtype=False, check_exact=True)
    return {"fixed_provenance": {k: a.get(k) for k in keys}, "nodes": len(x)}


def check_evaluation(evaluation: Path, selected: list[dict]):
    summary_file = evaluation / "summary.json"
    metrics_file = evaluation / "scenario_metrics.csv"
    if not summary_file.is_file() or not metrics_file.is_file():
        raise FileNotFoundError(
            f"Missing external GA inference outputs in {evaluation}. The ORIGINAL frozen GNN "
            "must first produce summary.json, scenario_metrics.csv and predictions/ga_*.npz. "
            "The old 05 original-test predictions do NOT cover these GA scenarios."
        )
    summary = jread(summary_file)
    if summary.get("target_type") != "road_congestion_delay_change" or summary.get("hour") != 8:
        raise ValueError("Wrong delay-target type or target hour in saved predictions")
    df = pd.read_csv(metrics_file, usecols=["scenario"])
    all_prediction_ids = list(df["scenario"].astype(str))
    if len(all_prediction_ids) != len(set(all_prediction_ids)):
        raise ValueError("Duplicate scenarios in prediction manifest")
    if summary.get("scenarios") != len(all_prediction_ids):
        raise ValueError("Prediction summary scenario count differs from scenario_metrics.csv")
    missing = sorted({r["scenario"] for r in selected} - set(all_prediction_ids))
    if missing:
        raise ValueError(f"Only {len(selected)-len(missing)}/{len(selected)} scenarios have predictions. "
                         f"Missing examples: {missing[:10]}")
    missing_files = [r["scenario"] for r in selected if not (evaluation / "predictions" / (r["scenario"]+".npz")).is_file()]
    if missing_files:
        raise FileNotFoundError(f"Missing prediction npz for {len(missing_files)} scenarios: {missing_files[:8]}")
    return summary


def load_generations(root: Path):
    """Read generation of *creation*, not full population membership, for cohorts."""
    p = root / "optimization_state.json"
    if not p.is_file():
        return {}, {}, None
    state = jread(p)
    archive = state.get("archive", {})
    if not isinstance(archive, dict):
        raise ValueError(f"Unexpected archive format in {p}")
    generation, stored = {}, {}
    for item in archive.values():
        sid = item.get("scenario")
        if not isinstance(sid, str):
            continue
        g = item.get("created_generation")
        if isinstance(g, int):
            generation[sid] = g
        fit = item.get("fitness")
        if fit is not None and isinstance(fit, (float,int)) and math.isfinite(fit):
            stored[sid] = float(fit)
    return generation, stored, p


def process_one(task):
    sid, source, old_gnn, evaluation, spatial, reference_file, baseline_hashes, completed_hashes, floor = task
    started = perf_counter()
    scenario_dir = Path(source)
    if scenario_dir.name != sid or not scenario_dir.is_dir():
        raise ValueError(f"Scenario source/ID mismatch: {sid} / {source}")
    meta = jread(scenario_dir / "scenario.json")
    if meta.get("id") != sid:
        raise ValueError(f"scenario.json id differs from folder name: {sid}")
    if meta.get("baseline_hashes") not in (baseline_hashes, completed_hashes):
        raise ValueError(f"Unexpected baseline hashes in GA scenario {sid}")
    run = jread(scenario_dir / "run_status.json")
    if run.get("state") not in ("completed", "complete") or run.get("postprocessing", {}).get("state") != "completed":
        raise ValueError(f"GA scenario not completed with postprocessing: {sid}")
    if not (scenario_dir / "targets/linkstats.txt.gz").is_file():
        raise FileNotFoundError(f"GA linkstats missing: {sid}")
    nodes = ev._graph_nodes(str(Path(old_gnn) / "gnn_static/dual_nodes.csv.gz"))
    prediction = ev._prediction_data(evaluation, sid)
    reference = ev.SufficiencyReference.load(Path(reference_file))
    baseline_mask = ev._load_or_create_baseline_mask(
        ev._baseline_mask_path(Path(reference_file)), pd.DataFrame(), Path(reference_file),
        "baseline_00000", reference, tuple(ev.settings.THRESHOLDS_MIN), is_baseline=False,
    )
    actual = ev.LinkTravelTimes.from_linkstats(
        scenario_dir / "targets/linkstats.txt.gz", hour_start=ev.settings.HOUR_START,
        statistic=ev.settings.LINKSTATS_STATISTIC,
    )
    reference_times, gnn_times, diagnostics = ev._paired_times(
        scenario_dir, nodes, prediction, actual, floor=floor,
    )
    true_score, true_summary = ev._calculate_and_score(
        scenario_dir, Path(spatial), reference_times, reference, baseline_mask,
    )
    gnn_score, gnn_summary = ev._calculate_and_score(
        scenario_dir, Path(spatial), gnn_times, reference, baseline_mask,
    )
    if (true_score["n_eligible_origins"] != gnn_score["n_eligible_origins"] or
            true_score["car_constraint"]["minimum_share"] != gnn_score["car_constraint"]["minimum_share"]):
        raise AssertionError("Inconsistent eligible origins or car constraint")
    comps = []
    for method, score in (("matsim", true_score), ("gnn", gnn_score)):
        for comp in score["components"]:
            comps.append({"scenario": sid, "source": method, **comp})
    return {
        "scenario": sid,
        "source_dir": str(scenario_dir.resolve()),
        "strategy": meta.get("strategy"),
        "budget_used_m": meta.get("budget_used_m"),
        "intervention_count": meta.get("intervention_count"),
        "fitness_matsim": true_score["fitness"],
        "fitness_gnn": gnn_score["fitness"],
        "fitness_error": gnn_score["fitness"] - true_score["fitness"],
        "raw_fitness_matsim": true_score["raw_fitness"],
        "raw_fitness_gnn": gnn_score["raw_fitness"],
        "raw_fitness_error": gnn_score["raw_fitness"] - true_score["raw_fitness"],
        "weighted_shortfall_matsim": true_score["weighted_shortfall"],
        "weighted_shortfall_gnn": gnn_score["weighted_shortfall"],
        "car_share_matsim": true_score["car_constraint"]["scenario_share"],
        "car_share_gnn": gnn_score["car_constraint"]["scenario_share"],
        "car_baseline_share": true_score["car_constraint"]["baseline_share"],
        "car_minimum_share": true_score["car_constraint"]["minimum_share"],
        "car_violation_pp_matsim": true_score["car_constraint"]["violation_pp"],
        "car_violation_pp_gnn": gnn_score["car_constraint"]["violation_pp"],
        "feasible_matsim": bool(true_score["feasible"]),
        "feasible_gnn": bool(gnn_score["feasible"]),
        "n_eligible_origins": true_score["n_eligible_origins"],
        "time_s": round(perf_counter() - started,3),
        "diagnostics": diagnostics,
        "components": comps,
        "sufficiency_matsim": true_summary,
        "sufficiency_gnn": gnn_summary,
    }


def make_generation_metrics(rows, generations, output):
    per_generation, topk = [], []
    cohorts = {}
    for row in rows:
        gen = generations.get(row["scenario"])
        if gen is None:
            continue
        cohorts.setdefault(gen, []).append(row)
    for gen, subset in sorted(cohorts.items()):
        if len(subset) < 2:
            continue
        m = ev._ranking(subset, (1,2,5,10,20,40,80))
        per_generation.append({
            "generation": gen, "n_new_evaluated": len(subset),
            "fitness_spearman": m["fitness_spearman"],
            "fitness_mae": m["fitness_mae"],
            "car_share_mae_pp": m["car_share_mae_pp"],
            "feasibility_recall": m["feasibility"]["recall"],
            "best_matsim_fitness": m["best_matsim_fitness"],
        })
        for k,v in m["top_k"].items():
            topk.append({"generation": gen, "n_new_evaluated": len(subset), "requested_k": int(k), **v})
    if per_generation:
        ev._write_csv(output / "generation_cohort_metrics.csv", per_generation)
        ev._write_csv(output / "generation_cohort_topk.csv", topk)
    return len(per_generation)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--old-gnn", type=Path, required=True)
    p.add_argument("--prepared-gnn", type=Path, required=True)
    p.add_argument("--evaluation", type=Path, required=True)
    p.add_argument("--historical", type=Path, required=True)
    p.add_argument("--spatial", type=Path, required=True)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--group", choices=GROUPS, default="ga_spatial_no_warm")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--floor", choices=("positive", "freeflow"), default="freeflow")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--plots", action="store_true")
    a = p.parse_args()
    if a.workers < 1 or a.limit < 0:
        p.error("--workers must be >=1, --limit must be >=0")
    if ev.FITNESS_EVALUATION_VERSION != "squared_sustainable_car_constraint_v1":
        raise RuntimeError("Unexpected historical screening evaluator version")
    ev._check_settings(a.reference)
    check = check_fixed_graph(a.old_gnn, a.prepared_gnn)
    selected, manifest = load_manifest(a.prepared_gnn, a.group)
    if a.limit:
        selected = selected[:a.limit]
    summary = check_evaluation(a.evaluation, selected)
    baseline = a.historical / "baseline_00000"
    sources, completed = hash_identity(baseline)
    if not sources:
        raise ValueError("Missing historical baseline hashes")
    grouped_dirs = {Path(r["source"]).parent for r in selected}
    if len(grouped_dirs) != 1:
        raise ValueError(f"Expected scenarios from one GA run root, got {grouped_dirs}")
    ga_root = next(iter(grouped_dirs))
    gens, ga_fitness, ga_state_file = load_generations(ga_root)

    if a.out.exists() and any(a.out.iterdir()) and not a.resume:
        raise FileExistsError(f"Existing output {a.out}; use new path or --resume")
    a.out.mkdir(parents=True, exist_ok=True)
    config = {
        "group": a.group, "scenario_ids": [r["scenario"] for r in selected],
        "frozen_gnn_dir": str(a.old_gnn.resolve()),
        "prepared_gnn_dir": str(a.prepared_gnn.resolve()),
        "evaluation_dir": str(a.evaluation.resolve()),
        "prediction_summary_sha256": sha(a.evaluation / "summary.json"),
        "manifest_sha256": sha(manifest),
        "splits_sha256": sha(a.prepared_gnn / "splits.json"),
        "reference_sha256": sha(a.reference),
        "attachment_sha256": sha(ev._baseline_mask_path(a.reference)),
        "screening_evaluator_sha256": sha(Path(ev.__file__)),
        "floor": a.floor, "fitness_version": ev.FITNESS_EVALUATION_VERSION,
        "baseline_source_hashes": sources, "completed_baseline_file_hashes": completed,
        "graph_check": check,
        "ga_state_sha256": sha(ga_state_file) if ga_state_file else None,
        "target_type": summary.get("target_type"), "hour": summary.get("hour"),
        "evaluation_split_label": summary.get("split"),
        "limit": a.limit,
    }
    confpath = a.out / "evaluation_config.json"
    if a.resume:
        if not confpath.is_file() or jread(confpath) != config:
            raise ValueError("--resume requires identical original output and configuration")
    else:
        writejson(confpath, config)
    perdir = a.out / "per_scenario"
    perdir.mkdir(exist_ok=True)
    old = {}
    tasks = []
    for row in selected:
        sid = row["scenario"]
        cp = perdir / (sid + ".json")
        if a.resume and cp.is_file():
            saved = jread(cp)
            if saved.get("scenario") != sid:
                raise ValueError(f"Cached checkpoint mismatch: {cp}")
            old[sid] = saved
        else:
            tasks.append((sid, row["source"], str(a.old_gnn), str(a.evaluation),
                          str(a.spatial), str(a.reference), sources, completed, a.floor))
    print(f"GROUP={a.group} total={len(selected)} cached={len(old)} todo={len(tasks)} ",
          f"workers={a.workers} prediction_split_label={summary.get('split')}",flush=True)
    failures = []
    t0=perf_counter()
    if a.workers == 1:
        for task in tasks:
            try:
                saved = process_one(task)
                writejson(perdir / (saved["scenario"]+".json"), saved)
                old[saved["scenario"]] = saved
            except Exception as exc:
                failures.append({"scenario":task[0],"error":repr(exc)})
                print(f"FAILED {task[0]}: {exc}",flush=True)
    else:
        with ProcessPoolExecutor(max_workers=a.workers) as pool:
            futures={pool.submit(process_one,t):t[0] for t in tasks}
            for count, f in enumerate(as_completed(futures),1):
                sid=futures[f]
                try:
                    saved=f.result()
                    writejson(perdir/(sid+".json"), saved)
                    old[sid]=saved
                except Exception as exc:
                    failures.append({"scenario":sid,"error":repr(exc)})
                    print(f"FAILED {sid}: {exc}",flush=True)
                if count%25==0 or count==len(tasks):
                    print(f"Processed {count}/{len(tasks)}; total completed={len(old)}",flush=True)
    if failures:
        ev._write_csv(a.out / "failures.csv", failures)
        raise RuntimeError(f"{len(failures)} failures. See {a.out/'failures.csv'}; set RESUME=1 to retry")
    (a.out / "failures.csv").unlink(missing_ok=True)
    ordered = [old[r["scenario"]] for r in selected]
    rows, comps = [], []
    for payload in ordered:
        sid=payload["scenario"]
        r={k:v for k,v in payload.items() if k not in ("diagnostics","components","sufficiency_matsim","sufficiency_gnn")}
        r["created_generation"] = gens.get(sid)
        r["ga_state_fitness"] = ga_fitness.get(sid)
        if sid in ga_fitness:
            r["fitness_delta_vs_ga_state"] = r["fitness_matsim"]-ga_fitness[sid]
        rows.append(r)
        comps.extend(payload["components"])
    ev._write_csv(a.out/"fitness_comparison.csv",rows)
    if comps:
        ev._write_csv(a.out/"mode_comparison.csv",comps)
    metrics=ev._ranking(rows, (1, 2, 5, 10, 20, 40, 50, 80, 100, 160, 320, 640, 1280))
    metrics["group"] = a.group
    metrics["elapsed_s"] = perf_counter()-t0
    metrics["created_generation_coverage"] = sum(r["created_generation"] is not None for r in rows)
    state_diff=[r["fitness_delta_vs_ga_state"] for r in rows if "fitness_delta_vs_ga_state" in r]
    metrics["ga_state_fitness_comparison"] = {
        "compared":len(state_diff),
        "mae":float(np.abs(state_diff).mean()) if state_diff else None,
        "max_abs":float(np.abs(state_diff).max()) if state_diff else None,
        "count_abs_over_0.01":sum(abs(x)>0.01 for x in state_diff),
        "note":"MATSim paired car-only reference may differ from original GA fitness",
    }
    ev._write_csv(a.out/"screening_topk.csv",[{"requested_k":int(k),**v} for k,v in metrics["top_k"].items()])
    ev._write_csv(a.out/"feasibility_confusion.csv",[{
        "scenario":r["scenario"],
        "feasible_matsim":r["feasible_matsim"],
        "feasible_gnn":r["feasible_gnn"],
        "car_share_matsim":r["car_share_matsim"],
        "car_share_gnn":r["car_share_gnn"],
        "raw_fitness_matsim":r["raw_fitness_matsim"],
        "raw_fitness_gnn":r["raw_fitness_gnn"],
    } for r in rows])
    metrics["generation_cohorts"] = make_generation_metrics(rows,gens,a.out)
    writejson(a.out/"ranking_metrics.json",metrics)
    if a.plots:
        ev._make_plots(rows,a.out,metrics)
    print(json.dumps({"group":a.group,"scenarios":len(rows),
                      "fitness_spearman":metrics["fitness_spearman"],
                      "fitness_mae":metrics["fitness_mae"],
                      "feasibility":metrics["feasibility"],
                      "global_top_k":metrics["top_k"],
                      "generation_cohorts":metrics["generation_cohorts"]}, indent=2),flush=True)
    print("COMPLETE",a.out,flush=True)


if __name__ == "__main__":
    main()

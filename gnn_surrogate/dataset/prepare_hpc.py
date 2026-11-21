"""Prepare the Locarno GNN dataset from completed MATSim scenarios only.

Run from the Master_Thesis repository root, normally via hpc/01_prepare_dataset.slurm.
The source simulation directory is read-only. Outputs contain compact tensors,
not copies of simulation networks or routes.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import traceback

import numpy as np

from .build_splits import build_split_manifest
from .dataset import ScenarioDataset, load_splits
from preprocessing.utils import PROJECT_ROOT

BASELINE_NAME = "baseline_00000"
FLOW_REL = Path("targets/output_detailed_link_statistics.csv.gz")
LINKSTATS_REL = Path("targets/linkstats.txt.gz")
REQUIRED_SCENARIO = ("scenario.json", "interventions.json", "network.xml.gz", "link_mapping.csv.gz")
STATIC_FEATURES = (
    "length_m", "baseline_car_capacity", "baseline_car_freespeed_m_s",
    "mid_x", "mid_y", "baseline_active", "baseline_allows_car",
    "baseline_allows_bike", "baseline_allows_bus",
)


def _json(path: Path):
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def _save(path: Path, value: dict):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _completion_reason(scenario: Path, baseline_hashes: dict | None = None) -> str | None:
    """Return a reason to exclude this scenario, or None if complete."""
    for name in REQUIRED_SCENARIO:
        if not (scenario / name).is_file():
            return f"missing {name}"
    status_file = scenario / "run_status.json"
    if not status_file.is_file():
        return "no run_status.json (not confirmed completed)"
    try:
        status = _json(status_file)
        if status.get("state") not in ("completed", "complete"):
            return f"simulation state={status.get('state')!r}"
        if status.get("postprocessing", {}).get("state") != "completed":
            return "postprocessing not completed"
        metadata = _json(scenario / "scenario.json")
        if metadata.get("id") != scenario.name:
            return "scenario.json ID does not match directory"
        interventions = _json(scenario / "interventions.json")
        if not isinstance(interventions, list):
            return "interventions.json is not a list"
        if scenario.name != BASELINE_NAME and (metadata.get("strategy") == "baseline" or not interventions):
            return "not a nonempty intervention scenario"
        if scenario.name == BASELINE_NAME and (metadata.get("strategy") != "baseline" or interventions):
            return "baseline must have strategy=baseline and no interventions"
        if baseline_hashes:
            other = metadata.get("baseline_hashes")
            if other != baseline_hashes:
                return "scenario was generated from different baseline hashes"
        if not (scenario / FLOW_REL).is_file() or (scenario / FLOW_REL).stat().st_size == 0:
            return "missing/empty compacted link statistics"
        if not (scenario / LINKSTATS_REL).is_file() or (scenario / LINKSTATS_REL).stat().st_size == 0:
            return "missing/empty linkstats travel-time targets"
        summary_file = scenario / "targets/target_summary.json"
        if not summary_file.is_file():
            return "missing target_summary.json (compaction not verified)"
        summary = _json(summary_file)
        if summary.get("flow_file") != str(FLOW_REL):
            return "target_summary.json refers to a different flow file"
        if summary.get("flow", {}).get("rows", 0) <= 0:
            return "target summary contains no link-hour rows"
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        return f"invalid metadata: {exc}"
    return None


def _run(module: str, *arguments: object, log: Path | None = None) -> None:
    command = [sys.executable, "-m", module, *(str(x) for x in arguments)]
    env = os.environ.copy()
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[name] = "1"
    if log is None:
        subprocess.run(command, check=True, cwd=PROJECT_ROOT, env=env)
    else:
        with log.open("a", encoding="utf-8") as handle:
            handle.write("COMMAND: " + " ".join(command) + "\n")
            handle.flush()
            subprocess.run(command, check=True, cwd=PROJECT_ROOT, env=env,
                           stdout=handle, stderr=subprocess.STDOUT)


def _fixed_inputs(source: Path, output: Path, baseline: Path, hour: int) -> tuple[int, dict]:
    graph, inputs = output / "gnn_static", output / "gnn_inputs"
    provenance_path = output / "fixed_provenance.json"
    provenance = {
        "baseline_network_sha256": _sha256(baseline / "network.xml.gz"),
        "street_segments_sha256": _sha256(source / "street_segments.csv.gz"),
        "baseline_flows_sha256": _sha256(baseline / FLOW_REL),
        "baseline_linkstats_sha256": _sha256(baseline / LINKSTATS_REL),
        "travel_time_hour": hour,
        "static_features": list(STATIC_FEATURES),
        "target_representation": "road_travel_time_delta_v2",
        "baseline_flow_window": "target_hour",
    }
    if provenance_path.exists():
        if _json(provenance_path) != provenance:
            raise ValueError("Fixed preprocessing was made from different baseline/features; use a NEW output directory")
        if not all((output / p).is_file() for p in (
            "gnn_static/dual_nodes.csv.gz", "gnn_static/dual_edge_index.npy",
            "gnn_static/baseline_tt.npz",
            "gnn_inputs/fixed_inputs.npz", "gnn_inputs/input_metadata.json"
        )):
            raise ValueError("Fixed preprocessing artifacts are incomplete; use a NEW output directory")
        print("Reusing fixed graph and inputs (baseline hashes match)", flush=True)
    else:
        if graph.exists() or inputs.exists():
            raise ValueError("Fixed output folders exist without provenance; choose a NEW output directory")
        _run("gnn_surrogate.preprocessing.graph.build_graph",
             "--segments", source / "street_segments.csv.gz",
             "--baseline-network", baseline / "network.xml.gz", "--out", graph)
        _run("gnn_surrogate.preprocessing.static.build_inputs",
             "--graph", graph, "--baseline-scenario", baseline,
             "--features", *STATIC_FEATURES, "--hour", hour, "--out", inputs)
        _save(provenance_path, provenance)
    metadata = _json(inputs / "input_metadata.json")
    if metadata.get("static_feature_names") != [*STATIC_FEATURES, "baseline_road_tt_s"] or metadata.get("travel_time_hour") != hour:
        raise ValueError(f"Wrong selected static features: {metadata.get('static_feature_names')}")
    with np.load(inputs / "fixed_inputs.npz", allow_pickle=False) as file:
        n = file["x_static"].shape[0]
        assert file["x_static"].shape == (n, 10)
        assert file["baseline_hourly_raw"].shape == (n, 30, 4)
    return n, provenance


def _validate_pair(output: Path, scenario_id: str, n: int, hour: int) -> None:
    dyn = output / "dynamic" / scenario_id
    tar = output / "targets" / scenario_id
    x = np.load(dyn / "x_dynamic.npy", allow_pickle=False)
    with np.load(tar / "target_delta.npz", allow_pickle=False) as arrays:
        y, mask = arrays["y_delta"], arrays["valid_mask"]
    d = _json(dyn / "dynamic_metadata.json")
    t = _json(tar / "target_metadata.json")
    if (x.shape != (n, 9) or y.shape != (n, 1) or mask.shape != y.shape
            or mask.dtype != np.bool_ or not mask[:, 0].any()
            or not np.isfinite(x).all() or not np.isfinite(y).all()
            or d.get("scenario") != scenario_id or t.get("scenario") != scenario_id
            or len(d.get("feature_names", [])) != 9
            or t.get("channels") != ["road"] or t.get("hour") != hour):
        raise ValueError(f"Invalid TT target/eligibility mask for {scenario_id}")


def _process_one(scenario: Path, output: Path, n: int, hour: int) -> tuple[str, str | None]:
    sid = scenario.name
    dyn_root, tar_root = output / "dynamic", output / "targets"
    dyn, tar = dyn_root / sid, tar_root / sid
    log = output / "logs" / f"{sid}.log"
    try:
        if dyn.exists() or tar.exists():
            if not dyn.is_dir() or not tar.is_dir():
                raise RuntimeError("Only one of dynamic and target outputs exists; manual cleanup required")
            _validate_pair(output, sid, n, hour)
            return sid, None

        tmp_dyn = Path(tempfile.mkdtemp(prefix=f".{sid}.", dir=dyn_root))
        tmp_tar = Path(tempfile.mkdtemp(prefix=f".{sid}.", dir=tar_root))
        try:
            _run("gnn_surrogate.preprocessing.dynamic.build_scenario_features",
                 "--graph", output / "gnn_static", "--scenario", scenario,
                 "--out", tmp_dyn, log=log)
            _run("gnn_surrogate.preprocessing.targets.build_targets",
                 "--graph", output / "gnn_static",
                 "--scenario", scenario, "--hour", hour, "--out", tmp_tar, log=log)
            # Validate before publishing either output directory.
            # The temporary directory names differ; metadata still uses the source scenario ID.
            x = np.load(tmp_dyn / "x_dynamic.npy", allow_pickle=False)
            with np.load(tmp_tar / "target_delta.npz", allow_pickle=False) as arrays:
                y, mask = arrays["y_delta"], arrays["valid_mask"]
            if (x.shape != (n, 9) or y.shape != (n, 1) or mask.shape != y.shape
                    or mask.dtype != np.bool_ or not mask[:, 0].any()
                    or not np.isfinite(x).all() or not np.isfinite(y).all()):
                raise ValueError("Wrong TT target or eligibility mask")
            tmp_dyn.rename(dyn)
            tmp_tar.rename(tar)
            _validate_pair(output, sid, n, hour)
        finally:
            shutil.rmtree(tmp_dyn, ignore_errors=True)
            shutil.rmtree(tmp_tar, ignore_errors=True)
        return sid, None
    except Exception as exc:
        with log.open("a", encoding="utf-8") as handle:
            handle.write(f"ERROR: {type(exc).__name__}: {exc}\n")
            handle.write(traceback.format_exc())
        return sid, f"{type(exc).__name__}: {exc}"


def prepare(source: Path, output: Path, *, workers: int = 4, seed: int = 42,
            max_scenarios: int = 0, hour: int = 8) -> dict:
    if not 0 <= hour <= 23:
        raise ValueError("hour must be in 0..23")
    source = source.resolve(strict=True)
    output = output.resolve()
    if not (source / "street_segments.csv.gz").is_file():
        raise FileNotFoundError(source / "street_segments.csv.gz")
    baseline = source / BASELINE_NAME
    reason = _completion_reason(baseline)
    if reason:
        raise RuntimeError(f"Baseline {BASELINE_NAME} is not usable: {reason}")
    baseline_hashes = _json(baseline / "scenario.json").get("baseline_hashes")
    if workers < 1 or max_scenarios < 0:
        raise ValueError("workers must be positive and max_scenarios nonnegative")

    candidates = sorted(p for p in source.iterdir() if p.is_dir() and p.name != BASELINE_NAME
                        and (p.name.startswith(("cluster_", "corridor_", "random_"))
                             or (p / "scenario.json").exists()))
    eligible, excluded = [], {}
    for scenario in candidates:
        reason = _completion_reason(scenario, baseline_hashes)
        if reason:
            excluded[scenario.name] = reason
        else:
            eligible.append(scenario)
    if max_scenarios:
        eligible = eligible[:max_scenarios]
    if len(eligible) < 3:
        raise ValueError(f"Only {len(eligible)} eligible scenarios; need >=3 for splits")
    output.mkdir(parents=True, exist_ok=True)
    for name in ("logs", "dynamic", "targets"):
        (output / name).mkdir(exist_ok=True)
    print(f"MATSim scenarios: candidates={len(candidates)}, eligible={len(eligible)}, "
          f"excluded_failed_or_incomplete={len(excluded)}", flush=True)
    n, provenance = _fixed_inputs(source, output, baseline, hour)
    started = time.monotonic()
    processed, errors = [], {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        tasks = {executor.submit(_process_one, sc, output, n, hour): sc.name for sc in eligible}
        for index, future in enumerate(as_completed(tasks), 1):
            sid, error = future.result()
            if error:
                errors[sid] = error
                print(f"FAILED preprocessing {sid}: {error}", flush=True)
            else:
                processed.append(sid)
            if index % 100 == 0 or index == len(tasks):
                print(f"Progress: {index}/{len(tasks)}; processed={len(processed)}; "
                      f"errors={len(errors)}", flush=True)
    processed.sort()
    report = {
        "source": str(source), "output": str(output),
        "candidate_count": len(candidates), "simulation_eligible_count": len(eligible),
        "simulation_excluded_count": len(excluded), "excluded_simulations": excluded,
        "processed_count": len(processed), "processed_scenarios": processed,
        "preprocessing_error_count": len(errors), "preprocessing_errors": errors,
        "elapsed_seconds": round(time.monotonic() - started, 2),
        "fixed_provenance": provenance,
    }
    _save(output / "preparation_report.json", report)
    if errors:
        raise RuntimeError(f"{len(errors)} eligible simulations failed GNN preprocessing. "
                           f"See {output / 'preparation_report.json'} and logs; not creating splits")
    split_path = output / "splits.json"
    if split_path.is_file():
        splits = load_splits(split_path)
        existing = sorted(splits["train"] + splits["val"] + splits["test"])
        if existing != processed:
            raise ValueError("Existing splits.json covers different scenarios; review before replacing it")
        print("Reusing existing splits.json without reshuffling", flush=True)
    else:
        splits = build_split_manifest(processed, seed=seed)
        _save(split_path, splits)
    print("Split counts:", {name: len(splits[name]) for name in ("train", "val", "test")}, flush=True)

    # One real dataset item verifies the final loader (including road eligibility).
    dataset = ScenarioDataset(output / "gnn_static", output / "gnn_inputs",
                              output / "dynamic", output / "targets",
                              scenario_ids=[splits["train"][0]])
    sample = dataset[0]
    if sample["y_delta"].shape != (n, 1) or sample["valid_mask"].shape != (n, 1):
        raise AssertionError("Final TT dataset target or eligibility mask has wrong shape")
    output_bytes = sum(file.stat().st_size for folder in (output / "dynamic", output / "targets")
                       for file in folder.rglob("*") if file.is_file())
    print(f"PASS: {len(processed)} completed, validated GNN scenarios; "
          f"stored scenario tensors={output_bytes / 1024**3:.2f} GiB; "
          f"elapsed={report['elapsed_seconds'] / 60:.1f} minutes", flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hour", type=int, default=8,
                        help="Peak-hour start (8 = 08:00-09:00)")
    parser.add_argument("--max-scenarios", type=int, default=0,
                        help="Optional pilot cap on eligible scenarios; default 0 processes all")
    args = parser.parse_args()
    prepare(args.dataset, args.out, workers=args.workers,
            seed=args.seed, max_scenarios=args.max_scenarios, hour=args.hour)


if __name__ == "__main__":
    main()

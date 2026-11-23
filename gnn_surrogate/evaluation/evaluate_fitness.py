"""Compare GNN-predicted and MATSim-reference accessibility / GA fitness on test scenarios.

No GNN inference, MATSim execution, GA search or scenario files are modified.
Uses the saved evaluation_test/predictions/*.npz and the EXACT same
accessibility and baseline-aware sufficiency implementation as optimization.fitness.

From repository root:
    python -m gnn_surrogate.evaluation.evaluate_fitness \
      --gnn-dir /.../gnn_training_tt_delay \
      --evaluation /.../gnn_training_tt_delay/runs/tt_delay_001/evaluation_test \
      --dataset /.../training_dataset_v4 \
      --spatial-dir /.../spatial_baseline \
      --reference /.../GA_Optimization/references/sufficiency_reference_15min.json \
      --out /.../gnn_training_tt_delay/runs/tt_delay_001/fitness_test \
      --workers 2 --plots

Important: the saved target/linkstats provides observed times only for
car-eligible road links. For BOTH fitness branches, all non-car network links
(including pt_* connectors and bus-only *_spec links) use the scenario network
free-flow time. A pt_* connector with infinite free speed has ZERO time.
Only car-link times differ between the MATSim and GNN accessibility calls.
This is a deliberately car-only MATSim reference, NOT a validation of observed
PT vehicle congestion or public transport timetables.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from functools import lru_cache
from pathlib import Path
from time import perf_counter
from typing import Any
import zipfile

import numpy as np
import pandas as pd

from accessibility.calculations.evaluate import calculate_accessibility
from accessibility.calculations.sufficiency import SufficiencyReference, assess_sufficiency
from accessibility.inputs.links.linkstats import LinkTravelTimes
from accessibility.inputs.links.network import _load_matsim_network
from gnn_surrogate.preprocessing.targets.target_tensors import _eligible_links
from optimization.fitness import (
    _baseline_mask_path, _load_or_create_baseline_mask, evaluate_scenario,
    score_sufficiency, FITNESS_VERSION,
)
from optimization import inspect_fitness as settings


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _test_ids(splits_path: Path) -> list[str]:
    manifest = _json(splits_path)
    splits = [manifest.get(k) for k in ("train", "val", "test")]
    if any(not isinstance(group, list) or not group or len(group) != len(set(group))
           for group in splits):
        raise ValueError("Invalid train/val/test split lists")
    joined = [sid for group in splits for sid in group]
    if len(joined) != len(set(joined)) or manifest.get("scenario_count") != len(joined):
        raise ValueError("Inconsistent split manifest")
    return splits[2]


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    pd.DataFrame(rows).to_csv(tmp, index=False)
    os.replace(tmp, path)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@lru_cache(maxsize=2)
def _graph_nodes(path: str) -> pd.DataFrame:
    return pd.read_csv(
        path, dtype={"segment_id": str, "from_node": str, "to_node": str,
                     "baseline_link_ids": str},
    )


def _prediction_data(evaluation: str, scenario: str) -> dict[str, np.ndarray]:
    root = Path(evaluation)
    if root.suffix == ".zip":
        with zipfile.ZipFile(root) as zf:
            matches = [k for k in zf.namelist()
                       if k.endswith(f"predictions/{scenario}.npz")]
            if len(matches) != 1:
                raise FileNotFoundError(f"Expected exactly one prediction for {scenario} in {root}")
            import io
            with np.load(io.BytesIO(zf.read(matches[0])), allow_pickle=False) as f:
                return {k: f[k] for k in f.files}
    candidates = (root / "predictions" / f"{scenario}.npz",
                  root / f"{scenario}.npz")
    files = [path for path in candidates if path.is_file()]
    if len(files) != 1:
        raise FileNotFoundError(f"Expected one saved prediction for {scenario} below {root}")
    with np.load(files[0], allow_pickle=False) as f:
        return {k: f[k] for k in f.files}


def _network_free_flow(link) -> float:
    """Physical network free-flow TT; infinite speed means instantaneous travel.

    Do not take these values from the car-only linkstats: PT and bus-special
    links need to be reconstructed from the actual intervention network.
    """
    length, speed = float(link.length_m), float(link.freespeed_m_s)
    if not math.isfinite(length) or length < 0:
        raise ValueError(f"Invalid network length for {link.link_id}: {length}")
    if math.isinf(speed) and speed > 0:
        return 0.0
    if not math.isfinite(speed) or speed <= 0:
        raise ValueError(f"Invalid network free speed for {link.link_id}: {speed}")
    tt = length / speed
    if not math.isfinite(tt) or tt < 0:
        raise ValueError(f"Invalid network free-flow time for {link.link_id}: {tt}")
    return tt


def _paired_times(
    scenario_dir: Path, nodes: pd.DataFrame, prediction: dict,
    actual: LinkTravelTimes, *, floor: str,
    strict_tolerance_s: float = 0.05,
) -> tuple[LinkTravelTimes, LinkTravelTimes, dict]:
    """Build full-network times for reference and prediction without PT leakage.

    MATSim branch: observed car TT + scenario-network non-car free flow.
    GNN branch: predicted car TT + IDENTICAL non-car free flow.

    The GNN target masks, car link IDs and observed target values must match
    the original saved test data; no intervention network is reconstructed.
    """
    n = len(nodes)
    mask = np.asarray(prediction["valid_mask"])
    predicted = np.asarray(prediction["predicted_tt_s"], dtype=np.float64)
    actual_saved = np.asarray(prediction["actual_tt_s"], dtype=np.float64)
    if (mask.shape != (n, 1) or predicted.shape != mask.shape
            or actual_saved.shape != mask.shape or mask.dtype != np.bool_):
        raise ValueError("Incompatible prediction arrays or graph node count")
    if not np.isfinite(predicted[mask]).all() or not np.isfinite(actual_saved[mask]).all():
        raise ValueError("Nonfinite GNN/actual times on eligible road links")

    eligible = _eligible_links(nodes, scenario_dir, hour=actual.hour_start)
    ids = eligible["node_id"].to_numpy(dtype=np.int64)
    expected = np.zeros(n, dtype=bool)
    expected[ids] = True
    if not np.array_equal(mask[:, 0], expected):
        raise ValueError("Saved prediction mask differs from scenario car-link mapping")
    observed = eligible["tt_s"].to_numpy(dtype=np.float64)
    if not np.allclose(actual_saved[ids, 0], observed,
                       atol=strict_tolerance_s, rtol=1e-5):
        delta = np.max(np.abs(actual_saved[ids, 0] - observed))
        raise ValueError(f"MATSim linkstats / stored GNN actual_tt mismatch: max {delta:.6g}s")

    raw = _load_matsim_network(scenario_dir / "network.xml.gz")
    car_link_set = set(eligible["link_id"].astype(str))
    network_car = {link.link_id for link in raw.links if "car" in link.modes}
    if network_car != car_link_set:
        raise ValueError(
            "Car-link mapping incomplete: "
            f"missing={sorted(network_car-car_link_set)[:5]}, "
            f"extra={sorted(car_link_set-network_car)[:5]}"
        )
    if any(lid.endswith("_spec") for lid in car_link_set):
        raise ValueError("A bus-only *_spec link incorrectly allows car")

    # Assemble the full-network input independently of car-only linkstats.
    true_times: dict[str, float] = {}
    pred_times: dict[str, float] = {}
    full_freeflow: dict[str, float] = {}
    counts = {
        "non_car_network_freeflow": 0,
        "non_car_zero_pt_links": 0,
        "non_car_spec_links": 0,
        "non_car_other_zero_links": 0,
    }
    for link in raw.links:
        lid = link.link_id
        if lid in car_link_set:
            tt = actual.travel_times_s.get(lid)
            ff = actual.free_flow_times_s.get(lid)
            if (tt is None or not math.isfinite(tt) or tt <= 0
                    or ff is None or not math.isfinite(ff) or ff <= 0):
                raise ValueError(f"Missing positive observed/free-flow car TT: {lid}")
            true_times[lid] = float(tt)
            full_freeflow[lid] = float(ff)
        else:
            ff = _network_free_flow(link)
            if lid.endswith("_spec") and ff <= 0:
                # Existing PTNetwork requires a positive cost for *_spec.
                raise ValueError(f"Bus-special link {lid} has zero free-flow time")
            if lid.startswith("pt_") and ff == 0:
                counts["non_car_zero_pt_links"] += 1
            elif ff == 0:
                counts["non_car_other_zero_links"] += 1
            else:
                counts["non_car_network_freeflow"] += 1
            if lid.endswith("_spec"):
                counts["non_car_spec_links"] += 1
            full_freeflow[lid] = ff
            true_times[lid] = ff
            pred_times[lid] = ff

    below_freeflow = adjusted_positive = adjusted_freeflow = 0
    for row in eligible.itertuples(index=False):
        lid, idx = str(row.link_id), int(row.node_id)
        tt = float(predicted[idx, 0])
        ff = full_freeflow[lid]
        if tt < ff - 0.01:
            below_freeflow += 1
        if floor == "freeflow":
            adjusted_freeflow += int(tt < ff)
            tt = max(tt, ff)
        elif floor == "positive":
            adjusted_positive += int(tt <= 0.001)
            tt = max(tt, 0.001)
        else:
            raise ValueError(f"Unknown prediction floor: {floor}")
        pred_times[lid] = tt

    if set(true_times) != set(pred_times) or set(pred_times) != set(full_freeflow):
        raise AssertionError("Incomplete paired full-network travel times")
    # The supplied source may contain PT NaNs/zeros; our full travel times are
    # explicitly derived from network free flow for these links, so do not
    # propagate irrelevant linkstats-invalid flags into the routing inputs.
    both = dict(free_flow_times_s=full_freeflow,
                invalid_travel_times={}, hourly_volumes={})
    reference = replace(actual, travel_times_s=true_times, **both)
    surrogate = replace(actual, travel_times_s=pred_times, **both)
    diagnostics = {
        "car_predicted_count": len(eligible),
        "raw_predicted_below_freeflow": below_freeflow,
        "clipped_to_positive": adjusted_positive,
        "clipped_to_freeflow": adjusted_freeflow,
        "unpredicted_link_fallback": counts,
    }
    return reference, surrogate, diagnostics


def _calculate_and_score(
    scenario_dir: Path, spatial_dir: Path, linkstats: LinkTravelTimes,
    reference: SufficiencyReference, mask: dict,
) -> tuple[dict, list[dict]]:
    access = calculate_accessibility(
        scenario_dir, spatial_dir, thresholds_min=settings.THRESHOLDS_MIN,
        hour_start=settings.HOUR_START,
        linkstats_statistic=settings.LINKSTATS_STATISTIC,
        bike_speed_m_s=settings.BIKE_SPEED_M_S,
        pt_transfer_penalty_s=settings.PT_TRANSFER_PENALTY_S,
        origin_snap_m=settings.ORIGIN_SNAP_M,
        job_snap_m=settings.JOB_SNAP_M,
        linkstats_override=linkstats,
    )
    scored, summary = assess_sufficiency(access, reference, settings.THRESHOLDS_MIN)
    fitness = score_sufficiency(
        summary, scored, mask, reference,
        mode_weights=settings.MODE_WEIGHTS,
        time_weights=settings.TIME_WEIGHTS,
        level=settings.SUFFICIENCY_LEVEL,
    )
    return fitness, summary.to_dict("records")


def _process_one(task: tuple[str, str, str, str, str, str, str, str]) -> dict:
    scenario, dataset, gnn_nodes, evaluation, spatial, reference_path, baseline_name, floor = task
    started = perf_counter()
    scenario_dir = Path(dataset) / scenario
    if not scenario_dir.is_dir():
        raise FileNotFoundError(f"Missing original MATSim scenario: {scenario_dir}")
    metadata = _json(scenario_dir / "scenario.json")
    baseline_metadata = _json(Path(dataset) / baseline_name / "scenario.json")
    if metadata.get("id") != scenario or (not metadata.get("baseline_hashes") or
            metadata["baseline_hashes"] != baseline_metadata.get("baseline_hashes")):
        raise ValueError(f"Scenario {scenario} does not match the fixed baseline")
    status = scenario_dir / "run_status.json"
    if status.is_file():
        run = _json(status)
        if run.get("state", run.get("status")) not in ("completed", "complete"):
            raise ValueError(f"MATSim scenario is not completed: {status}")
    prediction = _prediction_data(evaluation, scenario)
    nodes = _graph_nodes(gnn_nodes)
    reference_file = Path(reference_path)
    reference = SufficiencyReference.load(reference_file)
    baseline_mask = _load_or_create_baseline_mask(
        _baseline_mask_path(reference_file), pd.DataFrame(), reference_file,
        baseline_name, is_baseline=False,
    )
    actual = LinkTravelTimes.from_linkstats(
        scenario_dir / "targets" / "linkstats.txt.gz",
        hour_start=settings.HOUR_START, statistic=settings.LINKSTATS_STATISTIC,
    )
    actual_full, predicted, diagnostics = _paired_times(
        scenario_dir, nodes, prediction, actual, floor=floor,
    )
    actual_score, actual_summary = _calculate_and_score(
        scenario_dir, Path(spatial), actual_full, reference, baseline_mask,
    )
    predicted_score, predicted_summary = _calculate_and_score(
        scenario_dir, Path(spatial), predicted, reference, baseline_mask,
    )
    comp = []
    for method, score in (("matsim", actual_score), ("gnn", predicted_score)):
        for item in score["components"]:
            comp.append({"scenario": scenario, "source": method, **item})
    return {
        "scenario": scenario,
        "strategy": metadata.get("strategy"),
        "budget_used_m": metadata.get("budget_used_m"),
        "intervention_count": metadata.get("intervention_count"),
        "fitness_matsim": actual_score["fitness"],
        "fitness_gnn": predicted_score["fitness"],
        "fitness_error": predicted_score["fitness"] - actual_score["fitness"],
        "weighted_shortfall_matsim": actual_score["weighted_shortfall"],
        "weighted_shortfall_gnn": predicted_score["weighted_shortfall"],
        "time_s": round(perf_counter() - started, 3),
        "diagnostics": diagnostics,
        "components": comp,
        "sufficiency_matsim": actual_summary,
        "sufficiency_gnn": predicted_summary,
    }


def _ranking(rows: list[dict], k_values: tuple[int, ...]) -> dict:
    df = pd.DataFrame(rows)
    true = df["fitness_matsim"].to_numpy(dtype=float)
    pred = df["fitness_gnn"].to_numpy(dtype=float)
    err = pred - true
    if not np.isfinite(true).all() or not np.isfinite(pred).all():
        raise ValueError("Nonfinite paired fitness")
    ordering_true = np.argsort(-true, kind="stable")
    ordering_pred = np.argsort(-pred, kind="stable")
    select_pred = ordering_pred[0]
    best_actual = true[ordering_true[0]]
    metrics = {
        "scenarios": len(df),
        "fitness_mae": float(np.abs(err).mean()),
        "fitness_rmse": float(np.sqrt(np.square(err).mean())),
        "fitness_bias": float(err.mean()),
        "fitness_spearman": (float(df["fitness_gnn"].corr(df["fitness_matsim"], method="spearman"))
                             if len(set(pred)) > 1 and len(set(true)) > 1 else None),
        "fitness_pearson": (float(df["fitness_gnn"].corr(df["fitness_matsim"], method="pearson"))
                            if len(set(pred)) > 1 and len(set(true)) > 1 else None),
        "best_matsim_scenario": df.iloc[ordering_true[0]]["scenario"],
        "best_matsim_fitness": float(best_actual),
        "gnn_top_scenario": df.iloc[select_pred]["scenario"],
        "gnn_top_actual_fitness": float(true[select_pred]),
        "best_selection_regret": float(max(0, best_actual - true[select_pred])),
        "top_k": {},
    }
    for k in sorted(set(k_values)):
        if k < 1 or k > len(df):
            continue
        a = set(ordering_true[:k]); b = set(ordering_pred[:k])
        best_in_top_k = float(np.max(true[ordering_pred[:k]]))
        metrics["top_k"][str(k)] = {
            "overlap": len(a & b),
            "overlap_fraction": len(a & b) / k,
            "best_actual_fitness_in_predicted_top_k": best_in_top_k,
            "best_in_top_k_regret": float(max(0, best_actual - best_in_top_k)),
        }
    return metrics


def _make_plots(rows: list[dict], out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    df = pd.DataFrame(rows)
    x = df["fitness_matsim"].to_numpy(float)
    y = df["fitness_gnn"].to_numpy(float)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(x, y, s=14, alpha=0.55)
    lo, hi = min(x.min(), y.min()), max(x.max(), y.max())
    ax.plot([lo, hi], [lo, hi], "k--", lw=1)
    ax.set(xlabel="MATSim fitness", ylabel="GNN fitness", title="Paired test-scenario fitness")
    fig.tight_layout()
    fig.savefig(out / "fitness_scatter.png", dpi=170)
    plt.close(fig)
    ranked = df.sort_values("fitness_matsim", ascending=False).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(np.arange(1, len(ranked) + 1), ranked["fitness_matsim"], label="MATSim", lw=1.5)
    ax.scatter(np.arange(1, len(ranked) + 1), ranked["fitness_gnn"], s=10,
               alpha=0.55, label="GNN")
    ax.set(xlabel="Rank by MATSim fitness", ylabel="Fitness", title="Predictions ordered by true rank")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "fitness_by_true_rank.png", dpi=170)
    plt.close(fig)


def _validate_eval(evaluation: Path, test_ids: list[str]) -> dict:
    if evaluation.suffix == ".zip":
        with zipfile.ZipFile(evaluation) as zf:
            names = zf.namelist()
            if not any(n.endswith("summary.json") for n in names):
                raise FileNotFoundError("Missing evaluation summary in zip")
            matches = [n for n in names if n.endswith("summary.json")]
            if len(matches) != 1:
                raise ValueError("Ambiguous test summary in zip")
            summary = json.loads(zf.read(matches[0]))
            metrics_csv = [n for n in names if n.endswith("scenario_metrics.csv")]
            if len(metrics_csv) != 1:
                raise ValueError("Cannot verify scenario IDs in archived evaluation")
            import io
            scenarios = pd.read_csv(io.BytesIO(zf.read(metrics_csv[0])), usecols=["scenario"])
    else:
        summary = _json(evaluation / "summary.json")
        scenarios = pd.read_csv(evaluation / "scenario_metrics.csv", usecols=["scenario"])
    if summary.get("split") != "test" or summary.get("target_type") != "road_congestion_delay_change":
        raise ValueError("Fitness evaluation requires saved TEST split of delay-target GNN")
    if summary.get("hour") != settings.HOUR_START:
        raise ValueError("Saved predictions use a different hour than accessibility")
    if (summary.get("scenarios") != len(test_ids)
            or len(scenarios) != len(test_ids)
            or set(scenarios["scenario"].astype(str)) != set(test_ids)):
        raise ValueError("Evaluation scenarios differ from original frozen test split")
    return summary


def _ensure_baseline(reference: Path, baseline: Path, spatial_dir: Path,
                     baseline_name: str, initialize: bool) -> None:
    mask = _baseline_mask_path(reference)
    if reference.is_file() and mask.is_file():
        _load_or_create_baseline_mask(mask, pd.DataFrame(), reference, baseline_name,
                                      is_baseline=False)
        SufficiencyReference.load(reference)
        return
    if not initialize:
        raise FileNotFoundError(f"Missing fixed reference {reference} or attachment mask {mask}. "
                                "Run with --initialize-baseline to compute them once")
    print("Building missing fixed baseline reference/mask from baseline MATSim", flush=True)
    evaluate_scenario(
        baseline, spatial_dir, reference, baseline_name=baseline_name,
        mode_weights=settings.MODE_WEIGHTS, time_weights=settings.TIME_WEIGHTS,
        level=settings.SUFFICIENCY_LEVEL, hour_start=settings.HOUR_START,
        linkstats_statistic=settings.LINKSTATS_STATISTIC,
        bike_speed_m_s=settings.BIKE_SPEED_M_S,
        pt_transfer_penalty_s=settings.PT_TRANSFER_PENALTY_S,
        origin_snap_m=settings.ORIGIN_SNAP_M, job_snap_m=settings.JOB_SNAP_M,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gnn-dir", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True,
                        help="Saved evaluation_test folder OR zip containing it")
    parser.add_argument("--dataset", type=Path, required=True,
                        help="Historical MATSim scenario dataset with baseline_00000")
    parser.add_argument("--spatial-dir", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--baseline-name", default="baseline_00000")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--limit", type=int, default=0,
                        help="0=entire test; positive value for quick smoke test")
    parser.add_argument("--floor", choices=("positive", "freeflow"), default="positive",
                        help="Positive floor keeps below-freeflow predictions; freeflow is a separate policy")
    parser.add_argument("--initialize-baseline", action="store_true")
    parser.add_argument("--plots", action="store_true")
    parser.add_argument("--resume", action="store_true",
                        help="Continue a partially completed run with identical inputs/settings")
    args = parser.parse_args()
    if args.workers < 1 or args.limit < 0:
        parser.error("--workers must be >=1 and --limit >=0")
    all_test_ids = _test_ids(args.gnn_dir / "splits.json")
    eval_summary = _validate_eval(args.evaluation, all_test_ids)
    ids = all_test_ids
    if args.limit:
        ids = ids[:args.limit]
    if not ids:
        raise ValueError("Empty test set")
    _ensure_baseline(args.reference, args.dataset / args.baseline_name,
                     args.spatial_dir, args.baseline_name, args.initialize_baseline)
    nodes_path = args.gnn_dir / "gnn_static" / "dual_nodes.csv.gz"
    if not nodes_path.is_file():
        raise FileNotFoundError(nodes_path)
    args.out.mkdir(parents=True, exist_ok=True)
    parameters = {
        "split": "test", "scenarios": len(ids),
        "fitness_version": FITNESS_VERSION,
        "model_target": "road_congestion_delay_change",
        "prediction_floor": args.floor,
        "non_car_prediction_fallback": "network_freeflow_for_both_branches_v1",
        "reference_sha256": _sha(args.reference),
        "attachment_mask_sha256": _sha(_baseline_mask_path(args.reference)),
        "spatial_dir": str(args.spatial_dir.resolve()),
        "test_scenario_ids": ids,
        "gnn_dir": str(args.gnn_dir.resolve()),
        "evaluation": str(args.evaluation.resolve()),
        "dataset": str(args.dataset.resolve()),
        "best_epoch": eval_summary.get("best_epoch"),
        "fitness_options": {
            "hour": settings.HOUR_START, "linkstats_statistic": settings.LINKSTATS_STATISTIC,
            "mode_weights": settings.MODE_WEIGHTS, "time_weights": settings.TIME_WEIGHTS,
            "sufficiency_level": settings.SUFFICIENCY_LEVEL,
            "bike_speed_m_s": settings.BIKE_SPEED_M_S,
            "pt_transfer_penalty_s": settings.PT_TRANSFER_PENALTY_S,
            "origin_snap_m": settings.ORIGIN_SNAP_M, "job_snap_m": settings.JOB_SNAP_M,
        },
    }
    config_file = args.out / "evaluation_config.json"
    if args.resume and config_file.is_file():
        if _json(config_file) != parameters:
            saved = list((args.out / "per_scenario").glob("*.json"))
            if saved:
                raise ValueError("Existing successful scenario checkpoints use a different "
                                 "travel-time policy. Use a new --out directory")
            print("Replacing obsolete evaluation configuration: no completed scenario "
                  "checkpoints exist", flush=True)
            _write_json(config_file, parameters)
    else:
        if args.resume and (args.out / "per_scenario").exists():
            raise ValueError("Existing checkpoints have no matching configuration")
        _write_json(config_file, parameters)

    per_scenario = args.out / "per_scenario"
    per_scenario.mkdir(exist_ok=True)
    tasks = [(sid, str(args.dataset), str(nodes_path), str(args.evaluation),
              str(args.spatial_dir), str(args.reference), args.baseline_name, args.floor)
             for sid in ids]
    results: dict[str, dict] = {}
    todo = []
    for task in tasks:
        saved = per_scenario / f"{task[0]}.json"
        if args.resume and saved.is_file():
            payload = _json(saved)
            if payload.get("scenario") != task[0]:
                raise ValueError(f"Corrupt result cache: {saved}")
            results[task[0]] = payload
        else:
            todo.append(task)
    failures = []
    started = perf_counter()
    print(f"Evaluating {len(ids)} test scenarios, {len(results)} cached, "
          f"{len(todo)} pending, {args.workers} worker(s)", flush=True)

    def record(scenario: str, result: dict | None, error: Exception | None) -> None:
        if error is None:
            assert result is not None and result["scenario"] == scenario
            _write_json(per_scenario / f"{scenario}.json", result)
            results[scenario] = result
        else:
            failures.append({"scenario": scenario,
                             "error": f"{type(error).__name__}: {error}"})
        print(f"{len(results)+len(failures)}/{len(ids)} finished ({scenario})", flush=True)

    if args.workers == 1:
        for task in todo:
            try:
                record(task[0], _process_one(task), None)
            except Exception as exc:
                record(task[0], None, exc)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            submitted = {pool.submit(_process_one, task): task[0] for task in todo}
            for future in as_completed(submitted):
                scenario = submitted[future]
                try:
                    record(scenario, future.result(), None)
                except Exception as exc:
                    record(scenario, None, exc)

    rows, components = [], []
    for sid in ids:
        if sid not in results:
            continue
        result = results[sid]
        row = {k: v for k, v in result.items()
               if k not in {"components", "diagnostics", "sufficiency_matsim", "sufficiency_gnn"}}
        diag = result["diagnostics"]
        row.update({"raw_below_freeflow": diag["raw_predicted_below_freeflow"],
                    "clipped_positive": diag["clipped_to_positive"],
                    "clipped_freeflow": diag["clipped_to_freeflow"],
                    **{f"fallback_{k}": v for k, v in diag["unpredicted_link_fallback"].items()}})
        for comp in result["components"]:
            components.append(comp)
            row[f"fitness_{comp['source']}_{comp['mode']}_{comp['time_min']:g}min"] = comp["mode_score"]
        rows.append(row)
    components.sort(key=lambda row: (row["scenario"], row["source"], row["mode"], row["time_min"]))
    if rows:
        _write_csv(args.out / "fitness_comparison.csv", rows)
        _write_csv(args.out / "mode_comparison.csv", components)
    if failures:
        _write_csv(args.out / "failures.csv", failures)
        raise RuntimeError(f"Fitness evaluation failed for {len(failures)}/{len(ids)} scenarios. "
                           f"See {args.out / 'failures.csv'}; rerun with --resume")
    (args.out / "failures.csv").unlink(missing_ok=True)
    if len(rows) != len(ids):
        raise RuntimeError("Incomplete scenario results; cannot report rankings")
    metrics = _ranking(rows, (10, 20, 50))
    metrics["elapsed_s"] = perf_counter() - started
    metrics["total_raw_below_freeflow"] = int(sum(r["raw_below_freeflow"] for r in rows))
    metrics["total_clipped_positive"] = int(sum(r["clipped_positive"] for r in rows))
    metrics["total_clipped_freeflow"] = int(sum(r["clipped_freeflow"] for r in rows))
    _write_json(args.out / "ranking_metrics.json", metrics)
    if args.plots:
        try:
            _make_plots(rows, args.out)
        except ImportError as exc:
            print(f"Plots skipped (matplotlib unavailable): {exc}", flush=True)
    print(json.dumps(metrics, indent=2, allow_nan=False), flush=True)
    print(f"Saved results in {args.out}", flush=True)


if __name__ == "__main__":
    main()

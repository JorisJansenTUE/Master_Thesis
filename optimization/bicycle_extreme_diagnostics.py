"""Two independent bicycle-accessibility diagnostic experiments (MATSim NOT run).

A: Rank all previously evaluated MATSim scenarios by the bicycle component of
   one fixed STATPOP-weighted fitness trial, with and without its car constraint.
C: Hypothetical stress-free cycling on every *existing bike-accessible link* of
   the baseline MATSim network, with unchanged geometry, access, speed, and
   origin/job snapping. This is a comfort-only benchmark, not a realizable
   intervention scenario or a universal theoretical upper bound.

Usage:
 python -m optimization.bicycle_extreme_diagnostics \
   --trial .../fitness_experiments/trial_03 \
   --spatial .../spatial_baseline \
   --population .../spatial_baseline/population_by_hexagon.csv \
   --out .../fitness_experiments/bicycle_extremes_01

Reads original scenario/accessibility outputs, writes ONLY under --out.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from accessibility.calculations.cumulative import CumulativeAccessibility
from accessibility.inputs.grid.snapping import ScenarioRoadSnapper
from accessibility.inputs.links.network import BikeNetwork
from accessibility.routing.road import RoadRouter


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def digest(path: Path) -> str:
    hash_ = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hash_.update(chunk)
    return hash_.hexdigest()


def jobs_column(time_min: float) -> str:
    return CumulativeAccessibility.column_name(float(time_min))


def component(row: dict, mode: str, time_min: float) -> dict:
    hits = [x for x in row["components"]
            if x["mode"] == mode and abs(float(x["time_min"]) - float(time_min)) < 1e-8]
    if len(hits) != 1:
        raise ValueError(f"Expected exactly one {mode}/{time_min:g} min component, got {len(hits)}")
    return hits[0]


def read_population(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path, dtype={"hex_id": str})
    required = {"hex_id", "population"}
    if not required.issubset(data.columns) or data.hex_id.isna().any() or data.hex_id.duplicated().any():
        raise ValueError(f"Invalid STATPOP file (need unique hex_id,population): {path}")
    data["population"] = pd.to_numeric(data["population"], errors="raise")
    weights = data.population.to_numpy(float)
    if not np.isfinite(weights).all() or (weights < 0).any() or weights.sum() <= 0:
        raise ValueError("Invalid population weights")
    return data[["hex_id", "population"]].copy()


def prepared_accessibility(df: pd.DataFrame, mode: str, column: str, pop: pd.DataFrame) -> pd.DataFrame:
    if not {"hex_id", "mode", "attached", column}.issubset(df.columns):
        raise ValueError(f"Accessibility missing required columns for {mode}/{column}")
    selected = df.loc[df["mode"] == mode, ["hex_id", "attached", column]].copy()
    selected.hex_id = selected.hex_id.astype(str)
    if selected.hex_id.duplicated().any() or set(selected.hex_id) != set(pop.hex_id):
        raise ValueError(f"Accessibility hexagon IDs do not match STATPOP for {mode}")
    state = selected.attached.astype(str).str.strip().str.lower()
    if not state.isin(["true", "false", "1", "0"]).all():
        raise ValueError("Invalid attached flag")
    attached = state.isin(["true", "1"])
    values = pd.to_numeric(selected[column], errors="raise")
    if values[attached].isna().any() or not np.isfinite(values[attached].to_numpy(float)).all():
        raise ValueError("Attached origins require finite job counts")
    if (values[attached] < 0).any():
        raise ValueError("Negative job counts")
    # Unattached points contribute zero accessibility, regardless of csv NaN.
    selected[column] = values.where(attached, 0).fillna(0).astype(float)
    selected["attached"] = attached.to_numpy(bool)
    return pop.merge(selected, on="hex_id", how="left", validate="one_to_one").sort_values("hex_id").reset_index(drop=True)


def cache_bike(path: Path, pop: pd.DataFrame, column: str) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    df = pd.read_csv(path, dtype={"hex_id": str, "mode": str})
    return prepared_accessibility(df, "bike", column, pop)


def weighted_mean(a: np.ndarray, p: np.ndarray) -> float:
    return float(np.dot(np.asarray(a, float), np.asarray(p, float)) / np.asarray(p, float).sum())


def score_bike(jobs: np.ndarray, pop: np.ndarray, threshold: float, exponent: float) -> float:
    if threshold <= 0 or exponent <= 0:
        raise ValueError("Sufficiency threshold and exponent must be positive")
    utilities = 1 - np.maximum(0.0, 1.0 - np.asarray(jobs, float)/threshold)**exponent
    return weighted_mean(utilities, pop)


def physical_metrics(baseline: pd.DataFrame, scenario: pd.DataFrame, col: str,
                     threshold: float, exponent: float) -> dict:
    if not baseline.hex_id.equals(scenario.hex_id) or not baseline.population.equals(scenario.population):
        raise ValueError("Physical comparison requires identical hexagon grid and population")
    original = baseline[col].to_numpy(float)
    new = scenario[col].to_numpy(float)
    weight = baseline.population.to_numpy(float)
    total = weight.sum()
    delta = new - original
    old_score = score_bike(original, weight, threshold, exponent)
    new_score = score_bike(new, weight, threshold, exponent)
    return {
        "baseline_bike_fitness": old_score,
        "scenario_bike_fitness": new_score,
        "delta_bike_fitness": new_score - old_score,
        "baseline_mean_accessible_jobs": weighted_mean(original, weight),
        "scenario_mean_accessible_jobs": weighted_mean(new, weight),
        "delta_mean_accessible_jobs": weighted_mean(delta, weight),
        "baseline_sufficient_resident_share": weighted_mean((original >= threshold).astype(float), weight),
        "scenario_sufficient_resident_share": weighted_mean((new >= threshold).astype(float), weight),
        "delta_sufficient_resident_share_pp": 100 * weighted_mean(
            (new >= threshold).astype(float) - (original >= threshold).astype(float), weight),
        "resident_share_with_more_jobs": float(weight[delta > 1e-7].sum()/total),
        "resident_share_with_fewer_jobs": float(weight[delta < -1e-7].sum()/total),
        "resident_share_unchanged": float(weight[np.abs(delta) <= 1e-7].sum()/total),
        "baseline_underserved_resident_share": float(weight[original < threshold].sum()/total),
        "mean_gain_jobs_baseline_underserved": (
            weighted_mean(delta[original < threshold], weight[original < threshold])
            if weight[original < threshold].sum() > 0 else None),
    }


def read_trial(trial: Path) -> tuple[dict, list[dict]]:
    conf_path = trial / "experiment.json"
    conf = json.loads(conf_path.read_text(encoding="utf-8"))
    files = [trial / "chunks" / f"chunk_{i:05d}.json" for i in range(conf["n_chunks"])]
    missing = [p for p in files if not p.is_file()]
    if missing:
        raise RuntimeError(f"Trial incomplete: {len(missing)} chunk files missing, e.g. {missing[:2]}")
    rows = []
    failures = 0
    for p in files:
        payload = json.loads(p.read_text(encoding="utf-8"))
        rows.extend(payload["completed"])
        failures += len(payload["errors"])
    if len(rows) + failures != conf["n_scenarios"]:
        raise RuntimeError("Trial has missing/duplicated results")
    if not rows:
        raise RuntimeError("No valid scenarios in trial")
    return conf, rows


def rank_existing(conf: dict, records: list[dict], cutoff: float) -> tuple[pd.DataFrame, dict]:
    base_bike = float(component(conf["baseline_score"], "bike", cutoff)["score"])
    base_pt = float(component(conf["baseline_score"], "pt", cutoff)["score"])
    base_car = float(component(conf["baseline_score"], "car", cutoff)["score"])
    lines = []
    for r in records:
        bike = float(component(r, "bike", cutoff)["score"])
        pt = float(component(r, "pt", cutoff)["score"])
        car = float(component(r, "car", cutoff)["score"])
        lines.append({
            "scenario": r["scenario"], "source": r["source"],
            "feasible_under_trial_constraint": bool(r["feasible"]),
            "car_sufficient_share": r.get("car_sufficient_share"),
            "bike_score": bike, "delta_bike_score": bike-base_bike,
            "pt_score": pt, "delta_pt_score": pt-base_pt,
            "car_score": car, "delta_car_score": car-base_car,
            "combined_original_score": float(r["raw_fitness"]),
        })
    df = pd.DataFrame(lines).sort_values("bike_score", ascending=False).reset_index(drop=True)
    good = df[df.feasible_under_trial_constraint]
    top_all = df.iloc[0].to_dict()
    top_good = good.iloc[0].to_dict() if len(good) else None
    summary = {
        "baseline_bike_score": base_bike,
        "baseline_pt_score": base_pt,
        "baseline_car_score": base_car,
        "n_scored": len(df), "n_feasible": len(good),
        "n_bike_improving": int((df.delta_bike_score > 1e-12).sum()),
        "n_bike_improving_feasible": int((good.delta_bike_score > 1e-12).sum()),
        "bike_gain_quantiles": {str(q): float(df.delta_bike_score.quantile(q))
                                for q in (0, .1, .25, .5, .75, .9, .99, 1)},
        "best_bike_no_constraint": top_all,
        "best_bike_with_trial_car_constraint": top_good,
    }
    return df, summary


def load_spatial(spatial: Path, pop: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    origins = pd.read_csv(spatial/"origins.csv", dtype={"hex_id": str})
    jobs = pd.read_csv(spatial/"jobs.csv", dtype={"job_id": str})
    if len(origins) != len(pop) or set(origins.hex_id) != set(pop.hex_id):
        raise ValueError("Spatial origins.csv and STATPOP population refer to different hexagon grids")
    if jobs.job_id.duplicated().any() or (pd.to_numeric(jobs.jobs, errors="raise") < 0).any():
        raise ValueError("Invalid jobs.csv")
    return origins, jobs


def bike_accessibility(graph, origins: pd.DataFrame, jobs: pd.DataFrame,
                       cutoff: float, snapper: ScenarioRoadSnapper) -> pd.DataFrame:
    attachments = snapper.attach(graph)
    router = RoadRouter(graph, attachments.origins.connectors, attachments.jobs.connectors, jobs)
    print(f"  attached origins={attachments.origins.attached_points}/{len(origins)}, "
          f"jobs={attachments.jobs.attached_points}/{len(jobs)}", flush=True)
    return CumulativeAccessibility((cutoff,)).run(router, origins.hex_id.astype(str), mode="bike")


def stress_free_bike_graph(bike: BikeNetwork) -> tuple[int, float]:
    """Remove ONLY eQasim cycling discomfort multipliers; preserve physical link times."""
    changed = 0
    penalty_seconds = 0.0
    for _, _, _, data in bike.graph.edges(keys=True, data=True):
        cost = float(data["routing_cost_s"])
        physical = float(data["travel_time_s"])
        if physical > cost + 1e-8:
            raise AssertionError("Existing cycling cost lower than physical time")
        if cost > physical + 1e-8:
            changed += 1
            penalty_seconds += cost - physical
        data["routing_cost_s"] = physical
        data["safety_multiplier"] = 1.0
        data["infrastructure_disutility_s"] = 0.0
    return changed, penalty_seconds


def make_graph_benchmark(network_path: Path, spatial: Path, pop: pd.DataFrame,
                         config: dict, cutoff: float, threshold: float, out: Path,
                         baseline_cache: pd.DataFrame, tolerance: float) -> dict:
    origins, jobs = load_spatial(spatial, pop)
    cfg = config["config"]
    bike = BikeNetwork.from_matsim_network(network_path, bike_speed_m_s=float(cfg["BIKE_SPEED_M_S"]))
    snapper = ScenarioRoadSnapper(origins, jobs,
                                   max_origin_distance_m=float(cfg["ORIGIN_SNAP_M"]),
                                   max_job_distance_m=float(cfg["JOB_SNAP_M"]))
    print("Rerouting baseline on original bike cost graph...", flush=True)
    normal = bike_accessibility(bike.graph, origins, jobs, cutoff, snapper)
    baseline_direct = prepared_accessibility(normal, "bike", jobs_column(cutoff), pop)
    old = baseline_cache[jobs_column(cutoff)].to_numpy(float)
    regenerated = baseline_direct[jobs_column(cutoff)].to_numpy(float)
    mismatch = np.abs(regenerated-old)
    n_mismatch = int((mismatch > tolerance).sum())
    print(f"  cached vs rerouted baseline: mismatched hexagons={n_mismatch}, "
          f"max absolute job difference={float(mismatch.max()):.3f}", flush=True)
    if n_mismatch:
        diagnostic = pd.DataFrame({"hex_id": baseline_direct.hex_id,
                                   "cached_jobs": old, "rerouted_jobs": regenerated,
                                   "difference_jobs": regenerated-old})
        diagnostic.to_csv(out/"baseline_routing_mismatch.csv.gz", compression="gzip", index=False)
        raise RuntimeError(
            f"Baseline routing differs from experiment cache at {n_mismatch} origins. "
            f"Wrote {out/'baseline_routing_mismatch.csv.gz'}. "
            "Do not directly compare idealized and trial candidate fitness until "
            "network/routing settings have been reconciled. Existing-sample results are saved."
        )
    print("Routing idealized stress-free cycling network...", flush=True)
    changed, penalty_seconds = stress_free_bike_graph(bike)
    ideal = bike_accessibility(bike.graph, origins, jobs, cutoff, snapper)
    ideal = prepared_accessibility(ideal, "bike", jobs_column(cutoff), pop)
    results = physical_metrics(baseline_direct, ideal, jobs_column(cutoff), threshold,
                               exponent=float(cfg["SHORTFALL_EXPONENT"]))
    delta = ideal[jobs_column(cutoff)].to_numpy(float)-regenerated
    if (delta < -tolerance).any():
        raise AssertionError("Idealized bike-only change reduced accessibility; check routing")
    comparison = baseline_direct[["hex_id", "population", "attached", jobs_column(cutoff)]].rename(
        columns={"attached": "baseline_attached", jobs_column(cutoff): "baseline_jobs"})
    comparison["idealized_attached"] = ideal.attached.to_numpy(bool)
    comparison["idealized_jobs"] = ideal[jobs_column(cutoff)].to_numpy(float)
    comparison["delta_jobs"] = delta
    comparison.to_csv(out/"idealized_bike_hexagon_changes.csv.gz", index=False, compression="gzip")
    return {
        "experiment": "C_stress_free_existing_bike_network",
        "interpretation": "Hypothetical comfort-only benchmark. Not a real road-space allocation, MATSim scenario, or universal physical upper bound.",
        "baseline_network": str(network_path), "bike_link_count": bike.num_links,
        "bike_links_disutility_reduced": changed,
        "total_removed_perceived_link_penalties_s": penalty_seconds,
        "routing_baseline_equals_fitness_cache": True,
        "cutoff_min": cutoff, "sufficiency_threshold_jobs": threshold,
        "bike_speed_m_s": float(cfg["BIKE_SPEED_M_S"]),
        "fitness_exponent": float(cfg["SHORTFALL_EXPONENT"]),
        "results": results,
    }


def plot_existing(frame: pd.DataFrame, out: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping plots", flush=True)
        return
    fig, ax = plt.subplots(figsize=(8, 4.6))
    gains = frame.delta_bike_score.to_numpy(float)
    ax.hist(gains, bins=55, edgecolor="white", linewidth=.3)
    ax.axvline(0, color="black", linewidth=1.2, linestyle="--")
    ax.axvline(np.max(gains), color="tab:orange", linewidth=1.4,
               label="Best observed bicycle-only score")
    ax.set(title="Existing MATSim scenarios: bicycle-only fitness changes",
           xlabel="Change in population-weighted bicycle score vs baseline",
           ylabel="Number of scenarios")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out/"existing_bike_fitness_distribution.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    subset = frame[frame.feasible_under_trial_constraint]
    rejected = frame[~frame.feasible_under_trial_constraint]
    if len(rejected):
        ax.scatter(rejected.delta_bike_score, rejected.delta_pt_score,
                   s=9, alpha=.25, label="Car-constraint infeasible")
    if len(subset):
        ax.scatter(subset.delta_bike_score, subset.delta_pt_score,
                   s=9, alpha=.28, label="Car-constraint feasible")
    ax.axvline(0, color="grey", lw=.8)
    ax.axhline(0, color="grey", lw=.8)
    ax.set(xlabel="Bicycle fitness change", ylabel="PT fitness change",
           title="Cycling versus PT trade-offs in existing MATSim scenarios")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out/"existing_bike_vs_pt_scatter.png", dpi=180)
    plt.close(fig)


def plot_comparison(existing: dict, ideal: dict, out: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    baseline = float(existing["baseline_bike_score"])
    best_feasible = existing["best_bike_with_trial_car_constraint"]
    best_all = existing["best_bike_no_constraint"]
    names = ["Baseline", "Best existing\n(car feasible)",
             "Best existing\n(unconstrained)", "Stress-free\nbenchmark"]
    vals = [baseline,
            float(best_feasible["bike_score"]) if best_feasible is not None else float("nan"),
            float(best_all["bike_score"]),
            float(ideal["results"]["scenario_bike_fitness"])]
    fig, ax = plt.subplots(figsize=(8.5, 5))
    bars = ax.bar(names, vals)
    for bar, value in zip(bars, vals):
        if np.isfinite(value):
            ax.text(bar.get_x()+bar.get_width()/2, value+.004, f"{value:.4f}",
                    ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0, min(1.09, max(v for v in vals if np.isfinite(v)) + .065))
    ax.set_ylabel("Population-weighted bicycle-only fitness (same threshold/exponent)")
    ax.set_title("Observed sample potential versus network-wide comfort benchmark")
    fig.tight_layout()
    fig.savefig(out/"sample_vs_idealized_bike.png", dpi=180)
    plt.close(fig)


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial", type=Path, required=True, help="Completed trial_03 or equivalent with car constraint")
    parser.add_argument("--spatial", type=Path, required=True, help="Spatial baseline directory with origins.csv, jobs.csv")
    parser.add_argument("--population", type=Path, required=True, help="STATPOP population_by_hexagon.csv")
    parser.add_argument("--out", type=Path, required=True, help="NEW, empty results directory")
    parser.add_argument("--cutoff", type=float, default=20.0)
    parser.add_argument("--no-idealized", action="store_true", help="Only rank existing scenarios (Experiment A)")
    parser.add_argument("--baseline-network", type=Path, default=None,
                        help="Override derived baseline network path")
    parser.add_argument("--baseline-tolerance-jobs", type=float, default=1e-6)
    args = parser.parse_args(argv)
    trial = args.trial.expanduser().resolve(strict=True)
    spatial = args.spatial.expanduser().resolve(strict=True)
    population_path = args.population.expanduser().resolve(strict=True)
    out = args.out.expanduser().resolve()
    if out.exists() and any(out.iterdir()):
        parser.error(f"Refusing to overwrite existing output directory: {out}")
    if args.baseline_tolerance_jobs < 0:
        parser.error("--baseline-tolerance-jobs must be nonnegative")
    out.mkdir(parents=True, exist_ok=True)
    pop = read_population(population_path)
    conf, records = read_trial(trial)
    cfg = conf["config"]
    cutoff = float(args.cutoff)
    if cutoff not in map(float, conf["required_times"]):
        raise ValueError(f"Trial has no cached cutoff {cutoff:g} min: {conf['required_times']}")
    col = jobs_column(cutoff)
    if col not in conf["reference"]:
        raise ValueError(f"Trial baseline has no benchmark {col}")
    threshold = float(conf["reference"][col]) * float(cfg["SUFFICIENCY_FRACTION"])
    base_source = Path(conf["baseline_source"]).expanduser().resolve(strict=True)
    if digest(base_source) != conf.get("baseline_sha256"):
        raise ValueError("Baseline accessibility cache changed since experiment")
    if digest(population_path) != conf["population_sha256"]:
        raise ValueError("STATPOP population file changed since experiment")
    if not cfg.get("CAR_CONSTRAINT_ENABLED"):
        print("WARNING: source trial has no active car constraint; 'feasible' means all", flush=True)

    print(f"Experiment A: ranking {len(records)} evaluated scenarios by BIKE only", flush=True)
    frame, summary = rank_existing(conf, records, cutoff)
    frame.to_csv(out/"existing_bike_ranked.csv.gz", compression="gzip", index=False)
    plot_existing(frame, out)
    base_bike = cache_bike(base_source, pop, col)
    for key in ("best_bike_with_trial_car_constraint", "best_bike_no_constraint"):
        chosen = summary[key]
        if chosen is None:
            continue
        candidate = cache_bike(Path(chosen["source"]), pop, col)
        measure = physical_metrics(base_bike, candidate, col, threshold,
                                   exponent=float(cfg["SHORTFALL_EXPONENT"]))
        if not np.isclose(measure["scenario_bike_fitness"], chosen["bike_score"], atol=1e-7):
            raise RuntimeError(f"Bike fitness mismatch reading scenario {chosen['scenario']}")
        chosen["physical_accessibility"] = measure
    write_json(out/"experiment_A_existing_samples.json", summary)
    write_json(out/"inputs.json", {
        "source_trial": str(trial), "trial_config": cfg, "baseline_source": str(base_source),
        "population": str(population_path), "population_sha256": digest(population_path),
        "baseline_sha256": digest(base_source),
        "spatial": str(spatial), "cutoff_min": cutoff,
        "sufficiency_threshold_jobs": threshold,
        "interpretation": "A ranks original MATSim outputs, C is a hypothetical static bicycle routing benchmark."
    })
    print(f"  baseline bike score={summary['baseline_bike_score']:.6f}", flush=True)
    for key in ("best_bike_with_trial_car_constraint", "best_bike_no_constraint"):
        r = summary[key]
        if r is not None:
            print(f"  {key}: {r['scenario']} "
                  f"F_bike={r['bike_score']:.6f} delta={r['delta_bike_score']:+.6f}", flush=True)

    if args.no_idealized:
        print(f"Experiment A results: {out}", flush=True)
        return
    print("Experiment C: replacing all baseline bike VTT multipliers with 1.0", flush=True)
    baseline_scenario = base_source.parent.parent if base_source.parent.name == "accessibility" else base_source.parent
    net = args.baseline_network if args.baseline_network else baseline_scenario / "network.xml.gz"
    net = net.expanduser().resolve(strict=True)
    ideal = make_graph_benchmark(net, spatial, pop, conf, cutoff, threshold, out, base_bike,
                                 args.baseline_tolerance_jobs)
    write_json(out/"experiment_C_stress_free_network.json", ideal)
    plot_comparison(summary, ideal, out)
    max_existing_gain = float(summary["best_bike_no_constraint"]["delta_bike_score"])
    ideal_gain = float(ideal["results"]["delta_bike_fitness"])
    combined = {
        "baseline_bike_fitness": summary["baseline_bike_score"],
        "best_existing_car_feasible_gain": (
            float(summary["best_bike_with_trial_car_constraint"]["delta_bike_score"])
            if summary["best_bike_with_trial_car_constraint"] is not None else None),
        "best_existing_unconstrained_gain": max_existing_gain,
        "hypothetical_stress_free_gain": ideal_gain,
        "observed_to_idealized_gain_ratio": max_existing_gain/ideal_gain if ideal_gain > 0 else None,
        "important_limitation": (
            "The stress-free case changes only perceived cost on baseline bike links, "
            "not permissions, travel speed, route topology, or MATSim demand. "
            "It is a sensitivity benchmark, NOT a global upper bound."
        ),
    }
    write_json(out/"comparison.json", combined)
    # Compare fixed physical outcomes, not just scores under a changed objective.
    comparison_rows = []
    for key, label in (("best_bike_with_trial_car_constraint", "Existing best car-feasible"),
                       ("best_bike_no_constraint", "Existing best unconstrained")):
        candidate = summary[key]
        if candidate is None:
            continue
        physical = candidate["physical_accessibility"]
        comparison_rows.append({
            "alternative": label, "scenario": candidate["scenario"],
            "car_constraint_feasible": candidate["feasible_under_trial_constraint"],
            "baseline_bike_fitness": physical["baseline_bike_fitness"],
            "scenario_bike_fitness": physical["scenario_bike_fitness"],
            "delta_bike_fitness": physical["delta_bike_fitness"],
            "population_weighted_mean_jobs_gain": physical["delta_mean_accessible_jobs"],
            "sufficient_resident_share_gain_pp": physical["delta_sufficient_resident_share_pp"],
            "resident_share_with_more_jobs": physical["resident_share_with_more_jobs"],
            "resident_share_with_fewer_jobs": physical["resident_share_with_fewer_jobs"],
        })
    physical = ideal["results"]
    comparison_rows.append({
        "alternative": "Idealized stress-free baseline network", "scenario": "hypothetical",
        "car_constraint_feasible": "not evaluated (bike-only benchmark)",
        "baseline_bike_fitness": physical["baseline_bike_fitness"],
        "scenario_bike_fitness": physical["scenario_bike_fitness"],
        "delta_bike_fitness": physical["delta_bike_fitness"],
        "population_weighted_mean_jobs_gain": physical["delta_mean_accessible_jobs"],
        "sufficient_resident_share_gain_pp": physical["delta_sufficient_resident_share_pp"],
        "resident_share_with_more_jobs": physical["resident_share_with_more_jobs"],
        "resident_share_with_fewer_jobs": physical["resident_share_with_fewer_jobs"],
    })
    pd.DataFrame(comparison_rows).to_csv(out/"comparison_accessibility_metrics.csv", index=False)
    lines = ["# Bicycle accessibility potential — diagnostic experiments", "",
             f"Source trial: `{trial}`", "",
             f"Population-weighted bicycle baseline fitness: **{summary['baseline_bike_score']:.6f}**", "",
             "| Alternative | Bicycle-only fitness improvement |", "|---|---:|",
             f"| Best existing car-feasible sample | {combined['best_existing_car_feasible_gain']:+.6f} |" if combined['best_existing_car_feasible_gain'] is not None else "| Best car-feasible sample | none |",
             f"| Best existing sample (no car constraint) | {max_existing_gain:+.6f} |",
             f"| Idealized stress-free baseline bicycle network | {ideal_gain:+.6f} |", "",
             "**Interpretation:** C uses the same fixed bike network, speeds, jobs and population "
             "but removes the perceived-discomfort multiplier. It is not a feasible road-space "
             "scenario, and does not provide a universal upper bound on intervention gains.", "",
             f"Idealized population-weighted accessible-jobs gain: "
             f"**{ideal['results']['delta_mean_accessible_jobs']:+.1f} jobs** per resident-weighted origin.", ""]
    (out/"SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")
    print("Comparison:", flush=True)
    print(f"  highest observed bicycle score gain={max_existing_gain:+.6f}", flush=True)
    print(f"  stress-free bike gain={ideal_gain:+.6f}", flush=True)
    print(f"  resident-weighted mean job gain={ideal['results']['delta_mean_accessible_jobs']:+.1f}", flush=True)
    print(f"Results and figures: {out}", flush=True)


if __name__ == "__main__":
    main()

"""Compare four bicycle budget potential experiments (cached results only).

python -m optimization.summarize_bicycle_budget_potential \
    --out /.../bicycle_budget_potential_01 \
    --extreme /.../bicycle_extremes_01
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def _best_incomplete(folder: Path) -> dict | None:
    best = None
    for p in (folder / "evaluations").glob("*.json"):
        item = _json(p)
        if item and item.get("feasible") and (best is None or item["bike_score"] > best["bike_score"]):
            best = item
    return best


def summarize(out: Path, extreme: Path | None) -> pd.DataFrame:
    out = out.resolve(strict=True)
    greedy_manifest = _json(out / "greedy" / "experiment_manifest.json")
    if greedy_manifest is None:
        raise FileNotFoundError("Greedy experiment_manifest.json missing")
    baseline = float(greedy_manifest["baseline_bike_score"])
    result = [{"experiment": "Baseline", "status": "reference", "budget_limit_km": 0,
               "used_budget_km": 0, "bike_score": baseline,
               "delta_bike_score": 0, "mean_job_gain": 0,
               "intervention_count": 0, "java_passed": True}]
    for p in sorted((out / "greedy" / "results").glob("*.json")):
        record = _json(p)
        result.append({"experiment": "Greedy: " + record["strategy"], "status": "complete",
                       "budget_limit_km": None, **{
                           key: record.get(key) for key in (
                               "used_budget_km", "bike_score", "bike_score_delta",
                               "mean_job_gain", "intervention_count", "java_passed")},
                       "delta_bike_score": record["bike_score_delta"]})
    for km in (15, 50, 150):
        folder = out / f"ga_{km}km"
        best = _json(folder / "best_validated.json")
        if best is not None:
            status = "complete"
        else:
            best = _best_incomplete(folder)
            status = "in progress / partial" if best is not None else "not started"
        row = {"experiment": f"GA {km} km", "budget_limit_km": km, "status": status}
        for key in ("used_budget_km", "bike_score", "bike_score_delta", "mean_job_gain",
                    "intervention_count", "java_passed"):
            row[key] = best.get(key) if best is not None else None
        result.append(row)
    if extreme is not None:
        a = _json(extreme / "experiment_A_existing_samples.json")
        c = _json(extreme / "experiment_C_stress_free_network.json")
        if a:
            best = a["best_bike_no_constraint"]
            result.append({"experiment": "Best historical MATSim sample (bike-only)",
                           "status": "reference", "budget_limit_km": None,
                           "used_budget_km": None, "bike_score": best["bike_score"],
                           "delta_bike_score": best["delta_bike_score"],
                           "mean_job_gain": best.get("physical_accessibility", {}).get("delta_mean_accessible_jobs"),
                           "intervention_count": None, "java_passed": None})
        if c:
            physical = c["results"]
            result.append({"experiment": "Idealized stress-free bicycle network",
                           "status": "hypothetical benchmark", "budget_limit_km": None,
                           "used_budget_km": None, "bike_score": physical["scenario_bike_fitness"],
                           "delta_bike_score": physical["delta_bike_fitness"],
                           "mean_job_gain": physical["delta_mean_accessible_jobs"],
                           "intervention_count": None, "java_passed": False})
    frame = pd.DataFrame(result)
    frame.to_csv(out / "comparison_all_four.csv", index=False)
    lines = ["# Bicycle budget-potential comparison", "",
             f"Trial 03 population-weighted bicycle baseline = **{baseline:.6f}**.", "",
             "The objective is **bicycle-only** (20 minutes, high sufficiency threshold, squared shortfall).",
             "Road-network feasibility is checked; the trial's population-level car constraint is NOT evaluated.",
             "", "| Experiment | Status | Used km | Bicycle F | ΔF | Mean additional jobs |",
             "|---|---|---:|---:|---:|---:|"]
    for r in result:
        fmt = lambda x, n=5: f"{x:.{n}f}" if x is not None and pd.notna(x) else "—"
        lines.append(f"| {r['experiment']} | {r['status']} | "
                     f"{fmt(r.get('used_budget_km'), 1)} | {fmt(r.get('bike_score'))} | "
                     f"{fmt(r.get('delta_bike_score'))} | {fmt(r.get('mean_job_gain'), 1)} |")
    lines.extend(["", "**Interpretation:** unlimited greedy is a feasible constructive reference, not a proven upper bound. "
             "The stress-free scenario is hypothetical and is not constrained by eligible interventions."])
    (out / "COMPARISON.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed: CSV/Markdown created, plots skipped")
    else:
        plot = frame.loc[frame.bike_score.notna()].copy()
        fig, ax = plt.subplots(figsize=(11, max(4, len(plot)*.5)))
        plot["gain"] = plot.bike_score - baseline
        bars = ax.barh(plot.experiment, plot.gain)
        ax.invert_yaxis()
        ax.axvline(0, color="grey", lw=.8)
        ax.set_xlabel("Change in population-weighted bicycle fitness from Trial 03 baseline")
        ax.set_title("Bicycle intervention potential by feasible budget")
        for bar, value in zip(bars, plot.gain):
            ax.text(value + 0.0007, bar.get_y()+bar.get_height()/2, f"{value:+.4f}", va="center", fontsize=9)
        fig.tight_layout()
        fig.savefig(out / "comparison_bicycle_fitness.png", dpi=180)
        plt.close(fig)
    return frame


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--extreme", type=Path,
                   help="Completed previous Experiment A/C output folder for benchmark lines")
    args = p.parse_args(argv)
    f = summarize(args.out, args.extreme)
    print(f.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print("\nWrote comparison_all_four.csv and COMPARISON.md")


if __name__ == "__main__":
    main()

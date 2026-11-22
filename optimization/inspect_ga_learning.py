"""Read-only offline diagnostic for Operationalizing Transport Justice GA results.

Usage:
  python inspect_ga_learning.py /path/to/GA_Optimization [/path/to/training_dataset_v4]

Reads optimization_state.json, scenario.json and accessibility/fitness.json.
Does NOT run MATSim, change checkpoints, or generate new candidates.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
import statistics
import sys


def load_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return None


def fmt(value, digits=6):
    return "n/a" if value is None else f"{value:.{digits}f}"


def mean_or_none(values):
    return statistics.mean(values) if values else None


def main() -> None:
    if len(sys.argv) not in (2, 3):
        raise SystemExit("Usage: python inspect_ga_learning.py GA_DATASET [HISTORICAL_DATASET]")
    root = Path(sys.argv[1])
    historical = Path(sys.argv[2]) if len(sys.argv) == 3 else None
    state_path = root / "optimization_state.json"
    state = load_json(state_path)
    if state is None:
        raise SystemExit(f"Missing or invalid {state_path}")
    archive = state["archive"]
    fitness_version = state.get("fitness_version")

    baseline_fitness_file = Path(state["baseline"]) / "accessibility" / "fitness.json"
    baseline_payload = load_json(baseline_fitness_file)
    baseline = None
    if baseline_payload and baseline_payload.get("fitness_version") == fitness_version:
        baseline = baseline_payload.get("fitness")
    elif baseline_payload:
        print("WARNING baseline fitness version differs; baseline comparison omitted")

    rows = []
    missing_scores = 0
    for key, entry in archive.items():
        fitness = entry.get("fitness")
        if fitness is None:
            continue
        source = entry.get("source", "unknown")
        path = None
        if entry.get("scenario"):
            path = root / entry["scenario"]
        elif historical is not None and entry.get("historical_scenario"):
            path = historical / entry["historical_scenario"]
        meta = load_json(path / "scenario.json") if path else None
        scored = load_json(path / "accessibility" / "fitness.json") if path else None
        parts = {}
        if scored and scored.get("fitness_version") == fitness_version:
            parts = {x["mode"]: x["mode_score"] for x in scored.get("components", [])}
        elif fitness >= 0:
            missing_scores += 1
        budget = meta.get("budget_used_m") if meta else None
        if budget is None and path:
            interventions = load_json(path / "interventions.json")
            if isinstance(interventions, list) and all("length_m" in x for x in interventions):
                budget = sum(float(x["length_m"]) for x in interventions)
        rows.append({
            "key": key,
            "source": source,
            "generation": int(entry.get("created_generation", 0)),
            "fitness": float(fitness),
            "budget": budget,
            "parts": parts,
            "interventions": entry.get("interventions", []),
            "scenario": entry.get("scenario") or entry.get("historical_scenario") or "(unknown)",
        })

    valid = [x for x in rows if x["fitness"] >= 0]
    ga = [x for x in rows if x["source"] == "ga"]
    ga_valid = [x for x in ga if x["fitness"] >= 0]
    historical_valid = [x for x in valid if x["source"] == "historical"]
    print(f"GA checkpoint: {state_path}")
    print(f"Fitness version: {fitness_version}")
    print(f"Baseline fitness: {fmt(baseline)}")
    print(f"Archive: {len(archive)} configurations; evaluated={len(rows)}; valid={len(valid)}; failed={sum(x['fitness'] < 0 for x in rows)}")
    print(f"Warm-start valid: {len(historical_valid)}; GA-created valid: {len(ga_valid)}; GA-created failed: {sum(x['fitness'] < 0 for x in ga)}")
    print(f"Best warm start: {fmt(max((x['fitness'] for x in historical_valid), default=None))}")
    print(f"Best new GA:   {fmt(max((x['fitness'] for x in ga_valid), default=None))}")
    print(f"Missing matching per-mode fitness.json: {missing_scores} valid entries")
    if baseline is not None:
        print(f"GA-created valid beating baseline: {sum(x['fitness'] > baseline for x in ga_valid)} / {len(ga_valid)}")

    groups = defaultdict(list)
    for x in ga:
        groups[x["generation"]].append(x)
    print("\nNEW GA CANDIDATES BY GENERATION (not elites/reused individuals)")
    print("gen  created  valid  failed   mean_valid    best_valid    median_valid")
    for gen, entries in sorted(groups.items()):
        successful = [x["fitness"] for x in entries if x["fitness"] >= 0]
        nfailed = sum(x["fitness"] < 0 for x in entries)
        print(f"{gen:>3}  {len(entries):>7}  {len(successful):>5}  {nfailed:>6}   "
              f"{fmt(mean_or_none(successful)):>10}    {fmt(max(successful) if successful else None):>10}    "
              f"{fmt(statistics.median(successful) if successful else None):>10}")

    print("\nTOP 12 SUCCESSFUL CONFIGURATIONS (all origins)")
    print("rank  source      gen   fitness   budget_km  n_interventions  car       bike       pt        scenario")
    for idx, x in enumerate(sorted(valid, key=lambda x: x["fitness"], reverse=True)[:12], 1):
        km = "  n/a" if x["budget"] is None else f"{float(x['budget'])/1000:>6.2f}"
        p = x["parts"]
        print(f"{idx:>4}  {x['source']:<10}  {x['generation']:>3}  {x['fitness']:.6f}     {km}   "
              f"{len(x['interventions']):>14}  {fmt(p.get('car'), 4):>8}  {fmt(p.get('bike'), 4):>8}  "
              f"{fmt(p.get('pt'), 4):>8}  {x['scenario']}")

    print("\nGA VALID FITNESS BY USED BUDGET")
    buckets = [(0, 3000), (3000, 7000), (7000, 11000), (11000, 15000), (15000, float('inf'))]
    for lo, hi in buckets:
        selected = [x["fitness"] for x in ga_valid
                    if x["budget"] is not None and lo <= float(x["budget"]) < hi]
        if selected:
            label = f"{lo/1000:g}-{hi/1000:g}km" if hi != float('inf') else ">=15km"
            print(f"{label:>8}: n={len(selected):>4}, mean={statistics.mean(selected):.6f}, "
                  f"best={max(selected):.6f}")

    scored = [x for x in ga_valid if all(m in x["parts"] for m in ("car", "bike", "pt"))]
    print("\nPER-MODE FITNESS SPREAD, GA-CREATED VALID SCENARIOS")
    for mode in ("car", "bike", "pt"):
        values = [x["parts"][mode] for x in scored]
        if values:
            print(f"{mode:>4}: n={len(values):>4}, mean={statistics.mean(values):.6f}, "
                  f"min={min(values):.6f}, max={max(values):.6f}, "
                  f"stdev={statistics.pstdev(values):.6f}")
    if baseline_payload and baseline_payload.get("fitness_version") == fitness_version:
        components = {x["mode"]: x["mode_score"] for x in baseline_payload.get("components", [])}
        print("Baseline mode scores: " + ", ".join(f"{k}={fmt(components.get(k))}" for k in ("car", "bike", "pt")))

    if len(scored) >= 20:
        ranked = sorted(scored, key=lambda x: x["fitness"])
        n = max(10, len(scored) // 10)
        print("\nINTERVENTIONS PER SCENARIO: BOTTOM vs TOP FITNESS DECILE (GA-created valid)")
        for title, group in (("Bottom 10%", ranked[:n]), ("Top 10%", ranked[-n:])):
            cnt = Counter(iv.get("intervention_type", "unknown") for x in group for iv in x["interventions"])
            meanbudget = mean_or_none([float(x["budget"])/1000 for x in group if x["budget"] is not None])
            print(title + f": {len(group)} scenarios; mean budget={fmt(meanbudget, 2)}km; "
                  f"mean genes={statistics.mean(len(x['interventions']) for x in group):.1f}")
            print("    mean interventions per scenario: " + ", ".join(
                f"{key}={cnt[key]/len(group):.1f}" for key in sorted(cnt)))


if __name__ == "__main__":
    main()

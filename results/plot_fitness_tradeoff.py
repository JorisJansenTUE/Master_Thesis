#!/usr/bin/env python3
"""Visualize car vs (bike+PT) trade-offs using existing STATPOP fitness chunks.

Usage:
 python plot_fitness_tradeoff.py --trial /path/to/fitness_experiments/trial_01

Reads experiment.json and chunks/chunk_*.json. Uses Python stdlib only;
requires no MATSim, routing, pandas, matplotlib, or additional installations.
Does not alter scored scenarios. Writes SVG, CSV, and Markdown under trial/tradeoff.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from html import escape
from pathlib import Path

MODES = ("car", "bike", "pt")
COLORS = {
    "GA (warm start)": "#7051a6",
    "GA (new seed)": "#c06b21",
    "Training: cluster": "#557da6",
    "Training: corridor": "#46a397",
    "Training: random": "#a4a7b6",
    "Other": "#7b8490",
}


def classify(source: str, scenario: str) -> str:
    path = source.lower()
    name = scenario.lower()
    if "ga_20min_park5_15km_seed1234_no_warm" in path:
        return "GA (new seed)"
    if "ga_20min_park5_15km_run01" in path:
        return "GA (warm start)"
    if name.startswith("cluster_"):
        return "Training: cluster"
    if name.startswith("corridor_"):
        return "Training: corridor"
    if name.startswith("random_"):
        return "Training: random"
    return "Other"


def component_scores(obj: dict, cutoff: float = 20.0) -> dict:
    found = {}
    for component in obj["components"]:
        mode = component["mode"]
        if mode in MODES and abs(float(component["time_min"]) - cutoff) < 1e-8:
            if mode in found:
                raise ValueError(f"Duplicate component: {mode}")
            found[mode] = float(component["score"])
    if set(found) != set(MODES):
        raise ValueError(f"Expected car/bike/pt at {cutoff}min; found {found}")
    if not all(math.isfinite(v) and 0 <= v <= 1+1e-9 for v in found.values()):
        raise ValueError(f"Invalid modal scores: {found}")
    return found


def read_trial(folder: Path) -> tuple[list[dict], dict, dict, int]:
    meta = json.loads((folder / "experiment.json").read_text(encoding="utf-8"))
    cfg = meta["config"]
    # The interpretation of the diagonal depends on these exact settings.
    weights = {k:float(v) for k,v in cfg["MODE_WEIGHTS"].items()}
    times = {float(k):float(v) for k,v in cfg["TIME_WEIGHTS"].items()}
    if (weights != {"car":1.0,"bike":1.0,"pt":1.0}
            or times != {20.0:1.0}
            or float(cfg["SUFFICIENCY_FRACTION"]) != 0.5
            or float(cfg["SHORTFALL_EXPONENT"]) != 2.0
            or bool(cfg["CAR_CONSTRAINT_ENABLED"])):
        raise ValueError("This plot is specifically for Trial 01: 20min, low threshold, "
                         "quadratic, equal car/bike/PT weights, and NO car constraint. "
                         "Use a matching experiment rather than mislabeling the objective.")
    base = component_scores(meta["baseline_score"])
    base_total = float(meta["baseline_score"]["raw_fitness"])
    if not math.isclose(base_total, sum(base.values())/3, abs_tol=1e-8):
        raise ValueError("Baseline component scores do not reproduce the baseline fitness")
    n_chunks = int(meta["n_chunks"])
    rows = []
    error_count = 0
    seen = set()
    for index in range(n_chunks):
        file = folder / "chunks" / f"chunk_{index:05d}.json"
        if not file.is_file():
            raise FileNotFoundError(f"Missing chunk {file}; finalize the experiment first")
        block = json.loads(file.read_text(encoding="utf-8"))
        error_count += len(block["errors"])
        for item in block["completed"]:
            s = component_scores(item)
            delta = {m: s[m]-base[m] for m in MODES}
            sustained = (s["bike"]+s["pt"])/2
            delta_sustained = (delta["bike"]+delta["pt"])/2
            delta_total = (delta["car"]+2*delta_sustained)/3
            if not math.isclose(float(item["raw_fitness"]), sum(s.values())/3, abs_tol=2e-8):
                raise ValueError(f"Objective mismatch for {item['scenario']}")
            if not math.isclose(float(item["raw_fitness"])-base_total, delta_total, abs_tol=2e-8):
                raise ValueError(f"Delta mismatch for {item['scenario']}")
            key = str(item.get("key") or item["source"])
            if key in seen:
                raise ValueError(f"Duplicate scenario key: {key}")
            seen.add(key)
            rows.append({
                "key": key, "scenario": str(item["scenario"]),
                "source": str(item["source"]),
                "group": classify(str(item["source"]), str(item["scenario"])),
                "car_score": s["car"], "bike_score":s["bike"], "pt_score":s["pt"],
                "sustainable_score": sustained, "total_fitness": float(item["raw_fitness"]),
                "delta_car_pp": 100*delta["car"],
                "delta_bike_pp": 100*delta["bike"],
                "delta_pt_pp": 100*delta["pt"],
                "delta_sustainable_pp": 100*delta_sustained,
                "delta_total_pp":100*delta_total,
            })
    if len(rows)+error_count != int(meta["n_scenarios"]):
        raise ValueError("Mismatched number of completed and failed scenarios")
    return rows, base, meta, error_count


def pareto_indices(rows: list[dict]) -> list[int]:
    """Undominated points in the (car delta, sustainable delta) plane.

    Compare against baseline as a genuine additional option (0,0).
    Keep at most one representative for exact ties.
    """
    ranked = sorted(range(len(rows)), key=lambda i:
                    (-rows[i]["delta_car_pp"], -rows[i]["delta_sustainable_pp"], rows[i]["key"]))
    frontier = []
    # Treat baseline as a candidate, but only where higher-car candidates didn't already dominate it.
    if ranked:
        best_y = -math.inf
    for i in ranked:
        x = rows[i]["delta_car_pp"]
        y = rows[i]["delta_sustainable_pp"]
        if y > best_y + 1e-10:
            frontier.append(i)
            best_y = y
    # Remove any points dominated by baseline, even if they survived the scan.
    frontier = [i for i in frontier if not (rows[i]["delta_car_pp"] <= 0 and rows[i]["delta_sustainable_pp"] <= 0)]
    return sorted(frontier, key=lambda i:rows[i]["delta_car_pp"])


def write_csv(path: Path, records: list[dict]) -> None:
    if not records:
        return
    keys = list(records[0])
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(records)


def fmt_tick(x: float) -> str:
    return f"{x:+.2f}" if abs(x) < 1 and x != 0 else f"{x:+.1f}" if x != 0 else "0"


def ticks(lo: float, hi:float, approx:int = 7) -> list[float]:
    step_raw = max((hi-lo)/approx, 1e-9)
    decade = 10**math.floor(math.log10(step_raw))
    step = min([1,2,2.5,5,10], key=lambda k: abs(k*decade-step_raw))*decade
    start = math.ceil(lo/step-1e-10)
    end = math.floor(hi/step+1e-10)
    return [v*step for v in range(start,end+1)]


def nice_bounds(values:list[float], include_zero:bool = True) -> tuple[float,float]:
    lo=min(values);hi=max(values)
    if include_zero:
        lo=min(lo,0);hi=max(hi,0)
    if hi-lo < 1e-8:
        lo-=.1;hi+=.1
    p=.055*(hi-lo)
    return lo-p, hi+p


def plot_svg(rows:list[dict], indices:list[int], frontier: list[int],
             best_idx:int, best_sust_idx:int, filename:Path,
             title:str, subtitle:str, include_baseline:bool=True):
    W,H=1200,780
    left,right,top,bottom=110,855,90,620
    panel_w=right-left;panel_h=bottom-top
    subset=[rows[i] for i in indices]
    xs=[r["delta_car_pp"] for r in subset]
    ys=[r["delta_sustainable_pp"] for r in subset]
    xmin,xmax=nice_bounds(xs,include_baseline)
    ymin,ymax=nice_bounds(ys,include_baseline)
    px=lambda x:left+(x-xmin)/(xmax-xmin)*panel_w
    py=lambda y:bottom-(y-ymin)/(ymax-ymin)*panel_h
    out=[]
    def add(s):out.append(s)
    add(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">')
    add('<defs><style>text{font-family:Arial,Helvetica,sans-serif;}</style>')
    add(f'<clipPath id="plotClip"><rect x="{left}" y="{top}" width="{panel_w}" height="{panel_h}"/></clipPath></defs>')
    add('<rect width="100%" height="100%" fill="#ffffff"/>')
    add(f'<text x="{left}" y="35" fill="#202b3c" font-size="23" font-weight="700">{escape(title)}</text>')
    add(f'<text x="{left}" y="60" fill="#687383" font-size="12">{escape(subtitle)}</text>')
    add(f'<rect x="{left}" y="{top}" width="{panel_w}" height="{panel_h}" fill="#fafbfc" stroke="#bec6ce"/>')
    for x in ticks(xmin,xmax):
        xx=px(x)
        add(f'<line x1="{xx:.2f}" y1="{top}" x2="{xx:.2f}" y2="{bottom}" stroke="#e1e6eb" stroke-width="1"/>')
        add(f'<text x="{xx:.2f}" y="{bottom+23}" text-anchor="middle" fill="#526073" font-size="11">{fmt_tick(x)}</text>')
    for y in ticks(ymin,ymax):
        yy=py(y)
        add(f'<line x1="{left}" y1="{yy:.2f}" x2="{right}" y2="{yy:.2f}" stroke="#e1e6eb" stroke-width="1"/>')
        add(f'<text x="{left-12}" y="{yy+4:.2f}" text-anchor="end" fill="#526073" font-size="11">{fmt_tick(y)}</text>')
    add(f'<text x="{(left+right)/2}" y="{bottom+64}" text-anchor="middle" fill="#263447" font-size="15" font-weight="600">Change in car fitness (score percentage points)</text>')
    add(f'<text x="25" y="{(top+bottom)/2}" text-anchor="middle" fill="#263447" font-size="15" font-weight="600" transform="rotate(-90 25 {(top+bottom)/2})">Change in mean bike + PT fitness (score percentage points)</text>')
    add('<g clip-path="url(#plotClip)">')
    # Equal-weight combined fitness is positive above y=-x/2.
    add(f'<path d="M {px(xmin):.2f},{py(-xmin/2):.2f} L {px(xmax):.2f},{py(-xmax/2):.2f}" stroke="#bf7538" stroke-width="1.7" stroke-dasharray="6 5" fill="none"/>')
    if xmin<=0<=xmax:
        xx=px(0);add(f'<path d="M {xx:.2f},{top} L {xx:.2f},{bottom}" stroke="#7f8a98" stroke-dasharray="4 5" stroke-width="1.3"/>')
    if ymin<=0<=ymax:
        yy=py(0);add(f'<path d="M {left},{yy:.2f} L {right},{yy:.2f}" stroke="#7f8a98" stroke-dasharray="4 5" stroke-width="1.3"/>')
    # Training underneath GA points. Points have hover tooltips in browsers.
    for i in sorted(indices,key=lambda i:(0 if rows[i]['group'].startswith('Training') else 1, rows[i]['group'])):
        r=rows[i]
        label=escape(f"{r['scenario']} | {r['group']} | car {r['delta_car_pp']:+.4f} pp | bike/PT {r['delta_sustainable_pp']:+.4f} pp | all-mode {r['delta_total_pp']:+.4f} pp")
        add(f'<circle cx="{px(r["delta_car_pp"]):.2f}" cy="{py(r["delta_sustainable_pp"]):.2f}" r="2.3" fill="{COLORS[r["group"]]}" fill-opacity="0.36"><title>{label}</title></circle>')
    view_frontier=[i for i in frontier if i in set(indices)]
    if len(view_frontier)>=2:
        d=' '.join(('M' if j==0 else 'L')+f' {px(rows[i]["delta_car_pp"]):.2f} {py(rows[i]["delta_sustainable_pp"]):.2f}' for j,i in enumerate(view_frontier))
        add(f'<path d="{d}" stroke="#1c7d66" stroke-width="2.3" fill="none"/>')
    for i in view_frontier:
        r=rows[i];add(f'<circle cx="{px(r["delta_car_pp"]):.2f}" cy="{py(r["delta_sustainable_pp"]):.2f}" r="3" stroke="#1c7d66" stroke-width="1.4" fill="#ffffff"/>')
    if include_baseline:
        add(f'<circle cx="{px(0):.2f}" cy="{py(0):.2f}" r="6" fill="#fff" stroke="#262c33" stroke-width="2.1"/>')
    for i,label,color in [(best_idx,"Best all-mode",'#b51d3b'),(best_sust_idx,"Best bike/PT",'#2b6777')]:
        if i not in indices:continue
        r=rows[i];xx=px(r["delta_car_pp"]);yy=py(r["delta_sustainable_pp"])
        add(f'<circle cx="{xx:.2f}" cy="{yy:.2f}" r="7.3" fill="{color}" stroke="white" stroke-width="2"><title>{escape(label)}: {escape(r["scenario"])}</title></circle>')
    add('</g>')
    add('<text x="902" y="106" font-size="15" font-weight="700" fill="#263447">Scenario sources</text>')
    offset=136
    for j,(name,color) in enumerate(COLORS.items()):
        add(f'<circle cx="908" cy="{offset+j*26}" r="5" fill="{color}"/>')
        add(f'<text x="924" y="{offset+j*26+4}" font-size="12" fill="#38465a">{escape(name)}</text>')
    y0=329
    add(f'<line x1="904" y1="{y0-11}" x2="949" y2="{y0-11}" stroke="#1c7d66" stroke-width="2.3"/>')
    add(f'<text x="958" y="{y0-7}" font-size="12" fill="#38465a">Pareto frontier</text>')
    add(f'<line x1="904" y1="{y0+19}" x2="949" y2="{y0+19}" stroke="#bf7538" stroke-dasharray="6 5" stroke-width="2"/>')
    add(f'<text x="958" y="{y0+23}" font-size="12" fill="#38465a">No net fitness change</text>')
    add(f'<circle cx="916" cy="{y0+53}" r="6" fill="#b51d3b" stroke="white" stroke-width="1"/>')
    add(f'<text x="933" y="{y0+57}" font-size="12" fill="#38465a">Best Trial 01 F</text>')
    add(f'<circle cx="916" cy="{y0+82}" r="6" fill="#2b6777" stroke="white" stroke-width="1"/>')
    add(f'<text x="933" y="{y0+86}" font-size="12" fill="#38465a">Best bike+PT F</text>')
    add(f'<circle cx="916" cy="{y0+111}" r="6" fill="#fff" stroke="#262c33" stroke-width="2"/>')
    add(f'<text x="933" y="{y0+115}" font-size="12" fill="#38465a">Baseline (0, 0)</text>')
    add(f'<text x="904" y="{y0+160}" font-size="13" font-weight="700" fill="#263447">Interpretation</text>')
    for j,t in enumerate(["Upper left: sustainable gain,", "but car fitness declines.", "Above dashed diagonal:", "combined Trial 01 F improves.", "Car scores can conceal losses", "above sufficiency threshold."]):
        add(f'<text x="904" y="{y0+184+j*20}" font-size="12" fill="#526073">{escape(t)}</text>')
    add(f'<text x="{left}" y="{H-35}" font-size="11" fill="#627086">Source: existing cached fitness scenarios; identical 20-min cutoff and 50% sufficiency threshold. Scores are dimensionless.</text>')
    add('</svg>')
    filename.write_text('\n'.join(out),encoding='utf-8')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--trial',type=Path,required=True,help='Path to existing fitness_experiments/trial_01')
    ap.add_argument('--out',type=Path,default=None,help='Output directory (default: TRIAL/tradeoff)')
    args=ap.parse_args()
    folder=args.trial.resolve(strict=True)
    out=(args.out or folder/'tradeoff').resolve()
    out.mkdir(parents=True,exist_ok=True)
    rows,base,meta,n_failed=read_trial(folder)
    if not rows: raise ValueError('No scored scenarios found')
    rows.sort(key=lambda r:r['delta_total_pp'],reverse=True)
    front=pareto_indices(rows)
    best_idx=max(range(len(rows)),key=lambda i:rows[i]['delta_total_pp'])
    best_sust_idx=max(range(len(rows)),key=lambda i:rows[i]['delta_sustainable_pp'])
    frontset={rows[i]['key'] for i in front}
    for r in rows:r['pareto_frontier']=r['key'] in frontset
    write_csv(out/'all_scenario_tradeoffs.csv',rows)
    write_csv(out/'pareto_frontier.csv',[rows[i] for i in front])
    write_csv(out/'top_25_combined.csv',rows[:25])
    # Zoom focuses on 10% highest combined fitness, permitting a close look at the selection frontier.
    n_top=min(len(rows),max(15,math.ceil(.1*len(rows))))
    top_indices=list(range(n_top))
    groups=Counter(r['group'] for r in rows)
    above=sum(r['delta_total_pp']>1e-9 for r in rows)
    quadrants={
        'sustainable_gain_car_loss': sum(r['delta_sustainable_pp']>0 and r['delta_car_pp']<0 for r in rows),
        'both_gain':sum(r['delta_sustainable_pp']>0 and r['delta_car_pp']>0 for r in rows),
        'both_loss':sum(r['delta_sustainable_pp']<0 and r['delta_car_pp']<0 for r in rows),
        'car_gain_sustainable_loss':sum(r['delta_sustainable_pp']<0 and r['delta_car_pp']>0 for r in rows),
    }
    plot_svg(rows,list(range(len(rows))),front,best_idx,best_sust_idx,
             out/'car_vs_sustainable_all.svg',
             'Car versus sustainable accessibility | All scenarios',
             f'{len(rows):,} evaluated | {n_failed} incompatible caches excluded | {len(front)} Pareto-optimal scenarios')
    plot_svg(rows,top_indices,front,best_idx,best_sust_idx,
             out/'car_vs_sustainable_top10pct.svg',
             'Car versus sustainable accessibility | Best 10% of scenarios',
             f'Top {n_top:,} ranked by the equal-weight Trial 01 objective | Pareto frontier highlighted',
             include_baseline=False)
    winner=rows[best_idx];best_sust=rows[best_sust_idx]
    report=[
        '# Car / bicycle + PT trade-off | Trial 01','',
        '**Objective:** 20-minute population-weighted sufficiency; common low (50%) threshold; squared shortfall; equal weights for car, bicycle and PT; no car constraint.','',
        f'**Analyzed scenarios:** {len(rows):,}; excluded/failed: {n_failed:,}.',
        f'**Scenarios improving combined F:** {above:,} ({100*above/len(rows):.1f}%).',
        f'**Pareto frontier:** {len(front)} scenarios (computed in low-threshold modal fitness, with baseline as a possible comparator).','',
        '| Baseline modal score | Score |','|---|---:|',
        *[f'| {mode} | {base[mode]:.6f} |' for mode in MODES],'',
        '| Candidate | Scenario | Δcar (score-pp) | Δbike+PT (score-pp) | Δall (score-pp) |',
        '|---|---|---:|---:|---:|',
        f'| Best combined | `{winner["scenario"]}` | {winner["delta_car_pp"]:+.4f} | {winner["delta_sustainable_pp"]:+.4f} | {winner["delta_total_pp"]:+.4f} |',
        f'| Best bike+PT | `{best_sust["scenario"]}` | {best_sust["delta_car_pp"]:+.4f} | {best_sust["delta_sustainable_pp"]:+.4f} | {best_sust["delta_total_pp"]:+.4f} |','',
        '## Scenario outcome quadrants','',
        *[f'- {name.replace("_"," ")}: {n:,}' for name,n in quadrants.items()],'',
        '## Sources','',
        *[f'- {name}: {groups[name]:,}' for name in COLORS if groups[name]],'',
        '## Interpreting the graph','',
        '- Horizontal: car score change; vertical: average bike/PT score change; both in score percentage points.','- Dashed diagonal is ΔF = 0: a point above the line improves the equally weighted combined objective.','- Upper left indicates the policy trade-off: improvements for bicycle/PT despite lower car scores.','- A nearly unchanged car *score* does **not** imply negligible losses in car accessible jobs. Car utility saturates above the sufficiency threshold.','- Pareto optimality is relative only to the evaluated scenarios, not to all physically possible interventions.','- The 319 excluded scenarios are not assessed; no MATSim simulations or accessibility routing is performed.',''
    ]
    (out/'SUMMARY.md').write_text('\n'.join(report),encoding='utf-8')
    print(f'Trial 01 trade-off: {len(rows)} scored; {n_failed} failed/excluded; {len(front)} Pareto-frontier points')
    print(f'Improved combined score: {above}/{len(rows)} ({100*above/len(rows):.1f}%)')
    print(f'Best combined: {winner["scenario"]}, Δcar={winner["delta_car_pp"]:+.4f} pp, Δbike+PT={winner["delta_sustainable_pp"]:+.4f} pp, ΔF={winner["delta_total_pp"]:+.4f} pp')
    print(f'Best bike+PT: {best_sust["scenario"]}, Δcar={best_sust["delta_car_pp"]:+.4f} pp, Δbike+PT={best_sust["delta_sustainable_pp"]:+.4f} pp, ΔF={best_sust["delta_total_pp"]:+.4f} pp')
    print(f'Wrote: {out}')

if __name__=='__main__':main()

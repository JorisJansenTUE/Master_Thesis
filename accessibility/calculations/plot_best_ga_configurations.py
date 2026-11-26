#!/usr/bin/env python3
"""Plot the best intervention configuration from two GA optimization runs.

Expected inputs:
- two optimization_state.json files (warm and no-warm runs)
- baseline MATSim network.xml.gz used to derive node coordinates

The script identifies the best feasible candidate from each run (preferring GA-generated
candidates when available), reconstructs each intervention's geometry from its segment_id
("nodeA__nodeB"), and creates a side-by-side map with colors by intervention type.

This is intended for the comparative GA analysis in the thesis.
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

COLORS = {
    "bicycle_street": "#1f78b4",
    "protected_bike_lane": "#33a02c",
    "dedicated_bus_lane": "#e31a1c",
    "destination_only": "#ff7f00",
}
ORDER = [
    "bicycle_street",
    "protected_bike_lane",
    "dedicated_bus_lane",
    "destination_only",
]
LABELS = {
    "bicycle_street": "Bicycle street",
    "protected_bike_lane": "Protected bike lane",
    "dedicated_bus_lane": "Dedicated bus lane",
    "destination_only": "Destination only",
}


def _open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, "rt", encoding="utf-8")


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_network_nodes(network_path: Path) -> Dict[str, Tuple[float, float]]:
    """Parse MATSim network and return node_id -> (x, y)."""
    with _open_text(network_path) as f:
        tree = ET.parse(f)
    root = tree.getroot()
    nodes = {}
    # MATSim structure: <network><nodes><node id="..." x="..." y="..."/></nodes>...
    for node in root.findall(".//node"):
        nid = node.attrib["id"]
        nodes[nid] = (float(node.attrib["x"]), float(node.attrib["y"]))
    if not nodes:
        raise RuntimeError(f"No nodes found in network: {network_path}")
    return nodes


def archive_records(state: dict) -> Iterable[dict]:
    arch = state.get("archive", {})
    if isinstance(arch, dict):
        return arch.values()
    if isinstance(arch, list):
        return arch
    raise TypeError(f"Unsupported archive type: {type(arch)!r}")


def candidate_sort_key(rec: dict):
    feasible = bool(rec.get("feasible", rec.get("car_constraint", {}).get("feasible", True)))
    fitness = rec.get("fitness", -math.inf)
    generated = rec.get("scenario") is not None or rec.get("source") == "ga"
    gen = rec.get("created_generation", -1)
    return (1 if feasible else 0, 1 if generated else 0, fitness, gen)


def choose_best_candidate(state: dict, require_generated: bool = True) -> dict:
    recs = list(archive_records(state))
    feasible_generated = [
        r for r in recs
        if (r.get("scenario") is not None or r.get("source") == "ga")
        and bool(r.get("feasible", r.get("car_constraint", {}).get("feasible", True)))
    ]
    if require_generated and feasible_generated:
        return max(feasible_generated, key=candidate_sort_key)
    feasible_any = [r for r in recs if bool(r.get("feasible", r.get("car_constraint", {}).get("feasible", True)))]
    if feasible_any:
        return max(feasible_any, key=candidate_sort_key)
    return max(recs, key=candidate_sort_key)


def intervention_segments(candidate: dict, node_xy: Dict[str, Tuple[float, float]]):
    by_type: Dict[str, List[List[Tuple[float, float]]]] = {k: [] for k in ORDER}
    skipped = []
    for item in candidate.get("interventions", []):
        itype = item.get("intervention_type")
        if itype not in by_type:
            continue
        seg_id = item.get("segment_id", "")
        if "__" not in seg_id:
            skipped.append(seg_id)
            continue
        a, b = seg_id.split("__", 1)
        if a not in node_xy or b not in node_xy:
            skipped.append(seg_id)
            continue
        by_type[itype].append([node_xy[a], node_xy[b]])
    return by_type, skipped


def candidate_summary(candidate: dict) -> str:
    counts = Counter([i.get("intervention_type", "?") for i in candidate.get("interventions", [])])
    lines = [
        f"fitness = {candidate.get('fitness', float('nan')):.6f}",
        f"created_generation = {candidate.get('created_generation', 'NA')}",
        f"source = {candidate.get('source', 'NA')}",
        f"scenario = {candidate.get('scenario', 'NA')}",
    ]
    lines.extend(f"{LABELS[k]}: {counts.get(k,0)}" for k in ORDER)
    return "\n".join(lines)


def plot_run(ax, by_type, title: str, summary: str, background_segments=None):
    if background_segments:
        bg = LineCollection(background_segments, colors="#d9d9d9", linewidths=0.4, alpha=0.45, zorder=1)
        ax.add_collection(bg)
    for itype in ORDER:
        segs = by_type.get(itype, [])
        if not segs:
            continue
        lw = 1.5 if itype in ("bicycle_street", "destination_only") else 2.2
        lc = LineCollection(segs, colors=COLORS[itype], linewidths=lw, alpha=0.95, zorder=3)
        ax.add_collection(lc)
    ax.set_title(title, fontsize=12)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.text(
        0.01, 0.99, summary,
        transform=ax.transAxes, va="top", ha="left", fontsize=8,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#bdbdbd", alpha=0.9),
        zorder=10,
    )


def collect_all_segments(node_xy: Dict[str, Tuple[float, float]], stride: int = 15):
    # lightweight gray context network by connecting all link endpoints is possible,
    # but parsing links would require more XML work. We instead skip background by default.
    return None


def set_shared_extent(axs, all_segments, pad_frac=0.04):
    xs, ys = [], []
    for segs in all_segments:
        for line in segs:
            for x, y in line:
                xs.append(x)
                ys.append(y)
    if not xs:
        return
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    dx = xmax - xmin
    dy = ymax - ymin
    padx = dx * pad_frac if dx > 0 else 100
    pady = dy * pad_frac if dy > 0 else 100
    for ax in axs:
        ax.set_xlim(xmin - padx, xmax + padx)
        ax.set_ylim(ymin - pady, ymax + pady)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--warm-state", required=True, type=Path)
    ap.add_argument("--no-warm-state", required=True, type=Path)
    ap.add_argument("--network", required=True, type=Path,
                    help="Baseline MATSim network.xml.gz with node coordinates")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--title", default="Best GA configurations by intervention type")
    ap.add_argument("--allow-historical", action="store_true",
                    help="Allow a historical warm-start candidate if it remains the best")
    args = ap.parse_args()

    warm_state = load_json(args.warm_state)
    no_warm_state = load_json(args.no_warm_state)
    node_xy = load_network_nodes(args.network)

    warm_best = choose_best_candidate(warm_state, require_generated=not args.allow_historical)
    no_warm_best = choose_best_candidate(no_warm_state, require_generated=not args.allow_historical)

    warm_by_type, warm_skipped = intervention_segments(warm_best, node_xy)
    now_by_type, now_skipped = intervention_segments(no_warm_best, node_xy)

    fig, axs = plt.subplots(1, 2, figsize=(15, 8), constrained_layout=True)
    bg = None

    plot_run(axs[0], warm_by_type, "Warm start GA — best candidate", candidate_summary(warm_best), background_segments=bg)
    plot_run(axs[1], now_by_type, "No-warm GA — best candidate", candidate_summary(no_warm_best), background_segments=bg)

    all_plotted = []
    for bt in (warm_by_type, now_by_type):
        for k in ORDER:
            all_plotted.extend(bt.get(k, []))
    set_shared_extent(axs, [all_plotted])

    handles = [Line2D([0], [0], color=COLORS[k], lw=3, label=LABELS[k]) for k in ORDER]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=10)
    fig.suptitle(args.title, fontsize=15)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=250, bbox_inches="tight")

    # Save a sidecar summary JSON for reproducibility
    sidecar = args.out.with_suffix('.json')
    with open(sidecar, 'w', encoding='utf-8') as f:
        json.dump({
            'warm_best': {
                'fitness': warm_best.get('fitness'),
                'created_generation': warm_best.get('created_generation'),
                'source': warm_best.get('source'),
                'scenario': warm_best.get('scenario'),
                'skipped_segment_ids': warm_skipped,
            },
            'no_warm_best': {
                'fitness': no_warm_best.get('fitness'),
                'created_generation': no_warm_best.get('created_generation'),
                'source': no_warm_best.get('source'),
                'scenario': no_warm_best.get('scenario'),
                'skipped_segment_ids': now_skipped,
            }
        }, f, indent=2)

    print(f"Saved figure: {args.out}")
    print(f"Saved metadata: {sidecar}")
    if warm_skipped or now_skipped:
        print(f"Skipped warm segments: {len(warm_skipped)}; skipped no-warm segments: {len(now_skipped)}")

if __name__ == '__main__':
    main()

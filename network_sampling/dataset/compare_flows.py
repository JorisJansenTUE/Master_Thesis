"""Compare all MATSim scenario flows with a baseline (24h and full 30h).

Reports both total vehicle-link traversals and the spatial/hourly redistribution.
Uses the stable directed StreetSegment mapping when provided, so new bus *_spec
links and protected-cycle links are compared against their physical road slot.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

try:
    from network_sampling.dataset.compact_outputs import MODES, read_flows
except ImportError:  # allow direct execution in this folder
    from compact_outputs import MODES, read_flows

KEYS = ["segment_id", "from_node", "to_node", "hour"]


def canonical_flows(scenario: Path, flows: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    mapping_path = scenario / "link_mapping.csv.gz"
    if mapping_path.is_file():
        mapping = pd.read_csv(mapping_path, dtype="string")
        if not set(["link_id", *KEYS[:-1]]).issubset(mapping.columns):
            raise ValueError(f"Invalid mapping: {mapping_path}")
        if mapping["link_id"].duplicated().any():
            raise ValueError(f"Duplicate mapping link_id in {mapping_path}")
        joined = flows.merge(mapping[["link_id", *KEYS[:-1]]], on="link_id", how="inner", validate="many_to_one")
        result = joined.groupby(KEYS, as_index=False)[list(MODES)].sum()
        return result, "street_segment_direction"
    # Without mapping, exact link IDs may not be comparable if *_spec links were introduced.
    result = flows.rename(columns={"link_id": "segment_id"}).copy()
    result["from_node"] = ""
    result["to_node"] = ""
    return result[[*KEYS, *MODES]], "raw_link_id"


def load_mode_stats(scenario: Path) -> pd.DataFrame | None:
    path = scenario / "targets" / "mode_statistics.csv"
    if not path.is_file():
        return None
    table = pd.read_csv(path)
    if "main_mode" not in table.columns and "mode" in table.columns:
        table = table.rename(columns={"mode": "main_mode"})
    if not {"main_mode", "trips"}.issubset(table.columns):
        return None
    return table[["main_mode", "trips"]].groupby("main_mode", as_index=False).sum()


def compare(baseline: Path, scenario: Path, b_flows: pd.DataFrame,
            b_canonical: pd.DataFrame, b_basis: str, b_mode_stats: pd.DataFrame | None):
    s_flows = read_flows(scenario)
    s_canonical, s_basis = canonical_flows(scenario, s_flows)
    if b_basis != s_basis:
        # Use link ID comparison only if both datasets lack a shared mapping.
        b_canonical, _ = canonical_flows_without_mapping(b_flows)
        s_canonical, _ = canonical_flows_without_mapping(s_flows)
        basis = "raw_link_id (baseline/scenario mapping mismatch)"
    else:
        basis = b_basis
    joined = b_canonical.merge(s_canonical, on=KEYS, how="outer", suffixes=("_base", "_scenario")).fillna(0)
    sums_b = b_flows.groupby("hour")[list(MODES)].sum()
    sums_s = s_flows.groupby("hour")[list(MODES)].sum()
    hourly = sums_b.join(sums_s, how="outer", lsuffix="_base", rsuffix="_scenario").fillna(0)

    comparison = []
    byhour = []
    for hour, row in hourly.iterrows():
        for mode in MODES:
            before = int(row[f"{mode}_base"])
            after = int(row[f"{mode}_scenario"])
            byhour.append({"scenario": scenario.name, "hour": int(hour), "mode": mode,
                           "baseline": before, "scenario_total": after, "delta": after-before,
                           "delta_pct": 100*(after-before)/before if before else None})

    for window, start, end in (("day_0_23", 0, 24), ("full_0_29", 0, 30)):
        b_in = b_flows.loc[b_flows.hour.between(start, end - 1)]
        s_in = s_flows.loc[s_flows.hour.between(start, end - 1)]
        j_in = joined.loc[joined.hour.between(start, end - 1)]
        for mode in MODES:
            before, after = int(b_in[mode].sum()), int(s_in[mode].sum())
            diff = j_in[f"{mode}_scenario"] - j_in[f"{mode}_base"]
            abs_diff = float(diff.abs().sum())
            changed = j_in.loc[diff != 0, KEYS]
            comparison.append({
                "scenario": scenario.name, "window": window, "mode": mode,
                "baseline_total": before, "scenario_total": after,
                "delta": after-before, "delta_pct": 100*(after-before)/before if before else None,
                "link_hour_L1": int(abs_diff),
                "link_hour_L1_pct_of_baseline": 100*abs_diff/before if before else None,
                "changed_link_hours": len(changed),
                "changed_road_slots": len(changed[KEYS[:-1]].drop_duplicates()),
                "comparison_basis": basis,
            })
    modes = []
    s_mode_stats = load_mode_stats(scenario)
    if b_mode_stats is not None and s_mode_stats is not None:
        joined_modes = b_mode_stats.merge(s_mode_stats, on="main_mode", how="outer",
                                           suffixes=("_base", "_scenario")).fillna(0)
        b_total = float(joined_modes.trips_base.sum())
        s_total = float(joined_modes.trips_scenario.sum())
        for _, row in joined_modes.iterrows():
            modes.append({"scenario": scenario.name, "main_mode": row.main_mode,
                          "baseline_trips": int(row.trips_base), "scenario_trips": int(row.trips_scenario),
                          "delta_trips": int(row.trips_scenario-row.trips_base),
                          "baseline_share": row.trips_base / b_total if b_total else None,
                          "scenario_share": row.trips_scenario / s_total if s_total else None})
    return comparison, byhour, modes


def canonical_flows_without_mapping(flows: pd.DataFrame):
    result = flows.rename(columns={"link_id": "segment_id"}).copy()
    result["from_node"] = ""
    result["to_node"] = ""
    return result[[*KEYS, *MODES]], "raw_link_id"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True, help="Baseline scenario directory, possibly outside dataset")
    parser.add_argument("--out", type=Path, help="Summary CSV (default: DATASET/flow_comparison.csv)")
    args = parser.parse_args()
    dataset = args.dataset.resolve()
    baseline = args.baseline.resolve()
    base_flows = read_flows(baseline)
    b_canonical, b_basis = canonical_flows(baseline, base_flows)
    mode_stats = load_mode_stats(baseline)

    scenarios = sorted(p for p in dataset.iterdir() if p.is_dir() and p.resolve() != baseline
                       and ((p / "targets" / "flows_hourly.csv.gz").is_file() or
                            (p / "raw_output").is_dir()))
    rows, byhour, modes = [], [], []
    failures = []
    for scenario in scenarios:
        try:
            a, b, c = compare(baseline, scenario, base_flows, b_canonical, b_basis, mode_stats)
            rows.extend(a); byhour.extend(b); modes.extend(c)
        except Exception as exc:
            failures.append((scenario.name, str(exc)))
            print(f"SKIP {scenario.name}: {exc}", file=sys.stderr)

    if not rows:
        raise SystemExit("No comparable scenarios found. Check --baseline and completed scenario outputs.")
    dest = args.out or dataset / "flow_comparison.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(rows)
    summary.to_csv(dest, index=False)
    pd.DataFrame(byhour).to_csv(dest.with_name("flow_comparison_by_hour.csv"), index=False)
    if modes:
        pd.DataFrame(modes).to_csv(dest.with_name("mode_comparison.csv"), index=False)
    print(f"Compared {len(scenarios)-len(failures)} scenarios against {baseline.name}.")
    print("Saved:", dest, "/", dest.with_name("flow_comparison_by_hour.csv"))
    if modes:
        print("Saved:", dest.with_name("mode_comparison.csv"))
    day = summary[summary.window == "day_0_23"]
    for name, group in day.groupby("scenario", sort=False):
        changes = ", ".join(f"{r['mode']} {r['delta_pct']:+.1f}% (L1 {r['link_hour_L1_pct_of_baseline']:.1f}%)"
                            for _, r in group.iterrows() if pd.notna(r["delta_pct"]) and pd.notna(r["link_hour_L1_pct_of_baseline"]))
        print(f"  {name}: {changes}")
    if failures:
        print(f"WARNING: {len(failures)} scenarios could not be compared", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()

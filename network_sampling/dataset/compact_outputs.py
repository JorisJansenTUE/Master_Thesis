"""Extract compact MATSim GNN targets and *optionally* remove large raw outputs.

Run this only after simulations have finished. Raw outputs are NEVER deleted unless
--delete-raw is explicitly passed, target extraction succeeds, and run_status.json
reports a successful run.
"""
from __future__ import annotations

import argparse
import gzip
import json
import shutil
import sys
import tempfile
from pathlib import Path

import pandas as pd

MODES = ("cars", "trucks", "bikes", "buses")
SUCCESS_STATUSES = {"complete", "completed", "success", "succeeded"}


def find_flow_file(scenario: Path) -> Path | None:
    # Prefer fresh raw output over earlier extracted targets when rerunning a scenario.
    raw = scenario / "raw_output"
    for name in ("output_detailed_link_statistics.csv.gz", "output_detailed_link_statistics.csv"):
        path = raw / name
        if path.is_file():
            return path
    matches = sorted(raw.glob("*output_detailed_link_statistics.csv*"))
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(f"Multiple possible detailed link-statistics files in {raw}")
    for name in ("flows_hourly.csv.gz", "flows_hourly.csv"):
        path = scenario / "targets" / name
        if path.is_file():
            return path
    return None


def read_flows(scenario: Path) -> pd.DataFrame:
    path = find_flow_file(scenario)
    if path is None:
        raise FileNotFoundError(f"Detailed hourly link statistics missing: {scenario}")
    flows = pd.read_csv(path, dtype={"link_id": "string"})
    required = {"link_id", "hour", *MODES}
    if not required.issubset(flows.columns):
        raise ValueError(f"{path}: missing {sorted(required - set(flows.columns))}")
    if flows.empty or flows[["link_id", "hour", *MODES]].isna().any().any():
        raise ValueError(f"{path}: empty data or null link/hour/flow value")
    if flows.duplicated(["link_id", "hour"]).any():
        raise ValueError(f"{path}: duplicate link_id/hour rows")
    if (flows[[*MODES]] < 0).any().any():
        raise ValueError(f"{path}: negative flow counts")
    if not ((flows["hour"] >= 0) & (flows["hour"] == flows["hour"].astype(int))).all():
        raise ValueError(f"{path}: invalid hour values")
    if not ((flows[[*MODES]] % 1) == 0).all().all():
        raise ValueError(f"{path}: non-integral flow values")
    flows = flows[["link_id", "hour", *MODES]].copy()
    flows["hour"] = flows["hour"].astype("int16")
    for mode in MODES:
        flows[mode] = flows[mode].astype("int64")
    return flows.sort_values(["link_id", "hour"], kind="stable").reset_index(drop=True)


def find_trips(scenario: Path) -> Path | None:
    raw = scenario / "raw_output"
    paths = list(raw.glob("*trips.csv*")) + list((raw / "analysis").glob("*trips.csv*"))
    paths = [p for p in paths if p.is_file()]
    paths.sort(key=lambda p: ("output_trips" not in p.name, len(str(p))))
    return paths[0] if paths else None


def read_trips(path: Path) -> pd.DataFrame:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8-sig") as handle:
        header = handle.readline()
    separator = max((",", ";", "\t"), key=header.count)
    return pd.read_csv(path, sep=separator, low_memory=False)


def extract_trip_modes(scenario: Path, dest: Path) -> dict:
    source = find_trips(scenario)
    if source is None:
        # Allow idempotent use after previous successful extraction.
        existing = scenario / "targets" / "trip_modes.csv.gz"
        existing_stats = scenario / "targets" / "mode_statistics.csv"
        if not (scenario / "raw_output").exists() and existing.is_file() and existing_stats.is_file():
            shutil.copy2(existing, dest / "trip_modes.csv.gz")
            shutil.copy2(existing_stats, dest / "mode_statistics.csv")
            n = len(pd.read_csv(existing))
            return {"source": "previously_extracted", "trips": n}
        raise FileNotFoundError(
            f"No final trip table in {scenario / 'raw_output'}; "
            "cannot reliably calculate PT / other passenger main modes. "
            "Keep raw_output and inspect output_trips.csv.gz or final plans."
        )

    trips = read_trips(source)
    mode_col = next((c for c in ("main_mode", "mainMode", "trip_mode", "mode") if c in trips), None)
    person_col = next((c for c in ("person", "person_id", "personId") if c in trips), None)
    if mode_col is None or person_col is None or trips.empty:
        raise ValueError(f"{source}: need nonempty trip table with person ID and main mode")
    if trips[[mode_col, person_col]].isna().any().any():
        raise ValueError(f"{source}: missing person IDs or main modes")

    optional = [c for c in ("trip_id", "tripId", "trip_number", "dep_time",
                            "start_time", "end_time", "trav_time",
                            "origin_activity_type", "destination_activity_type") if c in trips]
    lean = trips[[person_col, *optional, mode_col]].copy()
    lean = lean.rename(columns={person_col: "person_id", mode_col: "main_mode"})
    lean["person_id"] = lean["person_id"].astype(str)
    lean.to_csv(dest / "trip_modes.csv.gz", index=False, compression="gzip")

    counts = trips[mode_col].astype(str).value_counts().rename_axis("main_mode").reset_index(name="trips")
    counts["share"] = counts["trips"] / len(trips)
    counts.to_csv(dest / "mode_statistics.csv", index=False)
    return {"source": str(source.relative_to(scenario)), "trips": len(trips),
            "mode_counts": dict(zip(counts["main_mode"], counts["trips"].astype(int)))}


def write_dual_hourly(scenario: Path, flows: pd.DataFrame, dest: Path) -> dict:
    """Map newly created *_spec and other links onto stable directed street slots."""
    mapping_file = scenario / "link_mapping.csv.gz"
    slots_file = scenario.parent / "dual_slots.csv.gz"
    if not mapping_file.is_file() or not slots_file.is_file():
        return {"status": "not_available", "note": "Keep link-level flows; mapping/dual_slots missing"}

    fields = ["segment_id", "from_node", "to_node"]
    mapping = pd.read_csv(mapping_file, dtype="string")
    slots = pd.read_csv(slots_file, dtype="string")
    required = {"link_id", *fields}
    if not required.issubset(mapping.columns) or not set(fields).issubset(slots.columns):
        raise ValueError("Invalid link_mapping.csv.gz or dual_slots.csv.gz schema")
    if mapping["link_id"].duplicated().any() or slots[fields].duplicated().any():
        raise ValueError("Duplicate link mapping or dual slot")
    if mapping[["link_id", *fields]].isna().any().any():
        raise ValueError("Null identifiers in link mapping")

    joined = flows.merge(mapping[["link_id", *fields]], on="link_id", how="left", validate="many_to_one")
    missing = joined["segment_id"].isna()
    unmapped = {m: int(joined.loc[missing, m].sum()) for m in MODES}
    grouped = joined.loc[~missing].groupby([*fields, "hour"], as_index=False, dropna=False)[list(MODES)].sum()

    outside = grouped[fields].drop_duplicates().merge(slots[fields], how="left", on=fields, indicator=True)
    if (outside["_merge"] == "left_only").any():
        raise ValueError("Mapped link refers to a segment/direction outside dual_slots.csv.gz")

    hours = pd.DataFrame({"hour": sorted(flows["hour"].unique())})
    full = slots[fields].merge(hours, how="cross")
    full = full.merge(grouped, on=[*fields, "hour"], how="left", validate="one_to_one")
    full[list(MODES)] = full[list(MODES)].fillna(0).astype("int64")
    full.to_csv(dest / "flows_dual_hourly.csv.gz", index=False, compression="gzip")
    return {"status": "ready", "slots": len(slots), "rows": len(full),
            "unmapped_link_flow_totals": unmapped}


def status_success(scenario: Path) -> bool:
    path = scenario / "run_status.json"
    if not path.is_file():
        return False
    try:
        return str(json.loads(path.read_text()).get("status", "")).lower() in SUCCESS_STATUSES
    except (ValueError, OSError):
        return False


def compact(scenario: Path, delete_raw: bool) -> None:
    flows = read_flows(scenario)
    # Write into staging first: never remove raw output after a partial extraction.
    with tempfile.TemporaryDirectory(prefix="targets_tmp_", dir=scenario) as temp:
        staging = Path(temp)
        flows.to_csv(staging / "flows_hourly.csv.gz", index=False, compression="gzip")
        trip_report = extract_trip_modes(scenario, staging)
        dual_report = write_dual_hourly(scenario, flows, staging)
        report = {
            "hours": sorted(int(h) for h in flows["hour"].unique()),
            "link_hour_rows": len(flows),
            "link_flow_totals": {m: int(flows[m].sum()) for m in MODES},
            "trips": trip_report, "dual": dual_report,
        }
        (staging / "target_summary.json").write_text(json.dumps(report, indent=2) + "\n")
        # Verify compressed targets can be read before moving them into place.
        check = pd.read_csv(staging / "flows_hourly.csv.gz", usecols=["hour", *MODES])
        if len(check) != len(flows) or any(int(check[m].sum()) != report["link_flow_totals"][m] for m in MODES):
            raise RuntimeError("Hourly targets failed post-write verification")
        final = scenario / "targets"
        final.mkdir(exist_ok=True)
        for p in staging.iterdir():
            p.replace(final / p.name)

    print(f"{scenario.name}: saved {len(flows):,} link-hour rows; modes {trip_report.get('mode_counts', 'existing')}")
    if delete_raw:
        if not status_success(scenario):
            raise RuntimeError(f"{scenario.name}: NOT deleting raw output: run_status.json is not complete")
        raw = scenario / "raw_output"
        if raw.is_dir():
            shutil.rmtree(raw)
            print(f"  removed {raw}")
        routechecked = scenario / "population_routechecked.xml.gz"
        if routechecked.is_file():
            routechecked.unlink()
            print("  removed routechecked population (source population is unchanged)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    location = parser.add_mutually_exclusive_group(required=True)
    location.add_argument("--dataset", type=Path, help="Postprocess all scenarios under this directory")
    location.add_argument("--scenario", type=Path, help="Postprocess one scenario (safe inside a SLURM array task)")
    parser.add_argument("--delete-raw", action="store_true", help="Delete raw_output and routechecked pop ONLY after extraction and successful run status")
    args = parser.parse_args()
    if args.scenario is not None:
        scenarios = [args.scenario]
    else:
        scenarios = sorted(p for p in args.dataset.iterdir() if p.is_dir()
                           and ((p / "raw_output").is_dir() or find_flow_file(p) is not None))
    print(f"Found {len(scenarios)} scenarios; deletion={'enabled' if args.delete_raw else 'DISABLED'}")
    failed = []
    for scenario in scenarios:
        try:
            compact(scenario, args.delete_raw)
        except Exception as exc:
            failed.append((scenario.name, str(exc)))
            print(f"FAILED {scenario.name}: {exc}", file=sys.stderr)
    if failed:
        print(f"{len(failed)} scenarios need attention; their raw outputs were retained on extraction failure.", file=sys.stderr)
        raise SystemExit(1)
    if not scenarios:
        raise SystemExit("No scenarios with hourly data found")


if __name__ == "__main__":
    main()

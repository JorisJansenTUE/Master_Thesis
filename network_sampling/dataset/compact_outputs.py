"""Preserve original MATSim flow, mode-share and final-iteration linkstats targets.

Retains the ORIGINAL columns, link IDs and hourly bins of:
    targets/output_detailed_link_statistics.csv.gz
    targets/modestats.csv
    targets/linkstats.txt.gz          # final iteration, original TSV columns

All original columns are preserved. No GNN feature engineering or aggregation.
By default raw_output is left intact. --delete-raw deletes it and the
regenerable route-checked population ONLY after a completed run and verified
copies of ALL THREE target files.

Usage:
    python -m network_sampling.dataset.compact_outputs --scenario "$SCENARIO"
    python -m network_sampling.dataset.compact_outputs --dataset "$DATASET" --delete-raw
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

FLOW_FILE = "output_detailed_link_statistics.csv.gz"
MODE_FILE = "modestats.csv"
LINKSTATS_FILE = "linkstats.txt.gz"
FLOW_MODES = ("cars", "trucks", "bikes", "buses")
MODES = FLOW_MODES  # Retain compare_flows.py compatibility
SUCCEEDED = {"completed", "complete", "succeeded", "success"}


def _open_text(path: Path):
    return gzip.open(path, "rt", newline="", encoding="utf-8-sig") if path.suffix == ".gz" else path.open("r", newline="", encoding="utf-8-sig")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _find_exact(raw: Path, names: tuple[str, ...]) -> Path | None:
    for name in names:
        p = raw / name
        if p.is_file():
            return p
    if not raw.is_dir():
        return None
    matches = sorted(p for p in raw.rglob("*") if p.is_file() and p.name in names)
    if len(matches) > 1:
        raise ValueError(f"Ambiguous files in {raw}: {matches}")
    return matches[0] if matches else None


def _source_flow(scenario: Path) -> Path:
    src = _find_exact(scenario / "raw_output", (FLOW_FILE, "output_detailed_link_statistics.csv"))
    if src is None:
        target = scenario / "targets" / FLOW_FILE
        if target.is_file():
            return target
        raise FileNotFoundError(f"Missing {FLOW_FILE} under {scenario}/raw_output")
    return src


def _source_modes(scenario: Path) -> Path:
    src = _find_exact(scenario / "raw_output", (
        "modestats.csv", "modestats.csv.gz", "output_modestats.csv", "output_modestats.csv.gz"
    ))
    if src is None:
        target = scenario / "targets" / MODE_FILE
        if target.is_file():
            return target
        raise FileNotFoundError(f"Missing modestats.csv under {scenario}/raw_output")
    return src


def _source_linkstats(scenario: Path) -> tuple[Path, int | None]:
    """Find the latest MATSim linkstats output, usually ITERS/it.N/N.linkstats.txt.gz.

    When last_iteration is recorded in run_status.json, require exactly that
    iteration. This prevents silently saving an earlier reporting window.
    """
    raw = scenario / "raw_output"
    status = json.loads((scenario / "run_status.json").read_text(encoding="utf-8"))
    last = status.get("last_iteration")
    last = int(last) if last is not None else None
    candidates = []
    if raw.is_dir():
        for file in raw.rglob("*linkstats.txt*"):
            if not file.is_file():
                continue
            match = re.fullmatch(r"(?:(\d+)\.)?linkstats\.txt(?:\.gz)?", file.name)
            if match is not None:
                iteration = int(match.group(1)) if match.group(1) else None
                candidates.append((iteration, file))
    if candidates:
        numbered = [(n, p) for n, p in candidates if n is not None]
        if last is not None:
            matching = [(n, p) for n, p in numbered if n == last]
            if not matching:
                raise FileNotFoundError(
                    f"{scenario.name}: no linkstats for lastIteration={last}; "
                    "set linkStats.writeLinkStatsInterval to a divisor of lastIteration "
                    "and keep raw_output for diagnosis."
                )
            selected = matching
        elif numbered:
            top = max(n for n, _ in numbered)
            selected = [(n, p) for n, p in numbered if n == top]
        else:
            selected = candidates
        if len(selected) != 1:
            raise ValueError(f"Ambiguous linkstats files for {scenario.name}: {selected}")
        return selected[0][1], selected[0][0]

    target = scenario / "targets" / LINKSTATS_FILE
    if target.is_file():
        # An already compacted scenario does not need raw_output again.
        summary = scenario / "targets" / "target_summary.json"
        iteration = None
        if summary.is_file():
            iteration = json.loads(summary.read_text(encoding="utf-8")).get("linkstats_iteration")
        return target, iteration
    raise FileNotFoundError(
        f"{scenario.name}: missing ITERS/it.N/N.linkstats.txt.gz; "
        "raw output will NOT be deleted. Previous compacted runs cannot "
        "recover linkstats if raw_output was already removed."
    )


def _validate_linkstats(path: Path) -> dict:
    """Validate gzip integrity and basic MATSim TSV structure without changing columns.

    Read to EOF so gzip CRC errors are detected before cleanup.
    """
    rows = 0
    with _open_text(path) as stream:
        reader = csv.reader(stream, delimiter="\t")
        header = next(reader, [])
        if len(header) < 3 or not any("TRAVELTIME" in h.upper() for h in header):
            raise ValueError(f"{path}: not a MATSim linkstats file with TRAVELTIME columns")
        # In MATSim outputs the first column is the link ID.
        for row in reader:
            if not row or not row[0].strip():
                raise ValueError(f"{path}: empty row/link identifier")
            if len(row) != len(header):
                raise ValueError(f"{path}: inconsistent column count at row {rows+2}")
            rows += 1
    if not rows:
        raise ValueError(f"{path}: no link statistics rows")
    return {"rows": rows, "columns": len(header),
            "travel_time_columns": [name for name in header if "TRAVELTIME" in name.upper()]}


def _validate_flows(path: Path) -> dict:
    """Stream and validate the file, including gzip CRC and all original columns."""
    rows = 0
    hours = set()
    ids = set()
    link_hours = set()
    totals = defaultdict(int)
    with _open_text(path) as stream:
        reader = csv.DictReader(stream)
        columns = reader.fieldnames or []
        required = {"link_id", "hour", *FLOW_MODES}
        if not required.issubset(columns):
            raise ValueError(f"{path}: missing hourly flow columns {sorted(required-set(columns))}")
        for row in reader:
            lid = str(row["link_id"] or "").strip()
            if not lid:
                raise ValueError(f"{path}: empty link ID")
            hour = int(row["hour"])
            if hour < 0:
                raise ValueError(f"{path}: negative hour {hour}")
            key = (lid, hour)
            if key in link_hours:
                raise ValueError(f"{path}: duplicate link/hour {key}")
            link_hours.add(key)
            ids.add(lid)
            hours.add(hour)
            for mode in FLOW_MODES:
                value = float(row[mode])
                if not value.is_integer() or value < 0:
                    raise ValueError(f"{path}: invalid {mode} count on {key}: {value}")
                totals[mode] += int(value)
            rows += 1
    if not rows:
        raise ValueError(f"{path}: no flow rows")
    return {"rows": rows, "links": len(ids), "hours": sorted(hours),
            "columns": columns, "flow_totals": dict(totals)}


def _validate_modes(path: Path) -> dict:
    """Use original iteration-mode shares, no inferred PT passenger counts."""
    with _open_text(path) as stream:
        header = stream.readline()
        delimiter = max((";", ",", "\t"), key=header.count)
        stream.seek(0)
        reader = csv.DictReader(stream, delimiter=delimiter)
        cols = reader.fieldnames or []
        modes = [c for c in cols if c != "iteration"]
        if "iteration" not in cols or not modes:
            raise ValueError(f"{path}: missing iteration or mode columns")
        found = {}
        for row in reader:
            iteration = int(row["iteration"])
            if iteration in found or iteration < 0:
                raise ValueError(f"{path}: duplicate/negative iteration {iteration}")
            shares = {mode: float(row[mode]) for mode in modes}
            if any(not 0.0 <= value <= 1.0 for value in shares.values()):
                raise ValueError(f"{path}: shares outside [0, 1]")
            if abs(sum(shares.values()) - 1.0) > 0.02:
                raise ValueError(f"{path}: shares at iteration {iteration} do not sum to 1")
            found[iteration] = shares
    if not found:
        raise ValueError(f"{path}: no mode statistics rows")
    last = max(found)
    return {"iterations": sorted(found), "final_iteration": last,
            "final_shares": found[last], "columns": cols}


def _completed(scenario: Path) -> bool:
    status_path = scenario / "run_status.json"
    if not status_path.is_file():
        return False
    status = json.loads(status_path.read_text(encoding="utf-8"))
    return str(status.get("state", status.get("status", ""))).lower() in SUCCEEDED


def _copy_as_gzip(source: Path, destination: Path) -> None:
    if source.resolve() == destination.resolve():
        return
    if source.suffix == ".gz":
        shutil.copyfile(source, destination)
    else:
        with source.open("rb") as inp, gzip.open(destination, "wb") as out:
            shutil.copyfileobj(inp, out)


def _copy_as_plain(source: Path, destination: Path) -> None:
    if source.resolve() == destination.resolve():
        return
    if source.suffix == ".gz":
        with gzip.open(source, "rb") as inp, destination.open("wb") as out:
            shutil.copyfileobj(inp, out)
    else:
        shutil.copyfile(source, destination)


def compact(scenario: Path, delete_raw: bool = False) -> dict:
    """Extract verified originals and optionally remove redundant MATSim output."""
    scenario = Path(scenario).resolve(strict=True)
    for required in ("scenario.json", "interventions.json", "network.xml.gz",
                     "transit_schedule.xml.gz", "link_mapping.csv.gz"):
        if not (scenario / required).is_file():
            raise FileNotFoundError(scenario / required)
    if not (scenario.parent / "street_segments.csv.gz").is_file():
        raise FileNotFoundError(scenario.parent / "street_segments.csv.gz")
    # Never delete failed or unverified runs, even if both CSVs exist.
    if delete_raw and not _completed(scenario):
        raise RuntimeError(f"{scenario.name}: no successful run_status.json; not deleting outputs")

    src_flow, src_mode = _source_flow(scenario), _source_modes(scenario)
    src_linkstats, linkstats_iteration = _source_linkstats(scenario)
    original_flows = _validate_flows(src_flow)
    original_modes = _validate_modes(src_mode)
    original_linkstats = _validate_linkstats(src_linkstats)

    # Stage on same filesystem, validate copies before touching final targets.
    with tempfile.TemporaryDirectory(dir=scenario, prefix=".targets_tmp_") as tmp:
        stage = Path(tmp)
        flow_dest = stage / FLOW_FILE
        mode_dest = stage / MODE_FILE
        linkstats_dest = stage / LINKSTATS_FILE
        _copy_as_gzip(src_flow, flow_dest)
        _copy_as_plain(src_mode, mode_dest)
        _copy_as_gzip(src_linkstats, linkstats_dest)
        if _validate_flows(flow_dest) != original_flows:
            raise RuntimeError(f"{scenario.name}: hourly flow copy verification failed")
        if _validate_modes(mode_dest) != original_modes:
            raise RuntimeError(f"{scenario.name}: mode-share copy verification failed")
        if _validate_linkstats(linkstats_dest) != original_linkstats:
            raise RuntimeError(f"{scenario.name}: linkstats copy verification failed")
        report = {
            "format": "original_matsim_statistics",
            "flow_file": f"targets/{FLOW_FILE}",
            "modestats_file": f"targets/{MODE_FILE}",
            "linkstats_file": f"targets/{LINKSTATS_FILE}",
            "linkstats_iteration": linkstats_iteration,
            "linkstats": original_linkstats,
            "flow": original_flows,
            "modestats": original_modes,
            "flow_sha256": _sha256(flow_dest),
            "modestats_sha256": _sha256(mode_dest),
            "linkstats_sha256": _sha256(linkstats_dest),
        }
        (stage / "target_summary.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
        final = scenario / "targets"
        final.mkdir(exist_ok=True)
        for file in stage.iterdir():
            os.replace(file, final / file.name)

    if delete_raw:
        # Explicit completion + verified destination sources are required.
        final = scenario / "targets"
        if _sha256(final / FLOW_FILE) != report["flow_sha256"] or \
           _sha256(final / MODE_FILE) != report["modestats_sha256"] or \
           _sha256(final / LINKSTATS_FILE) != report["linkstats_sha256"]:
            raise RuntimeError("Final target integrity verification failed; raw output retained")
        raw = scenario / "raw_output"
        if raw.is_dir():
            shutil.rmtree(raw)
        # Regenerated by PrepareInterventionPopulation on a future simulation.
        (scenario / "population_routechecked.xml.gz").unlink(missing_ok=True)
        print(f"CLEANED {scenario.name}: removed raw_output and routechecked population", flush=True)
    print(f"SAVED {scenario.name}: {original_flows['rows']:,} link-hour rows, "
          f"hours={len(original_flows['hours'])}, mode iteration={original_modes['final_iteration']}, "
          f"linkstats iteration={linkstats_iteration}", flush=True)
    return report



# Read-only compatibility helpers for the existing compare_flows.py.
# Neither function writes new targets or changes the MATSim CSVs.
def read_flows(scenario: Path):
    import pandas as pd
    source = _source_flow(Path(scenario))
    columns = ["link_id", "hour", *FLOW_MODES]
    return pd.read_csv(source, usecols=columns, dtype={"link_id": "string"})[columns]


def read_final_mode_shares(scenario: Path):
    import pandas as pd
    try:
        source = _source_modes(Path(scenario))
    except FileNotFoundError:
        return None
    with _open_text(source) as fh:
        header = fh.readline()
    separator = max((";", ",", "\t"), key=header.count)
    data = pd.read_csv(source, sep=separator)
    if "iteration" not in data.columns or data.empty:
        raise ValueError(f"{source}: missing iteration mode shares")
    final = data.loc[data["iteration"].idxmax()]
    names = [c for c in data.columns if c != "iteration"]
    return (pd.DataFrame({"main_mode": names, "share": [final[c] for c in names]}),
            int(final["iteration"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--scenario", type=Path)
    group.add_argument("--dataset", type=Path)
    parser.add_argument("--delete-raw", action="store_true",
                        help="Only after all three targets are verified and simulation status is completed")
    args = parser.parse_args()
    scenarios = [args.scenario] if args.scenario is not None else sorted(
        p for p in args.dataset.iterdir()
        if p.is_dir() and (p / "scenario.json").is_file()
    )
    errors = []
    for scenario in scenarios:
        try:
            compact(scenario, args.delete_raw)
        except Exception as exc:
            print(f"FAILED {scenario.name}: {exc}", file=sys.stderr)
            errors.append(scenario.name)
    if not scenarios or errors:
        raise SystemExit(f"{len(errors)} failed; {len(scenarios)} processed")


if __name__ == "__main__":
    main()

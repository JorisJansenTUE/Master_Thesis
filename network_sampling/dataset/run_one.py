"""Run one generated scenario with the existing Swiss eQasim JAR.

Sequence: PrepareInterventionPopulation (route mode + PT repair) -> MATSim ->
verified compact extraction. Nothing about GNN feature creation occurs here.
The --delete-raw switch is deliberately opt-in.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from network_sampling.dataset.compact_outputs import compact

PREPARE_MAIN = "org.thesis.PrepareInterventionPopulation"
SIMULATION_MAIN = "org.eqasim.switzerland.ch_cmdp.RunSimulation"
EQASIM_CONFIGURATOR = "org.eqasim.switzerland.ch_cmdp.SwitzerlandConfigurator"


def _time() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _required_file(path: Path) -> Path:
    path = path.resolve(strict=True)
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Missing/empty required file: {path}")
    return path


def _execute(command: list[str], cwd: Path, log_path: Path, env: dict[str, str]) -> None:
    print("Running:", " ".join(command), flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"Started: {_time()}\nWorking directory: {cwd}\n")
        log.write("Command: " + " ".join(command) + "\n\n")
        log.flush()
        subprocess.run(command, cwd=cwd, env=env, stdout=log,
                       stderr=subprocess.STDOUT, check=True)


def _status_write(path: Path, status: dict) -> None:
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    os.replace(temp, path)


def run(args: argparse.Namespace) -> None:
    scenario = args.scenario.resolve(strict=True)
    base_config = _required_file(args.base_config)
    population = _required_file(args.population)
    jar = _required_file(args.jar)
    workdir = (args.workdir or base_config.parent).resolve(strict=True)
    for name in ("scenario.json", "interventions.json", "network.xml.gz",
                 "transit_schedule.xml.gz", "link_mapping.csv.gz"):
        _required_file(scenario / name)
    _required_file(scenario.parent / "street_segments.csv.gz")
    if not 1 <= args.threads:
        raise ValueError("--threads must be positive")
    if args.last_iteration is not None and args.last_iteration < 0:
        raise ValueError("--last-iteration must be nonnegative")
    if args.linkstats_interval < 1 or args.linkstats_average_iterations < 1:
        raise ValueError("linkstats interval/average must be positive")
    if args.linkstats_average_iterations > args.linkstats_interval:
        raise ValueError("linkstats averaging window cannot exceed the write interval")
    if args.last_iteration is not None and args.last_iteration % args.linkstats_interval != 0:
        raise ValueError("last iteration must be a multiple of the linkstats interval so a final target exists")
    if args.delete_raw and args.no_compact:
        raise ValueError("--delete-raw cannot be combined with --no-compact")
    metadata = json.loads((scenario / "scenario.json").read_text(encoding="utf-8"))
    if metadata.get("id") != scenario.name:
        raise ValueError("scenario.json id does not match scenario directory")

    status_path = scenario / "run_status.json"
    if status_path.exists():
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if status.get("state", status.get("status")) not in ("completed", "complete"):
            raise RuntimeError(f"Existing {status_path} is not completed. Inspect logs and manually clear failed outputs before rerunning.")
        print(f"SKIP simulation (already completed): {scenario.name}", flush=True)
        if not args.no_compact and not args.dry_run:
            report = compact(scenario, delete_raw=args.delete_raw)
            status["postprocessing"] = {
                "state": "completed", "delete_raw": args.delete_raw,
                "finished": _time(), "flow_file": report["flow_file"],
                "linkstats_file": report["linkstats_file"],
            }
            _status_write(status_path, status)
        return

    network = _required_file(scenario / "network.xml.gz")
    schedule = _required_file(scenario / "transit_schedule.xml.gz")
    raw = scenario / "raw_output"
    if raw.exists() and any(raw.iterdir()):
        raise RuntimeError(f"Existing raw output at {raw}; refusing to overwrite")
    checked = scenario / "population_routechecked.xml.gz"
    if checked.exists():
        raise RuntimeError(f"Existing prepared population {checked}; refusing to overwrite")

    env = os.environ.copy()
    env["eqasim_configurator"] = EQASIM_CONFIGURATOR
    java_threads = f"-XX:ActiveProcessorCount={args.threads}"
    preprocess = [
        args.java, f"-Xmx{args.reset_heap}", java_threads, "-cp", str(jar),
        PREPARE_MAIN, str(base_config), str(network), str(schedule),
        str(population), str(checked),
    ]
    simulate = [
        args.java, f"-Xms{args.initial_heap}", f"-Xmx{args.heap}",
        java_threads, "-cp", str(jar), SIMULATION_MAIN,
        "--config-path", str(base_config),
        "--config:network.inputNetworkFile", str(network),
        "--config:plans.inputPlansFile", str(checked),
        "--config:transit.transitScheduleFile", str(schedule),
        "--config:controller.outputDirectory", str(raw),
        "--config:global.numberOfThreads", str(args.threads),
        "--config:qsim.numberOfThreads", str(args.threads),
        "--config:linkStats.writeLinkStatsInterval", str(args.linkstats_interval),
        "--config:linkStats.averageLinkStatsOverIterations", str(args.linkstats_average_iterations),
    ]
    if args.last_iteration is not None:
        simulate += ["--config:controller.lastIteration", str(args.last_iteration)]

    if args.dry_run:
        print("Preprocessing:\n" + " ".join(preprocess))
        print("Simulation:\n" + " ".join(simulate))
        print("Compaction:", "disabled" if args.no_compact else
              "enabled; delete_raw=" + str(args.delete_raw))
        return

    shutil.copy2(base_config, scenario / "base_config.xml")  # provenance only
    status = {
        "state": "running", "scenario": scenario.name,
        "started": _time(), "base_config": str(base_config),
        "population": str(population), "jar": str(jar),
        "working_directory": str(workdir), "threads": args.threads,
        "last_iteration": args.last_iteration,
        "linkstats_interval": args.linkstats_interval,
        "linkstats_average_iterations": args.linkstats_average_iterations,
    }
    _status_write(status_path, status)
    try:
        _execute(preprocess, cwd=workdir, log_path=scenario / "routecheck.log", env=env)
        _required_file(checked)
        raw.mkdir(exist_ok=True)
        _execute(simulate, cwd=workdir, log_path=scenario / "simulation.log", env=env)
        if not any(raw.iterdir()):
            raise RuntimeError("MATSim exited without producing any raw output")
        # Save successful simulation status BEFORE postprocessing, so a missing
        # mode CSV doesn't incorrectly mark the MATSim simulation itself failed.
        status["state"] = "completed"
        status["simulation_finished"] = _time()
        _status_write(status_path, status)
        print(f"Simulation completed: {scenario.name}", flush=True)
        if not args.no_compact:
            report = compact(scenario, delete_raw=args.delete_raw)
            status["postprocessing"] = {
                "state": "completed", "finished": _time(),
                "delete_raw": args.delete_raw, "flow_file": report["flow_file"],
                "linkstats_file": report["linkstats_file"],
            }
    except Exception as exc:
        if status["state"] != "completed":
            status["state"] = "failed"
            status["error"] = f"{type(exc).__name__}: {exc}"
        else:
            status["postprocessing"] = {
                "state": "failed", "error": f"{type(exc).__name__}: {exc}",
                "raw_output_retained": (scenario / "raw_output").exists(),
            }
        print(f"FAILED {scenario.name}: {exc}", file=sys.stderr)
        raise
    finally:
        status["finished"] = _time()
        _status_write(status_path, status)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", type=Path, required=True)
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--population", type=Path, required=True)
    parser.add_argument("--jar", type=Path, required=True)
    parser.add_argument("--workdir", type=Path, help="Original eQasim config directory (default config parent)")
    parser.add_argument("--java", default="java")
    parser.add_argument("--threads", type=int, default=24)
    parser.add_argument("--heap", default="96G")
    parser.add_argument("--initial-heap", default="16G")
    parser.add_argument("--reset-heap", default="24G")
    parser.add_argument("--last-iteration", type=int, default=None)
    parser.add_argument("--linkstats-interval", type=int, default=10)
    parser.add_argument("--linkstats-average-iterations", type=int, default=5)
    parser.add_argument("--no-compact", action="store_true", help="Disable target extraction")
    parser.add_argument("--delete-raw", action="store_true", help="After compaction succeeds, delete raw_output and regenerated population")
    parser.add_argument("--dry-run", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()

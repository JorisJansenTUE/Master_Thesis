"""Step 2: Run one generated scenario in Eqasim/MATSim.

Responsibilities:
  1. Reset population routes incompatible with the generated network.
  2. Run Eqasim with the original config and explicit scenario overrides.
  3. Preserve logs and raw MATSim outputs for inspection.

Nothing here writes MATSim networks, modifies transit schedules, extracts training
features/targets, or deletes simulation output. Those tasks belong elsewhere.

Requires org.thesis.CheckRouteModeAndResetPopulation in the Swiss Eqasim JAR.
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

ROUTE_RESET_MAIN = "org.thesis.CheckRouteModeAndResetPopulation"
SIMULATION_MAIN = "org.eqasim.switzerland.ch_cmdp.RunSimulation"
EQASIM_CONFIGURATOR = "org.eqasim.switzerland.ch_cmdp.SwitzerlandConfigurator"


def _time() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _required_file(path: Path) -> Path:
    path = path.resolve(strict=True)
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Missing or empty input: {path}")
    return path


def _execute(command: list[str], *, cwd: Path, log_path: Path, env: dict[str, str]):
    print("Running:", " ".join(command), flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"Started: {_time()}\nWorking directory: {cwd}\n")
        log.write("Command: " + " ".join(command) + "\n\n")
        log.flush()
        subprocess.run(command, cwd=cwd, env=env, stdout=log,
                       stderr=subprocess.STDOUT, check=True)


def run(args: argparse.Namespace):
    # This scenario was created by dataset/generate.py. Do not regenerate it.
    scenario = args.scenario.resolve(strict=True)
    manifest = _required_file(scenario / "scenario.json")
    _required_file(scenario / "interventions.json")
    network = _required_file(scenario / "network.xml.gz")
    schedule = _required_file(scenario / "transit_schedule.xml.gz")
    base_config = _required_file(args.base_config)
    population = _required_file(args.population)
    jar = _required_file(args.jar)
    workdir = (args.workdir or base_config.parent).resolve(strict=True)

    if args.threads < 1:
        raise ValueError("--threads must be >= 1")
    if args.last_iteration is not None and args.last_iteration < 0:
        raise ValueError("--last-iteration must be >= 0")
    if not workdir.is_dir():
        raise NotADirectoryError(workdir)

    metadata = json.loads(manifest.read_text(encoding="utf-8"))
    if metadata.get("id") != scenario.name:
        raise ValueError("scenario.json id does not match directory name")

    status_path = scenario / "run_status.json"
    if status_path.exists():
        previous = json.loads(status_path.read_text(encoding="utf-8"))
        if previous.get("state") == "completed":
            print(f"Already completed: {scenario.name}")
            return
        raise RuntimeError(
            f"Existing run_status.json has state {previous.get('state')!r}. "
            "Inspect the previous run before manually removing it to retry."
        )

    raw_output = scenario / "raw_output"
    if raw_output.exists() and any(raw_output.iterdir()):
        raise RuntimeError(
            f"Nonempty output folder: {raw_output}. "
            "Refusing to overwrite a previous simulation."
        )

    checked_population = scenario / "population_routechecked.xml.gz"
    if checked_population.exists():
        raise RuntimeError(f"Population already exists: {checked_population}")

    env = os.environ.copy()
    env["eqasim_configurator"] = EQASIM_CONFIGURATOR
    threads = f"-XX:ActiveProcessorCount={args.threads}"
    # reset_command = [
    #     args.java, f"-Xmx{args.reset_heap}", threads, "-cp", str(jar),
    #     ROUTE_RESET_MAIN, str(network), str(population), str(checked_population),
    # ]
    reset_command = [
        "java",
        "-Xmx24G",
        "-XX:ActiveProcessorCount=24",
        "-cp", str(jar),
        "org.thesis.PrepareInterventionPopulation",
        str(base_config),
        str(scenario / "network.xml.gz"),
        str(scenario / "transit_schedule.xml.gz"),
        str(population),
        str(scenario / "population_routechecked.xml.gz"),
    ]


    simulation_command = [
        args.java, f"-Xms{args.initial_heap}", f"-Xmx{args.heap}",
        threads, "-cp", str(jar), SIMULATION_MAIN,
        "--config-path", str(base_config),
        "--config:network.inputNetworkFile", str(network),
        "--config:plans.inputPlansFile", str(checked_population),
        "--config:transit.transitScheduleFile", str(schedule),
        "--config:controller.outputDirectory", str(raw_output),
    ]
    if args.last_iteration is not None:
        simulation_command += [
            "--config:controller.lastIteration", str(args.last_iteration)
        ]

    if args.dry_run:
        print("\nRoute reset command:\n  " + " ".join(reset_command))
        print("\nSimulation command:\n  " + " ".join(simulation_command))
        print("\nNo Java process started; no status recorded.")
        return

    # Copy only for provenance. RunSimulation uses the original config, so
    # its relative YAML/CSV/facilities/vehicle paths keep their original context.
    shutil.copy2(base_config, scenario / "base_config.xml")
    status = {
        "state": "running",
        "scenario": scenario.name,
        "started": _time(),
        "base_config": str(base_config),
        "population": str(population),
        "jar": str(jar),
        "working_directory": str(workdir),
        "threads": args.threads,
        "last_iteration": args.last_iteration,
    }
    status_path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    try:
        _execute(reset_command, cwd=workdir,
                 log_path=scenario / "routecheck.log", env=env)
        _required_file(checked_population)

        # Each scenario has a unique raw_output folder, preventing conflicts.
        raw_output.mkdir(exist_ok=True)
        _execute(simulation_command, cwd=workdir,
                 log_path=scenario / "simulation.log", env=env)
        if not any(raw_output.iterdir()):
            raise RuntimeError("Simulation returned successfully but raw_output is empty")
        status["state"] = "completed"
        print(f"Finished: {scenario.name}; outputs in {raw_output}")
    except Exception as error:
        status["state"] = "failed"
        status["error"] = f"{type(error).__name__}: {error}"
        print(f"FAILED: {status['error']}", file=sys.stderr)
        raise
    finally:
        status["finished"] = _time()
        status_path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", type=Path, required=True,
                        help="A directory produced by generate.py")
    parser.add_argument("--base-config", type=Path, required=True,
                        help="Original Eqasim config, still next to its resources")
    parser.add_argument("--population", type=Path, required=True,
                        help="Prepared baseline population, same network model version")
    parser.add_argument("--jar", type=Path, required=True,
                        help="Swiss Eqasim JAR with the route-check helper class")
    parser.add_argument("--workdir", type=Path,
                        help="Original Eqasim working directory (default: config parent)")
    parser.add_argument("--java", default="java")
    parser.add_argument("--threads", type=int, default=24)
    parser.add_argument("--heap", default="96G")
    parser.add_argument("--initial-heap", default="16G")
    parser.add_argument("--reset-heap", default="24G")
    parser.add_argument("--last-iteration", type=int,
                        help="Override config's lastIteration; omit to use original value")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the Java commands without running them")
    run(parser.parse_args())


if __name__ == "__main__":
    main()

"""Overnight SLURM orchestration for the existing restartable MATSim GA.

No new genetic or MATSim logic lives here. This process starts and waits for
existing submit_generation.sh arrays, then calls optimization.run step. Its
checkpoint (optimization_state.json) remains the single source of GA progress.

An active_array.json marker makes the overnight script safe to restart after
walltime/preemption: it waits for an already-submitted array rather than
launching duplicate evaluations.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time

STATE_FILE = "optimization_state.json"
ACTIVE_FILE = "hpc_active_array.json"
DEFAULT_JOB_WALLTIME_S = 24 * 3600  # fallback outside SLURM only
# Stop monitoring before SLURM sends SIGTERM, allowing a clean final log.
HARD_EXIT_BUFFER_S = 180


def log(msg: str) -> None:
    print(f"[{datetime.now(timezone.utc).isoformat(timespec='seconds')}] {msg}", flush=True)


def actual_job_walltime_s() -> int:
    """Use the effective Slurm allocation, including sbatch --time overrides.

    SLURM_TIMELIMIT is provided in minutes by Slurm. In non-Slurm tests,
    fall back to 24h rather than claiming an artificial 8-hour deadline.
    """
    raw = os.environ.get('SLURM_TIMELIMIT')
    if not raw:
        return DEFAULT_JOB_WALLTIME_S
    if raw.isdecimal():
        return int(raw) * 60
    # Some Slurm installations export a formatted timelimit.
    day, sep, clock = raw.partition('-')
    days = int(day) if sep else 0
    clock = clock if sep else raw
    fields = [int(x) for x in clock.split(':')]
    if len(fields) == 2:
        hours, minutes = fields
        seconds = 0
    elif len(fields) == 3:
        hours, minutes, seconds = fields
    else:
        raise ValueError(f'Cannot interpret SLURM_TIMELIMIT={raw!r}')
    return ((days * 24 + hours) * 60 + minutes) * 60 + seconds


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, content: dict) -> None:
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(content, indent=2) + "\n", encoding="utf-8")
    os.replace(temp, path)


def invoke(command: list[str], *, capture: bool = False) -> str:
    log("RUN " + " ".join(command))
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    if result.stdout:
        print(result.stdout, end="" if result.stdout.endswith("\n") else "\n", flush=True)
    if result.stderr:
        print(result.stderr, end="" if result.stderr.endswith("\n") else "\n", file=sys.stderr, flush=True)
    if result.returncode:
        raise RuntimeError(f"Command exited {result.returncode}: {command!r}")
    return result.stdout if capture else ""


def active_slurm_job(job_id: str) -> bool:
    result = subprocess.run(
        ["squeue", "-h", "-j", job_id, "-o", "%i"],
        text=True,
        capture_output=True,
        check=False,
    )

    if result.returncode:
        if "invalid job id specified" in result.stderr.lower():
            return False
        raise RuntimeError(f"squeue failed: {result.stderr.strip()}")

    return bool(result.stdout.strip())


def state_summary(state: dict) -> tuple[int, int]:
    gen = int(state["generation"])
    pending = sum(state["archive"][key]["fitness"] is None for key in state["population"])
    return gen, pending


def init_command(args: argparse.Namespace) -> list[str]:
    return [
        str(args.python), "-m", "optimization.run", "init",
        "--dataset", str(args.dataset),
        "--baseline", str(args.baseline),
        "--warm-start", str(args.warm_start),
        "--matsim-jar", str(args.jar),
        "--reference", str(args.reference),
        "--spatial-dir", str(args.spatial_dir),
        "--population", str(args.population),
        "--generations", str(args.generations),
        "--budget", str(args.budget),
        "--seed", str(args.seed),
        "--elites", str(args.elites),
        "--immigrants", str(args.immigrants),
    ] + (["--no-warm-start"] if os.environ.get("GA_NO_WARM_START", "0") == "1" else [])


def submission_command(args: argparse.Namespace, manifest: Path) -> list[str]:
    command = [
        "bash", str(args.repo / "optimization" / "submit_generation.sh"),
        str(args.dataset), str(manifest), str(args.config),
        str(args.population_file), str(args.jar), str(args.python),
        str(args.max_parallel),
    ]
    if args.delete_raw:
        command += ["--delete-raw"]
    return command


def step_command(args: argparse.Namespace) -> list[str]:
    cmd = [str(args.python), "-m", "optimization.run", "step",
           "--dataset", str(args.dataset)]
    if args.penalize_failed:
        cmd.append("--penalize-failed")
    return cmd


def wait_for_job(job_id: str, *, hard_deadline: float, poll_sec: int) -> bool:
    """Return False if nearly out of SLURM walltime; leave job marker intact."""
    while active_slurm_job(job_id):
        if time.monotonic() >= hard_deadline:
            log(f"Walltime approaching; array {job_id} still active. "
                "Leaving it running and preserving the marker for the next overnight job.")
            return False
        time.sleep(min(poll_sec, max(1.0, hard_deadline - time.monotonic())))
    return True


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("repo", "python", "dataset", "baseline", "warm-start", "reference",
                 "spatial-dir", "config", "population-file", "jar"):
        p.add_argument("--" + name, required=True, type=Path)
    p.add_argument("--budget", required=True, type=float)
    p.add_argument("--population", required=True, type=int)
    p.add_argument("--generations", required=True, type=int)
    p.add_argument("--seed", required=True, type=int)
    p.add_argument("--elites", required=True, type=int)
    p.add_argument("--immigrants", required=True, type=float)
    p.add_argument("--max-parallel", required=True, type=int)
    p.add_argument("--delete-raw", required=True, type=int, choices=(0, 1))
    p.add_argument("--penalize-failed", required=True, type=int, choices=(0, 1))
    p.add_argument("--stop-margin-min", required=True, type=int)
    p.add_argument("--poll-sec", required=True, type=int)
    p.add_argument("--dry-run", required=True, type=int, choices=(0, 1))
    args = p.parse_args()
    if args.max_parallel < 1 or args.poll_sec < 1 or args.stop_margin_min < 5:
        p.error("max-parallel/poll-sec must be positive and stop margin >=5 minutes")
    job_walltime_s = actual_job_walltime_s()
    if args.stop_margin_min * 60 + HARD_EXIT_BUFFER_S >= job_walltime_s:
        p.error(f"stop margin too large for SLURM walltime ({job_walltime_s / 3600:g} hours)")
    if args.population < 2 or args.generations < 1:
        p.error("population>=2 and generations>=1 required")

    args.dataset.mkdir(parents=True, exist_ok=True)
    state_path = args.dataset / STATE_FILE
    active_path = args.dataset / ACTIVE_FILE
    hard_deadline = time.monotonic() + job_walltime_s - HARD_EXIT_BUFFER_S
    submit_deadline = hard_deadline - args.stop_margin_min * 60
    log(f"Starting/restarting GA, SLURM walltime={job_walltime_s / 3600:g}h; "
        f"submitting until ~{args.stop_margin_min} min before walltime")

    if not state_path.is_file():
        init = init_command(args)
        if args.dry_run:
            invoke(init + ["--dry-run"])
            log("DRY RUN complete; no checkpoint, MATSim jobs or scenario folders created")
            return
        invoke(init)
    elif args.dry_run:
        invoke([str(args.python), "-m", "optimization.run", "status",
                "--dataset", str(args.dataset)])
        state = read_json(state_path)
        gen, pending = state_summary(state)
        manifest = args.dataset / f"generation_{gen:03d}.txt"
        if pending and manifest.is_file():
            invoke(submission_command(args, manifest) + ["--dry-run"])
        log("DRY RUN complete")
        return

    while True:
        state = read_json(state_path)
        gen, pending = state_summary(state)
        log(f"Generation {gen + 1}/{state['generations']}: "
            f"phase={state['phase']}, waiting for {pending} fitness values")
        if state["phase"] == "complete":
            log("Optimisation finished all configured generations")
            return

        marker = read_json(active_path) if active_path.is_file() else None
        if marker:
            marker_gen = int(marker["generation"])
            if marker_gen != gen:
                if marker_gen < gen:
                    # The checkpoint advanced before a previous process removed its marker.
                    active_path.unlink()
                    continue
                raise RuntimeError(f"Active array belongs to future generation {marker_gen}")
            if not wait_for_job(str(marker["job_id"]), hard_deadline=hard_deadline,
                                poll_sec=args.poll_sec):
                return
            log(f"Array {marker['job_id']} finished; collecting fitness")
            invoke(step_command(args))
            updated = read_json(state_path)
            updated_gen, updated_pending = state_summary(updated)
            if updated_gen == gen and updated["phase"] != "complete" and updated_pending:
                raise RuntimeError(
                    f"Generation {gen} still has {updated_pending} unscored scenarios after "
                    f"array {marker['job_id']} finished. Inspect run_status.json and "
                    "slurm_logs before retrying; no duplicate submission was made."
                )
            active_path.unlink()
            continue

        if pending == 0:
            invoke(step_command(args))
            updated = read_json(state_path)
            if updated["generation"] == gen and updated["phase"] != "complete":
                raise RuntimeError("GA step did not advance despite zero pending fitness values")
            continue

        manifest = args.dataset / f"generation_{gen:03d}.txt"
        if not manifest.is_file():
            raise FileNotFoundError(f"Missing GA generation manifest: {manifest}")
        entries = [s.strip() for s in manifest.read_text(encoding="utf-8").splitlines() if s.strip()]
        if not entries:
            raise RuntimeError(f"No simulations in {manifest} but {pending} unknown scores")

        if time.monotonic() >= submit_deadline:
            log(f"Reached overnight submission cutoff; generation {gen} will continue "
                "on the next overnight job. Nothing was canceled.")
            return

        out = invoke(submission_command(args, manifest), capture=True)
        match = re.search(r"Submitted\s+\d+\s+GA scenarios as SLURM job array\s+(\d+)", out)
        if match:
            job_id = match.group(1)
            write_json(active_path, {
                "generation": gen,
                "job_id": job_id,
                "manifest": str(manifest),
                "submitted_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            })
            log(f"Submitted generation {gen} as array {job_id}")
            continue
        if "Nothing to submit." in out:
            # May occur if all MATSim outputs were already present, but fitness
            # has not yet been collected (e.g. after a previous interruption).
            log("All MATSim files already present; collecting cached evaluations")
            invoke(step_command(args))
            updated = read_json(state_path)
            if updated["generation"] == gen and updated["phase"] != "complete":
                raise RuntimeError("MATSim submission had nothing to do, but fitness is incomplete")
            continue
        raise RuntimeError(f"Could not identify submitted job array in output: {out!r}")


if __name__ == "__main__":
    main()

#!/usr/bin/env bash
# Submit explicitly selected GA scenarios as a throttled SLURM array.
#
# Usage (on the HPC, from the repository root):
#   bash optimization/submit_generation.sh \
#     DATASET MANIFEST CONFIG POPULATION JAR PYTHON [max_parallel] [--dry-run] [--delete-raw]
#
# MANIFEST contains one scenario ID (e.g. ga_2c5238eb86d4f1c8) per line.
# Pass DATASET/scenarios.txt for a small pilot, or a generation-specific manifest.
#
# Reuses network_sampling/dataset/run_all_scenarios.sh --worker and hence the
# established run_one.py MATSim settings, postprocessing and status handling.
# Neither the original submission script nor run_one.py is modified.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
DATASET_WORKER="$REPO_ROOT/network_sampling/dataset/run_all_scenarios.sh"

if [[ "${1:-}" == "--worker" ]]; then
    shift
    if (( $# < 5 )); then
        echo "Internal: --worker MANIFEST CONFIG POPULATION JAR PYTHON [--delete-raw]" >&2
        exit 2
    fi
    : "${SLURM_ARRAY_TASK_ID:?This entrypoint must run as a SLURM array task}"
    SCENARIO="$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$1")"
    [[ -d "$SCENARIO" ]] || { echo "Missing GA scenario $SCENARIO" >&2; exit 1; }

    # Prevent two independently submitted arrays from writing to the same
    # scenario concurrently. Keep the file descriptor open until worker exits.
    exec 9>"$SCENARIO/.ga_worker.lock"
    if ! flock -n 9; then
        echo "BLOCKED: scenario already being evaluated by another worker: $SCENARIO" >&2
        exit 1
    fi
    bash "$DATASET_WORKER" --worker "$@"
    exit $?
fi

if (( $# < 6 )); then
    echo "Usage: bash $0 DATASET MANIFEST CONFIG POPULATION JAR PYTHON [max_parallel] [--dry-run] [--delete-raw]" >&2
    exit 2
fi

DATASET="$(readlink -f -- "$1")"
INPUT_MANIFEST="$(readlink -f -- "$2")"
CONFIG="$(readlink -f -- "$3")"
POPULATION="$(readlink -f -- "$4")"
JAR="$(readlink -f -- "$5")"
PYTHON="$6"
shift 6

# Preserve virtualenv Python symlinks (do not resolve .venv/bin/python).
if [[ "$PYTHON" != */* ]]; then
    PYTHON="$(command -v "$PYTHON")" || { echo "Python not found" >&2; exit 2; }
elif [[ "$PYTHON" != /* ]]; then
    PYTHON="$PWD/$PYTHON"
fi

MAX_PARALLEL=3
if (( $# > 0 )) && [[ "$1" =~ ^[0-9]+$ ]]; then
    MAX_PARALLEL="$1"
    shift
fi
[[ "$MAX_PARALLEL" =~ ^[1-9][0-9]*$ ]] || { echo "max_parallel must be positive" >&2; exit 2; }

DRY_RUN=0
DELETE_RAW=0
for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=1 ;;
        --delete-raw) DELETE_RAW=1 ;;
        *) echo "Unknown option: $arg" >&2; exit 2 ;;
    esac
done

for file in "$INPUT_MANIFEST" "$CONFIG" "$POPULATION" "$JAR" "$PYTHON" "$DATASET_WORKER"; do
    [[ -s "$file" ]] || { echo "Missing/empty required file: $file" >&2; exit 2; }
done
for file in "$DATASET/dataset_manifest.json" "$DATASET/street_segments.csv.gz"; do
    [[ -s "$file" ]] || { echo "Incomplete GA dataset: $file" >&2; exit 2; }
done

mkdir -p "$DATASET/slurm_logs"
# SLURM must read the exact same snapshot even if the source manifest is
# modified during a run. Retain it for real submissions; remove on dry-run.
RUN_MANIFEST="$(mktemp "$DATASET/slurm_logs/ga_manifest.XXXXXXXX.txt")"
PERSIST_MANIFEST=0
cleanup() {
    if (( PERSIST_MANIFEST == 0 )); then rm -f -- "$RUN_MANIFEST"; fi
}
trap cleanup EXIT

"$PYTHON" - "$DATASET" "$INPUT_MANIFEST" "$RUN_MANIFEST" <<'PY'
import json
import re
import sys
from pathlib import Path

dataset, source, output = map(Path, sys.argv[1:])
expected = (
    "network.xml.gz", "transit_schedule.xml.gz", "scenario.json",
    "interventions.json", "link_mapping.csv.gz",
)
targets = (
    "output_detailed_link_statistics.csv.gz", "modestats.csv", "linkstats.txt.gz",
)
seen = set()
selected = []
completed = 0
blocked = []

for line_no, raw in enumerate(source.read_text(encoding="utf-8-sig").splitlines(), 1):
    name = raw.strip()
    if not name or name.startswith("#"):
        continue
    # Accept names, not Windows paths or other dataset directories.
    if re.fullmatch(r"ga_[A-Za-z0-9_-]+", name) is None:
        raise SystemExit(f"Invalid GA scenario ID at line {line_no}: {name!r}")
    if name in seen:
        print(f"SKIP duplicate manifest entry: {name}")
        continue
    seen.add(name)
    folder = dataset / name
    missing = [part for part in expected if not (folder / part).is_file()]
    if missing:
        raise SystemExit(f"Incomplete scenario {name}: missing {missing}")
    metadata = json.loads((folder / "scenario.json").read_text(encoding="utf-8"))
    if metadata.get("id") != name or metadata.get("strategy") != "ga":
        raise SystemExit(f"Invalid GA scenario metadata in {folder}")

    status_file = folder / "run_status.json"
    if status_file.is_file():
        status = json.loads(status_file.read_text(encoding="utf-8"))
        state = str(status.get("state", status.get("status", ""))).lower()
        if state in {"completed", "complete"}:
            if all((folder / "targets" / file).is_file() for file in targets):
                completed += 1
                print(f"SKIP completed: {name}")
                continue
            # run_one.py re-runs compaction only, not MATSim, for completed runs.
            print(f"REPROCESS: {name} (MATSim completed; compact targets incomplete)")
        elif state in {"failed", "running"}:
            blocked.append((name, state))
            continue
        else:
            raise SystemExit(f"Unknown run_status.json state {state!r} for {name}")
    selected.append(str(folder.resolve()))

if blocked:
    for name, state in blocked:
        print(f"BLOCKED {name}: status={state}; inspect and recover manually", file=sys.stderr)
    raise SystemExit("Refusing submission while manifest includes failed/running scenarios")

output.write_text("".join(f"{item}\n" for item in selected), encoding="utf-8")
print(f"Manifest: {len(seen)} unique GA scenarios, {completed} already completed, "
      f"{len(selected)} needing processing")
PY

N="$(wc -l < "$RUN_MANIFEST")"
if (( N == 0 )); then
    echo "Nothing to submit."
    exit 0
fi
if (( N > 1000 )); then
    echo "A single generation is limited to 1000 SLURM tasks; got $N. Split the manifest." >&2
    exit 2
fi

# Resource requests below mirror the CURRENT working dataset SLURM launcher.
# Actual MATSim flags stay exclusively in its --worker entrypoint:
# threads=2, heap=4G, reset_heap=3G, last_iteration=100,
# linkstats_interval=25, linkstats_average_iterations=5.
# Keep resource requests aligned if that worker is changed later.
SBATCH_ARGS=(
    --partition=mech-cst-mov.cpu.q
    --nodes=1
    --ntasks=1
    --cpus-per-task=2
    --mem=6G
    --time=00:30:00
    --job-name=matsim-ga
    --output="$DATASET/slurm_logs/%A_%a.out"
    --error="$DATASET/slurm_logs/%A_%a.err"
    --array="0-$((N - 1))%$MAX_PARALLEL"
)
WORKER_ARGS=("$SCRIPT_DIR/submit_generation.sh" --worker "$RUN_MANIFEST" "$CONFIG" "$POPULATION" "$JAR" "$PYTHON")
if (( DELETE_RAW == 1 )); then WORKER_ARGS+=(--delete-raw); fi

if (( DRY_RUN == 1 )); then
    echo "DRY RUN: would submit $N GA scenarios, max $MAX_PARALLEL in parallel"
    echo "SLURM resources: 2 CPUs, 6G RAM, 30 min/task (matching dataset runner)"
    sed 's/^/  /' "$RUN_MANIFEST"
    echo "Raw MATSim output: $([[ "$DELETE_RAW" == 1 ]] && echo 'delete AFTER verified compaction' || echo 'retain')"
    exit 0
fi

# One array handles this generation. The immutable manifest is kept so queued
# workers never depend on a source manifest being edited or regenerated.
JOB_ID="$(sbatch --parsable "${SBATCH_ARGS[@]}" "${WORKER_ARGS[@]}")"
PERSIST_MANIFEST=1
echo "Submitted $N GA scenarios as SLURM job array $JOB_ID (maximum $MAX_PARALLEL concurrent)"
echo "Scenario snapshot: $RUN_MANIFEST"
echo "Inspect with: squeue -j ${JOB_ID%%;*}"
echo "Logs: $DATASET/slurm_logs/"
echo "Raw output: $([[ "$DELETE_RAW" == 1 ]] && echo 'delete after verified compaction' || echo 'retain')"
